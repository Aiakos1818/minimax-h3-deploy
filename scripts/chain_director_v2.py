#!/usr/bin/env python3
"""chain_director_v2.py -- ChainDirector persist mode (FSDP resident across segments).

Same engine as chain_director_v1.py (Herrgotts masked-AV, segment queue per clip)
but the RayInitializer runs with clear_vram_after_sampling=False so the FSDP
UNet stays resident in VRAM between queues. Paired with the raylight
idempotent-RayInitializer + persist-aware ensure_fresh_actors patches, later
segments reuse the loaded model instead of ray.shutdown()/recreating actors and
reloading FSDP every queue (and the cond cache skips CLIP dispatch).

First-segment material routing:
  * no material            -> fl2va  (H3ContinuousStartV14, fl2va unet) -- text only
  * --first-image/--last-image -> fl2va  (Start.first_frame/last_frame anchors)
  * --ref-image/--ref-video    -> ref2va (official MiniMaxH3ReferenceToVideo,
                               ref2va unet; ref images/videos enter the DiT
                               condition). Continuation segments always run the
                               fl2va unet (new reuse_epoch rebuild after clip 1).

Everything else (params, resume, --merge, stitch time-base fix) matches v1.
v1 is no longer maintained; all changes land here.

Managed ComfyUI lifecycle: this script owns the ComfyUI process for a real run.
Before any segment work it takes the service over (stop if up, then start, then
wait for :8188); on exit -- success, failure/SystemExit, or SIGTERM/SIGINT -- it
stops the service so GPUs are free between runs. Merge-only and `all clips
present` runs never touch the service. The --clear option is ignored here.
"""
import argparse, glob, json, os, random, re, shutil, signal, subprocess, sys, time, urllib.request

import numpy as np

from safetensors import safe_open

import av

HOME = os.path.expanduser("~/MiniMax-H3-Deploy")
API = "http://127.0.0.1:8188"
OUTPUT = HOME + "/output"
INPUT_DIR = os.path.expanduser("~/ComfyUI-Deploy/input")
CLIP = "qwen3vl_32b_minimax_h3_int8_convrot.safetensors"
VAE_VIDEO = "minimax_h3_video_vae_fp16.safetensors"
VAE_AUDIO = "minimax_h3_audio_vae_fp32.safetensors"
UNET_FL2VA = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
UNET_REF2VA = "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
PREFIX = "h3_continuous/chain"          # saved under OUTPUT
SLOT_DIR = "h3_continuous"
SLOT_BASE = "chain"
FPS = 24
HEAD = 39
ANALYZE_CLS = "H3ContinuousAnalyzeHandoverV14"
START_CLS = "H3ContinuousStartV14"
CONT_CLS = "H3ContinuousContinueV14"
SAVE_CLS = "H3ContinuousSaveLatent"
LOAD_CLS = "H3ContinuousLoadLatent"

_OBJ = {}

# Unique tag per script run: every segment of this run shares it, so the first
# queue rebuilds ray (clearing any resident FSDP from a previous run and freeing
# VRAM for CLIP dispatch), while segments 2..N reuse the resident model.
_RUN_EPOCH = int(time.time_ns())


def obj(name):
    if name not in _OBJ:
        _OBJ[name] = json.load(urllib.request.urlopen(API + "/object_info/" + name, timeout=30))[name]
    return _OBJ[name]


def widget_defaults(name):
    out = {}
    for cat in ("required", "optional"):
        for k, spec in (obj(name).get("input", {}).get(cat, {}) or {}).items():
            if isinstance(spec, list):
                meta = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
                if "default" in meta:
                    out[k] = meta["default"]
            elif isinstance(spec, dict) and "default" in spec:
                out[k] = spec["default"]
    return out


def slot_path(clip_idx):
    return os.path.join(OUTPUT, SLOT_DIR, "%s_%05d.safetensors" % (SLOT_BASE, clip_idx))


def http_json(url, data=None, timeout=120):
    if data is None:
        return json.load(urllib.request.urlopen(url, timeout=timeout))
    req = urllib.request.Request(url, data=json.dumps(data).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def _stage_image(src, tag, idx):
    """Copy a local image into ComfyUI's input dir; return its file name."""
    if not os.path.isfile(src):
        sys.exit("image not found: %s" % src)
    name = "%s_%d_%s" % (tag, idx, os.path.basename(src))
    dst = os.path.join(INPUT_DIR, name)
    if os.path.abspath(src) != os.path.abspath(dst):
        shutil.copy2(src, dst)
    print("[material] image -> input/%s" % name)
    return name


def _refit_24fps(src, tag, idx):
    """Re-encode a reference video to a 24 fps mp4 in ComfyUI's input dir.

    No ffmpeg binary on the host, so pyav is used: decode video + optional audio,
    then write an h264(24fps)/aac mp4. Returns (file name, has_audio).
    """
    if not os.path.isfile(src):
        sys.exit("video not found: %s" % src)
    name = "%s_%d_%s" % (tag, idx, os.path.basename(src))
    stem, ext = os.path.splitext(name)
    if ext.lower() != ".mp4":
        name = stem + ".mp4"
    dst = os.path.join(INPUT_DIR, name)
    tmp = dst + ".tmp"
    vframes, w, h = [], 0, 0
    with av.open(src) as ins:
        vs = next((s for s in ins.streams if s.type == "video"), None)
        if vs is None:
            sys.exit("video %s has no video stream" % src)
        w, h = vs.codec_context.width, vs.codec_context.height
        p = 0
        for fr in ins.decode(vs):
            f = fr.reformat(width=w, height=h, format="yuv420p")
            # Reformat keeps the decoded frame time base (e.g. mp4 1/12288);
            # advance pts by one frame in that base like stitch() does.
            den = f.time_base.denominator if f.time_base else FPS
            f.pts = p
            p += round(den / FPS)
            vframes.append(f)
    # Audio read in a second pass: demuxing video to EOF consumes the shared
    # packet queue, so decoding audio in the same container returns nothing.
    achan, arate, adata = 0, 0, None
    with av.open(src) as ins:
        as_ = next((s for s in ins.streams if s.type == "audio"), None)
        if as_ is not None:
            arate = as_.codec_context.sample_rate or 32000
            achan = as_.codec_context.channels or 2
            n = []
            for fr in ins.decode(as_):
                a = fr.to_ndarray()
                if a.shape[0] == 1 and achan > 1:
                    a = a.repeat(achan, axis=0)
                n.append(a)
            if n:
                adata = np.concatenate(n, axis=-1).astype(np.float32)
    lay = ""
    with av.open(tmp, "w", format="mp4") as out:
        vout = out.add_stream("h264", rate=FPS)
        vout.width, vout.height = w, h
        vout.pix_fmt = "yuv420p"
        vout.options = {"crf": "18", "preset": "medium"}
        # Declare the audio stream up front (before any muxing): the mp4 muxer
        # rebases packets against streams known at start; adding it after the
        # video flush raises "Cannot rebase to zero time".
        aout = None
        if adata is not None:
            lay = "stereo" if achan > 1 else "mono"
            aout = out.add_stream("aac", rate=arate)
            aout.layout = lay
            aout.sample_rate = arate
        for f in vframes:
            for pkt in vout.encode(f):
                out.mux(pkt)
        for pkt in vout.encode(None):
            out.mux(pkt)
        if aout is not None:
            step = 1024
            for i in range(0, adata.shape[-1], step):
                blk = adata[:, i:i + step]
                af = av.AudioFrame.from_ndarray(blk, format="fltp", layout=lay)
                af.sample_rate = arate
                af.pts = i
                for pkt in aout.encode(af):
                    out.mux(pkt)
            for pkt in aout.encode(None):
                out.mux(pkt)
    os.replace(tmp, dst)
    print("[material] video refit 24fps -> input/%s (audio=%s)" % (name, adata is not None))
    return name, adata is not None


def _stage_audio(src, tag, idx):
    """Copy a local audio file into ComfyUI's input dir; return its file name."""
    if not os.path.isfile(src):
        sys.exit("audio not found: %s" % src)
    name = "%s_%d_%s" % (tag, idx, os.path.basename(src))
    dst = os.path.join(INPUT_DIR, name)
    if os.path.abspath(src) != os.path.abspath(dst):
        shutil.copy2(src, dst)
    print("[material] audio -> input/%s" % name)
    return name


def ref2v_length(dur):
    """--dur seconds -> ref2v node frame count, aligned to the model's
    L % 17 == 5 lattice (official template: 5 s -> 124 frames)."""
    base = max(5, int(round(dur * FPS)))
    return base + ((5 - base) % 17)


def ray_initializer_node(epoch=_RUN_EPOCH):
    return {"class_type": "RayInitializer", "inputs": {
        "ray_cluster_address": "local", "ray_cluster_namespace": "default",
        "GPU": 2, "ulysses_degree": 2, "ring_degree": 1, "cfg_degree": 1,
        "dp_degree": 1, "sync_ulysses": False, "clear_vram_after_sampling": False,
        "FSDP": True, "FSDP_CPU_OFFLOAD": False, "XFuser_attention": "SAGE_FP16",
        "skip_comm_test": False, "use_mmap": True, "RAYLIGHT_ULYSSES_KV_INT8": "off",
        "reuse_epoch": epoch}}


def ray_base(unet, w, h, save_prefix, steps, epoch=_RUN_EPOCH):
    return {
        "141": ray_initializer_node(epoch),
        "130": {"class_type": "H3MultiGPUCLIPLoader", "inputs": {
            "clip_name": CLIP, "gpu_ids": "0,1", "offload_after_encode": True}},
        "121": {"class_type": "VAELoader", "inputs": {"vae_name": VAE_VIDEO}},
        "122": {"class_type": "VAELoader", "inputs": {"vae_name": VAE_AUDIO}},
        "142": {"class_type": "RayUNETLoader", "inputs": {
            "unet_name": unet, "weight_dtype": "default", "ray_actors_init": ["141", 0]}},
        "125": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "res_multistep"}},
        "145": {"class_type": "RayBasicGuider", "inputs": {"ray_actors": ["142", 0], "conditioning": ["133", 0]}},
        "143": {"class_type": "RayBasicScheduler", "inputs": {
            "ray_actors": ["142", 0], "scheduler": "simple", "steps": steps, "denoise": 1}},
        "144": {"class_type": "XFuserSamplerCustomAdvanced", "inputs": {
            "add_noise": True, "noise_seed": 0,
            "guider": ["145", 0], "sampler": ["125", 0], "sigmas": ["143", 0],
            "latent_image": ["133", 1]}},
        "124": {"class_type": "VAEDecode", "inputs": {"samples": ["144", 0], "vae": ["121", 0]}},
        "123": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["144", 0], "vae": ["122", 0]}},
        "132": {"class_type": "CreateVideo", "inputs": {
            "images": ["124", 0], "fps": 24, "audio": ["123", 0], "bit_depth": 8}},
        "92": {"class_type": "SaveVideo", "inputs": {
            "video": ["132", 0], "filename_prefix": save_prefix, "format": "auto", "codec": "auto"}},
    }


def add_analyze_and_save(g, clip_idx):
    d = widget_defaults(ANALYZE_CLS)
    d.pop("images", None)
    g["200"] = {"class_type": ANALYZE_CLS, "inputs": {"images": ["124", 0], **d}}
    g["180"] = {"class_type": SAVE_CLS, "inputs": {
        "latent": ["144", 0], "filename_prefix": PREFIX, "clip_index": clip_idx,
        "handover": ["200", 0]}}


def seg0_graph(prompt, w, h, dur, seed, tag, steps, first_img=None, last_img=None,
               epoch=_RUN_EPOCH):
    """fl2va first-segment graph (H3ContinuousStartV14). Optional first/last
    image become Start.first_frame / last_frame (frame-locked anchors)."""
    g = ray_base(UNET_FL2VA, w, h, "video/chain/%s/seg_0" % tag, steps, epoch)
    g["144"]["inputs"]["noise_seed"] = seed
    g["133"] = {"class_type": START_CLS, "inputs": {
        "clip": ["130", 0], "vae": ["121", 0], "prompt": prompt,
        "width": w, "height": h, "duration": dur, "ref_image_size": "match"}}
    if first_img is not None:
        fname = _stage_image(first_img, tag, 1)
        g["300"] = {"class_type": "LoadImage", "inputs": {"image": fname}}
        g["133"]["inputs"]["first_frame"] = ["300", 0]
    if last_img is not None:
        fname = _stage_image(last_img, tag, 2)
        g["301"] = {"class_type": "LoadImage", "inputs": {"image": fname}}
        g["133"]["inputs"]["last_frame"] = ["301", 0]
    add_analyze_and_save(g, 1)
    return g


def ref2va_graph(prompt, w, h, dur, seed, tag, steps, ref_images, ref_videos,
                 ref_audios=None, ref_image_size="match", epoch=_RUN_EPOCH):
    """ref2va first-segment graph: official MiniMaxH3ReferenceToVideo +
    minimax_h3_ref2va unet. ref_images -> ref_image_N (LoadImage), ref_videos ->
    ref_video_N (+ref_video_audio_N if it carries sound) via LoadVideo +
    GetVideoComponents (video refit to 24 fps first), standalone audio ->
    ref_audio_N (LoadAudio)."""
    g = ray_base(UNET_REF2VA, w, h, "video/chain/%s/seg_0" % tag, steps, epoch)
    g["144"]["inputs"]["noise_seed"] = seed
    g["133"] = {"class_type": "MiniMaxH3ReferenceToVideo", "inputs": {
        "clip": ["130", 0], "vae": ["121", 0], "audio_vae": ["122", 0],
        "prompt": prompt, "width": w, "height": h,
        "length": ref2v_length(dur), "ref_image_size": ref_image_size}}
    for i, p in enumerate(ref_images):
        fname = _stage_image(p, tag, 100 + i)
        nid = "320%d" % i
        g[nid] = {"class_type": "LoadImage", "inputs": {"image": fname}}
        g["133"]["inputs"]["ref_images.ref_image_%d" % i] = [nid, 0]
    for i, p in enumerate(ref_videos):
        fname, has_audio = _refit_24fps(p, tag, 200 + i)
        vn = "340%d" % i
        cn = "341%d" % i
        g[vn] = {"class_type": "LoadVideo", "inputs": {"file": fname}}
        g[cn] = {"class_type": "GetVideoComponents", "inputs": {"video": [vn, 0]}}
        g["133"]["inputs"]["ref_videos.ref_video_%d" % i] = [cn, 0]
        if has_audio:
            g["133"]["inputs"]["ref_video_audios.ref_video_audio_%d" % i] = [cn, 1]
    for i, p in enumerate(ref_audios or []):
        fname = _stage_audio(p, tag, 300 + i)
        an = "360%d" % i
        g[an] = {"class_type": "LoadAudio", "inputs": {"audio": fname}}
        g["133"]["inputs"]["ref_audios.ref_audio_%d" % i] = [an, 0]
    add_analyze_and_save(g, 1)
    return g


def cont_graph(prompt, w, h, dur, seed, tag, clip_idx, ctx_abs, steps, epoch=_RUN_EPOCH):
    g = ray_base(UNET_FL2VA, w, h, "video/chain/%s/seg_%d" % (tag, clip_idx - 1), steps, epoch)
    g["144"]["inputs"]["noise_seed"] = seed
    g["150"] = {"class_type": LOAD_CLS, "inputs": {"latent_path": ctx_abs, "clip_index": clip_idx - 1}}
    g["133"] = {"class_type": CONT_CLS, "inputs": {
        "clip": ["130", 0], "vae": ["121", 0], "previous_latent": ["150", 0],
        "prompt": prompt, "width": w, "height": h, "duration": dur,
        "masked_context_frames": "39", "audio_feather_ticks": 0,
        "ref_image_size": "match", "duration_mode": "Net New Content",
        "audio_tail_carryover": "Full Previous Tail",
        "handover": ["150", 3]}}
    add_analyze_and_save(g, clip_idx)
    return g


def submit(g, client):
    r = http_json(API + "/prompt", {"prompt": g, "client_id": client})
    if r.get("error"):
        sys.exit("submit error: %s" % json.dumps(r["error"], ensure_ascii=False)[:2500])
    return r["prompt_id"]


def wait_done(pid, client, timeout_s=9000):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            h = http_json(API + "/history/%s" % pid)
        except Exception:
            h = {}
        if h and pid in h:
            st = h[pid].get("status", {})
            if st.get("status_str") == "error":
                for msg in st.get("messages", []):
                    if msg[0] == "execution_error":
                        m = msg[1]
                        print("NODE ERROR %s %s\n%s" % (
                            m.get("node_id"), m.get("exception_type"),
                            str(m.get("exception_message"))[-8000:]))
                sys.exit(1)
            return h[pid], time.time() - t0
        time.sleep(5)
    sys.exit("wait timeout")


def saved_mp4(hist):
    out = hist.get("outputs", {}).get("92", {})
    for key in ("videos", "gifs", "images", "preview_videos", "preview"):
        v = out.get(key)
        if isinstance(v, list):
            for it in v:
                if isinstance(it, dict):
                    f = os.path.join(OUTPUT, it.get("subfolder", ""), it.get("filename", ""))
                    if os.path.isfile(f):
                        return f
    return None


def handover_end(clip_idx):
    p = slot_path(clip_idx)
    if not os.path.isfile(p):
        return None
    with safe_open(p, framework="numpy") as h:
        raw = (h.metadata() or {}).get("handover_json")
    if not raw:
        return None
    d = json.loads(raw)
    return int(d.get("handover_end_frame", -1))


_BEAT_RE = re.compile(r"^([\d.]+)\s*(?:s|秒)?\s*[:：]\s*(.+)$")


def split_beats(beat_list, dur):
    """--beat list -> {seg_idx: [prompt-suffix,...]}, keyed by beat time.

    Each element is ONE independent argv string, so the description may contain
    arbitrary characters (;, :, full-width punctuation...) - no separator
    splicing anywhere. A beat that cannot be parsed aborts the run (no silent
    drops). seg = int(t // dur); rel is the offset within that shot.
    """
    out = {}
    for spec in beat_list or []:
        spec = spec.strip()
        if not spec:
            continue
        m = _BEAT_RE.match(spec)
        if not m:
            sys.exit("bad --beat: %r\n  format: <seconds>s:<description>, "
                     "e.g. --beat \"10s:爆炸\" (repeatable; one beat per argument)" % spec)
        t = float(m.group(1))
        txt = m.group(2).strip()
        if not txt:
            sys.exit("bad --beat %r: empty description" % spec)
        seg = int(t // dur)
        rel = t - seg * dur
        out.setdefault(seg, []).append("at about %.1fs into this shot: %s" % (rel, txt))
    return out


def clip_len(path):
    with av.open(path) as c:
        return sum(1 for _ in c.decode(video=0))


def video_pass(path, start, end):
    vframes = []
    with av.open(path) as c:
        vs = next((s for s in c.streams if s.type == "video"), None)
        idx = 0
        for fr in c.decode(vs):
            if start <= idx < end:
                vframes.append(fr)
            idx += 1
            if idx > end:
                break
    return vframes


def audio_pass(path, start, end):
    with av.open(path) as c:
        as_ = next((s for s in c.streams if s.type == "audio"), None)
        if as_ is None:
            return None, 0
        sr = as_.codec_context.sample_rate or 32000
        a0 = int(start * sr / FPS)
        a1 = int(end * sr / FPS)
        chunks = []
        n = 0
        for fr in c.decode(as_):
            arr = fr.to_ndarray()
            L = arr.shape[-1]
            if n + L <= a0:
                n += L
                continue
            if n >= a1:
                break
            c0 = max(0, a0 - n)
            c1 = min(L, a1 - n)
            chunks.append(arr[..., c0:c1])
            n += L
        audio = np.concatenate(chunks, axis=-1) if chunks else None
    return audio, sr


def stitch(pieces, out_path):
    all_v = []
    all_a = []
    sr = 32000
    for path, s0, e0 in pieces:
        all_v += video_pass(path, s0, e0)
        aud, sr_i = audio_pass(path, s0, e0)
        if aud is not None:
            sr = sr_i
            all_a.append(aud)
    width = all_v[0].width
    height = all_v[0].height
    from fractions import Fraction
    with av.open(out_path, "w") as out:
        vs = out.add_stream("h264", rate=FPS)
        vs.width, vs.height = width, height
        vs.pix_fmt = "yuv420p"
        vs.time_base = Fraction(1, FPS)
        vs.options = {"crf": "18", "preset": "medium"}
        a_s = None
        if all_a:
            C = max(a.shape[0] for a in all_a)
            lay = "stereo" if C > 1 else "mono"
            a_s = out.add_stream("aac", rate=sr)
            a_s.layout = lay
            a_s.sample_rate = sr
            a_s.time_base = Fraction(1, sr)
            a_s.codec_context.time_base = Fraction(1, sr)
        vpts = 0
        for f in all_v:
            fr = f.reformat(width=width, height=height, format="yuv420p")
            # Decoded/reformatted frames carry time_base 1/12288 (h264 mp4);
            # encode() interprets frame.pts in that time base, so stepping pts
            # by 1 would land every frame near t=0 (all-0 timestamps). Advance
            # pts by one frame in the frame time base instead.
            den = fr.time_base.denominator if fr.time_base else FPS
            fr.pts = vpts
            vpts += round(den / FPS)
            for pkt in vs.encode(fr):
                out.mux(pkt)
        for pkt in vs.encode(None):
            out.mux(pkt)
        if a_s is not None:
            segs = []
            for a in all_a:
                if a.shape[0] == 1 and C > 1:
                    a = a.repeat(C, axis=0)
                segs.append(a)
            st = np.concatenate(segs, axis=-1).astype(np.float32)
            step = 1024
            for i in range(0, st.shape[-1], step):
                blk = st[:, i:i + step]
                af = av.AudioFrame.from_ndarray(blk, format="fltp", layout=lay)
                af.sample_rate = sr
                af.pts = i
                for pkt in a_s.encode(af):
                    out.mux(pkt)
            for pkt in a_s.encode(None):
                out.mux(pkt)


def count_existing():
    n = 0
    while os.path.isfile(slot_path(n + 1)):
        n += 1
    return n


_comfy_managed = {"needed": False}


def service_stop():
    """Stop the ComfyUI service (stop.sh). check=False: harmless if already down."""
    print("[lifecycle] stopping ComfyUI (stop.sh)...", flush=True)
    subprocess.run(["bash", os.path.expanduser("~/ComfyUI-Deploy/stop.sh")], check=False)


def _signal_cleanup(signum, frame):
    name = signal.Signals(signum).name
    print("received signal %d (%s) -> stopping ComfyUI" % (signum, name), flush=True)
    if _comfy_managed.get("needed"):
        try:
            service_stop()
        except Exception as e:
            print("service_stop error: %r" % e, flush=True)
    os._exit(128 + signum)


def restart_service():
    """Default per-run clear: stop.sh then start-comfyui-for-minimax-h3.sh.
    Kills ComfyUI + ray workers so any resident FSDP from a previous run is gone
    and the GPU is empty for this run's CLIP dispatch. Measured ~10-12 s to :8188
    back up (vs ~30 s for the in-place prewarm rebuild) and fully resets
    process/VRAM state."""
    import subprocess
    print("[clear] stopping ComfyUI/ray (stop.sh)...")
    subprocess.run(["bash", os.path.expanduser("~/ComfyUI-Deploy/stop.sh")], check=False)
    time.sleep(2)
    _start_sh = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "start-comfyui-for-minimax-h3.sh")
    print("[clear] starting ComfyUI (start-comfyui-for-minimax-h3.sh)...")
    subprocess.run(["bash", _start_sh], check=False)
    t0 = time.time()
    while time.time() - t0 < 240:
        try:
            http_json(API + "/system_stats", timeout=5)
            print("[clear] service up after %.1fs" % (time.time() - t0))
            return
        except Exception:
            time.sleep(2)
    sys.exit("restart: service did not come up")


def prewarm_clear(client):
    """Clear WITHOUT restarting the service. Submits a queue containing only the
    RayInitializer (this run's reuse_epoch) plus a pass-through OUTPUT node. If a
    previous run left resident FSDP, the reuse_epoch mismatch makes spawn_actor
    rebuild ray (ray.shutdown + init + fresh workers), which kills the old workers
    and frees VRAM (~30 s measured) before any segment queue runs. If the service
    is already clean/current, it just (re)inits once. Segment queues then reuse
    these workers (segments 2..N keep the model)."""
    graph = {
        "141": ray_initializer_node(),
        "999": {"class_type": "RayCleanVRAMUsed",
                "inputs": {"anything": ["141", 0]}},
    }
    print("[clear] prewarm queue (rebuild/reuse ray workers)...")
    pid = submit(graph, client)
    wait_done(pid, client)
    print("[clear] ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--beat", action="append", default=None, metavar="SECs:DESC",
                    help="scene beat, repeatable, one per argument: "
                         "'--beat \"10s:爆炸\" --beat \"25s:风暴将至\"'. Time is seconds "
                         "(integer/decimal, optional s/秒 suffix), then ':' or '：' and a "
                         "free-form description (may contain ; , : etc.). Each --beat is "
                         "applied to the shot whose time-window contains it.")
    ap.add_argument("--segments", type=int, default=4)
    ap.add_argument("--dur", type=float, default=5.0)
    ap.add_argument("--width", type=int, default=864)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--tag", default="chain")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--first-image", default=None, metavar="IMG",
                    help="fl2va first segment: Start.first_frame (first-frame anchor).")
    ap.add_argument("--last-image", default=None, metavar="IMG",
                    help="fl2va first segment: Start.last_frame (end-frame anchor).")
    ap.add_argument("--ref-image", action="append", default=None, metavar="IMG",
                    help="ref2va first segment: reference image (repeatable, <=9).")
    ap.add_argument("--ref-video", action="append", default=None, metavar="MP4",
                    help="ref2va first segment: reference video (repeatable, <=3).")
    ap.add_argument("--ref-audio", action="append", default=None, metavar="WAV/MP4",
                    help="ref2va first segment: standalone reference audio, e.g. footsteps/"
                         "music to steer generated sound (repeatable, <=3).")
    ap.add_argument("--ref-image-size", choices=["match", "max"], default="match",
                    help="ref2va reference sizing (default match).")
    ap.add_argument("--clear", choices=["restart", "prewarm", "none"], default="restart",
                    help="(Ignored in managed lifecycle mode) Per-run GPU clear before "
                         "generating segments: restart=stop/start service; prewarm=in-place "
                         "ray rebuild; none=keep current workers. With managed ComfyUI the "
                         "setup always does stop-then-start (and stops again on exit).")
    a = ap.parse_args()
    # Managed ComfyUI lifecycle: on SIGTERM/SIGINT (e.g. a web cancel sends
    # SIGTERM) we stop the service before exiting so the GPU is never left busy.
    signal.signal(signal.SIGTERM, _signal_cleanup)
    signal.signal(signal.SIGINT, _signal_cleanup)
    beats = split_beats(a.beat, a.dur)
    client = "cd2-" + a.tag + "-" + str(random.randint(1000, 9999))
    seed = a.seed if a.seed is not None else random.randint(0, 2**63 - 1)
    use_ref2v = bool(a.ref_image or a.ref_video or a.ref_audio)
    if use_ref2v and (a.first_image or a.last_image):
        sys.exit("--first/--last-image (fl2va) cannot be combined with "
                 "--ref-image/--ref-video/--ref-audio (ref2va); pick one first-segment engine")
    if a.ref_image and len(a.ref_image) > 9:
        sys.exit("ref2va supports at most 9 reference images")
    if a.ref_video and len(a.ref_video) > 3:
        sys.exit("ref2va supports at most 3 reference videos")
    if a.ref_audio and len(a.ref_audio) > 3:
        sys.exit("ref2va supports at most 3 reference audios")
    # Continuation segments (2..N) always run the fl2va unet. reuse_epoch is
    # chosen dynamically below (see change-point comment): kept stable while
    # consecutive segments share a prompt, bumped at each prompt change. A
    # material-bearing first segment additionally restarts the service before
    # segment 2 (VAE pool + fl2va weight switch), which clears ray anyway.
    epoch_first = _RUN_EPOCH
    existing = count_existing()
    run_needed = existing < a.segments
    print("existing clips:", existing, "target:", a.segments,
          "first-segment:", "ref2va" if use_ref2v else "fl2va")
    if not run_needed:
        print("all clips present")
        if not a.merge:
            return
    # ---- managed ComfyUI lifecycle ----
    # This script owns the ComfyUI process for the duration of a real run. Before
    # any segment work we take the service over: stop it if it is up, then start
    # it, and wait for :8188 (the historical restart_service behaviour, now the
    # mandatory front door instead of the --clear option). On exit -- however it
    # happens (success, failure/SystemExit, or SIGTERM/SIGINT from a web cancel)
    # -- we stop it again so the GPUs are free between runs. Merge-only runs and
    # `all clips present` runs never touch the service.
    if run_needed:
        # take over no matter the current state: run down -> start; up -> stop+start
        restart_service()
        _comfy_managed["needed"] = True
    try:
        raw = {}
        material = use_ref2v or bool(a.first_image or a.last_image)
        # FSDP reuse across segments is only safe when the next segment re-encodes
        # the SAME prompt as the previous one (cond cache HIT -> no Qwen dispatch).
        # The moment seg_prompt changes (beats inject text per shot, or a resume
        # swaps the story) the continuation must dispatch Qwen (~24 GB over both
        # cards), which cannot coexist with resident FSDP workers (11.3 GB/card).
        # We therefore keep the reuse_epoch only while consecutive segments share a
        # prompt, and bump the epoch at each prompt change point so that segment's
        # encode runs on an empty GPU (ray rebuild); following same-prompt segments
        # then reuse the freshly loaded FSDP again.
        prev_prompt, prev_epoch = None, epoch_first
        for ci in range(existing + 1, a.segments + 1):
            seg_prompt = a.prompt
            if beats.get(ci - 1):
                seg_prompt = a.prompt + "\n\n" + " ".join(beats[ci - 1])
            if ci == 1:
                epoch = epoch_first
                if use_ref2v:
                    g = ref2va_graph(seg_prompt, a.width, a.height, a.dur, seed, a.tag, a.steps,
                                     ref_images=a.ref_image or [], ref_videos=a.ref_video or [],
                                     ref_audios=a.ref_audio or [], ref_image_size=a.ref_image_size,
                                     epoch=epoch_first)
                else:
                    g = seg0_graph(seg_prompt, a.width, a.height, a.dur, seed, a.tag, a.steps,
                                   a.first_image, a.last_image, epoch_first)
            else:
                # The material-bearing first segment VAE-encodes images/videos in the
                # ComfyUI process, leaving a GPU0 allocator pool that pushes resident
                # continuation sampling OOM by ~100 MB. Restart once before segment 2
                # so the continuation starts from a clean process (measured same
                # effect as the fl2va->ref2va weight switch already handled above).
                if ci == 2 and existing == 0 and material:
                    restart_service()
                # prompt change point -> epoch bump so the guard rebuilds ray
                # (shutdown workers, freeing the GPU for this segment's Qwen
                # dispatch); identical prompt keeps the previous epoch -> FSDP reuse.
                if prev_prompt is not None and seg_prompt == prev_prompt:
                    epoch = prev_epoch
                else:
                    epoch = epoch_first + ci
                g = cont_graph(seg_prompt, a.width, a.height, a.dur, seed, a.tag, ci,
                               slot_path(ci - 1), a.steps, epoch)
            print("[clip%d] submit..." % ci)
            prev_prompt, prev_epoch = seg_prompt, epoch
            pid = submit(g, client)
            hist, el = wait_done(pid, client)
            mp4 = saved_mp4(hist)
            if mp4 is None:
                sys.exit("clip%d: no mp4 saved" % ci)
            raw[ci] = mp4
            print("[clip%d] done %.1fs -> %s" % (ci, el, os.path.basename(mp4)))
        if a.merge:
            pieces = []
            for ci in range(1, a.segments + 1):
                mp4 = raw.get(ci)
                if not mp4:
                    cands = sorted(glob.glob(os.path.join(
                        OUTPUT, "video/chain/%s/seg_%d_*.mp4" % (a.tag, ci - 1))))
                    mp4 = cands[-1] if cands else None
                if not mp4:
                    sys.exit("merge: no raw mp4 for clip %d" % ci)
                n = clip_len(mp4)
                h0 = 0 if ci == 1 else HEAD
                safe = handover_end(ci)
                e0 = safe if (safe and safe > h0 and ci < a.segments) else n
                pieces.append((mp4, h0, e0))
            final_path = os.path.join(OUTPUT, "final_%s.mp4" % a.tag)
            print("stitching:", [(os.path.basename(p[0]), p[1], p[2]) for p in pieces])
            stitch(pieces, final_path)
            print("FINAL", final_path)
    finally:
        if _comfy_managed.get("needed"):
            service_stop()


if __name__ == "__main__":
    main()
