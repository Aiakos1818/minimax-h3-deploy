#!/usr/bin/env python3
"""chain_director_v3.py -- ChainDirector resident-UNet mode (int4 CLIP on demand).

Same continuation engine as chain_director_v2.py (Herrgotts masked-AV, one queue
per clip, slot latents, handover-based merge), but the segment graph is wired
like the verified workflow
(~/ComfyUI-Deploy/user/default/workflows/minimax_h3_int4clip_int8unet_raylight.json):

  * RayInitializer runs clear_vram_after_sampling=False so the FSDP UNet shards
    stay resident across segments and no segment pays an FSDP reload. v2 bumped
    reuse_epoch at every prompt change point to force a ray rebuild; here the epoch
    is a single value for the whole run.
  * H3MultiGPUCLIPLoader runs the int4 text encoder with offload_after_encode=True:
    the encoder is dispatched for an encode and rebound to its CPU masters right
    after (measured 11.3s up / 0.9s down). The cond cache is consulted before the
    dispatch, so the normal chain -- one prompt, re-encoded never -- encodes once
    for the whole run and every later segment reuses the conditioning with no
    dispatch at all. Keeping the encoder off the cards during sampling is what
    leaves room for long segments; the resident-encoder variant only managed ~107
    frames at 864x480 before its second segment ran out of VRAM.
  * The video VAE is the int8 convrot one and is evicted three times per segment
    through UnloadVideoVAE: before the (cached) CLIP loader runs, after the
    start/continue conditioning, and after decoding. The post-decode eviction also
    clears the ray workers' cached CUDA pool -- that pool is what otherwise makes
    the next segment's keyframe/decode work OOM next to the resident shards.
  * Audio decode runs on gpu:1 (SelectVAEDevice).

First-segment material is fl2va only: text, or --first-image/--last-image
first/last-frame anchors. Those anchors become VAE keyframes and never touch the
Qwen vision tower, so H3_MP_KEEP_VISUAL_CPU=1 keeps that ~1.1 GiB off GPU0.
Continuation segments are always H3ContinuousContinueV14 on the fl2va unet.

Managed ComfyUI lifecycle: same as v2. Before the first real segment the service
is taken over (stop, start, wait for :8188) and it is stopped again on exit --
success, failure/SystemExit, or SIGTERM/SIGINT. --clear is accepted and ignored;
the service is always restarted per run. --clean wipes the global chain slots so a
new chain does not resume a previous tag's segments.

Usage:
  ~/ComfyUI-Deploy/comfyenv/bin/python scripts/chain_director_v3.py --tag myfilm \
    --segments 6 --dur 5 --prompt "..." --beat "10s:..." --merge
"""
import argparse, glob, json, os, random, re, shutil, signal, subprocess, sys, time, urllib.request

import numpy as np

from safetensors import safe_open

import av

HOME = os.path.expanduser("~/MiniMax-H3-Deploy")
API = "http://127.0.0.1:8188"
OUTPUT = HOME + "/output"
INPUT_DIR = os.path.expanduser("~/ComfyUI-Deploy/input")
CLIP = "qwen3vl_32b_minimax_h3_int4_convrot.safetensors"
VAE_VIDEO = "minimax_h3_video_vae_int8_convrot.safetensors"
VAE_AUDIO = "minimax_h3_audio_vae_fp32.safetensors"
UNET_FL2VA = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
AUDIO_VAE_DEVICE = "gpu:1"              # audio VAE decode on the second card
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

# One tag for the whole run: every segment queues the same RayInitializer, so the
# persist guard reuses the resident workers/FSDP instead of rebuilding ray.
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


def ray_initializer_node(epoch=_RUN_EPOCH):
    return {"class_type": "RayInitializer", "inputs": {
        "ray_cluster_address": "local", "ray_cluster_namespace": "default",
        "GPU": 2, "ulysses_degree": 2, "ring_degree": 1, "cfg_degree": 1,
        "dp_degree": 1, "sync_ulysses": True, "clear_vram_after_sampling": False,
        "FSDP": True, "FSDP_CPU_OFFLOAD": False, "XFuser_attention": "SAGE_FP16",
        "skip_comm_test": False, "use_mmap": True, "RAYLIGHT_ULYSSES_KV_INT8": "v",
        "reuse_epoch": epoch}}


def ray_base(save_prefix, steps, epoch=_RUN_EPOCH):
    """Resident base graph. 904/905 evict the VAE around the CLIP use; 903 evicts it
    after decoding and also frees the ray workers' cached CUDA pool."""
    return {
        "141": ray_initializer_node(epoch),
        "130": {"class_type": "H3MultiGPUCLIPLoader", "inputs": {
            "clip_name": CLIP, "gpu_ids": "0,1", "offload_after_encode": True}},
        "121": {"class_type": "VAELoader", "inputs": {"vae_name": VAE_VIDEO}},
        "122": {"class_type": "VAELoader", "inputs": {"vae_name": VAE_AUDIO}},
        "902": {"class_type": "SelectVAEDevice", "inputs": {
            "vae": ["122", 0], "device": AUDIO_VAE_DEVICE}},
        "142": {"class_type": "RayUNETLoader", "inputs": {
            "unet_name": UNET_FL2VA, "weight_dtype": "default", "ray_actors_init": ["141", 0]}},
        "125": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "res_multistep"}},
        "145": {"class_type": "RayBasicGuider", "inputs": {
            "ray_actors": ["142", 0], "conditioning": ["905", 0]}},
        "143": {"class_type": "RayBasicScheduler", "inputs": {
            "ray_actors": ["142", 0], "scheduler": "simple", "steps": steps, "denoise": 1}},
        "144": {"class_type": "XFuserSamplerCustomAdvanced", "inputs": {
            "add_noise": True, "noise_seed": 0,
            "guider": ["145", 0], "sampler": ["125", 0], "sigmas": ["143", 0],
            "latent_image": ["133", 1]}},
        "124": {"class_type": "VAEDecode", "inputs": {"samples": ["144", 0], "vae": ["121", 0]}},
        "123": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["144", 0], "vae": ["902", 0]}},
        "903": {"class_type": "UnloadVideoVAE", "inputs": {
            "vae": ["121", 0], "anything": ["124", 0], "ray_actors": ["142", 0]}},
        "132": {"class_type": "CreateVideo", "inputs": {
            "images": ["903", 0], "fps": 24, "audio": ["123", 0], "bit_depth": 8}},
        "92": {"class_type": "SaveVideo", "inputs": {
            "video": ["132", 0], "filename_prefix": save_prefix, "format": "auto", "codec": "auto"}},
        "904": {"class_type": "UnloadVideoVAE", "inputs": {"vae": ["121", 0], "anything": ["130", 0]}},
        "905": {"class_type": "UnloadVideoVAE", "inputs": {"vae": ["121", 0], "anything": ["133", 0]}},
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
    """fl2va first segment (H3ContinuousStartV14). Optional first/last image become
    Start.first_frame / last_frame (VAE-encoded anchors)."""
    g = ray_base("video/chain/%s/seg_0" % tag, steps, epoch)
    g["144"]["inputs"]["noise_seed"] = seed
    g["133"] = {"class_type": START_CLS, "inputs": {
        "clip": ["904", 0], "vae": ["121", 0], "prompt": prompt,
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


def cont_graph(prompt, w, h, dur, seed, tag, clip_idx, ctx_abs, steps, epoch=_RUN_EPOCH):
    g = ray_base("video/chain/%s/seg_%d" % (tag, clip_idx - 1), steps, epoch)
    g["144"]["inputs"]["noise_seed"] = seed
    g["150"] = {"class_type": LOAD_CLS, "inputs": {"latent_path": ctx_abs, "clip_index": clip_idx - 1}}
    g["133"] = {"class_type": CONT_CLS, "inputs": {
        "clip": ["904", 0], "vae": ["121", 0], "previous_latent": ["150", 0],
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
    arbitrary characters (;, :, full-width punctuation...). A beat that cannot be
    parsed aborts the run (no silent drops). seg = int(t // dur); rel is the offset
    within that shot.
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


def clean_slots():
    """Slot names are global (chain_0000N.safetensors, no tag in them), so a new
    chain must wipe them or it would resume whatever the last tag produced."""
    removed = 0
    for p in sorted(glob.glob(os.path.join(OUTPUT, SLOT_DIR, "%s_*.safetensors" % SLOT_BASE))):
        os.remove(p)
        removed += 1
    print("[clean] removed %d chain slot(s)" % removed)


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
    """Per-run clear: stop.sh then the H3 launcher (which exports the RAY env,
    --output-directory, --lowvram --reserve-vram 11.5, --use-sage-attention and
    H3_MP_RETAIN_CPU_WEIGHTS=1). Kills ComfyUI + ray workers so no FSDP shard from
    a previous run survives into this one."""
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
    ap.add_argument("--clean", action="store_true",
                    help="wipe the global chain slots before running (fresh chain).")
    ap.add_argument("--clear", choices=["restart", "prewarm", "none"], default="restart",
                    help="(Ignored) The managed lifecycle always does stop-then-start "
                         "before a real run and stops the service on exit.")
    a = ap.parse_args()
    # Managed ComfyUI lifecycle: on SIGTERM/SIGINT (e.g. a web cancel sends
    # SIGTERM) we stop the service before exiting so the GPU is never left busy.
    signal.signal(signal.SIGTERM, _signal_cleanup)
    signal.signal(signal.SIGINT, _signal_cleanup)
    beats = split_beats(a.beat, a.dur)
    client = "cd3-" + a.tag + "-" + str(random.randint(1000, 9999))
    seed = a.seed if a.seed is not None else random.randint(0, 2**63 - 1)
    if a.clean:
        clean_slots()
    existing = count_existing()
    run_needed = existing < a.segments
    print("existing clips:", existing, "target:", a.segments, "steps:", a.steps)
    if not run_needed:
        print("all clips present")
        if not a.merge:
            return
    # ---- managed ComfyUI lifecycle ----
    # The service is restarted once per run (clean CUDA state, no resident shards
    # from a previous run) and stopped on exit. Segments inside the run never
    # restart it: the FSDP UNet stays resident for all of them and the text
    # encoder is dispatched only when the conditioning cache misses.
    if run_needed:
        restart_service()
        _comfy_managed["needed"] = True
    try:
        raw = {}
        # Every segment queues the same epoch, so the FSDP UNet is never reloaded
        # (v2 bumped it at prompt change points to let a fresh encode dispatch onto
        # an empty GPU). A miss on the conditioning cache here costs one encoder
        # round trip (~12s) on top of the resident shards; a repeated prompt -- the
        # normal case for a single-prompt chain -- costs nothing.
        epoch = _RUN_EPOCH
        for ci in range(existing + 1, a.segments + 1):
            seg_prompt = a.prompt
            if beats.get(ci - 1):
                seg_prompt = a.prompt + "\n\n" + " ".join(beats[ci - 1])
            if ci == 1:
                g = seg0_graph(seg_prompt, a.width, a.height, a.dur, seed, a.tag, a.steps,
                               a.first_image, a.last_image, epoch)
            else:
                g = cont_graph(seg_prompt, a.width, a.height, a.dur, seed, a.tag, ci,
                               slot_path(ci - 1), a.steps, epoch)
            print("[clip%d] submit..." % ci)
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
