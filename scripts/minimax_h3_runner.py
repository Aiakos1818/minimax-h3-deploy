#!/usr/bin/env python3
"""minimax_h3_runner.py -- one MiniMax H3 clip per invocation.

Single-segment sibling of chain_director_v3.py: same resident int4-CLIP /
int8-UNet raylight base.  Two modes:
  --mode ref2v : node 133 is MiniMaxH3ReferenceToVideo; clip from a prompt plus
                 reference images / videos / audios.
  --mode t2v   : node 133 is MiniMaxH3ImageToVideo (fl2va UNet); prompt plus
                 optional first/last keyframes (none = t2v, first = i2v,
                 both = fl2v).
No slots, no continuation, no merge -- one task in, one mp4 out.

ComfyUI is resident: a run starts the service only when it is down, and always
leaves it up so the next run reuses the loaded FSDP shards and raylight workers.
The web console owns the explicit "release VRAM" action.

Usage (called by the web console; can be run by hand too):
  ~/ComfyUI-Deploy/comfyenv/bin/python scripts/minimax_h3_runner.py \
    --mode ref2v --prompt "..." --image ref.png --dur 5 --steps 20 \
    --out ~/MiniMax-H3-Deploy/output/ref2v/<job>.mp4
"""
import argparse, glob, json, os, random, re, shutil, signal, subprocess, sys, time, urllib.request

import numpy as np
import av

HOME = os.path.expanduser("~/MiniMax-H3-Deploy")
API = "http://127.0.0.1:8188"
OUTPUT = HOME + "/output"
INPUT_DIR = os.path.expanduser("~/ComfyUI-Deploy/input")
START_SH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "start-comfyui-for-minimax-h3.sh")
STOP_SH = os.path.expanduser("~/ComfyUI-Deploy/stop.sh")
STATE_PATH = HOME + "/.ref2v_service.json"
COMFY_LOG = os.path.expanduser("~/ComfyUI-Deploy/comfy.log")

CLIP = "qwen3vl_32b_minimax_h3_int4_convrot.safetensors"
VAE_VIDEO = "minimax_h3_video_vae_int8_convrot.safetensors"
VAE_AUDIO = "minimax_h3_audio_vae_fp32.safetensors"
UNET_REF2VA = "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
UNET_FL2VA = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
AUDIO_VAE_DEVICE = "gpu:1"
FPS = 24
MAX_IMAGES, MAX_VIDEOS, MAX_AUDIOS = 9, 3, 3
OUT_SUBDIRS = {"ref2v": "ref2v", "t2v": "t2v"}
MODE_UNET = {"ref2v": UNET_REF2VA, "t2v": UNET_FL2VA}


def http_json(url, data=None, timeout=120):
    if data is None:
        return json.load(urllib.request.urlopen(url, timeout=timeout))
    req = urllib.request.Request(url, data=json.dumps(data).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def log(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------- materials
def _stage_image(src, tag, idx):
    if not os.path.isfile(src):
        sys.exit("image not found: %s" % src)
    name = "%s_%d_%s" % (tag, idx, os.path.basename(src))
    dst = os.path.join(INPUT_DIR, name)
    if os.path.abspath(src) != os.path.abspath(dst):
        shutil.copy2(src, dst)
    log("[material] image -> input/%s" % name)
    return name


def _stage_audio(src, tag, idx):
    if not os.path.isfile(src):
        sys.exit("audio not found: %s" % src)
    name = "%s_%d_%s" % (tag, idx, os.path.basename(src))
    dst = os.path.join(INPUT_DIR, name)
    if os.path.abspath(src) != os.path.abspath(dst):
        shutil.copy2(src, dst)
    log("[material] audio -> input/%s" % name)
    return name


def _refit_24fps(src, tag, idx):
    """Re-encode a reference video to a 24 fps h264/aac mp4 in ComfyUI's input
    dir (no ffmpeg binary on the host). Returns (file name, has_audio)."""
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
            den = f.time_base.denominator if f.time_base else FPS
            f.pts = p
            p += round(den / FPS)
            vframes.append(f)
    achan, arate, adata = 0, 0, None
    with av.open(src) as ins:
        as_ = next((s for s in ins.streams if s.type == "audio"), None)
        if as_ is not None:
            arate = as_.codec_context.sample_rate or 32000
            achan = as_.codec_context.channels or 2
            chunks = []
            for fr in ins.decode(as_):
                a = fr.to_ndarray()
                if a.shape[0] == 1 and achan > 1:
                    a = a.repeat(achan, axis=0)
                chunks.append(a)
            if chunks:
                adata = np.concatenate(chunks, axis=-1).astype(np.float32)
    lay = ""
    with av.open(tmp, "w", format="mp4") as out:
        vout = out.add_stream("h264", rate=FPS)
        vout.width, vout.height = w, h
        vout.pix_fmt = "yuv420p"
        vout.options = {"crf": "18", "preset": "medium"}
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
    log("[material] video refit 24fps -> input/%s (audio=%s)" % (name, adata is not None))
    return name, adata is not None


def ref2v_length(dur):
    """seconds -> frame count on the model's L % 17 == 5 lattice (5s -> 124)."""
    base = max(5, int(round(dur * FPS)))
    return base + ((5 - base) % 17)


# ------------------------------------------------------------------- graph
def ray_base(save_prefix, steps, epoch, unet=UNET_REF2VA):
    return {
        "141": {"class_type": "RayInitializer", "inputs": {
            "ray_cluster_address": "local", "ray_cluster_namespace": "default",
            "GPU": 2, "ulysses_degree": 2, "ring_degree": 1, "cfg_degree": 1,
            "dp_degree": 1, "sync_ulysses": True, "clear_vram_after_sampling": False,
            "FSDP": True, "FSDP_CPU_OFFLOAD": False, "XFuser_attention": "SAGE_FP16",
            "skip_comm_test": False, "use_mmap": True, "RAYLIGHT_ULYSSES_KV_INT8": "v",
            "reuse_epoch": epoch}},
        "130": {"class_type": "H3MultiGPUCLIPLoader", "inputs": {
            "clip_name": CLIP, "gpu_ids": "0,1", "offload_after_encode": True}},
        "121": {"class_type": "VAELoader", "inputs": {"vae_name": VAE_VIDEO}},
        "122": {"class_type": "VAELoader", "inputs": {"vae_name": VAE_AUDIO}},
        "902": {"class_type": "SelectVAEDevice", "inputs": {
            "vae": ["122", 0], "device": AUDIO_VAE_DEVICE}},
        "142": {"class_type": "RayUNETLoader", "inputs": {
            "unet_name": unet, "weight_dtype": "default",
            "ray_actors_init": ["141", 0]}},
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


def ref2v_graph(prompt, w, h, dur, seed, tag, steps, epoch,
                images, videos, audios, ref_image_size="match",
                aspect="16:9 (Widescreen)", megapixels=0.4, multiple=32):
    g = ray_base("video/%s/%s" % (OUT_SUBDIRS["ref2v"], tag), steps, epoch)
    g["144"]["inputs"]["noise_seed"] = seed
    g["133"] = {"class_type": "MiniMaxH3ReferenceToVideo", "inputs": {
        "clip": ["904", 0], "vae": ["121", 0], "audio_vae": ["122", 0],
        "prompt": prompt,
        "length": ref2v_length(dur), "ref_image_size": ref_image_size}}
    if w and h:
        g["133"]["inputs"]["width"] = w
        g["133"]["inputs"]["height"] = h
    else:
        # let the ResolutionSelector node pick the canvas, same as the workflow
        g["115"] = {"class_type": "ResolutionSelector", "inputs": {
            "aspect_ratio": aspect, "megapixels": megapixels, "multiple": multiple}}
        g["133"]["inputs"]["width"] = ["115", 0]
        g["133"]["inputs"]["height"] = ["115", 1]
    for i, p in enumerate(images):
        fname = _stage_image(p, tag, 100 + i)
        nid = "320%d" % i
        g[nid] = {"class_type": "LoadImage", "inputs": {"image": fname}}
        g["133"]["inputs"]["ref_images.ref_image_%d" % i] = [nid, 0]
    for i, p in enumerate(videos):
        fname, has_audio = _refit_24fps(p, tag, 200 + i)
        vn, cn = "340%d" % i, "341%d" % i
        g[vn] = {"class_type": "LoadVideo", "inputs": {"file": fname}}
        g[cn] = {"class_type": "GetVideoComponents", "inputs": {"video": [vn, 0]}}
        g["133"]["inputs"]["ref_videos.ref_video_%d" % i] = [cn, 0]
        if has_audio:
            g["133"]["inputs"]["ref_video_audios.ref_video_audio_%d" % i] = [cn, 1]
    for i, p in enumerate(audios):
        fname = _stage_audio(p, tag, 300 + i)
        an = "360%d" % i
        g[an] = {"class_type": "LoadAudio", "inputs": {"audio": fname}}
        g["133"]["inputs"]["ref_audios.ref_audio_%d" % i] = [an, 0]
    return g


def t2v_graph(prompt, w, h, dur, seed, tag, steps, epoch,
              first_frame=None, last_frame=None,
              aspect="16:9 (Widescreen)", megapixels=0.4, multiple=32):
    """t2v / i2v / fl2v: prompt plus optional first and/or last keyframe."""
    g = ray_base("video/%s/%s" % (OUT_SUBDIRS["t2v"], tag), steps, epoch, unet=UNET_FL2VA)
    g["144"]["inputs"]["noise_seed"] = seed
    g["133"] = {"class_type": "MiniMaxH3ImageToVideo", "inputs": {
        "clip": ["904", 0], "vae": ["121", 0], "prompt": prompt,
        "length": ref2v_length(dur)}}
    if w and h:
        g["133"]["inputs"]["width"] = w
        g["133"]["inputs"]["height"] = h
    else:
        g["115"] = {"class_type": "ResolutionSelector", "inputs": {
            "aspect_ratio": aspect, "megapixels": megapixels, "multiple": multiple}}
        g["133"]["inputs"]["width"] = ["115", 0]
        g["133"]["inputs"]["height"] = ["115", 1]
    for key, src, idx in (("first_frame", first_frame, 400), ("last_frame", last_frame, 401)):
        if src:
            fname = _stage_image(src, tag, idx)
            nid = str(idx)
            g[nid] = {"class_type": "LoadImage", "inputs": {"image": fname}}
            g["133"]["inputs"][key] = [nid, 0]
    return g


# ----------------------------------------------------------------- service
def service_pid():
    out = subprocess.run(["pgrep", "-f", "main.py --listen 0.0.0.0 --port 8188"],
                         capture_output=True, text=True).stdout.split()
    return int(out[0]) if out else None


def _service_up():
    try:
        http_json(API + "/system_stats", timeout=5)
        return True
    except Exception:
        return False


def _state_read():
    try:
        with open(STATE_PATH) as fh:
            return json.load(fh)
    except Exception:
        return {}


def _state_write(epoch, unet):
    with open(STATE_PATH, "w") as fh:
        json.dump({"pid": service_pid(), "epoch": int(epoch), "unet": unet}, fh)


def ensure_service(unet=UNET_REF2VA):
    """Return the reuse_epoch. Resumes the running service and its loaded FSDP
    UNet when the state file vouches for it; otherwise starts the service and
    mints a new epoch (ray rebuild + UNet reload on the first prompt). A mode
    switch changes the UNet and therefore always mints a fresh epoch."""
    pid = service_pid()
    st = _state_read()
    if pid and st.get("pid") == pid and st.get("unet") == unet and st.get("epoch") and _service_up():
        log("[resident] reusing service pid=%s (epoch=%s)" % (pid, st["epoch"]))
        return int(st["epoch"])
    if not _service_up():
        log("[lifecycle] starting ComfyUI...")
        subprocess.run(["bash", START_SH], check=False)
        t0 = time.time()
        while time.time() - t0 < 300:
            if _service_up():
                log("[lifecycle] service up after %.1fs" % (time.time() - t0))
                break
            time.sleep(2)
        else:
            sys.exit("service did not come up")
    epoch = int(time.time_ns())
    _state_write(epoch, unet)
    return epoch


_cancel = {"hit": False}


def _on_signal(signum, frame):
    _cancel["hit"] = True
    log("[cancel] signal %d -> interrupting prompt" % signum)
    try:
        http_json(API + "/interrupt", {})
    except Exception:
        pass
    os._exit(128 + signum)


# ---------------------------------------------------------------- execution
def submit(g, client):
    r = http_json(API + "/prompt", {"prompt": g, "client_id": client})
    if r.get("error"):
        sys.exit("submit error: %s" % json.dumps(r["error"], ensure_ascii=False)[:2500])
    return r["prompt_id"]


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
# (pattern, stage key) -- first match wins, checked in order.
_COMFY_STAGE_RE = (
    (re.compile(r"reuse path exception -> REBUILD|skip reuse .*-> REBUILD|doing ray\.shutdown"), "ray_rebuild"),
    (re.compile(r"Applying FSDP to \w*MiniMaxH3Model"), "load_unet"),
    (re.compile(r"Requested to load MiniMaxH3TEModel_"), "load_clip"),
    (re.compile(r"H3 Qwen model-parallel timing"), "encode_clip"),
    (re.compile(r"Requested to load MiniMaxH3VideoVAE|Requested to load MiniMaxH3AudioVAE"), "load_vae"),
)
# Lines worth echoing to the job log so the console's diagnostic pane shows the
# model-load / ray / encode detail that otherwise only lives in comfy.log.
_COMFY_DETAIL_RE = re.compile(
    r"H3 Qwen model-parallel (timing|placement)|Applying FSDP|\[GUARD\]|"
    r"Requested to load|loaded partially|Parallel Degree|USP\] Initializing|"
    r"Using XFuser|NCCL version|COMM test passed|FSDP registered|Prompt executed in|VRAM\[")


def _clean_comfy_line(line):
    line = _ANSI_RE.sub("", line)
    m = re.search(r"\((?:RayWorker|pid=\d+)[^)]*\)\s*(.*)", line)
    if m:
        line = m.group(1)
    return line.strip()


def comfy_events(chunk):
    """Yield ("stage", key) / ("detail", text) events from a chunk of comfy.log."""
    for raw in chunk.decode("utf-8", "replace").splitlines():
        if "|" in raw and "%|" in raw:      # tqdm progress bars
            continue
        line = _clean_comfy_line(raw)
        if not line:
            continue
        for rx, key in _COMFY_STAGE_RE:
            if rx.search(line):
                yield ("stage", key)
                break
        if _COMFY_DETAIL_RE.search(line):
            yield ("detail", line[:300])


def sample_progress(start=0):
    """Latest 'done/total' step from ComfyUI's tqdm line in comfy.log (or None).

    Serial execution means at most one prompt is sampling, so the tail of the
    shared server log is ours to read. Only lines written after `start` (the
    byte offset captured when this prompt started sampling) count, so the
    previous run's last tqdm line is not mistaken for ours."""
    try:
        with open(COMFY_LOG, "rb") as f:
            size = f.seek(0, 2)
            f.seek(max(start, size - 16384))
            tail = f.read().decode("utf-8", "replace")
    except Exception:
        return None
    ms = re.findall(r"(\d+(?:\.\d+)?)/(\d+(?:\.\d+)?)\s*\[", tail)
    if not ms:
        return None
    a, b = ms[-1]
    return int(float(a)), int(float(b))


def wait_done(pid, timeout_s=9000):
    t0 = time.time()
    announced = False
    prog_off = 0
    last_prog = None
    last_stage = None
    comfy_off = os.path.getsize(COMFY_LOG) if os.path.exists(COMFY_LOG) else 0
    while time.time() - t0 < timeout_s:
        if _cancel["hit"]:
            sys.exit("cancelled")
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
                        log("NODE ERROR %s %s\n%s" % (
                            m.get("node_id"), m.get("exception_type"),
                            str(m.get("exception_message"))[-8000:]))
                sys.exit(1)
            return h[pid], time.time() - t0
        # surface ComfyUI-side phases (model loads, ray rebuild, encode)
        try:
            with open(COMFY_LOG, "rb") as f:
                f.seek(comfy_off)
                chunk = f.read()
                comfy_off = f.tell()
        except Exception:
            chunk = b""
        for kind, payload in comfy_events(chunk):
            if kind == "stage":
                if payload != last_stage:
                    last_stage = payload
                    log("[stage] %s" % payload)
            else:
                log("[comfy] %s" % payload)
        pr = sample_progress(prog_off)
        if pr and pr != last_prog:
            if not announced:
                # the tail may still hold the previous run's final 'N/N' bar;
                # only a genuine first bar (or a 1-step run) starts sampling
                if pr[0] == pr[1] and pr[1] != 1:
                    pr = None
                else:
                    announced = True
                    prog_off = os.path.getsize(COMFY_LOG) if os.path.exists(COMFY_LOG) else 0
                    log("[stage] sampling")
            if pr is not None:
                last_prog = pr
                log("[progress] %d/%d" % pr)
        time.sleep(3)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--image", action="append", default=[])
    ap.add_argument("--video", action="append", default=[])
    ap.add_argument("--audio", action="append", default=[])
    ap.add_argument("--dur", type=float, default=5.0)
    ap.add_argument("--width", type=int, default=None)
    ap.add_argument("--height", type=int, default=None)
    ap.add_argument("--aspect", default="16:9 (Widescreen)")
    ap.add_argument("--megapixels", type=float, default=0.4)
    ap.add_argument("--multiple", type=int, default=32)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--mode", choices=("ref2v", "t2v"), default="ref2v")
    ap.add_argument("--first-frame", default=None)
    ap.add_argument("--last-frame", default=None)
    ap.add_argument("--ref-image-size", choices=("match", "max"), default="match")
    ap.add_argument("--tag", default="h3")
    ap.add_argument("--out", default=None, help="final mp4 path (default output/<mode>/<tag>.mp4)")
    a = ap.parse_args()

    if a.mode == "ref2v":
        if len(a.image) > MAX_IMAGES:
            sys.exit("too many reference images (max %d)" % MAX_IMAGES)
        if len(a.video) > MAX_VIDEOS:
            sys.exit("too many reference videos (max %d)" % MAX_VIDEOS)
        if len(a.audio) > MAX_AUDIOS:
            sys.exit("too many reference audios (max %d)" % MAX_AUDIOS)
        if not a.image and not a.video and not a.audio:
            sys.exit("at least one reference image, video or audio is required")
    elif a.image or a.video or a.audio:
        sys.exit("reference media is only valid in ref2v mode")
    for p in (a.first_frame, a.last_frame):
        if p and not os.path.isfile(p):
            sys.exit("keyframe not found: %s" % p)

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)

    seed = a.seed if a.seed is not None else random.randint(0, 2**63 - 1)
    log("[stage] material")
    epoch = ensure_service(MODE_UNET[a.mode])
    client = "h3-" + a.tag + "-" + str(random.randint(1000, 9999))
    canvas = ("%dx%d" % (a.width, a.height)) if (a.width and a.height) \
        else "%s @ %.2fMP (x%d)" % (a.aspect, a.megapixels, a.multiple)
    log("mode %s | seed %d | length %d | %s | steps %d"
        % (a.mode, seed, ref2v_length(a.dur), canvas, a.steps))

    if a.mode == "ref2v":
        g = ref2v_graph(a.prompt, a.width, a.height, a.dur, seed, a.tag, a.steps, epoch,
                        a.image, a.video, a.audio, a.ref_image_size,
                        a.aspect, a.megapixels, a.multiple)
    else:
        g = t2v_graph(a.prompt, a.width, a.height, a.dur, seed, a.tag, a.steps, epoch,
                      a.first_frame, a.last_frame,
                      a.aspect, a.megapixels, a.multiple)
    log("[stage] queue")
    pid = submit(g, client)
    log("prompt_id %s" % pid)
    hist, el = wait_done(pid)
    mp4 = saved_mp4(hist)
    if mp4 is None:
        sys.exit("no mp4 saved")
    out = a.out or os.path.join(OUTPUT, OUT_SUBDIRS[a.mode], "%s.mp4" % a.tag)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if os.path.abspath(mp4) != os.path.abspath(out):
        shutil.move(mp4, out)
    log("[stage] done %.1fs" % el)
    log("RESULT %s" % out)
    log("SEED %d" % seed)


if __name__ == "__main__":
    main()
