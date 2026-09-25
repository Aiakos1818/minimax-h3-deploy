#!/usr/bin/env python3
"""ref2v_runner.py -- one reference-to-video clip per invocation.

Single-segment sibling of chain_director_v3.py: same resident int4-CLIP /
int8-UNet raylight base, but node 133 is MiniMaxH3ReferenceToVideo and the
clip is produced from a prompt plus reference images / videos / audios.  No
slots, no continuation, no merge -- one task in, one mp4 out.

ComfyUI is resident: a run starts the service only when it is down, and always
leaves it up so the next run reuses the loaded FSDP shards and raylight workers.
The web console owns the explicit "release VRAM" action.

Usage (called by the web console; can be run by hand too):
  ~/ComfyUI-Deploy/comfyenv/bin/python scripts/ref2v_runner.py \
    --prompt "..." --image ref.png --dur 5 --steps 20 \
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
AUDIO_VAE_DEVICE = "gpu:1"
FPS = 24
MAX_IMAGES, MAX_VIDEOS, MAX_AUDIOS = 9, 3, 3
OUT_SUBDIR = "ref2v"


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
def ray_base(save_prefix, steps, epoch):
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
            "unet_name": UNET_REF2VA, "weight_dtype": "default",
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
    g = ray_base("video/%s/%s" % (OUT_SUBDIR, tag), steps, epoch)
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


def _state_write(epoch):
    with open(STATE_PATH, "w") as fh:
        json.dump({"pid": service_pid(), "epoch": int(epoch), "unet": UNET_REF2VA}, fh)


def ensure_service():
    """Return the reuse_epoch. Resumes the running service and its loaded FSDP
    UNet when the state file vouches for it; otherwise starts the service and
    mints a new epoch (ray rebuild + UNet reload on the first prompt)."""
    pid = service_pid()
    st = _state_read()
    if pid and st.get("pid") == pid and st.get("unet") == UNET_REF2VA and st.get("epoch") and _service_up():
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
    _state_write(epoch)
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


def running_prompts():
    try:
        q = http_json(API + "/queue", timeout=5)
    except Exception:
        return set()
    return {it[1] for it in (q.get("queue_running") or [])}


def sample_progress():
    """Latest 'done/total' step from ComfyUI's tqdm line in comfy.log (or None).

    Serial execution means at most one prompt is sampling, so the tail of the
    shared server log is ours to read."""
    try:
        with open(COMFY_LOG, "rb") as f:
            f.seek(0, 2)
            start = max(0, f.tell() - 16384)
            f.seek(start)
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
    last_prog = None
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
        if not announced and pid in running_prompts():
            announced = True
            log("[stage] sampling")
        if announced:
            pr = sample_progress()
            if pr and pr != last_prog:
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
    ap.add_argument("--ref-image-size", choices=("match", "max"), default="match")
    ap.add_argument("--tag", default="ref2v")
    ap.add_argument("--out", default=None, help="final mp4 path (default output/ref2v/<tag>.mp4)")
    a = ap.parse_args()

    if len(a.image) > MAX_IMAGES:
        sys.exit("too many reference images (max %d)" % MAX_IMAGES)
    if len(a.video) > MAX_VIDEOS:
        sys.exit("too many reference videos (max %d)" % MAX_VIDEOS)
    if len(a.audio) > MAX_AUDIOS:
        sys.exit("too many reference audios (max %d)" % MAX_AUDIOS)
    if not a.image and not a.video and not a.audio:
        sys.exit("at least one reference image, video or audio is required")

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)

    seed = a.seed if a.seed is not None else random.randint(0, 2**63 - 1)
    log("[stage] material")
    epoch = ensure_service()
    client = "ref2v-" + a.tag + "-" + str(random.randint(1000, 9999))
    canvas = ("%dx%d" % (a.width, a.height)) if (a.width and a.height) \
        else "%s @ %.2fMP (x%d)" % (a.aspect, a.megapixels, a.multiple)
    log("seed %d | length %d | %s | steps %d"
        % (seed, ref2v_length(a.dur), canvas, a.steps))

    g = ref2v_graph(a.prompt, a.width, a.height, a.dur, seed, a.tag, a.steps, epoch,
                    a.image, a.video, a.audio, a.ref_image_size,
                    a.aspect, a.megapixels, a.multiple)
    log("[stage] queue")
    pid = submit(g, client)
    log("prompt_id %s" % pid)
    hist, el = wait_done(pid)
    mp4 = saved_mp4(hist)
    if mp4 is None:
        sys.exit("no mp4 saved")
    out = a.out or os.path.join(OUTPUT, OUT_SUBDIR, "%s.mp4" % a.tag)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if os.path.abspath(mp4) != os.path.abspath(out):
        shutil.move(mp4, out)
    log("[stage] done %.1fs" % el)
    log("RESULT %s" % out)
    log("SEED %d" % seed)


if __name__ == "__main__":
    main()
