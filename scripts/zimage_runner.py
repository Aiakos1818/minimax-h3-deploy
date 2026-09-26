#!/usr/bin/env python3
"""zimage_runner.py -- Z-Image Turbo image driver for the web console.

Stdlib only. Text-to-image loads workflows/api/api_image_z_image_turbo.json and
image-to-image loads workflows/api/api_image_z_image_turbo_i2i.json (LoadImage ->
ImageScaleToTotalPixels -> VAEEncode -> KSampler with denoise = strength).
Overrides prompt / size / seed / steps, submits the graph to ComfyUI (:8188),
waits for the SaveImage result and copies the PNG to --out. Emits the same
"[stage]", "[progress]" and "[comfy]" markers minimax_h3_runner.py does so the
web console can show phases and a step bar.

Usage:
  zimage_runner.py --prompt "..." --aspect "16:9 (Widescreen)" --megapixels 0.4 \
                   --steps 8 --out output/<project>/t2i/<tag>.png [--seed N]
  zimage_runner.py --mode i2i --init-image <path> --strength 0.6 --prompt "..." \
                   --megapixels 0.4 --steps 8 --out output/<project>/i2i/<tag>.png
"""
import argparse, json, math, os, re, shutil, signal, subprocess, sys, time, urllib.request, uuid

HOME = os.path.expanduser("~/MiniMax-H3-Deploy")
API = "http://127.0.0.1:8188"
OUTPUT = HOME + "/output"
TEMPLATE = os.path.join(HOME, "workflows", "api", "api_image_z_image_turbo.json")
TEMPLATE_I2I = os.path.join(HOME, "workflows", "api", "api_image_z_image_turbo_i2i.json")
COMFY_LOG = os.path.expanduser("~/ComfyUI-Deploy/comfy.log")
INPUT_DIR = os.path.expanduser("~/ComfyUI-Deploy/input")
START_SH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "start-comfyui-for-minimax-h3.sh")

SAVE_NODE = "9"
PROMPT_NODE = "27"
LATENT_NODE = "13"
SAMPLER_NODE = "3"
I2I_LOAD_NODE = "40"
I2I_SCALE_NODE = "42"
I2I_ENCODE_NODE = "41"

RATIOS = {
    "16:9 (Widescreen)": 16 / 9,
    "9:16 (Portrait Widescreen)": 9 / 16,
    "1:1 (Square)": 1.0,
    "4:3 (Standard)": 4 / 3,
    "3:4 (Portrait Standard)": 3 / 4,
    "3:2 (Photo)": 3 / 2,
    "2:3 (Portrait Photo)": 2 / 3,
    "21:9 (Ultrawide)": 21 / 9,
}


def http_json(url, data=None, timeout=120):
    if data is None:
        return json.load(urllib.request.urlopen(url, timeout=timeout))
    req = urllib.request.Request(url, data=json.dumps(data).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def log(msg):
    print(msg, flush=True)


def _service_up():
    try:
        http_json(API + "/system_stats", timeout=5)
        return True
    except Exception:
        return False


def ensure_service():
    """Start ComfyUI if it is down so image modes share the same engine and
    queue as the video modes (which start it the same way)."""
    if _service_up():
        return
    log("[lifecycle] starting ComfyUI...")
    subprocess.run(["bash", START_SH], check=False)
    t0 = time.time()
    while time.time() - t0 < 300:
        if _service_up():
            log("[lifecycle] service up after %.1fs" % (time.time() - t0))
            return
        time.sleep(2)
    sys.exit("service did not come up")


def _round(x, multiple=16):
    return max(multiple, int(round(x / multiple)) * multiple)


def dims(aspect, megapixels, multiple=16):
    ratio = RATIOS.get(aspect, 16 / 9)
    area = max(0.05, float(megapixels)) * 1e6
    w = math.sqrt(area * ratio)
    h = w / ratio
    return _round(w, multiple), _round(h, multiple)


def build_graph(prompt, w, h, seed, steps, mode="t2i", init_name=None,
                megapixels=0.4, strength=0.6):
    path = TEMPLATE_I2I if mode == "i2i" else TEMPLATE
    with open(path, encoding="utf-8") as f:
        g = json.load(f)
    g[PROMPT_NODE]["inputs"]["text"] = prompt
    g[SAMPLER_NODE]["inputs"]["seed"] = seed
    g[SAMPLER_NODE]["inputs"]["steps"] = steps
    if mode == "i2i":
        g[I2I_LOAD_NODE]["inputs"]["image"] = init_name
        g[I2I_SCALE_NODE]["inputs"]["megapixels"] = megapixels
        g[SAMPLER_NODE]["inputs"]["latent_image"] = [I2I_ENCODE_NODE, 0]
        g[SAMPLER_NODE]["inputs"]["denoise"] = strength
    else:
        g[LATENT_NODE]["inputs"]["width"] = w
        g[LATENT_NODE]["inputs"]["height"] = h
    return g


def _stage_image(src, tag):
    if not os.path.isfile(src):
        sys.exit("init image not found: %s" % src)
    ext = os.path.splitext(src)[1].lower() or ".png"
    name = "%s_i2i_init%s" % (tag, ext)
    dst = os.path.join(INPUT_DIR, name)
    if os.path.abspath(src) != os.path.abspath(dst):
        shutil.copy2(src, dst)
    log("[material] init -> input/%s" % name)
    return name


def submit(g, client):
    r = http_json(API + "/prompt", {"prompt": g, "client_id": client})
    if r.get("error"):
        sys.exit("submit error: %s" % json.dumps(r["error"], ensure_ascii=False)[:2500])
    return r["prompt_id"]


_PROG_RE = re.compile(r"(\d+(?:\.\d+)?)/(\d+(?:\.\d+)?)\s*\[")
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def sample_progress(start=0):
    try:
        with open(COMFY_LOG, "rb") as f:
            size = f.seek(0, 2)
            f.seek(max(start, size - 16384))
            tail = f.read().decode("utf-8", "replace")
    except Exception:
        return None
    ms = _PROG_RE.findall(tail)
    if not ms:
        return None
    a, b = ms[-1]
    return int(float(a)), int(float(b))


def _comfy_detail(start=0):
    try:
        with open(COMFY_LOG, "rb") as f:
            f.seek(start)
            chunk = f.read()
            return f.tell(), chunk.decode("utf-8", "replace")
    except Exception:
        return start, ""


def wait_done(pid, timeout_s=9000):
    t0 = time.time()
    announced = False
    prog_off = 0
    last_prog = None
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
        comfy_off, chunk = _comfy_detail(comfy_off)
        for raw in chunk.splitlines():
            line = _ANSI_RE.sub("", raw).strip()
            if not line or ("|" in line and "%|" in line):
                continue
            if re.search(r"Requested to load|Prompt executed in|VRAM\[", line):
                log("[comfy] %s" % line[:300])
        pr = sample_progress(prog_off)
        if pr and pr != last_prog:
            if not announced:
                if pr[0] == pr[1] and pr[1] != 1:
                    pr = None
                else:
                    announced = True
                    prog_off = os.path.getsize(COMFY_LOG) if os.path.exists(COMFY_LOG) else 0
                    log("[stage] sampling")
            if pr is not None:
                last_prog = pr
                log("[progress] %d/%d" % pr)
        time.sleep(2)
    sys.exit("wait timeout")


def saved_png(hist):
    out = hist.get("outputs", {}).get(SAVE_NODE, {})
    for it in out.get("images", []):
        if not isinstance(it, dict):
            continue
        f = os.path.join(OUTPUT, it.get("subfolder", ""), it.get("filename", ""))
        if os.path.isfile(f):
            return f
    return None


_cancel = {"hit": False}


def _on_signal(signum, frame):
    _cancel["hit"] = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--mode", choices=["t2i", "i2i"], default="t2i")
    ap.add_argument("--init-image", default=None)
    ap.add_argument("--strength", type=float, default=0.6)
    ap.add_argument("--aspect", default="16:9 (Widescreen)")
    ap.add_argument("--megapixels", type=float, default=0.4)
    ap.add_argument("--multiple", type=int, default=16)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--tag", default="t2i")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)

    mode = "i2i" if (a.init_image or a.mode == "i2i") else "t2i"
    if mode == "i2i" and not a.init_image:
        sys.exit("i2i requires --init-image")
    strength = max(0.05, min(1.0, a.strength))

    w, h = dims(a.aspect, a.megapixels, a.multiple)
    seed = a.seed if a.seed is not None else int.from_bytes(os.urandom(8), "little") & ((1 << 63) - 1)
    init_name = _stage_image(a.init_image, a.tag) if mode == "i2i" else None
    ensure_service()
    log("[stage] queue")
    if mode == "i2i":
        log("z-image-turbo i2i %.2fMP (x%d) strength=%.2f seed=%d steps=%d"
            % (a.megapixels, a.multiple, strength, seed, a.steps))
    else:
        log("z-image-turbo %dx%d %.2fMP (x%d) seed=%d steps=%d" % (
            w, h, a.megapixels, a.multiple, seed, a.steps))

    g = build_graph(a.prompt, w, h, seed, a.steps, mode=mode, init_name=init_name,
                    megapixels=a.megapixels, strength=strength)
    pid = submit(g, uuid.uuid4().hex)
    log("[stage] queue")
    hist, took = wait_done(pid)

    src = saved_png(hist)
    if not src:
        sys.exit("no png saved")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    shutil.copy2(src, a.out)
    log("[stage] done")
    log("saved %s (%.1fs, %d bytes)" % (a.out, took, os.path.getsize(a.out)))


if __name__ == "__main__":
    main()
