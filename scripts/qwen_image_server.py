#!/usr/bin/env python3
"""qwen_image_server.py -- resident Qwen-Image-2.1 service for the web console.

Loads Qwen-Image-2.1 (bf16, diffusers QwenImage21Pipeline) once with
device_map="balanced" (text encoder on one card, transformer + VAE on the
other) and keeps it in VRAM, so consecutive Qwen jobs reuse the same weights
instead of paying the ~11s reload. The web console stops this service before
any other model's job (Z-Image Turbo / video) so ComfyUI can own the GPUs.

Endpoints (all POST unless noted):
  GET  /health   -> {"ok":1,"loaded":1,...}
  POST /generate -> streams "[stage]"/"[progress]"/"[qwen]" lines, then "saved ..."
  POST /unload   -> release the weights and exit the process

Run: ~/ComfyUI-Deploy/comfyenv/bin/python scripts/qwen_image_server.py --port 8193
The web console starts it on demand and writes the pidfile under ~/.cache/h3qwen.
"""
import argparse, gc, json, math, os, random, signal, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

SITE = os.path.expanduser("~/.venvs/qwen21/site")
MODEL = os.path.expanduser("~/Downloads/qwen-image-2.1")
CACHE = os.path.expanduser("~/.cache/h3qwen")

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


def pidfile(port):
    return os.path.join(CACHE, "server-%d.pid" % port)


def logfile(port):
    return os.path.join(CACHE, "server-%d.log" % port)


def _round(x, multiple=32):
    return max(multiple, int(round(x / multiple)) * multiple)


def dims(aspect, megapixels, multiple=32):
    ratio = RATIOS.get(aspect, 16 / 9)
    area = max(0.05, float(megapixels)) * 1e6
    w = math.sqrt(area * ratio)
    h = w / ratio
    return _round(w, multiple), _round(h, multiple)


def _log(msg):
    print("[qwen-server] %s" % msg, flush=True)


class State:
    def __init__(self, port):
        self.port = port
        self.pipe = None
        self.httpd = None
        self.lock = threading.Lock()
        self.cancel = False


def load_pipe():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    if os.path.isdir(SITE) and SITE not in sys.path:
        sys.path.insert(0, SITE)
    if not os.path.isdir(MODEL):
        sys.exit("model dir not found: %s" % MODEL)
    import torch
    from diffusers import QwenImage21Pipeline
    _log("loading %s (bf16, balanced)" % MODEL)
    t0 = time.time()
    pipe = QwenImage21Pipeline.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map="balanced")
    torch.cuda.synchronize()
    for name in ("text_encoder", "transformer", "vae"):
        m = getattr(pipe, name, None)
        dev = getattr(m, "device", None)
        if dev is None and hasattr(m, "hf_device_map"):
            dev = sorted(set(str(v) for v in m.hf_device_map.values()))
        _log("%s -> %s" % (name, dev))
    _log("loaded in %.1fs" % (time.time() - t0))
    return pipe


def run_generation(state, req, emit):
    pipe = state.pipe
    if pipe is None:
        emit("[error] model not loaded")
        return
    prompt = (req.get("prompt") or "").strip()
    if not prompt:
        emit("[error] empty prompt")
        return
    mode = req.get("mode") if req.get("mode") in ("t2i", "i2i") else "t2i"
    steps = max(1, min(80, int(req.get("steps") or 20)))
    seed = int(req["seed"]) if req.get("seed") is not None else \
        random.randint(0, 2 ** 63 - 1)
    import torch
    gen = torch.Generator("cpu").manual_seed(seed)
    kwargs = {"prompt": prompt, "num_inference_steps": steps, "true_cfg_scale": 1.0,
              "generator": gen, "output_type": "pil"}
    if mode == "i2i":
        from PIL import Image
        init = req.get("init_image")
        if not init or not os.path.isfile(init):
            emit("[error] init image not found")
            return
        res = int(round(math.sqrt(max(0.05, float(req.get("megapixels") or 0.4)) * 1e6)))
        kwargs["image"] = Image.open(init).convert("RGB")
        kwargs["output_resolution"] = res
        emit("[qwen] i2i edit, output_resolution=%d (input %s)"
             % (res, kwargs["image"].size))
    else:
        w, h = dims(req.get("aspect"), req.get("megapixels") or 0.4)
        kwargs["width"], kwargs["height"] = w, h
        emit("[qwen] t2i %dx%d %.2fMP seed=%d"
             % (w, h, float(req.get("megapixels") or 0.4), seed))

    def cb(_pipe, step, _ts, cbkw):
        if state.cancel:
            _pipe._interrupt = True
            return cbkw
        try:
            emit("[progress] %d/%d" % (step + 1, steps))
        except Exception:
            state.cancel = True
            _pipe._interrupt = True
        return cbkw

    kwargs["callback_on_step_end"] = cb
    emit("[stage] sampling")
    t1 = time.time()
    out = pipe(**kwargs)
    torch.cuda.synchronize()
    if state.cancel:
        emit("[cancelled]")
        return
    img = out.images[0]
    out_path = req.get("out")
    if not out_path:
        emit("[error] missing out path")
        return
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    img.save(out_path)
    took = time.time() - t1
    emit("[stage] done")
    emit("[qwen] gen=%.1fs size=%s" % (took, img.size))
    emit("saved %s (%.1fs, %d bytes)" % (out_path, took, os.path.getsize(out_path)))


def make_handler(state):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"

        def log_message(self, *args):
            pass

        def _json(self, code, obj):
            body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if urlparse(self.path).path == "/health":
                self._json(200, {"ok": 1, "loaded": bool(state.pipe is not None),
                                 "pid": os.getpid(), "port": state.port})
            else:
                self._json(404, {"ok": 0})

        def do_POST(self):
            path = urlparse(self.path).path
            n = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(n) if n else b""
            if path == "/unload":
                self._json(200, {"ok": 1, "msg": "unloading"})
                with state.lock:
                    state.pipe = None
                gc.collect()
                _log("unload requested, shutting down")
                threading.Thread(target=state.httpd.shutdown, daemon=True).start()
                return
            if path == "/generate":
                self._generate(raw)
                return
            self._json(404, {"ok": 0})

        def _generate(self, raw):
            try:
                req = json.loads(raw.decode("utf-8")) if raw else {}
            except Exception:
                self._json(400, {"ok": 0, "err": "bad json"})
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()

            def emit(line):
                self.wfile.write((line + "\n").encode("utf-8"))
                self.wfile.flush()

            try:
                with state.lock:
                    state.cancel = False
                    run_generation(state, req, emit)
            except (BrokenPipeError, ConnectionResetError):
                state.cancel = True
                _log("client disconnected, cancelling")
            except Exception as e:
                state.cancel = True
                try:
                    emit("[error] %s" % e)
                except Exception:
                    pass

    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8193)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()

    os.makedirs(CACHE, exist_ok=True)
    with open(pidfile(a.port), "w") as f:
        f.write(str(os.getpid()))
    httpd = None
    state = State(a.port)
    try:
        state.pipe = load_pipe()

        def on_signal(signum, frame):
            if state.httpd:
                threading.Thread(target=state.httpd.shutdown, daemon=True).start()

        signal.signal(signal.SIGTERM, on_signal)
        signal.signal(signal.SIGINT, on_signal)

        httpd = ThreadingHTTPServer((a.host, a.port), make_handler(state))
        httpd.daemon_threads = True
        state.httpd = httpd
        _log("ready on http://%s:%d/ pid=%d" % (a.host, a.port, os.getpid()))
        httpd.serve_forever(poll_interval=0.5)
    finally:
        if httpd is not None:
            httpd.server_close()
        try:
            os.remove(pidfile(a.port))
        except OSError:
            pass
        _log("bye")


if __name__ == "__main__":
    main()
