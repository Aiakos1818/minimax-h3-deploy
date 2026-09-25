#!/usr/bin/env python3
"""qwen_image_runner.py -- Qwen-Image-2.1 client for the web console.

The actual model lives in the resident qwen_image_server.py (see its docstring),
so this process only forwards one request and relays the progress lines the
console parses ("[stage]", "[progress]", "[qwen]"). The web console starts and
stops the server, so this client assumes it is already healthy.

Usage:
  qwen_image_runner.py --port 8193 --prompt "..." --aspect "16:9 (Widescreen)" \
      --megapixels 0.4 --steps 8 --out output/<project>/t2i/<tag>.png [--seed N]
  qwen_image_runner.py --port 8193 --mode i2i --init-image <path> --prompt "..." \
      --megapixels 0.4 --steps 8 --out output/<project>/i2i/<tag>.png
"""
import argparse, json, sys, urllib.request


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8193)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--mode", choices=["t2i", "i2i"], default="t2i")
    ap.add_argument("--init-image", default=None)
    ap.add_argument("--strength", type=float, default=0.6)
    ap.add_argument("--aspect", default="16:9 (Widescreen)")
    ap.add_argument("--megapixels", type=float, default=0.4)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--tag", default="t2i")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    mode = "i2i" if (a.init_image or a.mode == "i2i") else "t2i"
    if mode == "i2i" and not a.init_image:
        sys.exit("i2i requires --init-image")
    body = {"prompt": a.prompt, "mode": mode, "aspect": a.aspect,
            "megapixels": a.megapixels, "steps": a.steps, "seed": a.seed,
            "out": a.out}
    if mode == "i2i":
        body["init_image"] = a.init_image
        body["strength"] = a.strength

    req = urllib.request.Request(
        "http://127.0.0.1:%d/generate" % a.port, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    try:
        resp = urllib.request.urlopen(req, timeout=7200)
    except Exception as e:
        sys.exit("qwen server not reachable on :%d (%s)" % (a.port, e))

    failed = False
    try:
        for raw in resp:
            line = raw.decode("utf-8", "replace").rstrip("\n")
            if line:
                print(line, flush=True)
            if line.startswith("[error]") or line.startswith("[cancelled]"):
                failed = line.startswith("[error]")
    except Exception as e:
        sys.exit("qwen generation stream failed: %s" % e)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
