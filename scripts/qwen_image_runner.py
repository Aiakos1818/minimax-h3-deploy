#!/usr/bin/env python3
"""qwen_image_runner.py -- Qwen-Image-2.1 driver for the web console.

Runs Qwen-Image-2.1 (bf16, diffusers QwenImage21Pipeline) on the local 2x
RTX 2080 Ti box with device_map="balanced": the text encoder lands on one
card, the transformer + VAE on the other, so both GPUs stay nearly full.
Text-to-image derives the canvas from --aspect/--megapixels. Image-to-image
feeds the reference image as vision context and lets the pipeline keep its
aspect ratio, sized by --megapixels. Qwen samples without classifier-free
guidance (true_cfg_scale=1.0).

The web console launches this with the ComfyUI venv interpreter; this file
prepends the overlay site dir that carries the newer transformers/diffusers.
It emits "[stage]", "[progress]" and "[qwen]" markers so the console can show
phases, a step bar and a little detail.

Usage:
  qwen_image_runner.py --prompt "..." --aspect "16:9 (Widescreen)" \
      --megapixels 0.4 --steps 8 --out output/<project>/t2i/<tag>.png [--seed N]
  qwen_image_runner.py --mode i2i --init-image <path> --prompt "..." \
      --megapixels 0.4 --steps 8 --out output/<project>/i2i/<tag>.png
"""
import argparse, math, os, signal, sys, time

SITE = os.path.expanduser("~/.venvs/qwen21/site")
MODEL = os.path.expanduser("~/Downloads/qwen-image-2.1")

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

_cancel = {"hit": False}


def log(msg):
    print(msg, flush=True)


def _on_signal(signum, frame):
    _cancel["hit"] = True


def _round(x, multiple=32):
    return max(multiple, int(round(x / multiple)) * multiple)


def dims(aspect, megapixels, multiple=32):
    ratio = RATIOS.get(aspect, 16 / 9)
    area = max(0.05, float(megapixels)) * 1e6
    w = math.sqrt(area * ratio)
    h = w / ratio
    return _round(w, multiple), _round(h, multiple)


def main():
    ap = argparse.ArgumentParser()
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

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)

    if os.path.isdir(SITE) and SITE not in sys.path:
        sys.path.insert(0, SITE)
    if not os.path.isdir(MODEL):
        sys.exit("model dir not found: %s" % MODEL)

    mode = "i2i" if (a.init_image or a.mode == "i2i") else "t2i"
    if mode == "i2i" and (not a.init_image or not os.path.isfile(a.init_image)):
        sys.exit("i2i requires a valid --init-image")
    steps = max(1, min(80, a.steps))
    seed = a.seed if a.seed is not None else int.from_bytes(os.urandom(8), "little") & ((1 << 63) - 1)

    import torch
    from PIL import Image
    from diffusers import QwenImage21Pipeline

    log("[stage] load_model")
    log("[qwen] loading Qwen-Image-2.1 (bf16, balanced) mode=%s steps=%d seed=%d"
        % (mode, steps, seed))
    t0 = time.time()
    pipe = QwenImage21Pipeline.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map="balanced")
    torch.cuda.synchronize()
    for name in ("text_encoder", "transformer", "vae"):
        m = getattr(pipe, name, None)
        dev = getattr(m, "device", None)
        if dev is None and hasattr(m, "hf_device_map"):
            dev = sorted(set(str(v) for v in m.hf_device_map.values()))
        log("[qwen] %s -> %s" % (name, dev))
    log("[qwen] load=%.1fs" % (time.time() - t0))
    if _cancel["hit"]:
        sys.exit("cancelled")

    gen = torch.Generator("cpu").manual_seed(seed)
    kwargs = {"prompt": a.prompt, "num_inference_steps": steps, "true_cfg_scale": 1.0,
              "generator": gen, "output_type": "pil"}
    if mode == "i2i":
        res = int(round(math.sqrt(max(0.05, a.megapixels) * 1e6)))
        kwargs["image"] = Image.open(a.init_image).convert("RGB")
        kwargs["output_resolution"] = res
        log("[qwen] i2i edit, output_resolution=%d (input %s)" % (res, kwargs["image"].size))
    else:
        w, h = dims(a.aspect, a.megapixels)
        kwargs["width"], kwargs["height"] = w, h
        log("[qwen] t2i %dx%d %.2fMP seed=%d" % (w, h, a.megapixels, seed))

    def cb(_pipe, step, _ts, cbkw):
        if _cancel["hit"]:
            _pipe._interrupt = True
        log("[progress] %d/%d" % (step + 1, steps))
        return cbkw

    kwargs["callback_on_step_end"] = cb
    torch.cuda.synchronize()
    log("[stage] sampling")
    t1 = time.time()
    out = pipe(**kwargs)
    torch.cuda.synchronize()
    took = time.time() - t1
    if _cancel["hit"]:
        sys.exit("cancelled")

    img = out.images[0]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    img.save(a.out)
    log("[stage] done")
    log("[qwen] gen=%.1fs size=%s" % (took, img.size))
    log("saved %s (%.1fs, %d bytes)" % (a.out, took, os.path.getsize(a.out)))


if __name__ == "__main__":
    main()
