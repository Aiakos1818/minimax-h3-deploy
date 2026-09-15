#!/usr/bin/env python3
"""anchors.py -- Z-Image-Turbo first-frame anchors for a film spec.

  anchors.py generate [--only s1 m2] [--dry-run]   queue every candidate still (shot x seed)
  anchors.py contact                                write output/<film>/anchors.html
  anchors.py pick s1=b s2=a [--film ...]            chosen candidate -> projects/<film>/anchors/<shot>_best.png

Candidates land in ComfyUI's output dir (= <root>/output/<film>/), so the web console can
preview them:  http://127.0.0.1:8190/files/<film>/anchors.html
Run the whole batch *before* starting any H3 chain: the resident FSDP UNet holds ~12G/card
and Z-Image cannot fit next to it.
"""
import argparse
import glob
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.request

DEFAULT_SPEC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "film_spec.example.json")
WAIT_SEC = 1200


def load_spec(path):
    with open(path, encoding="utf-8") as fh:
        spec = json.load(fh)
    spec["_path"] = os.path.abspath(path)
    spec["_root"] = os.path.expanduser(spec["root"])
    spec["_out"] = os.path.join(spec["_root"], "output", spec["film"])
    if not os.path.isdir(spec["_root"]):
        sys.exit("root not found: %s" % spec["_root"])
    return spec


def graph(prompt, seed, prefix, size, steps):
    return {
        "30": {"class_type": "CLIPLoader",
               "inputs": {"clip_name": "qwen_3_4b.safetensors", "type": "lumina2", "device": "default"}},
        "27": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["30", 0]}},
        "33": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["27", 0]}},
        "28": {"class_type": "UNETLoader",
               "inputs": {"unet_name": "z_image_turbo_bf16.safetensors", "weight_dtype": "default"}},
        "11": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["28", 0], "shift": 3.0}},
        "13": {"class_type": "EmptySD3LatentImage",
               "inputs": {"width": size[0], "height": size[1], "batch_size": 1}},
        "3": {"class_type": "KSampler",
              "inputs": {"model": ["11", 0], "seed": seed, "steps": steps, "cfg": 1.0,
                         "sampler_name": "res_multistep", "scheduler": "simple",
                         "positive": ["27", 0], "negative": ["33", 0],
                         "latent_image": ["13", 0], "denoise": 1.0}},
        "29": {"class_type": "VAELoader", "inputs": {"vae_name": "ae.safetensors"}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["29", 0]}},
        "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": prefix}},
    }


def post(comfy, g, client):
    req = urllib.request.Request(comfy + "/prompt",
                                 data=json.dumps({"prompt": g, "client_id": client}).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        return json.load(urllib.request.urlopen(req, timeout=60))["prompt_id"]
    except urllib.error.HTTPError as e:
        sys.exit("ComfyUI rejected the graph: %s\n%s" % (e, e.read().decode()[:500]))
    except urllib.error.URLError as e:
        sys.exit("ComfyUI unreachable at %s (%s)\nstart it with: %s/scripts/start-comfyui-for-minimax-h3.sh"
                 % (comfy, e, os.path.expanduser("~/MiniMax-H3-Deploy")))


def wait(comfy, pid):
    t0 = time.time()
    while time.time() - t0 < WAIT_SEC:
        try:
            h = json.load(urllib.request.urlopen("%s/history/%s" % (comfy, pid), timeout=15)).get(pid)
        except Exception:
            h = None
        if h:
            if h.get("status", {}).get("status_str") == "error":
                return None
            files = [im["filename"] for node in h.get("outputs", {}).values() for im in node.get("images", [])]
            if files:
                return files
        time.sleep(3)
    return None


def cmd_generate(spec, only, dry):
    comfy = spec["comfy"]
    shots = [s for s in spec["anchors"] if not only or s in only]
    todo = [(s, letter, seed) for s in shots for letter, seed in spec["anchors"][s]["seeds"].items()]
    print("%s: %d still(s) for %s" % (spec["film"], len(todo), ", ".join(shots)), flush=True)
    for shot, letter, seed in todo:
        cfg = spec["anchors"][shot]
        prefix = "%s/%s_%s" % (spec["film"], shot, letter)
        if dry:
            print("  [dry] %s seed=%s size=%sx%s steps=%s -> output/%s/*.png"
                  % (prefix, seed, cfg["size"][0], cfg["size"][1], cfg["steps"], prefix), flush=True)
            continue
        pid = post(comfy, graph(cfg["prompt"], seed, prefix, cfg["size"], cfg["steps"]),
                   "anchor-%s-%s" % (shot, letter))
        files = wait(comfy, pid)
        print("  %-12s seed=%-8s -> %s" % (prefix, seed, files or "FAILED"), flush=True)
    if not dry:
        print("next: anchors.py contact  (then pick s1=b s2=a ...)", flush=True)


def cmd_contact(spec):
    out = spec["_out"]
    rows = []
    for shot, cfg in spec["anchors"].items():
        cells = []
        for p in sorted(glob.glob(os.path.join(out, "%s_*.png" % shot))):
            rel = "%s/%s" % (spec["film"], os.path.basename(p))
            cells.append('<figure><img src="/files/%s"><figcaption>%s</figcaption></figure>'
                         % (rel, os.path.basename(p)))
        rows.append("<h2>%s</h2><div class=row>%s</div>" % (shot, "".join(cells) or "<i>missing</i>"))
    from string import Template
    html = Template("""<!doctype html><meta charset=utf-8><title>$film anchors</title>
<style>body{background:#15171b;color:#dfe3ea;font-family:system-ui,sans-serif;margin:16px}
h1{font-size:18px}h2{font-size:14px;margin:18px 0 6px;color:#ffcb6b}
.row{display:flex;gap:10px;flex-wrap:wrap}figure{margin:0}
img{width:420px;border:1px solid #333;border-radius:4px;display:block}
figcaption{font-size:11px;color:#8b93a1;padding-top:4px}</style>
<h1>$film anchors -- reply with picks like "s1=b s2=a"</h1>$body
<p style="color:#8b93a1;font-size:12px">pick copies the chosen file to
projects/$film/anchors/&lt;shot&gt;_best.png for the run.</p>
""").substitute(film=spec["film"], body="".join(rows))
    path = os.path.join(out, "anchors.html")
    os.makedirs(out, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(html)
    print("wrote", path, flush=True)
    print("open  %s/files/%s/anchors.html" % (spec["console"], spec["film"]), flush=True)


def cmd_pick(spec, pairs):
    dst_dir = os.path.join(spec["_root"], "projects", spec["film"], "anchors")
    os.makedirs(dst_dir, exist_ok=True)
    for pair in pairs:
        if "=" not in pair:
            sys.exit("bad pick %r (expected shot=letter, e.g. s1=b)" % pair)
        shot, letter = pair.split("=", 1)
        cands = sorted(glob.glob(os.path.join(spec["_out"], "%s_%s_*.png" % (shot, letter))))
        if not cands:
            sys.exit("no candidate for %s (%s_%s_*.png in %s)" % (pair, shot, letter, spec["_out"]))
        dst = os.path.join(dst_dir, "%s_best.png" % shot)
        shutil.copy2(cands[-1], dst)
        print("  %s -> %s" % (pair, dst), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("verb", choices=["generate", "contact", "pick"])
    ap.add_argument("pairs", nargs="*", help="pick: shot=letter ...")
    ap.add_argument("--spec", default=DEFAULT_SPEC)
    ap.add_argument("--only", nargs="*", default=None, help="generate: only these shot keys")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    spec = load_spec(a.spec)
    if a.verb == "generate":
        cmd_generate(spec, a.only, a.dry_run)
    elif a.verb == "contact":
        cmd_contact(spec)
    else:
        cmd_pick(spec, a.pairs)


if __name__ == "__main__":
    main()
