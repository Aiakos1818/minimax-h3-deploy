import argparse
import json
import os
import random
import sys
import time
import urllib.request

HOME = os.path.expanduser("~/MiniMax-H3-Deploy")
API = "http://127.0.0.1:8188"
WF = HOME + "/workflows/api/api_raylight_h3_i2v.json"


def http_json(url, data=None):
    if data is None:
        return json.load(urllib.request.urlopen(url, timeout=40))
    req = urllib.request.Request(url, data=json.dumps(data).encode(), headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=40))


def upload_image(path):
    fname = os.path.basename(path)
    boundary = "----g2" + str(random.randint(10**8, 10**9))
    with open(path, "rb") as f:
        body = f.read()
    head = ("--%s\r\nContent-Disposition: form-data; name=\"image\"; filename=\"%s\"\r\n"
            "Content-Type: image/png\r\n\r\n" % (boundary, fname)).encode() + body + ("\r\n--%s--\r\n" % boundary).encode()
    req = urllib.request.Request(API + "/upload/image", data=head,
        headers={"Content-Type": "multipart/form-data; boundary=" + boundary})
    r = json.load(urllib.request.urlopen(req, timeout=60))
    return r.get("name", fname)


def find_nodes(g, cls):
    return [k for k, s in g.items() if s.get("class_type") == cls]


def build(args):
    g = json.loads(json.dumps(json.load(open(WF))))
    for k in find_nodes(g, "MiniMaxH3ImageToVideo"):
        inp = g[k]["inputs"]
        inp["prompt"] = args.prompt
        if not args.image:
            inp.pop("first_frame", None)
        if args.width:
            inp["width"] = args.width
        if args.height:
            inp["height"] = args.height
        if args.length:
            inp["length"] = args.length
        if args.seconds is not None:
            inp.pop("length", None)
    for k in find_nodes(g, "LoadImage"):
        if args.image:
            name = upload_image(args.image) if os.path.isfile(args.image) else args.image
            g[k]["inputs"]["image"] = name
    for k in find_nodes(g, "XFuserSamplerCustomAdvanced"):
        g[k]["inputs"]["noise_seed"] = args.seed
    for k in find_nodes(g, "RayBasicScheduler"):
        g[k]["inputs"]["steps"] = args.steps
    for k, s in g.items():
        if s.get("class_type") == "PrimitiveFloat" and s.get("_prim") is None:
            pass
    if args.seconds is not None:
        for k in find_nodes(g, "ComfyMathExpression"):
            g[k]["inputs"]["values.a"] = args.seconds
    return g


def main():
    p = argparse.ArgumentParser(description="MiniMax H3 dual-GPU (Raylight TP) generation via ComfyUI API")
    p.add_argument("--prompt", required=True)
    p.add_argument("--image", default=None, help="first-frame image (server path to upload, or filename already in ComfyUI/input)")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--height", type=int, default=None)
    p.add_argument("--length", type=int, default=None, help="exact frame count (overrides seconds)")
    p.add_argument("--seconds", type=float, default=None, help="duration in seconds (ComfyMath 17k+5 snap), default template 2.0")
    p.add_argument("--steps", type=int, default=20, help="sampling steps (default 20, min 8)")
    p.add_argument("--wait", action="store_true")
    a = p.parse_args()
    if a.steps < 8:
        p.error("--steps must be >= 8")

    g = build(a)
    r = http_json(API + "/prompt", {"prompt": g, "client_id": "opencode-gen-dual"})
    pid = r.get("prompt_id")
    if r.get("error"):
        sys.exit("submit error: %s" % r["error"])
    print("SUBMITTED %s" % pid)
    if not a.wait:
        return
    t0 = time.time()
    while time.time() - t0 < 3600:
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
                        print("ERROR node=%s %s %s" % (m.get("node_id"), m.get("exception_type"), str(m.get("exception_message"))[:1000]))
                sys.exit(1)
            for o in h[pid].get("outputs", {}).values():
                for f in (o.get("videos") or o.get("images") or []):
                    print("DONE %.1f min -> %s/%s" % ((time.time() - t0) / 60, f.get("subfolder"), f.get("filename")))
            return
        time.sleep(15)
    print("TIMEOUT_WAIT")


if __name__ == "__main__":
    main()
