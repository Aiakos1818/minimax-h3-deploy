import argparse, json, os, random, sys, time, urllib.request

HOME = os.path.expanduser("~/MiniMax-H3-Deploy")
API = "http://127.0.0.1:8188"
WF_DIR = HOME + "/workflows/api"
T2V_WF = WF_DIR + "/api_local_t2v.json"
I2V_WF = WF_DIR + "/api_local_i2v.json"

def http_json(url, data=None):
    if data is None:
        return json.load(urllib.request.urlopen(url, timeout=30))
    req = urllib.request.Request(url, data=json.dumps(data).encode(), headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=30))

def upload_image(path):
    fname = os.path.basename(path)
    boundary = "----genpy" + str(random.randint(10**8, 10**9))
    with open(path, "rb") as f:
        body = f.read()
    head = ("--%s\r\nContent-Disposition: form-data; name=\"image\"; filename=\"%s\"\r\n"
            "Content-Type: image/png\r\n\r\n" % (boundary, fname)).encode() + body + ("\r\n--%s--\r\n" % boundary).encode()
    req = urllib.request.Request(API + "/upload/image", data=head,
        headers={"Content-Type": "multipart/form-data; boundary=" + boundary})
    r = json.load(urllib.request.urlopen(req, timeout=60))
    return r.get("name", fname)

def find_by_class(g, cls):
    return [k for k, s in g.items() if s.get("class_type") == cls]

def build(args):
    base = I2V_WF if (args.mode == "i2v") else T2V_WF
    g = json.loads(json.dumps(json.load(open(base))))
    for k in find_by_class(g, "MiniMaxH3ImageToVideo"):
        inp = g[k]["inputs"]
        inp["prompt"] = args.prompt
        if args.image:
            name = upload_image(args.image) if os.path.isfile(args.image) else args.image
            img_key = "loadimg"
            g[img_key] = {"class_type": "LoadImage", "inputs": {"image": name}}
            inp["first_frame"] = [img_key, 0]
        elif "first_frame" in inp:
            inp.pop("first_frame", None)
        if args.width:  inp["width"] = args.width
        if args.height: inp["height"] = args.height
        if args.length: inp["length"] = args.length
    for k in find_by_class(g, "RandomNoise"):
        g[k]["inputs"]["noise_seed"] = args.seed
    for k, s in g.items():
        if s["class_type"] == "PrimitiveBoolean":
            g[k]["inputs"]["value"] = bool(args.turbo)
    return g

def main():
    p = argparse.ArgumentParser(description="MiniMax H3 generation via ComfyUI API")
    p.add_argument("--mode", choices=["t2v", "i2v"], default="t2v")
    p.add_argument("--prompt", required=True)
    p.add_argument("--image", default=None, help="first-frame image (file path on server, or a name already in ComfyUI/input)")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--height", type=int, default=None)
    p.add_argument("--length", type=int, default=None, help="frame count, default 124 (~5s)")
    p.add_argument("--seconds", type=float, default=None, help="duration in seconds (snapped to 17k+5)")
    p.add_argument("--turbo", action="store_true", help="turbo LoRA 8-step path")
    p.add_argument("--wait", action="store_true", help="wait for completion and print result path")
    a = p.parse_args()

    g = build(a)
    if a.seconds is not None:
        for k in find_by_class(g, "PrimitiveFloat"):
            g[k]["inputs"]["value"] = float(a.seconds)
    r = http_json(API + "/prompt", {"prompt": g, "client_id": "opencode-gen"})
    pid = r.get("prompt_id")
    if r.get("error"):
        sys.exit("submit error: %s" % r["error"])
    print("SUBMITTED %s mode=%s" % (pid, a.mode))
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
                    print("DONE %.1f min -> %s/%s" % ((time.time()-t0)/60, f.get("subfolder"), f.get("filename")))
            return
        time.sleep(15)
    print("TIMEOUT_WAIT")

if __name__ == "__main__":
    main()
