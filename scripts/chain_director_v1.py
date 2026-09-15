#!/usr/bin/env python3
"""chain_director_v1.py -- MiniMax-H3 ChainDirector-RayLight (Herrgotts masked-AV engine).

Server-side automatic multi-clip chain on raylight dual-GPU TP:

  clip1 : H3ContinuousStartV14 -> sampler -> Analyze -> SaveLatent(slot1) + raw mp4
  clipN : LoadLatent(slot N-1) + H3ContinuousContinueV14 -> sampler -> Analyze
          -> SaveLatent(slot N) + raw mp4

--prompt = global script; --beats "t:note;t:note" injects each beat (relative to that
segment's own clock) into its owning segment (t / --dur picks the segment).

--merge: stitch delivered ranges (clip1[0:safe], clipN[head:safe], last [head:end])
into final.mp4.
"""
import argparse, glob, json, os, random, re, sys, time, urllib.request

import numpy as np

from safetensors import safe_open

import av

HOME = os.path.expanduser("~/MiniMax-H3-Deploy")
API = "http://127.0.0.1:8188"
OUTPUT = HOME + "/output"
CLIP = "qwen3vl_32b_minimax_h3_int8_convrot.safetensors"
VAE_VIDEO = "minimax_h3_video_vae_fp16.safetensors"
VAE_AUDIO = "minimax_h3_audio_vae_fp32.safetensors"
UNET_FL2VA = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
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


def ray_base(unet, w, h, save_prefix):
    return {
        "141": {"class_type": "RayInitializer", "inputs": {
            "ray_cluster_address": "local", "ray_cluster_namespace": "default",
            "GPU": 2, "ulysses_degree": 2, "ring_degree": 1, "cfg_degree": 1,
            "dp_degree": 1, "sync_ulysses": False, "clear_vram_after_sampling": True,
            "FSDP": True, "FSDP_CPU_OFFLOAD": False, "XFuser_attention": "SAGE_FP16",
            "skip_comm_test": False, "use_mmap": True, "RAYLIGHT_ULYSSES_KV_INT8": "off"}},
        "130": {"class_type": "H3MultiGPUCLIPLoader", "inputs": {
            "clip_name": CLIP, "gpu_ids": "0,1", "offload_after_encode": True}},
        "121": {"class_type": "VAELoader", "inputs": {"vae_name": VAE_VIDEO}},
        "122": {"class_type": "VAELoader", "inputs": {"vae_name": VAE_AUDIO}},
        "142": {"class_type": "RayUNETLoader", "inputs": {
            "unet_name": unet, "weight_dtype": "default", "ray_actors_init": ["141", 0]}},
        "125": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "res_multistep"}},
        "145": {"class_type": "RayBasicGuider", "inputs": {"ray_actors": ["142", 0], "conditioning": ["133", 0]}},
        "143": {"class_type": "RayBasicScheduler", "inputs": {
            "ray_actors": ["142", 0], "scheduler": "simple", "steps": 8, "denoise": 1}},
        "144": {"class_type": "XFuserSamplerCustomAdvanced", "inputs": {
            "add_noise": True, "noise_seed": 0,
            "guider": ["145", 0], "sampler": ["125", 0], "sigmas": ["143", 0],
            "latent_image": ["133", 1]}},
        "124": {"class_type": "VAEDecode", "inputs": {"samples": ["144", 0], "vae": ["121", 0]}},
        "123": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["144", 0], "vae": ["122", 0]}},
        "132": {"class_type": "CreateVideo", "inputs": {
            "images": ["124", 0], "fps": 24, "audio": ["123", 0], "bit_depth": 8}},
        "92": {"class_type": "SaveVideo", "inputs": {
            "video": ["132", 0], "filename_prefix": save_prefix, "format": "auto", "codec": "auto"}},
    }


def add_analyze_and_save(g, clip_idx):
    d = widget_defaults(ANALYZE_CLS)
    d.pop("images", None)
    g["200"] = {"class_type": ANALYZE_CLS, "inputs": {"images": ["124", 0], **d}}
    g["180"] = {"class_type": SAVE_CLS, "inputs": {
        "latent": ["144", 0], "filename_prefix": PREFIX, "clip_index": clip_idx,
        "handover": ["200", 0]}}


def seg0_graph(prompt, w, h, dur, seed, tag):
    g = ray_base(UNET_FL2VA, w, h, "video/chain/%s/seg_0" % tag)
    g["144"]["inputs"]["noise_seed"] = seed
    g["133"] = {"class_type": START_CLS, "inputs": {
        "clip": ["130", 0], "vae": ["121", 0], "prompt": prompt,
        "width": w, "height": h, "duration": dur, "ref_image_size": "match"}}
    add_analyze_and_save(g, 1)
    return g


def cont_graph(prompt, w, h, dur, seed, tag, clip_idx, ctx_abs):
    g = ray_base(UNET_FL2VA, w, h, "video/chain/%s/seg_%d" % (tag, clip_idx - 1))
    g["144"]["inputs"]["noise_seed"] = seed
    g["150"] = {"class_type": LOAD_CLS, "inputs": {"latent_path": ctx_abs, "clip_index": clip_idx - 1}}
    g["133"] = {"class_type": CONT_CLS, "inputs": {
        "clip": ["130", 0], "vae": ["121", 0], "previous_latent": ["150", 0],
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


def split_beats(spec, dur):
    beats = []
    if spec:
        for tok in re.split(r"[;,]", spec):
            tok = tok.strip()
            if not tok:
                continue
            m = re.match(r"^([\d.]+)\s*:\s*(.*)$", tok)
            if m:
                beats.append((float(m.group(1)), m.group(2).strip()))
    out = {}
    for t, txt in beats:
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--beats", default=None)
    ap.add_argument("--segments", type=int, default=4)
    ap.add_argument("--dur", type=float, default=5.0)
    ap.add_argument("--width", type=int, default=864)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--tag", default="chain")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    beats = split_beats(a.beats, a.dur)
    client = "cd1-" + a.tag + "-" + str(random.randint(1000, 9999))
    seed = a.seed if a.seed is not None else random.randint(0, 2**63 - 1)
    existing = count_existing()
    print("existing clips:", existing, "target:", a.segments)
    if existing >= a.segments:
        print("all clips present")
        if not a.merge:
            return
    raw = {}
    for ci in range(existing + 1, a.segments + 1):
        seg_prompt = a.prompt
        if beats.get(ci - 1):
            seg_prompt = a.prompt + "\n\n" + " ".join(beats[ci - 1])
        if ci == 1:
            g = seg0_graph(seg_prompt, a.width, a.height, a.dur, seed, a.tag)
        else:
            g = cont_graph(seg_prompt, a.width, a.height, a.dur, seed, a.tag, ci, slot_path(ci - 1))
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


if __name__ == "__main__":
    main()
