#!/usr/bin/env python3
"""minimax_h3_edit.py -- render a timeline of clips into one film.

Reads a sequence JSON (written by minimax_h3_web.py) describing an ordered list
of clips plus the transition entering each clip, and renders them to a single
mp4 with PyAV/numpy (no ffmpeg binary needed). Supported transitions:

  cut       hard cut (audio gets a short edge fade so it does not click)
  fade      dip to black: outgoing tail fades out, incoming head fades in
  dissolve  cross dissolve over the overlap (video blend, audio crossfade)
  push      incoming clip slides in from the right, outgoing slides to the left

The whole film is centre cropped to --aspect (0 = keep the source aspect) and
scaled to the first clip's size, and optional global fade in/out are baked in.

Sequence JSON:
  {"fps":24, "aspect":0.0, "fade_in":0.0, "fade_out":0.0,
   "clips":[{"path":"...mp4","trans":{"type":"cut","dur":0.0}}, ...]}

Progress is printed as "[edit] <done>/<total>" and "[edit] stage <name>".
"""
import argparse
import json
from collections import deque
from fractions import Fraction

import av
import numpy as np
from PIL import Image

FPS = 24
EDGE_FADE = 0.03          # seconds, applied at cuts to avoid audio clicks
TRANSITIONS = ("cut", "fade", "dissolve", "push")
DEFAULT_SR = 32000


def log(msg):
    print(msg, flush=True)


# ------------------------------------------------------------------ metadata
def clip_meta(path):
    with av.open(path) as c:
        vs = next((s for s in c.streams if s.type == "video"), None)
        if vs is None:
            raise ValueError("no video stream: %s" % path)
        ar = next((s for s in c.streams if s.type == "audio"), None)
        w, h = vs.codec_context.width, vs.codec_context.height
        rate = float(vs.average_rate) if vs.average_rate else FPS
        frames = 0
        if vs.duration:
            frames = int(round(float(vs.duration * vs.time_base) * rate))
        if frames <= 0:
            frames = sum(1 for _ in c.decode(vs))
        sr = ar.codec_context.sample_rate if ar else 0
        ch = ar.codec_context.channels if ar else 0
    return {"path": path, "w": w, "h": h, "frames": frames, "sr": sr or 0, "ch": ch or 0}


def crop_box(w, h, aspect):
    if aspect <= 0:
        return 0, w, 0, h
    if w / h > aspect:                      # too wide -> crop width
        nw = int(round(h * aspect)); nw -= nw % 2
        x0 = (w - nw) // 2
        return x0, x0 + nw, 0, h
    nh = int(round(w / aspect)); nh -= nh % 2   # too tall -> crop height
    y0 = (h - nh) // 2
    return 0, w, y0, y0 + nh


def trans_type(t):
    tt = (t or {}).get("type") or "cut"
    return tt if tt in TRANSITIONS else "cut"


def trans_dur(t, fps):
    if trans_type(t) == "cut":
        return 0
    try:
        d = float((t or {}).get("dur") or 0)
    except (TypeError, ValueError):
        d = 0
    d = max(0.1, min(1.5, d))
    return max(1, int(round(d * fps)))


def effective_overlaps(clips, fps):
    """Overlap (frames) for each clip index; only dissolve/push overlap."""
    n = len(clips)
    eff = [0] * n
    for i in range(1, n):
        if trans_type(clips[i].get("trans")) in ("dissolve", "push"):
            want = trans_dur(clips[i].get("trans"), fps)
            eff[i] = max(0, min(want, clips[i - 1]["frames"] // 2, clips[i]["frames"] // 2))
    return eff


# ------------------------------------------------------------------- frames
def scale_crop(arr, box, size):
    x0, x1, y0, y1 = box
    a = arr[y0:y1, x0:x1]
    if (x1 - x0, y1 - y0) == size:
        return np.ascontiguousarray(a)
    im = Image.fromarray(a).resize(size, Image.LANCZOS)
    return np.ascontiguousarray(np.asarray(im))


def blend(a, b, t, kind):
    if kind == "dissolve":
        return ((1.0 - t) * a + t * b).astype(np.uint8)
    w = a.shape[1]                                     # push (left)
    shift = int(round(t * w))
    if shift <= 0:
        return a
    if shift >= w:
        return b
    out = np.empty_like(a)
    out[:, :w - shift] = a[:, shift:]
    out[:, w - shift:] = b[:, :shift]
    return out


def video_frames(path):
    with av.open(path) as c:
        vs = next((s for s in c.streams if s.type == "video"), None)
        for fr in c.decode(vs):
            yield fr


# -------------------------------------------------------------------- audio
def clip_audio(path, sr, ch):
    chunks = []
    with av.open(path) as c:
        st = next((s for s in c.streams if s.type == "audio"), None)
        if st is None:
            return np.zeros((ch, 0), dtype=np.float32)
        src_sr = st.codec_context.sample_rate or sr
        for fr in c.decode(st):
            chunks.append(fr.to_ndarray())
    if not chunks:
        return np.zeros((ch, 0), dtype=np.float32)
    a = np.concatenate(chunks, axis=-1).astype(np.float32)
    if a.shape[0] == 1 and ch > 1:
        a = np.repeat(a, ch, axis=0)
    elif a.shape[0] > ch:
        a = a[:ch]
    if src_sr != sr:                                   # linear resample
        n = int(round(a.shape[-1] * sr / src_sr))
        idx = np.linspace(0, a.shape[-1] - 1, n)
        a = np.stack([np.interp(idx, np.arange(a.shape[-1]), a[c]) for c in range(a.shape[0])])
    return a.astype(np.float32)


def edge_fade(a, sr, head, tail):
    n = int(EDGE_FADE * sr)
    if n <= 0 or a.shape[-1] < 2 * n:
        return a
    a = a.copy()
    if head:
        a[..., :n] *= np.linspace(0.0, 1.0, n, dtype=np.float32)
    if tail:
        a[..., -n:] *= np.linspace(1.0, 0.0, n, dtype=np.float32)
    return a


def build_audio(clips, eff, fps, sr, ch, total_frames):
    out, reserved = [], None
    for i, clip in enumerate(clips):
        a = clip_audio(clip["path"], sr, ch)
        want = int(round(clip["frames"] / fps * sr))   # match the clip's video length
        if a.shape[-1] < want:
            a = np.concatenate([a, np.zeros((ch, want - a.shape[-1]), dtype=np.float32)], axis=-1)
        elif a.shape[-1] > want:
            a = a[:, :want]
        start, head_cut, tail_cut = 0, True, True
        if i > 0:
            t = trans_type(clip.get("trans"))
            if t in ("dissolve", "push") and reserved is not None:
                n = min(eff[i], reserved.shape[-1], a.shape[-1])
                if n > 0:
                    ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
                    out.append(reserved[:, -n:] * (1.0 - ramp) + a[:, :n] * ramp)
                    start = n
                reserved = None
                head_cut = False
            elif t == "fade":
                n = min(trans_dur(clip.get("trans"), fps) * sr // fps, a.shape[-1])
                if n > 0:
                    a = a.copy()
                    a[..., :n] *= np.linspace(0.0, 1.0, n, dtype=np.float32)
                    head_cut = False
        rest = a[:, start:]
        if i + 1 < len(clips):
            nt = trans_type(clips[i + 1].get("trans"))
            if nt in ("dissolve", "push"):
                n = min(eff[i + 1], rest.shape[-1])
                if n > 0:
                    reserved = rest[:, -n:].copy()
                    rest = rest[:, :-n]
                tail_cut = False
            elif nt == "fade":
                n = min(trans_dur(clips[i + 1].get("trans"), fps) * sr // fps, rest.shape[-1])
                if n > 0:
                    rest = rest.copy()
                    rest[..., -n:] *= np.linspace(1.0, 0.0, n, dtype=np.float32)
                tail_cut = False
        out.append(edge_fade(rest, sr, head_cut, tail_cut))
    if not out:
        return None
    st = np.concatenate(out, axis=-1)
    want = int(round(total_frames / fps * sr))         # no drift vs the video
    if st.shape[-1] < want:
        st = np.concatenate([st, np.zeros((ch, want - st.shape[-1]), dtype=np.float32)], axis=-1)
    elif st.shape[-1] > want:
        st = st[:, :want]
    return st


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True, help="sequence json")
    ap.add_argument("--out", required=True, help="output mp4")
    a = ap.parse_args()
    with open(a.seq, encoding="utf-8") as f:
        seq = json.load(f)

    clips = [c for c in seq.get("clips", []) if c.get("path")]
    if not clips:
        log("[edit] error: empty sequence"); raise SystemExit(2)
    fps = int(seq.get("fps") or FPS)
    aspect = float(seq.get("aspect") or 0)

    metas = [clip_meta(c["path"]) for c in clips]
    for c, m in zip(clips, metas):
        c["frames"] = m["frames"]
    if any(c["frames"] <= 0 for c in clips):
        log("[edit] error: empty clip"); raise SystemExit(2)
    eff = effective_overlaps(clips, fps)
    total = sum(c["frames"] for c in clips) - sum(eff)

    cw, chh = metas[0]["w"], metas[0]["h"]
    bx0, bx1, by0, by1 = crop_box(cw, chh, aspect)
    tw, th = bx1 - bx0, by1 - by0
    log("[edit] stage 准备素材")
    log("[edit] clips=%d target=%dx%d aspect=%s total_frames=%d"
        % (len(clips), tw, th, aspect or "source", total))
    for i, c in enumerate(clips):
        log("[edit]  clip %d: %s frames=%d trans=%s"
            % (i + 1, c["path"].rsplit("/", 1)[-1], c["frames"],
               trans_type(c.get("trans")) if i else "start"))

    n_in_global = int(round(float(seq.get("fade_in") or 0) * fps))
    n_out_global = int(round(float(seq.get("fade_out") or 0) * fps))
    sr = max([m["sr"] for m in metas] or [DEFAULT_SR]) or DEFAULT_SR
    ach = max([m["ch"] for m in metas] or [1]) or 1
    want_audio = bool(seq.get("audio", True))
    layout = "stereo" if ach > 1 else "mono"

    with av.open(a.out, "w") as out:
        vs = out.add_stream("h264", rate=fps)
        vs.pix_fmt = "yuv420p"
        vs.time_base = Fraction(1, fps)
        vs.options = {"crf": "18", "preset": "medium"}
        vs.width, vs.height = tw, th
        a_s = None                       # muxer needs every stream before packets
        if want_audio:
            a_s = out.add_stream("aac", rate=sr)
            a_s.layout = layout
            a_s.sample_rate = sr
            a_s.time_base = Fraction(1, sr)
            a_s.codec_context.time_base = Fraction(1, sr)

        gi = 0
        last_pct = -1
        prev_tail = None
        log("[edit] stage 合成画面")
        for i, clip in enumerate(clips):
            it = video_frames(clip["path"])
            box = crop_box(cw, chh, aspect)

            def emit(frame):
                nonlocal gi, last_pct
                if n_in_global and gi < n_in_global:
                    frame = (frame * ((gi + 1) / n_in_global)).astype(np.uint8)
                if n_out_global and gi >= total - n_out_global:
                    frame = (frame * ((total - gi) / n_out_global)).astype(np.uint8)
                vf = av.VideoFrame.from_ndarray(np.ascontiguousarray(frame), format="rgb24")
                vf = vf.reformat(width=tw, height=th, format="yuv420p")
                vf.pts = gi
                vf.time_base = Fraction(1, fps)
                gi += 1
                for pkt in vs.encode(vf):
                    out.mux(pkt)
                pct = gi * 100 // total
                if pct != last_pct and pct % 2 == 0:
                    last_pct = pct
                    log("[edit] %d/%d" % (gi, total))

            if i > 0:
                tt = trans_type(clip.get("trans"))
                if tt in ("dissolve", "push"):
                    n = eff[i]
                    heads = [frames2rgb(next(it), box, (tw, th)) for _ in range(n)]
                    for k in range(n):
                        emit(blend(prev_tail[k], heads[k], (k + 1) / (n + 1), tt))
                elif tt == "fade":
                    n = min(trans_dur(clip.get("trans"), fps), clip["frames"] // 2)
                    for k in range(n):
                        f = frames2rgb(next(it), box, (tw, th))
                        emit((f * ((k + 1) / (n + 1))).astype(np.uint8))

            hold = 0
            if i + 1 < len(clips):
                nt = trans_type(clips[i + 1].get("trans"))
                if nt in ("dissolve", "push"):
                    hold = eff[i + 1]
                elif nt == "fade":
                    hold = min(trans_dur(clips[i + 1].get("trans"), fps), clip["frames"] // 2)
            dq = deque()
            for fr in it:
                dq.append(frames2rgb(fr, box, (tw, th)))
                if len(dq) > hold:
                    emit(dq.popleft())
            tail = list(dq)
            if i + 1 < len(clips):
                nt = trans_type(clips[i + 1].get("trans"))
                if nt in ("dissolve", "push"):
                    prev_tail = tail
                elif nt == "fade":
                    n = len(tail)
                    for idx, f in enumerate(tail):
                        emit((f * ((n - 1 - idx) / max(1, n - 1))).astype(np.uint8))
                else:
                    for f in tail:
                        emit(f)
            else:
                for f in tail:
                    emit(f)
        for pkt in vs.encode(None):
            out.mux(pkt)
        log("[edit] frames=%d/%d" % (gi, total))

        if a_s is not None:
            log("[edit] stage 合成音频")
            st = build_audio(clips, eff, fps, sr, ach, total)
            if st is not None:
                n_in_a = int(round(float(seq.get("fade_in") or 0) * sr))
                n_out_a = int(round(float(seq.get("fade_out") or 0) * sr))
                if n_in_a:
                    st[..., :n_in_a] *= np.linspace(0.0, 1.0, n_in_a, dtype=np.float32)
                if n_out_a and n_out_a < st.shape[-1]:
                    st[..., -n_out_a:] *= np.linspace(1.0, 0.0, n_out_a, dtype=np.float32)
                step = 1024
                for i in range(0, st.shape[-1], step):
                    af = av.AudioFrame.from_ndarray(
                        np.ascontiguousarray(st[:, i:i + step]), format="fltp", layout=layout)
                    af.sample_rate = sr
                    af.pts = i
                    for pkt in a_s.encode(af):
                        out.mux(pkt)
                for pkt in a_s.encode(None):
                    out.mux(pkt)
    log("[edit] wrote %s" % a.out)


def frames2rgb(fr, box, size):
    arr = fr.reformat(format="rgb24").to_ndarray()
    return scale_crop(arr, box, size)


if __name__ == "__main__":
    main()
