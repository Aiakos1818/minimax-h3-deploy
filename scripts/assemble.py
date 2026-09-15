#!/usr/bin/env python3
"""assemble.py -- stitch finished clips into one film.

Clips are given in final order and joined with hard cuts: each is decoded, center
cropped to --aspect, and re-encoded once so the crop, the global fade in/out and an
optional freeze frame before the closing fade are baked in. Audio keeps the source
rate (32k) and gets a short edge fade at every cut so hard cuts do not click.

  assemble.py --out output/final_pf01.mp4 --aspect 2.39 --fade-in 0.5 --fade-out 1.0 \
      --hold 0.6 --clip a.mp4 --clip b.mp4
"""
import argparse
from fractions import Fraction

import av
import numpy as np

FPS = 24
EDGE_FADE = 0.03


def clip_len(path):
    with av.open(path) as c:
        return sum(1 for _ in c.decode(video=0))


def crop_box(frame, aspect):
    w, h = frame.width, frame.height
    if aspect <= 0:
        return 0, w, 0, h
    ch = int(round(w / aspect))
    if ch % 2:
        ch -= 1
    if ch >= h:
        return 0, w, 0, h
    y0 = (h - ch) // 2
    return 0, w, y0, y0 + ch


def edge_fade(audio, sr):
    n = int(EDGE_FADE * sr)
    if audio is None or n * 2 >= audio.shape[-1]:
        return audio
    ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
    audio = audio.copy()
    audio[..., :n] *= ramp
    audio[..., -n:] *= ramp[::-1]
    return audio


def clip_audio(path):
    chunks = []
    sr = 32000
    with av.open(path) as c:
        st = next((s for s in c.streams if s.type == "audio"), None)
        if st is None:
            return None, sr
        sr = st.codec_context.sample_rate or 32000
        for fr in c.decode(st):
            chunks.append(fr.to_ndarray())
    if not chunks:
        return None, sr
    return np.concatenate(chunks, axis=-1).astype(np.float32), sr


def video_frames(path):
    with av.open(path) as c:
        vs = next((s for s in c.streams if s.type == "video"), None)
        for fr in c.decode(vs):
            yield fr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--clip", action="append", required=True, metavar="MP4")
    ap.add_argument("--aspect", type=float, default=0.0, help="center crop width/height, e.g. 2.39")
    ap.add_argument("--fade-in", type=float, default=0.0, metavar="SEC")
    ap.add_argument("--fade-out", type=float, default=0.0, metavar="SEC")
    ap.add_argument("--hold", type=float, default=0.0, metavar="SEC",
                    help="freeze the last frame this long inside the closing fade")
    a = ap.parse_args()

    n_in = int(round(a.fade_in * FPS))
    n_out = int(round(a.fade_out * FPS))
    n_hold = int(round(a.hold * FPS))
    total = sum(clip_len(p) for p in a.clip) + n_hold
    print("pieces:", [(p.rsplit("/", 1)[-1], clip_len(p)) for p in a.clip],
          "hold:", n_hold, "total frames:", total, flush=True)

    audios = []
    sr = 32000
    for p in a.clip:
        aud, sr = clip_audio(p)
        if aud is not None:
            audios.append(edge_fade(aud, sr))

    st = lay = None
    if audios:
        C = max(x.shape[0] for x in audios)
        lay = "stereo" if C > 1 else "mono"
        st = np.concatenate([x.repeat(C, axis=0) if x.shape[0] == 1 else x for x in audios], axis=-1)
        n_in_a = int(round(a.fade_in * sr))
        n_out_a = int(round(a.fade_out * sr))
        if n_in_a:
            st[..., :n_in_a] *= np.linspace(0.0, 1.0, n_in_a, dtype=np.float32)
        if n_out_a and n_out_a < st.shape[-1]:
            st[..., -n_out_a:] *= np.linspace(1.0, 0.0, n_out_a, dtype=np.float32)
        st = np.concatenate([st, np.zeros((C, int(round(a.hold * sr))), dtype=np.float32)], axis=-1)

    with av.open(a.out, "w") as out:
        vs = out.add_stream("h264", rate=FPS)
        vs.pix_fmt = "yuv420p"
        vs.time_base = Fraction(1, FPS)
        vs.options = {"crf": "18", "preset": "medium"}
        a_s = None
        if st is not None:
            a_s = out.add_stream("aac", rate=sr)
            a_s.layout = lay
            a_s.sample_rate = sr
            a_s.time_base = Fraction(1, sr)
            a_s.codec_context.time_base = Fraction(1, sr)
        w_out = h_out = None
        gi = 0
        last = None
        for p in a.clip:
            for fr in video_frames(p):
                x0, x1, y0, y1 = crop_box(fr, a.aspect)
                if w_out is None:
                    w_out, h_out = x1 - x0, y1 - y0
                    vs.width, vs.height = w_out, h_out
                arr = fr.reformat(format="rgb24").to_ndarray()[y0:y1, x0:x1]
                if n_in and gi < n_in:
                    arr = (arr * ((gi + 1) / n_in)).astype(np.uint8)
                if n_out and gi >= total - n_out:
                    arr = (arr * (total - gi) / n_out).astype(np.uint8)
                vf = av.VideoFrame.from_ndarray(np.ascontiguousarray(arr), format="rgb24")
                vf = vf.reformat(width=w_out, height=h_out, format="yuv420p")
                vf.pts = gi
                vf.time_base = Fraction(1, FPS)
                last = arr
                gi += 1
                for pkt in vs.encode(vf):
                    out.mux(pkt)
        for _ in range(n_hold):
            arr = (last * (total - gi) / max(1, n_out)).astype(np.uint8) if n_out else last
            vf = av.VideoFrame.from_ndarray(np.ascontiguousarray(arr), format="rgb24")
            vf = vf.reformat(width=w_out, height=h_out, format="yuv420p")
            vf.pts = gi
            vf.time_base = Fraction(1, FPS)
            gi += 1
            for pkt in vs.encode(vf):
                out.mux(pkt)
        for pkt in vs.encode(None):
            out.mux(pkt)
        assert gi == total, (gi, total)

        if a_s is not None:
            step = 1024
            for i in range(0, st.shape[-1], step):
                af = av.AudioFrame.from_ndarray(np.ascontiguousarray(st[:, i:i + step]), format="fltp", layout=lay)
                af.sample_rate = sr
                af.pts = i
                for pkt in a_s.encode(af):
                    out.mux(pkt)
            for pkt in a_s.encode(None):
                out.mux(pkt)
    print("wrote", a.out, flush=True)


if __name__ == "__main__":
    main()
