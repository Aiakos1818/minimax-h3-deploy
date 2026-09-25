#!/usr/bin/env python3
"""make_vthumb.py -- generate a small JPEG poster for a material video.

Usage: python make_vthumb.py <src> <dst> [max_width]

Run with the comfyenv interpreter (PyAV + Pillow); the web process keeps
stdlib-only and spawns this only when a poster is missing. A few frames are
sampled and the most detailed one is kept, to avoid blank intro cards.
"""
import sys

import av
from PIL import ImageStat

FRACTIONS = (0.1, 0.3, 0.5, 0.7)


def detail(im):
    return ImageStat.Stat(im.convert("L")).stddev[0]


def main():
    if len(sys.argv) < 3:
        sys.exit(2)
    src, dst = sys.argv[1], sys.argv[2]
    maxw = int(sys.argv[3]) if len(sys.argv) > 3 else 480
    im = None
    with av.open(src) as c:
        stream = c.streams.video[0]
        dur = float(c.duration / av.time_base) if c.duration else 0.0
        best = -1.0
        for frac in FRACTIONS:
            if dur <= 0 and frac != FRACTIONS[0]:
                break
            try:
                c.seek(int(dur * frac * av.time_base))
                frame = next(c.decode(stream))
                cand = frame.to_image().convert("RGB")
            except Exception:
                continue
            score = detail(cand)
            if score > best:
                best, im = score, cand
        if im is None:
            try:
                c.seek(0)
                im = next(c.decode(stream)).to_image().convert("RGB")
            except Exception:
                sys.exit(1)
    w, h = im.size
    if w > maxw:
        im = im.resize((maxw, max(1, round(h * maxw / w))))
    im.save(dst, "JPEG", quality=82)


if __name__ == "__main__":
    main()
