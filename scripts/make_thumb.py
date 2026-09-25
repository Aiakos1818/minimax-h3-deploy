#!/usr/bin/env python3
"""make_thumb.py -- generate a small JPEG thumbnail for a material image.

Usage: python make_thumb.py <src> <dst> [max_width]

Run with the comfyenv interpreter (Pillow); the web process keeps stdlib-only
and spawns this only when a thumbnail is missing.
"""
import sys
from PIL import Image, ImageOps


def main():
    if len(sys.argv) < 3:
        sys.exit(2)
    src, dst = sys.argv[1], sys.argv[2]
    maxw = int(sys.argv[3]) if len(sys.argv) > 3 else 480
    im = ImageOps.exif_transpose(Image.open(src))
    im = im.convert("RGB")
    w, h = im.size
    if w > maxw:
        im = im.resize((maxw, max(1, round(h * maxw / w))), Image.LANCZOS)
    im.save(dst, "JPEG", quality=82)


if __name__ == "__main__":
    main()
