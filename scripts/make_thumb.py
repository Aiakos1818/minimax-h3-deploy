#!/usr/bin/env python3
"""make_thumb.py -- make a size-capped copy of a material image, same format.

Usage: python make_thumb.py <src> <dst> [max_width] [max_bytes]

The destination keeps the source extension so the web console serves the
thumbnail with the right content type and alpha stays intact for PNG/WebP.
It downscales/re-encodes until the file is at most <max_bytes> (default
512000), or writes the smallest attempt it can. The console only calls this
for images above the "show the original" size threshold.
"""
import io, os, shutil, sys
from PIL import Image, ImageOps

LOSSY = ("JPEG", "WEBP")
FMT_BY_EXT = {".jpg": "JPEG", ".jpeg": "JPEG", ".png": "PNG", ".webp": "WEBP",
              ".bmp": "BMP", ".gif": "GIF"}


def has_alpha(im):
    return im.mode in ("RGBA", "LA", "PA") or (im.mode == "P" and "transparency" in im.info)


def main():
    if len(sys.argv) < 3:
        sys.exit(2)
    src, dst = sys.argv[1], sys.argv[2]
    maxw = int(sys.argv[3]) if len(sys.argv) > 3 else 480
    cap = int(sys.argv[4]) if len(sys.argv) > 4 else 500 * 1024
    fmt = FMT_BY_EXT.get(os.path.splitext(dst)[1].lower(), "PNG")

    im = ImageOps.exif_transpose(Image.open(src))
    if getattr(im, "n_frames", 1) > 1:
        shutil.copyfile(src, dst)
        return
    if fmt == "JPEG":
        im = im.convert("RGB")
    elif has_alpha(im):
        im = im.convert("RGBA")
    elif im.mode != "RGB":
        im = im.convert("RGB")

    def fit(img, w):
        if img.width > w:
            return img.resize((w, max(1, round(img.height * w / img.width))), Image.LANCZOS)
        return img

    def encode(img, quality):
        buf = io.BytesIO()
        if fmt == "JPEG":
            img.save(buf, fmt, quality=quality, optimize=True, progressive=True)
        elif fmt == "WEBP":
            img.save(buf, fmt, quality=quality, method=4)
        elif fmt == "PNG":
            img.save(buf, fmt, optimize=True)
        else:
            img.save(buf, fmt)
        return buf.getvalue()

    work = fit(im, maxw)
    quality = 82
    data = encode(work, quality)
    guard = 0
    while len(data) > cap and guard < 40:
        guard += 1
        if fmt in LOSSY and quality > 45:
            quality = max(45, quality - 10)
        else:
            w = int(work.width * 0.85)
            if w < 64:
                break
            work = fit(work, w)
            quality = 82
        data = encode(work, quality)

    with open(dst, "wb") as f:
        f.write(data)


if __name__ == "__main__":
    main()
