#!/usr/bin/env python3
"""film.py -- generate and assemble the pieces of a film spec.

  film.py run [--only pingfan_ride] [--dry-run]   drive chain_director_v3.py per job (serially)
  film.py assemble                          hard-cut + 2.39 crop + fades via <root>/scripts/assemble.py
  film.py verify                            pyav check of every piece and of the assembled film

`run` always passes --clean: chain slots are global (not per tag), so a leftover slot from a
previous job would make the engine think a segment already exists. Chain jobs get --merge.

Redirect to keep a log:  film.py run --spec ... > projects/<film>/run.log 2>&1
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time

DEFAULT_SPEC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "film_spec.example.json")
FPS = 24
CUT_DIFF = 25.0


def ensure_media_env(spec):
    """verify needs pyav/numpy, which live in ComfyUI's venv (the host has no ffmpeg)."""
    try:
        import av  # noqa: F401
        import numpy  # noqa: F401
        return
    except ImportError:
        pass
    if os.environ.get("H3_FILM_REEXEC") == "1":
        sys.exit("pyav/numpy still missing in %s" % sys.executable)
    py = os.path.expanduser(spec.get("python", ""))
    if not py or not os.path.isfile(py):
        sys.exit('set "python" in the spec to an env that has pyav/numpy '
                 "(e.g. ~/ComfyUI-Deploy/comfyenv/bin/python)")
    os.execve(py, [py, os.path.abspath(__file__)] + sys.argv[1:], dict(os.environ, H3_FILM_REEXEC="1"))



def load_spec(path):
    with open(path, encoding="utf-8") as fh:
        spec = json.load(fh)
    spec["_path"] = os.path.abspath(path)
    spec["_root"] = os.path.expanduser(spec["root"])
    spec["_out"] = os.path.join(spec["_root"], "output")
    if not os.path.isdir(spec["_root"]):
        sys.exit("root not found: %s" % spec["_root"])
    return spec


def job_argv(spec, job):
    p = spec["params"]
    drv = os.path.join(spec["_root"], "scripts", "chain_director_v3.py")
    argv = [os.path.expanduser(spec["python"]), "-u", drv,
            "--tag", job["tag"], "--segments", str(job["segments"]),
            "--dur", str(p["dur"]), "--steps", str(p["steps"]),
            "--width", str(p["width"]), "--height", str(p["height"]),
            "--seed", str(job["seed"]), "--clean", "--prompt", job["prompt"]]
    if job.get("anchor"):
        img = os.path.join(spec["_root"], "projects", spec["film"], "anchors", "%s_best.png" % job["anchor"])
        if os.path.isfile(img):
            argv += ["--first-image", img]
        else:
            print("  !! anchor missing, running text-only: %s" % img, flush=True)
    for i, beat in enumerate(job.get("beats") or []):
        argv += ["--beat", "%ds:%s" % (p["dur"] * (i + 1), beat)]
    if job["segments"] > 1:
        argv.append("--merge")
    return argv


def cmd_run(spec, only, dry):
    jobs = [j for j in spec["jobs"] if not only or j["tag"] in only]
    print("%s: %d job(s) -- %s" % (spec["film"], len(jobs), ", ".join(j["tag"] for j in jobs)), flush=True)
    for job in jobs:
        argv = job_argv(spec, job)
        print("\n===== %s (%s, %d segment(s)) =====" % (job["tag"], job["kind"], job["segments"]), flush=True)
        if dry:
            print("  [dry] " + " ".join(argv), flush=True)
            continue
        t0 = time.time()
        rc = subprocess.run(argv, cwd=spec["_root"]).returncode
        print("===== %s rc=%d in %.0fs =====" % (job["tag"], rc, time.time() - t0), flush=True)
        if rc != 0:
            sys.exit("job %s failed; fix and rerun (slots are cleaned by --clean on the next run)" % job["tag"])


def clip_path(spec, token):
    if os.path.isfile(os.path.join(spec["_root"], token)):
        return os.path.join(spec["_root"], token)
    kind, tag = token.split(":", 1)
    if kind == "shot":
        hits = sorted(glob.glob(os.path.join(spec["_out"], "video/chain", tag, "seg_0_*.mp4")))
        return hits[-1] if hits else None
    if kind == "chain":
        p = os.path.join(spec["_out"], "final_%s.mp4" % tag)
        return p if os.path.isfile(p) else None
    return None


def cmd_assemble(spec, dry):
    a = spec["assemble"]
    out = os.path.join(spec["_root"], a["out"])
    argv = [os.path.expanduser(spec["python"]), os.path.join(spec["_root"], "scripts", "assemble.py"),
            "--out", out, "--aspect", str(a["aspect"]),
            "--fade-in", str(a["fade_in"]), "--fade-out", str(a["fade_out"]), "--hold", str(a["hold"])]
    missing = []
    for token in a["clips"]:
        p = clip_path(spec, token)
        if p is None:
            missing.append(token)
        else:
            argv += ["--clip", p]
    if missing:
        if not dry:
            sys.exit("missing piece(s): %s (run `film.py run` first)" % ", ".join(missing))
        print("  (missing piece(s): %s -- run `film.py run` first)" % ", ".join(missing), flush=True)
    print("%s -> %s" % (" ".join(t for t in a["clips"]), out), flush=True)
    if dry:
        print("  [dry] " + " ".join(argv), flush=True)
        return
    rc = subprocess.run(argv, cwd=spec["_root"]).returncode
    if rc != 0:
        sys.exit("assemble.py failed (%d)" % rc)


def probe(path):
    import av
    if not os.path.isfile(path):
        return None
    with av.open(path) as c:
        v = next((s for s in c.streams if s.type == "video"), None)
        a = next((s for s in c.streams if s.type == "audio"), None)
        n = sum(1 for _ in c.decode(video=0))
        return {"w": v.codec_context.width, "h": v.codec_context.height, "frames": n,
                "secs": n / float(FPS), "audio": a.codec_context.name if a else None,
                "mb": os.path.getsize(path) / 1048576.0}


def cmd_verify(spec):
    import av
    import numpy as np
    bad = 0
    for job in spec["jobs"]:
        p = clip_path(spec, ("chain:" if job["segments"] > 1 else "shot:") + job["tag"])
        info = probe(p) if p else None
        if info is None:
            print("  MISSING %-8s %s" % (job["tag"], p), flush=True)
            bad += 1
        else:
            print("  ok      %-8s %dx%d %4d frames (%5.2fs) audio=%-4s %5.1fMB"
                  % (job["tag"], info["w"], info["h"], info["frames"], info["secs"], info["audio"], info["mb"]),
                  flush=True)
    out = os.path.join(spec["_root"], spec["assemble"]["out"])
    info = probe(out)
    if info is None:
        print("  MISSING  film %s" % out, flush=True)
        return 1
    print("  film    %-8s %dx%d %4d frames (%5.2fs) audio=%-4s %5.1fMB"
          % (spec["film"], info["w"], info["h"], info["frames"], info["secs"], info["audio"], info["mb"]), flush=True)
    with av.open(out) as c:
        v = next(s for s in c.streams if s.type == "video")
        frames = [f.to_ndarray(format="rgb24") for f in c.decode(v)]
    print("  fade    first=%.1f last=%.1f (both near 0 = fade in/out)" % (frames[0].mean(), frames[-1].mean()), flush=True)
    diff = [float(np.abs(frames[i].astype(np.int16) - frames[i - 1].astype(np.int16)).mean())
            for i in range(1, len(frames))]
    cuts = [i for i, d in enumerate(diff) if d > CUT_DIFF]
    print("  cuts    %s" % cuts, flush=True)
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("verb", choices=["run", "assemble", "verify"])
    ap.add_argument("--spec", default=DEFAULT_SPEC)
    ap.add_argument("--only", nargs="*", default=None, help="run: only these tags")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    spec = load_spec(a.spec)
    if a.verb == "verify":
        ensure_media_env(spec)
    if a.verb == "run":
        cmd_run(spec, a.only, a.dry_run)
    elif a.verb == "assemble":
        cmd_assemble(spec, a.dry_run)
    else:
        sys.exit(cmd_verify(spec))


if __name__ == "__main__":
    main()
