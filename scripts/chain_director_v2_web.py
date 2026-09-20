#!/usr/bin/env python3
"""chain_director_v2_web.py -- LAN web console for chain_director_v2.py (stdlib only).

Runs on the GPU host next to ComfyUI and exposes a dashboard on 0.0.0.0:8189 by
default so other LAN machines can drive chain_director_v2.py without ssh:
  * mirrors every chain_director_v2.py CLI parameter as form fields
  * multipart uploads for first/last-image and ref-image/video/audio
  * serial job queue (GPU fits one chain at a time), streaming log tail,
    cancel, history, "continue" (resume) from a finished job
  * output gallery: final_*.mp4 and per-tag segment list with Range playback

Runtime / control:
  start (foreground):   ~/ComfyUI-Deploy/comfyenv/bin/python ~/MiniMax-H3-Deploy/scripts/chain_director_v2_web.py
  start (background):   ~/ComfyUI-Deploy/comfyenv/bin/python ~/MiniMax-H3-Deploy/scripts/chain_director_v2_web.py --start
  stop:                 ~/ComfyUI-Deploy/comfyenv/bin/python ~/MiniMax-H3-Deploy/scripts/chain_director_v2_web.py --stop
  status:               ~/ComfyUI-Deploy/comfyenv/bin/python ~/MiniMax-H3-Deploy/scripts/chain_director_v2_web.py --status

Guards / conventions (mirror the CLI docs):
  * every fresh web job clears output/h3_continuous/chain_*.safetensors and the
    same-tag segment dir first (clean_slots, default on) -> one web job = one
    fresh chain; tick it off to resume manually
  * "continue" reuses a finished job's tag + staged media + slot files, forcing
    clean_slots off; it refuses if the slot files were rewritten/cleared since
  * busy_guard: if an external chain_director (e.g. a manual CLI run) is alive,
    queued jobs wait for it instead of racing it (clear=restart would kill it)
  * a job's default --clear restart stops/starts ComfyUI at its start (same as
    the CLI); cancel is TERM-grade - GPU leftovers are reset by the next run's
    clear

Layout under the data dir (default ~/MiniMax-H3-Deploy/.h3web):
  h3web.pid                 daemon pid
  h3web.log                 daemon stdout/stderr (only when --start)
  jobs/<job_id>/config.json  job config + staged upload paths
  jobs/<job_id>/log.txt      child process stdout/stderr
  jobs/<job_id>/status.json  status snapshot (recovery / history)
  jobs/<job_id>/uploads/     staged uploads (<seq>_<name>)

Only the Python stdlib is used; the web server never imports numpy/av/safetensors.
"""
import argparse, glob, io, json, mimetypes, os, re, shutil, signal, socket
import subprocess, sys, threading, time, uuid, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs, unquote, quote

DEFAULT_PORT = 8189
COMVFY_BASE = "http://127.0.0.1:8188"
UPLOAD_MAX = 2 * 1024 * 1024 * 1024   # reject single body > 2 GiB
BUSY_WAIT_MAX_S = 6 * 3600            # cap how long a job waits on external busy

NOW = lambda: time.strftime("%Y-%m-%d %H:%M:%S")


def _safe_name(name):
    name = os.path.basename(name.replace("\\", "/"))
    name = re.sub(r"[^\w.\- ]", "_", name)
    return name or "file"


# --------------------------------------------------------------------------
# minimal multipart/form-data parser (keeps file bytes in memory, no deps)
# --------------------------------------------------------------------------
def _parse_mpart_header(data):
    """data: bytes of one part's header block -> dict of disposition fields."""
    d = {}
    ct = None
    for line in data.split(b"\r\n"):
        line = line.decode("latin-1", "replace")
        low = line.lower()
        if low.startswith("content-disposition:"):
            m = re.search(r'name="([^"]*)"', line)
            if m:
                d["name"] = m.group(1)
            m = re.search(r'filename="(.*)"', line)
            if m:
                d["filename"] = m.group(1)
        elif low.startswith("content-type:"):
            ct = line.split(":", 1)[1].strip()
    d["content_type"] = ct
    return d


def parse_multipart(body, boundary):
    """Return (fields: dict[str, list[str]], files: list[dict]).

    fields values are decoded utf-8 strings (one list entry per occurrence, so
    repeated/multi values keep order); files are {name, filename, content_type,
    content: bytes} in body order.
    """
    fields, files = {}, []
    delim = b"--" + boundary
    if not body.startswith(delim):
        raise ValueError("bad multipart body: missing leading boundary")
    start = len(delim)
    while True:
        if body[start:start + 2] == b"--":
            break                       # closing boundary
        if body[start:start + 2] == b"\r\n":
            start += 2
        hdr_end = body.index(b"\r\n\r\n", start)
        hdr = _parse_mpart_header(body[start:hdr_end])
        c0 = hdr_end + 4
        m = body.find(b"\r\n" + delim, c0)
        if m < 0:
            raise ValueError("bad multipart body: missing boundary separator")
        content = body[c0:m]
        start = m + 2 + len(delim)
        name = hdr.get("name")
        if not name:
            continue
        if "filename" in hdr:
            files.append({"name": name, "filename": hdr["filename"],
                          "content_type": hdr.get("content_type"),
                          "content": content})
        else:
            fields.setdefault(name, []).append(content.decode("utf-8", "replace").strip())
    return fields, files


# --------------------------------------------------------------------------
# process helpers
# --------------------------------------------------------------------------
def _is_zombie(pid):
    """True if pid is a zombie (gone, waiting for its parent to reap)."""
    try:
        st = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)],
                            capture_output=True, text=True, timeout=5).stdout.strip()
        return st == "Z"
    except Exception:
        return False


def external_chain_pids(own_children):
    """Pids of external chain_director_v2.py processes (not spawned by us).

    NOTE: after a chain finishes, the raylight workers stay resident with FSDP /
    VAE in VRAM on purpose (cleared by the NEXT run's clear step). So an alive
    driver process alone never means "busy" - see Manager.blocking_busy which
    keys off the ComfyUI /queue instead. This is purely informational.
    """
    try:
        out = subprocess.run(["pgrep", "-af", "chain_director_v2"],
                             capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return []
    mine = set(own_children) | {os.getpid()}
    pids = []
    seen = set()
    for ln in out.splitlines():
        parts = ln.split(None, 1)
        if len(parts) != 2:
            continue
        pid_s, cmd = parts
        if not (re.search(r"chain_director_v2\.py", cmd) and not re.search(r"chain_director_v2_web", cmd)):
            continue
        try:
            pid = int(pid_s)
        except ValueError:
            continue
        if pid in mine or pid in seen:
            continue
        seen.add(pid)
        if _is_zombie(pid):
            continue
        pids.append(pid)
    return pids


class ComfyHealth:
    def __init__(self, base):
        self.base = base
        self._lock = threading.Lock()
        self._cached = (False, None, 0.0)
        self._ts = 0.0

    def check(self):
        with self._lock:
            if time.time() - self._ts < 5:
                return self._cached
        t0 = time.time()
        try:
            urllib.request.urlopen(self.base + "/system_stats", timeout=3).read()
            up, ms = True, int((time.time() - t0) * 1000)
        except Exception:
            up, ms = False, None
        with self._lock:
            self._cached = (up, ms, time.time())
            self._ts = time.time()
        return self._cached

    def running_prompts(self):
        """Number of prompts ComfyUI is executing right now (or None if down)."""
        try:
            q = json.load(urllib.request.urlopen(self.base + "/queue", timeout=3))
            return len(q.get("queue_running") or [])
        except Exception:
            return None


# --------------------------------------------------------------------------
# stage hints parsed out of the driver log
# --------------------------------------------------------------------------
_STAGE_PATTERNS = [
    (r"\[clear\] stopping", "清场中: 重启 ComfyUI"),
    (r"service up after", "清场完成(服务已就绪)"),
    (r"\[clip(\d+)\] submit", "第 %s 段: 提交生成"),
    (r"\[clip(\d+)\] done", "第 %s 段: 完成"),
    (r"stitching", "缝合中(merge)"),
    (r"FINAL", "产出 final"),
    (r"NODE ERROR", "节点出错"),
    (r"submit error", "提交错误"),
]
_PREV_CLIP = re.compile(r"\[clip(\d+)\] done")


def stage_from_log(text):
    """Scan the latest meaningful log text -> (label, clip)."""
    lines = [l for l in text.splitlines() if l.strip()]
    clip = None
    for l in lines:
        m = _PREV_CLIP.search(l)
        if m:
            clip = int(m.group(1))
    for pat, lab in reversed(_STAGE_PATTERNS):
        for l in reversed(lines):
            m = re.search(pat, l)
            if m:
                c = m.group(1) if len(m.groups()) else clip
                label = (lab % c) if (c is not None and "%s" in lab) else lab
                return label, clip
    if not lines:
        return "启动中...", None
    return lines[-1][:80], clip


# --------------------------------------------------------------------------
# job manager
# --------------------------------------------------------------------------
class Manager:
    def __init__(self, root, driver, comfy_base):
        self.root = root
        self.driver = os.path.abspath(driver)
        self.data = os.path.join(root, ".h3web")
        self.jobs_dir = os.path.join(self.data, "jobs")
        self.lock = threading.RLock()
        self._stop = False
        self.jobs = {}                 # id -> job dict
        self.queue = []                # ordered job ids (queued then running)
        self.current = None            # job id being executed
        self.children = set()          # pids spawned by this server
        self.comfy = ComfyHealth(comfy_base)
        self._cv = threading.Condition(self.lock)
        os.makedirs(self.jobs_dir, exist_ok=True)

    # ---- persistence -----------------------------------------------------
    def _job_dir(self, jid):
        return os.path.join(self.jobs_dir, jid)

    def persist_cfg(self, job):
        d = self._job_dir(job["id"])
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "config.json"), "w", encoding="utf-8") as f:
            json.dump({k: job[k] for k in ("cfg",)}, f, ensure_ascii=False, indent=2)
        # flatten for readability
        with open(os.path.join(d, "config.flat.json"), "w", encoding="utf-8") as f:
            json.dump(job["cfg"], f, ensure_ascii=False, indent=2)

    def persist_status(self, job):
        with open(os.path.join(self._job_dir(job["id"]), "status.json"), "w", encoding="utf-8") as f:
            json.dump(job["st"], f, ensure_ascii=False, indent=2)

    def recover(self):
        for d in sorted(os.listdir(self.jobs_dir)):
            jdir = os.path.join(self.jobs_dir, d)
            if not os.path.isdir(jdir):
                continue
            try:
                with open(os.path.join(jdir, "config.json"), encoding="utf-8") as f:
                    cfg = json.load(f)["cfg"]
            except Exception:
                cfg = None
            st = None
            try:
                with open(os.path.join(jdir, "status.json"), encoding="utf-8") as f:
                    st = json.load(f)
            except Exception:
                pass
            job = {"id": d, "cfg": cfg, "st": st or {"id": d}}
            job.setdefault("log", os.path.join(jdir, "log.txt"))
            if not st or st.get("status") == "running":
                job["st"].update(status="interrupted", ended=NOW(),
                                 err="web 重启中断: GPU 残留由下个任务 --clear restart 清场")
                self.persist_status(job)
            job.setdefault("child", None)
            self.jobs[d] = job

    # ---- submit ----------------------------------------------------------
    def new_id(self):
        return time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:4]

    def submit(self, cfg, jid=None):
        """Validate + queue a job. cfg already resolved (see build_cfg).
        When jid is given the caller has already created + staged the job dir
        (files must be on disk before the worker can touch them)."""
        with self.lock:
            if jid is None:
                jid = self.new_id()
            job = {"id": jid, "cfg": cfg, "log": os.path.join(self._job_dir(jid), "log.txt"),
                   "st": {"id": jid, "status": "queued", "created": NOW(), "created_ts": time.time(),
                          "tag": cfg["tag"], "engine": cfg["engine"],
                          "target_segments": cfg["params"]["segments"]},
                   "child": None}
            self.jobs[jid] = job
            self.persist_cfg(job)
            self.persist_status(job)
            self.queue.append(jid)
            with self._cv:
                self._cv.notify()
            return jid

    # ---- worker ----------------------------------------------------------
    def start_worker(self):
        threading.Thread(target=self._worker_loop, daemon=True).start()

    def _worker_loop(self):
        while not self._stop:
            with self._cv:
                while not self._stop and not self.queue:
                    self._cv.wait()
                if self._stop:
                    return
                jid = self.queue.pop(0)
            self.current = jid
            self._set(jid, "status", "running")
            try:
                self._run(jid)
            except Exception as e:                     # never wedge the worker
                self._set(jid, "status", "failed", err="内部错误: %r" % e)
            with self.lock:
                self.current = None
                with self._cv:
                    self._cv.notify_all()

    def _set(self, jid, key, value, **extra):
        with self.lock:
            job = self.jobs.get(jid)
            if not job:
                return
            job["st"][key] = value
            for k, v in extra.items():
                job["st"][k] = v
            if key in ("status", "stage") or extra:
                self.persist_status(job)

    def _tail_text(self, job):
        try:
            with open(job["log"], "rb") as f:
                return f.read().decode("utf-8", "replace")
        except Exception:
            return ""

    def _stage(self, job):
        text = self._tail_text(job)
        label, clip = stage_from_log(text)
        job["st"]["stage"] = {"label": label, "ts": NOW()}
        if clip:
            job["st"]["last_clip_done"] = clip

    def _clean_slots(self, cfg, log):
        """fresh-run cleanup mirroring the CLI's slot convention."""
        for f in sorted(glob.glob(os.path.join(self.root, "output/h3_continuous/chain_*.safetensors"))):
            os.remove(f)
            log.write("[web] rm %s\n" % os.path.relpath(f, self.root))
        tdir = os.path.join(self.root, "output/video/chain", cfg["tag"])
        if os.path.isdir(tdir):
            shutil.rmtree(tdir, ignore_errors=True)
            log.write("[web] rm -r video/chain/%s (old segments)\n" % cfg["tag"])
        fin = os.path.join(self.root, "output", "final_%s.mp4" % cfg["tag"])
        if os.path.isfile(fin):
            os.remove(fin)
            log.write("[web] rm final_%s.mp4 (old)\n" % cfg["tag"])
        log.flush()

    def _snapshot_slots(self):
        out = {}
        for p in sorted(glob.glob(os.path.join(self.root, "output/h3_continuous/chain_*.safetensors"))):
            rel = os.path.relpath(p, self.root)
            out[rel] = os.stat(p).st_mtime
        return out

    def _check_continue(self, cfg, log):
        base = cfg.get("base_slots") or {}
        if not base:
            log.write("[web] continue: base had no slot snapshot, refusing\n")
            return "基准任务没有可用槽位(续拍中止)"
        cur = self._snapshot_slots()
        if set(cur) != set(base):
            log.write("[web] continue: slot files changed since base job\n")
            return "槽位已被清理或覆盖,无法续拍; 请新建任务"
        for rel, mt in base.items():
            if abs(cur[rel] - mt) > 1.5:
                return "槽位 %s 时间不匹配,无法续拍; 请新建任务" % rel
        return None

    def blocking_busy(self):
        """Reasons a fresh run with --clear restart must wait (or empty = free).

        The authoritative signal is ComfyUI /queue: restarting the service would
        kill whatever ComfyUI is currently executing. Note that alive external
        chain_director processes alone do NOT count - after a chain finishes the
        raylight workers deliberately stay resident (FSDP/VAE in VRAM, cleared by
        the NEXT run's clear step), so "process alive" != "generating".
        """
        reasons = []
        qr = self.comfy.running_prompts()
        if qr:
            reasons.append("ComfyUI 正在执行 %d 个任务" % qr)
        elif qr is None and external_chain_pids(self.children):
            # ComfyUI is down but an external driver is alive: it may be inside
            # its own stop/start clear step; racing another restart would clash.
            reasons.append("ComfyUI 离线且外部 driver 存活(可能正在重启)")
        return reasons

    def _run(self, jid):
        job = self.jobs[jid]
        cfg = job["cfg"]
        p = cfg["params"]
        log_path = job["log"]
        with open(log_path, "a", encoding="utf-8") as log:
            log.write("=== job %s | tag=%s engine=%s ===\n"
                      % (jid, cfg["tag"], cfg["engine"]))
            # ---- busy guard: wait for external chain_director to finish ----
            if p.get("clear") == "restart":
                t0 = time.time()
                while not self._stop:
                    reasons = self.blocking_busy()
                    if not reasons:
                        break
                    if time.time() - t0 > BUSY_WAIT_MAX_S:
                        self._set(jid, "status", "failed",
                                  err="等待外部 GPU 任务超时(%s)" % "; ".join(reasons))
                        log.write("[web] busy timeout: %s\n" % "; ".join(reasons))
                        log.flush()
                        return
                    self._set(jid, "status", "running")
                    job["st"]["stage"] = {"label": "等待: %s" % "; ".join(reasons), "ts": NOW()}
                    self.persist_status(job)
                    log.write("[web] busy: %s ...\n" % "; ".join(reasons))
                    log.flush()
                    time.sleep(6)
                    if job["st"].get("cancel_requested"):
                        self._set(jid, "status", "cancelled", ended=NOW(), err="取消(排队等待中)")
                        return
            # ---- fresh cleanup / continue validation ----
            if cfg.get("clean_slots"):
                self._clean_slots(cfg, log)
            if cfg.get("continue_of"):
                err = self._check_continue(cfg, log)
                if err:
                    self._set(jid, "status", "failed", ended=NOW(), err=err)
                    log.write("[web] %s\n" % err)
                    log.flush()
                    return
            # ---- launch the driver ----
            argv = self._build_argv(cfg)
            log.write("cmd: %s\n\n" % shlex_join(argv))
            log.flush()
            env = dict(os.environ, PYTHONUNBUFFERED="1")
            if sys.platform != "win32":
                popen = subprocess.Popen(argv, cwd=self.root, env=env,
                                         stdout=log, stderr=subprocess.STDOUT,
                                         start_new_session=True)
            else:
                popen = subprocess.Popen(argv, cwd=self.root, env=env,
                                         stdout=log, stderr=subprocess.STDOUT)
            with self.lock:
                job["child"] = popen
                self.children.add(popen.pid)
            self._set(jid, "status", "running")
            # ---- monitor ----
            last_ps = 0.0
            while popen.poll() is None:
                if job["st"].get("cancel_requested"):
                    self._term(job)
                if time.time() - last_ps > 2:
                    self._stage(job)
                    self.persist_status(job)
                    last_ps = time.time()
                time.sleep(0.5)
            rc = popen.returncode
            with self.lock:
                self.children.discard(popen.pid)
                job["child"] = None
            self._stage(job)
            # finalize
            if job["st"].get("cancel_requested"):
                self._set(jid, "status", "cancelled", ended=NOW(), exit=rc,
                          err="已取消(TERM rc=%s)" % rc)
                log.write("\n[web] cancelled (rc=%s)\n" % rc)
            elif rc == 0:
                snap = self._snapshot_slots()
                fin_rel = None
                fin = os.path.join(self.root, "output", "final_%s.mp4" % cfg["tag"])
                if os.path.isfile(fin):
                    fin_rel = os.path.relpath(fin, os.path.join(self.root, "output"))
                dur = time.time() - job["st"].get("created_ts", time.time())
                self._set(jid, "status", "done", ended=NOW(), exit=0, err=None,
                          duration=int(dur),
                          slots_snapshot={k: round(v, 3) for k, v in snap.items()},
                          final_rel=fin_rel)
                log.write("\n[web] done rc=0 final=%s slots=%d\n"
                          % (fin_rel, len(snap)))
            else:
                self._set(jid, "status", "failed", ended=NOW(), exit=rc,
                          err="driver exited rc=%s" % rc)
                log.write("\n[web] failed rc=%s\n" % rc)
            log.flush()

    def _term(self, job):
        child = job.get("child")
        if child is None:
            return
        try:
            if sys.platform != "win32":
                os.killpg(child.pid, signal.SIGTERM)
            else:
                child.terminate()
        except Exception:
            pass

    def cancel(self, jid):
        with self.lock:
            job = self.jobs.get(jid)
            if not job or job["st"].get("status") not in ("queued", "running"):
                return False, "任务不存在或已结束"
            if job["st"]["status"] == "queued":
                job["st"]["status"] = "cancelled"
                job["st"]["ended"] = NOW()
                self.persist_status(job)
                self.queue = [x for x in self.queue if x != jid]
                return True, "已取消(排队中)"
            job["st"]["cancel_requested"] = True
            self._term(job)
            return True, "已发送取消(TERM)"

    def delete(self, jid):
        with self.lock:
            job = self.jobs.get(jid)
            if not job:
                return False, "不存在"
            st = job["st"].get("status")
            if st == "running":
                return False, "运行中的任务不能直接删除，请先取消"
            if st == "queued":
                self.queue = [x for x in self.queue if x != jid]
                job["st"]["status"] = "cancelled"
                job["st"]["ended"] = NOW()
            shutil.rmtree(self._job_dir(jid), ignore_errors=True)
            del self.jobs[jid]
            if st == "queued":
                return True, "已从队列移除(含已上传素材)"
            return True, "已删除记录(不删除 output 产物)"

    # ---- argv ------------------------------------------------------------
    def _build_argv(self, cfg):
        p = cfg["params"]
        staged = cfg.get("staged_files", {})
        a = [sys.executable, self.driver, "--tag", cfg["tag"],
             "--prompt", p["prompt"],
             "--segments", str(p["segments"]), "--dur", str(p["dur"]),
             "--width", str(p["width"]), "--height", str(p["height"]),
             "--steps", str(p["steps"])]
        for row in beats_to_rows(p.get("beats")):
            a += ["--beat", row]
        if p.get("seed") is not None:
            a += ["--seed", str(p["seed"])]
        if p.get("merge"):
            a += ["--merge"]
        clear = p.get("clear", "restart")
        if clear != "restart":
            a += ["--clear", clear]
        if p.get("ref_image_size") and p["ref_image_size"] != "match":
            a += ["--ref-image-size", p["ref_image_size"]]
        for f in staged.get("first_image", []):
            a += ["--first-image", f]
        for f in staged.get("last_image", []):
            a += ["--last-image", f]
        for f in staged.get("ref_image", []):
            a += ["--ref-image", f]
        for f in staged.get("ref_video", []):
            a += ["--ref-video", f]
        for f in staged.get("ref_audio", []):
            a += ["--ref-audio", f]
        return a

    # ---- API snapshots ---------------------------------------------------
    def info(self, job):
        st = job["st"]
        snap = st.get("slots_snapshot") or {}
        return {"id": st["id"], "status": st.get("status"), "tag": st.get("tag"),
                "engine": st.get("engine"), "target_segments": st.get("target_segments"),
                "created": st.get("created"), "ended": st.get("ended"),
                "duration": st.get("duration"), "exit": st.get("exit"),
                "err": st.get("err"), "final_rel": st.get("final_rel"),
                "slots_snapshot": snap, "slots_count": len(snap),
                "stage": st.get("stage"), "cancel_requested": bool(st.get("cancel_requested"))}

    def state(self):
        with self.lock:
            cur = self.info(self.jobs[self.current]) if self.current and self.current in self.jobs else None
            q = [self.info(self.jobs[j]) for j in self.queue if j in self.jobs]
            ext = external_chain_pids(self.children)
        up, ms, _ = self.comfy.check()
        return {"server_ts": NOW(), "comfy": {"up": up, "ms": ms},
                "busy_external": ext, "blocking": self.blocking_busy(),
                "current": cur, "queue": q}

    def jobs_list(self):
        with self.lock:
            lst = [self.info(j) for j in self.jobs.values()]
        lst.sort(key=lambda x: x.get("created") or "", reverse=True)
        return lst


def shlex_join(argv):
    try:
        import shlex
        return " ".join(shlex.quote(x) for x in argv)
    except Exception:
        return " ".join(str(x) for x in argv)


# --------------------------------------------------------------------------
# config resolution (form -> validated job config)
# --------------------------------------------------------------------------
def _to_int(v, default, lo=None, hi=None):
    try:
        n = int(v)
    except (TypeError, ValueError):
        n = default
    if lo is not None:
        n = max(n, lo)
    if hi is not None:
        n = min(n, hi)
    return n


def _to_float(v, default, lo=None, hi=None):
    try:
        n = float(v)
    except (TypeError, ValueError):
        n = default
    if lo is not None:
        n = max(n, lo)
    if hi is not None:
        n = min(n, hi)
    return n


def _to_bool(v, default=False):
    if v is None:
        return default
    return str(v).strip().lower() in ("on", "true", "1", "yes")


def _first(fields, key, default=None):
    vals = fields.get(key) or []
    return vals[0] if vals else default


_BEAT_LINE_RE = re.compile(r"^([\d.]+)\s*(?:s|秒)?\s*[:：]\s*(.+)$")


def normalize_beats(text):
    """One beat per line: '<sec>[s|秒][:：]<description>'.

    Newline is the only separator (descriptions may freely contain ; , : full
    or half width - no re-splicing on ; anywhere). Returns canonical multiline
    text '<sec>s:<desc>' (one per line) or None. Any unparseable line aborts
    the whole submission (no silent drops) with the offending line number.
    """
    if not text:
        return None
    out = []
    for i, ln in enumerate(text.splitlines(), start=1):
        s = ln.strip()
        if not s:
            continue
        m = _BEAT_LINE_RE.match(s)
        if not m:
            raise ValueError("beats 第 %d 行无法解析: %s; 格式 <秒>[s|秒]冒号描述, "
                             "每行一条, 例如 10s:爆炸" % (i, s[:60]))
        t = float(m.group(1))
        desc = m.group(2).strip()
        if not desc:
            raise ValueError("beats 第 %d 行描述为空: %s" % (i, s[:60]))
        out.append("%gs:%s" % (t, desc))
    return "\n".join(out) if out else None


def beats_to_rows(beats):
    """Canonical multiline beats text -> list of individual '<sec>s:desc' rows."""
    if not beats:
        return []
    return [ln.strip() for ln in beats.splitlines() if ln.strip()]


ALL_KINDS = ("first_image", "last_image", "ref_image", "ref_video", "ref_audio")
ALLOW_KINDS = {"text": (), "i2v": ("first_image", "last_image"),
               "ref": ("ref_image", "ref_video", "ref_audio")}
_KIND_LIMIT = {"first_image": 1, "last_image": 1, "ref_image": 9,
               "ref_video": 3, "ref_audio": 3}


def parse_orders(fields):
    """{kind: [tokens]} where each token is 'new:<seq>' or 'base:<idx>'.
    'new:<seq>' refers to the <seq>-th uploaded file of that kind (body
    order, 0-based); 'base:<idx>' to the base job's staged file of that kind."""
    orders = {}
    for kind in ALL_KINDS:
        vals = fields.get("order_" + kind) or []
        toks = []
        for v in vals:
            for ln in str(v).splitlines():
                s = ln.strip()
                if s:
                    toks.append(s)
        if toks:
            orders[kind] = toks
    return orders


def build_cfg(fields, files, manager, form_tag_hint=None, inherit_cfg=None):
    """Turn parsed form into a validated job config dict (or raise ValueError).

    inherit_cfg (a finished base job's config) marks a "duplicate" submission:
    engine is inherited and locked, media is resolved per-field by the
    order_<kind> tokens against the inherited staged files plus any new
    uploads (see resolve_staged). The form still fully controls prompt/beats/
    segments/dur/... so a duplicated job is a brand-new chain.
    """
    seg = _to_int(_first(fields, "segments"), 1, lo=1, hi=200)
    dur = _to_float(_first(fields, "dur"), 5.0, lo=1.0, hi=60.0)
    w = _to_int(_first(fields, "width"), 864, lo=256, hi=2048)
    h = _to_int(_first(fields, "height"), 480, lo=256, hi=2048)
    steps = _to_int(_first(fields, "steps"), 8, lo=4, hi=50)
    seed_raw = _first(fields, "seed", "").strip()
    seed = None
    if seed_raw:
        try:
            seed = int(seed_raw)
        except ValueError:
            raise ValueError("seed 必须为整数")
    prompt = _first(fields, "prompt", "").strip()
    if not prompt:
        raise ValueError("prompt 不能为空")
    tag = _first(fields, "tag", "").strip() or form_tag_hint or time.strftime("w_%Y%m%d_%H%M%S")
    if not re.match(r"^[\w\-]{1,60}$", tag):
        raise ValueError("tag 只允许字母/数字/_/- (<=60)")
    beats = normalize_beats(_first(fields, "beats", ""))
    if beats:
        _rows = beats.splitlines()
        if len(_rows) > seg:
            beats = "\n".join(_rows[:seg])
    merge = _to_bool(_first(fields, "merge"))
    clear = _first(fields, "clear", "restart")
    if clear not in ("restart", "prewarm", "none"):
        clear = "restart"
    clean_slots = _to_bool(_first(fields, "clean_slots"), True)
    rsz = _first(fields, "ref_image_size", "match")
    if rsz not in ("match", "max"):
        rsz = "match"

    # ---- files -> staged upload dirs ----
    staged = {k: [] for k in ALL_KINDS}
    counts = {}
    for f in files:
        kind = f["name"]
        if kind not in ALL_KINDS:
            continue
        counts[kind] = counts.get(kind, 0) + 1
        staged[kind].append(f)
    declared = (_first(fields, "engine", "") or "").strip()

    def _listk(ks):
        return "、".join(sorted(ks)) or "无"

    if inherit_cfg is not None:
        engine = inherit_cfg["engine"]
        if declared and declared != engine:
            raise ValueError("复制任务沿用原引擎(%s)，不能切换; 换引擎请新建任务" % engine)
        bad = [k for k in counts if k not in ALLOW_KINDS[engine]]
        if bad:
            raise ValueError("%s 引擎不允许上传 %s" % (engine, _listk(bad)))
        clean_slots = True        # a duplicated chain always starts from scratch
    elif declared == "text":
        if counts:
            raise ValueError("纯文本引擎却收到文件(%s); 若为残留选择请硬刷新 Ctrl+F5 后重试"
                             % _listk(set(counts)))
        engine = "text"
    elif declared == "i2v":
        if set(counts) & set(ALLOW_KINDS["ref"]):
            raise ValueError("首末帧图(i2v)与 ref 素材冲突: 收到 %s"
                             % _listk(set(counts) & set(ALLOW_KINDS["ref"])))
        engine = "i2v"
    elif declared == "ref":
        if set(counts) & {"first_image", "last_image"}:
            raise ValueError("参考素材(ref)与首末帧图冲突: 收到 %s"
                             % _listk(set(counts) & {"first_image", "last_image"}))
        if not (set(counts) & set(ALLOW_KINDS["ref"])):
            raise ValueError("ref 引擎需至少上传一个参考素材(图/视频/音频)")
        engine = "ref"
    else:
        # legacy clients / direct API without explicit engine: auto-detect
        fl2 = set(counts) & {"first_image", "last_image"}
        rfs = set(counts) & set(ALLOW_KINDS["ref"])
        if not counts:
            engine = "text"
        elif fl2 and not rfs:
            engine = "i2v"
        elif rfs and not fl2:
            engine = "ref"
        else:
            raise ValueError("fl2va 图集(first/last-image)与 ref 素材不能混用: 收到 %s; "
                             "若未主动上传文件请硬刷新 Ctrl+F5 重试" % _listk(set(counts)))
    for kind, mx in _KIND_LIMIT.items():
        if counts.get(kind, 0) > mx:
            raise ValueError("%s 最多 %d 个" % (kind, mx))

    return {"tag": tag, "engine": engine, "params": {
                "prompt": prompt, "beats": beats, "segments": seg, "dur": dur,
                "width": w, "height": h, "steps": steps, "seed": seed,
                "merge": merge, "clear": clear, "ref_image_size": rsz},
            "clean_slots": clean_slots, "continue_of": None, "duplicate_of": None,
            "staged_files": staged, "base_slots": {},
            "orders": parse_orders(fields) if inherit_cfg is not None else None}


def stage_uploaded_files(files, job_dir):
    """Persist file parts to disk; return {field: [abs paths]} in body order."""
    out = {}
    up_dir = os.path.join(job_dir, "uploads")
    os.makedirs(up_dir, exist_ok=True)
    seq = 0
    for f in files:
        if f["name"] not in ("first_image", "last_image", "ref_image", "ref_video", "ref_audio"):
            continue
        seq += 1
        name = "%02d_%s" % (seq, _safe_name(f["filename"]))
        dst = os.path.join(up_dir, name)
        with open(dst, "wb") as fh:
            fh.write(f["content"])
        out.setdefault(f["name"], []).append(dst)
    return out


def resolve_staged(engine, orders, uploaded, base_staged):
    """Merge uploaded (new, abs paths by kind) with the inherited base job's
    staged files according to the order_<kind> tokens and validate the result.

    Returns {kind: [abs paths]} in display order. A field's order must list
    exactly the items the UI wants: 'base:<idx>' keeps the base file,
    'new:<seq>' takes the <seq>-th upload, and a lone '-' means the field is
    intentionally EMPTY (the UI sends it when every item was deleted). Fields
    without an order fall back to base-then-uploaded (legacy / JSON clients).
    """
    final = {}
    for kind in ALL_KINDS:
        up = uploaded.get(kind) or []
        base = (base_staged or {}).get(kind) or []
        toks = (orders or {}).get(kind)
        items = []
        if toks:
            for t in toks:
                if t == "-":
                    continue
                if t.startswith("base:"):
                    items.append(base[int(t.split(":", 1)[1])])
                elif t.startswith("new:"):
                    items.append(up[int(t.split(":", 1)[1])])
                else:
                    raise ValueError("order_%s 未知 token: %s" % (kind, t))
        else:
            items = list(base) + list(up)
        final[kind] = items
    _validate_media(engine, final)
    return final


def _validate_media(engine, final):
    allowed = ALLOW_KINDS[engine]
    for kind, mx in _KIND_LIMIT.items():
        n = len(final.get(kind) or [])
        if n > mx:
            raise ValueError("%s 最多 %d 个" % (kind, mx))
        if kind not in allowed and n:
            raise ValueError("%s 引擎不允许携带 %s" % (engine, kind))
    if engine == "ref" and not any(final.get(k) for k in ALLOW_KINDS["ref"]):
        raise ValueError("ref 引擎需至少保留一个参考素材(图/视频/音频)")


# --------------------------------------------------------------------------
# HTTP handler
# --------------------------------------------------------------------------
_OUTPUT_RE = re.compile(r"^/files/(.+)$")


def make_handler(mgr):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format, *args):        # silence noise
            pass

        def _send_json(self, code, obj):
            body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _err(self, code, msg):
            self._send_json(code, {"error": msg})

        def _send_file_range(self, path, force_attachment=False):
            try:
                size = os.path.getsize(path)
            except OSError:
                self._err(404, "file not found")
                return
            rng = self.headers.get("Range")
            start, end = 0, size - 1
            if rng and rng.startswith("bytes="):
                spec = rng[6:].strip()
                if spec and spec[0] == "-":          # suffix range
                    start = max(0, size - int(spec[1:]))
                elif "-" in spec:
                    a, b = spec.split("-", 1)
                    start = int(a or 0)
                    end = int(b) if b else size - 1
                end = min(end, size - 1)
                if start > end or start >= size:
                    self.send_response(416)
                    self.send_header("Content-Range", "bytes */%d" % size)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
            ctype = mimetypes.guess_type(path)[0] or "application/octet-stream"
            length = end - start + 1
            if rng:
                self.send_response(206)
                self.send_header("Content-Range", "bytes %d-%d/%d" % (start, end, size))
            else:
                self.send_response(200)
            self.send_header("Content-Type", ctype)
            if force_attachment:
                name = os.path.basename(path)
                try:
                    name.encode("ascii")
                    ascii_name = name
                except UnicodeEncodeError:
                    ascii_name = "download" + os.path.splitext(name)[1] or ".bin"
                self.send_header("Content-Disposition",
                                 "attachment; filename=\"%s\"; filename*=UTF-8''%s"
                                 % (ascii_name, quote(name, safe="")))
            self.send_header("Content-Length", str(length))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            if self.command == "HEAD":
                return
            with open(path, "rb") as f:
                f.seek(start)
                remain = length
                while remain > 0:
                    chunk = f.read(min(256 * 1024, remain))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remain -= len(chunk)

        def _serve_output(self, rel, force_dl=False):
            rel = unquote(rel)
            base = os.path.realpath(os.path.join(mgr.root, "output"))
            cand = os.path.realpath(os.path.join(base, rel))
            if cand != base and not cand.startswith(base + os.sep):
                self._err(403, "path outside output")
                return
            if not os.path.isfile(cand):
                self._err(404, "file not found")
                return
            self._send_file_range(cand, force_attachment=force_dl)

        # ---- GET ----
        def do_GET(self):
            u = urlparse(self.path)
            if u.path == "/":
                self.send_response(200)
                body = HTML.encode("utf-8")
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
            elif u.path == "/api/state":
                self._send_json(200, mgr.state())
            elif u.path == "/api/jobs":
                self._send_json(200, {"jobs": mgr.jobs_list()})
            elif u.path.startswith("/api/jobs/"):
                rest = u.path[len("/api/jobs/"):]
                if "/" in rest:
                    jid, sub = rest.split("/", 1)
                    if sub.startswith("log") and jid in mgr.jobs:
                        q = parse_qs(u.query)
                        offset = int(q.get("offset", ["0"])[0] or 0)
                        job = mgr.jobs[jid]
                        try:
                            with open(job["log"], "rb") as f:
                                f.seek(offset)
                                raw = f.read()
                                new_off = f.tell()
                            data = raw.decode("utf-8", "replace")
                        except OSError:
                            data, new_off = "", offset
                        with mgr.lock:
                            if job["st"].get("stage") is None:
                                job["st"]["stage"] = {"label": "排队中...", "ts": NOW()}
                            st = job["st"].get("stage", {})
                            info = mgr.info(job)
                        self._send_json(200, {"id": jid, "status": job["st"].get("status"),
                                              "offset": new_off,
                                              "text": data, "stage": st, "info": info})
                    elif sub == "config" and jid in mgr.jobs:
                        job = mgr.jobs[jid]
                        cfg = json.loads(json.dumps(job["cfg"]))
                        media = {}
                        for k, lst in (cfg.get("staged_files") or {}).items():
                            media[k] = [os.path.basename(x) for x in lst]
                        cfg["media"] = media
                        cfg["slots_count"] = len(mgr._snapshot_slots())
                        self._send_json(200, cfg)
                    else:
                        self._err(404, "not found")
                else:
                    self._err(404, "not found")
            elif u.path == "/api/outputs":
                self._send_json(200, outputs_listing(mgr))
            else:
                m = _OUTPUT_RE.match(u.path)
                if m:
                    self._serve_output(m.group(1),
                                       force_dl=parse_qs(u.query).get("dl", ["0"])[0] == "1")
                else:
                    self._err(404, "not found")

        def do_HEAD(self):
            u = urlparse(self.path)
            m = _OUTPUT_RE.match(u.path)
            if m:
                self._serve_output(m.group(1))
            else:
                self._err(404, "not found")

        # ---- POST ----
        def do_POST(self):
            u = urlparse(self.path)
            # job actions carry no body
            m_action = re.match(r"^/api/jobs/([^/]+)/(cancel|delete)$", u.path)
            if m_action:
                jid, action = m_action.group(1), m_action.group(2)
                ok, msg = (mgr.cancel(jid) if action == "cancel" else mgr.delete(jid))
                self._send_json(200 if ok else 409, {"ok": ok, "msg": msg})
                return
            if u.path == "/api/output/delete":
                try:
                    length = int(self.headers.get("Content-Length") or 0)
                    body = self.rfile.read(length) if length > 0 else b"{}"
                    js = json.loads(body.decode("utf-8")) if body.strip() else {}
                except Exception:
                    self._err(400, "bad json"); return
                rel = (js.get("rel") or "").strip()
                if not rel:
                    self._err(400, "missing rel"); return
                base = os.path.realpath(os.path.join(mgr.root, "output"))
                cand = os.path.realpath(os.path.join(base, rel))
                if cand != base and not cand.startswith(base + os.sep):
                    self._err(403, "path outside output"); return
                if os.path.isfile(cand):
                    os.remove(cand)
                    self._send_json(200, {"ok": True, "msg": "已删除 " + rel})
                elif os.path.isdir(cand):
                    self._err(400, "不允许删除目录")
                else:
                    self._err(404, "file not found")
                return
            if u.path != "/api/run":
                self._err(404, "not found")
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
                if length <= 0:
                    self._err(400, "empty body"); return
                if length > UPLOAD_MAX:
                    self._err(413, "body too large (>2 GiB)"); return
                body = self.rfile.read(length)
            except Exception as e:
                self._err(400, "read body failed: %r" % e); return
            ct = self.headers.get("Content-Type", "")
            if ct.startswith("multipart/form-data"):
                m = re.search(r'boundary="?([^";]+)"?', ct)
                if not m:
                    self._err(400, "missing boundary"); return
                try:
                    fields, files = parse_multipart(body, m.group(1).encode("ascii"))
                except ValueError as e:
                    self._err(400, str(e)); return
            else:
                fields, files = {}, []
                try:
                    js = json.loads(body.decode("utf-8"))
                    if isinstance(js, dict):
                        fields = {k: [str(v)] for k, v in js.items()}
                except Exception:
                    self._err(400, "expected multipart/form-data"); return

            self._api_run(fields, files)

        def _api_run(self, fields, files):
            continue_of = (_first(fields, "continue_of", "") or "").strip()
            duplicate_of = (_first(fields, "duplicate_of", "") or "").strip()
            if continue_of and duplicate_of:
                self._err(400, "续拍与复制不能同时使用"); return
            with mgr.lock:
                # ---- continue: pure segment resume, every param frozen ----
                if continue_of:
                    base = mgr.jobs.get(continue_of)
                    if not base or not base["cfg"]:
                        self._err(404, "基准任务 %s 不存在或配置缺失" % continue_of); return
                    if base["st"].get("status") not in ("done", "failed", "cancelled", "interrupted"):
                        self._err(409, "基准任务尚未结束, 不能续拍"); return
                    if files:
                        self._err(400, "续拍沿用原素材, 请不要上传文件"); return
                    base_cfg = base["cfg"]
                    seg = _to_int(_first(fields, "segments"), 0, lo=1, hi=200)
                    if seg <= 0:
                        self._err(400, "请指定目标段数 segments"); return
                    base_slots = base["st"].get("slots_snapshot") or {}
                    if not base_slots:
                        base_slots = {os.path.relpath(p, mgr.root): os.stat(p).st_mtime
                                      for p in sorted(glob.glob(os.path.join(
                                          mgr.root, "output/h3_continuous/chain_*.safetensors")))}
                    n_slots = len(base_slots)
                    if seg <= n_slots:
                        self._err(400, "磁盘已有 %d 段槽位; 续拍 segments 需 > %d" % (n_slots, n_slots)); return
                    # clone the base config verbatim - the form parameters are
                    # ignored so a resume can never accidentally change anything
                    cfg = json.loads(json.dumps(base_cfg))
                    cfg["params"]["segments"] = seg
                    cfg["clean_slots"] = False
                    cfg["continue_of"] = continue_of
                    cfg["duplicate_of"] = None
                    cfg["base_slots"] = base_slots
                    cfg["tag"] = base_cfg["tag"]
                    cfg["engine"] = base_cfg["engine"]
                    cfg["staged_files"] = base_cfg.get("staged_files") or {}
                    cfg.pop("orders", None)
                    jid = mgr.new_id()
                    mgr.submit(cfg, jid=jid)
                    self._send_json(202, {"id": jid, "status": "queued", "tag": cfg["tag"]})
                    return
                # ---- duplicate: copy params into a brand-new chain ----
                inherit_cfg = None
                if duplicate_of:
                    base = mgr.jobs.get(duplicate_of)
                    if not base or not base["cfg"]:
                        self._err(404, "基准任务 %s 不存在或配置缺失" % duplicate_of); return
                    if base["st"].get("status") in ("queued", "running"):
                        self._err(409, "基准任务正在运行, 请结束后再复制"); return
                    inherit_cfg = base["cfg"]
                try:
                    cfg = build_cfg(fields, files, mgr, inherit_cfg=inherit_cfg)
                except ValueError as e:
                    self._err(400, str(e)); return
                jid = mgr.new_id()
                if inherit_cfg is not None:
                    cfg["duplicate_of"] = duplicate_of
                    jdir = mgr._job_dir(jid)
                    os.makedirs(jdir, exist_ok=True)
                    uploaded = stage_uploaded_files(files, jdir) if files else {}
                    try:
                        # merge new uploads with the base job's kept media per order
                        cfg["staged_files"] = resolve_staged(
                            cfg["engine"], cfg.get("orders"), uploaded,
                            inherit_cfg.get("staged_files") or {})
                    except (ValueError, IndexError) as e:
                        self._err(400, str(e)); return
                    cfg.pop("orders", None)
                else:
                    # fresh job: stage uploads into a pre-allocated job dir so
                    # the worker never sees files mid-move
                    if files:
                        jdir = mgr._job_dir(jid)
                        os.makedirs(jdir, exist_ok=True)
                        cfg["staged_files"] = stage_uploaded_files(files, jdir)
                mgr.submit(cfg, jid=jid)
                self._send_json(202, {"id": jid, "status": "queued", "tag": cfg["tag"]})

    return H


def outputs_listing(mgr):
    out_root = os.path.join(mgr.root, "output")
    finals = []
    for p in sorted(glob.glob(os.path.join(out_root, "final_*.mp4"))):
        rel = os.path.relpath(p, out_root)
        finals.append({"rel": rel.replace("\\", "/"), "name": os.path.basename(p),
                       "size": os.path.getsize(p), "ts": time.strftime(
                           "%m-%d %H:%M", time.localtime(os.path.getmtime(p))),
                       "tag": os.path.basename(p)[len("final_"):-4]})
    chains = []
    cdir = os.path.join(out_root, "video/chain")
    for tag in sorted(os.listdir(cdir)) if os.path.isdir(cdir) else []:
        segs = []
        tdir = os.path.join(cdir, tag)
        for p in sorted(glob.glob(os.path.join(tdir, "seg_*.mp4"))):
            segs.append({"rel": os.path.relpath(p, out_root).replace("\\", "/"),
                         "name": os.path.basename(p), "size": os.path.getsize(p)})
        chains.append({"tag": tag, "segs": segs[-40:]})
    chains.sort(key=lambda x: x["tag"])
    return {"finals": finals, "chains": chains}


# --------------------------------------------------------------------------
# HTML dashboard (single page, no CDN)
# --------------------------------------------------------------------------
HTML = r"""<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ChainDirector V2 · Web</title>
<style>
:root{--bg:#0f1115;--panel:#171b22;--line:#2a3140;--fg:#e6e9f0;--dim:#8b93a3;
--acc:#4da3ff;--ok:#35c96a;--warn:#f0b23c;--bad:#f05a5a;--mono:ui-monospace,Consolas,monospace}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
a{color:var(--acc);text-decoration:none}
header{display:flex;gap:14px;align-items:center;flex-wrap:wrap;padding:10px 16px;
background:var(--panel);border-bottom:1px solid var(--line);position:sticky;top:0;z-index:5}
header h1{font-size:16px;margin:0}
.pill{padding:2px 10px;border-radius:12px;font-size:12px;border:1px solid var(--line)}
.pill.ok{color:var(--ok);border-color:var(--ok)}.pill.bad{color:var(--bad);border-color:var(--bad)}
.pill.run{color:var(--acc);border-color:var(--acc)}.pill.que{color:var(--warn);border-color:var(--warn)}
main{display:grid;grid-template-columns:minmax(340px,460px) 1fr;gap:14px;padding:14px;align-items:start}
@media(max-width:980px){main{grid-template-columns:1fr}}
section{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px}
section h2{font-size:13px;margin:0 0 10px;color:var(--dim);text-transform:uppercase;letter-spacing:.5px}
label{display:block;font-size:12px;color:var(--dim);margin:8px 0 3px}
input[type=text],input[type=number],textarea,select{width:100%;background:#0d1014;border:1px solid var(--line);
color:var(--fg);border-radius:6px;padding:6px 8px;font-size:13px}
textarea{font-family:var(--mono);resize:vertical}
button{cursor:pointer;border:1px solid var(--acc);background:transparent;color:var(--acc);
border-radius:6px;padding:6px 12px;font-size:13px}
button:hover{background:rgba(77,163,255,.12)}
button.primary{background:var(--acc);color:#081018}
button.danger{border-color:var(--bad);color:var(--bad)}
button.danger:hover{background:rgba(240,90,90,.15)}
.row{display:flex;gap:8px}.row>*{flex:1}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:0 10px}
.chk{display:flex;gap:6px;align-items:center;margin:8px 0}
.chk input{accent-color:var(--acc)}
details{margin-top:8px;border-top:1px solid var(--line);padding-top:6px}
summary{cursor:pointer;font-size:12px;color:var(--dim)}
.tabs{display:flex;gap:4px;margin:4px 0 10px}
.tabs button{flex:1;font-size:12px;border-color:var(--line);color:var(--dim)}
.tabs button.on{border-color:var(--acc);color:var(--acc)}
table{width:100%;border-collapse:collapse;font-size:12.5px}
th,td{text-align:left;padding:5px 6px;border-bottom:1px solid var(--line);vertical-align:top}
th{color:var(--dim);font-weight:500}
.mono{font-family:var(--mono)}
.log{background:#0a0c0f;border:1px solid var(--line);border-radius:6px;height:300px;overflow:auto;
font:12px/1.45 var(--mono);padding:8px;white-space:pre-wrap;word-break:break-all}
.card{border:1px solid var(--line);border-radius:8px;padding:8px;margin-bottom:8px}
.mlist{list-style:none;margin:4px 0 0;padding:0}
.mlist li{display:flex;align-items:center;gap:8px;padding:4px 7px;border:1px solid var(--line);
border-radius:6px;margin:3px 0;font-size:12px;background:#0d1014}
.mlist li[draggable=true]{cursor:grab}
.mlist li.dragging{opacity:.45;border-style:dashed}
.mlist .gh{color:var(--dim);cursor:grab;user-select:none}
.mlist .k{color:var(--acc);font-family:var(--mono);min-width:64px}
.mlist .src{color:var(--dim)}
.mlist .del{color:var(--bad);cursor:pointer;margin-left:auto;user-select:none}
.madd{font-size:12px;padding:1px 8px;margin:4px 0 0}
.mtip{font-size:11px;color:var(--dim);margin:2px 0 0}
.badge{font-size:11px;padding:1px 7px;border-radius:10px;border:1px solid var(--line);white-space:nowrap}
.badge.done{color:var(--ok);border-color:var(--ok)}
.badge.failed,.badge.cancelled,.badge.interrupted{color:var(--bad);border-color:var(--bad)}
.badge.running{color:var(--acc);border-color:var(--acc)}
.badge.queued{color:var(--warn);border-color:var(--warn)}
progress{width:100%;height:8px;accent-color:var(--acc)}
#modal{position:fixed;inset:0;background:rgba(0,0,0,.7);display:none;align-items:center;justify-content:center;z-index:50}
#modal .box{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px;max-width:920px;width:94%}
#modal video{width:100%;max-height:70vh;background:#000;border-radius:6px}
#logmodal{position:fixed;inset:0;background:rgba(0,0,0,.7);display:none;align-items:center;justify-content:center;z-index:50}
#logmodal .box{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px;max-width:980px;width:94%}
.hint{font-size:11px;color:var(--dim);margin-top:3px}
footer{padding:8px 16px;color:var(--dim);font-size:11px}
.statusline{display:flex;flex-wrap:wrap;gap:6px;align-items:center}
.muted{color:var(--dim)}
.linkdel{color:var(--bad);cursor:pointer}
.tblwrap{overflow-x:auto;-webkit-overflow-scrolling:touch}
.otblwrap{overflow-x:auto;-webkit-overflow-scrolling:touch}
@media(max-width:640px){
  main{padding:8px;gap:8px}
  section{padding:9px}
  .grid2{grid-template-columns:1fr}
  .tabs button{white-space:normal;line-height:1.25;padding:6px 4px}
  .row{flex-wrap:wrap}
  button{padding:6px 8px}
  input[type=text],input[type=number],textarea,select{font-size:16px}
  .tblwrap th:nth-child(4),.tblwrap td:nth-child(4){display:none}
  .tblwrap th:nth-child(5),.tblwrap td:nth-child(5){display:none}
  td{overflow-wrap:anywhere;word-break:break-word}
  .card a{word-break:break-all}
  .mlist li{flex-wrap:wrap}
  #modal .box{width:98%;padding:8px}
}
</style>
</head>
<body>
<header>
  <h1>ChainDirector V2 · Web</h1>
  <span class="statusline">
    <span class="pill" id="comfy">ComfyUI …</span>
    <span class="pill" id="busy">GPU: 检查中</span>
    <span class="pill" id="now"></span>
  </span>
</header>
<main>
  <!-- left column -->
  <div>
    <section>
      <h2>运行面板</h2>
      <div id="curpan">空闲 — 没有正在运行的任务</div>
    </section>

    <section style="margin-top:12px">
      <h2>新建任务</h2>
      <form id="jobform" onsubmit="return submitJob(event)">
        <input type="hidden" id="f_continue_of" name="continue_of">
        <input type="hidden" id="f_duplicate_of" name="duplicate_of">
        <label>提示词 prompt</label>
        <textarea id="f_prompt" name="prompt" rows="4"
          placeholder="全局设定/人物/风格/机位一句话（作用于全片）。续拍时 prompt 决定后续段风格。"></textarea>

        <div class="tabs" id="enginetabs">
          <button type="button" class="on" data-e="text" onclick="setEngine('text')">纯文本 fl2va</button>
          <button type="button" data-e="i2v" onclick="setEngine('i2v')">首末帧图 fl2va</button>
          <button type="button" data-e="ref" onclick="setEngine('ref')">参考素材 ref2va</button>
        </div>

        <div id="g_text" class="hint">无图锚，H3ContinuousStartV14 纯文生。</div>

        <div id="g_i2v" style="display:none">
          <div id="mb_first_image"></div>
          <div id="mb_last_image"></div>
        </div>

        <div id="g_ref" style="display:none">
          <div id="mb_ref_image"></div>
          <div id="mb_ref_video"></div>
          <div id="mb_ref_audio"></div>
          <label>ref-image-size</label>
          <select name="ref_image_size"><option value="match" selected>match</option><option value="max">max</option></select>
          <div class="hint">weak-audio 场景建议 --steps 20；带素材/强事件用 8 即可（见 audio.md）。</div>
        </div>

        <label>分段节拍 beats（每行一条，时间秒，作用于所属段；描述内可含任意标点）</label>
        <textarea id="f_beats" name="beats" rows="3" placeholder="10s:爆炸&#10;25.5s:传送门浮现，暴雨骤停"></textarea>
        <div class="hint">格式 <b>&lt;秒&gt;[s|秒]冒号描述</b>，例如 <code>10s:爆炸</code>、<code>5秒:鹰俯冲</code>、
          <code>0:开场镜头</code>。每行一条，换行即分隔，一行不合法会整体拒绝。</div>

        <div class="grid2">
          <div><label>segments 段数</label><input type="number" name="segments" value="1" min="1"></div>
          <div><label>每段净秒 dur</label><input type="number" name="dur" value="5.0" step="0.5" min="1"></div>
          <div><label>width</label><input type="number" name="width" value="864" min="256" step="8"></div>
          <div><label>height</label><input type="number" name="height" value="480" min="256" step="8"></div>
          <div><label>steps（弱音频场景用 20）</label><input type="number" name="steps" value="8" min="4"></div>
          <div><label>seed（留空随机）</label><input type="text" name="seed"></div>
        </div>

        <label>tag（留空自动生成 w_&lt;时间戳&gt;）</label>
        <input type="text" id="f_tag" name="tag" placeholder="w_... 或自定义">

        <details>
          <summary>高级</summary>
          <div class="grid2">
            <div>
              <label>清场方式 clear</label>
              <select name="clear">
                <option value="restart" selected>restart（重启服务 ~11s）</option>
                <option value="prewarm">prewarm（原地重建 ~30s）</option>
                <option value="none">none</option>
              </select>
            </div>
            <div><label>引擎(读取用)</label><select name="_engine_read" id="f_engine_read"><option>text</option></select></div>
          </div>
          <div class="chk"><input type="checkbox" id="f_merge" name="merge" checked><label style="margin:0">跑完自动 merge 成 final_&lt;tag&gt;.mp4</label></div>
          <div class="chk"><input type="checkbox" id="f_clean" name="clean_slots" checked><label style="margin:0">全新链：跑前清空 h3_continuous 槽位 + 本 tag 旧片段</label></div>
        </details>

        <div style="margin-top:10px;display:flex;gap:8px;align-items:center">
          <button class="primary" type="submit">提交任务</button>
          <span id="upinfo" class="muted" style="font-size:12px"></span>
        </div>
        <div id="submitmsg"></div>
      </form>
    </section>

    <section style="margin-top:12px">
      <h2>队列</h2>
      <div id="queuebox">—</div>
    </section>
  </div>

  <!-- right column -->
  <div>
    <section>
      <h2>当前任务 · 控制台</h2>
      <div id="stagewrap" class="muted" style="margin-bottom:6px"></div>
      <div class="log" id="console">(等待任务…)</div>
      <div style="margin-top:8px;display:flex;gap:8px;align-items:center">
        <button class="danger" id="btnCancel" onclick="cancelCurrent()" disabled>取消当前任务</button>
        <label class="chk" style="margin:0"><input type="checkbox" id="autoscroll" checked>自动滚动</label>
      </div>
    </section>

    <section style="margin-top:12px">
      <h2>历史任务</h2>
      <div class="tblwrap">
      <table>
        <thead><tr><th>时间</th><th>状态</th><th>tag / 引擎</th><th>段</th><th>耗时</th><th>操作</th></tr></thead>
        <tbody id="histbody"><tr><td colspan="6" class="muted">加载中…</td></tr></tbody>
      </table>
      </div>
    </section>

    <section style="margin-top:12px">
      <h2>输出库</h2>
      <div id="outbox">加载中…</div>
    </section>
  </div>
</main>

<div id="modal" onclick="if(event.target===this)closeModal()"><div class="box">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">
    <b id="modaltitle"></b><button onclick="closeModal()">关闭</button>
  </div>
  <video id="modvideo" controls preload="metadata" playsinline webkit-playsinline></video>
</div></div>

<div id="logmodal" onclick="if(event.target===this)closeLog()"><div class="box">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">
    <b id="logtitle"></b><button onclick="closeLog()">关闭</button>
  </div>
  <div class="log" id="logbody" style="height:60vh"></div>
</div></div>

<footer>内网部署 · 无鉴权 · 端口 8189 · 同时只串行跑一条链。驱动脚本：chain_director_v2.py（CLI 语义镜像）。</footer>

<script>
const $=id=>document.getElementById(id);
let ENGINE="text";
let pollTimer=null, logTimer=null, curId=null, logOff=0, hist=[];
const esc=s=>String(s??"").replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));

let engineLocked=false, mediaLocked=false;
const ALL_MEDIA_KINDS=["first_image","last_image","ref_image","ref_video","ref_audio"];
const MEDIA_META={
  first_image:{title:"首帧图 first-frame",acc:"image/*",multi:false,key:""},
  last_image:{title:"末帧图 last-frame",acc:"image/*",multi:false,key:""},
  ref_image:{title:"参考图 ref-image",acc:"image/*",multi:true,key:"img"},
  ref_video:{title:"参考视频 ref-video",acc:"video/*",multi:true,key:"video"},
  ref_audio:{title:"参考音频 ref-audio",acc:"audio/*,video/*",multi:true,key:"audio"}};
const MEDIA_LIMIT={first_image:1,last_image:1,ref_image:9,ref_video:3,ref_audio:3};
const media={};
ALL_MEDIA_KINDS.forEach(k=>{media[k]=[];});

function mediaLabel(kind){
  if(kind==="ref_image")return"参考图 ref-image（≤9，列表顺序=编号 img1..）";
  if(kind==="ref_video")return"参考视频 ref-video（≤3，24fps 重编码；序号对应其 &lt;Audio/&lt;Video&gt; k）";
  if(kind==="ref_audio")return"参考音频 ref-audio（≤3，独立引导声：脚步声/音乐…）";
  return MEDIA_META[kind].title;
}
function renderMedia(kind){
  const host=$("mb_"+kind);
  if(!host)return;
  host.innerHTML="";
  const meta=MEDIA_META[kind];
  const lab=document.createElement("div");
  lab.textContent=mediaLabel(kind);
  host.appendChild(lab);
  const btn=document.createElement("button"); btn.type="button"; btn.className="madd";
  btn.textContent="+ 添加文件";
  const inp=document.createElement("input"); inp.type="file"; inp.accept=meta.acc;
  if(meta.multi)inp.multiple=true;
  inp.style.display="none";
  btn.onclick=()=>inp.click();
  inp.onchange=()=>{addFiles(kind,[...inp.files]);inp.value="";};
  host.appendChild(btn); host.appendChild(inp);
  const ul=document.createElement("ul"); ul.className="mlist"; host.appendChild(ul);
  const items=media[kind];
  if(!items.length){
    const li=document.createElement("li"); li.style.cursor="default"; li.style.color="var(--dim)";
    li.textContent="（无）"; ul.appendChild(li);
    return;
  }
  let dragged=null;
  items.forEach((it,i)=>{
    const li=document.createElement("li");
    const gh=document.createElement("span"); gh.className="gh"; gh.textContent="⠿";
    const ks=document.createElement("span"); ks.className="k";
    ks.textContent=meta.key?meta.key+(i+1):(kind==="first_image"?"first":kind==="last_image"?"last":kind);
    const sr=document.createElement("span"); sr.className="src"; sr.textContent=it.b?"沿用原任务":"新增";
    const nm=document.createElement("span"); nm.textContent=it.name; nm.style.wordBreak="break-all";
    const del=document.createElement("span"); del.className="del"; del.textContent="✕";
    del.title="删除该项";
    del.onclick=()=>delMediaItem(kind,i);
    li.append(gh,ks,sr,nm,del);
    ul.appendChild(li);
    li.addEventListener("dragstart",e=>{
      dragged=i; e.dataTransfer.effectAllowed="move";
      setTimeout(()=>li.classList.add("dragging"),0);
    });
    li.addEventListener("dragend",()=>li.classList.remove("dragging"));
  });
  ul.addEventListener("dragover",e=>e.preventDefault());
  ul.addEventListener("drop",e=>{
    e.preventDefault();
    if(dragged==null)return;
    const over=e.target.closest("li");
    const overI=over?[...ul.children].indexOf(over):items.length;
    if(overI<0||dragged===overI){ul.querySelectorAll(".dragging").forEach(x=>x.classList.remove("dragging"));dragged=null;return;}
    const arr=media[kind];
    const it=arr.splice(dragged,1)[0];
    const ins=overI>dragged?overI-1:overI;
    arr.splice(ins,0,it);
    dragged=null;
    renderMedia(kind);
  });
}
function addFiles(kind,files){
  if(mediaLocked){alert("续拍模式沿用原素材，不能增删或排序素材");return;}
  const meta=MEDIA_META[kind], lim=MEDIA_LIMIT[kind];
  let arr=meta.multi?media[kind].slice():[];
  for(const f of files){
    if(arr.length>=lim){alert(MEDIA_META[kind].title.replace(/^./,c=>c.toUpperCase())+"最多 "+lim+" 个，多余已忽略");break;}
    arr.push({b:false,file:f,name:f.name});
  }
  media[kind]=arr; renderMedia(kind);
}
function delMediaItem(kind,i){
  if(mediaLocked){alert("续拍模式沿用原素材，不能增删或排序素材");return;}
  media[kind].splice(i,1);renderMedia(kind);
}

function clearForeignFiles(e){
  // drop any media belonging to an engine tab that is not the active one
  const allow=(e==="i2v"?["first_image","last_image"]:(e==="ref"?["ref_image","ref_video","ref_audio"]:[]));
  ALL_MEDIA_KINDS.forEach(k=>{
    if(!allow.includes(k)&&media[k].length){media[k]=[];renderMedia(k);}
  });
}

function setEngine(e){if(engineLocked&&e!==ENGINE)return;
  ENGINE=e;
  document.querySelectorAll("#enginetabs button").forEach(b=>b.classList.toggle("on",b.dataset.e===e));
  $("g_text").style.display=e==="text"?"":"none";
  $("g_i2v").style.display=e==="i2v"?"":"none";
  $("g_ref").style.display=e==="ref"?"":"none";
  clearForeignFiles(e);
}

async function jget(url){const r=await fetch(url);const j=await r.json();if(!r.ok)throw new Error(j.error||r.status);return j;}
function fmt(t){const d=t?" "+t:"";return d;}

/* ---------- submit ---------- */
const TEXT_FIELDS=["prompt","beats","tag","segments","dur","width","height","steps","seed",
                   "merge","clean_slots","clear","ref_image_size","continue_of","duplicate_of"];
const FORM_FIELDS=["prompt","beats","tag","segments","dur","width","height","steps","seed",
                   "merge","clean_slots","clear","ref_image_size"];
const FILE_ALLOW={text:[], i2v:["first_image","last_image"], ref:["ref_image","ref_video","ref_audio"]};

function lockEngine(lock){engineLocked=lock;document.querySelectorAll("#enginetabs button").forEach(b=>b.disabled=lock);}
function fillForm(cfg){
  const form=$("jobform");
  const p=cfg.params||{};
  $("f_prompt").value=p.prompt||"";
  const bv=p.beats||"";
  $("f_beats").value=bv.indexOf("\n")>=0?bv:bv.replace(/;/g,"\n");
  const map={segments:"segments",dur:"dur",width:"width",height:"height",steps:"steps",seed:"seed",clear:"clear"};
  Object.entries(map).forEach(([k,n])=>{const el=form.querySelector('[name="'+n+'"]');if(el&&p[k]!=null)el.value=p[k];});
  if(p.seed==null)form.querySelector('[name=seed]').value="";
  $("f_merge").checked=!!p.merge;
  form.querySelector('[name=ref_image_size]').value=p.ref_image_size||"match";
  setEngine(cfg.engine==="i2v"?"i2v":(cfg.engine==="ref"?"ref":"text"));
}
function lockParams(keep){
  ["prompt","beats","dur","width","height","steps","seed","clear","merge","clean_slots","ref_image_size"].forEach(n=>{
    const el=document.querySelector('[name="'+n+'"]'); if(el)el.disabled=!keep.includes(n);
  });
}
function resetForm(){
  $("jobform").reset();
  $("f_clean").checked=true;$("f_merge").checked=true;
  $("f_tag").disabled=false;
  $("f_continue_of").value="";$("f_duplicate_of").value="";
  lockParams(FORM_FIELDS);               // re-enable every parameter
  lockEngine(false);mediaLocked=false;
  ALL_MEDIA_KINDS.forEach(k=>{media[k]=[];renderMedia(k);});
  setEngine("text");
}

function buildFd(){
  // Whitelist build: text fields + (for the active engine) the ordered media
  // lists. Newly picked files are uploaded in display order; per-field
  // order_<kind> tokens tell the backend how to merge them with the base
  // job's kept files when this is a duplicate submission.
  const fd=new FormData();
  const els=$("jobform").elements;
  TEXT_FIELDS.forEach(n=>{
    const el=els.namedItem(n);
    if(!el)return;
    if(el.disabled)return;                  // locked fields never go out
    if(el.type==="checkbox"){if(el.checked)fd.append(n,"on");}
    else fd.append(n,el.value);
  });
  fd.set("engine",ENGINE);
  const allow=FILE_ALLOW[ENGINE]||[];
  for(const kind of allow){
    const items=media[kind]||[];
    if(!items.length){fd.append("order_"+kind,"-");continue;}
    let k=0;
    for(const it of items){
      if(it.b){fd.append("order_"+kind,"base:"+it.j);}
      else{fd.append("order_"+kind,"new:"+(k++));fd.append(kind,it.file,it.name);}
    }
  }
  return fd;
}

async function submitJob(ev){
  ev.preventDefault();
  const fd=buildFd();
  $("upinfo").textContent="上传中…";
  const xhr=new XMLHttpRequest();
  const done=new Promise(res=>{xhr.onloadend=res;});
  xhr.open("POST","/api/run");
  xhr.upload.onprogress=e=>{if(e.lengthComputable)$("upinfo").textContent="上传 "+Math.round(e.loaded/e.length*100)+"%";};
  xhr.onload=()=>{
    let j=null;try{j=JSON.parse(xhr.responseText);}catch(_){}
    $("upinfo").textContent="";
    if(xhr.status>=200&&xhr.status<300){
      $("submitmsg").innerHTML='<span class="badge running">已入队 '+esc(j.id)+'</span>';
      resetForm();
      refreshAll();
    }else{
      $("submitmsg").innerHTML='<span class="badge failed">'+esc(j?.error||xhr.statusText)+'</span>';
    }
  };
  xhr.onerror=()=>{$("upinfo").textContent="请求失败";};
  xhr.send(fd);
  await done;
  return false;
}

/* ---------- polling ---------- */
async function refreshState(){
  let st;
  try{st=await jget("/api/state");}catch(_){return;}
  const cu=st.comfy;
  $("comfy").textContent="ComfyUI: "+(cu.up?"在线":"离线");
  $("comfy").className="pill "+(cu.up?"ok":"bad");
  $("busy").textContent = st.blocking && st.blocking.length ? "GPU 忙: " + st.blocking.join("；")
    : (st.busy_external && st.busy_external.length ? "GPU: 空闲 (旧链 worker 常驻, 下任务清场)"
    : "GPU: 空闲");
  $("busy").className = "pill " + ((st.blocking && st.blocking.length) ? "que" : "ok");
  $("now").textContent=st.server_ts;

  // queue
  const q=st.queue.filter(x=>x.status!=="cancelled");
  $("queuebox").innerHTML=q.length?q.map(x=>'<div class="card" style="display:flex;justify-content:space-between;gap:8px">'+
    '<span><span class="badge queued">排队</span> '+esc(x.tag)+' · '+esc(x.engine)+' · '+esc(x.target_segments)+' 段</span>'+
    '<button class="danger" style="padding:2px 8px" onclick="delJob(\''+x.id+'\')">移除</button></div>').join("")
    :"—";

  // current panel
  const c=st.current;
  if(c && c.id!==curId){curId=c.id;logOff=0;$("console").textContent="";}
  if(c){
    const stg=c.stage||{};
    $("curpan").innerHTML='<b>'+esc(c.tag)+'</b> · 引擎 '+esc(c.engine)+' · 目标 '+esc(c.target_segments)+' 段<br>'+
      '<span class="badge running">'+esc(c.status)+'</span> '+
      '<span class="mono">'+esc(stg.label||"")+'</span> <span class="muted">'+esc(stg.ts||"")+'</span>';
    $("btnCancel").disabled=false;
  }else{
    if(curId){curId=null;logOff=0;}
    $("curpan").textContent="空闲 — 没有正在运行的任务";
    $("btnCancel").disabled=true;
    const cur=$("console").textContent==="(等待任务…)"?"":"";
  }
}

async function pollLog(){
  if(!curId)return;
  try{
    const j=await jget("/api/jobs/"+curId+"/log?offset="+logOff);
    if(j.text){
      const box=$("console");
      box.textContent+=j.text;
      const stg=j.stage||{};
      $("stagewrap").innerHTML='<span class="badge running">'+esc(j.status)+'</span> <b>'+esc(stg.label||"")+
        '</b> <span class="muted">'+esc(stg.ts||"")+'</span>';
      if($("autoscroll").checked)box.scrollTop=box.scrollHeight;
      logOff=j.offset;
    }
    if(["done","failed","cancelled","interrupted"].includes(j.status)){
      curId=null;logOff=0;
      refreshAll();
    }
  }catch(_){}
}

async function refreshJobs(){
  let j;
  try{j=await jget("/api/jobs");}catch(_){return;}
  hist=j.jobs;
  const rows=hist.slice(0,40).map(x=>{
    const d=(x.duration!=null)?Math.round(x.duration/60*10)/10+"min":(x.ended?"":x.status);
    return '<tr><td class="muted">'+esc(x.created)+'</td>'+
      '<td><span class="badge '+esc(x.status)+'">'+esc(x.status)+'</span></td>'+
      '<td>'+esc(x.tag)+'<br><span class="muted">'+esc(x.engine)+'</span></td>'+
      '<td>'+esc(x.target_segments)+'</td><td class="muted">'+esc(d)+'</td>'+
      '<td>'+(x.final_rel?'<a href="/files/'+x.final_rel+'?dl=1" download>下载</a> ':'' )+
        (x.status==="done"?'<a href="#" onclick="openModal(this,\'/files/'+esc(x.final_rel||"")+'\',\''+esc(x.tag)+'\');return false">预览</a> ':'' )+
        '<button style="padding:1px 6px;font-size:11px" onclick="openLog(\''+x.id+'\')">日志</button> '+
        (["done","failed","cancelled","interrupted"].includes(x.status)
           ?'<button style="padding:1px 6px;font-size:11px" onclick="startContinue(\''+x.id+'\')">续拍</button> '
           +'<button style="padding:1px 6px;font-size:11px" onclick="startDuplicate(\''+x.id+'\')">复制</button> ':'' )+
        (["done","failed","cancelled","interrupted"].includes(x.status)?'<button class="danger" style="padding:1px 6px;font-size:11px" onclick="delJob(\''+x.id+'\')">删</button>':'')+
      '</td></tr>';
  }).join("");
  $("histbody").innerHTML=rows||'<tr><td colspan="6" class="muted">暂无记录</td></tr>';
}

async function refreshOutputs(){
  let j;
  try{j=await jget("/api/outputs");}catch(_){return;}
  let h='';
  if(j.finals.length){
    h+='<div style="margin-bottom:6px"><b>final 成片</b></div><div class="otblwrap"><table><tbody>';
    j.finals.forEach(f=>{
      h+='<tr><td><a href="#" onclick="openModal(this,\'/files/'+esc(f.rel)+'\',\''+esc(f.tag)+'\');return false">'+esc(f.name)+'</a></td>'+
        '<td class="muted">'+(f.size/1048576).toFixed(1)+'MB</td><td class="muted">'+esc(f.ts)+'</td>'+
        '<td><a href="#" class="linkdel" onclick="delFile(\''+encodeURIComponent(f.rel)+'\');return false">删</a></td></tr>';
    });
    h+='</tbody></table></div>';
  }
  if(j.chains.length){
    h+='<div style="margin:8px 0 4px"><b>片段库（每段 raw）</b></div>';
    j.chains.forEach(c=>{
      const boxid="seglist_"+c.tag;
      h+='<div class="card"><b>'+esc(c.tag)+'</b> <span class="muted">'+c.segs.length+' 段</span> '+
        '<button style="padding:1px 6px;font-size:11px" onclick="toggleSegs(\''+boxid+'\')">展开</button><br>'+
        '<div id="'+boxid+'" style="display:none">'+c.segs.map(s=>
          '<a href="#" onclick="openModal(this,\'/files/'+esc(s.rel)+'\',\''+esc(s.name)+'\');return false">'+esc(s.name)+'</a> '+
          '<span class="muted">'+(s.size/1048576).toFixed(1)+'MB</span> '+
          '<a href="#" class="linkdel" style="font-size:11px" onclick="delFile(\''+encodeURIComponent(s.rel)+'\');return false">删</a><br>').join("")+'</div></div>';
    });
  }
  if(!j.finals.length&&!j.chains.length)h='<span class="muted">尚无产物</span>';
  $("outbox").innerHTML=h;
}
function toggleSegs(id){const el=$(id);el.style.display=el.style.display==="none"?"block":"none";}

/* ---------- actions ---------- */
function startContinue(id){
  const job=hist.find(x=>x.id===id);
  if(!job)return;
  fetch("/api/jobs/"+id+"/config").then(r=>r.json()).then(cfg=>{
    resetForm();
    fillForm(cfg);                       // shows the frozen params for review
    const nSlots=cfg.slots_count||0;
    $("f_clean").checked=false;          // continue: keep slots
    $("f_continue_of").value=id;
    $("f_tag").value=cfg.tag;
    $("f_tag").disabled=true;
    $("f_prompt").placeholder="续拍=从缺失段继续，参数锁定不改";
    lockEngine(true);mediaLocked=true;   // continue: frozen params + frozen media
    lockParams(["segments"]);            // only the target segment count is editable
    $("submitmsg").innerHTML='<span class="badge queued">续拍模式: 磁盘已有 '+nSlots+' 段槽位 · 参数全部沿用原任务 · 仅可调目标段数(需 &gt; '+nSlots+')，将自动从第 '+(nSlots+1)+' 段补拍到目标段</span>';
    window.scrollTo({top:0,behavior:"smooth"});
  }).catch(e=>alert(e.message));
}

function startDuplicate(id){
  const job=hist.find(x=>x.id===id);
  if(!job)return;
  fetch("/api/jobs/"+id+"/config").then(r=>r.json()).then(cfg=>{
    resetForm();
    fillForm(cfg);                       // copy params for editing
    $("f_duplicate_of").value=id;
    $("f_tag").value="";
    $("f_tag").disabled=false;
    $("f_tag").placeholder="留空自动生成新 tag（复制=新链）";
    $("f_clean").checked=true;
    lockEngine(true);                    // engine inherited + locked
    lockParams(FORM_FIELDS);             // unlock all parameters (duplicate = editable new chain)
    // pre-fill base media (default keep), user may delete / reorder / re-add
    ALL_MEDIA_KINDS.forEach(kind=>{
      media[kind]=((cfg.media&&cfg.media[kind])||[]).map((nm,j)=>({b:true,j:j,name:nm}));
      renderMedia(kind);
    });
    let has=ALL_MEDIA_KINDS.some(k=>media[k].length);
    $("submitmsg").innerHTML='<span class="badge queued">复制模式: 参数可改(引擎沿用 '+esc(cfg.engine)+') · 素材默认沿用原任务(可删/拖序/重传) · tag 留空自动生成 · 将作为全新链从第 1 段跑</span>';
    window.scrollTo({top:0,behavior:"smooth"});
  }).catch(e=>alert(e.message));
}

async function openLog(id){
  const job=hist.find(x=>x.id===id);
  $("logtitle").textContent=(job?job.tag+" ":"")+"· "+id+" 日志";
  $("logbody").textContent="加载中…";
  $("logmodal").style.display="flex";
  try{
    const r=await fetch("/api/jobs/"+id+"/log?offset=0");
    const j=await r.json();
    $("logbody").textContent=j.text||"(无日志)";
    $("logbody").scrollTop=$("logbody").scrollHeight;
  }catch(e){ $("logbody").textContent="加载失败: "+(e.message||e); }
}
function closeLog(){ $("logmodal").style.display="none"; }
async function delJob(id){
  if(!confirm("确定删除该任务记录？(不删除 output 产物)"))return;
  const r=await fetch("/api/jobs/"+id+"/delete",{method:"POST"});
  const j=await r.json().catch(()=>({}));
  alert(j.msg||"done"); refreshAll();
}
async function delFile(relEnc){
  if(!confirm("确定删除该输出文件？(不可恢复)"))return;
  const r=await fetch("/api/output/delete",{method:"POST",
    headers:{"Content-Type":"application/json"},body:JSON.stringify({rel:decodeURIComponent(relEnc)})});
  const j=await r.json().catch(()=>({}));
  alert(j.msg||"done"); refreshOutputs();
}
async function cancelCurrent(){
  if(!curId)return;
  const r=await fetch("/api/jobs/"+curId+"/cancel",{method:"POST"});
  const j=await r.json().catch(()=>({}));
  alert(j.msg||"已发送"); refreshAll();
}

/* ---------- modal ---------- */
function openModal(el,rel,title){
  if(!rel||rel==="/files/")return;
  $("modaltitle").textContent=title||rel;
  $("modvideo").src=rel;
  $("modal").style.display="flex";
  $("modvideo").play().catch(()=>{});
}
function closeModal(){ $("modal").style.display="none"; $("modvideo").pause(); }

function refreshAll(){ refreshState(); refreshJobs(); refreshOutputs(); }

ALL_MEDIA_KINDS.forEach(renderMedia);

pollTimer=setInterval(refreshState,2000);
logTimer=setInterval(pollLog,1500);
refreshAll();
setInterval(refreshJobs,5000);
</script>
</body>
</html>
"""


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="LAN web console for chain_director_v2.py")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--root", default=None,
                    help="deploy root (default ~/MiniMax-H3-Deploy); data lives under .h3web/")
    ap.add_argument("--driver", default=None,
                    help="chain_director_v2.py path (default: same dir as this script)")
    ap.add_argument("--comfy-base", default=COMVFY_BASE)
    ap.add_argument("--start", action="store_true", help="background via setsid+nohup")
    ap.add_argument("--stop", action="store_true")
    ap.add_argument("--status", action="store_true")
    a = ap.parse_args()

    root = os.path.abspath(a.root or os.path.expanduser("~/MiniMax-H3-Deploy"))
    driver = os.path.abspath(a.driver or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                      "chain_director_v2.py"))
    data = os.path.join(root, ".h3web")
    os.makedirs(data, exist_ok=True)
    pidfile = os.path.join(data, "h3web.pid")
    logfile = os.path.join(root, "h3web.log")

    def read_pid():
        try:
            with open(pidfile) as f:
                return int(f.read().strip())
        except Exception:
            return None

    if a.stop:
        pid = read_pid()
        if not pid:
            print("no pid file (nothing running?)")
            sys.exit(1)
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            print("pid %d not alive" % pid)
        for _ in range(50):
            try:
                os.kill(pid, 0)
                time.sleep(0.2)
            except ProcessLookupError:
                print("stopped pid %d" % pid)
                try:
                    os.remove(pidfile)
                except OSError:
                    pass
                return
        print("pid %d still alive after 10s; check h3web.log" % pid)
        sys.exit(1)

    if a.status:
        pid = read_pid()
        print("pid:", pid if pid else "(none)")
        print("port:", a.port if not pid else DEFAULT_PORT)
        try:
            if pid:
                s = json.load(urllib.request.urlopen("http://127.0.0.1:%d/api/state" % a.port, timeout=3))
                print("comfy:", s["comfy"])
                print("current:", (s["current"] or {}).get("id"))
                print("queue:", len(s["queue"]))
        except Exception as e:
            print("state query failed:", e)
        return

    if a.start:
        if sys.platform == "win32":
            print("--start 仅支持 POSIX; 前台运行")
        else:
            env = dict(os.environ, _H3WEB_DAEMON="1")
            pid = os.fork()
            if pid > 0:
                # parent: wait for the child to write its pid, then exit
                deadline = time.time() + 10
                p = None
                while time.time() < deadline:
                    p = read_pid()
                    if p:
                        try:
                            os.kill(p, 0)
                            break
                        except ProcessLookupError:
                            p = None
                            break
                    time.sleep(0.1)
                if not p:
                    print("daemon failed to start; check %s" % logfile, file=sys.stderr)
                    sys.exit(1)
                print("started (pid %s) log: %s" % (p, logfile))
                sys.exit(0)
            os.setsid()
            with open(pidfile, "w") as f:
                f.write(str(os.getpid()))
            log = open(logfile, "ab", buffering=0)
            os.dup2(log.fileno(), 1)
            os.dup2(log.fileno(), 2)
            devnull = os.open(os.devnull, os.O_RDONLY)
            os.dup2(devnull, 0)
    else:
        # foreground: write pidfile too so --status/--stop work
        with open(pidfile, "w") as f:
            f.write(str(os.getpid()))

    if not os.path.isfile(driver):
        print("driver not found: %s (pass --driver)" % driver, file=sys.stderr)
        if not os.environ.get("_H3WEB_DAEMON"):
            sys.exit(2)
    mgr = Manager(root, driver, a.comfy_base)
    mgr.recover()
    mgr.start_worker()

    try:
        srv = ThreadingHTTPServer((a.host, a.port), make_handler(mgr))
    except OSError as e:
        print("bind %s:%d failed: %s" % (a.host, a.port, e), file=sys.stderr)
        sys.exit(1)
    srv.daemon_threads = True

    def _sigterm(signum, frame):
        mgr._stop = True
        threading.Thread(target=srv.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, _sigterm)
    signal.signal(signal.SIGINT, _sigterm)
    print("ChainDirector V2 Web on http://%s:%d/ (driver=%s)" % (a.host, a.port, driver))
    print("ComfyUI:", a.comfy_base, "| data:", data)
    try:
        srv.serve_forever(poll_interval=0.5)
    finally:
        mgr._stop = True
        try:
            os.remove(pidfile)
        except OSError:
            pass
        print("bye")


if __name__ == "__main__":
    main()
