#!/usr/bin/env python3
"""ref2v_web.py -- LAN web console for single-segment reference-to-video.

Stdlib only (http.server); it never imports numpy/av/safetensors. A job is one
prompt + reference images/videos/audios; the worker spawns ref2v_runner.py, which
drives ComfyUI (:8188, resident int4 CLIP + ref2va int8 UNet) and drops one mp4
under output/ref2v/. Jobs run serially -- the GPUs render one clip at a time.

Usage:
  ~/ComfyUI-Deploy/comfyenv/bin/python scripts/ref2v_web.py --start|--stop|--status
Default: http://0.0.0.0:8191/   data: <root>/.h3ref2v/   log: <root>/ref2v_web.log
"""
import argparse, base64, glob, json, mimetypes, os, random, re, shutil, signal
import subprocess, sys, threading, time, uuid, urllib.request, urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs, unquote, quote

DEFAULT_PORT = 8191
DEFAULT_PASSWORD = "aiakos-ref2v"
COMFY_BASE = "http://127.0.0.1:8188"
UPLOAD_MAX = 2 * 1024 * 1024 * 1024
BUSY_WAIT_MAX_S = 6 * 3600
FPS = 24
MAX_IMAGES, MAX_VIDEOS, MAX_AUDIOS = 9, 3, 3
MEDIA_KINDS = ("ref_image", "ref_video", "ref_audio")
ASPECTS = ["16:9 (Widescreen)", "9:16 (Portrait Widescreen)", "1:1 (Square)",
           "4:3 (Standard)", "3:4 (Portrait Standard)", "3:2 (Photo)",
           "2:3 (Portrait Photo)", "21:9 (Ultrawide)"]

NOW = lambda: time.strftime("%Y-%m-%d %H:%M:%S")

# ---------------------------------------------------- prompt optimizer (cloud)
LLM_CONF_PATH = os.path.expanduser("~/.config/h3ref2v/llm.conf")
LLM_SYSTEM_PROMPT = (
    "你是 MiniMax H3 参考生视频（reference-to-video）模型的提示词工程师。"
    "请在保持原意、不改语言（中文输入→中文输出）的前提下，把用户提示词改写得更具体、更适合视频生成："
    "补充主体与动作、镜头运动、景别、光线、氛围、风格和画质描述，写成一段连贯自然的话，不要分点、不要解释。"
    "必须严格保持 <Picture N> / <Video N> / <Audio N> 引用标签原样不变（含编号），"
    "不要新增不存在的引用，不要改动标签内容。只输出优化后的提示词本身。"
)


def load_llm_conf():
    conf = {
        "base": os.environ.get("REF2V_LLM_BASE", "https://api.deepseek.com/v1"),
        "model": os.environ.get("REF2V_LLM_MODEL", "deepseek-chat"),
        "key": os.environ.get("REF2V_LLM_KEY", ""),
    }
    try:
        with open(LLM_CONF_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k, v = k.strip().lower(), v.strip().strip('"').strip("'")
                if k in ("base", "base_url", "url"):
                    conf["base"] = v
                elif k == "model":
                    conf["model"] = v
                elif k in ("key", "api_key"):
                    conf["key"] = v
    except OSError:
        pass
    return conf


def llm_optimize(prompt, counts):
    conf = load_llm_conf()
    if not conf["key"]:
        return False, ("未配置云端 API key：请在 %s 写 key=...，或设置环境变量 REF2V_LLM_KEY"
                       % LLM_CONF_PATH)
    have = "、".join("%s %d" % (cn, counts.get(k, 0)) for k, cn in
                     (("ref_image", "参考图"), ("ref_video", "参考视频"), ("ref_audio", "参考音频"))
                     if counts.get(k))
    user = "可用参考素材：%s。\n\n原始提示词：\n%s" % (have or "无", prompt)
    payload = {"model": conf["model"], "temperature": 0.7, "stream": False,
               "messages": [{"role": "system", "content": LLM_SYSTEM_PROMPT},
                            {"role": "user", "content": user}]}
    req = urllib.request.Request(
        conf["base"].rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + conf["key"]},
        method="POST")
    try:
        with urllib.request.urlopen(req, timeout=90) as r:
            resp = json.loads(r.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as e:
        try:
            detail = e.read().decode("utf-8", "replace")[:300]
        except Exception:
            detail = ""
        return False, "云端 API 错误 %s %s" % (e.code, detail)
    except Exception as e:
        return False, "调用云端失败: %r" % e
    try:
        text = (resp["choices"][0]["message"]["content"] or "").strip()
    except Exception:
        return False, "云端返回格式异常: %s" % json.dumps(resp, ensure_ascii=False)[:300]
    text = re.sub(r"^```[a-zA-Z0-9]*\s*", "", text)
    text = re.sub(r"\s*```$", "", text).strip()
    return (True, text) if text else (False, "云端返回为空")


def ref2v_length(dur):
    base = max(5, int(round(dur * FPS)))
    return base + ((5 - base) % 17)


def _safe_name(name):
    name = os.path.basename(name.replace("\\", "/"))
    name = re.sub(r"[^\w.\- ]", "_", name)
    return name or "file"


def _first(fields, key, default=None):
    v = fields.get(key)
    return v[0] if isinstance(v, list) and v else default


def _to_int(s, default, lo=None, hi=None):
    try:
        v = int(float(s))
    except (TypeError, ValueError):
        return default
    if lo is not None and v < lo:
        return default
    if hi is not None and v > hi:
        return default
    return v


def _to_float(s, default, lo=None, hi=None):
    try:
        v = float(s)
    except (TypeError, ValueError):
        return default
    if lo is not None and v < lo:
        return default
    if hi is not None and v > hi:
        return default
    return v


# ---------------------------------------------------------------- multipart
def _parse_mpart_header(data):
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
    fields, files = {}, []
    delim = b"--" + boundary
    if not body.startswith(delim):
        raise ValueError("bad multipart body: missing leading boundary")
    start = len(delim)
    while True:
        if body[start:start + 2] == b"--":
            break
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
        if "filename" in hdr and hdr["filename"]:
            files.append({"name": name, "filename": hdr["filename"],
                          "content_type": hdr.get("content_type"), "content": content})
        else:
            fields.setdefault(name, []).append(content.decode("utf-8", "replace").strip())
    return fields, files


# ------------------------------------------------------------- comfy health
def _nvidia_vram():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.total,memory.used",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5).stdout
        res = {}
        for ln in out.splitlines():
            a = [x.strip() for x in ln.split(",")]
            if len(a) == 3:
                res[int(a[0])] = (int(a[1]) * 1024 * 1024, int(a[2]) * 1024 * 1024)
        return res
    except Exception:
        return {}


class ComfyHealth:
    def __init__(self, base):
        self.base = base
        self._lock = threading.Lock()
        self._cached = (False, None)
        self._vram = []
        self._ts = 0.0

    def check(self):
        with self._lock:
            if time.time() - self._ts < 3:
                return self._cached
        t0 = time.time()
        vram = []
        try:
            raw = json.loads(urllib.request.urlopen(self.base + "/system_stats", timeout=3).read())
            up, ms = True, int((time.time() - t0) * 1000)
            nv = _nvidia_vram()
            for dev in (raw.get("devices") or []):
                total = dev.get("vram_total") or 0
                if not total:
                    continue
                used = total - (dev.get("vram_free") or 0)
                nvd = nv.get(dev.get("index"))
                if nvd:
                    total, used = nvd
                vram.append({"name": (dev.get("name") or "?").split(" : ")[0],
                             "used": used, "total": total})
        except Exception:
            up, ms = False, None
        with self._lock:
            self._cached = (up, ms)
            self._vram = vram
            self._ts = time.time()
        return self._cached

    def vram(self):
        self.check()
        with self._lock:
            return self._vram

    def running_prompts(self):
        try:
            q = json.loads(urllib.request.urlopen(self.base + "/queue", timeout=3))
            return len(q.get("queue_running") or [])
        except Exception:
            return None


_STAGE_PATTERNS = [
    (r"\[resident\] reusing service", "复用常驻服务(免冷启动)"),
    (r"\[lifecycle\] starting", "启动 ComfyUI(冷启动)"),
    (r"service up after", "服务就绪"),
    (r"\[stage\] material", "预处理素材"),
    (r"\[stage\] queue", "已提交,等待采样"),
    (r"\[stage\] sampling", "采样生成中"),
    (r"\[stage\] done", "完成,正在收尾"),
    (r"NODE ERROR", "节点出错"),
    (r"submit error", "提交错误"),
]
_PROG_RE = re.compile(r"\[progress\]\s+(\d+)/(\d+)")


def stage_from_log(text):
    lines = [l for l in text.splitlines() if l.strip() and "[progress]" not in l]
    for pat, lab in reversed(_STAGE_PATTERNS):
        for l in reversed(lines):
            if re.search(pat, l):
                return lab
    return lines[-1][:80] if lines else "排队中..."


# -------------------------------------------------------------- job manager
class Manager:
    def __init__(self, root, driver, comfy_base, password=DEFAULT_PASSWORD):
        self.root = root
        self.driver = os.path.abspath(driver)
        self.password = password
        self.data = os.path.join(root, ".h3ref2v")
        self.jobs_dir = os.path.join(self.data, "jobs")
        self.out_dir = os.path.join(root, "output", "ref2v")
        self.lock = threading.RLock()
        self._stop = False
        self.jobs = {}
        self.queue = []
        self.current = None
        self.children = set()
        self.comfy = ComfyHealth(comfy_base)
        self._cv = threading.Condition(self.lock)
        os.makedirs(self.jobs_dir, exist_ok=True)
        os.makedirs(self.out_dir, exist_ok=True)

    # ---- persistence ----
    def _job_dir(self, jid):
        return os.path.join(self.jobs_dir, jid)

    def persist_cfg(self, job):
        d = self._job_dir(job["id"])
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "config.json"), "w", encoding="utf-8") as f:
            json.dump({"cfg": job["cfg"]}, f, ensure_ascii=False, indent=2)

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
            try:
                with open(os.path.join(jdir, "status.json"), encoding="utf-8") as f:
                    st = json.load(f)
            except Exception:
                st = None
            job = {"id": d, "cfg": cfg, "st": st or {"id": d},
                   "log": os.path.join(jdir, "log.txt"), "child": None}
            if not st or st.get("status") in ("running", "queued"):
                job["st"].update(status="interrupted", ended=NOW(),
                                 err="web 重启中断,需要重新提交")
                self.persist_status(job)
            self.jobs[d] = job

    def new_id(self):
        return time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:4]

    def submit(self, cfg, jid=None):
        with self.lock:
            if jid is None:
                jid = self.new_id()
            job = {"id": jid, "cfg": cfg, "log": os.path.join(self._job_dir(jid), "log.txt"),
                   "st": {"id": jid, "status": "queued", "created": NOW(),
                          "created_ts": time.time(), "tag": jid, "stage": None,
                          "params": cfg["params"], "seed": cfg["seed"],
                          "media": cfg["media"]},
                   "child": None}
            self.jobs[jid] = job
            self.persist_cfg(job)
            self.persist_status(job)
            self.queue.append(jid)
            with self._cv:
                self._cv.notify()
            return jid

    # ---- worker ----
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
            except Exception as e:
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
            st = job["st"]
            if key == "status" and value in ("done", "failed", "cancelled", "interrupted") \
                    and "duration" not in st and st.get("created_ts"):
                st["duration"] = int(time.time() - st["created_ts"])
            st[key] = value
            for k, v in extra.items():
                st[k] = v
            if key in ("status", "stage") or extra:
                self.persist_status(job)

    def _stage(self, job):
        try:
            with open(job["log"], "rb") as f:
                text = f.read().decode("utf-8", "replace")
        except Exception:
            text = ""
        job["st"]["stage"] = {"label": stage_from_log(text), "ts": NOW()}
        m = None
        for m in _PROG_RE.finditer(text):
            pass
        if m:
            cur, total = int(m.group(1)), int(m.group(2))
            if total > 0 and cur <= total:
                job["st"]["progress"] = {"cur": cur, "total": total}

    def blocking_busy(self):
        qr = self.comfy.running_prompts()
        if qr:
            return ["ComfyUI 正在执行其它任务"]
        return []

    def _build_argv(self, cfg):
        p = cfg["params"]
        a = [sys.executable, self.driver, "--tag", cfg["tag"], "--prompt", p["prompt"],
             "--dur", str(p["dur"]), "--aspect", p["aspect"],
             "--megapixels", str(p["megapixels"]), "--multiple", str(p["multiple"]),
             "--steps", str(p["steps"]), "--ref-image-size", p["ref_image_size"],
             "--out", os.path.join(self.out_dir, "%s.mp4" % cfg["tag"])]
        if cfg.get("seed") is not None:
            a += ["--seed", str(cfg["seed"])]
        m = cfg["media"]
        for f in m.get("ref_image", []):
            a += ["--image", f]
        for f in m.get("ref_video", []):
            a += ["--video", f]
        for f in m.get("ref_audio", []):
            a += ["--audio", f]
        return a

    def _run(self, jid):
        job = self.jobs[jid]
        cfg = job["cfg"]
        with open(job["log"], "a", encoding="utf-8") as log:
            log.write("=== job %s ===\n" % jid)
            t0 = time.time()
            while not self._stop:
                reasons = self.blocking_busy()
                if not reasons:
                    break
                if time.time() - t0 > BUSY_WAIT_MAX_S:
                    self._set(jid, "status", "failed", err="等待 GPU 超时")
                    log.write("[web] busy timeout\n"); log.flush(); return
                job["st"]["stage"] = {"label": "等待: " + "; ".join(reasons), "ts": NOW()}
                self.persist_status(job)
                log.write("[web] busy: %s\n" % "; ".join(reasons)); log.flush()
                time.sleep(5)
                if job["st"].get("cancel_requested"):
                    self._set(jid, "status", "cancelled", ended=NOW(), err="取消(等待中)")
                    return
            argv = self._build_argv(cfg)
            log.write("cmd: %s\n\n" % " ".join(argv)); log.flush()
            env = dict(os.environ, PYTHONUNBUFFERED="1")
            popen = subprocess.Popen(argv, cwd=self.root, env=env, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            with self.lock:
                job["child"] = popen
                self.children.add(popen.pid)
            self._set(jid, "status", "running")
            last = 0.0
            while popen.poll() is None:
                if job["st"].get("cancel_requested"):
                    self._term(job)
                if time.time() - last > 2:
                    self._stage(job)
                    self.persist_status(job)
                    last = time.time()
                time.sleep(0.5)
            rc = popen.returncode
            with self.lock:
                self.children.discard(popen.pid)
                job["child"] = None
            self._stage(job)
            if job["st"].get("cancel_requested"):
                self._set(jid, "status", "cancelled", ended=NOW(), exit=rc,
                          err="已取消")
                log.write("\n[web] cancelled rc=%s\n" % rc)
            elif rc == 0:
                rel = "ref2v/%s.mp4" % cfg["tag"]
                p = os.path.join(self.root, "output", rel)
                size = os.path.getsize(p) if os.path.isfile(p) else 0
                frames = ref2v_length(cfg["params"]["dur"])
                self._set(jid, "status", "done", ended=NOW(), exit=0, err=None,
                          duration=int(time.time() - job["st"].get("created_ts", time.time())),
                          clip_rel=rel if size else None, clip_size=size,
                          clip_frames=frames, clip_seconds=round(frames / FPS, 2))
                log.write("\n[web] done rc=0 rel=%s size=%d\n" % (rel, size))
            else:
                self._set(jid, "status", "failed", ended=NOW(), exit=rc,
                          err="runner exited rc=%s" % rc)
                log.write("\n[web] failed rc=%s\n" % rc)
            log.flush()

    def _term(self, job):
        child = job.get("child")
        if child is None:
            return
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except Exception:
            pass

    def cancel(self, jid):
        with self.lock:
            job = self.jobs.get(jid)
            if not job or job["st"].get("status") not in ("queued", "running"):
                return False, "任务不存在或已结束"
            if job["st"]["status"] == "queued":
                job["st"].update(status="cancelled", ended=NOW(),
                                 duration=int(time.time() - job["st"].get("created_ts", time.time())))
                self.persist_status(job)
                self.queue = [x for x in self.queue if x != jid]
                return True, "已取消(排队中)"
            job["st"]["cancel_requested"] = True
            self._term(job)
            return True, "已发送取消"

    def delete(self, jid):
        with self.lock:
            job = self.jobs.get(jid)
            if not job:
                return False, "不存在"
            st = job["st"].get("status")
            if st == "running":
                return False, "运行中的任务请先取消"
            if st == "queued":
                self.queue = [x for x in self.queue if x != jid]
                job["st"].update(status="cancelled", ended=NOW())
            shutil.rmtree(self._job_dir(jid), ignore_errors=True)
            del self.jobs[jid]
            return True, "已删除记录(产物保留)"

    # ---- snapshots ----
    def info(self, job):
        st = job["st"]
        return {"id": st["id"], "status": st.get("status"), "created": st.get("created"),
                "created_ts": st.get("created_ts"),
                "ended": st.get("ended"), "duration": st.get("duration"),
                "stage": st.get("stage"), "progress": st.get("progress"),
                "err": st.get("err"), "seed": st.get("seed"),
                "params": st.get("params"), "media": st.get("media"),
                "clip_rel": st.get("clip_rel"), "clip_size": st.get("clip_size"),
                "clip_frames": st.get("clip_frames"), "clip_seconds": st.get("clip_seconds"),
                "cancel_requested": bool(st.get("cancel_requested"))}

    def state(self):
        with self.lock:
            cur = self.info(self.jobs[self.current]) if self.current in self.jobs else None
            q = [self.info(self.jobs[j]) for j in self.queue if j in self.jobs]
        up, ms = self.comfy.check()
        return {"server_ts": NOW(), "comfy": {"up": up, "ms": ms},
                "vram": self.comfy.vram(), "blocking": self.blocking_busy(),
                "current": cur, "queue": q}

    def jobs_list(self):
        with self.lock:
            lst = [self.info(j) for j in self.jobs.values()]
        lst.sort(key=lambda x: x.get("created") or "", reverse=True)
        return lst

    def clips_list(self):
        out_root = os.path.join(self.root, "output")
        clips = []
        meta = {}
        for job in self.jobs.values():
            rel = job["st"].get("clip_rel")
            if rel:
                meta[os.path.basename(rel)] = self.info(job)
        for p in sorted(glob.glob(os.path.join(self.out_dir, "*.mp4")),
                        key=os.path.getmtime, reverse=True):
            rel = os.path.relpath(p, out_root).replace("\\", "/")
            m = meta.get(os.path.basename(p), {})
            clips.append({"rel": rel, "name": os.path.basename(p),
                          "size": os.path.getsize(p),
                          "ts": time.strftime("%m-%d %H:%M", time.localtime(os.path.getmtime(p))),
                          "job": m.get("id"), "seconds": m.get("clip_seconds"),
                          "frames": m.get("clip_frames"), "seed": m.get("seed"),
                          "params": m.get("params")})
        return {"clips": clips}


# ------------------------------------------------------------------- handler
_OUTPUT_RE = re.compile(r"^/files/(.+)$")


def make_handler(mgr):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            pass

        def _authed(self):
            """HTTP Basic auth; any username, the configured password decides."""
            hdr = self.headers.get("Authorization", "")
            if hdr.startswith("Basic "):
                try:
                    raw = base64.b64decode(hdr[6:].strip()).decode("utf-8", "replace")
                except Exception:
                    raw = ""
                if ":" in raw and raw.split(":", 1)[1] == mgr.password:
                    return True
            self.send_response(401)
            self.send_header("WWW-Authenticate", 'Basic realm="Ref2V"')
            self.send_header("Content-Length", "0")
            self.end_headers()
            return False

        def _json(self, code, obj):
            body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _err(self, code, msg):
            self._json(code, {"error": msg})

        def _send_file_range(self, path, force_dl=False):
            try:
                size = os.path.getsize(path)
            except OSError:
                self._err(404, "file not found"); return
            rng = self.headers.get("Range")
            start, end = 0, size - 1
            if rng and rng.startswith("bytes="):
                spec = rng[6:].strip()
                if spec and spec[0] == "-":
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
                    self.end_headers(); return
            ctype = mimetypes.guess_type(path)[0] or "application/octet-stream"
            length = end - start + 1
            if rng:
                self.send_response(206)
                self.send_header("Content-Range", "bytes %d-%d/%d" % (start, end, size))
            else:
                self.send_response(200)
            self.send_header("Content-Type", ctype)
            if force_dl:
                name = os.path.basename(path)
                try:
                    name.encode("ascii"); ascii_name = name
                except UnicodeEncodeError:
                    ascii_name = "download" + os.path.splitext(name)[1]
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
                self._err(403, "path outside output"); return
            if not os.path.isfile(cand):
                self._err(404, "file not found"); return
            self._send_file_range(cand, force_dl)

        def do_GET(self):
            if not self._authed():
                return
            u = urlparse(self.path)
            if u.path == "/":
                body = HTML.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
            elif u.path == "/api/state":
                self._json(200, mgr.state())
            elif u.path == "/api/jobs":
                self._json(200, {"jobs": mgr.jobs_list()})
            elif u.path.startswith("/api/jobs/"):
                rest = u.path[len("/api/jobs/"):]
                jid, _, sub = rest.partition("/")
                job = mgr.jobs.get(jid)
                if not job:
                    self._err(404, "not found"); return
                if sub == "log":
                    offset = int(parse_qs(u.query).get("offset", ["0"])[0] or 0)
                    try:
                        with open(job["log"], "rb") as f:
                            f.seek(offset)
                            raw = f.read(); new_off = f.tell()
                        data = raw.decode("utf-8", "replace")
                    except OSError:
                        data, new_off = "", offset
                    self._json(200, {"id": jid, "status": job["st"].get("status"),
                                     "offset": new_off, "text": data,
                                     "stage": job["st"].get("stage")})
                elif sub == "":
                    self._json(200, mgr.info(job))
                else:
                    self._err(404, "not found")
            elif u.path == "/api/outputs":
                self._json(200, mgr.clips_list())
            else:
                m = _OUTPUT_RE.match(u.path)
                if m:
                    self._serve_output(m.group(1),
                                       parse_qs(u.query).get("dl", ["0"])[0] == "1")
                else:
                    self._err(404, "not found")

        def do_HEAD(self):
            if not self._authed():
                return
            m = _OUTPUT_RE.match(urlparse(self.path).path)
            if m:
                self._serve_output(m.group(1))
            else:
                self._err(404, "not found")

        def do_POST(self):
            if not self._authed():
                return
            u = urlparse(self.path)
            m = re.match(r"^/api/jobs/([^/]+)/(cancel|delete)$", u.path)
            if m:
                jid, action = m.group(1), m.group(2)
                ok, msg = (mgr.cancel(jid) if action == "cancel" else mgr.delete(jid))
                self._json(200 if ok else 409, {"ok": ok, "msg": msg}); return
            if u.path == "/api/service/stop":
                if mgr.current:
                    self._json(409, {"ok": False, "msg": "有任务在跑, 先取消"}); return
                subprocess.run(["bash", os.path.expanduser("~/ComfyUI-Deploy/stop.sh")], check=False)
                self._json(200, {"ok": True, "msg": "已停止 ComfyUI, 显存已释放"}); return
            if u.path == "/api/output/delete":
                try:
                    length = int(self.headers.get("Content-Length") or 0)
                    js = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
                except Exception:
                    self._err(400, "bad json"); return
                rel = (js.get("rel") or "").strip()
                base = os.path.realpath(os.path.join(mgr.root, "output"))
                cand = os.path.realpath(os.path.join(base, rel)) if rel else base
                if not rel or (cand != base and not cand.startswith(base + os.sep)):
                    self._err(403, "bad path"); return
                if os.path.isfile(cand):
                    os.remove(cand)
                    self._json(200, {"ok": True, "msg": "已删除"}); return
                self._err(404, "file not found"); return
            if u.path == "/api/optimize":
                try:
                    length = int(self.headers.get("Content-Length") or 0)
                    js = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
                except Exception:
                    self._err(400, "bad json"); return
                prompt = (js.get("prompt") or "").strip()
                if not prompt:
                    self._err(400, "请先填写提示词"); return
                if len(prompt) > 4000:
                    self._err(400, "提示词过长（>4000 字符）"); return
                ok, text = llm_optimize(prompt, js.get("counts") or {})
                self._json(200, {"ok": True, "text": text}) if ok else self._err(502, text)
                return
            if u.path != "/api/run":
                self._err(404, "not found"); return
            try:
                length = int(self.headers.get("Content-Length") or 0)
                if length <= 0:
                    self._err(400, "empty body"); return
                if length > UPLOAD_MAX:
                    self._err(413, "body too large (>2GiB)"); return
                body = self.rfile.read(length)
            except Exception as e:
                self._err(400, "read body failed: %r" % e); return
            ct = self.headers.get("Content-Type", "")
            if ct.startswith("multipart/form-data"):
                bm = re.search(r'boundary="?([^";]+)"?', ct)
                if not bm:
                    self._err(400, "missing boundary"); return
                try:
                    fields, files = parse_multipart(body, bm.group(1).encode("ascii"))
                except ValueError as e:
                    self._err(400, str(e)); return
            else:
                self._err(400, "expected multipart/form-data"); return
            self._api_run(fields, files)

        def _api_run(self, fields, files):
            prompt = (_first(fields, "prompt", "") or "").strip()
            if not prompt:
                self._err(400, "请填写提示词"); return
            by_kind = {k: [] for k in MEDIA_KINDS}
            for f in files:
                if f["name"] in by_kind:
                    by_kind[f["name"]].append(f)
            n = {k: len(v) for k, v in by_kind.items()}
            if n["ref_image"] > MAX_IMAGES:
                self._err(400, "参考图最多 %d 张" % MAX_IMAGES); return
            if n["ref_video"] > MAX_VIDEOS:
                self._err(400, "参考视频最多 %d 段" % MAX_VIDEOS); return
            if n["ref_audio"] > MAX_AUDIOS:
                self._err(400, "参考音频最多 %d 段" % MAX_AUDIOS); return
            if not any(n.values()):
                self._err(400, "请至少上传一张参考图/视频/音频"); return

            seed = _to_int(_first(fields, "seed"), None, lo=0, hi=2**63 - 1)
            if seed is None:
                seed = random.randint(0, 2**63 - 1)
            cfg = {
                "tag": None,  # filled with the job id
                "seed": seed,
                "params": {
                    "prompt": prompt,
                    "dur": _to_float(_first(fields, "dur"), 5.0, lo=1.0, hi=15.0),
                    "aspect": _first(fields, "aspect", ASPECTS[0]),
                    "megapixels": _to_float(_first(fields, "megapixels"), 0.4, lo=0.1, hi=2.0),
                    "multiple": 32,
                    "steps": _to_int(_first(fields, "steps"), 8, lo=1, hi=50),
                    "ref_image_size": "max" if _first(fields, "ref_image_size") == "max" else "match",
                },
                "media": {},
            }
            jid = mgr.new_id()
            cfg["tag"] = jid
            jdir = mgr._job_dir(jid)
            os.makedirs(jdir, exist_ok=True)
            up_dir = os.path.join(jdir, "uploads")
            os.makedirs(up_dir, exist_ok=True)
            for kind in MEDIA_KINDS:
                paths = []
                for i, f in enumerate(by_kind[kind]):
                    safe = _safe_name(f["filename"])
                    dst = os.path.join(up_dir, "%s_%02d_%s" % (kind, i, safe))
                    with open(dst, "wb") as wf:
                        wf.write(f["content"])
                    paths.append(dst)
                cfg["media"][kind] = paths
            mgr.submit(cfg, jid=jid)
            self._json(202, {"id": jid, "status": "queued"})

    return H


HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Ref2V · MiniMax H3</title>
<style>
:root{--bg:#0f1115;--card:#171a21;--line:#252a34;--fg:#e7e9ee;--mut:#9aa3b2;
      --acc:#4f8cff;--ok:#35c07a;--warn:#e0a63a;--err:#e05a5a}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,"PingFang SC","Microsoft YaHei",sans-serif}
header{position:sticky;top:0;z-index:5;display:flex;flex-wrap:wrap;gap:8px;align-items:center;
       padding:10px 14px;background:rgba(15,17,21,.92);backdrop-filter:blur(8px);border-bottom:1px solid var(--line)}
header h1{font-size:16px;margin:0 8px 0 0}
.pill{font-size:12px;padding:3px 9px;border-radius:999px;background:#20242d;color:var(--mut);white-space:nowrap}
.pill.on{color:#0b0d11;background:var(--ok)} .pill.off{color:#0b0d11;background:var(--err)}
.spacer{flex:1}
main{display:grid;grid-template-columns:minmax(340px,460px) 1fr;gap:14px;padding:14px;max-width:1400px;margin:0 auto}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px;margin-bottom:14px}
.card h2{font-size:14px;margin:0 0 10px;color:var(--mut);font-weight:600;letter-spacing:.03em}
label{display:block;font-size:13px;color:var(--mut);margin:10px 0 4px}
textarea,input,select{width:100%;background:#0e1116;border:1px solid var(--line);color:var(--fg);
       border-radius:8px;padding:9px 10px;font-size:15px}
textarea{min-height:96px;resize:vertical}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.grid3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px}
.filebox{border:1px dashed var(--line);border-radius:8px;padding:8px;margin-top:4px}
.filebox .hint{font-size:12px;color:var(--mut)}
.filebox ul{list-style:none;margin:6px 0 0;padding:0;font-size:12px;color:var(--fg)}
.filebox li{display:flex;justify-content:space-between;align-items:center;gap:8px;padding:3px 0;border-bottom:1px dotted var(--line)}
.filebox li .fname{flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.filebox li .acts{display:flex;gap:6px;flex:0 0 auto}
.filebox li button.rm{background:none;border:0;color:var(--err);cursor:pointer;font-size:12px}
.filebox li button.chip{background:#22324d;color:#bcd4ff;border:1px solid #33507d;border-radius:999px;
       padding:1px 9px;font-size:12px;cursor:pointer}
.filebox li button.chip:active{background:#2f4a6e}
button.primary{width:100%;margin-top:14px;padding:12px;border:0;border-radius:9px;background:var(--acc);
       color:#fff;font-size:15px;font-weight:600;cursor:pointer}
button.primary:disabled{opacity:.5;cursor:default}
button.ghost{background:#20242d;color:var(--fg);border:1px solid var(--line);border-radius:8px;
       padding:6px 10px;font-size:13px;cursor:pointer}
.progress{height:6px;background:#20242d;border-radius:3px;overflow:hidden;margin-top:8px;display:none}
.progress>i{display:block;height:100%;width:0;background:var(--acc);transition:width .2s}
.curState{font-size:18px;font-weight:600;margin-bottom:8px}
.bar{height:8px;background:#20242d;border-radius:4px;overflow:hidden;margin:4px 0 8px}
.bar>i{display:block;height:100%;width:0;background:var(--acc);transition:width .3s}
.bar.indet>i{width:35%;animation:slide 1.3s ease-in-out infinite}
@keyframes slide{0%{margin-left:-35%}100%{margin-left:100%}}
.curMeta{font-size:12px;color:var(--mut);line-height:1.7}
details.logBox summary{cursor:pointer}
details.logBox pre{margin-top:8px}
details.sec>summary{list-style:none;cursor:pointer;font-size:14px;color:var(--mut);font-weight:600;
       letter-spacing:.03em;display:flex;align-items:center;gap:6px}
details.sec>summary::-webkit-details-marker{display:none}
details.sec>summary::before{content:"\25BC";font-size:.8em;line-height:1;color:var(--fg);transition:transform .2s}
details.sec:not([open])>summary::before{transform:rotate(-90deg)}
details.sec[open]>summary{margin-bottom:10px}
pre.log{max-height:240px;overflow:auto;background:#0b0d11;border:1px solid var(--line);border-radius:8px;
        padding:8px;font-size:12px;color:#c7cede;white-space:pre-wrap;word-break:break-all}
.job{display:flex;gap:10px;align-items:center;padding:9px 0;border-bottom:1px solid var(--line);flex-wrap:wrap}
.job .st{font-size:12px;padding:2px 8px;border-radius:999px;background:#20242d}
.st.done{background:var(--ok);color:#0b0d11}.st.running{background:var(--acc);color:#fff}
.st.failed,.st.cancelled,.st.interrupted{background:var(--err);color:#fff}.st.queued{background:var(--warn);color:#0b0d11}
.job .meta{font-size:12px;color:var(--mut);flex:1;min-width:160px}
.clips{display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:10px}
.clip{background:#0e1116;border:1px solid var(--line);border-radius:10px;overflow:hidden;cursor:pointer}
.clip video,.clip .ph{width:100%;aspect-ratio:16/9;background:#000;display:block;object-fit:cover}
.clip .cap{padding:6px 8px;font-size:12px;color:var(--mut)}
.modal{position:fixed;inset:0;background:rgba(0,0,0,.8);display:none;align-items:center;justify-content:center;z-index:20;padding:12px}
.modal.open{display:flex}
.modal .box{width:min(960px,98vw);background:#0e1116;border:1px solid var(--line);border-radius:12px;padding:10px}
.modal video{width:100%;max-height:76vh;background:#000;border-radius:8px}
.opttext{width:100%;min-height:220px;max-height:56vh;background:#0b0e13;color:var(--fg);
       border:1px solid var(--line);border-radius:8px;padding:10px;font-size:13px;line-height:1.5;resize:vertical}
.optrow{display:flex;justify-content:space-between;align-items:center;margin-bottom:8px}
.optacts{display:flex;gap:8px;margin-top:10px}
.optacts button.primary,.optacts button.ghost{flex:1 1 0;width:auto;height:38px;margin:0;padding:0 10px;
       display:flex;align-items:center;justify-content:center;box-sizing:border-box}
.muted{color:var(--mut);font-size:12px}
@media(max-width:980px){main{grid-template-columns:1fr}}
@media(max-width:640px){.grid3{grid-template-columns:1fr 1fr}textarea,input,select{font-size:16px}}
</style>
</head>
<body>
<header>
  <h1>Ref2V · MiniMax H3</h1>
  <span id="pComfy" class="pill off">ComfyUI ?</span>
  <span id="pVram" class="pill">VRAM --</span>
  <span id="pQ" class="pill">队列 0</span>
  <span class="spacer"></span>
  <button class="ghost" onclick="releaseVram()">释放显存</button>
</header>
<main>
  <div>
    <div class="card">
      <h2>新建任务</h2>
      <label>提示词（点下方素材后的「图片1 / 视频1 / 音频1」按钮即可插入 &lt;Picture 1&gt; 等引用）</label>
      <textarea id="prompt" placeholder="Cinematic shot of the subject in <Picture 1> ..."></textarea>
      <div style="margin-top:6px"><button class="ghost" id="optBtn" onclick="optimizePrompt()">提示词优化</button></div>
      <label>参考图（≤9）</label>
      <div class="filebox"><input id="fImg" type="file" accept="image/*" multiple><ul id="lImg"></ul></div>
      <label>参考视频（≤3，每段 2–15s，合计 ≤15s）</label>
      <div class="filebox"><input id="fVid" type="file" accept="video/*" multiple><ul id="lVid"></ul></div>
      <label>参考音频（≤3，合计 ≤15s）</label>
      <div class="filebox"><input id="fAud" type="file" accept="audio/*" multiple><ul id="lAud"></ul></div>
      <div class="grid3">
        <div><label>时长(秒)</label><input id="dur" type="number" value="5" min="1" max="15" step="0.5"></div>
        <div><label>步数</label><input id="steps" type="number" value="8" min="1" max="50"></div>
        <div><label>seed(空=随机)</label><input id="seed" type="number" placeholder="随机"></div>
      </div>
      <div class="grid2">
        <div><label>画幅</label><select id="aspect"></select></div>
        <div><label>分辨率(MP)</label>
          <select id="megapixels">
            <option value="0.2">0.2</option><option value="0.3">0.3</option>
            <option value="0.4" selected>0.4</option><option value="0.5">0.5</option>
            <option value="0.6">0.6</option><option value="0.8">0.8</option>
          </select></div>
      </div>
      <div><label>参考图缩放</label>
        <select id="ref_image_size"><option value="match">match(快)</option><option value="max">max(保真)</option></select></div>
      <button class="primary" id="submitBtn" onclick="submit()">提交任务</button>
      <div class="progress" id="prog"><i></i></div>
      <div class="muted" id="submitMsg" style="margin-top:8px"></div>
    </div>
  </div>
  <div>
    <div class="card">
      <h2>当前任务</h2>
      <div id="cur"><div class="muted">空闲</div></div>
      <details class="logBox" id="logBox" style="margin-top:12px">
        <summary class="muted">诊断日志</summary>
        <pre class="log" id="log"></pre>
      </details>
    </div>
    <div class="card">
      <details class="sec" open>
        <summary>任务记录</summary>
        <div id="jobs" class="muted">暂无</div>
      </details>
    </div>
    <div class="card">
      <details class="sec" open>
        <summary>产物</summary>
        <div id="clips" class="clips"></div>
      </details>
    </div>
  </div>
</main>
<div class="modal" id="modal" onclick="if(event.target===this)closeModal()">
  <div class="box">
    <video id="mvideo" controls playsinline webkit-playsinline></video>
    <div class="muted" id="mcap" style="margin-top:8px"></div>
  </div>
</div>
<div class="modal" id="optModal" onclick="if(event.target===this)closeOpt()">
  <div class="box">
    <div class="optrow"><b>提示词优化</b><button class="ghost" onclick="closeOpt()">关闭</button></div>
    <div class="muted" id="optMsg" style="margin-bottom:8px"></div>
    <textarea id="optText" class="opttext" readonly placeholder="优化结果将显示在这里"></textarea>
    <div class="optacts">
      <button class="primary" onclick="copyOpt()">复制</button>
      <button class="ghost" onclick="applyOpt()">替换输入框</button>
    </div>
  </div>
</div>
<script>
const $ = (id)=>document.getElementById(id);
const ASPECTS = __ASPECTS__;
let logOffset = 0, lastJob = null, jobsById = {}, curStart = 0, curRunning = false;
const STATUS_CN = {queued:'排队中', running:'进行中', done:'已完成', failed:'失败', cancelled:'已取消', interrupted:'已中断'};
const MEDIA_CN = {ref_image:'图', ref_video:'视频', ref_audio:'音频'};
function friendlyErr(e){
  if(!e) return '';
  if(/runner exited rc=/.test(e)) return '生成进程异常退出';
  if(/no mp4 saved/.test(e)) return '未产出视频';
  if(/NODE ERROR|out of memory|OOM/i.test(e)) return '显存不足或节点出错';
  if(/取消|cancel/i.test(e)) return '已取消';
  return e.slice(0,120);
}
function mediaBrief(m){
  if(!m) return '';
  return Object.keys(MEDIA_CN).map(k=> (m[k]&&m[k].length)? m[k].length+MEDIA_CN[k] : null).filter(Boolean).join(' · ');
}
function friendlyStatus(j){
  if(j.status==='running'){
    const lab = (j.stage&&j.stage.label) || '进行中';
    const p = j.progress;
    return p? (lab+' '+p.cur+'/'+p.total) : lab;
  }
  if(j.status==='done') return '已完成'+(j.clip_seconds? ' · '+j.clip_seconds+'s':'');
  if(j.status==='failed') return '失败：'+friendlyErr(j.err);
  if(j.status==='cancelled') return '已取消';
  if(j.status==='interrupted') return '已中断(web 重启)，请重新提交';
  return STATUS_CN[j.status]||j.status;
}
function fmtDur(sec){ sec=Math.max(0,Math.floor(sec)); return String(Math.floor(sec/60)).padStart(2,'0')+':'+String(sec%60).padStart(2,'0'); }
function renderCurrent(j){
  curStart = j.created_ts || curStart || 0;
  curRunning = (j.status==='running'||j.status==='queued');
  const p=j.params||{};
  const meta=[mediaBrief(j.media), p.dur?p.dur+'s':'', p.aspect?p.aspect.split(' ')[0]:'',
              p.megapixels?p.megapixels+'MP':'', p.steps?p.steps+'步':'', j.seed?('seed '+j.seed):''].filter(Boolean).join(' · ');
  let bar='';
  if(j.status==='running' && j.progress && j.progress.total){
    bar='<div class="bar"><i style="width:'+Math.round(j.progress.cur/j.progress.total*100)+'%"></i></div>';
  }else if(j.status==='running'||j.status==='queued'){
    bar='<div class="bar indet"><i></i></div>';
  }
  const cls=(j.status==='failed')?' style="color:var(--err)"':'';
  $('cur').innerHTML='<div class="curState"'+cls+'>'+friendlyStatus(j)+'</div>'+bar+
    '<div class="curMeta"><b>'+j.id+'</b>'+(meta?'<br>'+meta:'')+
    (curRunning&&curStart?'<br>已用时 <span id="curElapsed">'+fmtDur(Date.now()/1000-curStart)+'</span>':'')+
    (j.status==='done'&&j.clip_rel?'<br><button class="ghost" onclick="play(\''+j.clip_rel+'\')">查看产物</button>':'')+'</div>';
}
setInterval(()=>{ const el=$('curElapsed'); if(el&&curRunning&&curStart) el.textContent=fmtDur(Date.now()/1000-curStart); },1000);

for (const a of ASPECTS){ const o=document.createElement('option'); o.value=a; o.textContent=a; $('aspect').appendChild(o); }
$('aspect').value = ASPECTS[0];

function fmtSize(n){ if(n>1048576) return (n/1048576).toFixed(1)+'MB'; if(n>1024) return (n/1024).toFixed(0)+'KB'; return n+'B'; }
function stCls(s){ return 'st '+s; }

function insertRef(prefix, idx){
  const ta=$('prompt'), tag='<'+prefix+' '+(idx+1)+'>';
  const s=ta.selectionStart??ta.value.length, e=ta.selectionEnd??s;
  ta.value=ta.value.slice(0,s)+tag+ta.value.slice(e);
  const pos=s+tag.length; ta.focus(); ta.setSelectionRange(pos,pos);
}
function bindFiles(inputId, listId, max, cn, prefix){
  const input=$(inputId), list=$(listId); let files=[];
  function render(){
    list.innerHTML='';
    files.forEach((f,i)=>{
      const li=document.createElement('li');
      const nm=document.createElement('span'); nm.className='fname'; nm.textContent=f.name;
      const acts=document.createElement('span'); acts.className='acts';
      const chip=document.createElement('button'); chip.className='chip'; chip.textContent=cn+(i+1);
      chip.title='插入引用 <'+prefix+' '+(i+1)+'>';
      chip.onclick=()=>insertRef(prefix,i);
      const rm=document.createElement('button'); rm.className='rm'; rm.textContent='移除';
      rm.onclick=()=>{files.splice(i,1); input.value=''; render();};
      acts.appendChild(chip); acts.appendChild(rm);
      li.appendChild(nm); li.appendChild(acts); list.appendChild(li);
    });
  }
  input.onchange=()=>{
    for(const f of Array.from(input.files)){
      if(files.length>=max){ $('submitMsg').textContent=cn+' 最多 '+max+' 个'; break; }
      const dup=files.some(x=>x.name===f.name&&x.size===f.size&&x.lastModified===f.lastModified);
      if(!dup) files.push(f);
    }
    input.value='';            // allow picking the same file again / keep appending
    render();
  };
  return ()=>files;
}
const getImg = bindFiles('fImg','lImg',9,'图片','Picture');
const getVid = bindFiles('fVid','lVid',3,'视频','Video');
const getAud = bindFiles('fAud','lAud',3,'音频','Audio');

function submit(){
  const prompt=$('prompt').value.trim();
  if(!prompt){ alert('请填写提示词'); return; }
  const imgs=getImg(), vids=getVid(), auds=getAud();
  if(!imgs.length && !vids.length && !auds.length){ alert('请至少上传一个参考素材'); return; }
  const fd=new FormData();
  fd.append('prompt', prompt);
  fd.append('dur', $('dur').value); fd.append('steps', $('steps').value);
  fd.append('aspect', $('aspect').value); fd.append('megapixels', $('megapixels').value);
  fd.append('ref_image_size', $('ref_image_size').value);
  if($('seed').value) fd.append('seed', $('seed').value);
  imgs.forEach(f=>fd.append('ref_image', f));
  vids.forEach(f=>fd.append('ref_video', f));
  auds.forEach(f=>fd.append('ref_audio', f));
  const xhr=new XMLHttpRequest(); xhr.open('POST','/api/run');
  $('prog').style.display='block'; $('prog').firstElementChild.style.width='0%';
  $('submitBtn').disabled=true; $('submitMsg').textContent='上传中...';
  xhr.upload.onprogress=(e)=>{ if(e.lengthComputable) $('prog').firstElementChild.style.width=(e.loaded/e.total*100)+'%'; };
  xhr.onload=()=>{
    $('submitBtn').disabled=false;
    try{ const r=JSON.parse(xhr.responseText);
      if(xhr.status===202){ $('submitMsg').textContent='已提交: '+r.id; $('prompt').value=''; }
      else alert('提交失败: '+(r.error||xhr.status));
    }catch(e){ alert('提交失败: '+xhr.status); }
    setTimeout(()=>{ $('prog').style.display='none'; },600);
    refreshJobs();
  };
  xhr.onerror=()=>{ $('submitBtn').disabled=false; alert('网络错误'); };
  xhr.send(fd);
}

async function api(path){ const r=await fetch(path); return r.ok? r.json(): null; }

async function refreshState(){
  const s=await api('/api/state'); if(!s) return;
  $('pComfy').className='pill '+(s.comfy.up?'on':'off');
  $('pComfy').textContent='ComfyUI '+(s.comfy.up?'在线':'离线');
  if(s.vram && s.vram.length){
    $('pVram').textContent='VRAM '+s.vram.map(v=>Math.round(v.used/1073741824)+'/'+Math.round(v.total/1073741824)+'G').join(' ');
  }
  $('pQ').textContent='队列 '+((s.queue||[]).length+(s.current?1:0));
  if(s.current){
    const j=s.current;
    if(lastJob!==j.id){ lastJob=j.id; logOffset=0; $('log').textContent=''; }
    renderCurrent(j);
  }else if(lastJob && jobsById[lastJob]){
    renderCurrent(jobsById[lastJob]);
  }
}

async function pollLog(){
  if(!lastJob) return;
  const r=await api('/api/jobs/'+lastJob+'/log?offset='+logOffset);
  if(!r) return;
  if(r.text){ $('log').textContent += r.text; $('log').scrollTop=$('log').scrollHeight; }
  logOffset=r.offset;
}

async function refreshJobs(){
  const r=await api('/api/jobs'); if(!r) return;
  const box=$('jobs');
  if(!r.jobs.length){ box.innerHTML='<span class="muted">暂无</span>'; return; }
  box.innerHTML='';
  r.jobs.slice(0,50).forEach(j=>{
    jobsById[j.id]=j;
    const d=document.createElement('div'); d.className='job';
    const p=j.params||{};
    const info=[mediaBrief(j.media), p.dur?p.dur+'s':'', p.megapixels?p.megapixels+'MP':''].filter(Boolean).join(' · ');
    const used = (j.duration!=null)? j.duration : (j.created_ts? Math.max(0, Date.now()/1000-j.created_ts) : null);
    const usedTxt = (used!=null)? ('用时 '+fmtDur(used)) : '';
    const line2 = [info, usedTxt].filter(Boolean).join(' · ');
    const note = j.status==='failed'? '<span style="color:var(--err)">'+friendlyErr(j.err)+'</span>' : (j.stage&&j.status==='running'? j.stage.label : '');
    let acts='';
    if(j.status==='queued'||j.status==='running') acts='<button class="ghost" onclick="jobAct(\''+j.id+'\',\'cancel\')">取消</button>';
    else acts='<button class="ghost" onclick="jobAct(\''+j.id+'\',\'delete\')">删除</button>';
    if(j.clip_rel) acts+=' <button class="ghost" onclick="play(\''+j.clip_rel+'\')">查看</button>';
    d.innerHTML='<span class="'+stCls(j.status)+'">'+(STATUS_CN[j.status]||j.status)+'</span>'+
      '<span class="meta"><b>'+j.id+'</b><br>'+line2+'<br>'+(note||j.created||'')+'</span>'+acts;
    box.appendChild(d);
  });
}

function jobAct(id,act){ if(!confirm(act==='cancel'?'取消任务 '+id+'?':'删除记录 '+id+'?')) return;
  fetch('/api/jobs/'+id+'/'+act,{method:'POST'}).then(()=>{refreshJobs();refreshOutputs();}); }

async function refreshOutputs(){
  const r=await api('/api/outputs'); if(!r) return;
  const box=$('clips');
  if(!r.clips.length){ box.innerHTML='<span class="muted">暂无产物</span>'; return; }
  box.innerHTML='';
  r.clips.forEach(c=>{
    const d=document.createElement('div'); d.className='clip'; d.onclick=()=>play(c.rel);
    d.innerHTML='<video preload="metadata" muted playsinline src="/files/'+encodeURI(c.rel)+'"></video>'+
      '<div class="cap">'+c.name.slice(0,20)+'<br>'+fmtSize(c.size)+' · '+c.ts+'</div>';
    box.appendChild(d);
  });
}

function play(rel){ $('mvideo').src='/files/'+encodeURI(rel); $('mcap').textContent=rel;
  $('modal').classList.add('open'); $('mvideo').play().catch(()=>{}); }
function closeModal(){ $('mvideo').pause(); $('mvideo').src=''; $('modal').classList.remove('open'); }

function releaseVram(){ if(!confirm('停止 ComfyUI 并释放显存? 下次生成需冷启动。')) return;
  fetch('/api/service/stop',{method:'POST'}).then(async r=>{ const j=await r.json(); alert(j.msg||'ok'); refreshState(); }); }

async function optimizePrompt(){
  const prompt=$('prompt').value.trim();
  if(!prompt){ $('submitMsg').textContent='请先填写提示词'; return; }
  const counts={ref_image:getImg().length, ref_video:getVid().length, ref_audio:getAud().length};
  $('optText').value=''; $('optMsg').textContent='优化中…'; $('optBtn').disabled=true;
  $('optModal').classList.add('open');
  try{
    const r=await fetch('/api/optimize',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({prompt,counts})});
    const j=await r.json().catch(()=>({}));
    if(r.ok && j.ok){ $('optText').value=j.text; $('optMsg').textContent='优化完成'; }
    else $('optMsg').textContent='优化失败: '+(j.error||r.status);
  }catch(e){ $('optMsg').textContent='网络错误'; }
  $('optBtn').disabled=false;
}
function closeOpt(){ $('optModal').classList.remove('open'); }
function copyOpt(){
  const t=$('optText').value; if(!t){ $('optMsg').textContent='暂无可复制内容'; return; }
  if(navigator.clipboard && navigator.clipboard.writeText){
    navigator.clipboard.writeText(t).then(()=>{ $('optMsg').textContent='已复制到剪贴板'; })
      .catch(()=>fallbackCopy());
  } else fallbackCopy();
}
function fallbackCopy(){
  const ta=$('optText'); ta.removeAttribute('readonly'); ta.focus(); ta.select();
  try{ document.execCommand('copy'); $('optMsg').textContent='已复制到剪贴板'; }
  catch(e){ $('optMsg').textContent='复制失败，请手动选择文本'; }
  ta.setAttribute('readonly','');
}
function applyOpt(){ const t=$('optText').value; if(!t) return; $('prompt').value=t; closeOpt(); }

refreshState(); refreshJobs(); refreshOutputs();
setInterval(refreshState,2000); setInterval(refreshJobs,5000); setInterval(refreshOutputs,5000); setInterval(pollLog,1500);
</script>
</body>
</html>"""


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="LAN web console for ref2v_runner.py")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--root", default=None)
    ap.add_argument("--driver", default=None)
    ap.add_argument("--comfy-base", default=COMFY_BASE)
    ap.add_argument("--password", default=DEFAULT_PASSWORD,
                    help="HTTP Basic password (any username); default %s" % DEFAULT_PASSWORD)
    ap.add_argument("--start", action="store_true")
    ap.add_argument("--stop", action="store_true")
    ap.add_argument("--status", action="store_true")
    a = ap.parse_args()

    root = os.path.abspath(a.root or os.path.expanduser("~/MiniMax-H3-Deploy"))
    driver = os.path.abspath(a.driver or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "ref2v_runner.py"))
    data = os.path.join(root, ".h3ref2v")
    os.makedirs(data, exist_ok=True)
    pidfile = os.path.join(data, "ref2v_web.pid")
    logfile = os.path.join(root, "ref2v_web.log")

    def read_pid():
        try:
            with open(pidfile) as f:
                return int(f.read().strip())
        except Exception:
            return None

    if a.stop:
        pid = read_pid()
        if not pid:
            print("no pid file (nothing running?)"); sys.exit(1)
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            print("pid %d not alive" % pid)
        for _ in range(50):
            try:
                os.kill(pid, 0); time.sleep(0.2)
            except ProcessLookupError:
                print("stopped pid %d" % pid)
                try:
                    os.remove(pidfile)
                except OSError:
                    pass
                return
        print("pid %d still alive after 10s" % pid); sys.exit(1)

    if a.status:
        pid = read_pid()
        print("pid:", pid or "(none)")
        print("url: http://%s:%d/" % (a.host, a.port))
        if pid:
            try:
                auth = base64.b64encode(("aiakos:" + a.password).encode()).decode()
                req = urllib.request.Request(
                    "http://127.0.0.1:%d/api/state" % a.port,
                    headers={"Authorization": "Basic " + auth})
                s = json.load(urllib.request.urlopen(req, timeout=3))
                print("comfy:", s["comfy"]); print("current:", (s["current"] or {}).get("id"))
                print("queue:", len(s["queue"]))
            except Exception as e:
                print("state query failed:", e)
        return

    if a.start and sys.platform != "win32":
        pid = os.fork()
        if pid > 0:
            deadline = time.time() + 10
            p = None
            while time.time() < deadline:
                p = read_pid()
                if p:
                    try:
                        os.kill(p, 0); break
                    except ProcessLookupError:
                        p = None; break
                time.sleep(0.1)
            if not p:
                print("daemon failed; check %s" % logfile, file=sys.stderr); sys.exit(1)
            print("started (pid %s) log: %s" % (p, logfile)); sys.exit(0)
        os.setsid()
        with open(pidfile, "w") as f:
            f.write(str(os.getpid()))
        lg = open(logfile, "ab", buffering=0)
        os.dup2(lg.fileno(), 1); os.dup2(lg.fileno(), 2)
        devnull = os.open(os.devnull, os.O_RDONLY)
        os.dup2(devnull, 0)
    else:
        with open(pidfile, "w") as f:
            f.write(str(os.getpid()))

    if not os.path.isfile(driver):
        print("driver not found: %s" % driver, file=sys.stderr)
        if not a.start:
            sys.exit(2)

    global HTML
    HTML = HTML.replace("__ASPECTS__", json.dumps(ASPECTS))

    mgr = Manager(root, driver, a.comfy_base, password=a.password)
    mgr.recover()
    mgr.start_worker()

    try:
        srv = ThreadingHTTPServer((a.host, a.port), make_handler(mgr))
    except OSError as e:
        print("bind %s:%d failed: %s" % (a.host, a.port, e), file=sys.stderr); sys.exit(1)
    srv.daemon_threads = True

    def _sigterm(signum, frame):
        mgr._stop = True
        with mgr.lock:
            for j in mgr.jobs.values():
                if j.get("child"):
                    mgr._term(j)
        threading.Thread(target=srv.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, _sigterm)
    signal.signal(signal.SIGINT, _sigterm)
    print("Ref2V Web on http://%s:%d/ (driver=%s)" % (a.host, a.port, driver))
    print("ComfyUI:", a.comfy_base, "| data:", data, "| auth: Basic (any user + password)")
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
