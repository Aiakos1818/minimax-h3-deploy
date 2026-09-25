#!/usr/bin/env python3
"""minimax_h3_web.py -- LAN web console for single-segment MiniMax H3 video.

Stdlib only (http.server); it never imports numpy/av/safetensors. Jobs are
grouped into projects (e.g. a story); each job is one prompt plus either
reference images/videos/audios (ref2v) or optional first/last keyframes (t2v).
The worker spawns minimax_h3_runner.py, which drives ComfyUI (:8188, resident
int4 CLIP + int8 UNet) and drops one mp4 under output/<project>/{ref2v,t2v}/.
Jobs run serially across all projects -- the GPUs render one clip at a time.

Usage:
  ~/ComfyUI-Deploy/comfyenv/bin/python scripts/minimax_h3_web.py --start|--stop|--status
Default: http://0.0.0.0:8191/   data: <root>/.h3ref2v/   log: <root>/minimax_h3_web.log
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
FRAME_KINDS = ("first_frame", "last_frame")
MODES = ("t2v", "ref2v")
OUT_SUBDIRS = {"t2v": "t2v", "ref2v": "ref2v"}
DEFAULT_PROJECT = "default"
DEFAULT_PROJECT_NAME = "默认项目"
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
    (r"NODE ERROR", "节点出错"),
    (r"submit error", "提交错误"),
]
# runner-emitted "[stage] <key>" -> friendly, detailed console text
_STAGE_LABELS = {
    "material": "预处理素材(上传/转码)",
    "queue": "已提交,等待执行",
    "sampling": "采样生成中",
    "done": "生成完成,正在收尾",
    "ray_rebuild": "重建 Ray 工作进程(冷加载/换 UNet)",
    "load_unet": "加载 UNet(FSDP 分片)",
    "load_clip": "加载 CLIP 文本编码器",
    "encode_clip": "CLIP 编码(提示词/参考)",
    "load_vae": "加载/解码 VAE",
}
_STAGE_LINE_RE = re.compile(r"^\[stage\]\s*(\S+)")
_PROG_RE = re.compile(r"\[progress\]\s+(\d+)/(\d+)")


def _stage_lines(text):
    return [l for l in text.splitlines()
            if l.strip() and "[progress]" not in l
            and not l.startswith(("cmd:", "=== job", "[web]", "[comfy]"))]


def stage_key_from_log(text):
    for l in reversed(_stage_lines(text)):
        m = _STAGE_LINE_RE.match(l)
        if m:
            return m.group(1)
    return None


def stage_from_log(text):
    lines = _stage_lines(text)
    for l in reversed(lines):
        m = _STAGE_LINE_RE.match(l)
        if m:
            return _STAGE_LABELS.get(m.group(1), m.group(1))[:80]
    for pat, lab in reversed(_STAGE_PATTERNS):
        for l in reversed(lines):
            if re.search(pat, l):
                return lab
    return lines[-1][:80] if lines else "启动生成进程..."


# -------------------------------------------------------------- job manager
class Manager:
    def __init__(self, root, driver, comfy_base, password=DEFAULT_PASSWORD):
        self.root = root
        self.driver = os.path.abspath(driver)
        self.password = password
        self.data = os.path.join(root, ".h3ref2v")
        self.jobs_dir = os.path.join(self.data, "jobs")
        self.projects_dir = os.path.join(self.data, "projects")
        self.out_root = os.path.join(root, "output")
        self.lock = threading.RLock()
        self._stop = False
        self.jobs = {}
        self.projects = {}
        self.queue = []
        self.current = None
        self.children = set()
        self.comfy = ComfyHealth(comfy_base)
        self._cv = threading.Condition(self.lock)
        os.makedirs(self.jobs_dir, exist_ok=True)
        os.makedirs(self.projects_dir, exist_ok=True)
        self.load_projects()
        for sub in OUT_SUBDIRS.values():
            os.makedirs(os.path.join(self.out_root, sub), exist_ok=True)

    def _out_dir(self, mode, project=None):
        sub = OUT_SUBDIRS.get(mode, "ref2v")
        if project:
            return os.path.join(self.out_root, project, sub)
        return os.path.join(self.out_root, sub)

    # ---- projects ----
    def _proj_dir(self, pid):
        return os.path.join(self.projects_dir, pid)

    def _write_project(self, p):
        d = self._proj_dir(p["id"])
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "project.json"), "w", encoding="utf-8") as f:
            json.dump(p, f, ensure_ascii=False, indent=2)

    def load_projects(self):
        for d in os.listdir(self.projects_dir):
            pf = os.path.join(self.projects_dir, d, "project.json")
            if not os.path.isfile(pf):
                continue
            try:
                with open(pf, encoding="utf-8") as f:
                    p = json.load(f)
                p["id"] = d
                p.setdefault("name", d)
                p.setdefault("created_ts", os.path.getmtime(pf))
                self.projects[d] = p
            except Exception:
                continue
        if DEFAULT_PROJECT not in self.projects:
            self.projects[DEFAULT_PROJECT] = {"id": DEFAULT_PROJECT,
                                              "name": DEFAULT_PROJECT_NAME,
                                              "created_ts": time.time()}
            self._write_project(self.projects[DEFAULT_PROJECT])

    def create_project(self, name):
        name = (name or "").strip()
        if not name:
            return None, "请填写项目名称"
        if len(name) > 60:
            return None, "项目名称过长（>60 字符）"
        pid = "p" + uuid.uuid4().hex[:8]
        p = {"id": pid, "name": name, "created_ts": time.time()}
        with self.lock:
            self.projects[pid] = p
            self._write_project(p)
        return p, None

    def rename_project(self, pid, name):
        name = (name or "").strip()
        if not name:
            return False, "请填写项目名称"
        if len(name) > 60:
            return False, "项目名称过长（>60 字符）"
        with self.lock:
            p = self.projects.get(pid)
            if not p:
                return False, "项目不存在"
            p["name"] = name
            self._write_project(p)
        return True, "已重命名"

    def _project_jobs(self, pid):
        return [j for j in self.jobs.values()
                if (j.get("st") or {}).get("project") == pid
                or (j.get("cfg") or {}).get("project") == pid]

    def delete_project(self, pid, mode="detach"):
        if pid == DEFAULT_PROJECT:
            return False, "默认项目不能删除"
        with self.lock:
            if pid not in self.projects:
                return False, "项目不存在"
            jobs = self._project_jobs(pid)
            busy = [j["id"] for j in jobs
                    if j["st"].get("status") in ("running", "queued")]
            if busy:
                return False, "项目内仍有运行/排队中的任务，请先取消"
            if mode == "purge":
                for j in jobs:
                    rel = j["st"].get("clip_rel")
                    if rel:
                        cand = os.path.realpath(os.path.join(self.out_root, rel))
                        base = os.path.realpath(self.out_root)
                        if cand.startswith(base + os.sep) and os.path.isfile(cand):
                            try:
                                os.remove(cand)
                            except OSError:
                                pass
                    shutil.rmtree(self._job_dir(j["id"]), ignore_errors=True)
                    del self.jobs[j["id"]]
            else:
                for j in jobs:
                    if j.get("cfg") is not None:
                        j["cfg"]["project"] = DEFAULT_PROJECT
                        self.persist_cfg(j)
                    j["st"]["project"] = DEFAULT_PROJECT
                    self.persist_status(j)
            shutil.rmtree(self._proj_dir(pid), ignore_errors=True)
            del self.projects[pid]
        return True, ("已删除项目及其任务与产物" if mode == "purge" else "已删除项目，任务已转默认项目")

    def _project_from_rel(self, rel):
        parts = rel.split("/")
        if len(parts) >= 3 and parts[0] in self.projects:
            return parts[0]
        return DEFAULT_PROJECT

    def project_info(self, pid, jobs=None):
        p = self.projects[pid]
        js = self._project_jobs(pid) if jobs is None else jobs
        counts = {"total": len(js), "running": 0, "queued": 0}
        cover = None
        cover_ts = ""
        for j in js:
            s = j["st"].get("status")
            if s in ("running", "queued"):
                counts[s] += 1
            rel = j["st"].get("clip_rel")
            if rel and j["st"].get("created", "") >= cover_ts:
                cover_ts = j["st"].get("created", "")
                cover = rel
        return {"id": p["id"], "name": p.get("name") or p["id"],
                "created_ts": p.get("created_ts"), "note": p.get("note"),
                "counts": counts, "cover": cover}

    def projects_list(self):
        with self.lock:
            lst = [self.project_info(pid) for pid in self.projects]
        lst.sort(key=lambda x: (x["id"] != DEFAULT_PROJECT, -(x.get("created_ts") or 0)))
        return lst

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
            pid = (cfg or {}).get("project") or job["st"].get("project")
            if pid not in self.projects:
                pid = DEFAULT_PROJECT
            if cfg is not None and cfg.get("project") != pid:
                cfg["project"] = pid
                self.persist_cfg(job)
            job["st"]["project"] = pid
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
                          "mode": cfg.get("mode", "ref2v"),
                          "project": cfg.get("project", DEFAULT_PROJECT),
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
        key = stage_key_from_log(text)
        job["st"]["stage"] = {"label": _STAGE_LABELS.get(key) or stage_from_log(text), "key": key,
                              "ts": NOW()}
        detail = None
        for l in text.splitlines():
            if l.startswith("[comfy] "):
                detail = l[len("[comfy] "):].strip()
        if detail:
            job["st"]["detail"] = detail[:160]
        if key != "sampling":
            job["st"].pop("progress", None)
        else:
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
        mode = cfg.get("mode", "ref2v")
        a = [sys.executable, self.driver, "--mode", mode, "--tag", cfg["tag"],
             "--prompt", p["prompt"],
             "--dur", str(p["dur"]), "--aspect", p["aspect"],
             "--megapixels", str(p["megapixels"]), "--multiple", str(p["multiple"]),
             "--steps", str(p["steps"]), "--ref-image-size", p["ref_image_size"],
             "--out", os.path.join(self._out_dir(mode, cfg.get("project")), "%s.mp4" % cfg["tag"])]
        if cfg.get("seed") is not None:
            a += ["--seed", str(cfg["seed"])]
        m = cfg["media"]
        if mode == "ref2v":
            for f in m.get("ref_image", []):
                a += ["--image", f]
            for f in m.get("ref_video", []):
                a += ["--video", f]
            for f in m.get("ref_audio", []):
                a += ["--audio", f]
        else:
            for kind, flag in (("first_frame", "--first-frame"), ("last_frame", "--last-frame")):
                for f in m.get(kind, []):
                    a += [flag, f]
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
            os.makedirs(self._out_dir(cfg.get("mode", "ref2v"), cfg.get("project")), exist_ok=True)
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
                mode = cfg.get("mode", "ref2v")
                p = os.path.join(self._out_dir(mode, cfg.get("project")), "%s.mp4" % cfg["tag"])
                rel = os.path.relpath(p, self.out_root).replace("\\", "/")
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
                "project": st.get("project") or (job.get("cfg") or {}).get("project") or DEFAULT_PROJECT,
                "mode": st.get("mode") or (job.get("cfg") or {}).get("mode") or "ref2v",
                "ended": st.get("ended"), "duration": st.get("duration"),
                "stage": st.get("stage"), "progress": st.get("progress"),
                "detail": st.get("detail"),
                "err": st.get("err"),
                "seed": None if st.get("seed") is None else str(st.get("seed")),
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

    def jobs_list(self, project=None):
        with self.lock:
            lst = [self.info(j) for j in self.jobs.values()
                   if project is None or j["st"].get("project") == project
                   or (j.get("cfg") or {}).get("project") == project]
        lst.sort(key=lambda x: x.get("created") or "", reverse=True)
        return lst

    def delete_outputs(self, rels):
        base = os.path.realpath(self.out_root)
        names, removed = set(), 0
        for rel in rels or []:
            rel = (rel or "").strip()
            cand = os.path.realpath(os.path.join(base, rel)) if rel else base
            if not rel or (cand != base and not cand.startswith(base + os.sep)):
                continue
            if os.path.isfile(cand):
                try:
                    os.remove(cand)
                except OSError:
                    continue
                removed += 1
                names.add(os.path.basename(cand))
        if names:
            with self.lock:
                for job in self.jobs.values():
                    r = job["st"].get("clip_rel")
                    if r and os.path.basename(r) in names:
                        job["st"].pop("clip_rel", None)
                        job["st"].pop("clip_size", None)
                        self.persist_status(job)
        return removed

    def clips_list(self, project=None):
        out_root = self.out_root
        clips = []
        meta = {}
        for job in self.jobs.values():
            rel = job["st"].get("clip_rel")
            if rel:
                meta[os.path.basename(rel)] = self.info(job)
        paths = glob.glob(os.path.join(out_root, "**", "*.mp4"), recursive=True)
        for p in sorted(paths, key=os.path.getmtime, reverse=True):
            rel = os.path.relpath(p, out_root).replace("\\", "/")
            m = meta.get(os.path.basename(p), {})
            pid = m.get("project") or self._project_from_rel(rel)
            if project is not None and pid != project:
                continue
            clips.append({"rel": rel, "name": os.path.basename(p),
                          "size": os.path.getsize(p), "project": pid,
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
            elif u.path == "/api/projects":
                self._json(200, {"projects": mgr.projects_list()})
            elif u.path == "/api/jobs":
                proj = parse_qs(u.query).get("project", [None])[0] or None
                self._json(200, {"jobs": mgr.jobs_list(proj)})
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
                proj = parse_qs(u.query).get("project", [None])[0] or None
                self._json(200, mgr.clips_list(proj))
            elif u.path.startswith("/media/"):
                jid, _, fn = u.path[len("/media/"):].partition("/")
                if not jid or not fn or jid not in mgr.jobs:
                    self._err(404, "not found"); return
                base = os.path.realpath(os.path.join(mgr._job_dir(jid), "uploads"))
                cand = os.path.realpath(os.path.join(base, unquote(fn)))
                if cand != base and not cand.startswith(base + os.sep):
                    self._err(403, "bad path"); return
                if not os.path.isfile(cand):
                    self._err(404, "file not found"); return
                self._send_file_range(cand)
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
            pm = re.match(r"^/api/projects/([^/]+)/(rename|delete)$", u.path)
            if pm:
                pid, action = pm.group(1), pm.group(2)
                try:
                    length = int(self.headers.get("Content-Length") or 0)
                    js = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
                except Exception:
                    self._err(400, "bad json"); return
                if action == "rename":
                    ok, msg = mgr.rename_project(pid, js.get("name"))
                    self._json(200 if ok else 400, {"ok": ok, "msg": msg}); return
                ok, msg = mgr.delete_project(pid, js.get("mode") or "detach")
                self._json(200 if ok else 409, {"ok": ok, "msg": msg}); return
            if u.path == "/api/projects":
                try:
                    length = int(self.headers.get("Content-Length") or 0)
                    js = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
                except Exception:
                    self._err(400, "bad json"); return
                p, err = mgr.create_project(js.get("name"))
                if err:
                    self._err(400, err); return
                self._json(201, mgr.project_info(p["id"])); return
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
                rels = js.get("rels")
                if rels is None:
                    rel = (js.get("rel") or "").strip()
                    rels = [rel] if rel else []
                removed = mgr.delete_outputs(rels)
                if removed:
                    self._json(200, {"ok": True, "removed": removed,
                                     "msg": "已删除 %d 个" % removed}); return
                self._err(404, "没有可删除的文件"); return
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
            mode = _first(fields, "mode", "ref2v")
            if mode not in MODES:
                mode = "ref2v"
            by_kind = {k: [] for k in MEDIA_KINDS}
            frames = {}
            for f in files:
                if f["name"] in by_kind:
                    by_kind[f["name"]].append(f)
                elif f["name"] in FRAME_KINDS:
                    frames[f["name"]] = f
            n = {k: len(v) for k, v in by_kind.items()}
            if mode == "ref2v":
                if n["ref_image"] > MAX_IMAGES:
                    self._err(400, "参考图最多 %d 张" % MAX_IMAGES); return
                if n["ref_video"] > MAX_VIDEOS:
                    self._err(400, "参考视频最多 %d 段" % MAX_VIDEOS); return
                if n["ref_audio"] > MAX_AUDIOS:
                    self._err(400, "参考音频最多 %d 段" % MAX_AUDIOS); return
                if not any(n.values()):
                    self._err(400, "请至少上传一张参考图/视频/音频"); return
                if frames:
                    self._err(400, "参考生视频不支持首/尾帧"); return
            elif any(n.values()):
                self._err(400, "文生视频不支持参考素材"); return

            project = _first(fields, "project", DEFAULT_PROJECT) or DEFAULT_PROJECT
            if project not in mgr.projects:
                self._err(400, "项目不存在"); return
            seed = _to_int(_first(fields, "seed"), None, lo=0, hi=2**63 - 1)
            if seed is None:
                seed = random.randint(0, 2**63 - 1)
            cfg = {
                "mode": mode,
                "project": project,
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
            for kind in FRAME_KINDS:
                f = frames.get(kind)
                if not f:
                    cfg["media"][kind] = []
                    continue
                safe = _safe_name(f["filename"])
                dst = os.path.join(up_dir, "%s_%s" % (kind, safe))
                with open(dst, "wb") as wf:
                    wf.write(f["content"])
                cfg["media"][kind] = [dst]
            mgr.submit(cfg, jid=jid)
            self._json(202, {"id": jid, "status": "queued"})

    return H


HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MiniMax H3</title>
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
main{padding:14px;max-width:1400px;margin:0 auto}
.cols{display:grid;grid-template-columns:minmax(340px,460px) 1fr;gap:14px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px;margin-bottom:14px}
.card h2{font-size:14px;margin:0 0 10px;color:var(--mut);font-weight:600;letter-spacing:.03em}
.cardhead{display:flex;justify-content:space-between;align-items:center;gap:10px;margin-bottom:10px}
.cardhead h2{margin:0}
.cardhead select{width:auto;padding:6px 10px;font-size:13px}
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
.submitrow{display:flex;gap:10px;margin-top:14px}
.submitrow button{flex:1 1 0;margin:0;height:44px;display:flex;align-items:center;justify-content:center}
.submitrow button.primary{width:auto;padding:0}
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
.clip{position:relative;background:#0e1116;border:1px solid var(--line);border-radius:10px;overflow:hidden;cursor:pointer}
.clip video,.clip .ph{width:100%;aspect-ratio:16/9;background:#000;display:block;object-fit:cover}
.clip .cap{padding:6px 8px;font-size:12px;color:var(--mut)}
.clip .pick{position:absolute;top:6px;left:6px;width:22px;height:22px;border-radius:6px;line-height:1;
       background:rgba(0,0,0,.55);border:2px solid #fff;display:flex;align-items:center;justify-content:center;
       font-size:14px;color:#fff}
.clip.sel{outline:3px solid var(--acc);outline-offset:-3px}
.clip.sel .pick{background:var(--acc);border-color:var(--acc)}
.clipbar{display:flex;gap:8px;align-items:center;margin-bottom:10px;flex-wrap:wrap}
.clipbar .muted{margin-right:auto}
details.sec>summary .editbtn{margin-left:auto}
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
.projgrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:12px}
.projcard{background:#0e1116;border:1px solid var(--line);border-radius:10px;overflow:hidden;cursor:pointer;
          transition:border-color .15s}
.projcard:hover{border-color:var(--acc)}
.projcard .cover,.projcard .ph{width:100%;aspect-ratio:16/9;background:#000;display:block;object-fit:cover}
.projcard .ph{display:flex;align-items:center;justify-content:center;color:var(--mut);font-size:12px}
.projcard .body{padding:8px 10px}
.projcard .nm{font-weight:600;margin-bottom:2px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.projcard .meta2{font-size:12px;color:var(--mut)}
.bcbar{display:flex;align-items:center;gap:8px;flex-wrap:wrap}
.mediagrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(120px,1fr));gap:8px;margin-top:6px}
.mediagrid img,.mediagrid video{width:100%;aspect-ratio:16/9;object-fit:cover;background:#000;
       border-radius:8px;border:1px solid var(--line)}
.medialist{display:block;margin-top:6px}
.medialist audio{width:100%;display:block;margin-top:6px}
.detprompt{background:#0b0e13;border:1px solid var(--line);border-radius:8px;padding:8px;font-size:13px;
       white-space:pre-wrap;word-break:break-word;max-height:200px;overflow:auto;line-height:1.55}
.detrow{display:flex;gap:10px;font-size:13px;padding:2px 0}
.detrow .muted{min-width:52px}
@media(max-width:980px){.cols{grid-template-columns:1fr}}
@media(max-width:640px){.grid3{grid-template-columns:1fr 1fr}textarea,input,select{font-size:16px}}
</style>
</head>
<body>
<header>
  <h1 style="cursor:pointer" onclick="goHome()" title="全部项目">MiniMax H3</h1>
  <span id="pComfy" class="pill off">ComfyUI ?</span>
  <span id="pVram" class="pill">VRAM --</span>
  <span id="pQ" class="pill">队列 0</span>
  <span class="spacer"></span>
  <button class="ghost" onclick="releaseVram()">释放显存</button>
</header>
<main>
  <div id="homeView">
    <div class="card">
      <div class="cardhead"><h2>新建项目</h2></div>
      <div style="display:flex;gap:8px">
        <input id="newProjName" placeholder="如：超人大战蝙蝠侠" onkeydown="if(event.key==='Enter')newProject()">
        <button class="ghost" style="flex:0 0 auto" onclick="newProject()">创建</button>
      </div>
      <div class="muted" id="projMsg" style="margin-top:8px"></div>
    </div>
    <div class="card">
      <h2>项目</h2>
      <div id="projList" class="projgrid"><span class="muted">加载中…</span></div>
    </div>
  </div>
  <div id="projView" style="display:none">
    <div class="card" id="projHead"></div>
    <div class="cols">
      <div>
        <div class="card">
          <div class="cardhead">
            <h2>新建任务</h2>
        <select id="mode" onchange="onModeChange()">
          <option value="t2v">文生视频</option>
          <option value="ref2v">参考生视频</option>
        </select>
      </div>
      <label id="promptLabel"></label>
      <textarea id="prompt" placeholder="Cinematic shot of the subject ..."></textarea>
      <div style="margin-top:6px"><button class="ghost" id="optBtn" onclick="optimizePrompt()">提示词优化</button></div>
      <div id="t2vBox">
        <label>首帧（可选，单张图片）</label>
        <div class="filebox"><input id="fFirst" type="file" accept="image/*"><ul id="lFirst"></ul></div>
        <label>尾帧（可选，单张图片）</label>
        <div class="filebox"><input id="fLast" type="file" accept="image/*"><ul id="lLast"></ul></div>
      </div>
      <div id="ref2vBox">
        <label>参考图（≤9）</label>
        <div class="filebox"><input id="fImg" type="file" accept="image/*" multiple><ul id="lImg"></ul></div>
        <label>参考视频（≤3，每段 2–15s，合计 ≤15s）</label>
        <div class="filebox"><input id="fVid" type="file" accept="video/*" multiple><ul id="lVid"></ul></div>
        <label>参考音频（≤3，合计 ≤15s）</label>
        <div class="filebox"><input id="fAud" type="file" accept="audio/*" multiple><ul id="lAud"></ul></div>
      </div>
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
      <div id="refImageSizeRow"><label>参考图缩放</label>
        <select id="ref_image_size"><option value="match">match(快)</option><option value="max">max(保真)</option></select></div>
      <div class="submitrow">
        <button class="primary" id="submitBtn" onclick="submit()">提交</button>
        <button class="ghost" onclick="resetForm()">重置</button>
      </div>
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
        <summary>产物<button class="ghost editbtn" id="clipEditBtn" onclick="event.preventDefault();event.stopPropagation();toggleClipEdit()">编辑</button></summary>
        <div id="clipBar" class="clipbar" style="display:none">
          <span class="muted" id="clipCount">已选 0</span>
          <button class="ghost" onclick="clipSelectAll()">全选</button>
          <button class="ghost" onclick="clipSelectNone()">取消全选</button>
          <button class="ghost" style="color:var(--err)" onclick="clipDelete()">删除</button>
        </div>
        <div id="clips" class="clips"></div>
      </details>
    </div>
  </div>
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
<div class="modal" id="delModal" onclick="if(event.target===this)closeDel()">
  <div class="box" style="width:min(420px,96vw)">
    <div class="optrow"><b>删除项目</b><button class="ghost" onclick="closeDel()">关闭</button></div>
    <div class="muted" id="delMsg" style="margin-bottom:12px"></div>
    <label style="display:flex;align-items:center;gap:8px;margin:0;color:var(--fg);font-size:14px">
      <input id="delClips" type="checkbox" checked style="width:auto">同时删除产物
    </label>
    <div class="muted" id="delHint" style="margin-top:10px"></div>
    <div class="optacts">
      <button class="ghost" onclick="closeDel()">取消</button>
      <button class="primary" onclick="confirmDeleteProject()">删除</button>
    </div>
  </div>
</div>
<div class="modal" id="askModal" onclick="if(event.target===this)askResolve(false)">
  <div class="box" style="width:min(420px,96vw)">
    <div class="optrow"><b id="askTitle">确认</b></div>
    <div class="muted" id="askMsg" style="margin-bottom:14px"></div>
    <div class="optacts">
      <button class="ghost" onclick="askResolve(false)">取消</button>
      <button class="primary" id="askOk" onclick="askResolve(true)">确定</button>
    </div>
  </div>
</div>
<div class="modal" id="inputModal" onclick="if(event.target===this)inputResolve(null)">
  <div class="box" style="width:min(420px,96vw)">
    <div class="optrow"><b id="inputTitle">输入</b><button class="ghost" onclick="inputResolve(null)">关闭</button></div>
    <input id="inputVal" onkeydown="if(event.key==='Enter'){event.preventDefault();inputResolve($('inputVal').value)}">
    <div class="optacts">
      <button class="ghost" onclick="inputResolve(null)">取消</button>
      <button class="primary" id="inputOk" onclick="inputResolve($('inputVal').value)">确定</button>
    </div>
  </div>
</div>
<div class="modal" id="jobModal" onclick="if(event.target===this)closeJob()">
  <div class="box" style="width:min(720px,96vw);max-height:88vh;overflow:auto">
    <div class="optrow"><b id="jTitle">任务详情</b><button class="ghost" onclick="closeJob()">关闭</button></div>
    <div id="jBody"></div>
  </div>
</div>
<script>
const $ = (id)=>document.getElementById(id);
const esc = (s)=>(s||'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const ASPECTS = __ASPECTS__;
let logOffset = 0, lastJob = null, jobsById = {}, curStart = 0, curRunning = false;
let projects = [], projNames = {}, curProject = null, projectsLoaded = false;
let clipEdit = false, clipSel = new Set(), clipsCache = [];
const STATUS_CN = {queued:'排队中', running:'进行中', done:'已完成', failed:'失败', cancelled:'已取消', interrupted:'已中断'};
const MEDIA_CN = {ref_image:'图', ref_video:'视频', ref_audio:'音频'};
const MODE_CN = {t2v:'文生视频', ref2v:'参考生视频'};
function projName(pid){ return projNames[pid] || (pid==='default'?'默认项目':(pid||'')); }
function projSub(p){
  return [p.counts.total+' 个任务',
          p.counts.running?p.counts.running+' 进行中':null,
          p.counts.queued?p.counts.queued+' 排队':null].filter(Boolean).join(' · ');
}
async function refreshProjects(){
  const r=await api('/api/projects'); if(!r) return;
  projects=r.projects||[]; projNames={};
  projects.forEach(p=>projNames[p.id]=p.name);
  projectsLoaded=true;
  const box=$('projList');
  const sig=projects.map(p=>[p.id,p.name,p.counts.total,p.counts.running,p.counts.queued,p.cover||''].join(',')).join('\n');
  if(box._sig!==sig){
    box._sig=sig;
    box.innerHTML='';
    if(!projects.length) box.innerHTML='<span class="muted">暂无项目</span>';
    projects.forEach(p=>{
      const d=document.createElement('div'); d.className='projcard'; d.onclick=()=>openProject(p.id);
      const cover=(p.cover && p.cover.slice(-4)==='.mp4')
        ? '<video class="cover" preload="metadata" muted playsinline src="/files/'+encodeURI(p.cover)+'#t=0.1"></video>'
        : '<div class="ph">暂无产物</div>';
      d.innerHTML=cover+'<div class="body"><div class="nm">'+esc(p.name)+'</div>'+
        '<div class="meta2">'+projSub(p)+'</div></div>';
      box.appendChild(d);
    });
  }
  if(curProject) renderProjHead();
}
function renderProjHead(){
  const p=projects.find(x=>x.id===curProject); if(!p){ return; }
  const canEdit=(curProject!=='default');
  $('projHead').innerHTML='<div class="cardhead"><h2 style="color:var(--fg);font-size:16px">'+esc(p.name)+'</h2>'+
    '<span class="bcbar">'+(canEdit?'<button class="ghost" onclick="renameProject()">改名</button>'+
      '<button class="ghost" onclick="deleteProject()">删除</button>':'')+
      '<button class="ghost" onclick="goHome()">全部项目</button></span></div>'+
    '<div class="muted">'+projSub(p)+'</div>';
}
function openProject(pid){ location.hash='#/p/'+encodeURIComponent(pid); }
function goHome(){ location.hash='#/'; }
function route(){
  const m=location.hash.match(/^#\/p\/(.+)$/);
  const pid=m?decodeURIComponent(m[1]):null;
  if(pid && projNames[pid]!==undefined){
    if(curProject!==pid){ curProject=pid;
      $('jobs')._sig=null; $('jobs').innerHTML='';
      $('clips')._sig=null; $('clips').innerHTML='';
      jobsById={}; clipsCache=[]; clipSel.clear(); lastJob=null; logOffset=0; $('log').textContent='';
      if(clipEdit){ clipEdit=false; $('clipBar').style.display='none'; $('clipEditBtn').textContent='编辑'; } }
    $('homeView').style.display='none'; $('projView').style.display='';
    renderProjHead(); refreshJobs(); refreshOutputs();
  }else{
    curProject=null;
    $('homeView').style.display=''; $('projView').style.display='none';
  }
}
window.addEventListener('hashchange',route);
async function newProject(){
  const name=$('newProjName').value.trim();
  if(!name){ $('projMsg').textContent='请填写项目名称'; return; }
  const r=await fetch('/api/projects',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({name})});
  const j=await r.json().catch(()=>({}));
  if(r.status===201){ $('newProjName').value=''; $('projMsg').textContent='已创建：'+j.name;
    await refreshProjects(); openProject(j.id); }
  else $('projMsg').textContent='创建失败：'+(j.error||r.status);
}
async function renameProject(){
  const p=projects.find(x=>x.id===curProject); if(!p) return;
  const name=await askInput('重命名项目',p.name,'保存','项目名称');
  if(name==null) return;
  const r=await fetch('/api/projects/'+encodeURIComponent(curProject)+'/rename',{method:'POST',
    headers:{'Content-Type':'application/json'},body:JSON.stringify({name})});
  const j=await r.json().catch(()=>({}));
  if(!r.ok){ alert('改名失败：'+(j.error||j.msg||r.status)); return; }
  await refreshProjects();
}
let delPid=null;
function updDelHint(){
  $('delHint').textContent = $('delClips').checked
    ? '将同时删除该项目下的任务记录与视频文件，不可恢复。'
    : '不删除产物：项目内任务将转为「默认项目」，视频文件保留。';
}
function deleteProject(){
  const p=projects.find(x=>x.id===curProject); if(!p) return;
  delPid=p.id;
  $('delMsg').innerHTML='确定删除项目「<b>'+esc(p.name)+'</b>」？';
  $('delClips').checked=true; updDelHint();
  $('delModal').classList.add('open');
}
function closeDel(){ $('delModal').classList.remove('open'); delPid=null; }
async function confirmDeleteProject(){
  if(!delPid) return;
  const mode=$('delClips').checked?'purge':'detach';
  const pid=delPid;
  const r=await fetch('/api/projects/'+encodeURIComponent(pid)+'/delete',{method:'POST',
    headers:{'Content-Type':'application/json'},body:JSON.stringify({mode})});
  const j=await r.json().catch(()=>({}));
  closeDel();
  if(!r.ok){ alert('删除失败：'+(j.error||j.msg||r.status)); return; }
  goHome(); await refreshProjects();
}
$('delClips').onchange=updDelHint;
let askCb=null;
function askConfirm(msg,title,okText){
  return new Promise(res=>{
    askCb=res;
    $('askTitle').textContent=title||'确认';
    $('askMsg').innerHTML=msg;
    $('askOk').textContent=okText||'确定';
    $('askModal').classList.add('open');
  });
}
function askResolve(v){ $('askModal').classList.remove('open'); const cb=askCb; askCb=null; if(cb) cb(v); }
let inputCb=null;
function askInput(title,value,okText,ph){
  return new Promise(res=>{
    inputCb=res;
    $('inputTitle').textContent=title||'输入';
    $('inputOk').textContent=okText||'确定';
    $('inputVal').value=value||'';
    $('inputVal').placeholder=ph||'';
    $('inputModal').classList.add('open');
    setTimeout(()=>{ $('inputVal').focus(); $('inputVal').select(); },50);
  });
}
function inputResolve(v){ $('inputModal').classList.remove('open'); const cb=inputCb; inputCb=null; if(cb) cb(v); }
function mediaUrl(jid,p){ return '/media/'+encodeURIComponent(jid)+'/'+encodeURIComponent(String(p).split('/').pop()); }
function mediaSection(jid,m){
  const groups=[['ref_image','参考图','img'],['ref_video','参考视频','video'],
                ['ref_audio','参考音频','audio'],['first_frame','首帧','img'],['last_frame','尾帧','img']];
  let h='';
  groups.forEach(([k,label,kind])=>{
    const arr=m[k]||[]; if(!arr.length) return;
    h+='<div style="margin-top:12px"><div class="muted">'+label+' ('+arr.length+')</div>'+
       '<div class="'+(kind==='audio'?'medialist':'mediagrid')+'">';
    arr.forEach(p=>{
      const u=mediaUrl(jid,p);
      if(kind==='img') h+='<img src="'+u+'" loading="lazy">';
      else if(kind==='video') h+='<video controls preload="metadata" playsinline src="'+u+'"></video>';
      else h+='<audio controls preload="metadata" src="'+u+'"></audio>';
    });
    h+='</div></div>';
  });
  return h;
}
function bindSinglePlay(root){
  const els=root.querySelectorAll('video,audio');
  els.forEach(el=>el.addEventListener('play',()=>{
    els.forEach(o=>{ if(o!==el && !o.paused) o.pause(); });
  }));
}
function showJob(id){
  const j=jobsById[id]; if(!j) return;
  const p=j.params||{}, m=j.media||{};
  $('jTitle').textContent='任务详情 · '+id;
  const row=(k,v)=>'<div class="detrow"><span class="muted">'+k+'</span><span>'+v+'</span></div>';
  let h='';
  h+=row('类型', MODE_CN[j.mode]||j.mode||'-');
  h+=row('时长', p.dur!=null? p.dur+' 秒':'-');
  h+=row('步数', p.steps!=null? p.steps:'-');
  h+=row('seed', j.seed!=null? String(j.seed):'-');
  h+=row('画幅', esc(p.aspect||'-'));
  h+=row('分辨率', p.megapixels!=null? p.megapixels+' MP':'-');
  if(j.status==='failed' && j.err) h+=row('错误', '<span style="color:var(--err)">'+esc(friendlyErr(j.err))+'</span>');
  h+='<div style="margin-top:12px"><div class="muted">提示词</div><div class="detprompt">'+esc(p.prompt||'')+'</div></div>';
  h+=mediaSection(id,m);
  $('jBody').innerHTML=h;
  bindSinglePlay($('jBody'));
  $('jobModal').classList.add('open');
}
function closeJob(){
  $('jBody').querySelectorAll('video,audio').forEach(o=>o.pause());
  $('jobModal').classList.remove('open');
}
async function reuseJob(id){
  const j=jobsById[id]; if(!j) return;
  if(!curProject){ alert('请先进入一个项目'); return; }
  const p=j.params||{}, m=j.media||{};
  $('submitMsg').textContent='正在载入任务 '+id+' 的素材…';
  $('mode').value=(j.mode==='ref2v')?'ref2v':'t2v'; onModeChange();
  $('prompt').value=p.prompt||'';
  if(p.dur!=null) $('dur').value=p.dur;
  if(p.steps!=null) $('steps').value=p.steps;
  if(p.aspect) $('aspect').value=p.aspect;
  if(p.megapixels!=null) $('megapixels').value=p.megapixels;
  if(p.ref_image_size) $('ref_image_size').value=p.ref_image_size;
  $('seed').value='';   // 复用不沿用原 seed，留空=随机，避免复现成同样的视频
  getImg.set([]); getVid.set([]); getAud.set([]); getFirst.set(null); getLast.set(null);
  const load=async(kind,setter)=>{
    const files=[];
    for(const path of (m[kind]||[])){
      try{
        const b=await (await fetch(mediaUrl(id,path))).blob();
        files.push(new File([b],String(path).split('/').pop(),{type:b.type||''}));
      }catch(e){}
    }
    if(kind==='first_frame'||kind==='last_frame') setter.set(files[0]||null); else setter.set(files);
  };
  await load('ref_image',getImg); await load('ref_video',getVid); await load('ref_audio',getAud);
  await load('first_frame',getFirst); await load('last_frame',getLast);
  window.scrollTo({top:0,behavior:'smooth'});
  $('submitMsg').textContent='已复用任务 '+id+'（未提交）';
}
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
  const parts=Object.keys(MEDIA_CN).map(k=> (m[k]&&m[k].length)? m[k].length+MEDIA_CN[k] : null);
  if(m.first_frame&&m.first_frame.length) parts.push('首帧');
  if(m.last_frame&&m.last_frame.length) parts.push('尾帧');
  return parts.filter(Boolean).join(' · ');
}
function friendlyStatus(j){
  if(j.status==='running'){
    const lab = (j.stage&&j.stage.label) || '进行中';
    const p = j.progress;
    return p? (lab+' '+p.cur+'/'+p.total) : lab;
  }
  if(j.status==='done'){
    const parts=['已完成'];
    if(j.clip_seconds) parts.push('视频 '+j.clip_seconds+'s');
    if(j.duration!=null) parts.push('用时 '+fmtDur(j.duration));
    return parts.join(' · ');
  }
  if(j.status==='failed') return '失败：'+friendlyErr(j.err)+(j.duration!=null?' · 用时 '+fmtDur(j.duration):'');
  if(j.status==='cancelled') return '已取消'+(j.duration!=null?' · 用时 '+fmtDur(j.duration):'');
  if(j.status==='interrupted') return '已中断(web 重启)，请重新提交';
  return STATUS_CN[j.status]||j.status;
}
function fmtDur(sec){ sec=Math.max(0,Math.floor(sec)); return String(Math.floor(sec/60)).padStart(2,'0')+':'+String(sec%60).padStart(2,'0'); }
function renderCurrent(j){
  curStart = j.created_ts || curStart || 0;
  curRunning = (j.status==='running'||j.status==='queued');
  const sig=[j.id,j.status,friendlyStatus(j),j.mode||'',j.project||'',j.clip_rel||'',j.err||'',j.detail||'',mediaBrief(j.media)].join('|');
  if($('cur')._sig===sig) return;
  $('cur')._sig=sig;
  const p=j.params||{};
  const meta=[projName(j.project), MODE_CN[j.mode]||'', mediaBrief(j.media), p.dur?p.dur+'s':'', p.aspect?p.aspect.split(' ')[0]:'',
              p.megapixels?p.megapixels+'MP':'', p.steps?p.steps+'步':'', j.seed?('seed '+j.seed):''].filter(Boolean).join(' · ');
  let bar='';
  if(j.status==='running' && j.progress && j.progress.total){
    bar='<div class="bar"><i style="width:'+Math.round(j.progress.cur/j.progress.total*100)+'%"></i></div>';
  }else if(j.status==='running'||j.status==='queued'){
    bar='<div class="bar indet"><i></i></div>';
  }
  const cls=(j.status==='failed')?' style="color:var(--err)"':'';
  const detail=(curRunning&&j.detail)? '<span class="muted">'+esc(j.detail)+'</span>' : '';
  $('cur').innerHTML='<div class="curState"'+cls+'>'+friendlyStatus(j)+'</div>'+bar+
    '<div class="curMeta"><b>'+j.id+'</b>'+(meta?'<br>'+meta:'')+(detail?'<br>'+detail:'')+
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
  const get=()=>files;
  get.set=(arr)=>{ files=(arr||[]).slice(); render(); };
  return get;
}
function bindOne(inputId, listId){
  const input=$(inputId), list=$(listId); let file=null;
  function render(){
    list.innerHTML='';
    if(!file) return;
    const li=document.createElement('li');
    const nm=document.createElement('span'); nm.className='fname'; nm.textContent=file.name;
    const acts=document.createElement('span'); acts.className='acts';
    const rm=document.createElement('button'); rm.className='rm'; rm.textContent='移除';
    rm.onclick=()=>{file=null; input.value=''; render();};
    acts.appendChild(rm); li.appendChild(nm); li.appendChild(acts); list.appendChild(li);
  }
  input.onchange=()=>{ file=input.files[0]||null; input.value=''; render(); };
  const get=()=>file;
  get.set=(f)=>{ file=f||null; render(); };
  return get;
}
const getImg = bindFiles('fImg','lImg',9,'图片','Picture');
const getVid = bindFiles('fVid','lVid',3,'视频','Video');
const getAud = bindFiles('fAud','lAud',3,'音频','Audio');
const getFirst = bindOne('fFirst','lFirst');
const getLast = bindOne('fLast','lLast');

function onModeChange(){
  const m=$('mode').value, ref=(m==='ref2v');
  $('t2vBox').style.display = ref? 'none':'';
  $('ref2vBox').style.display = ref? '':'none';
  $('refImageSizeRow').style.display = ref? '':'none';
  $('promptLabel').textContent = ref
    ? '提示词（点下方素材后的「图片1 / 视频1 / 音频1」按钮即可插入 <Picture 1> 等引用）'
    : '提示词（可选：上传首帧/尾帧；都不传即纯文生视频）';
}

async function submit(){
  if(!curProject){ alert('请先进入一个项目'); goHome(); return; }
  const mode=$('mode').value;
  const prompt=$('prompt').value.trim();
  if(!prompt){ alert('请填写提示词'); return; }
  const dur=$('dur').value, steps=$('steps').value;
  let media='';
  if(mode==='ref2v'){
    const imgs=getImg(), vids=getVid(), auds=getAud();
    if(!imgs.length && !vids.length && !auds.length){ alert('请至少上传一个参考素材'); return; }
    media=[imgs.length?imgs.length+'图':null, vids.length?vids.length+'视频':null,
           auds.length?auds.length+'音频':null].filter(Boolean).join(' / ');
  }else{
    const ff=getFirst(), lf=getLast();
    media=[ff?'首帧':null, lf?'尾帧':null].filter(Boolean).join(' + ');
  }
  const ok=await askConfirm(
    '项目：'+esc(projName(curProject))+'<br>类型：'+(MODE_CN[mode]||mode)+
    '<br>时长：'+dur+'s · 步数：'+steps+(media?'<br>素材：'+media:'')+
    '<br>提示词：'+esc(prompt.slice(0,100))+(prompt.length>100?'…':''),
    '提交任务','提交');
  if(!ok) return;
  const fd=new FormData();
  fd.append('project', curProject);
  fd.append('mode', mode);
  fd.append('prompt', prompt);
  fd.append('dur', $('dur').value); fd.append('steps', $('steps').value);
  fd.append('aspect', $('aspect').value); fd.append('megapixels', $('megapixels').value);
  fd.append('ref_image_size', $('ref_image_size').value);
  if($('seed').value) fd.append('seed', $('seed').value);
  if(mode==='ref2v'){
    const imgs=getImg(), vids=getVid(), auds=getAud();
    imgs.forEach(f=>fd.append('ref_image', f));
    vids.forEach(f=>fd.append('ref_video', f));
    auds.forEach(f=>fd.append('ref_audio', f));
  }else{
    const ff=getFirst(), lf=getLast();
    if(ff) fd.append('first_frame', ff);
    if(lf) fd.append('last_frame', lf);
  }
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

async function resetForm(){
  const ok=await askConfirm('清空当前填写的内容并恢复默认参数？','重置','清空');
  if(!ok) return;
  $('mode').value='t2v'; onModeChange();
  $('prompt').value='';
  $('dur').value=5; $('steps').value=8; $('seed').value='';
  $('aspect').value=ASPECTS[0]; $('megapixels').value='0.4'; $('ref_image_size').value='match';
  getImg.set([]); getVid.set([]); getAud.set([]); getFirst.set(null); getLast.set(null);
  $('prog').style.display='none'; $('submitMsg').textContent='';
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
  if(!curProject) return;
  let j=s.current;
  if(j && j.project!==curProject) j=null;            // 只显示本项目任务
  if(!j) j=(lastJob && jobsById[lastJob]) || null;
  if(j && j.project!==curProject) j=null;
  if(j){
    if(lastJob!==j.id){ lastJob=j.id; logOffset=0; $('log').textContent=''; }
    renderCurrent(j);
  }else{
    lastJob=null; renderCurrentEmpty();
  }
}
function renderCurrentEmpty(){
  if($('cur')._sig==='EMPTY') return;
  $('cur')._sig='EMPTY'; $('cur').innerHTML='<div class="muted">空闲</div>';
}

async function pollLog(){
  if(!lastJob) return;
  const r=await api('/api/jobs/'+lastJob+'/log?offset='+logOffset);
  if(!r) return;
  if(r.text){ $('log').textContent += r.text; $('log').scrollTop=$('log').scrollHeight; }
  logOffset=r.offset;
}

function updateUsedElapsed(){
  const now=Date.now()/1000;
  document.querySelectorAll('#jobs [data-used]').forEach(el=>{
    const ts=parseFloat(el.getAttribute('data-used'));
    if(ts) el.textContent='用时 '+fmtDur(now-ts);
  });
}

async function refreshJobs(){
  if(!curProject) return;
  const r=await api('/api/jobs?project='+encodeURIComponent(curProject)); if(!r) return;
  const box=$('jobs');
  const jobs=r.jobs.slice(0,50);
  const sig=jobs.map(j=>[j.id,j.status,(j.stage&&j.stage.label)||'',
    (j.progress&&j.progress.cur)||'',(j.progress&&j.progress.total)||'',
    j.duration!=null?j.duration:'',j.clip_rel||'',j.err||'',j.mode||''].join(',')).join('\n');
  if(box._sig===sig){ updateUsedElapsed(); return; }
  box._sig=sig;
  if(!jobs.length){ box.innerHTML='<span class="muted">暂无</span>'; return; }
  box.innerHTML='';
  jobs.forEach(j=>{
    jobsById[j.id]=j;
    const d=document.createElement('div'); d.className='job';
    const p=j.params||{};
    const info=[mediaBrief(j.media), p.dur?p.dur+'s':'', p.megapixels?p.megapixels+'MP':''].filter(Boolean).join(' · ');
    const used = (j.duration!=null)? j.duration : (j.created_ts? Math.max(0, Date.now()/1000-j.created_ts) : null);
    const usedTxt = (used!=null)? (j.duration!=null? '用时 '+fmtDur(used)
                                 : '<span data-used="'+j.created_ts+'">用时 '+fmtDur(used)+'</span>') : '';
    const line2 = [MODE_CN[j.mode]||'', info, usedTxt].filter(Boolean).join(' · ');
    const note = j.status==='failed'? '<span style="color:var(--err)">'+friendlyErr(j.err)+'</span>' : (j.stage&&j.status==='running'? j.stage.label : '');
    let acts='<button class="ghost" onclick="showJob(\''+j.id+'\')">详情</button> ';
    if(j.status==='queued'||j.status==='running') acts+='<button class="ghost" onclick="jobAct(\''+j.id+'\',\'cancel\')">取消</button>';
    else acts+='<button class="ghost" onclick="jobAct(\''+j.id+'\',\'delete\')">删除</button>';
    if(j.clip_rel) acts+=' <button class="ghost" onclick="play(\''+j.clip_rel+'\')">查看</button>';
    acts+=' <button class="ghost" onclick="reuseJob(\''+j.id+'\')">复用</button>';
    d.innerHTML='<span class="'+stCls(j.status)+'">'+(STATUS_CN[j.status]||j.status)+'</span>'+
      '<span class="meta"><b>'+j.id+'</b><br>'+line2+'<br>'+(note||j.created||'')+'</span>'+acts;
    box.appendChild(d);
  });
}

async function jobAct(id,act){
  const ok=await askConfirm((act==='cancel'?'取消任务 ':'删除记录 ')+'<b>'+id+'</b>？',
                            act==='cancel'?'取消任务':'删除记录', act==='cancel'?'取消':'删除');
  if(!ok) return;
  fetch('/api/jobs/'+id+'/'+act,{method:'POST'}).then(()=>{refreshJobs();refreshOutputs();}); }

async function refreshOutputs(){
  if(!curProject) return;
  const r=await api('/api/outputs?project='+encodeURIComponent(curProject)); if(!r) return;
  clipsCache=r.clips||[];
  const box=$('clips');
  const sig=clipsCache.map(c=>c.rel+'|'+c.size).join('\n');
  if(box._sig===sig) return;
  box._sig=sig;
  if(!clipsCache.length){ box.innerHTML='<span class="muted">暂无产物</span>'; return; }
  box.innerHTML='';
  clipsCache.forEach(c=>{
    const sel=clipSel.has(c.rel);
    const d=document.createElement('div'); d.className='clip'+(sel?' sel':''); d.dataset.rel=c.rel;
    d.onclick=()=>{ if(clipEdit) toggleClip(c.rel,d); else play(c.rel); };
    const pick=clipEdit? '<span class="pick">'+(sel?'\u2713':'')+'</span>' : '';
    d.innerHTML=pick+'<video preload="metadata" muted playsinline src="/files/'+encodeURI(c.rel)+'"></video>'+
      '<div class="cap">'+c.name.slice(0,20)+'<br>'+fmtSize(c.size)+' · '+c.ts+'</div>';
    box.appendChild(d);
  });
}

function updClipBar(){ $('clipCount').textContent='已选 '+clipSel.size; }
function applyClipEditUI(){
  $('clips').querySelectorAll('.clip').forEach(el=>{
    if(clipEdit){
      if(!el.querySelector('.pick')){
        const pk=document.createElement('span'); pk.className='pick';
        el.insertBefore(pk, el.firstChild);
      }
      const on=clipSel.has(el.dataset.rel);
      el.classList.toggle('sel',on);
      el.querySelector('.pick').textContent=on?'\u2713':'';
    }else{
      const pk=el.querySelector('.pick'); if(pk) pk.remove();
      el.classList.remove('sel');
    }
  });
}
function toggleClipEdit(){
  clipEdit=!clipEdit; clipSel.clear();
  $('clipBar').style.display=clipEdit?'flex':'none';
  $('clipEditBtn').textContent=clipEdit?'完成':'编辑';
  applyClipEditUI(); updClipBar();
}
function toggleClip(rel,el){
  if(clipSel.has(rel)) clipSel.delete(rel); else clipSel.add(rel);
  el.classList.toggle('sel',clipSel.has(rel));
  const pk=el.querySelector('.pick'); if(pk) pk.textContent=clipSel.has(rel)?'\u2713':'';
  updClipBar();
}
function syncClipSel(){
  $('clips').querySelectorAll('.clip').forEach(el=>{
    const on=clipSel.has(el.dataset.rel);
    el.classList.toggle('sel',on);
    const pk=el.querySelector('.pick'); if(pk) pk.textContent=on?'\u2713':'';
  });
  updClipBar();
}
function clipSelectAll(){ clipsCache.forEach(c=>clipSel.add(c.rel)); syncClipSel(); }
function clipSelectNone(){ clipSel.clear(); syncClipSel(); }
function removeClipsLocal(rels){
  const gone=new Set(rels);
  $('clips').querySelectorAll('.clip').forEach(el=>{ if(gone.has(el.dataset.rel)) el.remove(); });
  clipsCache=clipsCache.filter(c=>!gone.has(c.rel));
  const box=$('clips');
  if(!clipsCache.length) box.innerHTML='<span class="muted">暂无产物</span>';
  box._sig=clipsCache.map(c=>c.rel+'|'+c.size).join('\n');
}
async function clipDelete(){
  const rels=[...clipSel];
  if(!rels.length){ alert('请先选择要删除的产物'); return; }
  const ok=await askConfirm('确定删除选中的 <b>'+rels.length+'</b> 个产物？删除后不可恢复。','删除产物','删除');
  if(!ok) return;
  const r=await fetch('/api/output/delete',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({rels})});
  const j=await r.json().catch(()=>({}));
  if(!r.ok){ alert('删除失败：'+(j.error||r.status)); return; }
  clipSel.clear(); updClipBar();
  removeClipsLocal(rels);
  if(clipEdit) toggleClipEdit();
  refreshProjects(); refreshJobs();
}

function play(rel){ $('mvideo').src='/files/'+encodeURI(rel); $('mcap').textContent=rel;
  $('modal').classList.add('open'); $('mvideo').play().catch(()=>{}); }
function closeModal(){ $('mvideo').pause(); $('mvideo').src=''; $('modal').classList.remove('open'); }

async function releaseVram(){
  const ok=await askConfirm('停止 ComfyUI 并释放显存？下次生成需冷启动。','释放显存','停止');
  if(!ok) return;
  fetch('/api/service/stop',{method:'POST'}).then(async r=>{ const j=await r.json(); alert(j.msg||'ok'); refreshState(); }); }

async function optimizePrompt(){
  const prompt=$('prompt').value.trim();
  if(!prompt){ $('submitMsg').textContent='请先填写提示词'; return; }
  const counts = $('mode').value==='ref2v'
    ? {ref_image:getImg().length, ref_video:getVid().length, ref_audio:getAud().length}
    : {};
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

onModeChange();
(async()=>{ await refreshProjects(); route(); })();
refreshState();
setInterval(refreshState,2000); setInterval(refreshProjects,5000);
setInterval(refreshJobs,5000); setInterval(refreshOutputs,5000); setInterval(pollLog,1500);
</script>
</body>
</html>"""


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="LAN web console for minimax_h3_runner.py")
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
        os.path.dirname(os.path.abspath(__file__)), "minimax_h3_runner.py"))
    data = os.path.join(root, ".h3ref2v")
    os.makedirs(data, exist_ok=True)
    pidfile = os.path.join(data, "minimax_h3_web.pid")
    logfile = os.path.join(root, "minimax_h3_web.log")

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
    print("MiniMax H3 Web on http://%s:%d/ (driver=%s)" % (a.host, a.port, driver))
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
