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
MAX_REFS = 12                       # 全部参考文件（图+视频+音频）合计上限
MAX_IMAGE_MB, MAX_VIDEO_MB, MAX_AUDIO_MB, MAX_REQUEST_MB = 30, 50, 15, 64
MEDIA_KINDS = ("ref_image", "ref_video", "ref_audio")
FRAME_KINDS = ("first_frame", "last_frame")
MATERIAL_EXTS = {".png": "image", ".jpg": "image", ".jpeg": "image", ".webp": "image",
                 ".bmp": "image", ".gif": "image",
                 ".mp4": "video", ".mov": "video", ".webm": "video", ".mkv": "video",
                 ".avi": "video",
                 ".mp3": "audio", ".wav": "audio", ".m4a": "audio", ".aac": "audio",
                 ".flac": "audio", ".ogg": "audio"}
MATERIAL_KIND_CN = {"image": "图片", "video": "视频", "audio": "音频"}
MATERIAL_NAME_MAX = 60
THUMB_MAX = 480
PREVIEW_MAX = 1600
THUMB_HELPER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "make_thumb.py")
VTHUMB_HELPER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "make_vthumb.py")
MODES = ("t2v", "ref2v")
OUT_SUBDIRS = {"t2v": "t2v", "ref2v": "ref2v", "edit": "edit"}
TRANSITIONS = ("cut", "fade", "dissolve", "push")
EDIT_ASPECT_OPTS = [("0", "原始画幅"), ("2.39", "2.39:1 宽银幕"), ("16:9", "16:9 横屏"),
                    ("9:16", "9:16 竖屏"), ("1:1", "1:1 方形"), ("4:3", "4:3 横版"),
                    ("3:4", "3:4 竖版")]
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


def aspect_ratio(s):
    """'2.39' / '16:9' / '0' (or '原始') -> a positive w/h float, 0 = source."""
    s = str(s or "0").strip().split(" ")[0]
    try:
        if ":" in s:
            a, b = s.split(":", 1)
            return max(0.0, float(a) / float(b))
        return max(0.0, float(s))
    except (ValueError, ZeroDivisionError):
        return 0.0


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

    def interrupt(self):
        """Ask ComfyUI to interrupt the current prompt and clear its pending queue."""
        try:
            req = urllib.request.Request(self.base + "/interrupt", data=b"", method="POST")
            urllib.request.urlopen(req, timeout=3).read()
        except Exception:
            pass
        try:
            data = json.dumps({"clear": True}).encode("utf-8")
            req = urllib.request.Request(self.base + "/queue", data=data, method="POST",
                                         headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=3).read()
        except Exception:
            pass


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
                p.setdefault("materials", [])
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
                return False, "项目内仍有运行/排队中的分镜，请先取消"
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
        return True, ("已删除项目及其分镜与产物" if mode == "purge" else "已删除项目，分镜已转默认项目")

    def _project_from_rel(self, rel):
        parts = rel.split("/")
        if len(parts) >= 3 and parts[0] in self.projects:
            return parts[0]
        return DEFAULT_PROJECT

    def project_info(self, pid, jobs=None):
        p = self.projects[pid]
        js = self._project_jobs(pid) if jobs is None else jobs
        js = [j for j in js if (j["st"].get("mode") or "ref2v") != "edit"]
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

    # ---- materials (per-project asset library) ----
    def _materials_dir(self, pid):
        return os.path.join(self._proj_dir(pid), "materials")

    def materials_list(self, pid):
        p = self.projects.get(pid)
        if not p:
            return []
        d = self._materials_dir(pid)
        helper_mt = 0
        for helper in (THUMB_HELPER, VTHUMB_HELPER):
            try:
                helper_mt = max(helper_mt, int(os.path.getmtime(helper)))
            except OSError:
                pass
        out = []
        for m in p.get("materials") or []:
            if not isinstance(m, dict) or not m.get("file"):
                continue
            fp = os.path.join(d, m["file"])
            exists = os.path.isfile(fp)
            kind = m.get("kind") or "image"
            v = 0
            if kind in ("image", "video") and exists:
                try:
                    v = max(int(os.path.getmtime(fp)), helper_mt)
                except OSError:
                    v = helper_mt
            out.append({"id": m.get("id"), "name": m.get("name") or "",
                        "kind": kind, "file": m.get("file"),
                        "size": m.get("size") or 0, "ts": m.get("ts"),
                        "thumb_v": v,
                        "exists": exists})
        return out

    def material_path(self, pid, mid):
        p = self.projects.get(pid)
        if not p or not mid:
            return None
        for m in p.get("materials") or []:
            if isinstance(m, dict) and m.get("id") == mid and m.get("file"):
                fp = os.path.join(self._materials_dir(pid), m["file"])
                return fp if os.path.isfile(fp) else None
        return None

    def material_file(self, pid, fn):
        """Resolve a stored material filename to an absolute path (no traversal)."""
        p = self.projects.get(pid)
        if not p or not fn or fn != os.path.basename(fn):
            return None
        base = os.path.realpath(self._materials_dir(pid))
        cand = os.path.realpath(os.path.join(base, fn))
        if cand.startswith(base + os.sep) and os.path.isfile(cand):
            return cand
        return None

    def _thumbs_dir(self, pid):
        return os.path.join(self._proj_dir(pid), "thumbs")

    def material_thumb(self, pid, fn, maxw=THUMB_MAX):
        """Resolve a material image/video to a cached thumbnail or poster (else the original)."""
        src = self.material_file(pid, fn)
        if not src:
            return None
        p = self.projects.get(pid)
        kind = None
        for m in (p.get("materials") or []) if p else []:
            if isinstance(m, dict) and m.get("file") == fn:
                kind = m.get("kind"); break
        if kind not in ("image", "video"):
            return src
        helper = THUMB_HELPER if kind == "image" else VTHUMB_HELPER
        suffix = ".jpg" if maxw == THUMB_MAX else ".p%d.jpg" % maxw
        dst = os.path.join(self._thumbs_dir(pid), fn + suffix)
        try:
            helper_mt = os.path.getmtime(helper) if os.path.isfile(helper) else 0
            fresh = (os.path.isfile(dst)
                     and os.path.getmtime(dst) >= max(os.path.getmtime(src), helper_mt))
        except OSError:
            fresh = False
        if fresh:
            return dst
        try:
            os.makedirs(self._thumbs_dir(pid), exist_ok=True)
            subprocess.run([sys.executable, helper, src, dst, str(maxw)],
                           check=True, timeout=90,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            return src
        return dst if os.path.isfile(dst) else src

    def clip_thumb(self, rel):
        """Resolve a generated clip to a cached poster JPEG (else None)."""
        base = os.path.realpath(self.out_root)
        cand = os.path.realpath(os.path.join(base, (rel or "").strip()))
        if not cand.startswith(base + os.sep) or not os.path.isfile(cand):
            return None
        rel = os.path.relpath(cand, base).replace("\\", "/")
        dst = os.path.join(self.out_root, ".thumbs", rel + ".jpg")
        try:
            helper_mt = os.path.getmtime(VTHUMB_HELPER) if os.path.isfile(VTHUMB_HELPER) else 0
            fresh = (os.path.isfile(dst)
                     and os.path.getmtime(dst) >= max(os.path.getmtime(cand), helper_mt))
        except OSError:
            fresh = False
        if not fresh:
            try:
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                subprocess.run([sys.executable, VTHUMB_HELPER, cand, dst, str(THUMB_MAX)],
                               check=True, timeout=120,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception:
                return None
        return dst if os.path.isfile(dst) else None

    def _mat_name_ok(self, name):
        name = (name or "").strip()
        if not name:
            return None, "请填写素材名称"
        if len(name) > MATERIAL_NAME_MAX:
            return None, "素材名称过长（>%d 字符）" % MATERIAL_NAME_MAX
        if "/" in name or "\\" in name:
            return None, "素材名称不能包含斜杠"
        return name, None

    def add_material(self, pid, name, filename, content):
        name, err = self._mat_name_ok(name)
        if err:
            return None, err
        ext = os.path.splitext(filename or "")[1].lower()
        kind = MATERIAL_EXTS.get(ext)
        if not kind:
            return None, "不支持的文件类型：%s" % (ext or "未知")
        if not content:
            return None, "文件内容为空"
        mid = "m" + uuid.uuid4().hex[:8]
        stored = mid + ext
        with self.lock:
            p = self.projects.get(pid)
            if not p:
                return None, "项目不存在"
            if any((m.get("name") or "") == name for m in (p.get("materials") or [])):
                return None, "素材名称「%s」已存在" % name
            d = self._materials_dir(pid)
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, stored), "wb") as f:
                f.write(content)
            mat = {"id": mid, "name": name, "kind": kind, "file": stored,
                   "size": len(content), "ts": NOW()}
            p.setdefault("materials", []).append(mat)
            self._write_project(p)
        return mat, None

    def delete_material(self, pid, mid):
        with self.lock:
            p = self.projects.get(pid)
            if not p:
                return False, "项目不存在"
            mats = p.get("materials") or []
            for i, m in enumerate(mats):
                if isinstance(m, dict) and m.get("id") == mid:
                    del mats[i]
                    self._write_project(p)
                    try:
                        os.remove(os.path.join(self._materials_dir(pid), m.get("file") or ""))
                    except OSError:
                        pass
                    return True, "已删除素材"
        return False, "素材不存在"

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
                          "name": cfg.get("name") or None,
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
        if (job["st"].get("mode") or (job.get("cfg") or {}).get("mode")) == "edit":
            return self._stage_edit(job)
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
        mode = cfg.get("mode", "ref2v")
        with open(job["log"], "a", encoding="utf-8") as log:
            log.write("=== job %s ===\n" % jid)
            t0 = time.time()
            while mode != "edit" and not self._stop:
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
            os.makedirs(self._out_dir(mode, cfg.get("project")), exist_ok=True)
            out_path = os.path.join(self._out_dir(mode, cfg.get("project")), "%s.mp4" % cfg["tag"])
            argv = self._edit_argv(cfg, out_path) if mode == "edit" else self._build_argv(cfg)
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
                p = os.path.join(self._out_dir(mode, cfg.get("project")), "%s.mp4" % cfg["tag"])
                rel = os.path.relpath(p, self.out_root).replace("\\", "/")
                size = os.path.getsize(p) if os.path.isfile(p) else 0
                frames = self._edit_total_frames(job) if mode == "edit" \
                    else ref2v_length(cfg["params"]["dur"])
                self._set(jid, "status", "done", ended=NOW(), exit=0, err=None,
                          duration=int(time.time() - job["st"].get("created_ts", time.time())),
                          clip_rel=rel if size else None, clip_size=size,
                          clip_frames=frames,
                          clip_seconds=(round(frames / FPS, 2) if frames else None))
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
                return False, "分镜不存在或已结束"
            if job["st"]["status"] == "queued":
                job["st"].update(status="cancelled", ended=NOW(),
                                 duration=int(time.time() - job["st"].get("created_ts", time.time())))
                self.persist_status(job)
                self.queue = [x for x in self.queue if x != jid]
                return True, "已取消(排队中)"
            job["st"]["cancel_requested"] = True
            self._term(job)
        self.comfy.interrupt()
        return True, "已发送取消"

    def retry(self, jid):
        """Re-queue a finished/failed/cancelled job with its stored parameters."""
        with self.lock:
            job = self.jobs.get(jid)
            if not job:
                return False, "不存在"
            st = job["st"]
            if st.get("status") in ("running", "queued"):
                return False, "分镜已在队列中"
            cfg = job.get("cfg")
            if not cfg or not cfg.get("params"):
                return False, "缺少参数，无法重新生成"
            for k in ("progress", "detail", "clip_rel", "clip_size", "clip_frames",
                      "clip_seconds", "duration", "ended", "err"):
                st.pop(k, None)
            st.update(status="queued", created=NOW(), created_ts=time.time(),
                      stage=None, cancel_requested=False, tag=jid)
            cfg["tag"] = jid
            self.persist_cfg(job)
            self.persist_status(job)
            self.queue.append(jid)
            with self._cv:
                self._cv.notify()
        return True, "已重新提交生成"

    def delete(self, jid):
        with self.lock:
            job = self.jobs.get(jid)
            if not job:
                return False, "不存在"
            st = job["st"].get("status")
            if st == "running":
                return False, "运行中的分镜请先取消"
            if st == "queued":
                self.queue = [x for x in self.queue if x != jid]
                job["st"].update(status="cancelled", ended=NOW())
            rel = job["st"].get("clip_rel")
            shutil.rmtree(self._job_dir(jid), ignore_errors=True)
            del self.jobs[jid]
        if rel:
            self.delete_outputs([rel])
        return True, "已删除分镜及其产物"

    # ---- snapshots ----
    def info(self, job):
        st = job["st"]
        return {"id": st["id"], "status": st.get("status"), "created": st.get("created"),
                "created_ts": st.get("created_ts"),
                "project": st.get("project") or (job.get("cfg") or {}).get("project") or DEFAULT_PROJECT,
                "mode": st.get("mode") or (job.get("cfg") or {}).get("mode") or "ref2v",
                "name": st.get("name") or (job.get("cfg") or {}).get("name") or None,
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

    def name_taken(self, project, name, keep=None):
        if not name:
            return False
        with self.lock:
            for jid, j in self.jobs.items():
                if jid == keep or j["st"].get("mode") == "edit":
                    continue
                if (j["st"].get("project") or (j.get("cfg") or {}).get("project")) != project:
                    continue
                if (j["st"].get("name") or (j.get("cfg") or {}).get("name") or "") == name:
                    return True
        return False

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
            if "/edit/" in rel:
                continue
            m = meta.get(os.path.basename(p), {})
            pid = m.get("project") or self._project_from_rel(rel)
            if project is not None and pid != project:
                continue
            clips.append({"rel": rel, "name": os.path.basename(p),
                          "size": os.path.getsize(p), "project": pid,
                          "ts": time.strftime("%m-%d %H:%M", time.localtime(os.path.getmtime(p))),
                          "job": m.get("id"), "shot": m.get("name"),
                          "seconds": m.get("clip_seconds"),
                          "frames": m.get("clip_frames"), "seed": m.get("seed"),
                          "params": m.get("params")})
        return {"clips": clips}

    # ---- timeline / edit sequence ----
    def _seq_path(self, pid):
        return os.path.join(self._proj_dir(pid), "edit.json")

    def load_sequence(self, pid):
        try:
            with open(self._seq_path(pid), encoding="utf-8") as f:
                seq = json.load(f)
        except Exception:
            seq = {}
        seq["aspect"] = str(seq.get("aspect") or "0")
        seq["fade_in"] = _to_float(seq.get("fade_in"), 0.0, lo=0.0, hi=3.0)
        seq["fade_out"] = _to_float(seq.get("fade_out"), 0.0, lo=0.0, hi=3.0)
        seq["clips"] = [c for c in (seq.get("clips") or [])
                        if isinstance(c, dict) and (c.get("rel") or "").strip()]
        return seq

    def _valid_clip(self, rel):
        rel = (rel or "").strip()
        if not rel or "/edit/" in rel:
            return None
        base = os.path.realpath(self.out_root)
        cand = os.path.realpath(os.path.join(base, rel))
        if not cand.startswith(base + os.sep) or not os.path.isfile(cand):
            return None
        return rel

    def save_sequence(self, pid, seq):
        if pid not in self.projects:
            return None, "项目不存在"
        clips = []
        for c in (seq or {}).get("clips") or []:
            rel = self._valid_clip((c or {}).get("rel"))
            if not rel:
                return None, "片段不存在：%s" % ((c or {}).get("rel") or "?")
            t = (c or {}).get("trans") or {}
            tt = t.get("type") if t.get("type") in TRANSITIONS else "cut"
            dur = 0.0 if tt == "cut" else _to_float(t.get("dur"), 0.5, lo=0.1, hi=1.5)
            clips.append({"rel": rel, "trans": {"type": tt, "dur": dur}})
        out = {"aspect": str((seq or {}).get("aspect") or "0"),
               "fade_in": _to_float((seq or {}).get("fade_in"), 0.0, lo=0.0, hi=3.0),
               "fade_out": _to_float((seq or {}).get("fade_out"), 0.0, lo=0.0, hi=3.0),
               "clips": clips}
        os.makedirs(self._proj_dir(pid), exist_ok=True)
        with open(self._seq_path(pid), "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        return out, None

    def submit_edit(self, pid, seq=None):
        if pid not in self.projects:
            return None, "项目不存在"
        if seq is not None:
            saved, err = self.save_sequence(pid, seq)
            if err:
                return None, err
        else:
            saved = self.load_sequence(pid)
        if not saved.get("clips"):
            return None, "时间线为空，请先添加片段"
        jid = self.new_id()
        cfg = {"mode": "edit", "project": pid, "tag": jid, "seed": None, "media": {},
               "params": {"prompt": "剪辑 %d 段" % len(saved["clips"]),
                          "dur": 0, "steps": 0, "aspect": saved["aspect"],
                          "megapixels": 0, "ref_image_size": "match",
                          "clips": len(saved["clips"])},
               "seq": saved}
        return self.submit(cfg, jid=jid), None

    def delete_edit(self, jid):
        with self.lock:
            job = self.jobs.get(jid)
            if not job:
                return False, "不存在"
            mode = job["st"].get("mode") or (job.get("cfg") or {}).get("mode")
            if mode != "edit":
                return False, "不是剪辑成片"
            if job["st"].get("status") in ("running", "queued"):
                return False, "运行中的分镜请先取消"
            rel = job["st"].get("clip_rel")
            if rel:
                base = os.path.realpath(self.out_root)
                cand = os.path.realpath(os.path.join(base, rel))
                if cand.startswith(base + os.sep) and os.path.isfile(cand):
                    try:
                        os.remove(cand)
                    except OSError:
                        pass
            shutil.rmtree(self._job_dir(jid), ignore_errors=True)
            del self.jobs[jid]
        return True, "已删除成片"

    def _edit_argv(self, cfg, out_path):
        seq = cfg["seq"]
        clips = []
        for c in seq.get("clips") or []:
            ap = os.path.realpath(os.path.join(self.out_root, c["rel"]))
            clips.append({"path": ap, "trans": c.get("trans")})
        rseq = {"fps": FPS, "aspect": aspect_ratio(seq.get("aspect")),
                "fade_in": seq.get("fade_in") or 0.0,
                "fade_out": seq.get("fade_out") or 0.0, "clips": clips}
        sp = os.path.join(self._job_dir(cfg["tag"]), "edit_seq.json")
        os.makedirs(os.path.dirname(sp), exist_ok=True)
        with open(sp, "w", encoding="utf-8") as f:
            json.dump(rseq, f, ensure_ascii=False)
        driver = os.path.join(self.root, "scripts", "minimax_h3_edit.py")
        return [sys.executable, driver, "--seq", sp, "--out", out_path]

    def _edit_total_frames(self, job):
        try:
            with open(job["log"], encoding="utf-8", errors="replace") as f:
                m = None
                for m in re.finditer(r"total_frames=(\d+)", f.read()):
                    pass
            return int(m.group(1)) if m else None
        except OSError:
            return None

    def _stage_edit(self, job):
        try:
            with open(job["log"], "rb") as f:
                text = f.read().decode("utf-8", "replace")
        except Exception:
            text = ""
        label, prog = None, None
        for line in text.splitlines():
            if line.startswith("[edit] stage "):
                label = line[len("[edit] stage "):].strip()
            elif line.startswith("[edit] "):
                m = re.match(r"\[edit\] (\d+)/(\d+)", line)
                if m:
                    prog = (int(m.group(1)), int(m.group(2)))
        if label:
            job["st"]["stage"] = {"label": label, "key": "edit", "ts": NOW()}
        if job["st"].get("status") == "done":
            job["st"].pop("progress", None)
        elif prog and prog[1] > 0 and prog[0] <= prog[1]:
            job["st"]["progress"] = {"cur": prog[0], "total": prog[1]}



# ------------------------------------------------------------------- handler
_OUTPUT_RE = re.compile(r"^/files/(.+)$")


def _tracked_path(rest):
    """Split `pid/file[/display-name]` -> (pid, file); the trailing display name
    is only there so the browser suggests it when saving the file."""
    pid, _, tail = rest.partition("/")
    fn = tail.split("/", 1)[0]
    return pid, unquote(fn)


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

        def _send_file_range(self, path, force_dl=False, cache=False):
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
            self.send_header("Cache-Control",
                             "public, max-age=86400" if cache else "no-store")
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
            elif u.path == "/api/materials":
                proj = parse_qs(u.query).get("project", [None])[0] or None
                if not proj or proj not in mgr.projects:
                    self._err(400, "项目不存在"); return
                self._json(200, {"project": proj, "materials": mgr.materials_list(proj)})
            elif u.path == "/api/sequence":
                proj = parse_qs(u.query).get("project", [None])[0] or None
                if not proj or proj not in mgr.projects:
                    self._err(400, "项目不存在"); return
                self._json(200, {"project": proj, "seq": mgr.load_sequence(proj),
                                 "clips": mgr.clips_list(proj)["clips"]})
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
            elif u.path.startswith("/thumb/"):
                pid, fn = _tracked_path(u.path[len("/thumb/"):])
                cand = mgr.material_thumb(pid, fn) if fn else None
                if not cand:
                    self._err(404, "not found"); return
                self._send_file_range(cand, cache=True)
            elif u.path.startswith("/preview/"):
                pid, fn = _tracked_path(u.path[len("/preview/"):])
                cand = mgr.material_thumb(pid, fn, PREVIEW_MAX) if fn else None
                if not cand:
                    self._err(404, "not found"); return
                self._send_file_range(cand, cache=True)
            elif u.path.startswith("/vthumb/"):
                cand = mgr.clip_thumb(unquote(u.path[len("/vthumb/"):]))
                if not cand:
                    self._err(404, "not found"); return
                self._send_file_range(cand)
            elif u.path.startswith("/material/"):
                pid, fn = _tracked_path(u.path[len("/material/"):])
                cand = mgr.material_file(pid, fn) if fn else None
                if not cand:
                    self._err(404, "not found"); return
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
            m = re.match(r"^/api/jobs/([^/]+)/(cancel|delete|retry)$", u.path)
            if m:
                jid, action = m.group(1), m.group(2)
                if action == "retry":
                    ok, msg = mgr.retry(jid)
                else:
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
            matm = re.match(r"^/api/materials/([^/]+)/([^/]+)/delete$", u.path)
            if matm:
                pid, mid = matm.group(1), matm.group(2)
                try:
                    length = int(self.headers.get("Content-Length") or 0)
                    if length:
                        self.rfile.read(length)
                except Exception:
                    pass
                ok, msg = mgr.delete_material(pid, mid)
                self._json(200 if ok else 409, {"ok": ok, "msg": msg,
                                                "materials": mgr.materials_list(pid)})
                return
            if u.path in ("/api/sequence", "/api/edit/render", "/api/edit/delete"):
                try:
                    length = int(self.headers.get("Content-Length") or 0)
                    js = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
                except Exception:
                    self._err(400, "bad json"); return
                if u.path == "/api/edit/delete":
                    ok, msg = mgr.delete_edit(js.get("id"))
                    self._json(200 if ok else 409, {"ok": ok, "msg": msg}); return
                pid = (js.get("project") or "").strip()
                if pid not in mgr.projects:
                    self._err(400, "项目不存在"); return
                if u.path == "/api/sequence":
                    saved, err = mgr.save_sequence(pid, js.get("seq") or {})
                    if err:
                        self._err(400, err); return
                    self._json(200, {"ok": True, "seq": saved}); return
                jid, err = mgr.submit_edit(pid, js.get("seq"))
                if err:
                    self._err(400, err); return
                self._json(202, {"id": jid, "status": "queued"}); return
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
                    self._json(409, {"ok": False, "msg": "有分镜在跑, 先取消"}); return
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
            if u.path == "/api/materials":
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
                bm = re.search(r'boundary="?([^";]+)"?', ct)
                if not (ct.startswith("multipart/form-data") and bm):
                    self._err(400, "expected multipart/form-data"); return
                try:
                    fields, files = parse_multipart(body, bm.group(1).encode("ascii"))
                except ValueError as e:
                    self._err(400, str(e)); return
                pid = (_first(fields, "project", "") or "").strip()
                if pid not in mgr.projects:
                    self._err(400, "项目不存在"); return
                f = files[0] if files else None
                if not f:
                    self._err(400, "请选择要上传的文件"); return
                mat, err = mgr.add_material(pid, _first(fields, "name", "") or "",
                                            f.get("filename"), f.get("content") or b"")
                if err:
                    self._err(400, err); return
                self._json(201, {"ok": True, "material": mat,
                                 "materials": mgr.materials_list(pid)})
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
            project = _first(fields, "project", DEFAULT_PROJECT) or DEFAULT_PROJECT
            if project not in mgr.projects:
                self._err(400, "项目不存在"); return
            name = (_first(fields, "name", "") or "").strip()[:60]
            if name and mgr.name_taken(project, name):
                self._err(409, "已存在同名分镜：%s" % name); return
            ids = {k: [x for x in (fields.get(k) or []) if x] for k in MEDIA_KINDS}
            frame_ids = {k: (_first(fields, k) or None) for k in FRAME_KINDS}
            n = {k: len(v) for k, v in ids.items()}
            if mode == "ref2v":
                if n["ref_image"] > MAX_IMAGES:
                    self._err(400, "参考图最多 %d 张" % MAX_IMAGES); return
                if n["ref_video"] > MAX_VIDEOS:
                    self._err(400, "参考视频最多 %d 段" % MAX_VIDEOS); return
                if n["ref_audio"] > MAX_AUDIOS:
                    self._err(400, "参考音频最多 %d 段" % MAX_AUDIOS); return
                if not any(n.values()):
                    self._err(400, "请至少选择一张参考图/视频/音频"); return
                if any(frame_ids.values()):
                    self._err(400, "参考生视频不支持首/尾帧"); return
            elif any(n.values()):
                self._err(400, "文生视频不支持参考素材"); return
            media = {}
            for kind in MEDIA_KINDS:
                paths = []
                for ref in ids[kind]:
                    if kind == "ref_video" and ref.startswith("clip:"):
                        rel = mgr._valid_clip(ref[len("clip:"):])
                        if not rel:
                            self._err(400, "产物不存在或不可用：%s" % ref[len("clip:"):]); return
                        paths.append(os.path.join(mgr.out_root, rel))
                        continue
                    fp = mgr.material_path(project, ref)
                    if not fp:
                        self._err(400, "素材不存在或已被删除：%s" % ref); return
                    paths.append(fp)
                media[kind] = paths
            for kind in FRAME_KINDS:
                mid = frame_ids[kind]
                if not mid:
                    media[kind] = []
                    continue
                fp = mgr.material_path(project, mid)
                if not fp:
                    self._err(400, "素材不存在或已被删除：%s" % mid); return
                media[kind] = [fp]
            if mode == "ref2v":
                ok = self._check_ref_limits(project, ids, media, n)
                if ok:
                    self._err(400, ok); return
            seed = _to_int(_first(fields, "seed"), None, lo=0, hi=2**63 - 1)
            if seed is None:
                seed = random.randint(0, 2**63 - 1)
            cfg = {
                "mode": mode,
                "project": project,
                "name": name,
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
                "media": media,
            }
            jid = mgr.new_id()
            cfg["tag"] = jid
            os.makedirs(mgr._job_dir(jid), exist_ok=True)
            mgr.submit(cfg, jid=jid)
            self._json(202, {"id": jid, "status": "queued"})

        def _check_ref_limits(self, project, ids, media, n):
            """Validate the MiniMax reference limits; return an error string or None."""
            total = n["ref_image"] + n["ref_video"] + n["ref_audio"]
            if total > MAX_REFS:
                return "参考文件合计最多 %d 个（当前 %d 个）" % (MAX_REFS, total)
            if n["ref_audio"] and not (n["ref_image"] or n["ref_video"]):
                return "音频不能单独作为参考，请至少再选 1 张图片或 1 段视频"
            lim = {"ref_image": MAX_IMAGE_MB, "ref_video": MAX_VIDEO_MB, "ref_audio": MAX_AUDIO_MB}
            cn = {"ref_image": "图片", "ref_video": "视频", "ref_audio": "音频"}
            names = {m.get("id"): (m.get("name") or m.get("file") or "")
                     for m in ((mgr.projects.get(project) or {}).get("materials") or [])}
            total_bytes = 0
            for kind in MEDIA_KINDS:
                for ref, path in zip(ids[kind], media[kind]):
                    label = os.path.basename(ref[len("clip:"):]) if ref.startswith("clip:") else names.get(ref, ref)
                    try:
                        size = os.path.getsize(path)
                    except OSError:
                        size = 0
                    total_bytes += size
                    if size > lim[kind] * 1048576:
                        return "%s「%s」%.1fMB，超过 %dMB 上限" % (
                            cn[kind], label, size / 1048576, lim[kind])
            if total_bytes > MAX_REQUEST_MB * 1048576:
                return "参考文件合计 %.1fMB，超过单次请求 %dMB 上限" % (
                    total_bytes / 1048576, MAX_REQUEST_MB)
            return None

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
.hdrright{display:flex;gap:8px;align-items:center;margin-left:auto;flex-wrap:wrap}
main{padding:14px;max-width:1400px;margin:0 auto}
.cols{display:grid;grid-template-columns:minmax(340px,460px) 1fr;gap:14px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px;margin-bottom:14px}
.card h2{font-size:14px;margin:0 0 10px;color:var(--mut);font-weight:600;letter-spacing:.03em}
.cardhead{display:flex;justify-content:space-between;align-items:center;gap:10px;margin-bottom:10px}
.cardhead h2{margin:0}
.cardhead.modehead{justify-content:flex-start;flex-wrap:wrap}
.cardhead.modehead h2{white-space:nowrap}
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
.headacts{display:flex;gap:10px;align-items:center}
.headacts button{width:auto;min-width:76px;height:34px;margin:0;padding:0 16px;border-radius:8px;
        font-size:14px;font-weight:600;cursor:pointer;display:flex;align-items:center;justify-content:center}
.headacts button.primary{width:auto;padding:0 16px;font-size:14px}
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
details.logBox summary{list-style:none}
details.logBox summary::-webkit-details-marker{display:none}
details.logBox pre{margin-top:8px}
details.sec>summary{list-style:none;font-size:14px;color:var(--mut);font-weight:600;
       letter-spacing:.03em;display:flex;align-items:center;gap:6px}
details.sec>summary::-webkit-details-marker{display:none}
.setoggle{display:inline-flex;align-items:center;gap:6px;cursor:pointer}
.setoggle::before{content:"\25BC";font-size:.8em;line-height:1;color:var(--fg);transition:transform .2s}
details:not([open])>summary .setoggle::before{transform:rotate(-90deg)}
details.sec[open]>summary{margin-bottom:10px}
pre.log{max-height:240px;overflow:auto;background:#0b0d11;border:1px solid var(--line);border-radius:8px;
        padding:8px;font-size:12px;color:#c7cede;white-space:pre-wrap;word-break:break-all}
.job{display:flex;gap:10px;align-items:flex-start;padding:9px 0;border-bottom:1px solid var(--line);flex-wrap:wrap}
.job .st{display:block;font-size:14px;font-weight:700;background:none;padding:0;border-radius:0}
.st.done{color:var(--ok)}.st.running{color:var(--acc)}
.st.failed,.st.cancelled,.st.interrupted{color:var(--err)}.st.queued{color:var(--warn)}
.job .meta{font-size:14px;line-height:normal;color:var(--mut);flex:1;min-width:160px;margin-top:-.1em}
.jobsacts{display:flex;gap:8px;flex-wrap:wrap;align-items:center}
.clips{display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:10px}
.clip{position:relative;background:#0e1116;border:1px solid var(--line);border-radius:10px;overflow:hidden;cursor:pointer}
.clip video,.clip img,.clip .ph{width:100%;aspect-ratio:16/9;background:#000;display:block;object-fit:cover}
.clip .cap{padding:6px 8px;font-size:12px;color:var(--mut)}
.clip .pick{position:absolute;top:6px;left:6px;width:22px;height:22px;border-radius:6px;line-height:1;
       background:rgba(0,0,0,.55);border:2px solid #fff;display:flex;align-items:center;justify-content:center;
       font-size:14px;color:#fff}
.clip.sel{outline:3px solid var(--acc);outline-offset:-3px}
.clip.sel .pick{background:var(--acc);border-color:var(--acc)}
#stJobs details>summary{width:100%}
.secbtn{margin-left:auto;display:flex;align-items:center}
.secbtn button{padding:3px 10px;font-size:13px;line-height:1.2}
.jthumb{width:120px;aspect-ratio:16/9;background:#000;border-radius:8px;object-fit:cover;flex:0 0 auto;cursor:pointer}
.jthumb.ph{position:relative;overflow:hidden;border:1px solid var(--line);cursor:default;
        background:linear-gradient(100deg,#12161d 30%,#1c2431 50%,#12161d 70%);
        background-size:200% 100%;animation:phshim 1.4s linear infinite}
.jthumb.ph>i{position:absolute;left:0;bottom:0;height:3px;background:var(--acc);transition:width .3s}
.jthumb.ph.static{animation:none;background:#0e1116}
@keyframes phshim{0%{background-position:100% 0}100%{background-position:-100% 0}}
.modal{position:fixed;inset:0;background:rgba(0,0,0,.8);display:none;align-items:center;justify-content:center;z-index:20;padding:12px}
.modal.open{display:flex}
.modal .box{width:min(960px,98vw);background:#0e1116;border:1px solid var(--line);border-radius:12px;padding:10px}
.modal video{width:100%;max-height:76vh;background:#000;border-radius:8px}
.opttext{width:100%;min-height:220px;max-height:56vh;background:#0b0e13;color:var(--fg);
       border:1px solid var(--line);border-radius:8px;padding:10px;font-size:13px;line-height:1.5;resize:vertical}
.optrow{display:flex;justify-content:space-between;align-items:center;gap:10px;margin-bottom:8px}
.optrow b{min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.optrow button{flex:0 0 auto;white-space:nowrap}
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
.projsub{display:flex;align-items:center;gap:10px;flex-wrap:wrap}
.mediagrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(120px,1fr));gap:8px;margin-top:6px}
.mediagrid img,.mediagrid video{width:100%;aspect-ratio:16/9;object-fit:cover;background:#000;
       border-radius:8px;border:1px solid var(--line)}
.medialist{display:block;margin-top:6px}
.medialist audio{width:100%;display:block;margin-top:6px}
.matup{display:flex;gap:8px;flex-wrap:wrap;align-items:center}
.matup #matName{flex:1 1 240px;min-width:0}
.matup #matFile{flex:1 1 260px;min-width:0;padding:7px 10px;font-size:13px}
.matgrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:10px;margin-top:12px;align-items:start}
.matgroups{display:flex;flex-direction:column;gap:16px;margin-top:12px}
.matgrouphead{font-size:13px;color:var(--mut);font-weight:600;margin-bottom:8px}
.matgroups .matgrid{margin-top:0}
.matcard{background:#0e1116;border:1px solid var(--line);border-radius:10px;overflow:hidden;
         display:grid;grid-template-columns:minmax(0,1fr) auto;grid-template-areas:"thumb thumb" "mb ma";position:relative}
.matcard.pick{cursor:pointer}
.matcard.pick:hover{border-color:var(--acc)}
.matcard.sel{outline:3px solid var(--acc);outline-offset:-3px}
.matcard .thumb{grid-area:thumb;width:100%;aspect-ratio:16/9;background:#000;object-fit:cover;display:block}
.matcard img.thumb{object-fit:contain;background:#0a0c11;cursor:zoom-in}
.matcard .thumbwrap{grid-area:thumb;position:relative;width:100%;aspect-ratio:16/9;background:#0a0c11}
.matcard .thumbwrap img.thumb{width:100%;height:100%;aspect-ratio:auto;object-fit:contain;display:block}
.matcard .thumbdl{position:absolute;inset:0;cursor:zoom-in}
.matcard.pick img.thumb{cursor:pointer}
.matcard video.thumb{cursor:pointer}
.matcard .thumbicon{grid-area:thumb;width:100%;aspect-ratio:16/9;background:#0a0c11;display:flex;flex-direction:column;
        align-items:center;justify-content:center;gap:4px;color:var(--mut);font-size:12px;letter-spacing:.05em;cursor:pointer}
.matcard .thumbicon .audplay{font-size:22px;line-height:1;color:var(--acc)}
.viewbody img{display:block;margin:0 auto;max-width:100%;max-height:82vh;border-radius:8px}
.viewbody video{display:block;width:100%;max-height:82vh;background:#000;border-radius:8px}
.viewbody audio{width:100%}
.matcard .mb{grid-area:mb;padding:6px 9px 8px;min-width:0}
.matcard .nm{font-weight:600;font-size:13px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.matcard .mm{font-size:11px;color:var(--mut);margin-top:2px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.matcard .ma{grid-area:ma;display:flex;align-items:center;padding:6px 9px 8px 0}
.matcard .ma button{background:#20242d;border:1px solid var(--line);color:var(--err);
        border-radius:6px;padding:4px 8px;font-size:12px;cursor:pointer}
.matcard .picktag{position:absolute;top:6px;right:6px;width:20px;height:20px;border-radius:6px;
        background:rgba(0,0,0,.6);border:2px solid #fff;display:flex;align-items:center;justify-content:center;
        font-size:12px;color:#fff}
.matcard.sel .picktag{background:var(--acc);border-color:var(--acc)}
.matcard .shotname{position:absolute;top:6px;left:6px;max-width:calc(100% - 40px);padding:2px 7px;
        border-radius:6px;background:rgba(0,0,0,.65);color:#fff;font-size:12px;font-weight:600;
        overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.matempty{color:var(--mut);font-size:13px;margin-top:10px}
.detprompt{background:#0b0e13;border:1px solid var(--line);border-radius:8px;padding:8px;font-size:13px;
       white-space:pre-wrap;word-break:break-word;max-height:200px;overflow:auto;line-height:1.55}
.detrow{display:flex;gap:10px;font-size:13px;padding:2px 0}
.detrow .muted{min-width:52px}
.timeline{display:flex;flex-direction:column;gap:8px;margin-top:8px}
.tslot{background:#0b0e13;border:1px solid var(--line);border-radius:10px;padding:8px}
.tslot.dragging{opacity:.45}
.trow{display:flex;gap:10px;align-items:center}
.trow .thumb{width:112px;flex:0 0 112px;aspect-ratio:16/9;background:#000;border-radius:8px;object-fit:cover}
.trow .tmeta{flex:1;min-width:0;font-size:12px;color:var(--mut)}
.trow .tmeta b{color:var(--fg);font-weight:600;display:block;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.thandle{cursor:grab;touch-action:none;user-select:none;color:var(--mut);font-size:20px;line-height:1;padding:2px 6px}
.tacts2{display:flex;flex-direction:column;gap:4px}
.tacts2 button{background:#20242d;border:1px solid var(--line);color:var(--fg);border-radius:6px;
       width:30px;height:26px;font-size:12px;cursor:pointer}
.tacts2 button.rm{color:var(--err)}
.junc{display:flex;gap:8px;align-items:center;font-size:12px;color:var(--mut);margin:0 0 8px}
.junc select{width:auto;padding:4px 8px;font-size:12px}
.junc .start{color:var(--ok)}
.addclip{position:absolute;top:6px;right:6px;background:var(--acc);color:#fff;border:0;border-radius:6px;
       padding:2px 8px;font-size:12px;cursor:pointer;z-index:1}
.formgrid,.params,.statgrid,.editgrid{display:block}
@media(min-width:981px){
  .formgrid{display:grid;grid-template-columns:minmax(0,1.15fr) minmax(0,1fr);gap:14px;align-items:start}
  .formgrid>.fcol{min-width:0}
  .formgrid>.fcol:first-child{order:2}
  .formgrid>.fcol:last-child{order:1}
  .formgrid textarea{min-height:214px}
  .params{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:10px;margin-top:4px}
  .params>.grid3,.params>.grid2{display:contents}
  .editgrid{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1.35fr);gap:14px}
  .editgrid>.card{margin-bottom:0}
  #taskCard{position:relative}
  #taskCard>.headacts{position:absolute;top:14px;right:14px}
}
@media(max-width:980px){.cols{grid-template-columns:1fr}
  #projHead .cardhead{flex-wrap:wrap}
  #projHead .cardhead h2{flex:1 1 100%}
  .headacts{margin-top:8px}
  .headacts button{flex:1 1 0;width:auto;min-width:0;height:42px;padding:0;font-size:15px}
  .headacts button.primary{width:auto;padding:0;font-size:15px}
}
@media(max-width:640px){.grid3{grid-template-columns:1fr 1fr}textarea,input,select{font-size:16px}
  .hdrright{margin-left:0}
  .params{display:grid;grid-template-columns:1fr 1fr;gap:10px}
  .params>.grid3,.params>.grid2{display:contents}
  .params .fseed{order:5}
  .cardhead.modehead{display:grid;grid-template-columns:1fr 1fr;gap:10px}
  .cardhead.modehead #mode{width:100%;min-width:0}
  .formgrid>.fcol:last-child{border-top:1px solid var(--line);margin-top:14px;padding-top:14px}
  .params{border-top:1px solid var(--line);margin-top:14px;padding-top:14px}
  #taskCard>.headacts{border-top:1px solid var(--line);margin-top:14px;padding-top:14px}
  .jobsacts{flex-basis:100%}
}
</style>
</head>
<body>
<header>
  <h1 style="cursor:pointer" onclick="goHome()" title="全部项目">MiniMax H3</h1>
  <span id="pComfy" class="pill off">ComfyUI ?</span>
  <span id="pQ" class="pill">队列 0</span>
  <span class="hdrright">
    <span id="pVram" class="pill">VRAM --</span>
    <button class="ghost" onclick="releaseVram()">释放显存</button>
    <button class="ghost" onclick="openLog()">诊断日志</button>
  </span>
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
    <div class="card" id="taskCard" style="display:none">
      <div class="cardhead modehead">
        <h2>新建分镜</h2>
        <select id="mode" onchange="onModeChange()">
          <option value="t2v">文生视频</option>
          <option value="ref2v">参考生视频</option>
        </select>
      </div>
      <div class="muted" id="shotLabel"></div>
      <div class="progress" id="prog"><i></i></div>
      <div class="muted" id="submitMsg" style="margin-top:8px"></div>
      <div class="formgrid">
        <div class="fcol">
          <label id="promptLabel"></label>
          <textarea id="prompt" placeholder="Cinematic shot of the subject ..."></textarea>
          <div style="margin-top:6px"><button class="ghost" id="optBtn" onclick="optimizePrompt()">提示词优化</button></div>
        </div>
        <div class="fcol">
          <div id="t2vBox">
            <label>首帧（可选，单张图片）</label>
            <div class="filebox">
              <button type="button" class="ghost" onclick="openPick('first_frame')">选择素材</button>
              <ul id="lFirst"></ul>
            </div>
            <label>尾帧（可选，单张图片）</label>
            <div class="filebox">
              <button type="button" class="ghost" onclick="openPick('last_frame')">选择素材</button>
              <ul id="lLast"></ul>
            </div>
          </div>
          <div id="ref2vBox">
            <label>参考图（≤9）</label>
            <div class="filebox">
              <button type="button" class="ghost" onclick="openPick('ref_image')">选择素材</button>
              <ul id="lImg"></ul>
            </div>
            <label>参考视频（≤3，每段 2–15s，合计 ≤15s）</label>
            <div class="filebox">
              <button type="button" class="ghost" onclick="openPick('ref_video')">选择素材</button>
              <button type="button" class="ghost" style="margin-left:6px" onclick="openPick('ref_video','clip')">选择产物</button>
              <ul id="lVid"></ul>
            </div>
            <label>参考音频（≤3，合计 ≤15s）</label>
            <div class="filebox">
              <button type="button" class="ghost" onclick="openPick('ref_audio')">选择素材</button>
              <ul id="lAud"></ul>
            </div>
          </div>
        </div>
      </div>
      <div class="params">
        <div class="grid3">
          <div><label>时长(秒)</label><input id="dur" type="number" value="5" min="1" max="15" step="0.5"></div>
          <div><label>步数</label><input id="steps" type="number" value="8" min="1" max="50"></div>
          <div class="fseed"><label>seed(空=随机)</label><input id="seed" type="number" placeholder="随机"></div>
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
        <div id="refImageSizeRow" class="refsize"><label>参考图缩放</label>
          <select id="ref_image_size"><option value="match">match(快)</option><option value="max">max(保真)</option></select></div>
      </div>
      <div class="headacts">
        <button class="primary" id="submitBtn" onclick="submit()">提交</button>
        <button class="ghost" onclick="resetForm()">重置</button>
        <button class="ghost" onclick="cancelShot()">取消</button>
      </div>
    </div>
    <div class="statgrid">
      <div class="card" id="matCard">
        <details class="sec">
          <summary onclick="toggleSec(event)"><span class="setoggle">素材库 <span class="muted" id="matSub"></span></span></summary>
          <div class="matup">
            <input id="matName" placeholder="素材名称（项目内唯一）" onkeydown="if(event.key==='Enter'){event.preventDefault();uploadMaterial()}">
            <input id="matFile" type="file" accept="image/*,video/*,audio/*" onchange="onMatFileChange()">
            <button type="button" class="ghost" onclick="uploadMaterial()">上传素材</button>
          </div>
          <div class="muted" id="matMsg" style="margin-top:6px"></div>
          <div id="matGrid" class="matgroups"></div>
        </details>
      </div>
      <div class="card" id="stJobs">
        <details class="sec" open>
          <summary onclick="toggleSec(event)"><span class="setoggle">分镜列表</span>
            <span class="secbtn"><button class="ghost" onclick="event.preventDefault();event.stopPropagation();openEdit()">剪辑</button></span>
          </summary>
          <div id="jobs" class="muted">暂无</div>
        </details>
      </div>
    </div>
  </div>
  <div id="editView" style="display:none">
    <div class="card" id="editHead"></div>
    <div class="card">
      <div class="cardhead">
        <h2>时间线</h2>
        <span class="bcbar">
          <button class="ghost" onclick="renderEdit()">预览/导出</button>
          <button class="ghost" onclick="saveEdit()">保存</button>
          <button class="ghost" onclick="backToProject()">返回项目</button>
        </span>
      </div>
      <div class="grid3" style="max-width:640px">
        <div><label>成片画幅</label><select id="eAspect"></select></div>
        <div><label>片头淡入(s)</label><input id="eFadeIn" type="number" value="0" min="0" max="3" step="0.1"></div>
        <div><label>片尾淡出(s)</label><input id="eFadeOut" type="number" value="0" min="0" max="3" step="0.1"></div>
      </div>
      <div class="muted" id="editMsg" style="margin-top:8px"></div>
      <div id="timeline" class="timeline"></div>
    </div>
    <div class="card">
      <div class="cardhead"><h2>添加片段</h2><span class="muted">点缩略图加入时间线</span></div>
      <div id="pickClips" class="clips"></div>
    </div>
    <div class="editgrid">
      <div class="card">
        <h2>当前渲染</h2>
        <div id="editCur"><div class="muted">空闲</div></div>
        <details class="logBox" style="margin-top:12px">
          <summary class="muted" onclick="toggleSec(event)"><span class="setoggle">日志</span></summary>
          <pre class="log" id="editLog"></pre>
        </details>
      </div>
      <div class="card">
        <details class="sec" open>
          <summary onclick="toggleSec(event)"><span class="setoggle">成片</span></summary>
          <div id="editList" class="clips"></div>
        </details>
      </div>
    </div>
  </div>
</main>
<div class="modal" id="pickModal" onclick="if(event.target===this)closePick()">
  <div class="box" style="width:min(900px,98vw);max-height:88vh;overflow:auto">
    <div class="optrow"><b id="pickTitle">选择素材</b><button class="ghost" onclick="closePick()">关闭</button></div>
    <div class="muted" id="pickHint" style="margin-bottom:8px"></div>
    <div id="pickGrid" class="matgrid"></div>
    <div class="optacts">
      <button class="ghost" onclick="closePick()">取消</button>
      <button class="primary" id="pickOk" onclick="pickOk()">加入</button>
    </div>
  </div>
</div>
<div class="modal" id="modal" onclick="if(event.target===this)closeModal()">
  <div class="box">
    <video id="mvideo" controls playsinline webkit-playsinline></video>
    <div class="muted" id="mcap" style="margin-top:8px"></div>
  </div>
</div>
<div class="modal" id="viewModal" onclick="if(event.target===this)closeView()">
  <div class="box" style="width:min(1100px,98vw)">
    <div class="optrow"><b id="viewCap"></b>
      <span style="display:flex;gap:8px;flex:0 0 auto">
        <button class="ghost" id="viewOrig" style="display:none" onclick="viewOriginal()">原图</button>
        <button class="ghost" onclick="closeView()">关闭</button>
      </span>
    </div>
    <div id="viewBody" class="viewbody"></div>
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
<div class="modal" id="noticeModal" onclick="if(event.target===this)closeNotice()">
  <div class="box" style="width:min(420px,96vw)">
    <div class="optrow"><b id="noticeTitle">提示</b><button class="ghost" onclick="closeNotice()">关闭</button></div>
    <div class="muted" id="noticeMsg" style="margin-bottom:14px;white-space:pre-wrap;word-break:break-word"></div>
    <div class="optacts">
      <button class="primary" onclick="closeNotice()">知道了</button>
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
<div class="modal" id="shotModal" onclick="if(event.target===this)closeShotModal()">
  <div class="box" style="width:min(420px,96vw)">
    <div class="optrow"><b id="shotTitle">添加分镜</b><button class="ghost" onclick="closeShotModal()">关闭</button></div>
    <input id="shotVal" placeholder="分镜名（项目内唯一）" onkeydown="if(event.key==='Enter'){event.preventDefault();confirmShot()}">
    <div class="muted" id="shotMsg" style="margin-top:8px;min-height:1.2em;color:var(--err)"></div>
    <div class="optacts">
      <button class="ghost" onclick="closeShotModal()">取消</button>
      <button class="primary" id="shotOk" onclick="confirmShot()">确定</button>
    </div>
  </div>
</div>
<div class="modal" id="jobModal" onclick="if(event.target===this)closeJob()">
  <div class="box" style="width:min(720px,96vw);max-height:88vh;overflow:auto">
    <div class="optrow"><b id="jTitle">分镜详情</b><button class="ghost" onclick="closeJob()">关闭</button></div>
    <div id="jBody"></div>
  </div>
</div>
<div class="modal" id="logModal" onclick="if(event.target===this)closeLog()">
  <div class="box" style="width:min(960px,98vw)">
    <div class="optrow"><b>诊断日志</b><button class="ghost" onclick="closeLog()">关闭</button></div>
    <pre class="log" id="log" style="height:60vh;max-height:60vh"></pre>
  </div>
</div>
<script>
const $ = (id)=>document.getElementById(id);
const esc = (s)=>(s||'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const ASPECTS = __ASPECTS__;
let logOffset = 0, lastJob = null, jobsById = {};
let shotName = null, shotCb = null;
let projects = [], projNames = {}, curProject = null, projectsLoaded = false;
let clipsCache = [];
let editSeq={aspect:'0',fade_in:0,fade_out:0,clips:[]}, editAvail=[], editLast=null, editLogOff=0;
let materials=[], pickKind=null, pickSel=new Set(), pickSingle=false, pickMode='mat';
const selMat={ref_image:[],ref_video:[],ref_audio:[],first_frame:[],last_frame:[]};
const selClip={ref_image:[],ref_video:[],ref_audio:[],first_frame:[],last_frame:[]};
const SLOT_CN={ref_image:{cn:'图片',prefix:'Picture',list:'lImg'},
               ref_video:{cn:'视频',prefix:'Video',list:'lVid'},
               ref_audio:{cn:'音频',prefix:'Audio',list:'lAud'},
               first_frame:{cn:'首帧',prefix:null,list:'lFirst'},
               last_frame:{cn:'尾帧',prefix:null,list:'lLast'}};
const SLOT_MEDIA={ref_image:'image',ref_video:'video',ref_audio:'audio',
                  first_frame:'image',last_frame:'image'};
const MAT_KIND_CN={image:'图片',video:'视频',audio:'音频'};
const STATUS_CN = {queued:'排队中', running:'进行中', done:'已完成', failed:'失败', cancelled:'已取消', interrupted:'已中断'};
const MEDIA_CN = {ref_image:'图', ref_video:'视频', ref_audio:'音频'};
const MODE_CN = {t2v:'文生视频', ref2v:'参考生视频', edit:'剪辑成片'};
const TRANS_CN = {cut:'硬切', fade:'黑场渐隐', dissolve:'交叉溶解', push:'推进/滑动'};
const EDIT_ASPECTS = [['0','原始画幅'],['2.39','2.39:1 宽银幕'],['16:9','16:9 横屏'],
  ['9:16','9:16 竖屏'],['1:1','1:1 方形'],['4:3','4:3 横版'],['3:4','3:4 竖版']];
function projName(pid){ return projNames[pid] || (pid==='default'?'默认项目':(pid||'')); }
function toggleSec(e){
  e.preventDefault();
  const st=e.target.closest('.setoggle');
  if(st){ const d=st.closest('details'); if(d) d.open=!d.open; }
}
function projSub(p){
  return [p.counts.total+' 个分镜',
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
        ? '<video class="cover" muted playsinline preload="none" title="'+esc(withExt(String(p.cover).split('/').pop().replace(/\.[^.]+$/,''), fileExt(p.cover)))+
          '" poster="/vthumb/'+encodeURI(p.cover)+'" src="/files/'+encodeURI(p.cover)+'"></video>'
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
    '<span class="bcbar">'+
      (canEdit?'<button class="ghost" onclick="renameProject()">改名</button>'+
      '<button class="ghost" onclick="deleteProject()">删除</button>':'')+
      '<button class="ghost" onclick="goHome()">全部项目</button></span></div>'+
    '<div class="projsub"><span class="muted">'+esc(projSub(p))+'</span>'+
      '<button class="ghost" onclick="addShot()">添加分镜</button></div>';
}
function openProject(pid){ location.hash='#/p/'+encodeURIComponent(pid); }
function openEdit(){ if(curProject) location.hash='#/p/'+encodeURIComponent(curProject)+'/edit'; }
function backToProject(){ if(curProject) location.hash='#/p/'+encodeURIComponent(curProject); }
function goHome(){ location.hash='#/'; }
function route(){
  const m=location.hash.match(/^#\/p\/([^/]+)(\/edit)?$/);
  const pid=m?decodeURIComponent(m[1]):null;
  const edit=!!(m&&m[2]);
  if(pid && projNames[pid]!==undefined){
    if(curProject!==pid){ curProject=pid;
      $('jobs')._sig=null; $('jobs').innerHTML='';
      jobsById={}; clipsCache=[]; lastJob=null; logOffset=0; $('log').textContent='';
      clearSelMat(); materials=[]; $('matGrid').innerHTML=''; }
    $('homeView').style.display='none';
    $('projView').style.display=edit?'none':'';
    $('editView').style.display=edit?'':'none';
    renderProjHead();
    refreshMaterials();
    if(edit){ refreshEdit(); }
    else { $('taskCard').style.display='none'; shotName=null; $('shotLabel').textContent=''; refreshJobs(); refreshOutputs(); }
  }else{
    curProject=null;
    $('homeView').style.display=''; $('projView').style.display='none'; $('editView').style.display='none';
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
  if(!r.ok){ notice('改名失败：'+(j.error||j.msg||r.status)); return; }
  await refreshProjects();
}
let delPid=null;
function updDelHint(){
  $('delHint').textContent = $('delClips').checked
    ? '将同时删除该项目下的分镜列表与视频文件，不可恢复。'
    : '不删除产物：项目内分镜将转为「默认项目」，视频文件保留。';
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
  if(!r.ok){ notice('删除失败：'+(j.error||j.msg||r.status)); return; }
  goHome(); await refreshProjects();
}
$('delClips').onchange=updDelHint;
let askCb=null;
function notice(msg,title){
  $('noticeTitle').textContent=title||'提示';
  $('noticeMsg').textContent=msg||'';
  $('noticeModal').classList.add('open');
}
function closeNotice(){ $('noticeModal').classList.remove('open'); }
function openLog(){
  $('logModal').classList.add('open');
  pollLog().then(()=>{ const el=$('log'); el.scrollTop=el.scrollHeight; });
}
function closeLog(){ $('logModal').classList.remove('open'); }
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
function openShotModal(title,value,okText,cb){
  shotCb=cb;
  $('shotTitle').textContent=title||'添加分镜';
  $('shotOk').textContent=okText||'确定';
  $('shotVal').value=value||'';
  $('shotMsg').textContent='';
  $('shotModal').classList.add('open');
  setTimeout(()=>{ $('shotVal').focus(); $('shotVal').select(); },50);
}
function closeShotModal(){ $('shotModal').classList.remove('open'); shotCb=null; }
async function confirmShot(){
  if(!curProject) return;
  const name=$('shotVal').value.trim();
  if(!name){ $('shotMsg').textContent='请填写分镜名'; return; }
  if(name.length>60){ $('shotMsg').textContent='分镜名过长（>60 字符）'; return; }
  const r=await api('/api/jobs?project='+encodeURIComponent(curProject));
  const used=((r&&r.jobs)||[]).some(j=>j.mode!=='edit' && (j.name||'')===name);
  if(used){ $('shotMsg').textContent='已存在同名分镜，请换一个名字'; return; }
  const cb=shotCb; shotCb=null; $('shotModal').classList.remove('open');
  if(cb) cb(name);
}
function addShot(){
  if(!curProject){ notice('请先进入一个项目'); return; }
  openShotModal('添加分镜','','确定',(name)=>{ shotName=name; showTaskCard(); });
}
function showTaskCard(){
  $('taskCard').style.display='';
  $('shotLabel').textContent=shotName? ('分镜：'+shotName) : '';
  $('submitMsg').textContent='';
  setTimeout(()=>{ $('taskCard').scrollIntoView({block:'start',behavior:'smooth'}); },30);
}
function fileExt(f){ const s=String(f), i=s.lastIndexOf('.'); return i>0? s.slice(i):''; }
function withExt(name, ext){ const s=String(name||''); return (ext && s.toLowerCase().endsWith(String(ext).toLowerCase()))? s : s+ext; }
function matSaveName(m){ return withExt(m.name, fileExt(m.file)); }
function tailName(name){ return name? '/'+encodeURIComponent(name) : ''; }
function matUrl(pid,file,name){ return '/material/'+encodeURIComponent(pid)+'/'+encodeURIComponent(String(file).split('/').pop())+tailName(name); }
function thumbUrl(pid,file,v,name){ return '/thumb/'+encodeURIComponent(pid)+'/'+encodeURIComponent(String(file).split('/').pop())+tailName(name)+(v?'?v='+v:''); }
function mediaUrl(jid,pid,p){
  p=String(p);
  if(pid && p.indexOf('/materials/')>=0) return matUrl(pid,p);
  const oi=p.indexOf('/output/');
  if(oi>=0) return '/files/'+encodeURI(p.slice(oi+8));
  return '/media/'+encodeURIComponent(jid)+'/'+encodeURIComponent(p.split('/').pop());
}
function mediaSection(jid,m,pid){
  const groups=[['ref_image','参考图','img'],['ref_video','参考视频','video'],
                ['ref_audio','参考音频','audio'],['first_frame','首帧','img'],['last_frame','尾帧','img']];
  let h='';
  groups.forEach(([k,label,kind])=>{
    const arr=m[k]||[]; if(!arr.length) return;
    h+='<div style="margin-top:12px"><div class="muted">'+label+' ('+arr.length+')</div>'+
       '<div class="'+(kind==='audio'?'medialist':'mediagrid')+'">';
    arr.forEach(p=>{
      const u=mediaUrl(jid,pid,p);
      if(kind==='img'){
        let src=u;
        if(pid && String(p).indexOf('/materials/')>=0){
          const mt=matByFile(String(p).split('/').pop());
          if(mt && mt.exists) src=previewUrl(pid,mt.file,mt.thumb_v,matSaveName(mt));
        }
        h+='<img src="'+src+'" loading="lazy">';
      }
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
  $('jTitle').textContent='分镜详情 · '+(j.name? j.name+' · ':'')+id;
  const row=(k,v)=>'<div class="detrow"><span class="muted">'+k+'</span><span>'+v+'</span></div>';
  let h='';
  if(j.name) h+=row('分镜名', esc(j.name));
  h+=row('类型', MODE_CN[j.mode]||j.mode||'-');
  h+=row('时长', p.dur!=null? p.dur+' 秒':'-');
  h+=row('步数', p.steps!=null? p.steps:'-');
  h+=row('seed', j.seed!=null? String(j.seed):'-');
  h+=row('画幅', esc(p.aspect||'-'));
  h+=row('分辨率', p.megapixels!=null? p.megapixels+' MP':'-');
  if(j.status==='failed' && j.err) h+=row('错误', '<span style="color:var(--err)">'+esc(friendlyErr(j.err))+'</span>');
  h+='<div style="margin-top:12px"><div class="muted">提示词</div><div class="detprompt">'+esc(p.prompt||'')+'</div></div>';
  h+=mediaSection(id,m,j.project);
  $('jBody').innerHTML=h;
  bindSinglePlay($('jBody'));
  $('jobModal').classList.add('open');
}
function closeJob(){
  $('jBody').querySelectorAll('video,audio').forEach(o=>o.pause());
  $('jobModal').classList.remove('open');
}
function reuseJob(id){
  const j=jobsById[id]; if(!j) return;
  if(!curProject){ notice('请先进入一个项目'); return; }
  openShotModal('复用分镜', j.name||'', '确定', (name)=>{ doReuse(id,name); });
}
async function doReuse(id,name){
  const j=jobsById[id]; if(!j) return;
  const p=j.params||{}, m=j.media||{};
  shotName=name;
  $('submitMsg').textContent='正在载入分镜 '+id+' 的素材…';
  $('mode').value=(j.mode==='ref2v')?'ref2v':'t2v'; onModeChange();
  $('prompt').value=p.prompt||'';
  if(p.dur!=null) $('dur').value=p.dur;
  if(p.steps!=null) $('steps').value=p.steps;
  if(p.aspect) $('aspect').value=p.aspect;
  if(p.megapixels!=null) $('megapixels').value=p.megapixels;
  if(p.ref_image_size) $('ref_image_size').value=p.ref_image_size;
  $('seed').value='';   // 复用不沿用原 seed，留空=随机，避免复现成同样的视频
  await refreshMaterials();
  await refreshOutputs();
  clearSelMat();
  const match=(kind)=>{
    const ids=[], cls=[];
    for(const path of (m[kind]||[])){
      const fn=String(path).split('/').pop();
      const mt=materials.find(x=>x.file===fn);
      if(mt){ ids.push(mt.id); continue; }
      const c=clipsCache.find(x=>x.name===fn);
      if(c) cls.push(c.rel);
    }
    selMat[kind]=ids; selClip[kind]=cls;
  };
  ['ref_image','ref_video','ref_audio','first_frame','last_frame'].forEach(match);
  renderAllSlots();
  let lost=Object.keys(m).some(k=>Array.isArray(m[k]) && m[k].length && m[k].some(p=>{
    const fn=String(p).split('/').pop();
    return !materials.some(x=>x.file===fn) && !clipsCache.some(x=>x.name===fn);
  }));
  showTaskCard();
  $('submitMsg').textContent='已复用分镜 '+(name||id)+(lost?'（部分素材已不在素材库/产物中，已跳过）':'（未提交）');
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

// ---- material library ----
function clearSelMat(){
  for(const k in selMat) selMat[k]=[];
  for(const k in selClip) selClip[k]=[];
  renderAllSlots();
}
function matById(id){ return materials.find(m=>m.id===id)||null; }
function matByFile(fn){ return materials.find(m=>m.file===fn)||null; }
function slotItems(kind){
  const mats=selMat[kind].map(matById).filter(Boolean).map(m=>({src:'mat',id:m.id,name:m.name}));
  const clips=selClip[kind].map(rel=>clipsCache.find(c=>c.rel===rel)).filter(Boolean)
                              .map(c=>({src:'clip',rel:c.rel,name:c.name}));
  return mats.concat(clips);
}
function itemKey(it){ return it.src==='mat'? ('mat:'+it.id) : ('clip:'+it.rel); }
function promptRefRe(prefix){ return new RegExp('<'+prefix+'\\s+(\\d+)>','g'); }
function promptHasRef(prefix,n){ return new RegExp('<'+prefix+'\\s+'+n+'>').test($('prompt').value); }
function remapPromptRefs(prefix,mapFn){
  const ta=$('prompt'), re=promptRefRe(prefix);
  ta.value=ta.value.replace(re,(m,n)=>{ const nn=mapFn(parseInt(n,10)); return nn? ('<'+prefix+' '+nn+'>') : m; });
}
// Replace a slot's selection: block dropping items still referenced in the prompt,
// then renumber the remaining <Prefix N> references to match the new order.
function setSlotSelection(kind,newItems){
  const prefix=SLOT_CN[kind].prefix, oldItems=slotItems(kind);
  const oldNumByKey=new Map(oldItems.map((it,i)=>[itemKey(it),i+1]));
  const newNumByKey=new Map(newItems.map((it,i)=>[itemKey(it),i+1]));
  if(prefix){
    for(const it of oldItems){
      const key=itemKey(it), n=oldNumByKey.get(key);
      if(!newNumByKey.has(key) && promptHasRef(prefix,n)){
        notice('无法移除：「'+it.name+'」已在提示词中被 <'+prefix+' '+n+'> 引用。\n请先删除提示词中的该引用，再移除。');
        return false;
      }
    }
  }
  selMat[kind]=newItems.filter(it=>it.src==='mat').map(it=>it.id);
  selClip[kind]=newItems.filter(it=>it.src==='clip').map(it=>it.rel);
  if(prefix){
    const oldKeyByNum=new Map(oldItems.map((it,i)=>[i+1,itemKey(it)]));
    remapPromptRefs(prefix,n=>{
      const key=oldKeyByNum.get(n);
      return key? newNumByKey.get(key) : null;
    });
  }
  renderSlot(kind);
  return true;
}
function renderSlot(kind){
  const cfg=SLOT_CN[kind], list=$(cfg.list); if(!list) return;
  const mats=selMat[kind].map(matById).filter(Boolean);
  selMat[kind]=mats.map(m=>m.id);
  const clips=selClip[kind].map(rel=>clipsCache.find(c=>c.rel===rel)).filter(Boolean);
  selClip[kind]=clips.map(c=>c.rel);
  const items=slotItems(kind);
  list.innerHTML='';
  items.forEach((it,i)=>{
    const li=document.createElement('li');
    const nm=document.createElement('span'); nm.className='fname'; nm.textContent=it.name;
    const acts=document.createElement('span'); acts.className='acts';
    if(cfg.prefix){
      const chip=document.createElement('button'); chip.className='chip'; chip.textContent=cfg.prefix+(i+1);
      chip.title='插入引用 <'+cfg.prefix+' '+(i+1)+'>';
      chip.onclick=()=>insertRef(cfg.prefix,i);
      acts.appendChild(chip);
    }
    const rm=document.createElement('button'); rm.className='rm'; rm.textContent='移除';
    rm.onclick=()=>setSlotSelection(kind, items.filter(x=>itemKey(x)!==itemKey(it)));
    acts.appendChild(rm);
    li.appendChild(nm); li.appendChild(acts); list.appendChild(li);
  });
}
function renderAllSlots(){ for(const k in selMat) renderSlot(k); }
function matPreview(m,pid,live){
  const nm=matSaveName(m);
  if(m.kind==='image') return '<img class="thumb" loading="lazy" src="'+thumbUrl(pid,m.file,m.thumb_v,nm)+'">';
  const u=matUrl(pid,m.file,nm);
  if(m.kind==='video') return '<video class="thumb" muted playsinline preload="none" title="'+esc(nm)+'" poster="'+thumbUrl(pid,m.file,m.thumb_v)+'" src="'+u+'"></video>';
  if(live) return '<div class="thumbicon"><span class="audplay">▶</span>'+MAT_KIND_CN[m.kind]+'</div>';
  return '<div class="thumbicon">'+MAT_KIND_CN[m.kind]+'</div>';
}
function renderMaterials(){
  const box=$('matGrid'), pid=curProject; if(!box) return;
  const sig=materials.map(m=>[m.id,m.name,m.kind,m.exists?1:0].join(',')).join('\n');
  if(box._sig!==sig){
    box._sig=sig; box.innerHTML='';
    if(!materials.length){ box.innerHTML='<div class="matempty">还没有素材，先在上方上传。</div>'; }
    [['image','图片'],['video','视频'],['audio','音频']].forEach(([kind,label])=>{
      const list=materials.filter(m=>m.kind===kind);
      if(!list.length) return;
      const sec=document.createElement('div'); sec.className='matgroup';
      const hd=document.createElement('div'); hd.className='matgrouphead';
      hd.textContent=label+' ('+list.length+')';
      const g=document.createElement('div'); g.className='matgrid';
      list.forEach(m=>{
        const d=document.createElement('div'); d.className='matcard';
        const nm=matSaveName(m);
        let pv=matPreview(m,pid,true);
        if(m.kind==='image' && m.exists){
          // show the cached thumbnail but let right-click save the original file
          pv='<div class="thumbwrap"><img class="thumb" loading="lazy" src="'+thumbUrl(pid,m.file,m.thumb_v,nm)+'">'+
             '<a class="thumbdl" href="'+matUrl(pid,m.file,nm)+'" download="'+esc(nm)+
             '" title="双击查看大图（右键另存为原图）" onclick="event.preventDefault()"></a></div>';
        }
        d.innerHTML=pv+'<div class="mb"><div class="nm" title="'+esc(m.name)+'">'+esc(m.name)+'</div>'+
          '<div class="mm">'+MAT_KIND_CN[m.kind]+' · '+fmtSize(m.size)+(m.exists?'':' · 文件缺失')+'</div></div>';
        if(m.exists){
          if(m.kind==='image'){
            const ov=d.querySelector('.thumbdl');
            if(ov) ov.addEventListener('dblclick',()=>viewMaterial(m.id));
          }else if(m.kind==='video'){
            d.firstElementChild.addEventListener('click',()=>viewMaterial(m.id));
          }else if(m.kind==='audio'){
            d.firstElementChild.addEventListener('click',()=>viewMaterial(m.id));
          }
        }
        const ma=document.createElement('div'); ma.className='ma';
        const rm=document.createElement('button'); rm.textContent='删除'; rm.onclick=()=>deleteMaterial(m.id);
        ma.appendChild(rm); d.appendChild(ma);
        g.appendChild(d);
      });
      sec.appendChild(hd); sec.appendChild(g); box.appendChild(sec);
    });
  }
  $('matSub').textContent=materials.length? (materials.length+' 个素材') : '';
}
async function refreshMaterials(){
  if(!curProject) return;
  const r=await api('/api/materials?project='+encodeURIComponent(curProject));
  if(!r) return;
  materials=r.materials||[];
  renderMaterials(); renderAllSlots();
}
function onMatFileChange(){
  const f=$('matFile').files[0]; if(!f) return;
  $('matName').value=f.name.replace(/\.[^.]+$/,'');
  $('matMsg').textContent='';
}
async function uploadMaterial(){
  if(!curProject){ notice('请先进入一个项目'); return; }
  const name=$('matName').value.trim();
  const f=$('matFile').files[0];
  if(!name){ $('matMsg').textContent='请填写素材名称'; return; }
  if(!f){ $('matMsg').textContent='请选择要上传的文件'; return; }
  const fd=new FormData();
  fd.append('project',curProject); fd.append('name',name); fd.append('file',f);
  $('matMsg').textContent='上传中…';
  try{
    const r=await fetch('/api/materials',{method:'POST',body:fd});
    const j=await r.json().catch(()=>({}));
    if(!r.ok){ $('matMsg').textContent='上传失败：'+(j.error||r.status); return; }
    materials=j.materials||materials;
    $('matName').value=''; $('matFile').value='';
    $('matMsg').textContent='已上传：'+((j.material&&j.material.name)||name);
    renderMaterials(); renderAllSlots();
  }catch(e){ $('matMsg').textContent='网络错误'; }
}
async function deleteMaterial(mid){
  const m=matById(mid); if(!m) return;
  const r0=await api('/api/jobs?project='+encodeURIComponent(curProject));
  const used=((r0&&r0.jobs)||[]).filter(j=>j.mode!=='edit' &&
    Object.keys(j.media||{}).some(k=>Array.isArray(j.media[k]) &&
      j.media[k].some(p=>String(p).split('/').pop()===m.file)));
  let msg='删除素材「<b>'+esc(m.name)+'</b>」？';
  if(used.length){
    msg+='<br><br>以下分镜列表用到了该素材：<br>'+
      used.map(j=>'· '+esc(j.name||j.id)).join('<br>')+
      '<br><br>删除后这些分镜的素材引用会失效。';
  }
  const ok=await askConfirm(msg,'删除素材','删除');
  if(!ok) return;
  const r=await fetch('/api/materials/'+encodeURIComponent(curProject)+'/'+encodeURIComponent(mid)+'/delete',
    {method:'POST',headers:{'Content-Type':'application/json'},body:'{}'});
  const j=await r.json().catch(()=>({}));
  if(!r.ok){ notice('删除失败：'+(j.error||j.msg||r.status)); return; }
  materials=j.materials||materials; renderMaterials(); renderAllSlots();
}
async function openPick(kind, mode){
  if(!curProject){ notice('请先进入一个项目'); return; }
  pickMode=(mode==='clip')?'clip':'mat';
  if(pickMode==='clip'){
    await refreshOutputs();
    if(!clipsCache.length){ notice('该项目还没有产物，先生成或用素材库素材。'); return; }
    pickKind=kind; pickSingle=false;
    pickSel=new Set(selClip[kind]);
    $('pickTitle').textContent='选择产物 · '+SLOT_CN[kind].cn;
    $('pickHint').textContent='可多选：点选多个产物作为参考视频';
  }else{
    const mk=SLOT_MEDIA[kind];
    if(!materials.some(m=>m.kind===mk)){
      notice('素材库中还没有'+MAT_KIND_CN[mk]+'素材，请先在「素材库」上传。'); return;
    }
    pickKind=kind; pickSingle=(kind==='first_frame'||kind==='last_frame');
    pickSel=new Set(selMat[kind]);
    $('pickTitle').textContent='选择素材 · '+SLOT_CN[kind].cn;
    $('pickHint').textContent=pickSingle? '单选：点一张'+MAT_KIND_CN[mk]+'素材'
                                       : '可多选：点选多个'+MAT_KIND_CN[mk]+'素材';
  }
  renderPickGrid();
  $('pickModal').classList.add('open');
}
function togglePick(key){
  if(pickSingle) pickSel=new Set(pickSel.has(key)?[]:[key]);
  else if(pickSel.has(key)) pickSel.delete(key); else pickSel.add(key);
  document.querySelectorAll('#pickGrid .matcard').forEach(d=>{   // 只更新选中态，不重建，避免视频重新加载
    const on=pickSel.has(d.dataset.key);
    d.classList.toggle('sel',on);
    const t=d.querySelector('.picktag'); if(t) t.textContent=on?'✓':'';
  });
}
function renderPickGrid(){
  const pid=curProject, box=$('pickGrid'); box.innerHTML='';
  if(pickMode==='clip'){
    if(!clipsCache.length){ box.innerHTML='<div class="matempty">暂无可用产物</div>'; return; }
    clipsCache.forEach(c=>{
      const on=pickSel.has(c.rel);
      const d=document.createElement('div'); d.className='matcard pick'+(on?' sel':'');
      d.dataset.key=c.rel;
      d.onclick=()=>togglePick(c.rel);
      d.innerHTML=(c.shot?'<div class="shotname" title="'+esc(c.shot)+'">'+esc(c.shot)+'</div>':'')+
        '<div class="picktag">'+(on?'✓':'')+'</div>'+
        '<video class="thumb" muted playsinline preload="none" title="'+esc(withExt(c.shot||String(c.name).replace(/\.[^.]+$/,''), fileExt(c.name)))+
          '" poster="/vthumb/'+encodeURI(c.rel)+'" src="/files/'+encodeURI(c.rel)+'"></video>'+
        '<div class="mb"><div class="nm" title="'+esc(c.name)+'">'+esc(c.name)+'</div>'+
        '<div class="mm">产物 · '+fmtSize(c.size)+' · '+c.ts+'</div></div>';
      box.appendChild(d);
    });
    return;
  }
  const list=materials.filter(m=>m.kind===SLOT_MEDIA[pickKind] && m.exists);
  if(!list.length){ box.innerHTML='<div class="matempty">暂无可用素材</div>'; return; }
  list.forEach(m=>{
    const d=document.createElement('div'); d.className='matcard pick'+(pickSel.has(m.id)?' sel':'');
    d.dataset.key=m.id;
    d.onclick=()=>togglePick(m.id);
    d.innerHTML='<div class="picktag">'+(pickSel.has(m.id)?'✓':'')+'</div>'+matPreview(m,pid)+
      '<div class="mb"><div class="nm" title="'+esc(m.name)+'">'+esc(m.name)+'</div>'+
      '<div class="mm">'+MAT_KIND_CN[m.kind]+' · '+fmtSize(m.size)+'</div></div>';
    box.appendChild(d);
  });
}
function pickOk(){
  if(!pickKind) return;
  const old=slotItems(pickKind);
  let newItems;
  if(pickMode==='clip'){
    const clips=Array.from(pickSel).map(rel=>{
      const c=clipsCache.find(x=>x.rel===rel);
      return c? {src:'clip',rel:c.rel,name:c.name} : null;
    }).filter(Boolean);
    newItems=old.filter(it=>it.src==='mat').concat(clips);
  }else{
    const mats=Array.from(pickSel).map(id=>{ const m=matById(id); return m? {src:'mat',id:m.id,name:m.name} : null; })
                              .filter(Boolean);
    newItems=mats.concat(old.filter(it=>it.src==='clip'));
  }
  if(!setSlotSelection(pickKind,newItems)) return;   // blocked by a live prompt reference
  closePick();
}
function closePick(){ $('pickModal').classList.remove('open'); pickKind=null; pickSel=new Set(); }

// selected reference files for one slot, with byte sizes (materials + products)
function refFiles(kind){
  const mats=slotItems(kind).filter(it=>it.src==='mat').map(it=>matById(it.id)).filter(Boolean)
    .map(m=>({name:m.name,size:m.size||0}));
  const clips=slotItems(kind).filter(it=>it.src==='clip').map(it=>clipsCache.find(c=>c.rel===it.rel)).filter(Boolean)
    .map(c=>({name:c.shot||c.name,size:c.size||0}));
  return mats.concat(clips);
}
// MiniMax reference limits; returns an error string or ''
function checkRefLimits(){
  const groups=[['ref_image','图片',30],['ref_video','视频',50],['ref_audio','音频',15]];
  const files=groups.map(([k])=>refFiles(k));
  const total=files.reduce((a,f)=>a+f.length,0);
  if(total>12) return '参考文件合计最多 12 个（当前 '+total+' 个）';
  if(files[2].length && !files[0].length && !files[1].length)
    return '音频不能单独作为参考，请至少再选 1 张图片或 1 段视频';
  let totalBytes=0;
  for(let i=0;i<groups.length;i++){
    const cn=groups[i][1], lim=groups[i][2];
    for(const f of files[i]){
      totalBytes+=f.size;
      if(f.size>lim*1048576)
        return cn+'「'+f.name+'」'+(f.size/1048576).toFixed(1)+'MB，超过 '+lim+'MB 上限';
    }
  }
  if(totalBytes>64*1048576) return '参考文件合计 '+(totalBytes/1048576).toFixed(1)+'MB，超过单次请求 64MB 上限';
  return '';
}

function onModeChange(){
  const m=$('mode').value, ref=(m==='ref2v');
  $('t2vBox').style.display = ref? 'none':'';
  $('ref2vBox').style.display = ref? '':'none';
  $('refImageSizeRow').style.display = ref? '':'none';
  $('promptLabel').textContent = ref
    ? '提示词（点素材后的「图片1 / 视频1 / 音频1」按钮即可插入 <Picture 1> 等引用）'
    : '提示词（可选：从素材库选首帧/尾帧；都不选即纯文生视频）';
}

async function submit(){
  if(!curProject){ notice('请先进入一个项目'); goHome(); return; }
  const mode=$('mode').value;
  const prompt=$('prompt').value.trim();
  if(!prompt){ notice('请填写提示词'); return; }
  const dur=$('dur').value, steps=$('steps').value;
  let media='';
  if(mode==='ref2v'){
    const imgs=selMat.ref_image, vids=selMat.ref_video.length+selClip.ref_video.length,
          auds=selMat.ref_audio;
    if(!imgs.length && !vids && !auds.length){ notice('请至少选择一个参考素材'); return; }
    const limErr=checkRefLimits();
    if(limErr){ notice(limErr,'参考素材超限'); return; }
    media=[imgs.length?imgs.length+'图':null, vids?vids+'视频':null,
           auds.length?auds.length+'音频':null].filter(Boolean).join(' / ');
  }else{
    const ff=selMat.first_frame[0], lf=selMat.last_frame[0];
    media=[ff?'首帧':null, lf?'尾帧':null].filter(Boolean).join(' + ');
  }
  const ok=await askConfirm(
    '项目：'+esc(projName(curProject))+(shotName?'<br>分镜：'+esc(shotName):'')+
    '<br>类型：'+(MODE_CN[mode]||mode)+
    '<br>时长：'+dur+'s · 步数：'+steps+(media?'<br>素材：'+media:'')+
    '<br>提示词：'+esc(prompt.slice(0,100))+(prompt.length>100?'…':''),
    '提交分镜','提交');
  if(!ok) return;
  const fd=new FormData();
  fd.append('project', curProject);
  fd.append('mode', mode);
  if(shotName) fd.append('name', shotName);
  fd.append('prompt', prompt);
  fd.append('dur', $('dur').value); fd.append('steps', $('steps').value);
  fd.append('aspect', $('aspect').value); fd.append('megapixels', $('megapixels').value);
  fd.append('ref_image_size', $('ref_image_size').value);
  if($('seed').value) fd.append('seed', $('seed').value);
  if(mode==='ref2v'){
    selMat.ref_image.forEach(id=>fd.append('ref_image', id));
    selMat.ref_video.forEach(id=>fd.append('ref_video', id));
    selClip.ref_video.forEach(rel=>fd.append('ref_video', 'clip:'+rel));
    selMat.ref_audio.forEach(id=>fd.append('ref_audio', id));
  }else{
    if(selMat.first_frame[0]) fd.append('first_frame', selMat.first_frame[0]);
    if(selMat.last_frame[0]) fd.append('last_frame', selMat.last_frame[0]);
  }
  const xhr=new XMLHttpRequest(); xhr.open('POST','/api/run');
  $('prog').style.display='block'; $('prog').firstElementChild.style.width='0%';
  $('submitBtn').disabled=true; $('submitMsg').textContent='上传中...';
  xhr.upload.onprogress=(e)=>{ if(e.lengthComputable) $('prog').firstElementChild.style.width=(e.loaded/e.total*100)+'%'; };
  xhr.onload=()=>{
    $('submitBtn').disabled=false;
    try{ const r=JSON.parse(xhr.responseText);
      if(xhr.status===202){ $('submitMsg').textContent='已提交: '+r.id; $('prompt').value='';
        shotName=null; $('shotLabel').textContent=''; $('taskCard').style.display='none'; }
      else notice('提交失败: '+(r.error||xhr.status));
    }catch(e){ notice('提交失败: '+xhr.status); }
    setTimeout(()=>{ $('prog').style.display='none'; },600);
    refreshJobs();
  };
  xhr.onerror=()=>{ $('submitBtn').disabled=false; notice('网络错误'); };
  xhr.send(fd);
}

function clearTaskForm(){
  $('mode').value='t2v'; onModeChange();
  $('prompt').value='';
  $('dur').value=5; $('steps').value=8; $('seed').value='';
  $('aspect').value=ASPECTS[0]; $('megapixels').value='0.4'; $('ref_image_size').value='match';
  clearSelMat();
  $('prog').style.display='none'; $('submitMsg').textContent='';
}
async function resetForm(){
  const ok=await askConfirm('清空当前填写的内容并恢复默认参数？','重置','清空');
  if(!ok) return;
  clearTaskForm();
}
function cancelShot(){
  shotName=null;
  $('shotLabel').textContent='';
  $('taskCard').style.display='none';
  clearTaskForm();
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
  if(j && (j.project!==curProject || j.mode==='edit')) j=null;   // 只显示本项目的生成分镜
  if(!j) j=(lastJob && jobsById[lastJob]) || null;
  if(j && (j.project!==curProject || j.mode==='edit')) j=null;
  if(j){
    if(lastJob!==j.id){ lastJob=j.id; logOffset=0; $('log').textContent=''; }
  }else{
    lastJob=null;
  }
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
  const jobs=r.jobs.filter(j=>j.mode!=='edit').slice(0,50);
  const sig=jobs.map(j=>[j.id,j.name||'',j.status,(j.stage&&j.stage.label)||'',
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
    if(j.status==='cancelled'||j.status==='failed'||j.status==='interrupted')
      acts+=' <button class="ghost" onclick="retryJob(\''+j.id+'\')">生成</button>';
    if(j.clip_rel) acts+=' <button class="ghost" onclick="play(\''+j.clip_rel+'\')">查看</button>';
    acts+=' <button class="ghost" onclick="reuseJob(\''+j.id+'\')">复用</button>';
    let thumb='';
    if(j.clip_rel){
      thumb='<video class="jthumb" muted playsinline preload="none" title="'+esc(withExt(j.name||j.id, fileExt(j.clip_rel)))+'" poster="/vthumb/'+encodeURI(j.clip_rel)+
        '" src="/files/'+encodeURI(j.clip_rel)+
        '" onclick="window.play(\''+j.clip_rel+'\')"></video>';
    }else if(j.status==='running'||j.status==='queued'){
      const pct=(j.status==='running'&&j.progress&&j.progress.total)
        ? Math.round(j.progress.cur/j.progress.total*100) : 0;
      thumb='<div class="jthumb ph" title="'+(STATUS_CN[j.status]||j.status)+'"><i style="width:'+pct+'%"></i></div>';
    }else{
      thumb='<div class="jthumb ph static" title="'+(STATUS_CN[j.status]||j.status)+'"></div>';
    }
    const head = j.name
      ? '<b>'+esc(j.name)+'</b><br><b>'+esc(j.id)+'</b>'
      : '<b>'+esc(j.id)+'</b>';
    d.innerHTML=thumb+
      '<span class="meta">'+head+'<br>'+line2+'<br>'+(note||j.created||'')+
      '<br><span class="'+stCls(j.status)+'">'+(STATUS_CN[j.status]||j.status)+'</span></span>'+
      '<span class="jobsacts">'+acts+'</span>';
    box.appendChild(d);
  });
}

async function retryJob(id){
  const r=await fetch('/api/jobs/'+id+'/retry',{method:'POST'});
  const j=await r.json().catch(()=>({}));
  if(!r.ok){ notice('生成失败：'+(j.msg||j.error||r.status)); return; }
  refreshJobs(); refreshProjects();
}
async function jobAct(id,act){
  const isDel=act!=='cancel';
  const j=jobsById[id]||{};
  const label=esc(j.name||id)+(j.name?' <span class="muted">'+id+'</span>':'');
  const ok=await askConfirm((isDel?'删除分镜 ':'取消分镜 ')+'<b>'+label+'</b>？'+
                            (isDel?'<br>将同时删除其产物视频，不可恢复。':''),
                            isDel?'删除分镜':'取消分镜', isDel?'删除':'确定');
  if(!ok) return;
  fetch('/api/jobs/'+id+'/'+act,{method:'POST'}).then(()=>{refreshJobs();refreshOutputs();refreshProjects();}); }

async function refreshOutputs(){
  if(!curProject){ clipsCache=[]; return; }
  const r=await api('/api/outputs?project='+encodeURIComponent(curProject)); if(!r) return;
  clipsCache=r.clips||[];
}

// ---------------------------------------------------------------- timeline
function editAspectInit(){
  if($('eAspect').options.length) return;
  EDIT_ASPECTS.forEach(([v,label])=>{ const o=document.createElement('option'); o.value=v; o.textContent=label; $('eAspect').appendChild(o); });
}
function editPick(){
  editSeq.aspect=$('eAspect').value;
  editSeq.fade_in=parseFloat($('eFadeIn').value)||0;
  editSeq.fade_out=parseFloat($('eFadeOut').value)||0;
}
async function refreshEdit(){
  editAspectInit();
  if(!curProject) return;
  const r=await api('/api/sequence?project='+encodeURIComponent(curProject)); if(!r) return;
  editSeq=r.seq||{aspect:'0',fade_in:0,fade_out:0,clips:[]};
  editAvail=r.clips||[];
  const p=projects.find(x=>x.id===curProject);
  $('editHead').innerHTML='<div class="cardhead"><h2 style="color:var(--fg);font-size:16px">剪辑 · '+esc(p?p.name:curProject)+'</h2>'+
    '<span class="bcbar"><button class="ghost" onclick="backToProject()">返回项目</button></span></div>'+
    '<div class="muted">拖动手柄排序；相邻片段之间可设转场</div>';
  $('eAspect').value=editSeq.aspect||'0';
  $('eFadeIn').value=editSeq.fade_in||0;
  $('eFadeOut').value=editSeq.fade_out||0;
  renderTimeline(); renderPickClips(); refreshEditJobs();
}
function clipLabel(rel){ return String(rel).split('/').pop(); }
function renderTimeline(){
  const box=$('timeline');
  if(!editSeq.clips.length){ box.innerHTML='<div class="muted">时间线为空，从下方「添加片段」加入</div>'; return; }
  let h='';
  editSeq.clips.forEach((c,i)=>{
    const t=c.trans||{type:'cut',dur:0};
    let junc;
    if(i===0){ junc='<div class="junc"><span class="start">起始</span></div>'; }
    else{
      const opts=Object.keys(TRANS_CN).map(k=>'<option value="'+k+'"'+(t.type===k?' selected':'')+'>'+TRANS_CN[k]+'</option>').join('');
      const durs=[0.3,0.5,0.8,1.0,1.5].map(d=>'<option value="'+d+'"'+(Math.abs((t.dur||0)-d)<1e-6?' selected':'')+'>'+d+'s</option>').join('');
      junc='<div class="junc"><span>转场</span><select onchange="setTrans('+i+',this.value)">'+opts+'</select>'+
        (t.type==='cut'?'':'<select onchange="setDur('+i+',this.value)">'+durs+'</select>')+'</div>';
    }
    h+='<div class="tslot" data-i="'+i+'">'+junc+
      '<div class="trow"><span class="thandle" onpointerdown="dragStart(event,'+i+')" title="拖动排序">\u2261</span>'+
      '<video class="thumb" muted playsinline preload="none" title="'+esc(withExt((clipsCache.find(x=>x.rel===c.rel)||{}).shot||String(c.rel).split('/').pop().replace(/\.[^.]+$/,''), fileExt(c.rel)))+
        '" poster="/vthumb/'+encodeURI(c.rel)+'" src="/files/'+encodeURI(c.rel)+'"></video>'+
      '<div class="tmeta"><b>'+esc(clipLabel(c.rel))+'</b>第 '+(i+1)+' 段 · '+esc(TRANS_CN[t.type]||t.type)+(t.type==='cut'?'':' '+t.dur+'s')+'</div>'+
      '<div class="tacts2">'+
        '<button onclick="moveClip('+i+',-1)" title="上移">\u25B2</button>'+
        '<button onclick="moveClip('+i+',1)" title="下移">\u25BC</button>'+
        '<button class="rm" onclick="removeClip('+i+')" title="移除">\u2715</button>'+
      '</div></div></div>';
  });
  box.innerHTML=h;
}
function renderPickClips(){
  const box=$('pickClips');
  if(!editAvail.length){ box.innerHTML='<span class="muted">暂无可用产物</span>'; return; }
  box.innerHTML='';
  editAvail.forEach(c=>{
    const d=document.createElement('div'); d.className='clip'; d.onclick=()=>addClip(c.rel);
    d.innerHTML='<span class="addclip">+</span><video muted playsinline preload="none" title="'+esc(withExt(c.shot||String(c.name).replace(/\.[^.]+$/,''), fileExt(c.name)))+
      '" poster="/vthumb/'+encodeURI(c.rel)+'" src="/files/'+encodeURI(c.rel)+'"></video>'+
      '<div class="cap">'+c.name.slice(0,20)+'<br>'+fmtSize(c.size)+'</div>';
    box.appendChild(d);
  });
}
function setTrans(i,v){
  editPick();
  editSeq.clips[i].trans={type:v,dur:(v==='cut'?0:((editSeq.clips[i].trans&&editSeq.clips[i].trans.dur)||0.5))};
  renderTimeline();
}
function setDur(i,v){ editPick(); editSeq.clips[i].trans.dur=parseFloat(v)||0.5; renderTimeline(); }
function addClip(rel){ editPick(); editSeq.clips.push({rel:rel,trans:{type:'cut',dur:0}}); renderTimeline(); $('editMsg').textContent='已加入，记得保存'; }
function removeClip(i){ editPick(); editSeq.clips.splice(i,1); renderTimeline(); $('editMsg').textContent='已移除，记得保存'; }
function moveClip(i,d){
  editPick(); const j=i+d, a=editSeq.clips;
  if(j<0||j>=a.length) return;
  [a[i],a[j]]=[a[j],a[i]]; renderTimeline(); $('editMsg').textContent='顺序已改，记得保存';
}
let dragSlot=null;
function dragStart(e,i){
  if(e.button!=null && e.button!==0) return;
  editPick();
  const slot=e.target.closest('.tslot'); if(!slot) return;
  dragSlot=slot; slot.classList.add('dragging');
  const move=(ev)=>{
    if(!dragSlot) return;
    const el=document.elementFromPoint(ev.clientX,ev.clientY);
    const over=el&&el.closest?el.closest('.tslot'):null;
    if(over&&over!==dragSlot){
      const r=over.getBoundingClientRect();
      over.parentNode.insertBefore(dragSlot, ev.clientY> r.top+r.height/2? over.nextSibling: over);
    }
    ev.preventDefault();
  };
  const up=()=>{
    document.removeEventListener('pointermove',move); document.removeEventListener('pointerup',up);
    slot.classList.remove('dragging');
    const order=[...$('timeline').querySelectorAll('.tslot')].map(x=>+x.dataset.i);
    const before=editSeq.clips.map(c=>c.rel).join('|');
    if(order.length===editSeq.clips.length){ const a=editSeq.clips; editSeq.clips=order.map(k=>a[k]); }
    dragSlot=null; renderTimeline();
    if(editSeq.clips.map(c=>c.rel).join('|')!==before) $('editMsg').textContent='顺序已改，记得保存';
  };
  document.addEventListener('pointermove',move,{passive:false});
  document.addEventListener('pointerup',up);
  e.preventDefault();
}
async function saveEdit(quiet){
  editPick();
  const r=await fetch('/api/sequence',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({project:curProject,seq:editSeq})});
  const j=await r.json().catch(()=>({}));
  if(!r.ok){ if(!quiet) notice('保存失败：'+(j.error||r.status)); return false; }
  editSeq=j.seq||editSeq;
  if(!quiet) $('editMsg').textContent='已保存';
  return true;
}
async function renderEdit(){
  editPick();
  if(!editSeq.clips.length){ notice('时间线为空'); return; }
  const ok=await askConfirm('按当前时间线渲染成片？共 <b>'+editSeq.clips.length+'</b> 段。','预览/导出','渲染');
  if(!ok) return;
  if(!await saveEdit(true)){ notice('保存失败'); return; }
  const r=await fetch('/api/edit/render',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({project:curProject})});
  const j=await r.json().catch(()=>({}));
  if(r.status!==202){ notice('提交失败：'+(j.error||r.status)); return; }
  editLast=j.id; editLogOff=0; $('editLog').textContent=''; $('editMsg').textContent='渲染已提交：'+j.id;
  refreshEditJobs();
}
async function refreshEditJobs(){
  if(!curProject) return;
  const r=await api('/api/jobs?project='+encodeURIComponent(curProject)); if(!r) return;
  const jobs=(r.jobs||[]).filter(j=>j.mode==='edit');
  const box=$('editList');
  const sig=jobs.map(j=>[j.id,j.status,j.clip_rel||'',j.clip_seconds||''].join(',')).join('\n');
  if(box._sig!==sig){
    box._sig=sig;
    if(!jobs.length){ box.innerHTML='<span class="muted">暂无成片</span>'; }
    else{
      box.innerHTML='';
      jobs.forEach(j=>{
        const d=document.createElement('div'); d.className='clip';
        const cap=(j.clip_seconds?j.clip_seconds+'s · ':'')+(STATUS_CN[j.status]||j.status)+' · '+(j.created||'');
        if(j.clip_rel){
          d.onclick=()=>play(j.clip_rel);
          d.innerHTML='<video muted playsinline preload="none" title="'+esc(withExt(String(j.clip_rel).split('/').pop().replace(/\.[^.]+$/,''), fileExt(j.clip_rel)))+
            '" poster="/vthumb/'+encodeURI(j.clip_rel)+'" src="/files/'+encodeURI(j.clip_rel)+'"></video>'+
            '<div class="cap">'+esc(j.id)+'<br>'+esc(cap)+'</div>'+
            '<button class="addclip" style="right:auto;left:6px;background:var(--err)" onclick="event.stopPropagation();delEdit(\''+j.id+'\')">删除</button>';
        }else{
          d.innerHTML='<div style="aspect-ratio:16/9;display:flex;align-items:center;justify-content:center;color:var(--mut)">'+esc(STATUS_CN[j.status]||j.status)+'</div>'+
            '<div class="cap">'+esc(j.id)+'<br>'+esc(cap)+'</div>';
        }
        box.appendChild(d);
      });
    }
  }
  if(editLast){ const j=(r.jobs||[]).find(x=>x.id===editLast); if(j) renderEditCurrent(j); }
}
async function delEdit(id){
  const ok=await askConfirm('删除成片 <b>'+id+'</b>？（视频文件一并删除）','删除成片','删除');
  if(!ok) return;
  const r=await fetch('/api/edit/delete',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({id})});
  const j=await r.json().catch(()=>({}));
  if(!r.ok){ notice('删除失败：'+(j.error||j.msg||r.status)); return; }
  if(editLast===id) editLast=null;
  $('editList')._sig=null; refreshEditJobs();
}
function renderEditCurrent(j){
  $('editCur').innerHTML='<div class="curState"'+(j.status==='failed'?' style="color:var(--err)"':'')+'>'+friendlyStatus(j)+'</div>'+
    (j.status==='running'&&j.progress&&j.progress.total?'<div class="bar"><i style="width:'+Math.round(j.progress.cur/j.progress.total*100)+'%"></i></div>':'')+
    '<div class="curMeta"><b>'+j.id+'</b>'+(j.clip_rel?'<br><button class="ghost" onclick="play(\''+j.clip_rel+'\')">播放成片</button>':'')+'</div>';
}
async function editTick(){
  if(!editLast) return;
  const j=await api('/api/jobs/'+editLast); if(!j) return;
  renderEditCurrent(j);
  const r=await api('/api/jobs/'+editLast+'/log?offset='+editLogOff);
  if(r){ if(r.text){ $('editLog').textContent+=r.text; $('editLog').scrollTop=$('editLog').scrollHeight; } editLogOff=r.offset; }
  if(j.status!=='running'&&j.status!=='queued'){ editLast=null; $('editList')._sig=null; refreshEditJobs(); refreshProjects(); }
}

function play(rel){ $('mvideo').src='/files/'+encodeURI(rel); $('mcap').textContent=rel;
  $('modal').classList.add('open'); $('mvideo').play().catch(()=>{}); }
function closeModal(){ $('mvideo').pause(); $('mvideo').src=''; $('modal').classList.remove('open'); }

function previewUrl(pid,file,v,name){ return '/preview/'+encodeURIComponent(pid)+'/'+encodeURIComponent(String(file).split('/').pop())+tailName(name)+(v?'?v='+v:''); }
function viewMaterial(mid){
  const m=matById(mid); if(!m) return;
  if(!m.exists){ notice('素材文件缺失：'+m.name); return; }
  const nm=matSaveName(m), u=matUrl(curProject,m.file,nm), pid=curProject;
  $('viewOrig').style.display = m.kind==='image' ? '' : 'none';
  let h;
  if(m.kind==='image'){
    // show the cached thumbnail at once, then swap in the downscaled preview
    h='<img src="'+thumbUrl(pid,m.file,m.thumb_v,nm)+'" data-preview="'+previewUrl(pid,m.file,m.thumb_v,nm)+'" data-orig="'+u+'">';
  }else if(m.kind==='video'){
    h='<video controls autoplay playsinline title="'+esc(nm)+'" src="'+u+'"></video>';
  }else{
    h='<audio controls autoplay title="'+esc(nm)+'" src="'+u+'"></audio>';
  }
  $('viewCap').textContent=m.name+' · '+MAT_KIND_CN[m.kind];
  $('viewBody').innerHTML=h;
  $('viewBody').querySelectorAll('video,audio').forEach(o=>o.play().catch(()=>{}));
  $('viewModal').classList.add('open');
  if(m.kind==='image'){
    const el=$('viewBody').querySelector('img');
    if(el){ const pre=new Image(); pre.onload=()=>{ if(el.isConnected) el.src=pre.src; }; pre.src=el.dataset.preview; }
  }
}
function viewOriginal(){
  const el=$('viewBody').querySelector('img'); if(!el) return;
  el.src=el.dataset.orig; $('viewOrig').style.display='none';
}
function closeView(){
  const b=$('viewBody');
  b.querySelectorAll('video,audio').forEach(o=>{ o.pause(); o.removeAttribute('src'); if(o.load) o.load(); });
  b.innerHTML=''; $('viewOrig').style.display='none';
  $('viewModal').classList.remove('open');
}

async function releaseVram(){
  const ok=await askConfirm('停止 ComfyUI 并释放显存？下次生成需冷启动。','释放显存','停止');
  if(!ok) return;
  fetch('/api/service/stop',{method:'POST'}).then(async r=>{ const j=await r.json(); notice(j.msg||'ok'); refreshState(); }); }

async function optimizePrompt(){
  const prompt=$('prompt').value.trim();
  if(!prompt){ $('submitMsg').textContent='请先填写提示词'; return; }
  const counts = $('mode').value==='ref2v'
    ? {ref_image:selMat.ref_image.length,
       ref_video:selMat.ref_video.length+selClip.ref_video.length,
       ref_audio:selMat.ref_audio.length}
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
setInterval(()=>{ const v=$('editView'); if(v && v.style.display!=='none') editTick(); },1500);
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
