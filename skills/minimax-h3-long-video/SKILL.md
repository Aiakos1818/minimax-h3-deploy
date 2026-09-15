---
name: minimax-h3-long-video
description: Use when 用户给出一段剧情大纲、小说片段或剧本，要把它做成多段长视频：先完善分镜（镜头表、景别机位、光线色调、每镜提示词），再出 Z-Image 首帧锚图、按段生成、pyav 拼接成片（2.39:1 裁幅）。Trigger words 大纲, 剧情, 分镜, 镜头表, 提示词, 长视频, 成片, 短剧, 首帧锚图, 续接链, chain_director, MiniMax H3, ~/MiniMax-H3-Deploy。Use ONLY for that pipeline; not for generic video editing.
---

# 剧情大纲 → 分镜 → 长视频成片（MiniMax H3 常驻档）

输入一段**剧情大纲/小说/剧本**：先产出**完整分镜表 + 每镜提示词**给用户审，通过后再出锚图、逐段生成、
拼接成片。创作部分（阶段 A）是主体；制作部分（阶段 B）只是执行后端。

## 0. 先读事实来源（不要凭记忆）

| 文件 | 内容 |
|---|---|
| `~/MiniMax-H3-Deploy/AGENTS.md` | 仓库规矩（提交前必须问用户） |
| `~/MiniMax-H3-Deploy/docs/chain_director_v3.md` | 续接链引擎：常驻服务、slot 语义、生命周期 |
| `~/MiniMax-H3-Deploy/workflows/api/README.md` | 单镜头 API 模板与打断注意 |
| 本 skill 的 `reference/` | `storyboard.md`（分镜方法与词库）、`film-spec.md`（spec 字段）、`pitfalls.md`（坑）、`pingfan_01.md`（实战） |

要点（先验证再用）：双 RTX 2080Ti、**21.48GB/卡**上限；生成 **864×480、24fps**；单段 ≤**226 帧**；
续接链每段"净新内容"由 `--dur` 秒数决定，raw 帧 ≈ `dur*24 + 39`（`HEAD=39` 合并时裁掉）；s/it 随帧数
6.6(107f) → 19.5(226f)；服务**默认常驻**（每实例只付一次冷启动 ~3min）；控制台 `:8190`；
宿主机**没有 ffmpeg**，用 `~/ComfyUI-Deploy/comfyenv/bin/python`（pyav）；本机内存只有 15GB，
换模型时会 swap 抖动（锚图阶段常常"卡" 2–4 分钟，其实在装载）。

## 1. 总流程

```
大纲 ──A1 规模测算──> 时长/镜头预算 ──A2 拆 beat──> 镜头与链的划分
     ──A3 视觉语法+人物块──> ──A4 镜头表 + A5 提示词──> film_spec.json ──用户审分镜──>
     ──B1 锚图（Z-Image，必须最先做完）──> 用户挑图 ──B2 逐段生成（chain_director_v3）
     ──B3 拼接（scripts/assemble.py）──> B4 验证 ──> 看片迭代
```

## 阶段 A：分镜（创作主体）

### A1 规模测算
- 有旁白文本：**字数 ÷ 3.3–4.5 字/秒** = 旁白时长（中文朗读）；成片时长按旁白定，不按"模型能出多久"定。
- 只有大纲：按 beat 数量估（1 个动作/场景 ≈ 4–6s）；不确定就直接问用户要几分钟的片子。
- 镜头预算 = 时长 ÷ 5s（日常 5s/段；要长镜头就 2–4 段串成链，单段上限 ~9s）。

### A2 拆 beat → 镜头 / 链
- 切镜依据：**时空变化、动作完成、情绪转折**。1 镜 = 1 个动作或 1 个画面。
- **同一动作连续推进 → 一条链**（长镜头，如"下车→走过来→上车"）；**跨时空/跨景别 → 独立镜头**（硬切）。
- ⚠️ 禁止把时间跳跃/闪回交给续接链：续接节点强制画面连续，跳跃会被抹成缓慢渐变。闪回要另开一条链，
  在后期用硬切/交叉溶解接。
- 结构技巧：设计**呼应镜头**（同一只手、同一构图的前后对照）作为全片支点；结尾用定格+淡出收。

### A3 视觉语法 + 人物块
- 先定总 look（年代/胶片/光比/调色/机位风格），必要时**按时段或情绪分套**（例：现实=冷青灰+手持呼吸；
  记忆=琥珀暖+稳定器流动）。写进 `looks{}`，每次原样复用。
- 人物写成 **CHARACTER 块**（年龄、脸、发型、服装、气质）**逐字复用**——无 LoRA 时这是唯一的一致性手段；
  跨时段（如 50 岁/25 岁）本就该不同脸，靠服装与构图呼应。
- 手、器物、文字是 AI 弱项：提示词里写死"one hand only, exactly five fingers"，负面里加
  "no text/subtitles/logos/watermarks, no distorted hands, no extra fingers, no morphing faces, no CG look"。

### A4 镜头表（交付给用户审的形态）
段号/tag｜时长｜生成方式（独立 / 链第 N 段）｜景别·机位·镜头｜画面动作｜光线色调｜声音｜转场｜锚图｜seed。
模板与镜头语言词库见 `reference/storyboard.md`。

### A5 提示词公式（每镜一条）
```
LOOK（本段所属的 look 块）+ CHARACTER（本镜人物块）+ SHOT（景别/机位/动作/运动）+
AUDIO（同期声，H3 是音视频联合生成）+ NEGATIVE（固定尾巴）
```
- 语言建议英文（镜头术语映射准，已验证）；中文注释给人看。
- 链的基底 `--prompt` 只能写**共有时空/人物/光**（同一 run 不可改），动作推进全部塞进 `--beat` 后缀。

### A6 产出 `film_spec.json`（唯一交付物）
按 `reference/film-spec.md` 填；可以直接复制 `scripts/film_spec.example.json`（pingfan 的真实数据）改。
**动手出图前必须把分镜表 + spec 给用户审**——GPU 时间贵，改分镜最便宜。

### A7 命名（tag / anchor / 文件名）——用有意义的名字，不要序号

| 项 | 规则 | 例（《平凡的人生》） |
|---|---|---|
| `film` | 作品短 slug，ASCII 小写 `[a-z0-9_]{2,20}` | `pingfan` |
| `tag` | **`<film>_<shot-slug>`**；slug = 2–3 个词描述**这一镜的内容**（地点/主体/动作） | `pingfan_balcony`、`pingfan_agespot`、`pingfan_screen`、`pingfan_ride`、`pingfan_gate` |
| `anchor` key | 复用同一个 slug → `anchors/<slug>_best.png` | `balcony`、`ride`… |
| 链 | 一条链**只用一个 tag**（段间推进靠 `--beat`，不靠编号） | `pingfan_ride` = 骑行 3 段 |
| 镜头顺序 | 只体现在 spec 的 `jobs[]` 与 `assemble.clips[]` | — |

- ❌ 不要 `pf01s1` / `shot1` / `m2` 这种纯序号：产物一多就分不清哪条是什么。
- 产物因此长这样：`output/video/chain/pingfan_balcony/seg_*.mp4`、`output/final_pingfan_ride.mp4`（链）、
  `output/final_pingfan.mp4`（成片）、`output/pingfan/`（锚图）、`projects/pingfan/`。
- 约束：≤60 字符、无空格/斜杠/点；中文可行（控制台校验 `^[\w\-]{1,60}$`，driver 不校验）但**建议 ASCII**
  ——URL、终端、跨工具都省转义。
- tag 只在“同一片内”需要唯一；**跨片靠 film 前缀区分**，所以前缀别省。

## 阶段 B：制作

### B1 锚图（Z-Image Turbo，必须在任何链开跑前做完）
```bash
cd ~/.config/opencode/skill/minimax-h3-long-video
python3 scripts/anchors.py generate --spec scripts/film_spec.example.json          # 可加 --dry-run
python3 scripts/anchors.py contact  --spec scripts/film_spec.example.json
# 用户在控制台看图：http://127.0.0.1:8190/files/<film>/anchors.html （/files/ 只能服务 output/ 下的文件）
python3 scripts/anchors.py pick s1=b s2=a --spec scripts/film_spec.example.json
```
- 每镜 2–3 个 seed（1024×576、8 步、cfg1、res_multistep、ModelSamplingAuraFlow shift3）；
  手部镜头（老人斑特写、握车把）多看几张。
- 常驻 FSDP 占 ~12G/卡，Z-Image 挤不进去：**先出完所有锚图**再开链（服务空闲时也该如此排期）。
- 锚图 1024×576 会被关键帧节点按生成尺寸（`ref_image_size: match`）缩放，不影响显存。

### B2 逐段生成（`chain_director_v3.py`）
```bash
python3 scripts/film.py run --spec scripts/film_spec.example.json          # 可加 --only pingfan_ride --dry-run
```
脚本按 spec 组命令，等价于：
- 独立镜头：`--segments 1 --first-image projects/<film>/anchors/<shot>_best.png`
- 链：`--segments N --first-image ... --beat "5s:<第2段正文>" --beat "10s:<第3段正文>" --merge`
- 每任务都要 `--clean`（slot 文件名**全局不分 tag**，残留会让引擎误判"该段已存在"）
- `--dur` 是净新秒数；`--beat` 归属 `seg = int(t // dur)`
- 服务默认常驻（跑完不释放），要释放加 `--stop-when-done` 或 `~/ComfyUI-Deploy/stop.sh`

### B3 拼接
```bash
python3 scripts/film.py assemble --spec scripts/film_spec.example.json
```
脚本调用仓库里的 `scripts/assemble.py`：硬切拼接 + 居中裁幅（`--aspect 2.39` → 864×362）+ 头淡入 +
末帧定格 + 淡出，音频 30ms 边沿淡化防爆音，输出 `output/final_<film>.mp4`（控制台「输出库」可见）。

### B4 验证
```bash
python3 scripts/film.py verify --spec scripts/film_spec.example.json
```
pyav 校验每段与成片的帧数/时长/尺寸/音频/首末帧亮度/切点帧号（切点应落在各段边界）。

## 决策规则速查

| 情况 | 做法 |
|---|---|
| 时长怎么定 | 有旁白按字数换算；只有大纲按 beat 估或问用户 |
| 一段多长 | 默认 5s（净新）；4–5s 最稳，>7s 变慢且更易漂 |
| 独立镜头 vs 链 | 跨时空/景别 → 独立硬切；同动作连续 → 链 |
| 一致性 | CHARACTER 块逐字复用 + 锚图；跨时段的角色本就换脸 |
| 质量不够 | 先调提示词，再提 `--steps`（8 → 12–16）；低步数音频底噪明显 |
| 重跑某段 | 独立镜头：`--clean` 后重跑该 tag；链：整链重跑（`--clean`） |
| 旁白/音乐 | 先出无旁白画面，最终时长定了再录/配，后期混音 |

## 常踩的坑（详见 `reference/pitfalls.md`）

1. 忘了 `--clean` → 引擎按残留 slot 续拍，产物张冠李戴。
2. 把闪回塞进一条链 → 画面被抹成渐变。
3. 锚图与链生成交叉进行 → 显存互抢 OOM；必须"锚图全部完成后才开链"。
4. 编辑 `scripts/assemble.py` 时**先建好所有流再编码**——在另一个流 flush 之后再 `add_stream`，muxer 会
   SIGFPE（本次真实踩过）；没 ffmpeg，验证一律用 pyav。
5. driver 的 stdout 是块缓冲：跑任务加 `python -u`，否则看不到进度；监控改看 ComfyUI 的 tqdm、
   `/history` 计数、`output/h3_continuous/` 槽位文件、`nvidia-smi`、`/queue` 里的 `client_id`。
6. `POST /interrupt` 打断后主进程 VAE 残留 ~2–4G，紧接着提交可能在第 1 步 OOM → 重启服务。
7. `ui2api.py` 不支持 subgraph（Z-Image 工作流就是），其 API 模板手工维护。
8. 本机 15GB 内存，换模型时 swap 抖动是"假卡死"（GPU 0%/几百 MB），再等 2–4 分钟。

## 与仓库工具的关系

- `scripts/chain_director_v3.py`：续接链引擎（`--segments/--dur/--steps/--beat/--first-image/--last-image/--clean/--merge/--stop-when-done`）。
- `scripts/chain_director_v3_web.py`（`:8190`）：控制台；`/api/outputs`、`/api/state`、`/files/<rel>`（只服务 `output/` 下）。
- `scripts/assemble.py`：拼接/裁幅/淡入淡出/定格（pyav）。
- `workflows/api/api_image_z_image_turbo.json`：出锚图的 API 模板；`scripts/ui2api.py`：UI→API 转换器（有 subgraph 会拒绝）。
- `projects/<film>/`（已 gitignore）：`anchors/<shot>_best.png`、`out/`、`run.log`。

## 规矩

- 分镜先给用户审；素材/成片先在控制台预览；**提交/推送前必须问用户**。
- 新增/修改 opencode skill 后需重启 opencode 才会加载。
