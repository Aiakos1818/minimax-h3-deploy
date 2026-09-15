# 工作节点 4：chain_director_v2.py（persist 链 + 首段素材分流，现行主线）

> 阶段：v2 = v1 引擎 + **persist 复用**（任务内 segments 2..N 复用常驻 FSDP，不每次
> reload）+ **首段素材分流**（fl2va 文/首末帧锚，ref2va 参考图/视频/音频）。
> v1 不再维护，一切改动落在 v2。

## 一、运行语义与 CLI

```bash
cd ~/MiniMax-H3-Deploy
# fl2va 首段：纯文本
~/ComfyUI-Deploy/comfyenv/bin/python scripts/chain_director_v2.py --tag myfilm --segments 6 --dur 5 \
  --prompt "全局设定/人物/风格/机位一句话" --beats "5:要点;12:要点"
# fl2va 首段：首/末帧图锚（I2V）
~/ComfyUI-Deploy/comfyenv/bin/python scripts/chain_director_v2.py --tag f --segments 4 \
  --prompt "..." --first-image img/open.png --last-image img/end.png
# ref2va 首段：参考图/视频/音频驱动
~/ComfyUI-Deploy/comfyenv/bin/python scripts/chain_director_v2.py --tag r --segments 4 \
  --prompt "the subject in <Picture 1> walks, camera follows the motion in <Video 1>" \
  --ref-image ref/a.png --ref-image ref/b.png   # <=9 张，<Picture N>
  --ref-video clip/in.mp4                        # <=3 段，24fps 帧，<Video k>；自带音轨作 <Audio k>
  --ref-audio steps.wav                          # <=3，独立参考音频（脚步声/音乐），见 audio.md
  --ref-image-size match                         # match|max
# 全部参数
#   --segments 段数 | --dur 每段净秒数 | --seed | --steps 默认 8 | --width/--height 864x480 起
#   --tag | --merge
```

- **ComfyUI 生命周期托管（现行）**：chain_director_v2.py **全权接管 ComfyUI 进程**。
  只要本次真正要跑段（`existing clips < --segments`）：
  - 跑段前无条件接管服务——**若已在运行则先 stop 再 start**（down 时 `stop.sh` 无害返回），
    并轮询 :8188 就绪（实测 ~10–16 s）；
  - 跑段结束（**不管什么原因**：成功、失败/异常 `sys.exit`、或 Web 取消发来的
    `SIGTERM`/`SIGINT`）一律 `stop.sh` 关闭，GPU 在任务间完全释放。
  - 不跑段的场景（`existing >= --segments` 且仅 merge、或所有段已在）**不启不停**。
  - Web（`chain_director_v2_web.py`）每个任务本来就是 spawn CLI，因此自动获得同样的
    启停，无需额外配置。
  - **`--clear` 在托管模式下被忽略**（跑段前恒为 stop→start），`restart|prewarm|none`
    仅保留解析，不再分支执行（见二节历史语义）。

- **互斥**：fl2va 图集（`--first/--last-image`）与 ref2va 素材集（`--ref-image/--ref-video/
  --ref-audio`）不可混用，混用即 exit。
- 约束：ref 图 ≤9、ref 视频 ≤3、ref 音频 ≤3。
- `--merge`：把各段 raw 按 `clip1[0:safe]`、`clipN[head:safe]`、`last[head:end]` 拼成
  `output/final_<tag>.mp4`（stitch 时间戳修复与 v1 相同，见 v1 文档六节）。
- 续拍：`existing clips < --segments` 时只补跑缺失段并重新 merge。

产物：`output/video/chain/<tag>/seg_*_*.mp4`（raw）、
`output/h3_continuous/chain_*.safetensors`（逐段 AV latent + handover，slot 即 clip 号，
**全局不区分 tag**）、`output/final_<tag>.mp4`。

## 二、persist 模式（raylight 复用改动）

**目标**：同一运行内 segments 2..N 复用常驻 FSDP（不每次 reload），同时每次运行前自动
清场一次，保证新 prompt 的 CLIP dispatch 不 OOM。v1 的 clear=True 每段重载被"任务内复用
+ 跨任务清场"取代。

**raylight 改动**（`custom_nodes/raylight`，在 gen_dual 部署适配之外的新增；改前原版存
`.bak_pre_persist`）：
1. `nodes.py` `RayInitializer.spawn_actor`：
   - 新参数 `reuse_epoch`（INT, advanced, 默认 0），注入 `parallel_dict["reuse_epoch"]`；
   - **幂等复用 guard**：`ray.is_initialized()` 且存活 `RayWorker:0` 且 `parallel_dict`
     关键字段（含 `reuse_epoch`）一致 → 直接组装现 workers 返回，跳过
     `ray.shutdown/init/新建`；
   - **persist-only**：guard 仅在 `clear_vram_after_sampling=False` 时启用；
     clear=True（v1）一律走原重建路径。
2. `distributed_worker/ray_worker.py` `ensure_fresh_actors`：`is_model_loaded` 触发重启
   仅当非 persist（`clear_vram_after_sampling` 为 True）；persist 下 loaded 是预期常驻态。
3. `ray_worker.py` `set_state_dict`：persist 复用时 `load_unet` fast-path 跳过重载、
   `state_dict` 已被采样置 None → 增加宽松（FSDP + model 已加载则 no-op），否则报
   "Worker state_dict is None"。

**chain_director_v2.py 清场语义（`--clear`）**：
- 模块级 `_RUN_EPOCH = time.time_ns()`（每次进程唯一），注入每段 RayInitializer。
- 正式段 queue 前，若有待生成段，先执行一次 **GPU 清场**：
  - `restart`（默认）：stop.sh + start-comfyui-for-minimax-h3.sh 重启服务（实测 **~10–12 s** 回到 :8188，
    全进程/显存彻底归零），首段 RayInitializer 在段 queue 内 init（~25 s）。
  - `prewarm`：不重启服务，提交 `RayInitializer(+本次 epoch)` + `RayCleanVRAMUsed`
    （OUTPUT 透传节点，使图有输出、不 load 模型）。上次残留（epoch 不同）→ guard 不匹配
    → `ray.shutdown`+新建 workers → GPU 归零（实测 **~30 s**，比 restart 慢，但可连跑
    不中断服务）。
  - `none`：不清场（热连跑，仅当已知 GPU 无残留）。
- 段 queue 复用同一 epoch → guard 命中 → 复用（segment 2..N 模型驻留不 reload）。

> **2026-09-05 托管模式修正**：上述 `--clear` 三态在**托管生命周期**开启时失效——跑段前
> 恒走 `restart`（stop→start）作为统一前门（见一节），`prewarm`/`none` 仅作历史语义保留，
> 不再被 driver 调用；prewarm 保留（`prewarm_clear()` 定义仍在）供手动复用。托管模式下
> 每次任务收尾都会 stop ComfyUI，因此跨任务"热连跑不清场"（`none`）失去意义。

**为何不能靠"段 queue 内重建"清场**：段内 Start/encode 节点不依赖 RayInitializer，
ComfyUI 可能先 encode（旧 FSDP 仍驻留）→ OOM；清场必须在独立 queue 完成。

**显存策略（2026-09-05 修正：勿段间 `unload_all_models`）**：persist 时 FSDP 分片常驻
ray worker（~11.5–12 G/卡），ComfyUI 主进程解码后还会把 video/audio VAE 留在 GPU
（大负载下 ~10 G）。曾用段间 `POST /free`（`unload_all_models`）清主进程，实测**会连带
驱逐 raylight worker actor** → 下段 RayInitializer guard lookup 失败 → rebuild + 重载
FSDP（续段不再省时）。修复：
- **start-comfyui-for-minimax-h3.sh 增加 `--lowvram --reserve-vram 11.5`**：让 Comfy 认知可用显存≈真实剩余
  （worker 驻留后），主进程 VAE 用后回落（实测段间 ~1.8 G）。验证 864×480/20 步/2 段：
  clip1 718s → clip2 670s，persist 生效且不 OOM。
- `raylight nodes.py` spawn_actor 的幂等 guard 内保留 `[GUARD]` 诊断打印（命中/异常原因），
  排障时看 `~/ComfyUI-Deploy/comfy.log`。

**persist 适用边界（2026-09-05）**：FSDP 只在本段与上一段 **prompt 相同**（cond 缓存
HIT → 免 Qwen 双卡 dispatch）时复用；一旦 seg_prompt 变化（beats 逐段注入、或续拍换了
剧情）→ cond MISS → 需 Qwen dispatch（~24 G/双卡），与驻留 worker（11.3 G/卡）物理冲突
必 OOM。driver 现在按 **seg_prompt 变化点**动态给 `reuse_epoch`：词变的那一段 epoch 递增
→ 空 GPU 完成 dispatch（ray rebuild + FSDP 重载，约 +1~2 min），其后同词段继续复用。
实测：同词 2 段 persist 快路径 OK；每段 beat 各异的 3 段（dur10/20 步/864×480 + 3 beats）
clip1 697s → clip2 904s → clip3 896s 全跑通无 OOM。

**CLIP cond 缓存落盘布局**：Qwen 文本→cond 的模型并行 encode 结果缓存在磁盘
`~/MiniMax-H3-Deploy/cond_cache/`（实现于 `H3MultiGPUCLIPLoader.encode_model_parallel`，
本地变更基线副本 `scripts/_h3mp/__init__.py`）。key = sha256(clip_name + tokens 规范
json + unprojected + add_dict)；纯文本只依赖 (clip, tokens)，是确定性函数，缓存**跨进程、
跨 ComfyUI 重启有效**。目录组织（两层，各一层名字）：

```
cond_cache/
└── <key 前 2 位 hex>/            # 第一层分片，避免单目录条目过多
    └── <完整 64 位 sha256 key>/  # 一条缓存
        ├── meta.json             # {"entries":[...], "ntensors":N}；张量型 meta 记
        │                         #   {"__tensor_file__":"meta<i>_<j>"}，非张量直接内联 JSON
        └── tensors.safetensors   # cond<i>（encode 输出）+ meta<i>_<j>（meta 张量）
```

读侧（HIT）：两文件齐全 → 按 meta.json 重组回 `[(cond, meta), ...]` 还原 conditioning，
**零 dispatch**；任一缺失/解析异常降级 miss 照常 encode 后重写。条目大小随文本长度/张量
规模浮动（实测 30 条 123 KB–17.5 MB）。**清缓存**：`rm -rf ~/MiniMax-H3-Deploy/cond_cache/*`
（目录保留无碍）。

**HIT/MISS 判定与"整链一次性"语义**：缓存按段文本查询，但**同文本全链共享同一个 key**。
无 beats、纯 prompt 的任务段 1..N 的 seg_prompt 完全相同 → 段 1 首次 encode MISS
（dispatch+encode+写盘一次），段 2..N 同 key 全 HIT；**连从未轮到采样的后续段，其 cond
也在段 1 那次已落盘**。因此任务中断再续拍（prompt 不变）时补段零 dispatch。只有带逐段不同
beats 的任务才每段一个 key，从未 queue 过的段（新 beats 文本）在续拍时首次 MISS 一次
（再写盘供下次复用）。详见"Web 续拍与复制"。

**Web 续拍与复制（2026-09-05）**：
- **续拍 = 纯补段，参数一字不改**。可对 `done/failed/cancelled/interrupted` 任务续拍；后端
  直接克隆基准 config（表单参数全部忽略，仅认目标段数），以磁盘槽位数为起点自动续
  （10 段拍到第 5 段失败 → 有 4 槽 → 自动补拍 5..10）。`segments` 必须 > 当前槽位数。
  槽已写完仅收尾失败的那段视作已完成、跳过。仅无法续拍的情况 = 基准无任何槽位（从头
  失败）→ 用"复制"。
- **复制 = 以原任务参数开一条新链**：全部参数回填可编辑（引擎沿用原任务并锁定，换引擎
  请新建），tag 留空自动生成，素材默认沿用原任务；素材列表支持逐项**删除 / 拖拽排序 /
  重传替换**——最终提交按 Web 列表显示顺序依次编号（ref_image→imgN、ref_video→videoN
  且带音轨的 <Audio N>、ref_audio→audioN）。提交以 `order_<kind>` token（`base:<idx>` /
  `new:<seq>` / `-`=清空）让后端把沿用文件与新上传按该顺序合并，再生成 driver argv。
  clean_slots 强制开启（全新链）。素材引用原任务 staging 目录，原任务被删除则复制任务
  素材失效。

## 三、首段素材分流

素材只作用于**首段**；2..N 段始终走 fl2va `H3ContinuousContinueV14` 无缝续接
（audio/video 已含）。按首段引擎自动分流：

- **fl2va 首段**（默认引擎，`H3ContinuousStartV14`）：
  - 纯文本（fl2va，无 keyframe）
  - `--first-image` = 首段视频第 0 帧锚点图（I2V 开画）
  - `--last-image` = 首段结束帧锚点图（决定段 1 终点画面，续段从该帧延伸）
  - 图 → 上传 ComfyUI input → `LoadImage` → Start 的 `first_frame`/`last_frame`，
    VAE encode 成 latent keyframe。
- **ref2va 首段**（`MiniMaxH3ReferenceToVideo` + ref2va unet）：
  - `--ref-image`（≤9，`<Picture N>`）、`--ref-video`（≤3，24fps 帧，`<Video k>`；
    自带音轨自动作 `<Audio k>`）、`--ref-audio`（≤3 独立参考音频，引导生成声）。
  - `minimax_refs` 进 DiT 条件、每步采样生效；空 AV latent 采样 `length` 帧（`--dur`
    换算 `L ≡ 5 (mod 17)`，5s→124）。参考视频先被**规整到 24fps**（远端无 ffmpeg，
    脚本用 pyav 重编码 h264/aac 到 input 目录），再 `LoadVideo → GetVideoComponents`。
  - `--ref-audio` 机制、测试与音色结论详见 **audio.md**。
  - clip1 后同样 `AnalyzeHandover → SaveLatent(slot1)`；续段逻辑零改动。

**模型切换与清场**：ref2va 首段用 ref2va unet，续段回 fl2va unet。带素材首段的 clip1
在 ComfyUI 进程里 VAE-encode 图像/视频，会留 GPU0 CUDA allocator 池（≈100 MB），压垮
resident 续段采样 → 脚本在 clip2 前**自动 restart 一次**（~16s）并以新 epoch 重建
（`epoch_rest = _RUN_EPOCH + 1`），续段 2..N 再驻留复用。

**slot 约定**：`h3_continuous/chain_*.safetensors` 全局不区分 tag，同时只跑一条链；
换任务前先 `rm output/h3_continuous/chain_*.safetensors`。

## 四、实测（2026-09-05）

- prewarm 清场（有残留 FSDP 18.3/12.4GB）：~30s，GPU → ~400MB；任务内同 epoch 复用：0.5s。
- **服务重启：~10–12 s**（两测 12.3/10.3/10.2s）——比 prewarm 重建更快更干净，故设为默认。
- 端到端（默认 restart）：cache-miss 新 prompt 首段正常 dispatch（无 OOM），clip1 ≈ 240–350s；
  续段 **~120s**（FSDP 复用，不再 reload/dispatch）；merge 正常（rate 24、7.958s/191 帧、pts 单调）。
- fl2va `--first-image/--last-image` 2 段 + merge 与 ref2va `--ref-image` 2 段 + merge 均成功
  （225 帧/9.38s、立体声）；ref2va `--ref-video`（含音轨）单段成功（seg=124 帧，ref2v
  length 与 fl2va dur 帧基数可不同，merge 按各段 handover 实际裁切）。
- ref2va 无 turbo lora，`--steps` 照常。

- **生命周期托管实测（2026-09-05）**：
  - 失败路径：ComfyUI 停机 → driver 跑段前 `stop.sh`(not running 无害)→`start-comfyui-for-minimax-h3.sh`→
    16.3 s 就绪(`:8188` 200)；段采样报错(dur=1.0 过短,<39 帧下限) → 最后仍
    `[lifecycle] stopping` → `:8188` DOWN。**失败也关** ✓
  - 成功路径：停机 → 拉起 → `[clip1] done 192.0s` + `FINAL` → 结束时 `stopped` →
    `:8188` DOWN。`[web] done rc=0`。**成功也关** ✓

## 五、脚本与组件清单（2026-09-05 状态）

远端运行目录 `~/MiniMax-H3-Deploy/`；一次性 POC/评测脚本已清理，`scripts/` 现仅存：

- **chain_director_v2.py**（**推荐 main**）：persist 模式，**托管 ComfyUI 生命周期**——
  跑段前 stop→start 接管并等待 :8188 就绪、跑段结束（含 SIGTERM/异常）恒 stop 释放 GPU
  （`--clear` 被忽略）；任务内 segments 2..N 复用 FSDP；含 merge 与 stitch 时间戳修复。
- **chain_director_v2_web.py**（局域网 Web 控制台，stdlib-only）：在 GPU 宿主与
  ComfyUI 同机常驻，`~/ComfyUI-Deploy/comfyenv/bin/python ~/MiniMax-H3-Deploy/scripts/chain_director_v2_web.py --daemon`
  （`--stop`/`--status`），默认监听 `0.0.0.0:8189`（可用 `http://<宿主IP>:8189` 从
  其它局域网机器访问）。它镜像 v2 全部 CLI 参数并支持素材上传、串行任务队列、
  流式日志/阶段徽标、取消、结果在线预览（Range）与"续拍"；每个 web 任务默认
  `clean_slots`（跑前清 h3_continuous 槽位 + 同 tag 旧片段 + 旧 final = 全新链）。
  busy 判据 = ComfyUI `/queue` 实际在跑（链结束后 ray worker 的 FSDP/VAE 驻留是
  正常持久态、不算忙，下个任务才清场）；排队任务可直接移除。数据目录
  `~/.h3web/jobs/<job_id>/`。docstring 内为完整 API/护栏说明；离线回归测试在本地
  `scripts` 同级 `.webtest/run_webtest.py`（mock driver，无 GPU）。
- **chain_director_v1.py**（对照/冷跑）：clear=True 逐段（每段重载），语义与旧版一致。
- **poc_samegraph.py**（POC）：同图单 queue N 段（借助 cond 缓存，见 v1 文档八节）。
- **start-comfyui-for-minimax-h3.sh**（`scripts/`）/ **stop.sh**（`~/ComfyUI-Deploy/`）：ComfyUI 服务启动（含 RAY env）/停止。

**目录布局（2026-09-06 拆分）**：ComfyUI 引擎独立于编排层：

```
~/ComfyUI-Deploy/          # 引擎（独立可运行）
├── main.py, models/ (74G), custom_nodes/, input/, comfyenv/ (7G venv)
├── stop.sh                 # 停止（引用 COMFY_HOME 绝对路径）
├── comfy.log              # ComfyUI 运行日志（随启动脚本落此）
~/MiniMax-H3-Deploy/       # 编排层
├── scripts/ (driver, web, workflows/), output/, cond_cache/, .h3web/
├── docs/, vendor/, raylightenv/, downloadenv/
```

一次性 A/B helper（本地会话留副本；若远端已删，以本地为准）：
- **cmp_official_t2v.py**：官方等价图（`MiniMaxH3ImageToVideo` 首段）API 提交，用于与链
  首段 1:1 对照（图等价结论见 audio.md 底噪调查 D 节）。

组件改动（非 scripts/）：

- **comfyui_h3_multigpu_clip/__init__.py**（已改，运行中生效）：内嵌 CLIP cond 预编译缓存
  （见 v1 文档七节）。
- **raylight/src/raylight/nodes.py** + **distributed_worker/ray_worker.py**（已改）：
  persist 幂等复用（reuse_epoch guard + persist-only + ensure_fresh_actors +
  set_state_dict 宽松），见本文件二节。改前原文件存 `.bak_pre_persist`。
- **raylight xdit_context_parallel.py**：cond_audio/denoise-mask 补丁见 audio.md。
- 本地 **scripts/_h3mp/**：comfyui_h3_multigpu_clip 修改版源码副本（变更基线）；
  **scripts/_raylight_patch/**：nodes.py/ray_worker.py persist 修改版副本。
- ComfyUI 主仓库与 Herrgotts suite：**未改**（见 README 改动总览）。
