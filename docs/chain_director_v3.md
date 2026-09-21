# 工作节点 5：chain_director_v3.py（常驻 UNet 续接链 · 现行）

> 阶段：v3 = v2 引擎（Herrgotts masked-AV、逐段 queue、slot latent、handover 合并）
> + **UNet 常驻**（`clear_vram_after_sampling=False`，段间零 FSDP 重载）
> + **int4 CLIP 按需上下卡**（`offload_after_encode=True`）。
> 首段只支持 fl2va（文本 / 首末帧锚图）。v2 保留为 persist 对照，v1 不再维护。

## 一、与 v2 的差异

| | v2 | v3 |
|---|---|---|
| CLIP | int8，`offload_after_encode=True` | **int4**，`offload_after_encode=True` |
| 视频 VAE | fp16 | **int8_convrot** + 3×`UnloadVideoVAE` |
| RayInitializer | `sync_ulysses=False`，`KV_INT8=off` | `sync_ulysses=True`，`KV_INT8=v`（见六节：当前不生效） |
| `reuse_epoch` | 按 prompt 变化点递增（触发 ray 重建 + FSDP 重载） | **整 run 恒定**（不重建、不重载） |
| 段间/首段后 restart 服务 | 有（换 unet / 有素材时） | **无** |
| run 之间的服务 | 每 run stop→start（跑完必 stop） | **默认常驻**，下一轮复用（见一·B） |
| 首段素材 | fl2va 锚图 或 ref2va（参考图/视频/音频） | **只 fl2va**（文本 / `--first-image` / `--last-image`） |
| slot 清理 | 靠 web 的 `clean_slots` | CLI `--clean` |

## 一·B、ComfyUI 生命周期：**默认常驻**

成功跑完一轮**不 stop**，服务与 ray worker、已装载的 FSDP 分片原地保留，下一轮直接复用——冷启动（服务 ~15s + ray 重建 + FSDP 首次装载，合计 ~240s）**每个服务实例只付一次**，不是每 run 付一次。

复用判定（`ensure_service()`）：

1. 服务进程存在（`pgrep main.py --listen ...`）；
2. 状态文件 `~/MiniMax-H3-Deploy/.v3_service.json` 记录的 `pid` 与当前一致；
3. 记录里的 `unet` 与本次一致；
4. `:8188` 有响应。

四条全中 → 复用（日志 `[resident] reusing service pid=... (epoch=...)`）且**沿用同一个 `reuse_epoch`** → persist guard 命中，worker 与 FSDP 都不重建。任一条不中 → `stop.sh` + `start-comfyui-for-minimax-h3.sh` 全新启动，并写入新 epoch（新 epoch 会强制重建 ray worker 并重载 FSDP，即"全新冷启动"）。

- `--stop-when-done`：成功后照样 stop（并清状态文件），空闲时把卡还回来。
- **失败/异常/超时**：一律 stop + 清状态文件（不留半坏的常驻态给下一轮）。
- **SIGTERM/SIGINT（取消）**：同上，stop + 清状态文件。
- 常驻空闲时每卡仍占 ~11.9G（FSDP 分片 + 主进程模型）；想立刻释放就 `~/ComfyUI-Deploy/stop.sh`。
- 想强制全新服务：先 `stop.sh`（状态判定自然失效）再跑。

run 内所有段仍共用同一批 worker 和同一份 FSDP 分片。

## 二、图接线（常驻底座）

段图 = `ray_base()` + Start/Continue 节点 + `add_analyze_and_save()`：

```
141 RayInitializer(local, GPU=2, ulysses=2, FSDP=true, sync_ulysses=true, KV="v",
                   clear_vram_after_sampling=False, reuse_epoch)
130 H3MultiGPUCLIPLoader(int4, gpu_ids="0,1", offload_after_encode=True)
        └─> 904 UnloadVideoVAE(vae=121) ─> Start/Continue 的 clip 输入
121 VAELoader(video int8_convrot) ─┬─> Start/Continue 的 vae
122 VAELoader(audio fp32) ─> 902 SelectVAEDevice("gpu:1") ─> 123 VAEDecodeAudio
142 RayUNETLoader(fl2va int8) ─┬─> 143 RayBasicScheduler ─┐
                              ├─> 145 RayBasicGuider ◄─ 905 UnloadVideoVAE(vae=121)
                              └─> 903 UnloadVideoVAE(ray_actors=142) 的 ray_actors 口
144 XFuserSamplerCustomAdvanced(guider=145, sampler=125, sigmas=143, latent_image=133:1)
124 VAEDecode(samples=144, vae=121) ─> 903 UnloadVideoVAE ─> 132 CreateVideo ─> 92 SaveVideo
133 H3ContinuousStartV14（段1）/ H3ContinuousContinueV14（段2..N，previous_latent/handover ← 150 LoadLatent）
200 H3ContinuousAnalyzeHandoverV14(images=124) ─> 180 H3ContinuousSaveLatent(latent=144, clip_index)
```

三个 `UnloadVideoVAE` 的分工（与已下单任务工作流一致）：

1. **904（CLIP 前）**：踢掉上一段/上一任务残留的视频 VAE，避免和 CLIP 同时占卡；
2. **905（条件编码后）**：踢掉关键帧编码用的 VAE，再进采样；
3. **903（解码后）**：踢掉解码 VAE，并 `free_cached_vae.remote()` **清 ray worker 的 CUDA 缓存池**——这一条是段间能连续跑的关键。

## 三、两个核心语义

### 3.1 条件缓存：一次性 encode，后面每段拿来就用

`H3MultiGPUCLIPLoader.encode_model_parallel()` 先查磁盘缓存（`_cond_cache_key = sha256(clip_name + tokens + unprojected + add_dict)`），**命中就直接返回，连 `dispatch()` 都不会调**。

- 固定 prompt 的链：段 1 首次 encode 写盘（或命中历史缓存），段 2..N 全 HIT → **每段额外开销 0**；
- 缓存**跨进程、跨 ComfyUI 重启有效**（实测：同一 prompt 第二次整链运行全程 HIT，一次上卡都没有）；
- 带锚图的段（key 里含图 token）与纯文本段的 key 不同，会 MISS 一次 → 一次 encoder 往返；
- 缓存目录 `~/MiniMax-H3-Deploy/cond_cache/`，清缓存 `rm -rf ~/MiniMax-H3-Deploy/cond_cache/*`。

### 3.2 reuse_epoch 恒定：段间不重建 ray、不重载 FSDP

v2 在 prompt 变化点递增 `reuse_epoch` 是为了让新的一次 CLIP dispatch 落在"空卡"上（int8 编码器要和常驻 FSDP 抢 ~24G）。v3 的编码器只占 ~6.5G/卡，且编码本来就是**在 FSDP 常驻的前提下**完成的（实测编码后仍剩 13.83/14.36G），所以整 run 一个 epoch 就够：

- `RayInitializer` 节点输出被 ComfyUI 缓存 → 段 2..N 连节点都不重跑；
- `RayUNETLoader.IS_CHANGED=NaN` 会重跑，但 `ensure_fresh_actors` 在 persist 模式下保留已加载模型，`load_unet` 的 `active_request_key` 相同 → 走 fast-path，不重载。

### 3.3 首段只用 fl2va

`--first-image/--last-image` 是**视频 VAE 的关键帧锚**（`minimax_keyframes`），不走 Qwen 视觉塔，所以 `H3_MP_KEEP_VISUAL_CPU=1`（视觉塔 int4 权重 ~1.1G 留在 CPU）依然成立——这正是单卡显存能兜住的原因。
ref2va 需要视觉塔上卡（`H3_MP_KEEP_VISUAL_CPU=0`），且在"UNet 常驻"前提下未验证，故 v3 不做。

## 四、CLI 与用法

```bash
cd ~/MiniMax-H3-Deploy
# 固定 prompt 的 4 段链（每段 4s 净新内容），8 步，输出并合并
~/ComfyUI-Deploy/comfyenv/bin/python scripts/chain_director_v3.py --tag film --segments 4 \
  --dur 4 --width 864 --height 480 --steps 8 --clean --merge --prompt "..."

# 首末帧锚图（只作用于首段）
... --first-image img/open.png --last-image img/end.png

# 续拍：不改参数直接重跑（按 slot 数补齐缺失段），例如上次跑到第 3 段
... --segments 6 --merge
```

- `--dur` 是"Net New Content"净新内容秒数；段 2..N 的总 latent = 净新 + 39 帧保护上下文（17k+5 对齐）。
- `--beat <秒:描述>`（可重复）仍保留（v2 语义）；**固定 prompt 的链用不到**，一次 encode 全段复用。
- `--clean`：清 `output/h3_continuous/chain_*.safetensors`（slot 文件名**全局不分 tag**，换任务必须清，否则会接着上一条链续拍）。
- `--stop-when-done`：跑完 stop（默认常驻不 stop，见一·B）。
- 产物：`output/video/chain/<tag>/seg_<i>_*.mp4`、slot `output/h3_continuous/chain_*.safetensors`、合并 `output/final_<tag>.mp4`（按 handover 元数据裁掉每段不可用尾/保护头，pts 单调）。

## 四·B、Web 控制台（`chain_director_v3_web.py`）

**新文件**，不动 `chain_director_v2_web.py`（后者继续驱动 v2 / 提供 ref2va）。两者可并用：

| | chain_director_v2_web.py（原样） | chain_director_v3_web.py（新） |
|---|---|---|
| 驱动 | chain_director_v2.py | **chain_director_v3.py** |
| 默认端口 | 8189 | **8190** |
| 数据目录 | `~/MiniMax-H3-Deploy/.h3web` | `~/MiniMax-H3-Deploy/.h3web_v3` |
| pid / 日志 | `h3web.pid` / `h3web.log` | `h3web_v3.pid` / `h3web_v3.log` |
| 引擎 | text / i2v / ref（ref2va 素材） | **text / i2v**（fl2va only） |

```bash
~/ComfyUI-Deploy/comfyenv/bin/python scripts/chain_director_v3_web.py --start   # 0.0.0.0:8190
~/ComfyUI-Deploy/comfyenv/bin/python scripts/chain_director_v3_web.py --status
~/ComfyUI-Deploy/comfyenv/bin/python scripts/chain_director_v3_web.py --stop
```

- `_build_argv` 只发 v3 认识的参数（`--tag/--prompt/--segments/--dur/--width/--height/--steps/--beat/--seed/--merge/--first-image/--last-image`）。
- ref2va 素材上传或 `engine=ref` 会被后端直接 400 拒绝，提示改用 CLI 跑 `chain_director_v2.py --ref-image/-video/-audio`。
- 表单移除了 ref2va 素材区、ref-image-size 与「清场方式 clear」（v3 无这些参数）；服务生命周期由 driver 自己管（默认常驻）。
- busy guard 恒开：ComfyUI `/queue` 有任务在跑，或"服务离线且外部 driver 存活（可能在重启）"时排队等待；跨控制台也生效（判据是 `/queue`，不是 pidfile）。
- `clean_slots`（默认开）仍在跑前清 `h3_continuous` 槽位 + 本 tag 旧片段 + 旧 final（等价且比 `--clean` 更彻底）。
- 取消 = TERM 级：v3 收到后 stop 服务 + 清状态文件，不留半坏常驻态。

## 五、实测（2×2080Ti，864×480，8 步，int4 CLIP + int8 UNet）

| 轮次 | 场景 | 段 1 | 段 2 | 合并 | 备注 |
|---|---|---|---|---|---|
| v3v2 | 固定 prompt，dur4 | 219.5s | **135.2s** | 191 帧 | 段 1 冷启动；段 2 缓存 HIT、零上卡 |
| v3v3 | 首+末帧锚图，dur4 | 338.2s | 140.2s | 174 帧 | 锚图段 MISS 一次（dispatch 11.06s + 带图 encode 11.31s + 卸载 0.13s） |
| v3v4 | 边界 dur8 | 426.9s | **250.2s** | 361 帧 / 15.04s | 段 1 = 192 帧、段 2 = 226 帧；峰值 **21.47/21.45G**（上限 21.48） |

常驻复用的收益（同一 prompt，dur4/8 步，连续两轮，第二轮起手复用上一轮的服务）：

| 轮次 | 段 1 | 段 2 | 说明 |
|---|---|---|---|
| v3r1（冷） | 242.8s | 135.1s | 服务 12.3s 拉起 + ray 重建 + FSDP 首次装载 |
| v3r2（复用） | **95.4s** | 135.1s | `[resident] reusing service pid=...`；不重启、不重建 worker、不重载 FSDP |
| v3r3（复用 + `--stop-when-done`） | 50.0s（dur2/48 帧） | — | 跑完 stop，显存归零、状态文件清除 |

即跨 run 复用让"每轮的第一个段"从 ~240s 降到 ~95s（2.5×）。

编码器成本（日志 `H3 Qwen model-parallel timing`）：

- 纯文本：`dispatch=11.295s encode=4.056s offload=0.915s`（上卡 11.3s / 卸载 0.9s）
- 带锚图：`dispatch=11.064s encode=11.311s offload=0.128s`

采样步速随帧数上升（同 8 步）：107 帧 6.58 s/it → 192 帧 14.7 s/it → 226 帧 19.5 s/it。

## 五·B、段内阶段分解与两卡负载（2026-09-22 实测）

一次冷启动两段跑（`--segments 2 --dur 2 --steps 8 --width 864 --height 480`，纯文本，无锚图），用 `nvidia-smi -lms 200`（CSV 带 timestamp 列）采两卡显存/利用率，ComfyUI 日志经时间戳管道对齐。原始数据：`~/Temp/opencode/{bench_gpu.csv,bench_comfy.ts.log,bench_driver.log}`。

### 段 1（Start，含冷启动）`Prompt executed in 124.67s`

| 阶段 | 时长 | GPU0 均值/峰值 MiB | GPU1 均值/峰值 MiB | GPU0 util | GPU1 util |
|---|---|---|---|---|---|
| 服务启动 | 18s | 76/175 | 8/9 | 0% | 0% |
| VAE+CLIP 载入（CPU，不上卡） | 7s | 175 | 161/165 | 0% | 0% |
| CLIP dispatch/encode/offload | 16s | 5703/7841 | 2508/7295 | 6% | 5% |
| `ray.init()` | 2s | 1713 | 219 | 5% | 8% |
| **worker + FSDP 首次装载** | **51s** | 5060/**15887** | 3598/**14393** | 2% | 2% |
| **采样 8 步** | 25s | 16502/16519 | 15008/15025 | **95%** | **95%** |
| VAE 上卡 + 解码 | 22s | 13781/16519 | 12051/15025 | **73%** | **6%** |
| 收尾 | 3s | 13502 | 11970 | 26% | 0% |

### 段 2（Continue，稳态）`Prompt executed in 79.20s`

| 阶段 | 时长 | GPU0 均值/峰值 MiB | GPU1 均值/峰值 MiB | GPU0 util | GPU1 util |
|---|---|---|---|---|---|
| 准备（LoadLatent + cond HIT + FSDP skip） | ~2s | 13467 | 11965 | 0% | 0% |
| **采样 8 步** | ~42s | 18294/**18417** | 16792/**16915** | **97%** | **97%** |
| **VAE 上卡 + 解码** | **29s** | 13786 | 11989 | **87%** | **2%** |
| 分析 + 保存 + 收尾 | 3s | 13458 | 11905 | 35% | 0% |

段 2 时间构成：采样 ~42s（53%）+ 解码 29s（37%）+ 其它 ~8s（10%）；cond HIT 使 CLIP 上卡为 0，FSDP 不重载。

### 关键观测（修正此前文档）

1. **采样期两卡不是镜像**：段 2 采样 GPU0 18417M vs GPU1 16915M，恒定差 **~1.5G**，且差异从 FSDP 装载期就存在（15887 vs 14393）。两卡 util 均 97%，接近饱和。
2. **解码期 GPU1 几乎空闲**（2–6%），GPU0 忙（73–87%）。视频 VAE 分双卡的空间就在这里，但 GPU0 未满算力，实际收益应低于 2×。
3. **VAE 上卡只要 1–3s**：日志 `Requested to load MiniMaxH3VideoVAE` → `loaded partially ... 2665.86 MB offloaded`（段 1 用 3s、段 2 用 1s）。此前 `gen_dual.md` §2.2 把"4.97G VideoVAE 装载"隐含成 28s 量级，实测否决——**"VAE 常驻"没有收益**（磁盘层被节点缓存命中，显存层仅一次 CPU→GPU）。
4. **解码耗时与帧数线性**：56 帧 ≈ 17s、90 帧 ≈ 28s（≈ **0.31 s/帧**）。
5. **本次冷启动远快于历史记录**：段 1 准备段合计 **76s**（CLIP encode 15s + `ray.init` 2s + worker/FSDP 装载 51s）。§六 记的"180–240s"是 int8 CLIP（dispatch 21–34s）+ 冷页缓存的结果；int4 CLIP（本次 `dispatch=11.966s encode=2.372s offload=0.115s`）与 warm mmap 下显著缩短。
6. `RAYLIGHT_ULYSSES_KV_INT8` 未生效再次确认（`HEAD_CHUNK is 0; using the regular FP16 Ulysses path`）。

### 由此得到的优化评估（均未实施）

- **唯一确定值得做的**：视频 VAE 按 temporal chunk 分双卡（段 2 解码 29s、同期 GPU1 空闲）。改 `comfy/ldm/minimax/vae.py` 的 `decode_temporal`，把 chunk 分派到 2 device；因 GPU0 解码 util 仅 87%，收益保守估 **8–14s/段**，不要按 2× 估。
- **已否决**：VAE 常驻（上卡仅 1–3s）；FSDP 软驻留（`gen_dual.md` §2.3，上游不支持，2 卡下不可靠）。
- **低风险小项**：`wait_done` 轮询 5s→0.5s（`chain_director_v3.py:242`），多段链累计可省。
- **无空间**：采样已 95–97% 饱和，除非动 KV int8（改采样轨迹，不采用）。

## 六、边界与注意

- **帧数边界 ≈226 帧 @864×480**（峰值 21.47/21.48G，已贴顶）。再抬帧数或分辨率会 OOM；要更长的单段只能降分辨率，或回到 v2 的"每段短一点但更多段"。
- **段 1 的固定成本**：服务拉起 12–18s + ray 重建 + FSDP 首次装载 ≈ 180–240s，**每个服务实例只付一次**（默认常驻 → 后续 run 复用，段 1 降到 ~95s）；段 2+ 是稳态（135–250s，取决于帧数）。该 180–240s 是 int8 CLIP + 冷页缓存时代的数；int4 CLIP + warm mmap 下准备段已降到 **~76s**（见 五·B）。
- **锚图的额外成本**：段 1 +120s 量级（VAE 关键帧编码 + 带图 encode），且 handover 会切掉更多不可用尾（锚图锁末帧 → tail 34 帧 vs 普通 17 帧）。
- **`RAYLIGHT_ULYSSES_KV_INT8="v"` 当前不生效**：raylight 打印 `enabled but RAYLIGHT_ULYSSES_HEAD_CHUNK is 0; using the regular FP16 Ulysses path`。要真启用需 `RAYLIGHT_ULYSSES_HEAD_CHUNK>0`，但那会改变采样轨迹（质量未验），v3 不采用——别把 `"v"` 当成省显存手段。
- slot 全局不分 tag；同时只跑一条链。

## 七、决策记录：为什么不用"CLIP 全常驻"

先做的版本是 CLIP 也常驻（`offload_after_encode=False`）。结果：

- 段 2 采样前每卡只剩 **4.52/3.61G**（CLIP 6.5G + FSDP 10.5G 都在卡上），107 帧的段在第 1 步就硬 OOM（申请 962MB，仅 601MB free；段 1 能过只是靠 allocator 回收侥幸）；
- 换言之全常驻档的单段上限 ~107 帧，是"贴着天花板跑"。

改成"编码后卸卡"后：采样期每卡多出 ~6.5G，边界抬到 ≥226 帧，代价只有**缓存 MISS 时**的 ~12s（11.3s 上卡 + 0.9s 卸载）——固定 prompt 的链上是 0s。UNet 仍然全程常驻，段间没有 FSDP 重载，v2 那种"换 prompt 就要 epoch bump + 重载"的代价也一并省掉。

`RAYLIGHT_ULYSSES_HEAD_CHUNK` 的 KV int8 路径（唯一还能再省显存的手段）因为会改采样轨迹，未采用。

## 八、复现/验证口径

```bash
# 1) 固定 prompt 2 段（应看到段 2 cond_cache HIT + [clip2] done 明显快于段 1）
~/ComfyUI-Deploy/comfyenv/bin/python scripts/chain_director_v3.py --tag t1 --segments 2 \
  --dur 4 --width 864 --height 480 --steps 8 --clean --merge --prompt "..."
# 2) 查日志：grep -a "cond_cache\|H3 Qwen model-parallel timing\|OutOfMemory" ~/ComfyUI-Deploy/comfy.log
# 3) 查产物：pts 单调 + aac 音轨（pyav 逐帧读）
```

判据：段 2 无 OOM、`h3_cond_cache: HIT`、无 `[GUARD] ... REBUILD`；合并产物 pts 单调且带音轨。

常驻复用判据：第一轮结束时 `pgrep -f "main.py --listen"` 仍在、`~/MiniMax-H3-Deploy/.v3_service.json` 存在；紧接着跑第二轮应打印 `[resident] reusing service pid=...`，且段 1 明显快于冷启动；`--stop-when-done` 之后服务应已停、状态文件消失。
