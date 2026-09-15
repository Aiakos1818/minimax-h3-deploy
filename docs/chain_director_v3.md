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
| 首段素材 | fl2va 锚图 或 ref2va（参考图/视频/音频） | **只 fl2va**（文本 / `--first-image` / `--last-image`） |
| slot 清理 | 靠 web 的 `clean_slots` | CLI `--clean` |

服务仍然**每 run 接管一次**（跑段前 stop→start，结束必 stop），但 run 内所有段共用同一批 ray worker 和同一份 FSDP 分片。

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
- `--clear`：保留解析但**忽略**（托管生命周期恒为 stop→start）。
- 产物：`output/video/chain/<tag>/seg_<i>_*.mp4`、slot `output/h3_continuous/chain_*.safetensors`、合并 `output/final_<tag>.mp4`（按 handover 元数据裁掉每段不可用尾/保护头，pts 单调）。

## 五、实测（2×2080Ti，864×480，8 步，int4 CLIP + int8 UNet）

| 轮次 | 场景 | 段 1 | 段 2 | 合并 | 备注 |
|---|---|---|---|---|---|
| v3v2 | 固定 prompt，dur4 | 219.5s | **135.2s** | 191 帧 | 段 1 冷启动；段 2 缓存 HIT、零上卡 |
| v3v3 | 首+末帧锚图，dur4 | 338.2s | 140.2s | 174 帧 | 锚图段 MISS 一次（dispatch 11.06s + 带图 encode 11.31s + 卸载 0.13s） |
| v3v4 | 边界 dur8 | 426.9s | **250.2s** | 361 帧 / 15.04s | 段 1 = 192 帧、段 2 = 226 帧；峰值 **21.47/21.45G**（上限 21.48） |

编码器成本（日志 `H3 Qwen model-parallel timing`）：

- 纯文本：`dispatch=11.295s encode=4.056s offload=0.915s`（上卡 11.3s / 卸载 0.9s）
- 带锚图：`dispatch=11.064s encode=11.311s offload=0.128s`

采样步速随帧数上升（同 8 步）：107 帧 6.58 s/it → 192 帧 14.7 s/it → 226 帧 19.5 s/it。

## 六、边界与注意

- **帧数边界 ≈226 帧 @864×480**（峰值 21.47/21.48G，已贴顶）。再抬帧数或分辨率会 OOM；要更长的单段只能降分辨率，或回到 v2 的"每段短一点但更多段"。
- **段 1 的固定成本**：服务拉起 14.3–18.4s + ray 重建 + FSDP 首次装载 ≈ 180–200s；段 2+ 才是稳态（135–250s，取决于帧数）。所以段数少时不划算。
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
