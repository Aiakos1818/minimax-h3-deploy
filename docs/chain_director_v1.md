# 工作节点 3：chain_director_v1.py（Herrgotts masked-AV 多段续接链 v1）

> 阶段：服务端自动多段续接链。v1 语义验证（`chain_director.py` v0）→ v1 正式
> （Herrgotts masked-AV 引擎，逐段独立 queue，clear=True 每段重载）。
> 含为链提速的 CLIP cond 预编译缓存与同图单 queue POC 探索。
> **自 2026-09-05 起 v1 不再维护，一切改动落在 v2**（见 chain_director_v2.md）。

## 一、演进与 v0 对照（为什么不用 first-frame 接力）

- **v0 `chain_director.py`**（P1 语义验证）：逐段独立 queue；seg0 官方
  `MiniMaxH3ImageToVideo`，seg1+ 用 `first_frame` = 上段 mp4 末帧抽帧接力。
  → 只有视频首帧继承，无音频/隐空间延续，接缝易跳变。
- **v1** 换 Herrgotts v1.4 native Masked-AV 引擎：把上一段 AV latent 前缀直接拷入下一段并
  用 video/audio 双流 denoise mask 保护，视频前缀与音频均**结构性延续**。段内续接用
  `H3ContinuousContinueV14`（masked 39f + handover 元数据），不再抽帧。

## 二、架构

```
clip1  H3ContinuousStartV14(fl2va) -> raylight sampler -> AnalyzeHandover -> SaveLatent(slot1) + raw mp4
clipN  LoadLatent(slot N-1) + H3ContinuousContinueV14(masked 39f, handover)
       -> raylight sampler -> AnalyzeHandover -> SaveLatent(slot N) + raw mp4
merge  clip1[0:safe_end] + clipN[head:safe_end] + ... + last[head:end]  -> final.mp4 (音画对齐)
```

- 逐段可 **resume**：`--segments` 大于已产 slot 数时只补跑缺失段。
- 每段可用不同提示；`--beats "秒:要点"` 自动把节拍注入其所属段（按 `t / --dur` 归段，
  转相对秒措辞）。
- 每段约 5s 净画面需 head 39 帧保护（约 1.6s 渲染开销被裁掉）。

## 三、依赖（本机已装/已打补丁）

- custom_nodes：`Herrgotts-H3-Infinite-Continuation-Suite`、`ComfyUI-H3-Motion-Context`（对照）、
  `comfyui_h3_multigpu_clip`、`raylight`。
- ComfyUI 需含 H3 native AV-mask（PR#15375）——部署 master 已具备。
- raylight 补丁（`src/raylight/diffusion_models/minimax/xdit_context_parallel.py`）：
  1. `cond_audio` 键补全（音频 keyframe 锚可用）；
  2. `denoise_mask/audio_denoise_mask` → 移植单卡逐行 timestep 逻辑（mask 前缀不被重新加噪）；
  3. `_split_packed_sequence` 支持向量 row（per-row 调度跨 SP 切分正确）。
  详见 audio.md（补丁细节与音频锚）。
- start-comfyui-for-minimax-h3.sh 增加 `RAY_memory_usage_threshold=1.0` / `RAY_memory_monitor_refresh_ms=2000`
  （15.5G RAM 下 Ray 会误杀 init 尖峰，借 swap 度过）。

## 四、用法（v1 CLI）

```bash
cd ~/MiniMax-H3-Deploy
comfyenv/bin/python scripts/chain_director_v1.py \
  --tag myfilm --segments 6 --dur 5 \
  --prompt "全局设定/人物/风格/机位一句话（作用于全片）" \
  --beats "5:人物加快脚步;12:身后天空浮现金色传送门;22:群武器齐射;30:雨落"
# 生成 6 段 raw + final_myfilm.mp4（merge 拼好的整片）
# 续拍：直接 --segments 8 再跑，自动补 clip7/8 并重新 merge
```

可选：`--steps`（默认 8）、`--seed`、`--width/--height`（864×480 起）、`--dur`
（每段净秒数，建议 ≥4）。

产物：`output/video/chain/<tag>/seg_*_*.mp4`（raw，未裁头尾）、
`output/h3_continuous/chain_*.safetensors`（逐段 AV latent + handover 元数据，slot 即
clip 号）、`output/final_<tag>.mp4`。

> v1 的 merge/stitch 与 v2 同构。stamp：stitch 时间戳修复见下（v1 首修，v2 继承）。

## 五、实测结论（双卡 TP）

- **Herrgotts masked-AV 前缀保护**：视频前缀 seg1[:39] vs seg0[68:105] ≈ **41.6 dB**
  （逐帧保真，mask 真实生效）。
- 接缝推进 seg0[105]→seg1[39] ≈ 28–32 dB；时间推进正常（非重播）。
- **音频延续**：seg0 尾 vs seg1 头 0ms 互相关 **0.83**（同一声源延续；对照 first-frame/
  Niko 方案仅 0.2–0.4）。机制详见 audio.md（audio_tail_carryover + 缝合实现）。
- 每段开销：clip1 ≈ 4–5 min、续段 ≈ 5–6 min（8 步 864×480）；head 39 帧为隐藏开销
  （净画面 = 总时长 − head）。

## 六、stitch 帧时间戳修复（播放"画面不动"）

**现象**：`final_*.mp4` 用播放器打开画面不动。根因：`stitch()` 视频重编码时
`fr.pts = vpts; vpts += 1` —— 解码/`reformat` 后帧的 time_base 是 **1/12288**，encode
按该时间基解释 pts，故全部帧 pts≈0 → 容器 duration≈1 帧（0.04s）。原生 seg mp4
（ComfyUI CreateVideo 产出）无此问题。

**修复**（v1 `stitch()`，v2 同构）：帧 pts 按其真实时间基步进，
`den = fr.time_base.denominator; vpts += round(den/FPS)`（24fps → 512/帧）。
修复后 191 帧 → duration 7.958s、pts 0→97280 单调、rate 24；逐帧与源 seg 对齐（含接缝
两侧）验证通过。

> 2026-09-04 修复前产出的 `final_cd1.mp4` 仍带旧 bug，且其 handover slot 已被清理，
> 需重跑链才能得到正常版。音频缝合实现细节（HEAD 窗口/样本换算/AAC stereo）见 audio.md。

## 七、CLIP 预编译 cond 缓存（v1 提速）

逐段 chain 里每段都会 `clip.encode_from_tokens_scheduled` 一次 → `comfyui_h3_multigpu_clip`
的模型并行 encode 每次 **dispatch（~25–30s，冷 ~70s）** 把 Qwen3VL-32B int8 权重搬上双卡，
encode 本身仅 1.5–4.6s。纯文本 encode 是**确定性函数**，其输出只依赖 `(clip, tokens)`
（含 `pooled_output`、`minimax_token_tags`，均无模型/分辨率/时长绑定 —— 帧相关 meta
是 Herrgotts 在 encode 之后用 `conditioning_set_values` 现加，不进缓存）。

**实现**（内嵌 `custom_nodes/comfyui_h3_multigpu_clip/__init__.py` 的
`encode_model_parallel`）：
- 缓存 key = `sha256(clip_name + tokens 规范 json + unprojected + add_dict)`。
- 磁盘布局 `~/MiniMax-H3-Deploy/cond_cache/<hex前2位>/<fullkey>/`：
  `tensors.safetensors` + `meta.json`。
- 命中：直接返回还原的 cpu conditioning（**0 dispatch**）；未命中照常 encode 后自动写入。
- 任何读写异常只告警降级为 miss，不影响生成。
- 命中前提 = 相同 prompt（同 clip + 同 tokenize 结果）；不同 prompt 自动 miss 走原路。

**收益**：同一 prompt 的多段链，从第 2 段起 encode 免 dispatch；**跨任务/跨重启复用**
（同 prompt 反复调 seed/dur 工程零 CLIP 加载）。同 prompt 重跑同一任务时 cond 全命中。
v2 persist 同样受益。

**清缓存**：`rm -rf ~/MiniMax-H3-Deploy/cond_cache/*`（缓存目录保留无碍）。

## 八、同图单 queue 流程（POC：`poc_samegraph.py`，探索附记，非正式路径）

把 N 段节点合成**一张 DAG、一次 `/prompt`**：共享 1 份 RayInitializer/CLIPLoader/UNETLoader，
段间**不落盘**直连 `sampler.latent → Continue.previous_latent`、`analyze.handover →
Continue.handover`，每段仍 Analyze + SaveLatent（保留 slot 与 merge 兼容）。

- 早期尝试在 seg1 卡死：`clear_vram_after_sampling=True` 时 seg1 sampler 的 worker
  `model=None`；改 `False` 则 FSDP shards 常驻，seg1 的 CLIP encode 直接 OOM（2×22GB
  放不下 CLIP+UNet 同驻）。
- **cond 缓存解锁**：seg1 Continue 的 encode 现在 HIT 缓存、不再 dispatch → 不抢显存 →
  `clear_vram_after_sampling=False` 下同图跑通。实测 2 段单 queue **522s**（逐段两段
  ~693s，省 ~25%，消除每 queue 的 CLIP dispatch + RayUNETLoader FSDP 重载固定开销）；
  全程 0 dispatch、2 次 cache HIT。
- 同图与逐段（同 seed/prompt/dur）产物**逐像素一致**（cond 同源），merge 裁切相同。
- 局限：单次 queue 无法中断续跑；`clear=False` 使显存常驻较高。
- 正式 main 是 v1/v2（逐段可 resume）；poc_samegraph 仅作收益验证与长链基准候选。
  其"FSDP 常驻 + 免 dispatch"思路被 v2 persist 吸收为正式能力。

## 九、限制与提示

- ~~每段 CLIP 重新 dispatch（~21s）~~ → **已解决（2026-09-04）**：cond 缓存，见上节。
- Turbo/Spectrum 会劣化音频与 pin 保真，建议默认路径跑。
- 链越长质量缓慢衰减（模型自身平滑），长片建议在中途自然乐段/换场处重启一段。
- H3 音频 32 kHz：拼接按源采样率处理，勿硬编码 48k（详见 audio.md）。
