# H3 音频专项（表示 / 段间连接 / 引导 / 底噪调查）

> 横切 gen、gen_dual 及各阶段。重点两件事：
> **① 段间 audio 如何结构性连接**；**② 首段"底噪/静音"现象的完整调查与结论**
> （与采样步数强相关，非工作流/节点缺陷）。含 `--ref-audio`/`cond_audio` 音频引导能力。
> 所有谱/rms 数值口径一致：音频抽到 32k mono，逐秒窗，频带能量为对应 FFT bin 的 RMS。

## A. 音频表示与数据通路

- H3 音频 latent 轴 **40 ticks/s**（`latent_math.py: AUDIO_HZ=40.0`），与 24fps 非整数倍
  （每视频帧 1.667 tick）。
- latent 形态 **`[B, 32, 2, T]`**（stereo，T = ticks）；audio VAE decode 输出
  **stereo 2ch / 32 kHz**（`comfy/ldm/minimax/audio_vae.py:MiniMaxH3AudioVAE`：
  decode 输出 `[B,2,L]`，两声道独立处理）。
- 落盘链路：`VAEDecodeAudio`（audio VAE decode）+ `CreateVideo` → 每段 raw seg mp4
  自带音轨（fltp / AAC stereo）。
- H3 是**多流模型**：text + video + audio 共享 DiT；prompt 里的事件描述（"…听到…
  脚步声/碎裂声"）经文本→audio 分支引导生成声。
- 拼接按源采样率 32k 处理，勿硬编码 48k。

## B. 音频引导素材（如何"点"声音）

- **ref2va `<Audio k>`**：`MiniMaxH3ReferenceToVideo` 的 `ref_video_audios` 接受自带音轨
  的参考视频，音轨自动规整（远端无 ffmpeg → 脚本 pyav 重编码 24fps h264/aac 进 input），
  再 `LoadVideo → GetVideoComponents` 拆出 ref audio 进 `minimax_refs`。
- **`--ref-audio`**：独立参考音频文件（脚步声/节拍/音乐等），
  ≤3 个，脚本经 `LoadAudio`（nodes_audio）→ `ref_audios.ref_audio_i` 接入 ref2va，
  与 `--ref-image`/`--ref-video` 并列。参考内容驱动生成声**跟着参考走**（有结构），
  而不是模型自由发挥的"噪声态"。
- **`cond_audio` keyframe 锚**（文本音频锚，xdit 补丁，见下）：让 `cond_audio` 段也进入
  采样布局（timestep 锚 + 行 tag），音频锚不被当作自由生成行。
- 模型/清场注意：ref2va 首段用 ref2va unet（`--steps` 照常，ref2va 无 turbo lora）；
  带素材首段后、续段前脚本会自动 restart（GPU0 allocator 残留 ≈100MB 压垮续段）。

### B1. xdit 补丁（raylight `xdit_context_parallel.py`，cond_audio / denoise mask）

1. `has_aud_cond` 支持 `"cond_audio"` 段：`any(k in ("ref_audio","cond_audio") …)`。
2. `seg_t` / `seg_tag` 增加 `"cond_audio"`（timestep 取 `max(t_a, aud_aug)`、tag=2），
   cond_audio keyframe 行与 audio 同组调度。
3. denoise/audio mask 逐行 timestep 逻辑移植：
   `denoise_mask` → `mask_row_values` → `rows_t=(1-m*σ_v).clamp(max=t_pin_v)`；
   `audio_denoise_mask` → 逐 tick 行 `rows_t=(1-m*σ_a).clamp(max=t_pin_a)`；
   `unique_t`/`t_row` 汇总 → `mod_segments` 逐行 append（行级 tag），
   `_split_packed_sequence` 支持 **tensor row**（`torch.is_tensor(row)` 按段切片）。
   效果：被 mask 的 video/audio **前缀行不被重新加噪**（保持上段拷贝的原始内容）。
   这组补丁是 v1 掩码续接（音频 carryover）在**多卡 xdit** 侧能正确工作的前提。

### B2. `--ref-audio` 验证实验（avt / avt2）

- 场景 prompt："在安静的夜晚走路，要听到脚步声"（无强事件词），`--steps 8`，seed 555，
  首帧图 `h3_i2v_test.png`。
- 对照：avt（无参考音频）→ 音频**全平**（flat，max/med≈1.3，无节拍结构）；
  avt2（`--ref-audio refsteps.wav`，0.5s 间隔合成脉冲脚步节奏）→ **11 个脉冲**分布在
  0.48/1.00/1.48/…/5.00s，**跟着参考节奏走**。
- 结论：弱场景下想让"该有的声音"真正出来，`--ref-audio` 是最直接手段（8 步即可干净）；
  光靠 prompt 词在弱场景 8 步常停在噪声/平态。

## C. 段间 audio 连接（生成侧 + 缝合侧）

### C1. 生成侧：audio_tail_carryover（连续性来源）

Herrgotts masked-AV 把上一段 **audio latent 尾部数字拷贝**进新段 head，并用
**audio denoise mask** 保护（拷贝行不参与 denoise，仍 40 ticks/s 行级），因此新段 head
音频与上段尾部**同源连续**，可越过视频边界延续真实音尾。

实测（v1）：seg0 尾 vs seg1 头 **0ms 互相关 0.83**（同一声源延续）；
对照 first-frame 接力 / Niko 方案仅 **0.2–0.4**（无 latent 继承）。

### C2. 缝合侧：stitch() 如何拼音频

（缝合实现 `stitch()`）
- 每段 raw seg mp4 由 VAEDecodeAudio + CreateVideo 直接带音轨落盘（stereo/32k，见 A）。
- 对每段取与视频相同的 `[HEAD=39, e0]` 窗口：`video_pass` 取视频帧，`audio_pass`
  按 `round(start*sr/FPS)` 把帧窗换算成样本窗切出音频；各段音频沿样本维 `concat` 成
  **AAC stereo**（保留 L/R）写入 final。无 crossfade、无 overlap blend。
- **早期实现曾把双声道 `mean()` 压成 mono，2026-09-05 已改为保留立体声**（早期产物
  final_cd1 等为 mono，重跑才有立体声版）。
- 为何硬拼接已连续：见 C1 —— 段 i+1 的 head 音频生成时就是段 i 尾部 latent 拷贝且被
  mask 保护 → 解码后与段 i 尾部同源，硬拼不断裂。

### C3. 已知简化点 / 潜在边界误差（留档，当前未处理）

- H3 audio latent 40 ticks/s 与 24fps 非整数倍，Herrgotts 用 `round`/`end_error` 单独对齐。
- Herrgotts 自带 `seamless_stitch.py`（`frame_trimmed_audio`/`context_aligned_audio_join`/
  `fit_audio_length`/`blend_audio_overlap`）是官方缝合 API，**脚本未接**，只用整数帧→样本
  `round` → 接缝理论上有 ±少量采样错位/时长舍入（几 ms 级），因内容同源听感风险低。
- 若日后需更严格接缝对齐，可切官方 seamless API（需各段 decoded audio + handover 帧信息）。

## D. 首段"底噪"现象调查（完整实验链）

### D1. 现象（final_f2i，2026-09-05 定位对象）

- final_f2i.mp4 首段（0–5s）**每秒钟都有恒定 ~12kHz ±200 高能**，被听感为"底噪/电流声"：
  | 段 | rms | 8–15k 带 | 12k±200 | 结构 |
  |---|---|---|---|---|
  | 0–5s clip1 | ~0.085（每秒恒定） | 8.5–17 | **27–29（每秒都在）** | 平、无事件结构 → 持续嘶声/高频嗡 |
  | 5–9.38s clip2 | 0.042–0.05 | 0.4–0.8 | ~1 | 干净，无固定 12k |
- 特征关键词：**恒定 + 无时间结构**的宽带高频能量，每秒同样强度，不是事件触发。

### D2. 排除链（每个都做了实验/静态验证）

| 假设 | 结论 |
|---|---|
| 单声道压平 | ✗ 已证伪：audio 实为 stereo（latent [B,32,2,T]，decode [B,2,L]）；同路径 r2i clip2 干净 |
| 编码器/容器/stitch 伪影 | ✗ raw seg mp4 源文件即带底噪；codec round-trip（fltp 往返）干净 |
| audio VAE decode 伪影 | ✗ 离线 `decode(zeros)` 与 `decode(latents mean 先验)` 均干净（仅低频 ~400Hz，无 12k） |
| 我们链的节点结构 | ✗ 官方 t2v 前端 json 与链首段图等价（见 D4） |

→ **12k 高频是模型在该 (prompt, seed, steps) 下的生成内容本身**，不是任何系统伪影。

### D3. 关键纠偏：高频 ≠ 伪影

- 官方 `video_minimax_h3_raylight_t2v.json` 产物（该文件现名 `video_minimax_h3_raylight_fl2v.json`）：
  - **00001**（天台弱场景，20 步）：音频近静音，rms≈0.0005（≈ **−66dB**），几乎无内容；
  - **00002**（杯子碎裂："…3秒时掉到地板上，听到清脆的碎裂声"，8 步）：rms≈0.10，
    8–15k≈10–17、**12k±200 峰值 39.3、peak 12000Hz**——高 12k 大量存在，但**听感完全正常**
    （碎裂/玻璃声本来就是 8–15k 瞬态），用户判定"非常正常"。
- 所以判断底噪不能看"有没有 12k"，而要看**时间结构**：
  - 事件驱动的 12k（碎裂/敲击/高频音色）＝合法内容；
  - **恒定平铺每秒同强度、无结构**的 12k 地板＝噪声态。

### D4. 决定性 A/B：官方图 vs 链首段（同参同 seed）

- 用官方等价图（`MiniMaxH3ImageToVideo` 节点，fl2va unet）+ 链首段（`H3ContinuousStartV14`
  fl2va）跑同一弱场景（天台）prompt、同一 steps=8 / seed=547879687678090 / 864×480 / 124f。
  helper：`cmp_official_t2v.py`（A/B 用，一次性）。
- 结果：两产物**音频逐秒数值完全一致**（rms/各频带/峰值一致到 4 位小数；
  t0–t4 每段 rms 0.020–0.025、8–15k=0.3、12k±200≈0.8–0.9）：
  | 文件 | 来源 | 量级 |
  |---|---|---|
  | cmp_official_t8.mp4 | 官方等价图 steps8 | rms ≈ 0.022（−33dB）轻环境内容 |
  | cmp_chain_t8.mp4 | 链首段 steps8 | **同上，逐秒相同** |
- 推论：`H3ContinuousStartV14` 纯文本首段 ≡ 官方 `MiniMaxH3ImageToVideo`；**同参数下链与
  官方产出同一个生成**。当年 final_f2i 的底噪若拿去官方 json 同 (prompt/seed) 跑，也会是
  同样结果——差异从来不在"我们 vs 官方的工作流"。

### D5. 根因定位：采样步数（收敛态）

- 弱音频 prompt 下，**8 步**停在"未收敛噪声态"（本次 -33dB 轻内容/噪声感；
  当年 final_f2i 是更强的 -20dB 左右恒定 12k 地板）；**20 步**收敛到干净端。
- 链首段 steps=20 复现官方 00001：
  | 文件 | 来源 | rms | 8–15k | 12k±200 |
  |---|---|---|---|---|
  | cmp_chain_t20.mp4 | 链首段 steps20（天台/同 seed） | **0.00047（−66.6dB）** | 0.00 | 0.00 |
  | MiniMax_H3_00001_.mp4 | 官方 json 前端 20 步（天台） | 0.0005（−66dB） | ~0 | ~0 |
  → 数值一致：**前端"无底噪"＝默认 20 步收敛到近静音**（干净但基本无内容），不是前端
  有什么链没有的东西。
- 综合：模型对弱音频 prompt 有"干净收敛点"，但要在足够步数下到达；步数不足就停在
  带恒定高频的噪声态。steps/seed/随机性决定音频质量，与走哪条工作流图无关。

### D6. 结论与操作建议

1. 音频质量 = 模型在 (prompt, seed, steps) 的输出。要"内容"（脚步声/碎裂/音乐）必须给
   **强事件 prompt** 或 **`--ref-audio`/参考视频音轨**；弱场景要"干净"用 **高 steps**。
2. 经验法则（默认 `--steps 8` 保持）：
   - **弱音频/氛围类 prompt**（纯画面描述）：用 `--steps 20`（≈官方 json 默认），
     否则可能出恒定高频噪声态；20 步弱场景通常收敛到近静音或稳定轻内容；
   - **强事件 prompt / 带 `--ref-audio`**：`--steps 8` 即可干净出结构内容。
3. 已归档产物（试听/复核用）：
   - 远端 `output/video/chain/cmp_ch/`、`cmp_ch20/`、`output/video/compare/official8_00001_.mp4`
   - 本地会话 `videos/cmp_chain_t8.mp4`、`videos/cmp_chain_t20.mp4`、`videos/cmp_official_t8.mp4`
   - 官方对照 `output/video/MiniMax_H3_00001_.mp4`（20 步静音）、`_00002_.mp4`（8 步碎裂，正常）

## E. 一键复核脚本口径

谱/rms 分析用本机 ffmpeg 抽 s16le 32k mono + numpy rfft（每整秒窗、Hann），指标：
`rms`、`8–15k 带`（8000–15000Hz FFT bin RMS）、`12k±200`（11800–12200Hz）、`peak Hz`。
历史评测脚本：`scripts/chain_metrics.py`、`audio_locate.py`、`encoding_probe.py`（本地会话
scripts/ 保留副本）。
