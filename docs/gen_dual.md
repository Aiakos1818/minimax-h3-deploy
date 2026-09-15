# 工作节点 2：gen_dual.py（双卡 Raylight TP 单任务）

> 阶段：双卡 FSDP/Ulysses 单段生成。包含整个双卡部署/版本锁/性能调优记录
> （并入原 `DUAL_TP_NOTES.md`），以及为链式任务装载优化做的 CLIP SSD-offload
> 往返实验归档（并入原 `clip_offload_experiment_20260904.md`）。
> 脚本能力此后被 chain_director v1/v2 的多段链继承；gen_dual 保留为单任务双卡 CLI。

## 一、gen_dual.py 职责与用法

- 向 `:8188` 提交**双卡 raylight** workflow（`api_raylight_h3_i2v.json`）：
  RayInitializer（FSDP/Ulysses）+ `MiniMaxH3ImageToVideo` + RayBasicScheduler +
  XFuserSamplerCustomAdvanced。默认 clear=True（B 策略）。
- 支持文本（不带 `--image` 即文生）与首帧图（`--image`），同模板即 t2v/i2v。
- `--length` 直接给帧数（覆盖 seconds）；`--seconds` 走 `ComfyMathExpression`（17k+5 snap）。

```bash
# 文生视频（864x480/124帧/20步/双卡）
python scripts/gen_dual.py --prompt "..." --width 864 --height 480 --length 124 --seed N --wait
# 文 + 首帧图（i2v）
python scripts/gen_dual.py --prompt "..." --image input/xxx.png --wait
# 降步快速出片（RayBasicScheduler steps，默认 20，最小 8）
python scripts/gen_dual.py --prompt "..." --steps 12 --wait   # e2e ~4.8 min
```
输出自动落 `output/video/MiniMax_H3_XXXX_.mp4`。

## 二、双卡部署与调优要点（原 DUAL_TP_NOTES 并入）

2× RTX 2080 Ti 22G（NVLink）上以 ComfyUI + Raylight FSDP/Ulysses + SM75 SageAttention
实现双卡并行推理的完整记录。针对 15G 内存与 Turing(sm75) 的每一处决策。

### 2.1 性能成果（同条件口径：864×480 / 124 帧 / 20 步 / INT8）

| 形态 | 采样步速 | 采样耗时 | 备注 |
|---|---|---|---|
| 单卡无 Sage（早期基线） | ~60 s/it | ~20 min | 纯 PyTorch attention |
| 单卡 + SM75 Sage | 14.3 s/it | 4:46 | Sage kernel 生效 |
| **双卡 FSDP+Ulysses+Sage** | **~7.9 s/it** | **~2:30** | GPU0/1 均满载 99% |

说明：3.5 s/it 出现在 56 帧/640×640 短任务（序列短、单步轻），长序列（124 帧）稳态为
7.9 s/it。相对单卡无 Sage 提速 ~7.7×，相对单卡 Sage 提速 ~1.8×。

- **B 模式端到端**（864×480/124帧：编码 55s + FSDP 重载 ~90s + 采样 ~150s + 解码保存）
  ≈ **6~7 min/任务**（两次实测 374s / 407s）。
- 编码器多卡放置 36s（dispatch+encode，GPU1 参与）；单卡 CPU-offload 参考 ~60s+。
- clear=True 后 worker 显存自动回落：11.5 GB/卡 → 1.1 GB/卡（VAE 等主进程缓存仍保留 ~8G，
  属 Comfy 模型缓存，下任务自动腾挪）。

### 2.2 单次双卡任务阶段流水（B 模式 / clear=True，864×480 / 124 帧 / 20 步实测）

以提交时刻为 0：

| 阶段 | 用时 | 累计 | 发生位置 | 说明 |
|---|---|---|---|---|
| 0 提交 & 校验 | ~1s | ~0s | driver | /prompt 校验通过即入队 |
| 1 编码器装载 | ~19s | ~19s | driver(CPU+mmap) | Qwen int8 权重 mmap 打开、CLIP 构造（懒页，不整读 RAM） |
| 2 Qwen 分发两卡 | ~27s | ~46s | GPU0/1 | `placement.dispatch()`：visual/embed/逐层 `.to(cuda:0/1)`，每卡约 24G（log: dispatch=26.9s） |
| 3 双卡并行编码 | ~9s | ~55s | GPU0/1 | prompt(+图) 前向；输出回 CPU，`load_state_dict(assign)` 重绑 mmap（offload 0.2s）→ 两卡归还 |
| 4 FSDP 重载 | ~91s | ~146s | worker ×2 | clear=True 使每任务重装 DiT int8 分片（2×~11.5G）+ NCCL init/通信自检（实测 submit→sampling_start=146s） |
| 5 采样 20 步 | ~150s | ~296s | worker ×2 | Ulysses all-to-all + FSDP all-gather + SM75 Sage；稳态 ~7.9s/it（log 2.0/20→100% 02:30） |
| 6 解码+音频+落盘 | ~60s | ~360s | driver(GPU0) | latent→VAE video(fp16)/audio decode→pyav(x264) 封装 |
| 7 清理 | 数秒 | ~374s | 全 | 采样结束 worker clear 释放分片（11.5G→1.1G/卡） |

关键口径：
- 首次冷任务额外含 ray 集群冷启动（raylet/GCS/dashboard），提交→采样起步约 **+2~3min**；
  连续任务复用已活 cluster，仅付阶段 4 的 ~91s 重载。
- 编码在 worker 拉起/重载之前完成，不占 worker 显存窗口 → 无 OOM。
- 阶段 2~3 内存峰值依赖 mmap 页缓存（可回收、非匿名）；worker 各 ~1.4G RSS；
  整体可用内存 ≥8G。
- 数字为分段推算口径（submit→encode_done=55s、→sampling_start=146s 为实测时间戳；
  e2e 374s / 407s 两任务实录）。

**固定等待拆解（"非采样的固定成本"≈137s≈2.3 min）**：装载 19s + dispatch 27s +
FSDP 重载 91s。其中只有 **FSDP 重载 91s 是 clear=True 引入的增量**；19+27（编码器装载
/dispatch）属固有成本——Qwen 编码器从不驻留（编码完即 `assign` 回 mmap），任何策略下每
任务必付，2 卡显存放不下"DiT 驻留 + Qwen"同驻。clear=False 省 91s 的前提是两者能同驻
（如 4×22G：DiT~11.5G+Qwen~12.6G≈18G/卡 <22G），2 卡下无此条件。

**解码+落盘（阶段 6，~60s）三分实测**（GPU/CPU 打点，20 步与 12 步数值一致、与步数无关）：

| 子段 | 时长 | 特征 |
|---|---|---|
| VAE 装载/音频/前置 | ~28s | latent gather、4.97G VideoVAE 装载、音频合成；GPU≈0、CPU 中 |
| **VideoVAE 帧解码** | **~26s** | GPU0 满载 100%（帧数驱动，算力固定） |
| **pyav 封装 + x264 写盘** | **~4s** | 进程内 pyav（无外部 ffmpeg 进程），CPU |

实测后驳回"CreateVideo 提速"方案：x264 封装仅 ~4s，即便 preset 极速上限也省 1~3s，ROI≈0。
若要继续压缩 ~60s，只能看 ~28s 前置段（含 VAE 装载与 latent 传输），与编码器复用同属低 ROI。

**降步快速档**：步速与步数无关（20 步 7.9 s/it、12 步 7.7 s/it）。采样是可砍的唯一大头：
- `--steps 12`（实测 00009）：采样 ~92s，e2e **4.8 min**；
- `--steps 8`（理论）：采样 ~60s，e2e ≈ 4.3 min。质量需按出片需求权衡。

### 2.3 reload 段构成与"FSDP 结构复用"探索（profiling 实测结论，勿重复踩坑）

对每任务 reload 段（encode 后 ~91s）打点实测（`load_unet`/`custom_sampler_advanced`
内 wall-time + GPU 曲线）：
- **load_unet 数据装配**（`fsdp_load_diffusion_model`，meta 模型构建 + mmap + FSDP
  分片源准备）：**5~52s**，波动大（页缓存/内存压力相关）。
- **sample 内首次准备**（`sample:start → guider_call`）：**93~134s**，主成分 =
  `patch_fsdp` 对 diffusion_model 执行 `fully_shard_bottom_up` + 权重物化 + comm 建立，
  且**每次任务重演**。H3 forward 不经 Comfy `model_function_runner`（打点不命中），无法进一步细拆。
- 归因修正：此前把 91s 视作"FSDP 数据重载"不准确；真正大头是每次 sample 前的 FSDP
  包裹/物化，`load_unet` 只占小头。

尝试的改造（Phase 1）与失败判定：
- 思路：worker **软驻留**——clear 时仅释放 GPU 分片存储，保留 FSDP 结构与 CPU state_dict；
  `patch_fsdp` 对"已注册 FSDP"分支支持仅重物化（不重 wrap/comm）。目标 reload 91s→~10s。
- 源码注入均在文件层验证正确（`inspect` 确认），但**运行行为与磁盘源码不一致**：
  `guider.sample()` 返回后注入代码不执行、clear 触发与软清日志缺失，疑 raylight 经 ray
  的 worker 字节码/对象复用机制所致（多次重启/换 worker 后依旧）。另 H3 int8/量化 DTensor
  分片在 `free_fsdp_vram` 存在 skip，需强制释放。
- 结论：**2 卡下该分支软驻留改造不可靠，已完整回退**（`ray_worker.py`/`model_patcher.py`
  0 残留补丁，默认硬清恢复、冒烟通过）。每任务 ~91s 重载为 2 卡硬约束，不建议再投入；
  结构性去掉需 raylight 上游支持或扩到 4×22G（分片与编码器同驻）。

> 后续 v2 persist 模式（chain_director_v2.md）用**不重启服务的幂等 RayInitializer 复用**
> 在任务内段间省掉 reload——与本节 clear=True 的每任务重载互补：任务内复用、跨任务仍清场。

### 2.4 版本锁（严格）

| 组件 | 版本/提交 | 备注 |
|---|---|---|
| ComfyUI | master `30bdda1` | 2026-09 主线 |
| torch / torchvision | 2.14.0+cu130 / 0.29.0 | aliyun pytorch-wheels/cu130 直下 |
| **xfuser** | **==0.4.5** | **勿升 0.6.0**（envs stub / RankGenerator 签名不兼容 raylight@903f50e） |
| raylight (custom node) | fork `komikndr/raylight` @`903f50e` + Eanya patch | 在 `ComfyUI/custom_nodes/raylight`，`-e` 安装 |
| SageAttention (SM75 fork) | `1506086927/SageAttention2_Optimized_Test` @`539026f` | 需自编译（见下） |
| H3 多卡 Qwen 节点 | `comfyui_h3_multigpu_clip` | `ComfyUI/custom_nodes/` 下 |
| ray | 2.58.0 | pypi |
| yunchang | 0.6.4 | pypi |
| kernels | 0.16.1 | pypi（xfuser 依赖） |
| NCCL | `nvidia-nccl-cu13==2.30.7` | **严禁装 cu12 系**，否则 `import torch` 报 `ncclCommResume` 未定义/缺库 |

SM75 Sage fork 编译要点（`setup.py` 三处本地改动）：
1. monkeypatch 绕过 torch `_check_cuda_version`（用 nvcc 12.8 编译 torch cu130 场景）；
2. NVCC 加 `--cudart=static`（避免运行时依赖 libcudart.so.12）；
3. 编译标准 `-std=c++17`→`-std=c++20`（torch 2.14 头硬性要求）；
4. qattn 扩展默认在 sm75 下也编（H3 长序列 self-attn 依赖 `_qattn_sm80` 内核，其名虽为
   sm80 但含 sm75 gencode）。

产物 `dist/sageattention-2.1.2-*.whl` 装入 comfyenv。

### 2.5 内存三律（15G RAM 生存指南）

1. **`H3_MP_RETAIN_CPU_WEIGHTS=1` 必须开**：H3MultiGPUCLIPLoader 编码结束后用
   `load_state_dict(assign=True)` 把 GPU 层重新绑回 mmap 主存储（offload 实测 0.2s）。
   若为 0 走 `.to("cpu")` 会复制出 25G 匿名内存 → swap 打爆、整机 thrash 死锁。
2. **44G 显存装不下 DiT 分片 + Qwen 编码器同时**：DiT FSDP 2×~11.5G 与 Qwen 2×~24G
   互斥，必须顺序执行。
3. swap 64G（/swapfile_h3）兜底；关 swap 必死。

### 2.6 worker 显存策略（B：clear_vram_after_sampling=True，已固化为默认）

- workflow 文件：`scripts/workflows/api_raylight_h3_i2v.json` 中
  `RayInitializer.clear_vram_after_sampling=True`。
- 效果：每个任务结束 worker 释放分片 → 下一次**换 prompt/图**的任务自动重载（~90s），
  不会因编码器 dispatch 而 CUDA OOM。
- 代价：任务间不能复用已加载分片（固定 ~90s 重载）。若要同 prompt 反复只调 seed/步数
  并希望秒级复用，可临时改回 False，但换内容前需 `RayKill`/重启释放（否则编码 OOM）。

### 2.7 启动与使用

```bash
~/MiniMax-H3-Deploy/scripts/start-comfyui-for-minimax-h3.sh
# H3_MP_RETAIN_CPU_WEIGHTS=1 + --use-sage-attention --disable-dynamic-vram --disable-cuda-malloc
```
生成与降步快速档见本文第一节命令。输出自动落 `output/video/MiniMax_H3_XXXX_.mp4`。

### 2.8 踩坑清单（每条都是真金白银）

- **NVFP4 编码器**：12.6G 更小但需 Blackwell；sm75 硬上 emulated 会崩 → 坚持 INT8（25.3G）。
- **nvidia-nccl-cu12==2.28.9**：误装后 torch cu130 import 挂（应装 cu13 系 2.30.7；
  若已装错先卸载再 `--reinstall nvidia-nccl-cu13==2.30.7`）。
- **torch 2.14 头**：Sage fork 的 pybind 需要 C++20（`#error C++20`），且 symint 报错本质是 C++17。
- **xfuser 0.6.0**：raylight compat 会预占 `xfuser.envs`（stub 缺 `get_device_name`）→
  修复 = raylight `__init__.py` 先 `import xfuser.envs` 再 `install_minimal_xfuser()`；
  RankGenerator 新增必填 `order` → 降到 0.4.5。
- **TORCH_FLASH attention**：Turing 不支持 → 设 `XFuser_attention=SAGE_FP16`。
- **Comfy 新版 `FinalLayer.forward`**：需 `sigma/sample_sigmas/shifts` 三参 → raylight
  `xdit_context_parallel.py:219` 补 `sigma_v, transformer_options.get("sample_sigmas"), (shift_v, shift_a)`。
- **pkill 自杀**：脚本里 `pkill -f "main.py --listen"` 会匹配自身命令行 →
  用 `pkill -f "[m]ain.py --listen"`。
- **swap thrash 死锁**：症状 = 整机 load>10、ssh 无响应、GPU1 空转；元凶多为主进程 25G
  匿名内存（见内存三律 1）。止损：`pkill -9 -f "[m]ain.py..."` 等系统缓过来。

### 2.9 上游项目与本地差异

参考项目：
- **Eanya-Tonic/2080ti-minimax-h3** —— 主参考，专为 RTX 2080Ti(SM75) 的 MiniMax-H3
  多卡 ComfyUI 方案（raylight 补丁、多卡 Qwen 节点、SM75 Sage 编译法、实测基准）。
- **komikndr/raylight** @`903f50e` —— 多卡框架本体（ray worker + FSDP + xFuser/Ulysses）。
- **1506086927/SageAttention2_Optimized_Test** @`539026f` —— Turing(SM75) Sage kernel fork。
- 间接上游：xFuser/xDiT、yunchang、ComfyUI、Comfy-Org MiniMax-H3 权重套。

相对上游我们做的改动：
1. **硬件拓扑**：上游面向 4 卡双 NVLink 岛；本机为 2 卡单 NVLink 岛（U2/R1，
   "480p 低延迟优先"档）。
2. **运行环境**：上游 Docker(NGC torch 2.11-cuda13)；本机裸机 pip cu130（torch 2.14.0+cu130）。
3. **Sage fork 编译**：为 torch 2.14 打 setup.py 补丁——monkeypatch 绕过 CUDA 版本强校验
   （nvcc 12.8）、`--cudart=static`、`c++17→c++20`、sm75 亦编入 `_qattn_sm80`。
4. **xfuser 锁 0.4.5**（0.6 与 raylight@903f50e 冲突）+ 修 raylight `__init__.py`
   （先真实 import `xfuser.envs` 再装 compat stub）。
5. **适配最新 ComfyUI master**：`xdit_context_parallel.py` 补 `FinalLayer` 新三参
   （`sigma_v/sample_sigmas/shifts`）。
6. **15G 内存落地**：`H3_MP_RETAIN_CPU_WEIGHTS=1` + `clear_vram_after_sampling=True`
   （B 策略，任务后释放分片，换 prompt 不 OOM）。
7. **attention 后端 `SAGE_FP16`**（Turing 无法跑 TORCH_FLASH）。
8. **单卡同享 Sage**：`--use-sage-attention` 全局开启（单卡 60→14.3 s/it）。
9. **工程化**：UI→API workflow 转换器、`gen_dual.py`/`gen.py` CLI、`start-comfyui-for-minimax-h3.sh` 参数固化、模板落盘。

> 注：本节为**双卡部署适配**阶段对 raylight 的全部改动；persist 复用（nodes.py/ray_worker.py
> reuse_epoch）是 v2 阶段新增，见 chain_director_v2.md；cond_audio/denoise-mask 逐行补丁
> 见 audio.md。

## 三、CLIP SSD-Offload 往返实验归档（2026-09-04，原文件并入）

> 结论先行：**SSD-offload 方向 no-go**。实验同时澄清了 MiniMax-H3 文本编码器 dispatch 慢
> 的真实机制（int8 冷读量主导 + page cache 驻留不足），并给出装载类优化收益天花板评估
> （端到端 ~7%）。可复现脚本保留于 `scripts/clip_ssd_roundtrip.py`，结构化数据见
> `~/MiniMax-H3-Deploy/.clip_offload_test/report.json`。

### 3.1 背景与动机

- 现象：CLIP（qwen3vl_32b int8_convrot 文本编码器）每次 encode 的 dispatch 稳定 26-34s
  （历史 comfy.log），成多段连续 i2v/ref2v 任务链的段间排队成本。
- 动机：为"unet 常驻 + 段间 CLIP 往返"设想一条 offload 到 SSD 的快路径：encode 后把权重
  落 SSD 分片、GPU 空出给 unet；下段从分片回载。

### 3.2 实验设计（单进程四轮）

1. S0 baseline：`load_clip(mmap)` → `H3MultiGPUPlacement.dispatch` → encode → stock offload。
2. S1 build：把每设备权重按 dispatch 计划顺序流式写入双分片 `.dev0.bin`/`.dev1.bin` + header。
3. S2 roundtrip：fresh load → 把模块参数 rebind 到分片 mmap view → dispatch → encode → SSD offload。
4. S3 幂等重复轮。

判据：encode 输出 bitwise（sha256）、offload 后 GPU `memory_allocated==0`、各阶段时序 +
RSS/VmSwap 采样。

### 3.3 关键数据

| 阶段 | dispatch | encode | offload | GPU 归零 | bitwise 同基线 |
|---|---|---|---|---|---|
| S0 基线(safetensors) | 20.9s | 3.6s | 28.5s（物化 13GiB 进 swap） | 是 | — |
| S2 SSD 轮 1/2 | 61.1/64.5s | ~1.4s | 0.2s（无效） | **否**（剩 10.8/12.6GiB） | 是 |

- `s1_verify_mismatches=0`：分片 51.5GB 与写源逐字节一致。
- 分片已删除（51.5GB 释放）；`report.json` 保留。

### 3.4 三条机制发现

1. **权重不是普通 bf16 Tensor，而是 `comfy_kitchen.tensor.base.QuantizedTensor`**
   （wrapper Tensor subclass）：外表 bf16 shape，真实数据在私有属性 `_qdata`
   （int8，22.71GiB）+ `_params`（scale/旋转元数据）。GPU 实际驻留 int8 ~13.4/12.6GiB/卡
   （placement 日志的 23.45GiB 是表面 bf16 虚算）——这解释了 32B 为何能在 2×22GiB 跑通。
2. **`param.data = 普通 mmap 张量` 对 QuantizedTensor 静默无效**（debug 实证：
   data_ptr/device 不变）。因为真实数据在构造时一次性填入 `_qdata/_params`，`.data`
   只是 `_make_wrapper_subclass` 的虚拟视图。SSD rebind/offload 因此全程未生效
   （S2 实际仍走原 CPU 权重，bitwise 一致只是普通幂等）。且若 rebind 真生效，dispatch
   会把 48GiB 浮点全量上卡 → 必然 OOM。双向无解。
3. **dispatch ~21s 的真实瓶颈 = 22.7GiB int8 冷读**，不是 PCIe 拷贝：
   - 文件为 file-backed mmap（maps 实证），数据靠 page cache 供给；
   - RAM 15GiB → 27GB 模型文件无法常驻 cache → 每次 dispatch≈冷读磁盘；
   - 实测磁盘读 25.3GiB = 18.9s（~1.3GB/s），有效 dispatch 带宽 ~1.05GB/s（PCIe3 理论 ~11GB/s）；
   - cache 清空后 dispatch 22.7s ≈ 自然态 21s ≈ S0 20.9s：15GB 下永远接近冷读。

### 3.5 结论：SSD-offload no-go

- 机制不兼容：QuantizedTensor 结构使".data 替换分片"无效；
- 无理论增益：权重本就在 SSD（safetensors mmap），瓶颈是 22.7GiB 读**量**，与数据源格式/布局无关；
- 排它替代：GPU 释放靠 stock offload（retain=1 时 rebind CPU 主副本，不物化）已可用。

### 3.6 收益天花板账本（为何装载类优化上限 ~7%）

单段 124帧/768p 端到端 ~250s：CLIP ~25-34s + UNet FSDP 加载（仅换模型时）+ **采样
100-200s+**（大头，吃分辨率×帧数×步数×算力）+ VAE decode 30-90s。

- 内存升级（最优装载项，预估 21s→5-8s 连续态）：单段省 ~15-20s ≈ **~7%**；且本机无扩容选项。
- nvfp4 真身（14.6GiB，未下载）：仅减装载 -35%，采样算力不变（Turing 软件解码）。
- **采样/解码才吃算力**，不吃装载优化。

### 3.7 后续做法（实验的落点）

装载方向投入回报低；真正被采纳的提速是：CLIP **cond 预编译缓存**（同 prompt 段间零
dispatch，见 chain_director_v1.md）与 **v2 persist**（同任务段间 FSDP 复用、免重载，
见 chain_director_v2.md）。

### 3.8 产物位置

- `~/MiniMax-H3-Deploy/.clip_offload_test/report.json`（结构化数据）
- `~/MiniMax-H3-Deploy/scripts/clip_ssd_roundtrip.py`（可复现主脚本，`--smoke`/默认全流程）

## 四、双卡底座权重与 r2v（ref2va）通道

- 扩散底座（`ComfyUI/models/diffusion_models/`，均 int8 convrot）：
  - `minimax_h3_fl2va_pruned_int8_convrot.safetensors`（20.97GB，t2v/i2v/首帧续写）
  - `minimax_h3_ref2va_pruned_int8_convrot.safetensors`（20.97GB，2026-09-04 新增，r2v/参考主体）
- 来源：**魔搭 `Comfy-Org/MiniMax-H3`**（`modelscope.snapshot_download` 或 resolve 直链；
  `downloads/` 留本地暂存）。两权重字节数完全一致 → 同一 H3 DiT 架构，仅微调目标不同
  （fl2v first/last 帧 vs ref 参考条件）；fl2va/ref2va 前端任务节点
  （`MiniMaxH3ImageToVideo`/`MiniMaxH3ReferenceToVideo`）差异全在 conditioning 层
  （`minimax_keyframes` vs `minimax_refs`）。**保留两权重不互删**（未经 ref2va 顶替 fl2va
  的 i2v/t2v 等质对照，不冒险删）。
- REF2VA 冒烟已通过：`scripts/workflows/api_raylight_h3_ref2v.json`（API 图；RayUNETLoader
  指 ref2va；`MiniMaxH3ReferenceToVideo` 传 `ref_images.ref_image_0:[LoadImage,0]`；
  480p/124帧/8步）→ `output/video/MiniMax_H3_00021_.mp4`，e2e 275s，worker 回落 620M。
  参考口最多 9 图 `<Picture N>`（`ref_image_0..8`），另有 ref_videos/ref_audios
  （Audio 需 audio_vae）。
- **前端 r2v 工作流**：`scripts/workflows/video_minimax_h3_raylight_ref2v.json` +
  `ComfyUI/user/default/workflows/` 同文件（浏览器 F5 → Workflow 列表双击
  `MiniMax H3 REF2VA`）。基于 raylight 官方示例精简：3 个 LoadImage（默认连好
  `input/example.png`→`ref_image_0`，另两个留空随传随连）、外部提示词框
  （`<Picture N>` 引用）、`ResolutionSelector`（0.4MP=864×480）、`PrimitiveFloat` 秒数
  （5→124 帧）。CLIP 已替换为 `H3MultiGPUCLIPLoader`（与 API 模板一致，勿改回单卡
  CLIPLoader）。UI 打开 Run 前若 139/160 空图口无需处理（未连不执行）。回归已验证
  （等价 API 4 步 205s 通过）。
- RayInitializer 注意：API 提交必须带 `use_mmap` 与 `RAYLIGHT_ULYSSES_KV_INT8`
  （服务 required，抄 i2v 模板值）。

## 五、目录速查

```
~/MiniMax-H3-Deploy/
├── ComfyUI/custom_nodes/{raylight, comfyui_h3_multigpu_clip}
├── vendor/SageAttention2_Optimized_Test/   # SM75 sage 源码（编译产物 dist/*.whl）
├── scripts/gen_dual.py                     # 双卡 CLI
├── scripts/workflows/api_raylight_h3_i2v.json   # 双卡图生模板（API 提交，clear=True 已固化）
├── scripts/workflows/api_raylight_h3_t2v.json   # 双卡文生模板（API 提交；纯文生，无图分支）
├── scripts/workflows/video_minimax_h3_raylight_t2v.json  # 双卡文生·前端版
├── ComfyUI/user/default/workflows/video_minimax_h3_raylight_t2v.json  # 同上前端版
├── scripts/workflows/api_raylight_h3_ref2v.json          # 双卡 r2v/参考生模板（API 提交）
├── scripts/workflows/video_minimax_h3_raylight_ref2v.json  # 双卡 r2v·前端版
├── ComfyUI/user/default/workflows/video_minimax_h3_raylight_ref2v.json  # 同上前端版
├── scripts/start-comfyui-for-minimax-h3.sh / stop.sh
└── output/video/
```

### 双卡文生视频（前端 / API 用法）
- 前端：浏览器打开 ComfyUI → Workflow 列表选 `video_minimax_h3_raylight_t2v`
  （若列表不刷新则按 F5）。默认即文生：改 prompt/seed/分辨率（ResolutionSelector）/帧数
  （MiniMaxH3ImageToVideo 的 length，124≈5.2s）后 Run。
- 图生：给画布里的 `LoadImage` 上传图，把它的 IMAGE 输出连到 `MiniMaxH3ImageToVideo`
  的 first_frame 端口（默认不连=文生）。
- API（脚本/curl）：用 `scripts/workflows/api_raylight_h3_t2v.json` 提交即可（示例替换
  prompt/seed 后 `POST /prompt`）。也可 `gen_dual.py --prompt "..."`（不带 --image 即文生）。
- 注意：`api_*.json` 是 API 格式，前端双击不会渲染；要可视化必须加载
  `video_minimax_*.json` 前端版。

## 六、退役说明

gen.py / gen_dual.py 是单任务驱动。多段续接与素材（图锚/参考音视频）能力此后移交
chain_director_v1/v2（见对应文档）；audio 专项结论在 audio.md。
