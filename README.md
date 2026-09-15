# MiniMax-H3-Deploy

MiniMax H3（原生音视频生成）在 **双 RTX 2080 Ti 22G（NVLink）** 上的部署配置、补丁与实测记录。

- **CLIP（int4）+ UNET（int8）双卡常驻**，生成过程中不卸载
- RayLight FSDP 权重分片 + Ulysses 序列并行 + pipelined Ulysses（KV int8 压缩，可选）
- 已验证：`864x480`、`124` 帧、`20` 步，t2v 与 首帧/首尾帧（fl2va）均可跑通
- 实测（模型常驻）：t2v **3 分 37 秒** / i2v 首尾帧 **4 分 12 秒**

---

## 从零复现（5 步）

### 1. 环境与依赖（版本锁定）

| 组件 | 版本 | 说明 |
|---|---|---|
| 系统 | Ubuntu 24.04，内存 16G+ | 内存越小越依赖 `use_mmap` |
| GPU | 2x RTX 2080 Ti 22G，NVLink | sm75；两张卡等价也可以 |
| Python | 3.12 | |
| torch / torchvision | **2.14.0+cu130** / 0.29.0 | **必须 cu130**：comfy-kitchen 对 20 系(GPU 20+)的优化内核要求 cu130+ |
| triton | 3.8.0 | |
| comfy-kitchen | 0.2.31 | int4/int8/fp8 量化内核 |
| comfy-aimdo | 0.4.15 | DynamicVRAM |
| ray | 2.58.0 | raylight 的 worker |
| xfuser | 0.4.5 | Ulysses 序列并行 |
| sageattention | 2.1.2 | **sm75 需用本仓库 fork 自行编译**（见第 2 步） |
| einops / safetensors / transformers | 0.8.2 / 0.8.0 / 5.16.1 | 其余依赖随 ComfyUI requirements |

```bash
# 先装 ComfyUI 自带 requirements.txt，再装我们的补充依赖
pip install -r ComfyUI/requirements.txt
pip install -r requirements-h3.txt          # 见本仓库同目录（torch 需 cu130 轮子）
# sageattention：sm75 需源码编译（见第 2 步的 fork）
```

### 2. 拉取代码

```bash
# (a) ComfyUI：用我们的分支（含 3 个必要修复）
git clone https://github.com/Aiakos1818/ComfyUI.git
cd ComfyUI && git checkout h3-sm75-deploy
#   或：官方 ComfyUI 上 cherry-pick 三个提交
#   git fetch https://github.com/Aiakos1818/ComfyUI.git h3-sm75-deploy
#   git cherry-pick 82000b6 dfd8c09 b26890d

# (b) raylight（双卡框架）：克隆即用，节点自身会把 src 注入 sys.path，无需 pip install
git clone -b dual-2080ti-persist https://github.com/Aiakos1818/raylight.git custom_nodes/raylight

# (c) 本仓库：自研节点 + 工作流
git clone https://github.com/Aiakos1818/minimax-h3-deploy.git /tmp/h3deploy
cp -r /tmp/h3deploy/nodes/* custom_nodes/
cp /tmp/h3deploy/workflows/*.json user/default/workflows/     # 只需放要用的
cp /tmp/h3deploy/start.sh /tmp/h3deploy/stop.sh .             # 双卡启动脚本

# (d) sm75 的 SageAttention（若要用 sage attention 后端）
git clone -b sm75-2080ti https://github.com/Aiakos1818/SageAttention2_Optimized_Test.git
cd SageAttention2_Optimized_Test && pip install -e . && cd -
```

### 3. 下载模型（放到 `models/` 对应目录）

```bash
# 魔搭（ModelScope）CLI：pip install modelscope
ms() { modelscope download --model "$1" --local_dir /tmp/h3 "${@:2}"; }

# (a) int8 UNET（20.97G）+ 音频 VAE（0.58G）
ms Comfy-Org/MiniMax-H3 \
   diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors \
   vae/minimax_h3_audio_vae_fp32.safetensors

# (b) int4 文本编码器（14.95G）+ int8 视频 VAE（3.17G）
ms gordonz/MiniMax-H3-int4-convrot-pruned \
   Text-Encoder/minimaxH3INT4Convrot_qwen3vl32bInt4.safetensors \
   VAE/minimax_h3_video_vae_int8_convrot.safetensors

# 归位（注意工作流里用的文件名）
mv /tmp/h3/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors models/diffusion_models/
mv /tmp/h3/vae/minimax_h3_audio_vae_fp32.safetensors                          models/vae/
mv "/tmp/h3/Text-Encoder/minimaxH3INT4Convrot_qwen3vl32bInt4.safetensors"     models/text_encoders/qwen3vl_32b_minimax_h3_int4_convrot.safetensors
mv "/tmp/h3/VAE/minimax_h3_video_vae_int8_convrot.safetensors"                 models/vae/
```

| 工作流期望的文件名 | 大小 | 来源 |
|---|---|---|
| `diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors` | 20.97G | 魔搭 `Comfy-Org/MiniMax-H3`（HF 同名） |
| `text_encoders/qwen3vl_32b_minimax_h3_int4_convrot.safetensors` | 14.95G | 魔搭 `gordonz/MiniMax-H3-int4-convrot-pruned` |
| `vae/minimax_h3_video_vae_int8_convrot.safetensors` | 3.17G | 同上 |
| `vae/minimax_h3_audio_vae_fp32.safetensors` | 0.58G | 魔搭 `Comfy-Org/MiniMax-H3` |
| （可选）`diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors` | 20.97G | `Comfy-Org/MiniMax-H3`，参考图/参考音模式用 |

> int4 CLIP 与 int8 视频 VAE 目前只在魔搭 gordonz 仓库有；Comfy-Org 官方仓库只有 bf16/int8/fp8/nvfp4。

### 4. 启动

```bash
cat start.sh
#   export H3_MP_RETAIN_CPU_WEIGHTS=1
#   setsid nohup "$PWD/comfyenv/bin/python" main.py --listen 0.0.0.0 --port 8188 > comfy.log 2>&1 &
./start.sh      # 停止用 ./stop.sh
```

关键环境变量：

| 变量 | 默认 | 说明 |
|---|---|---|
| `H3_MP_RETAIN_CPU_WEIGHTS` | `1` | 保留 CLIP 的 CPU 主存储（start.sh 已导出） |
| `H3_MP_KEEP_VISUAL_CPU` | `1` | t2v 用不到 vision tower，留 CPU 省 1.1GB（GPU0）；i2v/ref2v 也能用，代价是每张图多 3~5s；显存宽裕时才设 `0` |

### 5. 跑工作流

- `workflows/minimax_h3_int4clip_int8unet_raylight.json`：已验证的双卡常驻工作流
  - **t2v**：填 prompt 直接跑
  - **i2v**：`LoadImage` 接到 `MiniMaxH3ImageToVideo` 的 `first_frame` / `last_frame`
- 分辨率/帧数上限：`864x480` + `124` 帧（0.4MP / 5.2s）；**不要再往上加**，见"已知边界"
- 输出：`output/video/*.mp4`（含音轨）

---

## 上游 fork 与分支（三个必要改动）

| 上游 | 我的 fork / 分支 | 改动 |
|---|---|---|
| `Comfy-Org/ComfyUI` | `Aiakos1818/ComfyUI` : `h3-sm75-deploy` | ① `comfy_extras/nodes_multigpu.py`：`Select Model/CLIP/VAE Device` 传入模型的 `supported_inference_dtypes`。H3 声明 `[bf16, fp32]`，sm75 无 bf16 本应 fp32，旧代码却按设备能力选了 **fp16 → 整个前向 NaN**（表现为视频全黑 + 音频 NaN）。② `comfy_api/latest/_input_impl/video_types.py`：音频 mux 前 `nan_to_num().clamp(-1,1)`（AAC 拒收非有限值，报 `avcodec_send_frame` EINVAL 22）。③ 新增 `start.sh`/`stop.sh` |
| `komikndr/raylight` | `Aiakos1818/raylight` : `dual-2080ti-persist` | 8 文件 +622/-48：persist 复用（`reuse_epoch`，段间不重载）、cond_audio / denoise mask、H3 xdit 补丁（`diffusion_models/minimax/xdit_context_parallel.py`）、pipelined Ulysses（新模块 `distributed_modules/pipelined_ulysses.py`）、失败 actor 终止清理、`comfy_extra_dist/nodes_custom_sampler.py` 取样结果处理 |
| `1506086927/SageAttention2_Optimized_Test` | `Aiakos1818/SageAttention2_Optimized_Test` : `sm75-2080ti` | `setup.py`：`-std=c++20`、`--cudart=static`、跳过 torch CUDA 版本检查，使 SM75 可编译 |

仅记录 revision、未改动的参考项目：`HerrgottMargott/Herrgotts-H3-Infinite-Continuation-Suite` (v1.4.0)、`NikoDemon80/ComfyUI-H3-Motion-Context`。

## 本仓库内容

```
nodes/comfyui_h3_multigpu_clip/   # 多卡 Qwen3-VL-32B CLIP：按层拆卡 + 磁盘 cond 缓存 + 显存优化
nodes/h3_vae_unload/              # UnloadVideoVAE：VAE 三处腾挪 + 清 ray worker CUDA 池
workflows/                        # 已验证工作流 + 历史 API/UI 工作流
scripts/                          # chain_director_v2（persist 续接链）、web 控制台(:8189)、gen/gen_dual
docs/                             # 部署与踩坑文档（audio / chain_director / gen_dual / gen）
start.sh stop.sh                  # 双卡启动脚本
```

### `h3_vae_unload` 为什么必要

CLIP+UNET 常驻后每卡只剩 ~3.2GB，而视频 VAE 编/解码需要 ~2.4GB 且会与采样互相挤。三个挂点：

1. **CLIP 编码前**：踢掉可能残留的视频 VAE
2. **条件编码后**：踢掉关键帧 VAE，再进采样
3. **解码后**：踢掉 VAE + **清 ray worker 的 CUDA 缓存池**（`free_cached_vae` RPC，只 `empty_cache`，不释放模型）

第 3 条是 124 帧 i2v 能否跑通的关键：ray worker 的 `cudaMallocAsync` 池会保留采样期瞬时缓冲（124 帧约 3.9GB/卡），主进程用不上，导致下一轮条件阶段差几十 MB OOM。

## 实测（864x480 / 20 步 / 模型常驻）

| 场景 | 总耗时 | 采样 |
|---|---|---|
| t2v（124 帧） | **3 分 37 秒** | 8.86 s/it（2:49） |
| i2v 首尾帧（124 帧） | **4 分 12 秒** | 9.63 s/it（3:05） |
| 对照：单卡 int4 fp32 | ~22 分钟 | 61 s/it |

首/尾帧生效验证：输出首帧 vs 输入首图 MAE ≈ 2.9；末帧 vs 输入尾图 MAE ≈ 31，vs 首图 87。

## 已知边界

- 采样期每卡峰值 ~21.4/21.48GB（贴边，靠 cudaMallocAsync 池回收维持）。**不要再加分辨率或帧数**；要更大需加卡或降档。
- sm75 上 fp8 是模拟执行（慢）；int4 的 convrot 走反量化回退路径。选 int8 UNET + int4 CLIP 是显存与速度的折中。
- 内存 16G 时 `use_mmap=True` 必须开；ray 起 worker 会吃内存。
- `custom_nodes/comfyui_h3_multigpu_clip` 的 cond 缓存写在 `~/MiniMax-H3-Deploy/cond_cache`（可用 `COND_CACHE_ROOT` 源码常量改），纯缓存可随时删。
