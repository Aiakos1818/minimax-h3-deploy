# MiniMax-H3-Deploy

MiniMax H3（原生音视频生成）在 **双 RTX 2080 Ti 22G（NVLink）** 上的部署配置、补丁与实测记录。

- **CLIP（int4）+ UNET（int8）双卡常驻**，生成过程中不卸载
- RayLight FSDP 权重分片 + Ulysses 序列并行 + 可选 pipelined Ulysses（KV int8 压缩）
- 已验证：`864x480`、`124` 帧、`20` 步，t2v 与 首帧/首尾帧(fl2va) 均可跑通

## 环境版本锁

| 项 | 版本 |
|---|---|
| GPU | 2x RTX 2080 Ti 22G（sm75，NVLink），内存 16G |
| 系统 | Ubuntu 24.04 |
| Python / torch | 3.12 / **2.14.0+cu130**（cu130 是 comfy-kitchen 在 20 系上用优化内核的前提） |
| ComfyUI | `Comfy-Org/ComfyUI` @ `30bdda1` |
| raylight | `komikndr/raylight` @ `903f50e` |
| comfy-kitchen / comfy-aimdo | 0.2.31 / 0.4.15 |
| 启动 | `./start.sh`（`main.py --listen 0.0.0.0 --port 8188`，导出 `H3_MP_RETAIN_CPU_WEIGHTS=1`） |

## 上游 fork 与分支（我方改动都在这三个 branch 上）

| 上游 | 我的 fork / 分支 | 改动 |
|---|---|---|
| `Comfy-Org/ComfyUI` | `Aiakos1818/ComfyUI` : `h3-sm75-deploy` | ① `comfy_extras/nodes_multigpu.py`：`Select * Device` 传入模型的 `supported_inference_dtypes`，避免 H3 在 sm75 被强改 fp16（会 NaN）。② `comfy_api/latest/_input_impl/video_types.py`：音频 mux 前 `nan_to_num().clamp(-1,1)`（AAC 拒收 NaN，报 `avcodec_send_frame` EINVAL 22）。③ 新增 `start.sh`/`stop.sh` |
| `komikndr/raylight` | `Aiakos1818/raylight` : `dual-2080ti-persist` | 8 文件 +622/-48：persist 复用（`reuse_epoch`）、cond_audio / denoise mask、H3 xdit 补丁（`xdit_context_parallel.py`）、pipelined Ulysses（新模块 `distributed_modules/pipelined_ulysses.py`）、失败 actor 终止、`nodes_custom_sampler.py` 取样结果处理 |
| `1506086927/SageAttention2_Optimized_Test` | `Aiakos1818/SageAttention2_Optimized_Test` : `sm75-2080ti` | `setup.py`：`-std=c++20`、`--cudart=static`、跳过 torch CUDA 版本检查，使 SM75 可编译 |

未改、仅记录 revision 的参考项目：`HerrgottMargott/Herrgotts-H3-Infinite-Continuation-Suite` (v1.4.0)、`NikoDemon80/ComfyUI-H3-Motion-Context`。

## 本仓库内容

```
nodes/comfyui_h3_multigpu_clip/   # 多卡 Qwen3-VL-32B CLIP：按层拆卡 + 磁盘 cond 缓存 + 显存优化
nodes/h3_vae_unload/              # UnloadVideoVAE：VAE 三处腾挪 + 清 ray worker CUDA 池
workflows/                        # 已验证工作流 + 历史 API/UI 工作流
scripts/                          # chain_director_v2（persist 续接链）、web 控制台(:8189)、gen/gen_dual
docs/                             # 部署与踩坑文档（audio / chain_director / gen_dual / gen）
start.sh stop.sh                  # 双卡启动脚本
```

把 `nodes/*` 放进 `ComfyUI/custom_nodes/`，`workflows/*.json` 放进 `ComfyUI/user/default/workflows/`。

### `h3_vae_unload` 的作用（关键）

三个挂点，全程不影响 CLIP/UNET 常驻：

1. **CLIP 编码前**：踢掉可能残留的视频 VAE
2. **条件编码后**：踢掉关键帧 VAE，再进采样
3. **解码后**：踢掉 VAE + **清 ray worker 的 CUDA 缓存池**（`free_cached_vae` RPC，只 `empty_cache`）

第 3 条是 124 帧 i2v 能否跑通的关键：worker 的 `cudaMallocAsync` 池会保留采样期瞬时缓冲（124 帧约 3.9GB/卡），主进程用不到，导致下一轮条件阶段差几十 MB OOM。

### 环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `H3_MP_RETAIN_CPU_WEIGHTS` | `1`（start.sh 里导出） | 保留 CLIP 的 CPU 主存储，`offload_after_encode` 才能快速重绑 |
| `H3_MP_KEEP_VISUAL_CPU` | `1` | t2v 用不到 vision tower，留在 CPU 省 1.1GB（GPU0）；i2v/ref2v 也能用，代价是每张图多 3~5s 编码；仅当显存宽裕时才设 `0` |

## 模型清单（放到 `ComfyUI/models/`）

| 目录 | 文件 | 大小 | 来源 |
|---|---|---|---|
| `diffusion_models/` | `minimax_h3_fl2va_pruned_int8_convrot.safetensors` | 20.97G | Comfy-Org/MiniMax-H3 |
| `text_encoders/` | `qwen3vl_32b_minimax_h3_int4_convrot.safetensors` | 14.95G | 魔搭 `gordonz/MiniMax-H3-int4-convrot-pruned` |
| `vae/` | `minimax_h3_video_vae_int8_convrot.safetensors` | 3.17G | 同上 |
| `vae/` | `minimax_h3_audio_vae_fp32.safetensors` | 0.58G | Comfy-Org/MiniMax-H3 |
| `diffusion_models/` | `minimax_h3_ref2va_pruned_int8_convrot.safetensors`（可选） | 20.97G | Comfy-Org/MiniMax-H3 |

## 实测（864x480 / 20 步 / 模型常驻）

| 场景 | 总耗时 | 采样 |
|---|---|---|
| t2v（124 帧） | **3 分 37 秒** | 8.86 s/it（2:49） |
| i2v 首尾帧（124 帧） | **4 分 12 秒** | 9.63 s/it（3:05） |
| 对照：单卡 int4 fp32 | ~22 分钟 | 61 s/it |

首/尾帧生效验证：输出首帧 vs 输入首图 MAE ≈ 2.9，末帧 vs 输入尾图 MAE ≈ 31。

## 已知边界

- 采样期每卡峰值 ~21.4/21.48GB（贴边，靠 cudaMallocAsync 池回收维持）。**不要再加分辨率或帧数**；要更大需加卡或降档。
- sm75 上 fp8 是模拟执行（慢），int4 的 convrot 也走反量化回退路径；本部署选 int8 UNET + int4 CLIP 是显存与速度的折中。
- 内存只有 16G（本机）：`use_mmap=True` 必须开。
