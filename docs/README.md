# MiniMax-H3 Deploy 文档索引

MiniMax-H3 双 RTX 2080 Ti 22G（NVLink）raylight TP 部署的工作节点文档。
远端运行目录 `~/MiniMax-H3-Deploy/`，脚本在 `scripts/`，文档与运行目录同步于 `docs/`。

## 工作节点演进（开发顺序）

| # | 文档 | 脚本 | 阶段 | 状态 |
|---|---|---|---|---|
| 1 | [gen.md](gen.md) | `gen.py` | 单机/单卡，官方 `MiniMaxH3ImageToVideo` workflow（`api_local_*.json`）驱动，无续接 | 历史（workflow 模板已清理） |
| 2 | [gen_dual.md](gen_dual.md) | `gen_dual.py` | 双卡 raylight FSDP/Ulysses 单任务（含部署/版本锁/性能/CLIP offload 归档） | 历史（能力移交 v1/v2） |
| 3 | [chain_director_v1.md](chain_director_v1.md) | `chain_director_v1.py` | Herrgotts masked-AV 多段续接链 v1（逐段 queue，每段重载） | 自 2026-09-05 不再维护 |
| 4 | [chain_director_v2.md](chain_director_v2.md) | `chain_director_v2.py` | persist 模式链（FSDP 段间驻留）+ 首段素材分流（文/图锚/参考图视频音频），**现行主线** | 维护中 |
| 5 | [chain_director_v2.md](chain_director_v2.md) | `chain_director_v2_web.py` | 局域网 Web 控制台（`:8189`，stdlib-only）：参数镜像 + 上传 + 串行队列 + 续拍 + 在线预览；经 GPU 宿主 `scripts/` 常驻（`--daemon/--stop/--status`） | 现行 |
| — | [audio.md](audio.md) | —（横切所有阶段） | H3 音频专项：表示与通路、段间 audio 连接、`--ref-audio`/`cond_audio` 引导、**首段底噪现象调查**（steps 收敛结论） | 现行 |

## 快速入口

- **当前推荐做法**：`chain_director_v2.py`，见 [chain_director_v2.md](chain_director_v2.md)。
- **音频踩坑/底噪结论**：见 [audio.md](audio.md)（立体声 2ch/32kHz、8 步弱场景易出噪声态、弱音频 prompt 用 `--steps 20`）。
- **双卡部署版本锁与踩坑**（ComfyUI `30bdda1`、xfuser 0.4.5、NCCL cu13、SM75 Sage 编译法、内存三律）：并入 [gen_dual.md](gen_dual.md)。

## 对上游代码的改动总览（详见各文档对应节）

- **ComfyUI 主仓库**：`git` 干净（master `30bdda1`），H3 native AV-mask/audio_vae 靠 upstream 自带，**未改**。
- **custom_nodes/raylight**（fork `komikndr/raylight` @`903f50e`）：双卡部署适配 + persist 复用 + xdit 补丁，见 gen_dual.md（FinalLayer 三参等）/ chain_director_v2.md（reuse_epoch）/ audio.md（cond_audio/denoise mask）。`nodes.py`/`ray_worker.py` 留有 `.bak_pre_persist` 原版备份。
- **custom_nodes/comfyui_h3_multigpu_clip**：`encode_model_parallel` 内嵌 cond 磁盘缓存（`cond_cache/`），见 chain_director_v1.md。
- **custom_nodes/Herrgotts-H3-Infinite-Continuation-Suite**（`H3ContinuousStartV14`/`H3ContinuousContinueV14` 所在）：**未改**，v1.4 upstream 原样。
- **scripts/start-comfyui-for-minimax-h3.sh**：`RAY_memory_usage_threshold=1.0` / `RAY_memory_monitor_refresh_ms=2000` / **`--lowvram`**（主进程 VAE 段间自动卸载，替代会驱逐 worker 的段间 `/free`，见 chain_director_v2.md）。

## 目录约定

```
~/MiniMax-H3-Deploy/
├── ComfyUI/                     # ComfyUI master 主仓库（git clean）
│   └── custom_nodes/
│       ├── raylight/            # 双卡框架（被改：nodes.py/ray_worker.py/xdit_context_parallel.py）
│       ├── comfyui_h3_multigpu_clip/   # 多卡 Qwen CLIP（被改：cond 缓存）
│       ├── Herrgotts-H3-Infinite-Continuation-Suite/  # masked-AV 续接（未改）
│       └── ComfyUI-H3-Motion-Context/  # 对照用（未改）
├── vendor/SageAttention2_Optimized_Test/   # SM75 Sage 源码（编译产物 dist/*.whl）
├── cond_cache/                  # CLIP cond 磁盘缓存（可 rm -rf）
├── output/                      # 产物：video/chain/<tag>/、h3_continuous/chain_*.safetensors、final_<tag>.mp4
├── scripts/                     # 各阶段 CLI + start-comfyui-for-minimax-h3.sh/stop.sh + workflows/
├── docs/                        # 本文档集
├── comfyenv/                    # ComfyUI venv
└── DUAL_TP_NOTES.md 等旧文档    # 内容已并入 docs/（归档见本地 docs/_archive_20260905/）
```
