# MiniMax-H3 Deploy 文档索引

MiniMax-H3 双 RTX 2080 Ti 22G（NVLink）raylight TP 部署的工作节点文档。
运行目录即本仓库 `~/MiniMax-H3-Deploy/`（脚本在 `scripts/`，文档在 `docs/`）——单目录，无同步。

## 工作节点演进（开发顺序）

| # | 文档 | 脚本 | 阶段 | 状态 |
|---|---|---|---|---|
| 1 | [gen.md](gen.md) | ~~`gen.py`~~ | 单机/单卡，官方 `MiniMaxH3ImageToVideo` workflow 驱动，无续接 | 历史（脚本与能力均已退役） |
| 2 | [gen_dual.md](gen_dual.md) | ~~`gen_dual.py`~~ | 双卡 raylight FSDP/Ulysses 单任务（含部署/版本锁/性能/CLIP offload 归档） | 历史（能力移交 v1/v2/v3） |
| 3 | [chain_director_v1.md](chain_director_v1.md) | `chain_director_v1.py` | Herrgotts masked-AV 多段续接链 v1（逐段 queue，每段重载） | 自 2026-09-05 不再维护 |
| 4 | [chain_director_v2.md](chain_director_v2.md) | `chain_director_v2.py` | persist 模式链（FSDP 段间驻留）+ 首段素材分流（ref2va 参考图/视频/音频） | 维护中（ref2va 走这里） |
| 5 | [chain_director_v2.md](chain_director_v2.md) | `chain_director_v2_web.py` | 局域网 Web 控制台（`:8189`，stdlib-only）：参数镜像 + 上传 + 串行队列 + 续拍 + 在线预览 | 现行（v2/ref2va） |
| 6 | [chain_director_v3.md](chain_director_v3.md) | `chain_director_v3.py` | 常驻 UNet 链：段间零 FSDP 重载、服务默认常驻、int4 CLIP 按需上下卡、首段 fl2va | **现行主线** |
| 7 | [chain_director_v3.md](chain_director_v3.md) | `chain_director_v3_web.py` | 局域网 Web 控制台（`:8190`，独立数据目录 `.h3web_v3/`），驱动 v3 | 现行 |
| — | [audio.md](audio.md) | —（横切所有阶段） | H3 音频专项：表示与通路、段间 audio 连接、`--ref-audio`/`cond_audio` 引导、**首段底噪现象调查**（steps 收敛结论） | 现行 |

## 快速入口

- **当前推荐做法**：`chain_director_v3.py`（常驻 UNet 链），见 [chain_director_v3.md](chain_director_v3.md)；需要 ref2va 参考素材时用 `chain_director_v2.py`。
- **音频踩坑/底噪结论**：见 [audio.md](audio.md)（立体声 2ch/32kHz、8 步弱场景易出噪声态、弱音频 prompt 用 `--steps 20`）。
- **双卡部署版本锁与踩坑**（ComfyUI `30bdda1`、xfuser 0.4.5、NCCL cu13、SM75 Sage 编译法、内存三律）：并入 [gen_dual.md](gen_dual.md)。

## 对上游代码的改动总览（详见各文档对应节）

- **ComfyUI 主仓库**：`git` 干净（master `30bdda1`），H3 native AV-mask/audio_vae 靠 upstream 自带，**未改**。
- **custom_nodes/raylight**（fork `komikndr/raylight` @`903f50e`）：双卡部署适配 + persist 复用 + xdit 补丁，见 gen_dual.md（FinalLayer 三参等）/ chain_director_v2.md（reuse_epoch）/ audio.md（cond_audio/denoise mask）。`nodes.py`/`ray_worker.py` 留有 `.bak_pre_persist` 原版备份。
- **custom_nodes/comfyui_h3_multigpu_clip**：`encode_model_parallel` 内嵌 cond 磁盘缓存（`cond_cache/`），见 chain_director_v1.md。
- **custom_nodes/Herrgotts-H3-Infinite-Continuation-Suite**（`H3ContinuousStartV14`/`H3ContinuousContinueV14` 所在）：**未改**，v1.4 upstream 原样。
- **scripts/start-comfyui-for-minimax-h3.sh**：`RAY_memory_usage_threshold=1.0` / `RAY_memory_monitor_refresh_ms=2000` / **`--lowvram`**（主进程 VAE 段间自动卸载，替代会驱逐 worker 的段间 `/free`，见 chain_director_v2.md）。

## 目录约定

运行目录 `~/MiniMax-H3-Deploy/` 就是本仓库；ComfyUI 引擎在 `~/ComfyUI-Deploy`，两者通过软链共享节点与工作流。

```
~/MiniMax-H3-Deploy/            # 运行目录 = git 工作树（origin: Aiakos1818/minimax-h3-deploy）
├── nodes/                      # ← ComfyUI custom_nodes/{comfyui_h3_multigpu_clip,h3_vae_unload} 软链指向这里
├── workflows/                  # ← ComfyUI user/default/workflows 软链指向这里（浏览器工作流列表）
│   └── api/                    # API 模板（gen*.py 用，浏览器默认不列出）
├── scripts/                    # 各阶段 CLI + web 控制台 + start-comfyui-for-minimax-h3.sh
├── docs/                       # 本文档集
├── vendor/SageAttention2_Optimized_Test/   # SM75 Sage 源码（编译产物 dist/*.whl）
├── cond_cache/                 # CLIP cond 磁盘缓存（可 rm -rf）
├── output/                     # 产物：video/chain/<tag>/、h3_continuous/chain_*.safetensors、final_<tag>.mp4
├── .h3web/ .h3web_v3/          # 两个 web 控制台的数据目录
└── comfyenv 在 ComfyUI 侧       # ~/ComfyUI-Deploy/comfyenv

~/ComfyUI-Deploy/               # ComfyUI 引擎（独立仓库，分支 h3-sm75-deploy）
├── custom_nodes/               # raylight / Herrgotts / Motion-Context（真实目录）；自研两节点为软链
└── user/default/workflows ->   # 软链到 ~/MiniMax-H3-Deploy/workflows
```
