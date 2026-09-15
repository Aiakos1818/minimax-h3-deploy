# 工作节点 1：gen.py（单机单卡官方图驱动）

> 阶段：MiniMax-H3 部署最早的单任务生成器。直接复用官方 ComfyUI workflow
> （`MiniMaxH3ImageToVideo`），单机/单卡跑通 t2v/i2v。历史角色，workflow
> 模板 `api_local_*.json` 已随清理删除，本文件为用法与结论归档。

## 职责

- 向本机 ComfyUI `:8188` 提交**官方 `MiniMaxH3ImageToVideo`** 工作流：
  - `t2v`：纯文本 → `api_local_t2v.json`
  - `i2v`：文本 + 首帧图 → `api_local_i2v.json`（`LoadImage` → `first_frame`）
- 无 raylight / 无多卡：早期单卡（含 SM75 Sage `--use-sage-attention`）路径。
- workflow 含 `RandomNoise`（seed 控制）、`PrimitiveBoolean`（`--turbo` 切 turbo LoRA 8 步）、
  `PrimitiveFloat`（`--seconds`，snap 到 17k+5）。

## 用法

```bash
python scripts/gen.py --prompt "..." --mode t2v --wait
python scripts/gen.py --prompt "..." --mode i2v --image input/xxx.png --wait
python scripts/gen.py --prompt "..." --width 864 --height 480 --length 124 --seed N --wait
python scripts/gen.py --prompt "..." --seconds 5 --turbo --wait   # turbo LoRA 8 步
```

- `--image` 传服务端路径则先 multipart 上传，或直接给已在 `ComfyUI/input` 的文件名。
- 产物：`output/video/*.mp4`（`--wait` 打印实际落盘路径）。

## 单卡性能基线（同条件：864×480 / 124 帧 / 20 步 / INT8）

| 形态 | 采样步速 | 采样耗时 | 备注 |
|---|---|---|---|
| 单卡无 Sage（早期基线） | ~60 s/it | ~20 min | 纯 PyTorch attention |
| 单卡 + SM75 Sage | 14.3 s/it | 4:46 | Sage kernel 生效（`--use-sage-attention` 全局开启） |

单卡基线只到 ~5min/任务量级；多段连续、更长画面由后续节点承担（见 README 演进表）。

## 局限（本节点退役原因）

- 单段、无续接、每任务全量重跑（编码 + 装载 + 采样 + 解码）；
- 采样是大头（吃分辨率×帧数×步数×算力），单卡算力受限；
- 之后 gen_dual（双卡 TP）与 chain_director v1/v2（多段续接）逐步取代。
