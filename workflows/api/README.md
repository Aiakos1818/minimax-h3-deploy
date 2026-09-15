# API 格式模板

与上一级 `workflows/*.json`（浏览器用的 UI 工作流）**一一对应**的 API 格式图，供命令行
`POST :8188/prompt` 或脚本使用（浏览器的工作流列表不显示本目录，列表默认非递归）。

| UI（`workflows/`） | API（本目录） | 说明 |
|---|---|---|
| `minimax_h3_int4clip_int8unet_raylight.json` | `api_minimax_h3_int4clip_int8unet_raylight.json` | 现行常驻档（int4 CLIP + int8 UNet + 三处 `UnloadVideoVAE`），单任务最快路径 |
| `video_minimax_h3_t2v.json` | `api_video_minimax_h3_t2v.json` | 官方模板（单卡、turbo LoRA），历史 |
| `video_minimax_h3_i2v.json` | `api_video_minimax_h3_i2v.json` | 同上 i2v |
| `video_minimax_h3_raylight_fl2v.json` | `api_video_minimax_h3_raylight_fl2v.json` | 双卡 raylight 前端版（int8 CLIP + fp16 VAE + `clear=true`） |
| `video_minimax_h3_raylight_ref2v.json` | `api_video_minimax_h3_raylight_ref2v.json` | 双卡 raylight ref2v（参考图/视频/音频） |
| `image_z_image_turbo.json`（本地保留） | `api_image_z_image_turbo.json` | Z-Image-Turbo 文生图（非 H3；出片首帧锚图用） |

`image_z_image_turbo.json` 的 UI 文件是 subgraph 结构，`ui2api.py` 会拒绝转换，因此
`api_image_z_image_turbo.json` 是**手工转换**并对照 `/object_info` 校验过的版本（节点 id
沿用子图内部 id：27=文本、13=宽高、3=seed/steps、9=输出前缀）。

## 用法（curl）

```bash
cd ~/MiniMax-H3-Deploy
WF=workflows/api/api_minimax_h3_int4clip_int8unet_raylight.json

# 文生：改 prompt / 尺寸 / 长度 / seed / 步数（节点 id 与 UI 一致：133=Task, 144=Sampler, 143=Scheduler）
python3 - "$WF" "黄昏天台，青年凭栏远望城市，镜头缓慢环绕推近" > /tmp/h3.json <<'PY'
import json, sys
g = json.load(open(sys.argv[1]))
g["133"]["inputs"].update(prompt=sys.argv[2], width=864, height=480, length=124)  # length 用 17k+5
g["144"]["inputs"]["noise_seed"] = 123456789
g["143"]["inputs"]["steps"] = 20
print(json.dumps({"prompt": g, "client_id": "curl-h3"}))
PY
curl -s -X POST http://127.0.0.1:8188/prompt -H 'Content-Type: application/json' \
     --data-binary @/tmp/h3.json | jq .
# 产物落在 ~/MiniMax-H3-Deploy/output/video/MiniMax_H3_*.mp4（也可从 /history/<prompt_id> 的 outputs["92"].images 读）

# 图生：上传首帧后把 LoadImage(114) 指过去，并给 133 加 first_frame 链接
#   g["114"]["inputs"]["image"] = <upload/image 返回的 name>
#   g["133"]["inputs"]["first_frame"] = ["114", 0]
```

注意：常驻档 `clear_vram_after_sampling=False`，POST 完 FSDP 会留在显存（~11.9G/卡），
要释放用 `~/ComfyUI-Deploy/stop.sh`；第一次提交付冷启动（~2-3 分钟），之后同 prompt 走
cond 缓存、几乎零编码开销。

> 打断注意：`POST /interrupt` 会中止当前提交，但主进程里已经装载的视频/音频 VAE **会留在显存**
> （多占 ~2-4G）；紧接着再提交可能在采样第 1 步 OOM（实测碰到过一次）。打断后建议先
> `~/ComfyUI-Deploy/stop.sh` 再重新启动；走 chain_director 则不受影响（它的 `UnloadVideoVAE`
> 会显式腾挪并清 ray 池）。

## 重新生成

`api_minimax_h3_int4clip_int8unet_raylight` / `api_video_minimax_h3_raylight_fl2v` /
`api_video_minimax_h3_raylight_ref2v` 由 `scripts/ui2api.py` 从对应 UI 文件生成
（需要 ComfyUI 在跑，用 `/object_info` 对齐控件名）：

```bash
python3 scripts/ui2api.py workflows/video_minimax_h3_raylight_fl2v.json \
    -o workflows/api/api_video_minimax_h3_raylight_fl2v.json
```

`api_video_minimax_h3_t2v/i2v.json` 的两个官方模板是 **subgraph** 结构，`ui2api.py`
不支持（会拒绝），保持手工转换的版本。
