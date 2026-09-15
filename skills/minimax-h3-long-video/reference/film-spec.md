# film_spec.json 字段说明

一部片一个 spec，人审 + 脚本（`anchors.py` / `film.py`）都读它。可直接复制
`scripts/film_spec.example.json`（pingfan_01 的真实数据）改。

```jsonc
{
  "film": "pingfan",                                  // 片子标识：输出、锚图、项目目录都用它
  "root": "~/MiniMax-H3-Deploy",                   // 管线仓库根
  "python": "~/ComfyUI-Deploy/comfyenv/bin/python",// 带 pyav 的解释器（宿主机没有 ffmpeg）
  "comfy": "http://127.0.0.1:8188",                // ComfyUI
  "console": "http://127.0.0.1:8190",              // web 控制台（预览 /files/<film>/anchors.html）

  "params": { "width": 864, "height": 480, "dur": 5, "steps": 8 },

  "looks":      { "present": "<LOOK 块>", "memory": "<LOOK 块>" },
  "characters": { "man50": "<CHARACTER 块>", "young": "..." },
  "negative":   "<NEGATIVE 固定尾巴>",

  "anchors": {
    "s1": { "prompt": "<Z-Image 静帧提示词>", "size": [1024, 576], "steps": 8,
            "seeds": { "a": 111111, "b": 222222, "c": 333333 } }
  },

  "jobs": [
    { "tag": "pingfan_balcony", "kind": "shot",  "segments": 1, "anchor": "balcony", "seed": 31001001,
      "prompt": "<完整提示词：LOOK + SHOT + CHARACTER + AUDIO + NEGATIVE>" ,
      "look": "present", "character": "man50", "shot": "<SHOT 文本>", "audio": "AUDIO: ..." },
    { "tag": "pingfan_ride", "kind": "chain", "segments": 3, "anchor": "ride", "seed": 31004004,
      "prompt": "<链基底提示词>",
      "beats": ["<第 2 段动作正文>", "<第 3 段动作正文>"] }
  ],

  "assemble": {
    "out": "output/final_pingfan.mp4", "aspect": 2.39,
    "fade_in": 0.5, "fade_out": 1.0, "hold": 0.6, "fps": 24,
    "clips": ["shot:pingfan_balcony", "shot:pingfan_agespot", "shot:pingfan_screen",
              "chain:pingfan_ride", "chain:pingfan_gate"]
  }
}
```

## jobs

| 字段 | 说明 |
|---|---|
| `tag` | 任务标识 → `output/video/chain/<tag>/seg_*.mp4`；链的合并成 `output/final_<tag>.mp4`。**命名：`<film>_<shot-slug>`，用内容 slug（`pingfan_balcony`），不要序号（`pf01s1`）** |
| `kind` | `shot`（独立）／`chain`（续接链）；只作标注，实际由 `segments` 决定 |
| `segments` | 1 = 独立镜头；>1 = 链（自动加 `--merge`） |
| `anchor` | 锚图 key → `projects/<film>/anchors/<key>_best.png`；缺文件时脚本警告并降级为纯文本 |
| `seed` | 固定 seed，便于重跑对比 |
| `prompt` | **完整提示词**（推荐：作者自己拼好，所见即所得） |
| `look`/`character`/`shot`/`audio` | 可选：不写 `prompt` 时由脚本按 `look + shot + character + audio + negative` 拼 |
| `beats` | 链的第 2..N 段动作；脚本生成 `--beat "<dur*k>s:<正文>"`（k=1,2,…） |

## assemble

| 字段 | 说明 |
|---|---|
| `out` | 相对 `<root>` 的成片路径（放 `output/` 下，控制台才能预览） |
| `aspect` | 居中裁幅宽高比（`2.39` → 864×480 裁成 864×362；`0` = 不裁） |
| `fade_in`/`fade_out`/`hold` | 秒；`hold` = 结尾末帧定格，落在淡出区间内 |
| `clips` | 拼接顺序：`shot:<tag>`（取 `seg_0_*.mp4`）／`chain:<tag>`（取 `final_<tag>.mp4`）／直接给相对路径 |

## 命令

```bash
S=~/.config/opencode/skill/minimax-h3-long-video/scripts
python3 $S/anchors.py generate --spec MINE.json [--only s1] [--dry-run]
python3 $S/anchors.py contact  --spec MINE.json
python3 $S/anchors.py pick s1=b s2=a --spec MINE.json
python3 $S/film.py run      --spec MINE.json [--only pingfan_ride] [--dry-run]
python3 $S/film.py assemble --spec MINE.json [--dry-run]
python3 $S/film.py verify   --spec MINE.json
```

`--dry-run` 只打印将要执行的内容，不碰 GPU；建议第一次跑任何一部片都先 dry-run 一遍。
