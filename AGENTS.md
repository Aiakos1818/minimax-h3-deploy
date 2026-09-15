# AGENTS.md — MiniMax-H3-Deploy

## 这是什么

本目录**既是运行目录也是 git 工作树**（origin `Aiakos1818/minimax-h3-deploy`）。
ComfyUI 引擎在 `~/ComfyUI-Deploy`，通过软链共享本目录的节点与工作流：

- `~/ComfyUI-Deploy/custom_nodes/{comfyui_h3_multigpu_clip,h3_vae_unload}` → `nodes/*`
- `~/ComfyUI-Deploy/user/default/workflows` → `workflows/`（浏览器里看到的即这一份）
- `~/.config/opencode/skill/minimax-h3-long-video` → `skills/minimax-h3-long-video`（长视频 skill：
  大纲→分镜→锚图→逐段生成→拼成片；**只在本目录这份上改**，全局那个是软链）

节点 / 工作流 / 脚本 / 文档都在本目录直接改，**不需要往别处复制同步**。
产物与缓存（`output/ cond_cache/ downloads/ vendor/ downloadenv/ raylightenv/ .h3web/ .h3web_v3/ .clip_offload_test/ *.log *.bak_*`）已被 `.gitignore` 排除。

## 操作规范

- **提交/推送前必须先征得用户确认**：可以准备好改动并给出 `git status` + `git diff --stat` 摘要，但不要自行 `git commit` / `git push`。
- 提交前先看 `git status --porcelain`，确认没有模型/缓存/大文件混入（忽略规则见 `.gitignore`）。
- 运行入口：`scripts/start-comfyui-for-minimax-h3.sh`（RAY env + `--lowvram --reserve-vram 11.5` + `--use-sage-attention` + `--disable-dynamic-vram --disable-cuda-malloc`）与 `~/ComfyUI-Deploy/stop.sh`；`scripts/start.sh`/`scripts/stop.sh` 与 `~/ComfyUI-Deploy/` 的同名文件保持一致，基本不改。
- 续接链现行主线是 `scripts/chain_director_v3.py`（常驻 UNet、服务默认常驻、CLIP 按需上下卡）；需要 ref2va 参考素材时用 `scripts/chain_director_v2.py`。
- API 模板在 `workflows/api/`（与 `workflows/*.json` 一一对应，供手工 POST/curl；改过 UI 工作流后用 `python scripts/ui2api.py` 重新生成）。浏览器工作流列表只看 `workflows/*.json`（非递归）。
