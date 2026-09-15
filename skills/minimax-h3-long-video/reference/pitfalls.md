# 坑与验证速查（都来自 pingfan_01 的实战）

## 生产流程

1. **每个任务前必须 `--clean`**：slot 文件名是 `output/h3_continuous/chain_0000N.safetensors`，**全局不分
   tag**；残留会让 `count_existing()` 认为"第 N 段已完成"，直接续拍上一条链甚至产物张冠李戴。
   （独立镜头之间的 slot 也会互相覆盖/删除，这是设计如此，不是 bug。）
2. **闪回/换时间线不要塞进一条链**：续接节点强制画面连续，跳跃会被抹成缓慢渐变。另开链 + 后期硬切。
3. **锚图必须在链开跑前全部出完**：常驻 FSDP 占 ~12G/卡，Z-Image Turbo 挤不进去。
   ComfyUI 是 `--lowvram`，换模型时会互相卸载，硬抢会 OOM 或极慢。
4. **`--dur` 是净新秒数**：段 2..N 的 raw 帧 ≈ `dur*24 + 39`（`HEAD=39` 保护上下文，合并时裁掉）；
   `--beat` 的秒数归属 `seg = int(t // dur)`，所以第 2 段写 `5s:`、第 3 段写 `10s:`（dur=5）。
5. **`/interrupt` 打断后**主进程已装载的视频/音频 VAE 残留 ~2–4G，紧接着提交可能在第 1 步 OOM →
   `~/ComfyUI-Deploy/stop.sh` 重启服务（走 chain_director 的正常路径不受影响，它会显式腾挪）。
6. **单段上限 ~226 帧 @864×480**（峰值 21.4G/卡）；s/it 随帧数上升（107f 6.6 → 226f 19.5）。dur=5 最稳。
7. **常驻服务**：跑完默认不释放（每卡 ~11.9G）。`--stop-when-done` 或 stop.sh 才释放。第一节任务付冷启动
   （服务 ~15s + ray 重建 + FSDP 装载 ≈ 3min），之后的同服务实例都免。
8. **本机内存 15GB**：换模型时会 swap 抖动，表现为 GPU 0%/几百 MB、日志停在 "Requested to load ..."
   2–4 分钟——这是装载中，不是死锁，别急着杀。锚图 15 张实测花了 ~13min。

## 介质与组装

9. **宿主机没有 ffmpeg** → 一切用 `~/ComfyUI-Deploy/comfyenv/bin/python`（pyav）。
10. **pyav 里先建好所有流再编码**：在另一个流 flush 之后才 `out.add_stream(...)`，muxer 会
    `Floating point exception (core dumped)` 崩在 `out.mux(pkt)`（本次真实踩过）。`assemble.py` 已按此写。
11. **裁幅数学**：864×480 → 2.39:1 = **864×362**（居中裁，高取偶数）。
12. **`/files/` 只能服务 `output/` 下的文件**（path containment）；锚图联系表要写在
    `output/<film>/anchors.html`，图也放 `output/<film>/`。
13. **`ui2api.py` 不支持 subgraph**（会明确拒绝）；Z-Image 工作流是 subgraph 结构，其
    `workflows/api/api_image_z_image_turbo.json` 是手工转换并对照 `/object_info` 校验的。

## 监控（driver 的 stdout 是块缓冲）

- 跑任务用 **`python -u`**（`film.py run` 已加），否则 driver 的 `[clip1] ...` 要等缓冲满才出现。
- 实时进度看 ComfyUI 日志的 tqdm；判断"哪个任务在跑"看 `/queue` 里 `queue_running[*][3].client_id`
  （driver 的 client_id 形如 `cd3-<tag>-<rand>`）；进度里程碑看 `/history` 条目数、`output/video/chain/<tag>/`
  文件、`output/h3_continuous/` 槽位、`nvidia-smi`。
- 合成任务大文件（slot safetensors 几 MB～几 GB）落盘时，mp4 可能短暂不可读——别急着判定失败。

## 验证命令

```bash
S=~/.config/opencode/skill/minimax-h3-long-video/scripts
python3 $S/film.py verify --spec MINE.json          # 帧数/时长/尺寸/音频/淡入淡出/切点

# 单独看某个文件
~/ComfyUI-Deploy/comfyenv/bin/python - <<'PY'
import av
c=av.open("output/final_pingfan.mp4"); v=next(s for s in c.streams if s.type=="video")
print(v.codec_context.width, v.codec_context.height, sum(1 for _ in c.decode(video=0)))
PY

# 服务/显存
curl -s http://127.0.0.1:8188/system_stats | head -c 300
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
~/ComfyUI-Deploy/stop.sh          # 立刻释放常驻显存
```

预期：成片 `w=864 h=362`，帧数 = 各段帧数之和 + `hold*24`，音频 32kHz 且时长≈视频，首/末帧亮度≈0，
切点帧号落在各段边界（pingfan：124/248/372/697 + 链内 477/1013）。
