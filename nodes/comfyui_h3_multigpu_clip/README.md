# MiniMax H3 多卡文本编码器加载节点

节点名称：`MiniMax H3 Multi-GPU CLIP Loader`

它将 H3 使用的 Qwen3-VL-32B 文本编码器按连续 decoder layers 分配到多张GPU，并把层间 activation 自动移动到下一张卡。`gpu_ids` 接受逗号分隔的可见 GPU，例如 `0,1,2,3`。

建议设置：

```bash
export H3_MP_RETAIN_CPU_WEIGHTS=1
```

该设置保留 CPU 参数主存储，编码后直接重新绑定 CPU tensors，使节点能快速归还 GPU 显存。

同一 INT8 Qwen 和输入条件下，ComfyUI 默认 offload 的观测数据为：原生单卡DynamicVRAM 路径 21.43 秒；默认加载节点置于四卡 Raylight 完整工作流时冷运行 37.09 秒。该四卡节点冷运行 11.04 秒、暖运行约 9.17–9.25 秒。相对同类四卡完整工作流的默认加载路径，冷运行由 37.09 秒降至 11.04 秒，约为 3.36× 吞吐。

另行完成的节点内部数值回归表明，启用 CPU 主存储重绑定前后 conditioning逐字节一致；它验证的是 offload 实现变化不会改变输出，不等同于默认节点与自研节点已经做过逐字节 A/B。

将整个目录复制到 `ComfyUI/custom_nodes/`，重启 ComfyUI 后即可使用。
