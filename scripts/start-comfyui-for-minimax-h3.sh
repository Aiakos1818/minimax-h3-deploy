#!/usr/bin/env bash
COMFY_HOME="$HOME/ComfyUI-Deploy"
MINIMAX_H3_HOME="$HOME/MiniMax-H3-Deploy"
cd "$COMFY_HOME"
export RAY_memory_usage_threshold=1.0
export RAY_memory_monitor_refresh_ms=2000
H3_MP_RETAIN_CPU_WEIGHTS=1 setsid nohup "$COMFY_HOME/comfyenv/bin/python" main.py --listen 0.0.0.0 --port 8188 --output-directory "$MINIMAX_H3_HOME/output" --lowvram --reserve-vram 11.5 --use-sage-attention --disable-dynamic-vram --disable-cuda-malloc > "$COMFY_HOME/comfy.log" 2>&1 < /dev/null &
echo "ComfyUI (dual-GPU config) starting on :8188 (log: $COMFY_HOME/comfy.log)"
