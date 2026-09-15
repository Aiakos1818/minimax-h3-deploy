#!/usr/bin/env bash
COMFY_HOME="$HOME/ComfyUI-Deploy"
cd "$COMFY_HOME"
# Let the H3 text encoder keep references to its CPU weight masters so
# offload_after_encode can rebind them instead of re-reading the checkpoint.
# Needed by first/last-frame workflows, where the video VAE and the text
# encoder cannot both stay in VRAM during conditioning.
export H3_MP_RETAIN_CPU_WEIGHTS=1
setsid nohup "$COMFY_HOME/comfyenv/bin/python" main.py --listen 0.0.0.0 --port 8188 > "$COMFY_HOME/comfy.log" 2>&1 < /dev/null &
echo "ComfyUI (dual-GPU config) starting on :8188 (log: $COMFY_HOME/comfy.log)"
