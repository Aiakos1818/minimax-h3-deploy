#!/usr/bin/env bash
pkill -f "main.py --listen 0.0.0.0 --port 8188" && echo stopped || echo "not running"
