#!/bin/bash
# 半双工 Demo：启动 gradio + 自动验证（流式 TTS 连续性，落盘音频分片）
# 用法: ./demo_half_duplex.sh    （需默认 yaml 服务运行中）
set -euo pipefail
source "$(dirname "$0")/env.sh"
OUT=${1:-"$RESULTS/demo"}
"$PYTHON" "$CODE_DIR/verify_demo.py" --base http://127.0.0.1:8091 --out "$OUT"
echo "结果: $OUT/result.json"
