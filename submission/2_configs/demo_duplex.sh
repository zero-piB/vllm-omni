#!/bin/bash
# 全双工 Demo（WebSocket 实时语音，16k PCM 输入 + 参考音色）
# 用法: ./demo_duplex.sh [输入音频路径]    （默认 /tmp/duplex_input_16k.wav；需先切 duplex yaml 服务）
set -euo pipefail
source "$(dirname "$0")/env.sh"
INPUT=${1:-/tmp/duplex_input_16k.wav}
REF="$MODEL_DIR/assets/HT_ref_audio.wav"
OUT=${2:-"$RESULTS/demo/duplex"}

if [ ! -f "$INPUT" ]; then
  echo "输入音频不存在: $INPUT（需 16k 单声道 PCM16）" >&2
  echo "生成示例: python3 -c \"import soundfile as sf,numpy as np; w,sr=sf.read('assets/wav',dtype='float32'); sf.write('$INPUT',w,16000,subtype='PCM_16')\"" >&2
  exit 1
fi
"$PYTHON" \
  "$CODE_DIR/realtime_duplex_demo.py" \
  --url "ws://127.0.0.1:8091/v1/realtime?duplex=1" \
  --model openbmb/MiniCPM-o-4_5 \
  --input-wav "$INPUT" --ref-audio "$REF" \
  --output-dir "$OUT" --timeout-s 90
echo "结果: $OUT/result.json（ok 字段 + transcript + 音频分片）"
