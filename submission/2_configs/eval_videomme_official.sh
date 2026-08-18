#!/bin/bash
# VideoMME 精度评测（官方 videomme dataset 模块，minicpm-frames 96 帧协议；准入 ≥67.0）
# 前置：服务用默认 yaml（image≥96）运行；数据：--videomme-video-dir 需有 mp4 且被 allowlist 覆盖
# 用法: ./eval_videomme_official.sh [条数]   （默认全量 2700；并发 4 同官方）
set -euo pipefail
source "$(dirname "$0")/env.sh"
N=${1:-2700}
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1
LOG="$RESULTS/videomme_official_$(date +%H%M%S).log"

"$VLLM" bench serve --omni --port 8091 --max-concurrency 4 \
  --dataset-name videomme --dataset-path /workspace/shared_assets/datasets/lmms-lab/Video-MME \
  --num-prompts "$N" --trust-remote-code --no-oversample --disable-shuffle \
  --temperature 0 --output-len 128 \
  --videomme-pack-mode minicpm-frames --videomme-max-frames 96 --videomme-duration all \
  --videomme-parquet "$VIDEOMME_PARQUET" \
  --videomme-video-dir "$VIDEOMME_VIDEOS" \
  --videomme-subtitle-dir "$DATA_DIR/videomme_subtitles" \
  --model "$SERVED" --tokenizer "$MODEL_DIR" \
  --endpoint /v1/chat/completions --backend openai-chat-omni \
  --percentile-metrics ttft,tpot,itl,e2el \
  --extra_body '{"modalities": ["text"], "chat_template_kwargs": {"enable_thinking": false}}' \
  2>&1 | tee "$LOG"

ACC=$(grep -oE "Overall Accuracy: [0-9]+/[0-9]+ = [0-9.]+%" "$LOG" | tail -1 | grep -oE '[0-9.]+%$' | head -1)
echo "---"
echo "VideoMME Accuracy = ${ACC:-?}  （准入 ≥67.0，官方 minicpm-frames/96帧 协议）"
[ -n "${ACC:-}" ] && awk -v a="${ACC%\%}" 'BEGIN { print (a+0 >= 67.0) ? "判定: ✅ 达标" : "判定: ❌ 不达标" }'
