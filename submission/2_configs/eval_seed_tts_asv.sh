#!/bin/bash
# TTS-Seed ASV/SIM 评测（官方 wavlm-base-plus 代理协议，准入 ≥0.689）
# 用法: ./eval_seed_tts_asv.sh [条数]   （默认 32）
set -euo pipefail
source "$(dirname "$0")/env.sh"
N=${1:-32}
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1
LOG="$RESULTS/seed_tts_asv_$(date +%H%M%S).log"

SEED_TTS_SIM_EVAL=1 SEED_TTS_EVAL_DEVICE=npu:0 \
  "$VLLM" bench serve --omni --port 8091 --max-concurrency 4 --num-warmup 3 \
  --dataset-name seed-tts --dataset-path "$SEED_TTS_DATA" \
  --seed-tts-locale en --num-prompts "$N" --no-oversample \
  --seed-tts-wer-eval --seed-tts-wer-save-items --trust-remote-code \
  --model "$SERVED" --tokenizer "$MODEL_DIR" \
  --endpoint /v1/chat/completions --backend openai-chat-omni \
  --extra_body '{"modalities": ["text", "audio"], "chat_template_kwargs": {"enable_thinking": false, "use_tts_template": true}}' \
  2>&1 | tee "$LOG"

SIM=$(grep -oE "Mean SIM: +[0-9.]+" "$LOG" | grep -oE '[0-9.]+$' | head -1)
echo "---"
echo "TTS-Seed SIM/ASV = $SIM  （准入 ≥0.689）"
awk -v s="$SIM" 'BEGIN { print (s+0 >= 0.689) ? "判定: ✅ 达标" : "判定: ❌ 不达标" }'
