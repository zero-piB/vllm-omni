#!/bin/bash
# TTS-Seed WER 评测（官方 vllm bench，准入 ≤1.56）
# 用法: ./eval_seed_tts_wer.sh [条数]   （默认 32）
set -euo pipefail
source "$(dirname "$0")/env.sh"
N=${1:-32}
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1
export SEED_TTS_HF_WHISPER_MODEL="${SEED_TTS_HF_WHISPER_MODEL:-$WHISPER_DIR}"
LOG="$RESULTS/seed_tts_wer_$(date +%H%M%S).log"

SEED_TTS_EVAL_DEVICE=npu:0 \
  "$VLLM" bench serve --omni --port 8091 --max-concurrency 4 --num-warmup 3 \
  --dataset-name seed-tts --dataset-path "$SEED_TTS_DATA" \
  --seed-tts-locale ${LOCALE:-zh} --num-prompts "$N" --no-oversample \
  --seed-tts-wer-eval --seed-tts-wer-save-items --trust-remote-code \
  --percentile-metrics ttft,tpot,itl,e2el,audio_ttfp,audio_rtf \
  --model "$SERVED" --tokenizer "$MODEL_DIR" \
  --endpoint /v1/chat/completions --backend openai-chat-omni \
  --extra_body '{"modalities": ["text", "audio"], "chat_template_kwargs": {"enable_thinking": false, "use_tts_template": true}}' \
  2>&1 | tee "$LOG"

WER=$(grep -oE "Mean WER: +[0-9.]+" "$LOG" | grep -oE '[0-9.]+$' | head -1)
echo "---"
echo "TTS-Seed WER = $WER  （准入 ≤1.56）"
awk -v w="$WER" 'BEGIN { print (w+0 <= 1.56) ? "判定: ✅ 达标" : "判定: ❌ 超标" }'
