#!/bin/bash
# 性能测试（Seed-TTS en 小样本：TTFT / TTFP / RTF / E2EL，对比 910C 基线）
# 用法: ./perf_seed_tts.sh [条数] [并发]   （默认 32 / 1）
set -euo pipefail
source "$(dirname "$0")/env.sh"
N=${1:-32}
C=${2:-1}
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1
LOG="$RESULTS/perf_seed_tts.log"

"$VLLM" bench serve --omni --port 8091 --max-concurrency "$C" --num-warmup 3 \
  --dataset-name seed-tts --dataset-path "$SEED_TTS_DATA" \
  --seed-tts-locale en --num-prompts "$N" --no-oversample \
  --seed-tts-wer-save-items --trust-remote-code \
  --percentile-metrics ttft,tpot,itl,e2el,audio_ttfp,audio_rtf \
  --model "$SERVED" --tokenizer "$MODEL_DIR" \
  --endpoint /v1/chat/completions --backend openai-chat-omni \
  --extra_body '{"modalities": ["text", "audio"], "chat_template_kwargs": {"enable_thinking": false, "use_tts_template": true}}' \
  2>&1 | tee "$LOG"

echo "---"
echo "910B3 实测 vs 910C F16 官方基线："
awk -F: '/Mean TTFT/{v=$2; sub(/ /,"",v); printf "TTFT  %.0f ms  (基线 333.27 ms)\n", v} /Mean AUDIO_TTFP/{v=$2; sub(/ /,"",v); printf "TTFP  %.0f ms  (基线 986.47 ms)\n", v} /Mean AUDIO_RTF/{v=$2; sub(/ /,"",v); printf "RTF   %.2f      (基线 0.4423)\n", v}' "$LOG"
echo "注: 910B 算力约为 910C 的 55-60%，数值供相对量级参考"
