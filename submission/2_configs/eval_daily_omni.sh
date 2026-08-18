#!/bin/bash
# Daily-Omni 评测（官方 vllm bench，准入 ≥77.5；bench yaml: repetition_penalty 1.2）
# 用法: ./eval_daily_omni.sh [条数] [restart-server]
#   条数默认 32；全量 1197 传 1197；传 restart-server 自动切 bench 服务（否则需先切好）
#   调试: DAILY_OMNI_INLINE=1 时视频 base64 内嵌请求（无需 allowlist），仅小样本用
set -euo pipefail
source "$(dirname "$0")/env.sh"
N=${1:-32}
if [ "${2:-}" = "restart-server" ] || [ "${2:-}" = "restart_server" ]; then
  "$(dirname "$0")/server_restart.sh" minicpmo_4_5_bench.yaml
fi
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1
# 调试开关：DAILY_OMNI_INLINE=1 时视频 base64 内嵌请求（无需服务端 allowlist），仅小样本调试用
INLINE_ARGS=()
[ -n "${DAILY_OMNI_INLINE:-}" ] && INLINE_ARGS+=(--daily-omni-inline-local-video)
LOG="$RESULTS/daily_omni_$(date +%H%M%S).log"

"$VLLM" bench serve --omni --port 8091 --max-concurrency 10 \
  --dataset-name daily-omni --num-prompts "$N" --trust-remote-code --no-oversample \
  --temperature 0 --output-len 128 \
  --daily-omni-input-mode all --daily-omni-pack-mode minicpm-interleave \
  "${INLINE_ARGS[@]}" \
  --daily-omni-video-dir "$DAILY_OMNI_VIDEOS" \
  --daily-omni-qa-json "$DAILY_OMNI_QA" \
  --model "$SERVED" --tokenizer "$MODEL_DIR" \
  --endpoint /v1/chat/completions --backend openai-chat-omni \
  --percentile-metrics ttft,tpot,itl,e2el \
  --save-result --result-dir "$RESULTS/daily_omni" \
  --extra_body '{"modalities": ["text"], "chat_template_kwargs": {"enable_thinking": false}}' \
  2>&1 | tee "$LOG"

ACC=$(grep -E "Overall Accuracy" "$LOG" | grep -oE '[0-9.]+%' | head -1)
echo "---"
echo "Daily-Omni Accuracy = $ACC  （准入 ≥77.5）"
awk -v a="${ACC%\%}" 'BEGIN { print (a+0 >= 77.5) ? "判定: ✅ 达标" : "判定: ❌ 不达标" }'
