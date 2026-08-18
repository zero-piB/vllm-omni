#!/bin/bash
# Seed-TTS zh 全量（WER 转写用 NPU 加速；在 Seed-TTS en 完成、服务空闲时运行）
set -euo pipefail
source "$(dirname "$0")/env.sh"
nohup env SEED_TTS_HF_WHISPER_MODEL=${SEED_TTS_HF_WHISPER_MODEL:-"$WHISPER_DIR"} \
  SEED_TTS_EVAL_DEVICE=npu:0 \
  SEED_TTS_WER_SAVE_AUDIO_DIR=${SEED_TTS_WER_SAVE_AUDIO_DIR:-"$DATA_DIR/seed_tts_wavs_zh"} \
  "$VLLM" bench serve --omni --port 8091 --trust-remote-code \
  --max-concurrency 4 --num-warmup 3 \
  --dataset-name seed-tts --dataset-path "$SEED_TTS_DATA" \
  --seed-tts-locale zh --num-prompts 2020 --no-oversample \
  --seed-tts-wer-eval --seed-tts-wer-save-items \
  --model "$SERVED" \
  --tokenizer "$MODEL_DIR" \
  --endpoint /v1/chat/completions --backend openai-chat-omni \
  --percentile-metrics ttft,tpot,itl,e2el,audio_ttfp,audio_rtf \
  --extra_body '{"modalities": ["text", "audio"], "chat_template_kwargs": {"enable_thinking": false, "use_tts_template": true}}' \
  --save-result --result-dir "$RESULTS/seed_tts_zh" \
  > "$RESULTS/seed_tts_zh_full.log" 2>&1 &
echo "Seed-TTS zh 全量已启动 PID $! → $RESULTS/seed_tts_zh_full.log"
