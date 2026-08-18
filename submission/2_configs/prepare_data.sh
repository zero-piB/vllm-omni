#!/bin/bash
# 准备评测数据：从 RAW_DATA_DIR 原始数据源生成 7_runtime/media/ 下的转换产物（幂等，已存在则跳过）
# 用法: ./prepare_data.sh [--force]   （--force 强制重新生成）
set -euo pipefail
source "$(dirname "$0")/env.sh"
FORCE=0
[ "${1:-}" = "--force" ] && FORCE=1

ok() { echo "[prepare] ✅ $1"; }
skip() { echo "[prepare] ⏭ 已存在，跳过: $1"; }
warn() { echo "[prepare] ⚠️ $1"; }

mkdir -p "$DATA_DIR"

# 1. Seed-TTS 数据集（tar → seedtts_testset/）
if [ -f "$SEED_TTS_DATA/meta.lst" ] && [ "$FORCE" -eq 0 ]; then
  skip "Seed-TTS ($SEED_TTS_DATA)"
elif [ -f "$SEED_TTS_TAR" ]; then
  echo "[prepare] 解压 Seed-TTS..."; mkdir -p "$DATA_DIR/seed-tts"
  tar -xf "$SEED_TTS_TAR" -C "$DATA_DIR/seed-tts/" && ok "Seed-TTS"
else
  warn "未找到 $SEED_TTS_TAR（env.sh 的 SEED_TTS_TAR），跳过 Seed-TTS"
fi

# 2. Daily-Omni（parquet → qa.json + Videos/）
if [ -f "$DAILY_OMNI_QA" ] && [ "$FORCE" -eq 0 ]; then
  skip "Daily-Omni ($DAILY_OMNI_QA)"
elif [ -d "$DAILY_OMNI_RAW" ]; then
  echo "[prepare] 转换 Daily-Omni...（1196 条 QA + 视频，约 30-60 分钟）"
  "$PYTHON" "$CODE_DIR/convert_daily_omni_modelscope.py" \
    --src "$DAILY_OMNI_RAW" --dst "$DATA_DIR/daily_omni" && ok "Daily-Omni"
else
  warn "未找到 $DAILY_OMNI_RAW（env.sh 的 DAILY_OMNI_RAW），跳过 Daily-Omni"
fi

# 3. VideoMME 视频（按需从 zip 解压所需 ~900 个 mp4）
if [ -n "$(ls -A "$VIDEOMME_VIDEOS" 2>/dev/null)" ] && [ "$FORCE" -eq 0 ]; then
  skip "VideoMME 视频 ($VIDEOMME_VIDEOS)"
elif [ -f "$VIDEOMME_PARQUET" ]; then
  echo "[prepare] 解压 VideoMME 视频（按标注所需 ~900 个 mp4，约 20-60 分钟）"
  "$PYTHON" "$CODE_DIR/extract_videomme_videos.py" \
    --parquet "$VIDEOMME_PARQUET" --zips-dir "$VIDEOMME_ZIPS_DIR" \
    --out-dir "$VIDEOMME_VIDEOS" && ok "VideoMME 视频"
else
  warn "未找到 $VIDEOMME_PARQUET（env.sh 的 VIDEOMME_PARQUET），跳过 VideoMME"
fi

# 4. WER 转写模型 whisper-large-v3
if [ -d "$WHISPER_DIR/config.json" ] || [ -f "$WHISPER_DIR/config.json" ] && [ "$FORCE" -eq 0 ]; then
  skip "whisper-large-v3 ($WHISPER_DIR)"
elif [ -n "$WHISPER_SRC" ] && [ -d "$WHISPER_SRC" ]; then
  echo "[prepare] 拷贝 whisper-large-v3..."
  mkdir -p "$DATA_DIR/models"
  cp -r "$WHISPER_SRC" "$WHISPER_DIR" && ok "whisper-large-v3"
else
  warn "whisper-large-v3 缺失：设 WHISPER_SRC 指向已有目录，或设置后重跑；也可离线下载后放 $WHISPER_DIR"
fi

echo ""
echo "[prepare] 完成。数据目录: $DATA_DIR"
echo "缺失项见上方 ⚠️ 提示；如需强制重生成: ./prepare_data.sh --force"
