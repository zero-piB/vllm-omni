#!/bin/bash
# 统一环境变量（换机器只改这里：VLLM/PYTHON/MODEL_DIR/REPO_DIR/RAW_DATA_DIR）
# 所有运行脚本 source 本文件。
# 运行时产物（转换后的数据/日志/结果）全部在 $SUBMIT_DIR/7_runtime/ 下——submission 拷贝即用，
# 无需外部 eval 目录；原始数据（数据集源）由 RAW_DATA_DIR 指向，运行前跑 prepare_data.sh 生成。
set -euo pipefail

# submission 根目录（本文件在 2_configs/ 下）
SUBMIT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODE_DIR="$SUBMIT_DIR/1_code"
CONFIG_DIR="$SUBMIT_DIR/2_configs"
RUNTIME_DIR="$SUBMIT_DIR/7_runtime"          # 运行时根（临时文件都在这）
DATA_DIR="$RUNTIME_DIR/media"                # 转换后的评测数据（prepare_data.sh 生成）
RESULTS="$RUNTIME_DIR/results"               # 评测日志/结果

# 环境路径（按需修改）
VLLM=/usr/local/python3.12.13/bin/vllm
PYTHON=/usr/local/python3.12.13/bin/python3
MODEL_DIR=/workspace/local_models/MiniCPM-o-4_5
SERVED=openbmb/MiniCPM-o-4_5
REPO_DIR=/workspace/vllm-omni

# 原始数据源（唯一外部依赖；换机器改这里，然后跑 prepare_data.sh）
RAW_DATA_DIR=/workspace/shared_assets/datasets
SEED_TTS_TAR="$RAW_DATA_DIR/CowboyZ/seed-tts-eval/seedtts_testset.tar"
DAILY_OMNI_RAW="$RAW_DATA_DIR/MTEB/Daily-Omni"
VIDEOMME_PARQUET="$RAW_DATA_DIR/lmms-lab/Video-MME/videomme/test-00000-of-00001.parquet"
VIDEOMME_ZIPS_DIR="$RAW_DATA_DIR/lmms-lab/Video-MME"
# WER 转写模型（whisper-large-v3）来源：本地已有目录时拷贝，留空则跳过（见 prepare_data.sh）
WHISPER_SRC="${WHISPER_SRC:-}"

# 数据子路径（prepare_data.sh 生成后即就位）
SEED_TTS_DATA="$DATA_DIR/seed-tts/seedtts_testset"
DAILY_OMNI_QA="$DATA_DIR/daily_omni/qa.json"
DAILY_OMNI_VIDEOS="$DATA_DIR/daily_omni/Videos"
VIDEOMME_VIDEOS="$DATA_DIR/videomme_videos"
WHISPER_DIR="$DATA_DIR/models/whisper-large-v3"

# 网络（内网无 huggingface.co）
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1

mkdir -p "$RESULTS"
