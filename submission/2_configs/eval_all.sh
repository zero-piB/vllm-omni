#!/bin/bash
# 四项精度准入一键评测：WER + ASV → VideoMME → Daily-Omni，结束输出汇总表
# 用法: ./eval_all.sh [每项条数]   （默认 32；服务自动切换 yaml）
set -euo pipefail
source "$(dirname "$0")/env.sh"
N=${1:-32}
cd "$(dirname "$0")"
SUM="$RESULTS/eval_all_summary.txt"
: > "$SUM"

echo "==== 1/3 TTS-Seed WER（服务: 默认 yaml） ====" | tee -a "$SUM"
./server_restart.sh minicpmo_4_5.yaml
./eval_seed_tts_wer.sh "$N" 2>&1 | tee -a "$SUM"

echo "==== 2/3 TTS-Seed ASV/SIM（服务: 默认 yaml） ====" | tee -a "$SUM"
./eval_seed_tts_asv.sh "$N" 2>&1 | tee -a "$SUM"

echo "==== 3/3 Daily-Omni（服务: bench yaml） ====" | tee -a "$SUM"
./server_restart.sh minicpmo_4_5_bench.yaml
./eval_daily_omni.sh "$N" 2>&1 | tee -a "$SUM"

echo "==== 可选: VideoMME（服务: 默认 yaml，耗时较长） ====" | tee -a "$SUM"
./server_restart.sh minicpmo_4_5.yaml
./eval_videomme_official.sh "$N" 2>&1 | tee -a "$SUM"

echo ""
echo "===================== 汇总（准入参考） =====================" | tee -a "$SUM"
cat "$SUM" | grep -E "====|WER =|SIM/ASV =|Accuracy =|判定" | tee -a /dev/null
echo "详细日志: $RESULTS/，汇总: $SUM"
