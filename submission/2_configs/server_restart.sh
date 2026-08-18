#!/bin/bash
# 重启 vLLM-Omni 服务（fp16 + 指定 deploy yaml）
# 用法: ./server_restart.sh <yaml文件名>   （minicpmo_4_5 / _bench / _duplex，从 1_code/ 取）
set -euo pipefail
YAML=${1:?usage: server_restart.sh <yaml文件名>}
source "$(dirname "$0")/env.sh"
# 支持绝对/相对路径（实验 yaml 放 7_runtime/exp/）；纯文件名回退到 1_code/
case "$YAML" in
  */*) YAML_PATH="$YAML" ;;
  *)   YAML_PATH="$CODE_DIR/$YAML" ;;
esac

pkill -f "[v]llm serve" || true
# 等旧服务完全退出（NPU 显存释放），否则新服务初始化会卡死（优雅关闭可能 >5s）
for i in $(seq 1 24); do
  npu-smi info 2>/dev/null | grep -q "VLLMStageEngi" || break
  [ $((i % 6)) -eq 0 ] && echo "等待旧服务释放 NPU…（$((i * 5))s）"
  sleep 5
done
LOG="$RESULTS/../minicpmo_server.log"
LINE0=$(wc -l < "$LOG" 2>/dev/null || echo 0)
nohup "$VLLM" serve "$MODEL_DIR" --omni \
  --served-model-name "$SERVED" \
  --trust-remote-code --dtype float16 \
  --deploy-config "$YAML_PATH" \
  --stage-init-timeout 900 --init-timeout 1200 --host 0.0.0.0 --port 8091 \
  --allowed-local-media-path "$DATA_DIR" \
  >> "$LOG" 2>&1 &
last=""
for i in $(seq 1 60); do
  # 只在本次启动后的新日志行中找就绪标记（避免匹配旧服务历史行）
  tail -n +$((LINE0 + 1)) "$LOG" 2>/dev/null | grep -q "Application startup complete" && { echo "服务就绪 ($YAML)"; exit 0; }
  # 阶段感知心跳：stage0/1/2 各自初始化一次（日志标记 stage_init_utils.py:689）
  n=$(tail -n +$((LINE0 + 1)) "$LOG" 2>/dev/null | grep -cE "\[stage_init\] Stage-[0-9]+ set runtime devices" || true)
  s="stage 初始化 ${n:-0}/3"
  # 阶段变化即时打印；无变化时每 60s 心跳一次
  [ "$s" != "$last" ] && { echo "服务启动中: $s"; last="$s"; }
  [ $((i % 6)) -eq 0 ] && [ "$s" = "$last" ] && echo "服务启动中: $s（$((i / 6))/10 分钟，日志: $LOG）"
  sleep 10
done
echo "服务启动超时" >&2; exit 1
