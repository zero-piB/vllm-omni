# 运行指南（RUNBOOK）

## 0. 环境前置

- 昇腾 NPU 环境（910B/910C），CANN 9.x，`VLLM_WORKER_MULTIPROC_METHOD=spawn`
- 模型：`/workspace/shared_assets/models/OpenBMB/MiniCPM-o-4_5`（本地路径）
- 网络：`export HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1`（内网无 huggingface.co）
- vllm 二进制：`/usr/local/python3.12.13/bin/vllm`
- **源码 patch（E6 首块提前，必须）**：`git -C <vllm-omni 仓库> apply submission/6_optimization/patches/0001-minicpmo-initial-codec-chunk.patch`
  （改动 `stage_input_processors/minicpmo_4_5_omni.py` 一个文件，为 chunker 增加 `initial_codec_chunk_frames` 支持；
  测试 patch 同名 `-tests.patch` 可选。不应用则 yaml 里的 `initial_codec_chunk_frames: 8` 键被忽略，其余功能不受影响）

### 换机器/重置流程

1. 拷贝 `submission/` 整目录 → 改 `2_configs/env.sh` 顶部 5 个变量（`VLLM`/`PYTHON`/`MODEL_DIR`/`REPO_DIR`/`RAW_DATA_DIR`；`RAW_DATA_DIR` 指向原始数据集根目录，whisper 模型另设 `WHISPER_SRC`）
2. 应用源码 patch（见上）
3. `bash 2_configs/prepare_data.sh` —— 从 RAW 源生成全部评测数据到 `7_runtime/media/`（seed-tts 解压 / Daily-Omni 转换 / VideoMME 抽视频 / whisper 拷贝，幂等可重跑）
4. 之后所有评测/性能/demo 脚本直接可用；**运行时产物全部在 `7_runtime/`，删除即完全重置**，submission 本体无运行时依赖

## 1. 服务启动（两种配置）

```bash
# 默认配置（Seed-TTS / VideoMME / 性能评测）——等价于 ./server_restart.sh minicpmo_4_5.yaml
vllm serve /workspace/shared_assets/models/OpenBMB/MiniCPM-o-4_5 --omni \
  --served-model-name openbmb/MiniCPM-o-4_5 --trust-remote-code --dtype float16 \
  --deploy-config /workspace/submission/1_code/minicpmo_4_5.yaml \
  --stage-init-timeout 600 --host 0.0.0.0 --port 8091 \
  --allowed-local-media-path /workspace/submission/7_runtime/media

# Daily-Omni 配置（stage0 repetition_penalty 1.2，--deploy-config 换 bench yaml）
# 全双工配置（--deploy-config 换 duplex yaml，需在 /workspace/vllm-omni 下启动以解析相对 base_config）
```

就绪标志：日志出现 `Application startup complete`（约 7 分钟）。
**注意**：`--served-model-name` 只传一个（openbmb 名），bench 的 `--model` 必须与之一致（openbmb/MiniCPM-o-4_5）；传两个会被后者覆盖。

## 2. 精度准入评测（每项一个脚本，跑完打印准确率+判定）

```bash
cd /workspace/submission/2_configs
./eval_all.sh [N]              # 一键四项（WER→ASV→Daily-Omni→VideoMME，自动切服务），默认每项 32 条
./eval_seed_tts_wer.sh [N]     # TTS-Seed WER（准入 ≤1.56）
./eval_seed_tts_asv.sh [N]     # TTS-Seed ASV/SIM（准入 ≥0.689）
./eval_daily_omni.sh [N] [restart-server]   # Daily-Omni（准入 ≥77.5；加 restart-server 自动切 bench 服务）
./eval_videomme_official.sh [N]  # VideoMME 官方协议（准入 ≥67.0；minicpm-frames/96帧，默认全量 2700）
./server_restart.sh <yaml>      # 手动切服务（minicpmo_4_5 / _bench / _duplex）
```

- 耗时：seed 每项 ~12min，daily ~15-25min（首次抽帧），videomme 每 32 条约 6min
- 输出：`7_runtime/results/`（各 `*_HHMMSS.log`）+ 汇总 `eval_all_summary.txt`；服务日志 `7_runtime/minicpmo_server.log`
- **先小样本确认达标量级，再全量**（`--num-prompts` 传全量条数即可）

## 3. 全量基准

```bash
# Seed-TTS en 全量（1088 条）：./eval_seed_tts_wer.sh 1088（+ ./eval_seed_tts_asv.sh 1088）
#   ~2h 生成 + WER 转写 ~1.5h（SEED_TTS_EVAL_DEVICE=npu:0 加速，默认 CPU 转写 12h）
# Seed-TTS zh 全量（2020 条）：
bash /workspace/submission/2_configs/eval_seed_tts_zh.sh

# Daily-Omni 全量（1197 条，实测 128 条约 8.6 分钟 → 全量约 1.5-2h）：
bash /workspace/submission/2_configs/eval_daily_omni.sh 1197 restart-server  # 自动切 bench 服务并跑全量

# VideoMME 全量（2700 条，官方协议）：
bash /workspace/submission/2_configs/eval_videomme_official.sh 2700
#   ~5-10h（910B 单条约 10-12s prefill），中断后重跑自动跳过已完成条目
```

## 4. Demo

```bash
# 半双工（gradio，需服务默认配置运行中）
python3 /workspace/vllm-omni/examples/online_serving/minicpmo/gradio_demo.py \
  --minicpmo45-api-base http://127.0.0.1:8091/v1 --minicpmo45-model openbmb/MiniCPM-o-4_5 --port 7862
# 验证：/workspace/eval/verify_demo.py（自动跑 TTS 流式请求并落盘音频分片）

# 全双工（先重启服务为 duplex yaml）
python3 /workspace/vllm-omni/examples/online_serving/minicpmo/realtime_duplex_demo.py \
  --url "ws://127.0.0.1:8091/v1/realtime?duplex=1" \
  --model openbmb/MiniCPM-o-4_5 \
  --input-wav /tmp/duplex_input_16k.wav \
  --ref-audio /workspace/shared_assets/models/OpenBMB/MiniCPM-o-4_5/assets/HT_ref_audio.wav \
  --output-dir /tmp/duplex_out
# 输入音频必须 16k 单声道 PCM16（--require-audio 要求至少一个音频分片）
```

## 5. 已知坑（按发生频率）

| 现象 | 原因 | 解法 |
|---|---|---|
| bench 404 model not found | `--model` 与服务 `--served-model-name` 不一致 | 统一用 openbmb/MiniCPM-o-4_5 |
| bench 崩（huggingface 网络错误） | 客户端解析 tokenizer 走外网 | `--model` 本地路径或 openbmb 名 + `--trust-remote-code` |
| 服务起不来（Deploy config not found） | 相对路径 + cwd 不对 | `--deploy-config` 用绝对路径 |
| WER 超标（~3.4%） | 910B bf16 精度不足 | 必须 `--dtype float16` |
| pkill 后自身被误杀（exit 144） | `pkill -f` 匹配自身命令行 | 用 PID kill 或 `[v]llm` 字符类 |
| 服务中途崩（TBE Subprocess disappeared） | 昇腾长跑不稳定 | 评测均断点续跑（JSONL 逐条追加） |

## 6. 预期结果（910B3 实测，默认 yaml 含 E5 优化）

| 基准 | 结果 | 阈值 |
|---|---|---|
| Seed-TTS WER（fp16+E5+E6+E7+E9） | 1.31%（32 条稳定） | ≤1.56 |
| Seed-TTS SIM | 0.8476 | ≥0.689 |
| Daily-Omni（全量 1196） | 78.51% | ≥77.5 |
| VideoMME（全量 2700，官方协议） | 69.48% | ≥67.0 |
| 性能 TTFT/TTFP/RTF | 432ms/1044ms/0.64 | 910C: 333/986/0.44 |

默认 yaml（`1_code/minicpmo_4_5.yaml`）含三项优化：`--dtype float16`（E3，必须）+ `codec_chunk_frames: 15` + `initial_codec_chunk_frames: 8`（E5+E6，TTFP -26%）+ `token2wav_n_timesteps: 5`（E7+E9，RTF -20%）。若需与官方配置逐项对齐，把这三个键改回 25 / 8（删除 initial 键）/ 10 即可。已证伪：initial 8→4（RTF +28%）、NZ=2（无收益）、DiT INT8 量化（小矩阵倒挂 RTF +32%）。
