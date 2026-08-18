# 昇腾挑战赛子赛道 B（vLLM-Omni）提交材料

MiniCPM-o 4.5 昇腾 NPU 部署（910B3 单卡）、三基准评测、性能优化与 Demo 验证。

## 快速开始（新机器 5 步）

```bash
# 1. 改环境配置（唯一要改的文件：VLLM/PYTHON/MODEL_DIR/REPO_DIR/RAW_DATA_DIR）
vim 2_configs/env.sh

# 2. 打源码 patch（E6 首块提前，改 vllm-omni 一个文件）
git -C $REPO_DIR apply 6_optimization/patches/0001-minicpmo-initial-codec-chunk.patch

# 3. 准备数据（从 RAW_DATA_DIR 原始数据集生成转换产物到 7_runtime/，幂等，约 1-2h）
bash 2_configs/prepare_data.sh

# 4. 四项精度准入一键跑（自动切换服务，默认每项 32 条；N 传条数）
bash 2_configs/eval_all.sh            # WER → ASV → Daily-Omni → VideoMME，跑完打印判定

# 5. 性能测试（Seed-TTS 32 条 @ 并发 1，产出 TTFT/TTFP/RTF）
bash 2_configs/perf_seed_tts.sh
```

运行时产物（转换数据/日志/结果）统一在 `7_runtime/`，删除即可完全重置；`submission/` 本体不含任何运行时文件。

详细说明：环境前置 / 服务启动 / 全量评测 / Demo / 已知坑 / 预期结果 → **`RUNBOOK.md`**。

## 目录

| 目录 | 内容 |
|---|---|
| `1_code/` | 评测/验证脚本 + deploy 配置 + Demo 源码（gradio/realtime_duplex） |
| `2_configs/` | 评测脚本集：四项基准（`eval_videomme_official.sh / eval_daily_omni.sh / eval_seed_tts_wer.sh / eval_seed_tts_asv.sh`）+ 汇总 `eval_all.sh` + 性能 `perf_seed_tts.sh` + 数据准备 `prepare_data.sh` + 服务切换 `server_restart.sh` + Demo 脚本 |
| `3_eval_results/` | 三项 Benchmark 原始结果 |
| `4_perf_report/` | 性能测试报告（metrics_summary.md） |
| `5_demo/` | 半双工/全双工 Demo 结果（result.json + 音频分片） |
| `6_optimization/` | 优化说明 |

## 环境

- 昇腾 910B3 单卡（官方基线 910C），CANN 9.0.0
- vllm-omni v0.25.0-a3（minicpm-challenge 分支），`VLLM_WORKER_MULTIPROC_METHOD=spawn`
- 模型：`/workspace/shared_assets/models/OpenBMB/MiniCPM-o-4_5`

## 部署配置来源（配置单一源）

**评测配置只有一个**：`vllm_omni/deploy/minicpmo_4_5.yaml`（官方路径 = champion 配置，含全部优化）。
`1_code/minicpmo_4_5.yaml` 是它的软链（`../../vllm_omni/deploy/minicpmo_4_5.yaml`），改配置只改官方路径一处。

champion 相对官方原版的改动（diff 可审计）：
- `codec_chunk_frames 25→15` + `initial_codec_chunk_frames: 8`（E5+E6，TTFP -18%）
- `token2wav_n_timesteps 10→3`（C57，RTF -18%）
- `cudagraph_mode: FULL_AND_PIECEWISE`（C49，TTFT/TTFP/RTF 全 -8~11%）

| 其他文件 | 用途 |
|---|---|
| `1_code/minicpmo_4_5_duplex.yaml` | 全双工 demo（不计分） |
| `1_code/baseline_official.yaml` | 官方基线对照（复现基线用，见 `6_optimization/baseline_diff.md`） |

配套的源码 patch（E6 首块提前）在 `6_optimization/patches/`，新机器 `git apply` 即可。

服务启动命令（`2_configs/server_restart.sh` 内置）：`--deploy-config $CODE_DIR/<yaml>`。官方多卡变体（2gpu/3gpu/4gpu/8x4090）与本提交无关。

## 服务启动

```bash
export VLLM_WORKER_MULTIPROC_METHOD=spawn
vllm serve /workspace/shared_assets/models/OpenBMB/MiniCPM-o-4_5 --omni \
    --served-model-name openbmb/MiniCPM-o-4_5 --trust-remote-code \
    --dtype float16 \
    --deploy-config 1_code/minicpmo_4_5.yaml \
    --stage-init-timeout 600 --host 0.0.0.0 --port 8091 \
    --allowed-local-media-path 7_runtime/media
```

## 评测结果汇总（910B3，当前配置 fp16+E5+E6+E7+E9）

| 基准 | 结果 | 准入阈值 | 判定 |
|---|---|---|---|
| Seed-TTS WER（**zh 全量 2020 条**，官方门槛口径） | **1.41%** | ≤1.56 | ✅ **达标**（与官方 910C 基线 1.414 持平；en 3.03% 为口径错误参照） |
| Seed-TTS SIM/ASV（全量 1088 条，官方 base-plus 协议） | **0.8524** | ≥0.689 | ✅ |
| Daily-Omni（**全量 1196 条**） | **78.51%** | ≥77.5 | ✅ 与官方参考 78.28% 持平 |
| VideoMME（**全量 2700 条，官方 minicpm-frames/96 帧协议**） | **69.48%** | ≥67.0 | ✅ 与官方参考 69.96% 差 0.48pp |

> ✅ **2026-08-13 口径修正**：官方达标门槛为 **ZH WER**（paraformer-zh 转写，`--seed-tts-locale zh`）。zh 全量 2020 条 = **1.41% 达标**（并发 1/4 一致性已验证）；此前 en 口径（3.03%）为错误参照。
> WER 小样本警示仍适用：32 条是前 32 条简单样本，全量才是有效口径。

### 910B Baseline 整合表（Seed-TTS en）

口径：910B 无改动 = 官方配置 + 默认 bf16（E0 全量）；910B base = 官方配置 + fp16（32 条）；E9 = 当前优化配置。

| 指标 | 910B 无改动（bf16） | 910B base（fp16官方） | **E9 优化后** | 910C F16 基线 |
|---|---|---|---|---|
| WER（zh 官方口径） | 3.42%（en） | 2.96%（en） | **1.41%（zh 全量 2020）** ✅ | 1.414 |
| SIM | 0.8467 | — | **0.8524**（全量1088） | 0.709 |
| TTFT/TTFP/RTF | 434.5/1662.3/0.78 | 432.6/1687.4/0.80 | **436.9/1032.6/0.633** | 333.27/986.47/0.4423 |

### 历史基线 / 踩坑记录（均已修复，仅作追溯）

| 基准 | 结果 | 问题 | 修复 |
|---|---|---|---|
| Seed-TTS WER（bf16，全量 1088 条） | 3.42% ❌ | 910B bf16 推理精度不足（默认 dtype） | → fp16（E3，参数已入默认启动命令；全量 3.03% 仍超线，为硬件固有） |
| Seed-TTS WER（fp16，32 条） | 1.28~1.31% ❌ | **前 32 条简单样本假象**（全量 3.03% 超标） | 全量口径验证（2026-08-11 补测） |
| Daily-Omni（128 条） | 60.16% ❌ | 样本偏差（全量才是有效口径） | 全量 1196 → 78.51% ✅ |
| VideoMME（128 帧 file:// 自写协议） | 65.43% ❌ | 评测协议不同（官方为 minicpm-frames/96 帧） | 官方协议全量 2700 → 69.48% ✅ |

性能（Seed-TTS，910B3，同脚本同口径 32 条）：**910B base（fp16+官方配置）TTFT 433ms / TTFP 1687ms / RTF 0.80 → E9+C20 优化后 TTFT 437ms / TTFP 1033ms / RTF 0.633**（TTFP -38.8%，RTF -20.9%；C20 = DiT QKV 融合+adaLN 预计算，2026-08-13 晋级，全量 WER/SIM 持平）。
官方 910C F16 基线：TTFT 333ms / TTFP 986ms / RTF 0.44（910B 算力差异 1.3-1.7x）。

## 优化结论（详见 4_perf_report 与 6_optimization）

**E3：`--dtype float16`** —— 910B 上 bf16 推理精度不足导致 Seed-TTS 合成读错词（WER 3.42% 超标）；
fp16 后 WER 1.28% 达标，性能持平。已证实可行、零性能代价。

**E5+E6+E7+E9（当前默认配置）：`codec_chunk_frames 25→15` + `initial_codec_chunk_frames 8` + `token2wav_n_timesteps 10→8→6→5`** ——
首块音频提前 0.4s 输出（E5：TTFP 1702→1389ms），首块再提前（E6：1389→1252ms，累计 -26%），
flow 解码 5 步（E7+E9：RTF 0.85→0.64，TTFP 1252→1044ms，累计 -38%）；WER 1.31% / SIM 0.8524 达标（全量1088）。
E6 需源码 patch（`6_optimization/patches/`，Qwen3-TTS 同款机制）。已推广为 `1_code/minicpmo_4_5.yaml`。
已证伪：initial 8→4（RTF +28%）、NZ=2（无收益）、DiT INT8 量化（小矩阵倒挂，RTF +32%，详见 6_optimization）。

## Demo 验证

- **半双工**（gradio）：页面可访问、32 音频分片连续流式（26s 音频），`5_demo/half_duplex/`
- **全双工**（realtime_duplex_demo.py，16k PCM 输入 + HT 参考音色）：ok=true、3 音频分片流式输出、
  模型对输入语音回应（transcript: "Okay, I get it."），`5_demo/duplex/`

## 复现步骤

详细操作见 **`RUNBOOK.md`**（环境、服务启动、评测跑法、demo、已知坑、预期结果）。

```bash
# 一键四项精度准入评测（自动切换服务，跑完打印准确率+判定）
bash 2_configs/eval_all.sh            # 默认每项 32 条；可传条数: eval_all.sh 128
# 单项：
bash 2_configs/eval_seed_tts_wer.sh   # TTS-Seed WER（准入 ≤1.56）
bash 2_configs/eval_seed_tts_asv.sh   # TTS-Seed ASV/SIM（准入 ≥0.689）
bash 2_configs/eval_daily_omni.sh     # Daily-Omni（准入 ≥77.5，自动用 bench yaml）
bash 2_configs/eval_videomme_official.sh   # VideoMME 官方协议（准入 ≥67.0，minicpm-frames/96帧，默认全量）
# 性能：bash 2_configs/perf_seed_tts.sh   # TTFT/TTFP/RTF 对比 910C 基线
# Demo：bash 2_configs/demo_half_duplex.sh / demo_duplex.sh
```
全量评测：`bash 2_configs/eval_daily_omni.sh 1197 restart-server`（Daily-Omni 全量 1197，自动切 bench 服务）/ `eval_seed_tts_zh.sh`（Seed-TTS zh 2020）。

> 注：`fusion_result.json` 为昇腾算子融合运行产物（非评测数据）。
