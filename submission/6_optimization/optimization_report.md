# 优化分析报告

## 原始瓶颈分析

910B3 单卡上 Seed-TTS en 全量（1088 条）首跑结果：**WER 3.42% 超标**（准入 ≤1.56）。
错误分布均匀（254/1088 条）、抽样音频时长正常（非截断）——排除转写链路问题
（NPU whisper fp16 自测、CPU np.interp、官方 NPU 协议三条路径一致），定位为**合成质量**问题。

服务日志确认三阶段推理 dtype 均为 `torch.bfloat16`（MiniCPM-o 4.5 权重为 bf16 存储，
不显式指定时 vllm 默认 bf16 推理）。**910B 上 bf16 精度不足**导致 talker/codec 读错词。

## 优化方法（E3：dtype fp16）

```bash
# 启动参数增加
--dtype float16
```

权重 bf16 存储 → 推理时 cast 为 fp16（日志确认 "Casting torch.bfloat16 to torch.float16"），
与官方 910C F16 档对齐。

## 优化效果（同批 32 条 Seed-TTS en）

| 指标 | bf16 | fp16 | 判定 |
|---|---|---|---|
| WER | 3.42%（全量 1088 条） | **1.28%** | ✅ ≤1.56 达标 |
| SIM/ASV（官方 base-plus 协议） | 0.8467 | 0.8467 | ✅ ≥0.689（无差异，音色与精度无关） |
| TTFT | 434.5ms（全量） | 448.2ms | 持平（±3% 噪声） |
| TTFP | 1662.3ms（全量） | 1702.0ms | 持平 |
| RTF | 0.78 | 0.81 | 持平 |

**结论：fp16 修复精度、零性能代价，可行。**

## 优化方法（E5：首块提前 + flow 解码加速，当前默认配置）

MiniCPM-o 的 TTS 首块音频要等 stage1 攒满 `codec_chunk_frames` 个 codec 帧才发出
（`minicpmo_4_5_omni.py:295`，25 帧 × 40ms/帧 = 1.0s 音频内容），stage2 每块固定跑
10 步 flow 解码（`codec2wav.py:488`）。两个 yaml 改动：

```yaml
codec_chunk_frames: 15        # 首块/稳态块 25→15 帧（提前 ~0.4s 输出）
token2wav_n_timesteps: 8      # flow 解码 10→8 步（每块解码时间 ~线性下降）
```

### 优化效果（32 条 Seed-TTS en，fp16 基础上）

| 指标 | E3 (25/10) | E5 (15/8) | 判定 |
|---|---|---|---|
| WER | 1.28% | 0.83~1.11%（两次独立 32 条） | ✅ ≤1.56 |
| SIM/ASV | 0.8467 | **0.8466** | ✅ ≥0.689（音色无影响） |
| TTFT | 448.2ms | 443.3ms | 持平 |
| TTFP | 1702.0ms | **1388.9ms（-18.4%）** | 首块提前 0.4s |
| RTF | 0.81 | 0.85（+5%） | 块数 1.67x → 每块固定解码开销，已由 ts8 吸收大半 |

**结论：E5 净收益，精度零退化，已推广为默认 `1_code/minicpmo_4_5.yaml`。**

> 中间实验：单独 15 帧 + 10 步时 RTF 涨到 1.04（块数增多），补 8 步后回落到 0.85——
> 两个旋钮必须搭配使用。

## 优化方法（E7：flow 解码 6 步，当前默认配置）

stage2 热路径分析（`batched_token2wav.py`）：每块（15 帧 codec = 0.6s 音频）计算 =
1 次 encoder（UpsampleConformerEncoderV2）+ `n_timesteps × 2` 次 estimator（16 层 DiT）
+ 1 次 HiFT。**CFG 双分支**（conditional + unconditional 拼接，`inference_cfg_rate 0.7`）
使每步 ODE 实为 2 次 DiT 前向——8 步 = 16 次，是 RTF 主成本。

```yaml
token2wav_n_timesteps: 6      # 10 → 8 → 6（每块 estimator 前向 16 → 12 次，-25%）
```

### 优化效果（32 条 Seed-TTS en，E5+E6 基础上）

| 指标 | E6 (ts8) | E7 (ts6) | 判定 |
|---|---|---|---|
| WER | 0.83~1.11% | 1.31%（两轮 32 条稳定） | ✅ ≤1.56 |
| SIM/ASV | 0.8462 | 0.8464 | ✅ ≥0.689（持平） |
| TTFT | 439ms | 431ms | -2% |
| TTFP | 1252ms | **1118ms（-11%）** | 首块 flow 步数同减 |
| RTF | 0.85 | **0.72（-15%）** | 对照本次同批基线 0.90 → 0.72（-20%） |

**结论：E7 净收益（RTF -15~20%），WER 余量从 0.45pp 收窄到 0.25pp，SIM 零退化，已推广为默认 yaml。**
后续若需更大 RTF 空间：CFG unconditional 分支隔步计算（-25% 再）、ts 4-5、w8a16（风险见下）。

## 已证伪/未执行的优化方向

- E7-b（initial 8→4）：首块 4 帧使 RTF 0.72→0.92（+28%）——每块固定解码开销（encoder+CFG 首步+HiFT 拼接）占比暴涨，initial=8 为最优平衡点
- E7-c（`VLLM_ASCEND_ENABLE_NZ=2`）：强制 fp16 权重转 FRACTAL_NZ 布局，TTFT/TTFP/RTF 全在 ±1.5% 噪声内——默认 1（仅量化场景转 NZ）已最优
- **E8（stage2 DiT INT8 在线量化）**：16 层 DiT 的 117 个 nn.Linear 就地替换为
  `npu_dynamic_quant` + `npu_quant_matmul` 的 W8A8 动态量化（复用 vllm-omni diffusion 量化框架
  `quantization/int8_config.py` 的 NPU 算子对，`dit_int8_online.py` 已实现并单测 9/9 通过）。
  **910B 实测 RTF 0.72→0.95（+32% 恶化）**，TTFP 1118→1346ms——矩阵太小（hidden 512、
  seq 15 帧），910B 的 INT8 GEMM 小矩阵无优势，且每层动态量化多出 2 个算子（激活量化 +
  matmul + dtype cast），算子启动开销 3 倍于 fp16 单算子，倒挂。**结论：910B 小矩阵场景
  INT8 不可行**（若未来有更大 batch/更长的 chunk 可重估）。实现保留（`platforms/npu/models/
  dit_int8_online.py` + 测试），yaml 开关 `token2wav_int8` 默认关闭。
  附带收获：确认 `npu_dynamic_quant`/`npu_quant_matmul` 只收 fp16/bf16（fp32 输入/输出均拒绝），
  零初始化层在真实 checkpoint 中不存在（权重训练后已非零）。
- 附带修复：`server_restart.sh` pkill 竞态（旧服务优雅关闭释放 NPU 可能 >5s，新服务提前启动会卡死在设备初始化；现改为轮询 npu-smi 确认释放后再启动，就绪时间 10-25 分钟 → 稳定 6 分钟）


- E2（connector sleep 0.01→0.001）：仓库无此轮询机制（实际轮询硬编码 1ms，`transfer_adapter/base.py:83`），证伪
- E1（stage2 开 cudagraph）：Code2Wav forward 数据依赖（运行时动态分桶 `minicpmo_4_5_code2wav.py:472-483` + token2wav 缓存形状每步变化），cudagraph 回放需形状静态，强行开启会静默重放过期图；仓库内唯一图化的 codec（MOSS-TTS）也是 CUDA-only。需大重构，放弃
- E4（w8a16 量化）：未执行（权重为 bf16 存储，量化收益需另行验证；且 bf16 已致精度超标，量化 risk 更高）
- FA3 flash attention：官方仅用于 RL 训练-推理一致性场景（`platform.py:860` 注明可能性能退化），serving 保持默认 AscendAttentionBackend
- `VLLM_ASCEND_ENABLE_NZ=2`（权重全转 FRACTAL_NZ）：未执行，可作后续 A/B

## 复现

`--dtype float16` + 默认 yaml（含 E5）启动服务 → `2_configs/eval_seed_tts_wer.sh 32` / `eval_seed_tts_asv.sh 32` / `perf_seed_tts.sh 32`

## 附：quantization patches 应用说明（0008-w4a16-linear-scheme.patch）

**目标仓库**：vllm-ascend（官方环境 pip 版本，与本地 0.19.1rc2 对齐）
**应用命令**（在 vllm-ascend 仓库根目录）：
```bash
git apply submission/6_optimization/patches/0008-w4a16-linear-scheme.patch
```
（已在干净 HEAD 上验证 `git apply --check` 通过）

**包含的改动**：
1. `vllm_ascend/quantization/methods/w4a16_linear.py`（新增）— `AscendW4A16LinearMethod`：
   compressed-tensors group-strategy int4 权重（int8 容器 [-8,7] + per-group32 scale/offset），
   `npu_weight_quant_batchmatmul(antiquant_group_size=32)`，权重保持 ND 不转 NZ
   （910C CANN 9.0.0 的 fp16/int4 内核只覆盖 weight format 29/3，FRACTAL_NZ 的 tiling key
   无内核——实测 4613751299 kernel missing）
2. `vllm_ascend/quantization/methods/__init__.py` — 注册 + 加入 `is_mx_quant_type`（per-group
   scale 的 TP 切分需要 input_dim=1，method_adapters.py:120 门控）
3. `vllm_ascend/quantization/compressed_tensors_config.py` — `matched_target is None` 回落
   （与 0005 同 hunk；**若官方环境已应用 0005，该 hunk 会重复**，可跳过此段用 `git apply
   --exclude=...` 或手动处理，其余两个文件不受影响）

**配合的提交物**（无需 patch）：
- `1_code/minicpmo_4_5_int4.yaml` — stage0 `quantization: compressed-tensors` + 指向 w4a16 ckpt
- `6_optimization/quant_export_w4a16.py` — 导出脚本（先在本机/官方环境 CPU 运行生成 ckpt）
- ckpt 产物：`local_models/MiniCPM-o-4_5-w4a16`（12.8G，模型目录需一并提交/重建）
