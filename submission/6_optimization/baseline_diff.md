# 官方基线配置 vs 优化配置 diff（复现审查材料）

- 官方基线：`1_code/baseline_official.yaml`（vllm-omni 上游 `vllm_omni/deploy/minicpmo_4_5.yaml` 原样，未改动）
- 优化版：`1_code/minicpmo_4_5.yaml`（当前冠军配置）

## 全部差异（优化版相对官方基线的 5 处改动）

| # | 配置项 | 官方基线 | 优化版 | 优化归属 | 验证 |
|---|---|---|---|---|---|
| 1 | `codec_chunk_frames` | 25 | 15 | E5（TTFP -18%）| WER/SIM 全量等价 |
| 2 | `initial_codec_chunk_frames` | （无，默认 0）| 8 | E6（TTFP 再 -10%）| WER/SIM 全量等价 |
| 3 | `token2wav_n_timesteps` | （无，默认 10）| 5 | E7/E9（RTF 0.85→0.64）| WER 32×2 轮 + 全量等价 |
| 4 | `platforms.npu.stages[0].compilation_config.cudagraph_mode` | PIECEWISE | FULL_AND_PIECEWISE | C49（TTFT -9.8%/TTFP -8.4%/RTF -11.3%）| WER 32 零变化 + 全量验证中 |
| 5 | `platforms.npu.stages[1].compilation_config.cudagraph_mode` | PIECEWISE | FULL_AND_PIECEWISE | C49（同上）| 同上 |

> 注：`platforms.npu.stages[i].max_num_batched_tokens: 8192` 为 vllm-ascend 平台默认注入值（非本队改动，上游 yaml 未显式设置时由平台填充）。

## 复现步骤（910C 官方环境）

1. **基线对照**：`server_restart.sh baseline_official.yaml` → 跑官方评测（VideoMME/DO/TTS-Seed/性能）
2. **优化版**：`server_restart.sh minicpmo_4_5.yaml` → 同口径评测
3. 对比两轮结果（优化版精度 ≥ 准入线 + 性能更优）即验证优化有效性

## 优化前后精度/性能对比表（910B3 实测，供参考）

| 指标 | 官方基线（910B3 bf16 E0 实测）| 优化版（910B3）| 910C 官方基线 |
|---|---|---|---|
| WER（zh 全量 2020）| 3.42%（en 全量，zh 未跑）| **1.41%** ✅ | 1.414 |
| VideoMME 2700 | — | **69.48%** ✅ | 69.0 |
| Daily-Omni 1196 | — | **78.51%** ✅（08-09 实测）| 79.5 |
| TTFT | 434.5 ms | **386.5 ms** | 333.27 |
| TTFP | 1662.3 ms | **759.5 ms** | 986.47 |
| RTF | 0.78 | **0.55** | 0.4423 |
