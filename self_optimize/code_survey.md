# 代码勘探存档（2026-08-16 三轮 agent + 配置链勘察 + 自查）

> 与 `ledger.json` 互补：ledger 记候选/判定（F 系列），本文件记**看过的代码结构与平台机制**（可复查的地图）。
> 用途：存档 + 防重复挖掘。所有结论带 `文件:行号`，重查可复验。

## A. 平台特性（910B3 / 昇腾，实测钉死）

| # | 结论 | 证据 |
|---|---|---|
| A1 | 昇腾 graph capture **只支持 aclnn 算子**；aclop（TBE 旧式）不可捕获，捕获失败即 RuntimeError（无静默 fallback）| F9 + acl_graph.py:203-217 + platform.py:540-600 |
| A2 | **fp16 算力优势只在算力饱和的大矩阵兑现**：小矩阵（30×512/512×512）fp32/fp16 = 1.01-1.05x；大矩阵（4096³）3.43x → stage0（带宽受限）与 stage2 DiT（发射受限）都非算力受限 | c47b_gemm_probe.py 实测 |
| A3 | DiT 每 kernel 真实成本 ~68μs（launch 12.7μs + 执行），3400 kernel/块 → 发射 43ms/块 | C20 msprof |
| A4 | eager 38.6 vs 图化 16.8 ms/token（stage1）——图化消发射砍半 | C11 |
| A5 | 显存带宽 910B ~1.2TB/s：8B fp16 权重 16GB → 13ms/token 带宽下限 | stage 分解（C21）|
| A6 | 昇腾 DMA/计算不并行、host 发射结构性开销（多流 1.74x 验证）| F15 |
| A7 | kernel 清理类优化收益上限 <0.1ms/step | F22（C30/C31 先例）|

## B. cudagraph 传递链与决策（关键机制）

```
yaml platforms.npu.stages[i].compilation_config.cudagraph_mode
  → DeployConfig.platforms（自由 dict，无 schema）stage_config.py:476,668
  → _apply_platform_overrides 按 stage_id setattr（整 dict 替换，非深合并）stage_config.py:711-763
  → StageDeployConfig.compilation_config → engine_args stage_config.py:846-851
  → OmniEngineArgs → EngineArgs.__post_init__ dict→CompilationConfig arg_utils.py:737-741
  → VllmConfig.post_init：enforce_eager→NONE（vllm.py:1142-1148）、O2 默认 cudagraph_mode=FULL_AND_PIECEWISE（vllm.py:1216-1265）
  → vllm-ascend platform.py:484-517 强制（num_of_warmups=1、capture sizes、enable_npugraph_ex=False）
```

- **O2 默认就是 FULL_AND_PIECEWISE**——yaml 显式 PIECEWISE 是次优覆盖（C49 只是对齐回默认）
- FULL 分支：`splitting_ops=[]`（整图，不分段）；PIECEWISE 分支：`set_splitting_ops_for_v1` + mla/dsa extend（platform.py:566-591）
- encoder-decoder 模型强制回退 PIECEWISE/NONE（platform.py:542-557）
- `ENPU_ENABLE=true`：FULL 模式 update/replay 排序优化（跳过 replay 前 synchronize）model_runner_v1.py:481-483（C 层 getenv，不在 envs.py）
- 无 `VLLM_CUDAGRAPH_MODE` env；`VLLM_USE_BREAKABLE_CUDAGRAPH` 被平台强制关

## C. 配置面全表（已扫完，勿重复）

### CompilationConfig 字段有效性（compilation.py:380-763）
- 生效：`cudagraph_mode`（stage0/1 已用）、`cudagraph_capture_sizes`（默认 [1,2,4] 已最优）、`compile_sizes`（Ascend 当额外 warmup，worker.py:717-731）、`custom_ops`、`splitting_ops`（piecewise 切分）
- 被忽略/强制：`mode`（只允许 NONE/VLLM_COMPILE）、`cudagraph_num_of_warmups`（强制=1）、`backend`（非 inductor）、`pass_config.*`（NPU 自动关大部分）、`inductor_*`、`use_inductor_graph_partition`、`fast_moe_cold_start`（非 MoE）、`xlite_graph_config`（FULL 下 enforce_eager）
- 本版本无 `level` 字段、无 `enable_static_kernel`（那是 ascend_config 的）

### platforms.npu 参数通道（stage_config.py:695-761）
- 字段路由：`hasattr(base, key)` → StageDeployConfig（stage_config.py:317-421 全字段）或 `engine_args: {}` 嵌套或 `env: {}` 通道
- **未用且有价值**：`engine_args.additional_config.ascend_compilation_config.enable_static_kernel`（默认 False，ascend_config.py:572-621，静态 shape 图优化 → C55）
- ascend_config 其余开关：`enable_npugraph_ex`(默认True 但 PIECEWISE 下强制 False)、`fuse_norm_quant`/`fuse_qknorm_rope`(True)、`fuse_allreduce_rms`(False, TP1 无意义)、`fusion_ops_gmmswigluquant`(True)、`enable_mlapo`(env 默认1)、`weight_prefetch_config`(仅 MLA)、`short_request_first_config`(默认关，并发1无意义)、`VLLM_ASCEND_BALANCE_SCHEDULING`(仅多卡)
- 顶层非 `stages` 的 platforms key 全部被忽略（stage_config.py:727）

### stage2 配置无效性（F30 已证伪）
- stage2 `enforce_eager: true` 无条件短路 cudagraph（vllm.py:1142-1148 + model_runner_v1.py:3098-3099）
- 模型不可图化：forward 动态分桶（_bucket_key）、_parse_item host 分支、逐桶 stack → decode_batch（minicpmo_4_5_code2wav.py:321-337,380-473,475-490,659-673）

## D. stage2 代码结构（site-packages 双副本）

| 模块 | 运行文件 | 每块成本 | 关键点 |
|---|---|---|---|
| flow encoder | stepaudio2/cosyvoice2/transformer/upsample_encoder_v2.py | 13ms，~220 kernel | rel_shift 40 kernel 可省 ~2ms（低于门槛）|
| DiT cfm | stepaudio2/cosyvoice2/flow/decoder_dit.py | 171ms，~3400 kernel | mask 恒 None（587 行）；q/k_norm 已最小；QKV 融合只在 forward 非 chunk 路径（F13 证伪补 chunk）|
| HiFT | flashcosyvoice/modules/hifigan.py | 38ms，~390 kernel | **stft/istft CPU fallback**（torch_npu codegen 无 istft，stft op_api 不可靠）每块 2-4 次 D2H/CPU/H2D；f0 重叠区重算 ~21%；stft_window 每块 2 次 H2D |
| 适配层 | vllm_omni/.../minicpmo_4_5_code2wav.py + batched_token2wav.py | — | C27 setup 缓存（~220ms→1-2ms）；decode_batch 每 forward 调 len(buckets) 次（单请求 1 次）|

- fp16 实测无收益（F29）：flow fp32 397 vs fp16 409 ms/块；且 C7.patch 有第 6 处崩点（C20 adaLN 表按构建时 dtype 缓存）
- `token2wav_float16` 键 champion yaml 无（协议文档"已开"是错的——F29 记账时修正）

## E. 管线时序（TTFT/TTFP 分解）

```
t=0 请求 → t≈111 stage0 prefill → t≈456 stage0 生成完（FINAL_ONLY 才发文本，TTFT≈444）
→ t≈480 stage1 prefill → t≈614 stage1 首块 8 步 → t≈880 stage2 首块音频（TTFP≈880）
```
- `serving_chat.py:206-241` FINAL_ONLY 强制（C48 机制：_route_output 门槛 = output.finished 与 output_kind 无关，orchestrator.py:1296-1300）
- stage1 无 prewarm（orchestrator.py:2025-2029 注释自认；C50 用）
- stage2 有 prewarm（占位 prefill 并行）
- TTFP 预算：stage1 prefill+8 步 ~150-180 + stage2 首块 ~60-100 + handoff ~15-30 + **残差 130-210ms 未解释**（C21-instrument.md 设计已存在）
- warmup 已充分（patch.py:1780-1832：3 条真实请求预热，C27 缓存/JIT/cudagraph 全覆盖）
- connector：每 chunk 一个 shm 段，无队列深度概念，put/get ~0.2-0.5ms（F21）

## F. 未探明面（未来挖掘入口）

1. **上游 diff 扫描已完成（2026-08-17，无货）**：fb89ab43（merge-base）..origin/main 137 commits，perf 相关 22 个、MiniCPM-o/Qwen3-TTS 相关 ~25 个，逐一核查：
   - #5382 whisper mask 向量化：**已证伪**（08-13 实验 TTFT 447.7 vs 436.9 噪声内略差；08-15 批量应用后 revert，不重试）
   - #5165 SigLIP unpadding metadata 复用：唯一未试的 MiniCPM-o 专属 perf 增量，但收益上限 = 每层省 1 次 nonzero+.item()，27 层 ≈ 1-2ms TTFT < 采纳线 10ms，且撞 C30/C31/C37 "CPU/kernel 清理类 <0.1ms/step" 平台模式 → 直接证伪
   - #5638 TensorRT Code2Wav：CUDA-only（`is_cuda()` 门控），910B 不可用
   - #5503 RoPE cos/sin cache：目标文件 `common/qwen3_code_predictor.py` 不在比赛分支（比赛分支 code predictor 走 vllm 标准 RoPE 自带 cache）→ 不适用
   - #5608 NPU BNSD RoPE fallback：bugfix，四项全达标无动机；目标文件 `platforms/npu/layers/rotary_embedding.py` 比赛分支不存在
   - #5637 single-GPU stabilize：CUDA-only HiFT 预热
   - #4958 QKV fuse：上游自身已 revert + 省算力类（F29 平台证伪）
   - 其余：模型专属（Minimax/MOSS/GLM/Qwen2.5-Omni）或 CI/文档/配置重构
   - vllm-ascend 上游不扫：不在提交物内（官方环境 pip 固定版），改动无法生效
   - **结论：上游无可用增量，此面闭合**
2. 显存碎片/allocator（评分口径显存不紧张，低价值）
3. CPU 侧热点（评测客户端 1 核 vs 910C 多核是环境差异，低价值）

## G. 已探明即勿重复（交叉索引 ledger F 系列）

F4（stage2 数据依赖）/F9（aclop 不可捕获）/F11（inductor 动态 shape）/F13（F.linear cat 负优化）/F14（tensor op 参数慢）/F15（DMA 不并行）/F16/F21（序列化家族）/F17（采样链）/F18/F28（KV 缓存家族）/F19（duplex-only）/F20（sleep_s 零消费者）/F22（kernel 清理上限）/F23（CFG 语义必需）/F24（Ascend C 融合边缘）/F25（w8a16）/F26（npugraph_ex 依赖）/F27（profiling_chunk_config 无评分价值）/F29（flow fp16 无收益）/F30（stage2 图化配置死路）/F36（上游 diff 无增量）/F37（vision eager attn 无内核空间：SDPA 0.70x 负优化，Ascend C 评分收益 0）/F38（HiFT conv TransData 固有：weight_norm 无关 A/B 证伪、NZ 预转换被平台禁 allow_internel_format=False、aclnnConv2d 每调用 67us 含转换无缓存）

## H. C66 勘察结论（2026-08-20，MiniCPM-o stage2 live 路径 + torch.compile 机制，F44 全链）

- **live flow 构建链**：`minicpmo_4_5_code2wav.py:768`（NPU 分支）→ `MiniCPMO45Token2wav` → `StepAudio2Token2WavCore`（`model_executor/models/step_audio2/step_audio2_token2wav.py:69`）→ `_ensure_models_loaded` 用模型目录 `assets/token2wav/flow.yaml` 的 `!new:` 实例化 → **顶层 `cosyvoice2` 包**（`cosyvoice2.flow.flow.CausalMaskedDiffWithXvec`，encoder=`cosyvoice2.transformer.upsample_encoder_v2.UpsampleConformerEncoderV2`，decoder=CFM+DiT）
- **活跃 @torch.compile 全仓清单**：仅 `cosyvoice2/transformer/upsample_encoder_v2.py:408` `forward_chunk`（`dynamic=True, backend="eager"`）；343/352/364 已注释；`stepaudio2/` 与顶层 `flashcosyvoice/` 的 `qwen2_components/layers.py` 各 4 处（SiluAndMul/RMSNorm×2/RotaryEmbedding）但 **import flash_attn（本机未装）→ 不可 import，非 live 路径**（C66 原归因错误，作废）
- **patch 挂点教训**：`npu_token2wav_sdpa_context`（`platforms/npu/models/step_audio2_token2wav.py:116`）只包 core.forward/stream_chunk_for；MiniCPM-o 走 `BatchedToken2Wav._encode_chunk`（`minicpmo_4_5/batched_token2wav.py:170` 直接 `flow.encoder.forward_chunk`）**不经过它** → `apply_cosyvoice2_dit_attn_npu_patch()` 及其 compile-disable 在 MiniCPM-o 路径从未执行（8-19 日志无记录实证）。CosyVoice2 评分路径需另查（不在评分范围）
- **torch.compile 机制**（CPU 实证 torch 2.10）：`@torch.compile(backend="eager")` **每次调用都产生 Torch-Compiled Region**（3/3），即 region 事件 = compile 包装的调用边界，**不区分 cache hit/recompile**；`torch._dynamo.disable(compiled)` 与 `__wrapped__` 均可消除（0 regions）。8-19 profile 把 region×240 + aclopCompileAndExecute 50600 解读为"每次重编译 ~184 op"是**误读**——aclopCompileAndExecute 是 graph op 的正常执行入口（2.478s host 时间为执行开销），opCompile 与 aclop 计数完全相等（疑同事件双计数）
- **实证结果**：禁用 forward_chunk compile 后 RTF 0.32×3 vs 同日 champion 0.29/0.28/0.35 → +10.3% 劣化；eager-backend compile 产物在 NPU 执行效率高于裸 eager（dynamo graph 批提交或 aclop 路径优势），**编译类优化在 stage2 无收益路径**
