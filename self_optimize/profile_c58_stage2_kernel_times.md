# C58/C59 stage2 kernel 时间档案（2026-08-17 采集）

> 用途：用户后续自行分析。原始数据在 `submission/7_runtime/profile/`，本文为提取摘要 + 方法。

## 采集方法（可复现）

1. 实验 yaml：`submission/7_runtime/exp/PROF.yaml`（champion 配置副本 + `stages[2].engine_args.profiler_config: {profiler: torch, torch_profiler_dir: /workspace/submission/7_runtime/profile, torch_profiler_with_stack: False}`）
   - **注意**：profiler 端点条件注册——`profiler_config` 必须在 **stage 的 engine_args** 里（api_server.py:206 `_should_enable_profiler_endpoints`），放 yaml 顶层不生效（404）
2. 服务启动：`VLLM_CUSTOM_SCOPES_FOR_PROFILING=1` 必须在 worker 派生前 export（vllm/envs.py:247 默认 False）
3. 采集流程：`POST /start_profile {"stages":[2]}`（单阶段！同物理 NPU 多 profiler 会丢 device task）→ 1 条 seed-tts 请求（`--num-prompts 1 --num-warmup 0`）→ `POST /stop_profile`（**长超时 ≥600s**，stop 是同步重操作）
4. 验收：`ASCEND_PROFILER_OUTPUT/` 下 kernel_details.csv / step_trace_time.csv / trace_view.json 齐全；FRAMEWORK/torch.op_mark 存在（VLLM_CUSTOM_SCOPES 生效标志）

## 原始数据路径

| 轮次 | 目录 | 说明 |
|---|---|---|
| 轮 1 | `profile/stage2_rank0/298846aeccdd_2190063_20260817142150054_ascend_pt/` | 无 patch（14:21:50）|
| 轮 2 | `profile/stage2_rank0/298846aeccdd_2246551_20260817155312444_ascend_pt/` | weight_norm patch（15:53:12）|

kernel 明细：两目录下 `ASCEND_PROFILER_OUTPUT/kernel_details.csv`（43380 行）
窗口口径：`ASCEND_PROFILER_OUTPUT/step_trace_time.csv`
自定义范围（vLLM scope）：`FRAMEWORK/torch.op_mark`（二进制，配 ascend_pytorch_profiler_0.db 解析）

## 轮 1 kernel 时间汇总（1 条请求，无 patch）

总 kernel **43380**，累计执行 **400.9 ms**（≈ step_trace Computing 400878us ✓ 对上了）

### Top 25（按累计耗时）

| 累计 | 次数 | max | kernel |
|---|---|---|---|
| 57.4 ms | 891 | 77.1us | trans_TransData_7 |
| 33.0 ms | 2838 | 25.7us | aclnnLayerNormWithImplMode_LayerNormV3 |
| 32.0 ms | 4556 | 25.8us | aclnnMul_MulAiCore_Mul |
| 30.1 ms | 3246 | 48.0us | aclnnAddmm_MatMulCommon_MatMulV2 |
| 23.1 ms | 3063 | 233.6us | aclnnCat_ConcatD_ConcatD |
| 23.0 ms | 4203 | 13.6us | aclnnAdd_AddAiCore_Add |
| 17.1 ms | 1476 | 18.6us | aclnnCat_TransposeAiCore_Transpose |
| 16.6 ms | 1297 | 18.6us | aclnnLayerNormWithImplMode_TransposeAiCore_Transpose |
| 15.7 ms | 2694 | 156.5us | aclnnCat_SliceAiCore_Slice |
| 13.1 ms | 432 | 54.4us | aclnnFlashAttentionScore（DiT attention，仅 54us/次）|
| 9.9 ms | 891 | 15.9us | trans_TransData_8 |
| 9.6 ms | 8 | 1586.1us | Conv2DTranspose28（HiFT 上采样，单次 1.6ms！）|
| 8.5 ms | 656 | 25.6us | aclnnNorm_LpNormV2 |
| 8.2 ms | 891 | 20.5us | Conv2D3（HiFT conv，执行仅 20.5us）|
| 7.1 ms | 600 | 18.7us | aclnnInplaceCopy_TransposeAiCore_Transpose |
| 6.9 ms | 891 | 15.3us | trans_TransData_6 |
| 6.2 ms | 584 | 15.2us | aclnnPowTensorScalar |
| 5.9 ms | 601 | 16.9us | aclnnSin |
| 5.4 ms | 439 | 18.5us | aclnnMul_TransposeAiCore_Transpose |
| 3.9 ms | 331 | 24.1us | aclnnMatmul_TransposeAiCore_Transpose |
| 3.8 ms | 444 | 131.1us | aclnnInplaceCopy_TensorMove |
| 3.1 ms | 2646 | 1.7us | aclnnLayerNormWithImplMode（小）|
| 3.0 ms | 440 | 10.3us | aclnnGelu |
| 2.8 ms | 432 | 10.7us | aclnnMish |
| 2.5 ms | 1904 | 1.9us | aclnnAdds |

### 核心类型

| 类型 | kernel 数 | 累计 |
|---|---|---|
| AI_VECTOR_CORE | 36957 | **318.5 ms（79%）** |
| AI_CORE | 5246 | 59.3 ms（15%）|
| MIX_AIC | 455 | 14.1 ms |
| MIX_AIV | 698 | 8.8 ms |
| DSA_SQE | 24 | 0.2 ms |

### TransData 家族合计

**100.2 ms（25%）across 4749 calls**（trans_TransData_7 57.4ms 最大）

## 时间分布发现

- busy 段分散（20s/38-49s/50-51s/65s）——50-53s 为主窗口（273ms），其余为 prewarm/清理段
- 主体段构成：elementwise+搬运家族 ≈ **136ms（50%）**，Matmul 仅 18.7ms（7%）
- trans_TransData_7 594 次集中在请求前 1.16s（单簇），后续块不再出现（疑一次性）——但 weight_norm patch 后 trans 未减（轮 2：1043 次/58.2ms，**总 TransData 101.1ms 不变**）→ 转换与权重静态性无关
- 时序模式：`trans(66us) → Conv2D3(6.7us) → trans`——转换是 conv 执行的 10 倍

## 窗口口径（step_trace_time.csv）

- Stage 总窗口 66896796us（66.9s，含请求前后空闲）
- Computing 400878us（400.9ms）✓ 与 kernel 累计一致
- Preparing 882967us（883ms，host 侧）
- Free 66495918us（99.4%——窗口含大量等待）

## C59 证伪结论（F38）

1. 假设 weight_norm → remove_weight_norm：A/B perf 无差异（TTFP 745.7 vs 750）+ profile 复采 trans 未减 → 证伪
2. 假设 FRACTAL_NZ 预转换：`npu_format_cast` 被平台禁（`allow_internel_format=False`）→ 不可行
3. 微基准：Conv1d(512,512,3) 每调用 67us 稳定含转换，无缓存机制 → trans 是 aclnnConv2d **固有 layout 转换**，用户侧无缓解
4. 附带：Conv2DTranspose28 单次 1.6ms × 8 = 9.6ms（HiFT 上采样）——未深挖，可作后续分析点

## 未深挖的观察点（供后续分析）

1. **Conv2DTranspose28**：8 次 × 1.59ms 单次——HiFT 上采样 convT 为什么这么贵？（其他 conv 20us 级）
2. **aclnnCat 家族** 23.1+17.1+15.7 = 56ms——ConcatD/Transpose/Slice 三件套，DiT/CFG 的 k_cache 路径
3. **aclnnAddmm 30.1ms × 3246**（9.3us/次）——小 GEMM 发射密集，QKV 融合（F13 已证伪 cat 路径）或可另想
4. **AI_VECTOR_CORE 79%**——elementwise 主导平台特性（F29 同族），融合算子（C24）收益边缘
5. **Preparing 883ms**（host 侧）——未被分析，可能含可挖的 host 开销
