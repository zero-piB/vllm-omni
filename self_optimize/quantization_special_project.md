# 量化专项调研结论：MiniCPM-o stage0 权重量化（910C）

> 2026-08-19 调研 · 状态：**专项规划**（本机算子层现状已实测闭环，实施待立项）
> 目标：stage0（8B thinker）decode 是 memory-bound（每 token 读全部权重）→ 权重量化减半/减 1/4 权重体积 → decode 加速

---

## 1. 收益模型（为什么量化值得做）

- stage0 8B dense decode：每 token 必须读全部权重（26G fp16）过 GEMM → **memory-bound**
- 量化减半权重（int8/fp8）→ decode 理论上限 ~1.5-1.8x
- **TP=2 后依然成立**：每 worker 读自己的 13G 分片，量化后 6.5G，带宽收益同样生效
- 关键前提：**权重"小着算"**（原生量化 GEMM），不能有 fp16 展开中间

## 2. 方案全景对比

| 方案 | 机制 | 权重展开 | 反量化位置 | 精度（实测 L2 误差）| 本机状态 |
|---|---|---|---|---|---|
| **W8A16**（Q1 已试）| int8 存 + 算子内展开 fp16 GEMM | **有**（每 token 每层）| 计算前 | int8: 0.009（1%）| ❌ 910B F25 端到端证伪（TPOT +10%）；910C 单算子加权 -9% |
| **W8A8 静态**（推荐本机）| int8 激活（静态 scale）+ int8 权重 → `npu_quant_matmul` 原生 int8×int8 | **无** | **仅输出**（deq_scale，[M,N] 级）✅ | int8: 0.009（1%）| ✅ 全链路现成（scheme + 分发 + 算子），910C 可用 |
| **W4A16**（官方 Q4_0 同款形态）| int4 存 + 反量化 | 有 | 计算前 | int4 per-channel: 0.156 / per-group32: 0.096 | 性能最好（GEMM -23%），精度风险大（官方 Q4_0 用 GGML block 量化达标，vllm-ascend w4a16 需 per-group 双 scale） |
| **W4A4 原生**（npu_quant_matmul int32 模式）| 激活+权重都 int4 打包 | 无 | 仅输出 | 未测（精度理论更差）| ❌ 实测算子报错（QuantMatmulKernel shape 约束）|
| **FP8 原生**（910C 硬件）| 权重 fp8 e4m3 存 + 原生 fp8 GEMM | 无 | 仅输出 | 未测（e4m3 3bit 尾数，动态范围大）| ❌ **本机 CANN 9.0.0 全方位无 fp8 算子**（cast/matmul/aclnn 三层全空）→ 官方环境保留 |
| **官方 Q4_0**（llama.cpp 基线）| GGML block-32 量化 + 原生量化 GEMM | 无 | 仅输出 | **WER 1.387 达标（< F16 的 1.414！）**| 子赛道 A 口径，两赛道共用精度准入表 → **量化精度官方背书可行** |

## 3. 算子现状（910C · CANN 9.0.0 实测）

| 算子/API | 语义 | 状态 |
|---|---|---|
| `torch_npu.npu_weight_quant_batchmatmul` | W8A16/W4A16 反量化 matmul（scale fp16/bf16/int64）| ✅ 可用（慢：展开路径）|
| `torch_npu.npu_quant_matmul` | **原生量化 matmul**：int8×int8；int32(=int4×8 打包) 模式 | ✅ int8 可用；❌ int4 模式报错（shape 约束）|
| `torch_npu.npu_convert_weight_to_int4pack` | int4 打包工具（int32 承载 8 个 int4，交叠排放）| ✅ 可用 |
| `torch_npu.npu_trans_quant_param` | scale 转 int64 格式 | ✅ 可用 |
| fp16→fp8 cast（`CopyKernelOpApi`/`aclnnInplaceNormal`）| FP8 转换 | ❌ 报错（NPU function error）|
| aclnn Fp8Matmul / torch_npu fp8 matmul | FP8 GEMM | ❌ 不存在 |
| vllm-ascend `fp8.py` | FP8 支持 | 仅 **MoE**（fused_moe w13/w2），dense 无适配 |
| CANN `static_kernel/` 目录 | static kernel 二进制索引 | ❌ 缺失（910C OPP 裁剪，enable_static_kernel 静默 fallback —— F42）|

**核心结论**：910C 本机量化算子 = **int8 家族完整（w8a16/w8a8），int4 部分可用（打包工具在、matmul 报错），fp8 零支持**。CANN 9.0.0 对 910C（ascend910_93）是部分裁剪支持（fp8/static_kernel 均缺）。

## 4. B1 性能实测（910C · stage0 真实 GEMM shape · 50 次均值）

### 4.1 各方案 vs fp16（相对倍数，<1 为快）

| GEMM | M=1 w8a16 | M=1 w8a8 | M=1 w4a16 | M=4 w8a8 |
|---|---|---|---|---|
| qkv (4096×12288) | 1.32 | 1.62 | 1.04 | 0.86 |
| gate_up (4096×24576) | 0.70 | 0.61 | 0.72 | 0.65 |
| down (24576×4096) | 0.64 | 0.59 | 0.66 | 0.58 |
| o_proj (12288×4096) | 1.51 | 1.81 | 0.88 | 0.78 |
| **加权合计** | **0.91（-9%）** | **0.975（-2.5%）** | **0.77（-23%）** | **~0.75（-25%）** |

关键规律：
- **大 M（batch≥4）量化收益显著**（-25~-47%）：高并发档（64×4/128×8）是真受益场景
- **小 M（M=1，评分主指标 32×1）收益微薄**：w8a16 -9%（端到端 <3%）、w8a8 -2.5%（加激活量化开销后 ≈0）
- 小 GEMM（qkv/o_proj，N 小）在 M=1 时算子开销倒挂

### 4.2 精度实测（随机权重模拟，相对 fp16 输出 L2 误差）

| 方案 | 误差 |
|---|---|
| int8 per-channel | 0.009（1%，优秀）|
| int4 per-channel | 0.156（16%）|
| int4 per-group32 | 0.096（10%）|

## 5. 结论与判定

1. **量化本身可行**（官方 Q4_0 精度达标背书）；**F25（w8a16 慢）是实现问题**（反量化展开路径），非量化理论失败
2. **本机唯一可落地的原生路径 = W8A8 静态**（int8 算子全现成：scheme `w8a8_static.py` + 分发 `compressed_tensors_config.py` 自带 + `npu_quant_matmul` 原生计算，**权重零展开、只输出反量化**）
3. **主指标（32×1）量化收益预期 <3%**（低于采纳门槛）—— 量化的真实价值在高并发档（-25~-47%）与显存富余（stage0 13G→6.5G/worker）
4. **fp8 是形态最优解**（权重 fp8 + 激活 fp16 混合，无激活量化开销），本机 CANN 9.0.0 不支持 → **官方环境（CANN ≥9.1）保留**

## 6. 专项规划（后续大专项）

### 阶段 A：W8A8 静态落地（本机，4-6h）—— 决策待定
- [ ] 权重转换：Q1 脚本改造（int8 per-channel + deq_scale，TP 切分对齐）
- [ ] **激活校准**：跑 N 条真实样本收集各层激活范围 → input_scale（新工作，Q1 无）
- [ ] 模型 config：quantization_config 描述 input_activations
- [ ] 启动 `--quantization compressed-tensors`（框架参数，vllm 内置）
- [ ] WER zh 32 冒烟（红线）→ 达标则 perf 32×1 A/B
- 预期：主指标收益低（诚实提示），主要价值 = 验证链路 + 高并发档

### 阶段 B：fp8 官方环境（官方 CANN ≥9.1 时）
- [ ] 权重转换脚本（fp16 → fp8 e4m3，per-tensor scale；离线 CPU cast 即可，规避设备 cast 缺失）
- [ ] `--quantization fp8` 标准参数（若 vllm-ascend dense fp8 适配在官方版本存在）
- [ ] 无适配则写 custom scheme（aclnnFp8Matmul 或 hiF8 kernel）
- 收益预期：最优形态（无激活量化开销），TP=2 + fp8 = 每 worker 3.25G 权重

### 阶段 C：自研 kernel（cann-samples，天级，仅当 A/B 都不满足时）
- [ ] B2：`quant_grouped_matmul_hif8` 910C arch 验证（样例要求 ARCH 3510，910C 待确认）
- [ ] dense 改造（group_num=1）+ 注册 custom op（参考 `_cann_ops_custom/custom_transformer` 先例）
- [ ] custom scheme 接入 vllm
- 收益：可同时解决 fp8 原生 + 小 M 倒挂问题

### 关键文件
- 转换脚本：`submission/6_optimization/quant_export_w8a16.py`（Q1，可改）
- 910B patch：`submission/6_optimization/patches/0005-q1-w8a16-dispatch.patch`（910C 新版本只补 W8A16 分支，W8A8 自带）
- 方案实现：vllm-ascend `quantization/methods/w8a8_static.py` + `compressed_tensors_config.py`（W8A8 分发已在）
- 样例：cann-samples `quant_grouped_matmul_hif8`（3510）
- bench 脚本：`7_runtime/exp/preflight/quant_gemm_bench.py` / `quant_q4_bench.py` / `quant_q4_precision.py`

### 台账关联
- F25（w8a16 910B 端到端证伪）→ 定性为"反量化实现问题"（本调研修正）
- Q1-e2e（rejected）→ 产物保留，供 w8a8 转换改造
- F42（static kernel 910C 缺失）→ 与 fp8 缺失同源（CANN 910C 裁剪）
