# 量化专项调研结论：MiniCPM-o stage0 权重量化（910C）

> 2026-08-19 调研 · 状态：**实施中**（前置 preflight 完成；W4A16 主轨实现中，见"执行状态"）
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

---

# 附二：执行状态（2026-08-19，实施中）

## 前置 preflight 结论（910C CANN 9.0.0 实测，quant_mxfp4_preflight.py）

1. **MXFP4（2a）被排除**：`npu_dynamic_mx_quant` 报
   `ascend910_93 does not support opType [DynamicMxQuant]`（OPP 无该 kernel）；
   fp16→fp4 `npu_dtype_cast` 同样失败（CastKernelNpuOpApi）；`npu_quant_matmul` fp4 模式因
   无法产出 fp4 张量而不可用。→ **fp4 家族在 910C CANN 9.0.0 全灭**（与 fp8 缺失同源，CANN 裁剪）。
2. **W4A16 per-group32 存活**：`npu_weight_quant_batchmatmul(antiquant_group_size=32, scale (K/32,N))`
   实测 L2 = 0.0958（= B1 per-group32 基准）。**关键签名**：2D scale 只能是 (1,N) per-channel 或
   (K/G,N) group-major（带 group_size 参数时 scale 必须 (K/G,N)，传 (N,K/G) 报错）。
3. **W4A16 线性 scheme 在本版本 vllm-ascend（0.19.1rc2）不存在**——registry 只有 ("W4A16","moe")。
   `_is_w4a16` 分发判定已在上游（无需 patch config），缺的只是 linear scheme 类。

## 已产出

| 文件 | 状态 |
|---|---|
| `vllm-ascend/quantization/methods/w4a16_linear.py` | 新增：AscendW4A16LinearMethod（per-group32, group-major scale, NZ weight，端到端 L2 0.096 验证） |
| `vllm-ascend/quantization/methods/__init__.py` | 注册 + `AscendW4A16LinearMethod` 加入 is_mx_quant_type（原因：per-group scale 需 input_dim=1 供 row-parallel TP 切分，method_adapters.py:120 门控） |
| `vllm-ascend/quantization/compressed_tensors_config.py` | 补 patch 0005 的 matched_target None 回落（本版本缺失，无此 fix 非匹配层 KeyError） |
| `submission/6_optimization/patches/0008-w4a16-linear-scheme.patch` | 上述 vllm-ascend 三处 diff（140 行） |
| `submission/6_optimization/quant_export_w4a16.py` | Q1 骨架 + per-group32 对称 RTN + 离线自检（L2 0.097 ✅） |
| `submission/6_optimization/quant_export_w8a8_static.py` | W8A8 静态导出（per-channel 对称 + 校准 input_scale 嵌入） |
| `submission/6_optimization/quant_calibrate_act.py` | 激活校准（Qwen2ForCausalLM 直载 + hook，64 条 zh 真实文本；已跑通产出 act_scales.json） |
| `submission/1_code/minicpmo_4_5_int4.yaml` / `_w8a8.yaml` | champion 拷贝 + stage0 `quantization: compressed-tensors` + 模型目录 |

导出验证：w4a16 ckpt 26G→12.8G（144 层量化），权重加载 + TP=2 切分（input_dim fix）已在服务中通过（stage 3/3 初始化成功）。

## 进行中

- **W4A16 编译图 kernel gap**：eager 下全 shape/format 探针通过，但 FULL_AND_PIECEWISE 编译图
  （ACL graph capture）下 `aclnnWeightQuantBatchMatmulV2` 报 `BinaryGetFunctionByEntry failed,
  tiling key 4613751299`（kernel 二进制缺失，107000）。正在二分：eager 跑通 → 图捕获路径问题
  （查 aclgraph 对 2D scale 算子的支持）；eager 也挂 → 算子参数/运行时差异。
- 候选修复路径（若确认图捕获问题）：stage0 编译配置对该算子豁免 / npu_quant_matmul 替代 /
  W8A8 静态轨（无此算子，native scheme 现成）。

## 教训

- 图捕获路径（ACL graph）下 op 行为 ≠ eager 探针：验证必须带真实编译配置。
- 本版本 vllm-ascend 的 `is_mx_quant_type` 名单语义 = "per-group scale 需 input_dim=1"，非 MX 家族专用。

---

# 附：加载时量化执行计划（2026-08-19，plan 审定稿）

## Context

赛题评分 RTF > TTFP > TTFT，champion = C63（stage0 TP=2，256/445/0.29）。stage0（8B Qwen2 thinker）decode 是 memory-bound（每 token 读全部 26G fp16 权重），权重量化是最确定的减速杠杆。用户要求**加载时量化**（区别于 E8 的推理时动态量化，已证伪）。本机（/root/code）无 NPU 运行时——代码在本仓库写，验证全部在 910C 服务机另行执行。

依据 `self_optimize/quantization_special_project.md`（2026-08-19，B1 实测）：

| 方案 | M=1 GEMM 加权 | 精度(L2) | 状态 |
|---|---|---|---|
| W8A8 静态 | -2.5%（≈0 净收益） | 0.009 优秀 | 全链路现成，需新增激活校准 |
| **W4A16** | **-23%（主指标最大收益）** | per-channel 0.156 / **per-group32 0.096** | 算子可用，vllm-ascend 有 w4a16 scheme |
| W8A16（Q1） | -9% | 0.009 | 910B 端到端证伪（反量化展开），不复用 |
| W4A4 int4×8 int32 打包 | - | 未测 | npu_quant_matmul int4 整数模式报错，不做 |
| **W4A4 MXFP4 原生**（新候选） | 未测 | 未测 | 仓库已有完整实现（见下），910C 未排除 |

官方 Q4_0（GGML block-32）WER 1.387 < F16 的 1.414 —— **int4 块量化精度官方背书可行**。两条线共享 Q1 导出脚本骨架（`submission/6_optimization/quant_export_w8a16.py`：分片遍历 + qkv/gate_up 融合 + index/config 重写，纯 RTN 确定性）。

**启用路径已确认**：deploy yaml stage 块支持平铺 `quantization:` 引擎参数（`vllm_omni/deploy/hunyuan_image3_ar.yaml:45` fp8 先例；`vllm_omni/config/omni_config.py:54` `_QuantizationEngineOverrides`）——stage0 加 `quantization: compressed-tensors` + 换 checkpoint 目录即可，**不改 vllm_omni 核心代码**。

**框架机制已核实（本仓库源码）**："加载时量化"的机制在框架里是现成的--各 Online LinearMethod 经 `LazyWeightMixin`（`int8_config.py:226-311`，meta 设备 + numel 计数触发）在权重载完瞬间执行 `process_weights_after_loading` 量化 + `replace_parameter`；Online（加载时量化原始 ckpt）与 Offline（载入已量化 ckpt）归一到同一 canonical 布局、共享 apply。E8 失败在 apply 端（每前向动态量化激活的算子开销），不是加载端；W8A8 静态要改造的正是 apply 端激活量化方式。

**新候选（本仓库源码发现）**：`mxfp4_config.py:177-343` 已有 **W4A4 MXFP4 原生路径**的完整实现（`NPUMxfp4LinearMethod` / `NPUMxfp4OnlineLinearMethod`）--权重 `float4_e2m1fn_x2` + per-32-group e8m0 scale，`npu_quant_matmul(x1_dtype/x2_dtype=float4_e2m1fn_x2)` 原生 GEMM、**零权重展开**，且自带加载时量化的 Online 版（`npu_dynamic_mx_quant` at load）。这与专项文档里实测报错的 "int4×8 int32 整数打包" 是**不同模式**；文档的 "W4A4 算子报错" 结论只覆盖了整数打包路径，MXFP4 浮点 4-bit 路径在 910C CANN 9.0.0 上未被排除。若 preflight 验证可用，它同时具备：int4 级别的显存/带宽收益（M=1 预期与 W4A16 同级或更好）+ 原生计算无展开 + 仓库实现现成（含 online 加载时量化）。

## 前置：910C 机上的格式锁定（第一步，0.5-1h）

两线共用。vllm-ascend 不在本机，以下必须在 910C 机读源码/实测后才能写死导出格式：

1. 读官方 vllm-ascend 的 `quantization/methods/w8a8_static.py` 和 `w4a16.py`（site-packages），锁定 checkpoint 期望格式：
   - W8A8：weight_scale 粒度（per-channel/per-tensor）、对称性、input_scale 粒度与存放方式（权重内嵌 or config）
   - W4A16：int4 权重存储形态（int32 打包 via `npu_convert_weight_to_int4pack`？）、per-group scale+offset 形状、group_size 支持值（32/64/128）
2. 读 `compressed_tensors_config.py` 确认分发条件（W8A8 自带、W4A16 分支已在上游；若 W8A16 缺口影响识别顺序，参考 patch 0005 模式补丁——patch 0005 目标仓库是 vllm-ascend，按 910C checklist 谨慎改）
3. 用现成 preflight 脚本核对算子语义：`7_runtime/exp/preflight/quant_q4_bench.py` / `quant_q4_precision.py`（在 910C 机的 vllm-omni 下）；导出脚本的量化/反量化数学必须与算子闭环一致（复用 Q1 的 `q1_antiquant_semantics.py` 验证模式）
4. **MXFP4 原生路径 preflight（新增）**：直接用仓库 `NPUMxfp4LinearMethod` 的算子组合在 910C 实测（stage0 真实 GEMM shape，M=1/M=4）：
   ```python
   w_q, w_s = torch_npu.npu_dynamic_mx_quant(weight)          # fp4 e2m1 + e8m0 scale
   x_q, x_s = torch_npu.npu_dynamic_mx_quant(x)
   out = torch_npu.npu_quant_matmul(x_q, w_q.t(), w_s_reshaped.transpose(0,1),
           scale_dtype=torch_npu.float8_e8m0fnu,
           x1_dtype=torch_npu.float4_e2m1fn_x2, x2_dtype=torch_npu.float4_e2m1fn_x2,
           pertoken_scale=x_s, pertoken_scale_dtype=torch_npu.float8_e8m0fnu,
           output_dtype=torch.float16, group_sizes=[1,1,32])
   ```
   参照 `mxfp4_config.py:263-296` 的调用形状（权重 inline 转置、scale (N,S/2,2)->(S/2,N,2)、bias float32）。同时测精度（随机权重 L2，对比 int8 的 0.009 / int4-group32 的 0.096 基准）。可用则 MXFP4 晋级为 int4 线首选（原生 GEMM 优于 W4A16 展开路径）

## Part 1：W8A8 静态（专项阶段 A，链路验证 + 高并发档价值）

### 新文件 1：`submission/6_optimization/quant_export_w8a8_static.py`
改造 Q1 脚本：
- 权重：int8 per-channel（对称或按 w8a8_static 要求），产出 `weight`(int8) + `weight_scale`；激活为静态时按该 scheme 约定补 `input_scale`（从校准产物读入）
- 保留：`llm.*` 前缀过滤、qkv 三段融合 / gate_up 融合、embed_tokens 与非 llm 前缀跳过、跨分片合并、index.json 重写、其余文件透传
- config.json `quantization_config`：`quant_method: compressed-tensors`，`config_groups` 里 weights **和 input_activations** 都描述为 INT8 静态（`dynamic: false`）——这是与 Q1 的唯一结构性差异
- 确定性：纯 RTN + 固定校准产物，逐位可复现

### 新文件 2：`submission/6_optimization/quant_calibrate_act.py`（新工作，Q1 没有）
- 加载 fp16 stage0，hook 各 Linear 输入，跑 N≈64-128 条**真实 Seed-TTS zh prompt**（固定样本清单，复现友好），统计各层激活 absmax → 静态 input_scale
- 设备无关（CPU/910C 均可跑），输出一份 scales 清单（json/safetensors）供导出脚本嵌入
- 粒度按前置步骤 1 锁定的 scheme 要求（预期 per-tensor）

### 配置与验证（910C 机执行）
- 新 yaml 变体（不动 champion）：`submission/1_code/minicpmo_4_5_w8a8.yaml`（champion 全量拷贝，stage0 块加 `quantization: compressed-tensors`、模型指向转换后 ckpt 目录）
- 验证顺序（铁律）：WER zh 32 冒烟（红线 ≤1.56）→ perf 32×1 两三轮 A/B（**诚实预期：主指标 <3%，低于 10ms/0.02 采纳门槛**）→ 64×4 / 128×8 矩阵档（真实价值区，预期 -25% 级）→ 达门槛才晋级全量四件套
- 附带收益：stage0 每 worker 13G→6.5G 显存富余

## Part 2：int4 线（主指标候选，无需校准）-- MXFP4 原生与 W4A16 双轨

preflight 结果决定主轨：**MXFP4 可用则走 2a（原生 GEMM 零展开，性能形态最优），不可用回退 2b（W4A16，B1 已实测 M=1 -23%）**。两轨共享精度缓解策略与验证流程。

### 2a. MXFP4 原生（W4A4，首选-if-可用）
- **实现基本零开发**：仓库 `NPUMxfp4OnlineLinearMethod`（`mxfp4_config.py:304`）就是加载时量化（`npu_dynamic_mx_quant` at load）+ 原生 fp4 GEMM；stage0 走 vllm 标准 LinearBase，启用 = checkpoint config 写 `quant_method: mxfp4`（经 `factory.py` 的 `_build_mxfp4` -> `DiffusionMXFP4Config`）或离线导出 fp4 ckpt
- 需确认：`DiffusionMXFP4Config.get_quant_method` 对非 diffusion 的 vllm 标准 Linear 层是否同样命中（`mxfp4_config.py:142`）；TP=2 下 fp4 权重 (N,K) 不预转置的切分行顺（column-parallel 切 N 无碍，row-parallel 切 K 时 32-group 边界整除性同 2b 检查）
- 风险：fp4 e2m1 精度（2bit 尾数）低于 int4-group32 -- L2 精度 preflight 是第一道闸
- 额外注意：910B/C 上 mxfp4/mxfp8 是否被 vllm-ascend 的平台能力检查拦截（`get_min_capability` 等）需实测

### 2b. W4A16（回退轨，B1 实测背书）

### 新文件 3：`submission/6_optimization/quant_export_w4a16.py`
- int4 **per-group(32) 非对称** RTN（GGML Q4_0 同形态：块 scale + offset），按前置步骤锁定的 vllm-ascend w4a16 格式打包
- 复用 Q1 骨架（融合、过滤、index/config 重写）；config：`num_bits: 4`，`strategy: group`，`group_size: 32`，`symmetric: false`
- 内置离线精度自检：随机权重 L2 对比（B1 基准 per-group32 ≈0.096，劣于此值报警）
- **TP=2 对齐检查**（关键风险）：row-parallel 层（o_proj/down_proj 按 K/in 维切）的 int4 打包组边界必须与切分对齐（24576/32=768 整除，理论 OK，导出后单测验证）

### 精度风险缓解（两轨共用）
- 官方 Q4_0 背书 + per-group32 双 scale（0.096）；若 WER zh 32 超线：用 compressed-tensors `config_groups` 多组 targets 正则做**敏感层回退**（qkv/lm_head 留 fp16 或 int8，其余 int4），embed_tokens/lm_head 本就被 Q1 模式跳过；MXFP4 线同理（mxfp4_dualscale 本身就支持 BF16 回退层混排，`factory.py:112-125`）
- WER 检查同红线：≤1.56（zh 全量历史 1.41-1.43，int4 余量约 0.13pp，偏紧——这是本线最大风险）

### 性能预期（诚实）
- W4A16：M=1 GEMM 加权 -23% 是真实主指标收益（stage0 TPOT 降 → handoff/decode 提速 → TTFP/RTF 改善）；**风险**：B1 只测了 M=1/M=4，prefill 大 M 下反量化展开路径可能拖慢 TTFT。A/B 判定看 RTF 净值 + TTFP 不劣化（评分规则），TTFT 允许适度波动
- MXFP4：无展开路径，prefill/decode 双端形态都优，但 910C 实测数据为零；且每前向激活 fp4 量化有算子开销（与 E8 同类风险，量级待 preflight）
- W4A16 无激活量化 → 零校准、确定性 RTN（复现最友好）；MXFP4 的 mx quant 数据驱动、同样无校准集依赖

## 执行顺序建议

1. 前置格式锁定 + **MXFP4 算子 preflight**（910C 机，0.5-1h）--preflight 结果直接决定 int4 主轨
2. int4 线先行（主指标收益大、无校准依赖）：MXFP4 可用走 2a（近零开发），否则 W4A16 导出脚本（2b）
3. W8A8 线随后（校准是新工作量；价值在高并发档与显存）
4. 任一线达门槛 → 按 SELF_OPT_PROTOCOL 晋级流程（全量四件套）

## 修改/新增文件清单

| 文件 | 动作 |
|---|---|
| `submission/6_optimization/quant_export_w8a8_static.py` | 新增（改造 Q1 脚本） |
| `submission/6_optimization/quant_calibrate_act.py` | 新增 |
| `submission/6_optimization/quant_export_w4a16.py` | 新增（改造 Q1 脚本；仅 2b 轨需要） |
| `submission/1_code/minicpmo_4_5_w8a8.yaml` / `_int4.yaml` | 新增（champion 拷贝 + stage0 quantization 参数；int4 轨按 preflight 结果定 mxfp4 或 w4a16） |
| `self_optimize/quantization_special_project.md` | 追加执行状态 |
| vllm_omni 核心代码 | **预期零改动**（compressed-tensors 为 vllm 内置 + vllm-ascend 自带分发；如需 patch 按 0005 留档模式） |

## 验证（全部在 910C 机，按 SELF_OPT_PROTOCOL）

1. **算子语义闭环**：导出脚本量化数学 vs `npu_weight_quant_batchmatmul`/`npu_quant_matmul` 反量化结果逐位对比（q1_antiquant_semantics 模式）
2. **WER zh 32 冒烟**：≤1.56 红线，超线即停（int4 线先试敏感层回退再判死）
3. **perf 32×1 A/B**：同服务会话 2-3 轮均值 vs C63 champion 锚点（256/445/0.29）；采纳门槛 = 任一指标 TTFT/TTFP ≥10ms 或 RTF ≥0.02 且其他不劣化
4. **高并发档**：64×4 / 128×8（W8A8 线的主战场）
5. **晋级**：全量四件套（WER zh 2020 / ASV 1088 / DO 1196 / VideoMME 2700）


## 验证结果（2026-08-20，int4 线判定）

**W4A16 端到端验证（修复 ND 后）**：
- 服务启动 ✅（编译图 profile 通过——ND 修复生效）
- WER zh 32 = **1.0% 达标**（红线 ≤1.56）——int4 per-group32 精度无问题 ✅
- perf 32×1 两轮均值：TTFT 312ms（+22%）/ TTFP 500ms（+12%）/ **RTF 0.31（+7%）vs C63 锚点 256/445/0.29 → 判定不采纳**（RTF 优先规则，稳定劣化非噪声）

**根因**（B1 的 -23% 没兑现）：
1. 910C 的 WeightQuantBatchMatmulV2 fp16/int4 内核只覆盖 weight format 29/3；
   `maybe_trans_nz` 产生的是 FRACTAL_NZ(2)（tiling key 4613751299 无内核）→ 被迫用 ND 权重，
   ND 是慢路径（NZ 才是 cube 偏好布局）——GEMM 单点收益被抵消
2. W4A16 反量化路径 prefill 大 M 有额外开销 → TTFT/TTFP 劣化
3. 端到端 stage0 TPOT 占比有限，单点 GEMM 收益稀释

**结论**：W4A16 在 910C CANN 9.0.0 上是死路（NZ 内核缺失 + ND 慢），除非 CANN 升级补内核。
patch 0008 保留（可复现 + 未来 CANN 版本可用），int4 yaml 保留但**不启用**。

## 根因修正（2026-08-20，机器实测 quant_int4_vs_fp16nz_check.py）

"B1 的 -23% 没兑现"的说法不准确——**B1 的 -23% 本身就是测量假象**，且有两层：

1. **B1 基线错**：B1（quant_q4_bench.py:51）的 fp16 基线是 ND 权重 `torch.matmul`；champion 实际
   跑 FRACTAL_NZ 权重（vllm-ascend 全局换 Ascend*Linear + `AscendUnquantizedLinearMethod`
   process_weights_after_loading → maybe_trans_nz，utils.py:746 / ops/linear.py:87）。实测
   ND 比 NZ 慢 30-55%（qkv M=1: 0.041 vs 0.030ms；gate_up: 0.074 vs 0.052ms）。
2. **B1 调用签名错**：B1 的 w4a16 是 **per-channel** scale（`[N]`，无 antiquant_group_size）；
   scheme 实际是 **per-group32 group-major** scale + `antiquant_group_size=32`（per-channel L2
   0.156 精度不达标才选的 group32）。同 shape 实测 per-channel 0.040ms vs per-group32 0.099ms
   ——per-group 分支慢 2.5×。

**真实对比（i4g32 vs fp16-NZ，4 shape × M∈{1,4,64,256}，全实测）**：比值 1.1×~5.5×，
**没有一个格子赢**（qkv 1.3-3.3×、gate_up 1.1-5.5×、down 2.4-4.2×、o_proj 1.5-2.5×）。
机制：WeightQuantBatchMatmulV2 是展开路径（算子内 int4→fp16 再 GEMM），计算量与 fp16 相同，
还要付 int8 读 + fp16 展开写/读 + scale/offset 加载 + ND tiling——M=1 纯倒挂。
e2e +7% RTF 全部来自 TTFT/TTFP（+12~22%，prefill/首包路径），与上表自洽。

**顺带发现**：w8a8 静态（npu_quant_matmul int8，无展开）是唯一有真实收益的量化——down/o_proj
M=1 快 1.5-1.9×、M≥256 全 shape 快 1.3-1.6×，但 qkv/gate_up M=1 输 1.1-1.5× → 32×1 评分档
加权约持平（符合原诚实预期 <3%），仍过不了采纳门槛；价值只在并发档。

**修正后结论**：不是"ND 内核慢"单因，而是本机 CANN 9.0.0 上**没有任何量化内核能赢 NZ-fp16
真基线**（int4 展开慢 + ND，w8a8 只在 M≥64 赢）。量化线收尾结论不变；重启条件 = 官方环境
CANN ≥9.1（NZ int4 / fp8 内核）。

## W8A8 验证结果（2026-08-20，用户追问"能否不反量化"后补跑）

**结论：W8A8 动态（per-token 激活量化）通过冒烟，性能三项全面优于 C63 锚点——首个量化候选达标。**

| 配置 | WER zh 32 | TTFT | TTFP | RTF |
|---|---|---|---|---|
| **W8A8 动态**（图模式+关融合） | **0.86% ✅** | **230.6** | **416.9** | **0.28** |
| W8A8 静态（per-tensor 校准） | 3.44% ❌ | — | — | — |
| int4 W4A16（已判死） | 1.0% | 312 | 500 | 0.31 |
| C63 champion 锚点 | ~1.4% | 256 | 445 | 0.29 |

perf 32×1 三轮均值 230.6/416.9/0.28 vs 锚点 256/445/0.29 → TTFT -25ms、TTFP -28ms（均超 10ms 采纳门槛）、RTF -0.01 不劣化，方向三轮一致 → **按 perf-adoption 门槛采纳候选**；晋级仍须全量四件套。

**踩坑记录（按序，全部已修）**：
1. 导出脚本 act_scales 命名空间错位（校准是 thinker.llm.*，导出查 llm.*）→ 补前缀
2. 导出 OOM（fp32 中间量 13G/片 + 服务 page cache，32G 容器 exit 137）→ fp16 中间量 + 修 (N,1,1) 形状 bug
3. config format "int8" 不在 is_activation_quantization_format 名单 → 改 "int-quantized"
4. `npu_quant_matmul` scale 拒 fp16（只收 UINT64/BF16/INT64/FLOAT）→ static 版 deq_scale 提 fp32（w8a8_static.py）
5. **静态版图模式出乱码：根因 = `npu_add_rms_norm_quant` 融合 op 在 ACL 图捕获下静默算错**（eager 正确）→ yaml 关 `fuse_norm_quant: false`（融合 pass 的 bf16 offset 修复保留，eager 侧正确）
6. 动态 scheme `apply()` 传 fp16 weight_scale 报同样 dtype 错（上游 bf16-only 遗留：process 备了 `weight_scale_fp32` 没用）→ apply 改用 fp32（w8a8_dynamic.py）
7. **静态激活量化（per-tensor 校准）精度劣化 3.44%**：截断/离群（校准 64 条 absmax 覆盖不足）→ 换 per-token 动态量化（`npu_dynamic_quant`，运行时 scale 无截断）→ 0.86%

**关键发现**：910C CANN 9.0.0 的图捕获路径对量化算子不可信（int4 tiling key 缺失、融合 op 静默算错），
w8a8 部署必须 `fuse_norm_quant: false`。

**产物**：`quant_export_w8a8_dynamic.py`（新增）、`quant_export_w8a8_static.py`（修复）、
w8a8 相关 vllm-ascend 三处 dtype 修复（deq_scale fp32 / dynamic apply fp32 / 融合 offset bf16）、
ckpt `MiniCPM-o-4_5-w8a8-dyn`（13G）、yaml `7_runtime/exp/minicpmo_4_5_w8a8_dyn_nofusion.yaml`。

**待办**：晋级须全量四件套（WER zh 2020 + ASV 1088 + DO 1196 + VideoMME 2700），约 4-5h，待用户确认。

## 待办
- [ ] W8A8 线验证（链路完整性；预期主指标 <3% 不采纳，价值在高并发档）——ckpt/校准/yaml 已就绪
- [ ] 若 W8A8 也劣化 → 量化线整体收尾，结论入库

## 全量验证结果（2026-08-20，最终结论：量化线整体收尾）

| 评测 | 全量结果 | 准入 | 判定 |
|---|---|---|---|
| **WER zh 2020** | **1.79%** | ≤1.56% | ❌ 超线 |
| ASV 1088（SIM） | 0.8497 | ≥0.689 | ✅（历史 0.8524，本机首次跑通全量） |
| DO 1196 / VideoMME 2700 / demo | 中止 | — | 用户决定：WER 超线，剩余无必要 |

**W8A8 动态（per-token int8 激活）冒烟 32 条 0.86% 漂亮，但全量 2020 = 1.79%（champion 1.41-1.43）——
int8 激活量化带来真实 +0.38pp 劣化，超线 0.23pp**。又是 32 条假象（[[wer-full-validation-rule]]）。

**最终结论：910C CANN 9.0.0 上 stage0 权重量化线整体收尾**：
- int4（W4A16）——算子展开 + ND 慢，端到端 +7% 劣化（早期判死）
- W8A8 静态——校准截断，32 条 WER 3.44%（判死）
- W8A8 动态——性能最优（32×1 三轮 230.6/416.9/0.28 全面优于 C63 锚点，TTFT/TTFP 超采纳门槛），
  但全量 WER 1.79% 超线 0.23pp，精度不过关（判死）
- 量化精度劣化的根因：stage0 是 8B 文本 LLM，decode 每 token 权重全读 → 量化收益大，但 WER 准入余量
  只有 0.13-0.15pp（1.41 vs 1.56），任何激活量化噪声都越界；权重侧 int4/int8 本身误差小（ASV 0.8497
  证明音色路径无损），瓶颈在激活量化对文本生成的干扰

**量化不动 stage0 的结论成立；未来唯一机会 = 官方环境 CANN ≥9.1 的 fp8（e4m3 尾数 3bit 精度高于 int8
静态激活，且无激活量化开销）或更高 WER 余量时重试。** 产物保留（脚本/ckpt/yaml 均可复现），不再投入。

## 敏感层回退 W8A8(skip8) —— 2026-08-20 验证中

**动机**: 全量化 W8A8 动态 WER zh 1.79% 超线(1.56%),但性能最好(-25/-28ms)。回退激活量化噪声最大的首4+末4 层(0-3,32-35)保持 fp16,期望精度回线 + 性能保留大部分。

**实现**(quant_export_w8a8_dynamic.py `--skip-layers 0-3,32-35`):
- 敏感层 q/k/v、gate/up 不合并不量化,原样拷贝;不写 weight_scale/weight_offset
- config targets 枚举非 skip 层正则 `re:^thinker\.llm\.model\.layers\.(4|...|31)\.`
  → find_matched_target 等值/正则匹配完整层路径,不匹配层保持 UnquantizedLinearMethod(原生 fp16 Linear),加载即原生权重
- ckpt: MiniCPM-o-4_5-w8a8-skip8(112 量化层 + 8 回退层);yaml: minicpmo_4_5_w8a8_skip8.yaml

**结果**:
- 冒烟 32 条 WER: **0.48%**,无 failed(功能正常,混合结构加载 OK)
- perf 32×1: **TTFT 240.28ms / TTFP 431.82ms / RTF 0.29**(champion C63: 256 / 445 / 0.29)
  - TTFT -15.7ms、TTFP -13.2ms 双超 10ms 采纳门槛,RTF 持平不劣化 → perf 达标
- 全量 zh 2020: 进行中

**skip10 perf 3 轮修正(2026-08-20)**: 重启后首轮偏高(TTFT 250.06 / TTFP 441.57),热轮稳定 TTFT 240.47/239.91 / TTFP 426.98/426.69 / RTF 0.285 —— 与 skip8 热数据(240.28/431.82)基本一致。**教训: 服务重启后首 1-2 轮 perf 有 ~10ms 冷启动假象**(skip10 首批两轮 250.77/252.60 恰是冷窗口;skip8 当时 perf 前跑了 12 分钟 WER 冒烟属热数据)。层 4/31 回退真实成本 ≈ 0,多回退 2 层不掉性能。
- skip10 vs champion(256.1/445.4/0.29): TTFT -16ms / TTFP -18.6ms / RTF -0.005(不劣化) → **perf 达标**(双超 10ms 门槛)
- 全量 zh 2020: 进行中

## ═══════ 夜间交接快照(2026-08-20 20:00)═══════

### 当前服务状态
- **运行中**: skip10 服务(pid 见 `ps aux | grep "vllm serve"`),yaml = `7_runtime/exp/minicpmo_4_5_w8a8_skip10.yaml`
- 服务日志 = `7_runtime/minicpmo_server.log`
- **skip14 ckpt 已导出完成**: `/workspace/local_models/MiniCPM-o-4_5-w8a8-skip14`(量化层 5-28,22 层;回退 0-6,29-35 共 14 层)
- **skip14 yaml 未建**(下一步 cp skip10 yaml 改 model 路径)

### skip10 全量四件套(已完成)
| 指标 | skip10 | champion C63 | 准入 | 判定 |
|---|---|---|---|---|
| WER zh 2020 | 1.52% | 1.43% | ≤1.56 | ✓ 余量 0.04 |
| ASV SIM | 0.8492 | 0.8487 | ≥0.689 | ✓ |
| DO 1196 | 75.92% | 78.43% | ≥77.5 | ❌ -2.51pp |
| VideoMME 2700 | 68.33% | 69.59% | ≥67.0 | ✓ 余量 1.33 |
| perf 32×1(热轮) | TTFT 240.2 / TTFP 426.8 / RTF 0.285 | 256.1/445.4/0.29 | -16/-18.6ms | ✓ 双超10ms门槛 |

### 剩余任务(唯一红项 = DO)
- 思路: 回退单调性 → skip14 DO 必然 ≥75.92,预计接近 champion 78.43。协议 SELF_OPT_PROTOCOL §5: DO 128 条只作粗回归方向检测,不可当绝对口径;精确必须全量 1196(~1.5h)
- **下一步执行清单**(全部在 2_configs 下,服务需先重启为 skip14):
  1. `cp exp/minicpmo_4_5_w8a8_skip10.yaml exp/minicpmo_4_5_w8a8_skip14.yaml` + 改 model 路径 + 注释
  2. 杀服务 + `VLLM_WORKER_MULTIPROC_METHOD=spawn vllm serve /workspace/local_models/MiniCPM-o-4_5-w8a8-skip14 --omni --served-model-name openbmb/MiniCPM-o-4_5 --trust-remote-code --dtype float16 --deploy-config <skip14 yaml绝对路径> --stage-init-timeout 900 --init-timeout 1200 --host 0.0.0.0 --port 8091 --allowed-local-media-path 7_runtime/media`
  3. 启动完成(~7min)后 `bash eval_daily_omni.sh 128`(方向确认,~20min)
  4. 方向对 → `bash eval_daily_omni.sh 1196`(DO 全量,~1.5h)
  5. DO 过 → `bash eval_seed_tts_zh.sh`(WER 全量复核,~1.5h;skip10 是 1.52,skip14 应更安全)
  6. DO 过 + WER 过 → perf 3 轮热确认(TTFT/TTFP 是否仍超 10ms 门槛;skip14 预计 -10ms 附近,边缘)→ 决定采纳
- **失败兜底**: skip14 若 DO 仍不过(不可能,单调)→ 继续 skip18(0-8,27-35)

### 关键教训(冷启动假象)
- 服务重启后**首 1-2 轮 perf 偏高 ~10ms**(TTFT 250 vs 240),热轮才是真值;perf 前应先跑一轮 WER/DO 冒烟(~12min)当热场
- skip8→skip10 的"多回退2层掉10ms"是假象,真实成本 ≈0

### 产物清单
- 导出脚本: `submission/6_optimization/quant_export_w8a8_dynamic.py`(`--skip-layers` 参数,默认 0-3,32-35)
- ckpt: `local_models/MiniCPM-o-4_5-w8a8-{skip8,skip10,skip14}`(skip14 最新)
- yaml: `7_runtime/exp/minicpmo_4_5_w8a8_{dyn_nofusion,skip8,skip10}.yaml`(skip14 未建)
- vllm-ascend 修复(已生效): w8a8_dynamic.py / w8a8_static.py / norm_quant_fusion_pass.py(fuse_norm_quant 需 false)
- 日志: `7_runtime/results/{seed_tts_zh_full,daily_omni_163737,videomme_official_173608,seed_tts_asv_161105}.log`
- 后台队列: `7_runtime/results/full_eval_queue.log`

**DO 128 条基线(skip10,2026-08-20 19:57)**: 93/128 = 72.66%(全量 75.92%)。下一条 skip12 的 128 条同批对比可看方向。
**skip12 已备好(2026-08-20)**: ckpt `local_models/MiniCPM-o-4_5-w8a8-skip12`(量化 6-29 共 24 层,回退 0-5,30-35 共 12 层)+ yaml `7_runtime/exp/minicpmo_4_5_w8a8_skip12.yaml`。用户决策:从 skip12 继续优化 DO。
