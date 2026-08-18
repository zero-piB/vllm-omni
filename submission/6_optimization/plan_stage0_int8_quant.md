# 方案：stage0（Qwen2-8B thinker）W8A16 权重量化

> 状态：**方案待审**（2026-08-16 v2，动态量化→w8a16 修正）。用户审阅后决定是否实施。
> 关联：E4（w8a16 当年**未执行**——判断精度余量不足没实测）、F6（DiT 小矩阵 INT8 无优势，不适用 stage0 大矩阵）、
> 2026-08-16 动态量化（w8a8_dynamic）经评估不可行：激活 per-token 量化依赖 npu_quant_matmul 激活量化 kernel（E8 R1 堵点）+ 引入激活量化误差。

## 动机（测量数据，2026-08-16 stage 分解）

TTFT 444ms = prefill 111 + stage0 生成 345（15 token × TPOT 23.26ms）。
**TPOT 23.26ms 中 57% 是纯显存带宽物理下限**：8B fp16 权重 16GB / 910B HBM ~1.2TB/s = 13ms/token（权重全量重读，L2 128MB 远不够缓存 16GB）。

→ **权重体积是唯一可动的 TTFT 杠杆**。

## 路径：W8A16（weight-only，激活保持 fp16）

**vllm-ascend 现成方法**（`vllm_ascend/quantization/methods/w8a16.py`）：
- `@register_scheme("W8A16", "linear")` → `AscendW8A16LinearMethod`
- 权重存 int8（8GB）+ **per-channel `weight_scale` + `weight_offset`**（反量化精度保护）
- forward = `torch_npu.npu_weight_quant_batchmatmul(x, int8_w, antiquant_scale, antiquant_offset, bias)` —— **AICore 片上反量化**，激活 fp16 原样参与，**无激活量化误差**
- 带宽：读 8GB int8 权重（减半）+ 反量化不落显存

**为何比动态量化可行**：
| | 动态量化（w8a8_dynamic） | W8A16（weight-only） |
|---|---|---|
| 激活 | per-token 量化（额外 kernel + 精度损失）| **fp16 原样**（零激活误差）|
| 权重反量化 | 无（int8 GEMM 直算）| 片上 anti-quant（per-channel scale+offset）|
| 昇腾算子 | npu_quant_matmul（激活量化路径未验证）| **npu_weight_quant_batchmatmul（现成、专为 weight-only 设计）** |
| 注册 | 已注册但路径未验证 | **已注册（methods/w8a16.py）** |

## 预期收益（带宽模型）

| 指标 | 现值 | W8A16 后 | 差值 |
|---|---|---|---|
| 权重读带宽 | 13ms/token | ~7ms/token | -6ms |
| TPOT | 23.26ms | ~15-17ms（带宽 7 + 计算/调度不变，int8 matmul 算力 2x 部分吸收）| -6~-8ms |
| stage0 生成（15 token）| 345ms | ~240-260ms | -100ms |
| prefill | 111ms | ~85ms | -26ms |
| **TTFT** | **444ms** | **~320-345ms** | **-100~-125ms（RTF -0.029~-0.036）** |

保守口径 RTF -0.03，超过采纳门槛 0.02。E4 当年 w8a16"未执行"——**没有实测证据说明它精度不行**，值得跑 32 条验证。

## 技术路径（启用）

1. **L0 探查（≤30min）**：vllm-ascend quant_parser 如何把配置映射到 `get_scheme_class("W8A16","linear")`（CLI `--quantization` 值 or compressed-tensors config 的 quant_method）；stage0（Qwen2-8B）是 vLLM 标准模型 → 量化配置走标准 QuantConfig 注入即可（不同于 E8 的 diffusion 就地替换）。
2. **权重转换**：权重 int8 + scale + offset 从哪来——vllm-ascend 的 `quant_parser`/`modelslim_config` 支持加载已量化权重（modelslim 导出）或运行时 RTN 量化（`--quantization` 在线转换路径是否存在需确认；没有则用 30 行脚本 RTN 量化权重并保存为 modelslim 格式）。
3. **冒烟四件套**（stage0 改动）：WER 32 + ASV 32 + VideoMME 32 + DO 128 粗回归。
4. **perf 32×1**：TTFT 目标 <380ms，RTF <0.58。
5. **全量四件套**（stage0 铁律）：WER zh 2020 + ASV 1088 + DO 1196 + VideoMME 2700（~15h）。
6. 全绿 → 晋级 + 文档同步；任何红 → 回滚（yaml 键删掉即回 fp16）。

## 风险

| # | 风险 | 缓解 |
|---|---|---|
| R1 | 权重 8bit 舍入精度（E4 当年担心的点）| per-channel scale+offset 是精度保护设计；32 条 WER 直接测；退化则换 fp8（e4m3，同样 weight-only，vllm-ascend fp8_config.py 现成）|
| R2 | 在线 RTN 转换路径不存在 | modelslim 导出工具或 30 行脚本；RTN 无需校准数据 |
| R3 | 全量验证成本高（~15h）| 只有 32 条全绿才启动 |
| R4 | 昇腾 int8 权重布局（NZ/对齐）| 方法内部处理（get_weight 返回 int8 张量），启动日志核对替换数 |

## 备选：FP8 weight-only（若 int8 精度不过）

910B 原生支持 FP8 e4m3：权重 8GB（fp8 每权重 1B）+ 动态范围优于 int8 → `vllm_ascend/quantization/fp8_config.py` 现成。收益同带宽模型。

## 官方复现路径（baseline.md §七/§八 对照）

官方评测流程 = 主办方拿**提交的代码、配置与文档**在官方环境（910C）重新部署（工程复现审查，§7 第 5 步）；提交物不含模型权重。

**量化必须可复现**，提交物清单：

| 提交物 | 内容 | 对应文档 |
|---|---|---|
| 转换脚本 | `quant_export_w8a16.py`（确定性 RTN，无校准数据 → 逐位可复现）| 随代码提交 |
| 复现步骤 | 跑转换脚本 → 得量化模型目录 → `env.sh` 改 `MODEL_DIR` → 启动 | "优化与复现说明" |
| 精度对比表 | 量化前后 WER/SIM/DO/VideoMME 全量数据（证明 ≤2pp 准入）| "Benchmark 评测结果" |
| 可比性声明 | 量化定义为**推理性能优化**（非模型修改），附转换确定性说明 | 复现说明开头 |

**910C 复测要求**：
- 官方环境 910C 性能/精度全部重跑（int8 算子同源可用，但带宽更高 → 收益需实测）
- 910B 本地数据只作预验证，提交文档需注明两环境差异

## 与 C24 的关系

独立可并行。量化动 stage0（TTFT/RTF），C24 动 stage2（RTF）。叠加预期 RTF 0.62 → ~0.57。
