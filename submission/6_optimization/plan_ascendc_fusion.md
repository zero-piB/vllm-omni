# 方案：C24 stage2 DiT 发射融合（Ascend C 自定义算子）

> 状态：**方案待审**（2026-08-16）。用户审阅后决定是否实施。
> 关联：C20（发射分析）、F13（QKV 融合教训：F.linear cat 临时权重无优化布局）、F9（aclop 不支持 graph capture）。

## 动机（C20 msprof 数据）

stage2 每块 222ms = enc 13 + **cfm 171** + hift 38。cfm（DiT 5 步 × 2 分支 × 16 层）由 ~3400 次小 kernel 发射主导（12 万个 0-500μs kernel，launch 平均 **12.7μs**）→ 发射 ≈ 43ms/块 占 cfm 25%。

C24 目标：融合每 block 的 **modulate（adaLN）+ gate + 残差链**（720 个 kernel/块）为 1 个 Ascend C 算子 → 省 719 次发射/块。

## 预期收益（C20 数据换算）

- 消 720 发射/块 × 12.7μs ≈ **9.1ms/块**（cfm 171→~162）
- 7 块/条 → 64ms/条 → **RTF -0.016**（略低于 0.02 门槛——**边缘**）
- 附带：显存带宽省（中间张量不落内存）

**诚实评估**：RTF -0.016 < 0.02 采纳门槛，工程量大（Ascend C 算子开发 + 验证）。**收益边缘**——方案价值主要在于：若量化（stage0）也做，二者叠加才显著；或融合范围扩大（attn 输出 proj + modulate 全链，~2000 kernel）→ RTF -0.04 级。

## 技术路径

1. **Ascend C 算子**（`kernel_meta` 自定义算子包）：
   - 输入：adaLN 调制参数（9×512/块，C20 已预计算）+ x 残差 + gate
   - 运算：`x * (1 + gamma) + beta`（modulate）+ SiLU 前 gate + 残差加（每块 ~15 元素级运算，当前 720 kernel 拆发射）
   - 输出：fp16，shape 跟随 chunk（15 帧 × 512）
2. **插桩**：`decoder_dit.py`（site-packages 双副本）`DiTBlock.forward_chunk`——C20 已在此文件做过 adaLN 预计算，融合点明确。
3. **验证**：
   - 数值对齐：融合算子 vs 原 kernel 链 allclose（rtol 1e-2，fp16）+ 32 条 WER 逐位口径
   - 性能：perf 32×1（RTF 目标 <0.60）
   - 全量 WER 1088 + ASV 1088（TTS 改动铁律）
4. **回滚**：patch 反转（site-packages .bak 恢复，同 C20 流程）。

## 风险

| # | 风险 | 缓解 |
|---|---|---|
| R1 | 昇腾 AICore 上 elementwise 融合收益被单算子启动吃掉（发射 12.7μs → 融合算子更大）| 先测单算子 Launch 时间；若 >8ms/块收益消失则放弃 |
| R2 | Ascend C 编译链（算子工程）成本高 | 复用 vllm-omni 已有自定义算子工程（kernel_meta/）；MVP 只融合 modulate+gate 两段 |
| R3 | 数值差（fp16 融合顺序）| allclose + WER 把关 |
| R4 | 与 C20 的 adaLN 缓存交互 | 插桩点同一文件，改动隔离 |

## 实施步骤（若批准）

1. L0 探查（≤30min）：Ascend C 算子工程模板 + 单算子 launch 时间实测（决定 R1 去留）。
2. MVP 融合 modulate+gate（~300 kernel/块）→ 冒烟 32 → perf 32。
3. 若 MVP 达标（RTF -0.01+）：扩展融合范围（残差链全段，720 kernel）→ 冒烟 → 全量 WER/ASV。
4. 全绿晋级；任何红回滚。

## 决策建议

**先做量化（收益大 RTF -0.05）**；C24 作为其后增量（叠加 RTF -0.016~-0.04）。两者互不依赖。
