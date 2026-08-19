# 910C TTS perf 热点 Profiling 专项（2026-08-19）

> 状态：**结论已定**（三阶段采集 + 解析闭环，热点定位完成）
> 目标：定位 seed-tts perf 任务（TTFT/TTFP/RTF）在 910C 上的时间去向与算子热点，为后续优化提供证据
> 结构化数据：`profile_conclusions.json`（schema_version=1，其他 session 直接消费此文件）
> 采集/解析方法论踩坑：见 §6（profile.md §7/§1 实测命中，已修复）

---

## 1. 采集口径与方法

- 环境：910C（1 NPU × 2 Chip 64G），champion 配置（`submission/1_code/minicpmo_4_5.yaml`：stage0 TP=2 双 die / stage1 die0 / stage2 die1）
- profiler：`--profiler-config '{"profiler":"torch","torch_profiler_dir":".../profiling/traces"}'` + `VLLM_CUSTOM_SCOPES_FOR_PROFILING=1`（必须在 worker 派生前设置）
- 口径：seed-tts en 32×1（35 请求含 3 warmup），torch_npu Level1（AiCoreNone）Text export
- **每轮只采一个 stage**（profile.md §1：同物理 NPU 多 profiler 会损坏 device task）；三阶段各一轮
- 离线解析：`torch_npu.profiler.profiler.analyse(<raw_ascend_pt_dir>, export_type='text')`
- 同轮 perf（profiler 开启，**仅参考**）：warmup RTF 0.29 / stage0 轮 0.35 / stage1 轮 0.37 / stage2 轮 0.30 —— torch profiler 宿主开销使数值偏大，不作为性能结论

## 2. 核心结论：三阶段全部 host-bound

| stage | 角色 | 窗口 | Computing | Communication | Free（空闲） | host api 总时间 | kernel launch 数 |
|---|---|---|---|---|---|---|---|
| stage0 | thinker（TP=2） | 42.3s | **11.5%** | 8.5%（Allreduce） | **79.9%** | 8.5s | 545k（15.5k/请求） |
| stage1 | talker | 45.1s | **15.8%** | 0 | **84.2%** | 14.2s | 2.0M（57k/请求） |
| stage2 | code2wav | 43.2s | **29.5%** | 0 | **70.5%** | **28.4s > device 12.7s** | 1.36M（39k/请求） |

**NPU 每阶段 70-84% 时间在等 host 下发 / 跨 stage 数据，device 计算不是瓶颈**——继续在算子内抠时间收益有限，host 侧（launch 合并、kernel 融合、减少同步）与跨 stage 流水才是大头。

## 3. 各 stage 耗时在哪

### stage0（Thinker，TTFT 主体 ~256ms）— host 同步间隙 + TP 通信
- device 自耗时 8.6s/rank：**wait_event 67.2%**（7409 次，每请求 ~212 次，device 等 host 同步/下发）+ **HcclAllreduce 19.9%**（3078 次，TP=2 die 间通信）+ Matmul 仅 6.8%
- **TTFT 主体不是算子**：是 host 侧调度间隙 + TP 通信

### stage1（Talker，生成期逐 token）— launch-bound，簿记 kernel 主导
- device 自耗时 2.9s：Scatter 24.7%（KV 簿记）+ slot_mapping 21.9% + Bincount 16.3% 合计 64%，attention 仅 7.7%（每次 3us）、Matmul 2.9%
- host 侧 2.0M 次 launch（每请求 ~57k 次）——**AR 逐 token 循环被 launch 开销主导**

### stage2（Code2Wav，生成期每块 ~100ms）— 三阶段中 device 最忙，op 碎片化最重
- device 自耗时 12.8s：**Conv2D 27.5%**（flow encoder CNN + HiFiT 声码器）+ **aclnnCat 15.1%**（chunk 拼帧）+ **LayerNorm 12.8%** + Mul/Addmm/Add 逐元素风暴；Conv2DTranspose 3.1%（声码器上采样）
- **host launch 28.4s > device 计算 12.7s**：launch+LaunchKernel 13.9s 纯 launch 开销（1.36M kernel）
- **50.6k 次 aclopCompileAndExecute（2.5s）——运行时 shape 不稳触发重编译**（stage2 是 enforce_eager + 禁 chunked prefill，变长 chunk/initial 8 帧路径嫌疑）

## 4. 优化杠杆排序（证据驱动）

| rank | stage | 杠杆 | 依据 |
|---|---|---|---|
| 1 | stage2 | op 融合：Cat/LayerNorm/Mul 逐元素风暴（占 device 时间 ~36%）→ 少而大的 kernel | 1.36M launch / 请求 39k；host 28.4s > device 12.7s |
| 2 | stage2 | 查 50.6k 次 aclopCompileAndExecute 的 shape 不稳根因 | 每请求 ~1446 次运行时编译（2.5s/43s） |
| 3 | stage0 | TP=2 Allreduce 通信（窗口 8.5%）+ host 间隙 | 与 C63 的 TP 收益权衡；TTFT 主体非算子 |
| 4 | stage1 | 逐 token launch 减量（57k 次/请求），融合簿记 kernel | Scatter/slot/Bincount 占 device 时间 64% |

## 5. 与其他专项的交叉说明（重要）

1. **与量化专项的关系**（`quantization_special_project.md`）：量化收益模型假设 stage0 decode "memory-bound（每 token 读全部权重）"；本 profile 显示**窗口级 stage0 是 host-bound（Computing 仅 11.5%，Matmul 仅 6.8%）**。两者不矛盾（memory-bound 指 kernel 内部带宽、host-bound 指窗口级空闲），但**互相印证**：量化专项 §4.1 实测 M=1 时 w8a16 仅 -9%、w8a8 -2.5% —— host-bound 下权重体积减半的带宽收益确实无法兑现。**量化对主指标（32×1）收益预期 <3% 的结论被本 profile 独立佐证**。
2. **C63 TP=2 的代价量化**：Allreduce 3.6s/42.3s 窗口（8.5%），每请求 ~88 次 —— 通信是 stage0 第一大真实 kernel 开销；任何权重压缩（量化）不减通信量。
3. **cudagraph 现状**：stage0/1 已 FULL_AND_PIECEWISE，stage2 是 enforce_eager（数据依赖不可图化）——stage2 的逐元素碎片化无 cudagraph 兜底，是 op 融合收益最大的原因之一。
4. **profiler 开销警告**：torch profiler 自身 host 开销会放大 Free 占比与同轮 perf 数值（0.31→0.35-0.37）；但 device 侧硬数据（wait_event 时长、kernel launch 计数、op 自耗时）不受影响。结论以 device 测量为主。
5. **Free 的组成**：host 未下发（launch 间隙/引擎调度）+ 跨 stage 等待（stage2 等 stage1 的 chunk）。stage2 的 wait_event 仅 0.8%（区别于 stage0 的 67%）→ stage2 的空闲是**等上游数据/等 host 下发**，不是 device 侧同步。
6. **AiCoreNone 的边界**：本次未采集 aicore 内部管线指标（mte/mac/vec 比例），kernel 内部 memory-bound 与否需 Level2/aic_metrics 重采才能判定（profile.md §8）。

## 6. 采集踩坑（profile.md 实测记录）

- **§7 命中：默认 `/start_profile` 由 upstream vllm handler 服务**（`vllm/entrypoints/serve/profile/api_router.py:23`），无 stages 参数 → 首轮实际全 stage 同时 profiler
- **§1 命中：同 die 双 profiler 冲突** → stage0 双 rank 的 kernel_details 缺失，首轮数据作废（保留在 `traces/9e059ade9708_*_2026081901271*`，~11.7G，可删）
- 修复：`api_server.py` 移除 upstream profile 路由 + 强制 include omni `profiler_router`（TEMP 补丁，已回滚）；验证 = 日志出现 `[api_server.py:3374] Starting profiler for stages: [N]` + 仅目标 stage 产生原始数据

## 7. 数据路径（只列路径）

```
/workspace/vllm-omni/submission/7_runtime/profiling/profile_conclusions.json
/workspace/vllm-omni/submission/7_runtime/profiling/README.md
/workspace/vllm-omni/submission/7_runtime/profiling/make_conclusions_json.py
/workspace/vllm-omni/submission/7_runtime/profiling/analyze_hotspots.py
/workspace/vllm-omni/submission/7_runtime/profiling/profile_round.sh
/workspace/vllm-omni/submission/7_runtime/profiling/traces/stage0_rank0/9e059ade9708_928124_20260819014651183_ascend_pt
/workspace/vllm-omni/submission/7_runtime/profiling/traces/stage0_rank1/9e059ade9708_928125_20260819014651183_ascend_pt
/workspace/vllm-omni/submission/7_runtime/profiling/traces/stage1_rank0/9e059ade9708_930434_20260819014754613_ascend_pt
/workspace/vllm-omni/submission/7_runtime/profiling/traces/stage2_rank0/9e059ade9708_930839_20260819015301523_ascend_pt
/workspace/vllm-omni/submission/7_runtime/profiling/traces/stage0_rank0/9e059ade9708_928124_20260819014651183_ascend_pt/ASCEND_PROFILER_OUTPUT/operator_details.csv
/workspace/vllm-omni/submission/7_runtime/profiling/traces/stage0_rank0/9e059ade9708_928124_20260819014651183_ascend_pt/ASCEND_PROFILER_OUTPUT/step_trace_time.csv
/workspace/vllm-omni/submission/7_runtime/profiling/traces/stage0_rank0/9e059ade9708_928124_20260819014651183_ascend_pt/ASCEND_PROFILER_OUTPUT/api_statistic.csv
/workspace/vllm-omni/submission/7_runtime/profiling/traces/stage0_rank0/9e059ade9708_928124_20260819014651183_ascend_pt/ASCEND_PROFILER_OUTPUT/kernel_details.csv
/workspace/vllm-omni/submission/7_runtime/profiling/traces/stage1_rank0/9e059ade9708_930434_20260819014754613_ascend_pt/ASCEND_PROFILER_OUTPUT/operator_details.csv
/workspace/vllm-omni/submission/7_runtime/profiling/traces/stage1_rank0/9e059ade9708_930434_20260819014754613_ascend_pt/ASCEND_PROFILER_OUTPUT/step_trace_time.csv
/workspace/vllm-omni/submission/7_runtime/profiling/traces/stage1_rank0/9e059ade9708_930434_20260819014754613_ascend_pt/ASCEND_PROFILER_OUTPUT/api_statistic.csv
/workspace/vllm-omni/submission/7_runtime/profiling/traces/stage1_rank0/9e059ade9708_930434_20260819014754613_ascend_pt/ASCEND_PROFILER_OUTPUT/kernel_details.csv
/workspace/vllm-omni/submission/7_runtime/profiling/traces/stage2_rank0/9e059ade9708_930839_20260819015301523_ascend_pt/ASCEND_PROFILER_OUTPUT/operator_details.csv
/workspace/vllm-omni/submission/7_runtime/profiling/traces/stage2_rank0/9e059ade9708_930839_20260819015301523_ascend_pt/ASCEND_PROFILER_OUTPUT/step_trace_time.csv
/workspace/vllm-omni/submission/7_runtime/profiling/traces/stage2_rank0/9e059ade9708_930839_20260819015301523_ascend_pt/ASCEND_PROFILER_OUTPUT/api_statistic.csv
/workspace/vllm-omni/submission/7_runtime/profiling/traces/stage2_rank0/9e059ade9708_930839_20260819015301523_ascend_pt/ASCEND_PROFILER_OUTPUT/kernel_details.csv
/workspace/vllm-omni/submission/7_runtime/profiling/round_stage0.log
/workspace/vllm-omni/submission/7_runtime/profiling/round_stage1.log
/workspace/vllm-omni/submission/7_runtime/profiling/round_stage2.log
/workspace/vllm-omni/submission/7_runtime/profiling/warmup_after_fix.log
```

污染轮数据（可删，~11.7G）：
```
/workspace/vllm-omni/submission/7_runtime/profiling/traces/20260819-012713_stage0_rank0_1787102833/
/workspace/vllm-omni/submission/7_runtime/profiling/traces/20260819-012713_stage0_rank1_1787102833/
/workspace/vllm-omni/submission/7_runtime/profiling/traces/20260819-012713_stage1_rank0_1787102833/
/workspace/vllm-omni/submission/7_runtime/profiling/traces/20260819-012713_stage2_rank0_1787102833/
/workspace/vllm-omni/submission/7_runtime/profiling/traces/stage0_rank0/9e059ade9708_913117_20260819012713767_ascend_pt/
/workspace/vllm-omni/submission/7_runtime/profiling/traces/stage0_rank1/9e059ade9708_913118_20260819012713767_ascend_pt/
/workspace/vllm-omni/submission/7_runtime/profiling/traces/stage1_rank0/9e059ade9708_915390_20260819012713905_ascend_pt/
/workspace/vllm-omni/submission/7_runtime/profiling/traces/stage2_rank0/9e059ade9708_915137_20260819012713986_ascend_pt/
```

## 8. 复现步骤

1. 重启服务带 profiler：`--profiler-config '{"profiler":"torch","torch_profiler_dir":"<dir>"}'` + `VLLM_CUSTOM_SCOPES_FOR_PROFILING=1`
2. `POST /start_profile {"stages":[N]}` → `vllm bench serve`（32×1 seed-tts en）→ `POST /stop_profile`（每轮只采一个 stage；**默认路由是 upstream 无 stages 版，需先按 §6 修复**）
3. `torch_npu.profiler.profiler.analyse(<raw_ascend_pt_dir>, export_type='text')`（stage2 12G 约需 20-30 min）
4. `python3 analyze_hotspots.py <raw_dir>` 人读 / `python3 make_conclusions_json.py` 重生成 JSON

## 9. 待办

- [ ] 污染轮数据删除确认（§7 底部路径，~11.7G）
- [ ] （可选）Level2 + aic_metrics 重采 stage2，判定 kernel 内部 memory-bound 比例（profile.md §8）
- [ ] （可选）stage2 的 aclopCompileAndExecute 根因排查（初始 8 帧/变长 chunk shape 稳定性）
- [ ] 服务当前为 profiler 开启状态运行中（用户指示暂缓恢复）；下次重启即恢复 champion（无 --profiler-config）
