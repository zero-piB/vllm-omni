# 自我优化协议（子赛道 B · vLLM-Omni）

驱动 Claude 针对本赛题做推理优化迭代的规则书。**每次迭代先读本文件 + `ledger.json`**。
本目录（`/workspace/self_optimize/`）是设计产物，只读 submission 与 vllm-omni，执行期仅在
`submission/7_runtime/`（运行时区）落产物；改动 `1_code/*.yaml`、`2_configs/*.sh`、源码 patch
只发生在**晋级（promotion）**时。

---

## 0. 权威文档与现状

| 文档 | 用途 |
|---|---|
| `/workspace/baseline.md` | 评分规范：精度准入阈值、性能基线、提交材料要求 |
| `/workspace/vllm-omni-demo.md` | 操作指南：§6 服务启动、§7.6 Seed-TTS、§7.7 Daily-Omni、§9 全双工、§10 官方测试 |
| `/workspace/submission/4_perf_report/metrics_summary.md` | 优化历史与瓶颈分解（人类可读版台账） |
| `/workspace/submission/6_optimization/optimization_report.md` | E0–E9 各实验细节 |
| `ledger.json`（本目录） | 机器可读状态：阈值/冠军/已证伪/候选池/实验史 |
| `/root/.claude/CLAUDE.md` | 踩坑沉淀的硬知识（dtype、评测顺序、ASV 协议、pkill 陷阱等） |

## 1. 评测地图（哪些指标用哪个配置跑）

| 指标 | 准入 | 服务配置 | 脚本 | 32条耗时 | 全量 |
|---|---|---|---|---|---|
| Seed-TTS WER | ≤1.56 | **默认 yaml**（优化面） | `eval_seed_tts_wer.sh [N]` | ~12min | 1088 条 ~2h生成+1.5h转写 |
| Seed-TTS ASV/SIM | ≥0.689 | 默认 yaml | `eval_seed_tts_asv.sh [N]` | ~12min | 1088 条 |
| **性能 TTFT/TTFP/RTF** | 无阈值，与 910C 基线 333.27/986.47/0.4423 对比 | 默认 yaml | 同一 WER 运行即输出（`--percentile-metrics`） | 同上 | — |
| Daily-Omni | ≥77.5 | **bench yaml**（rep 1.2/64帧/25块，官方配方；TTS 旋钮无关） | `eval_daily_omni.sh [N]` | ~15-25min | 1196 条 ~75min |
| VideoMME | ≥67.0 | 默认 yaml（image≥96） | `eval_videomme_official.sh [N]`（minicpm-frames/96帧/并发4） | ~6min | 2700 条 ~10h |

要点：**WER/ASV/perf/VideoMME 吃优化配置，Daily-Omni 吃官方 bench 配置** →
改 TTS 旋钮的实验理论上不碰 DO；改 stage0（批大小/prefill/缓存）的实验才需要四件套验证。

## 2. 迭代循环（一轮 = 下面 8 步，墙钟 ≤2h）

1. **读状态**：`ledger.json` + `metrics_summary.md`（实验史）→ 确认当前冠军、已证伪项、候选池。
2. **选候选**：从 `ledger.candidates` 取 status=open 中风险×收益最优的一个；**一轮只验证一个旋钮**（E5 教训：chunk 与 ts 必须配对时视为一个假设）。若池中无可取项，先 `grep` 源码（`$REPO_DIR`）找新旋钮并登记入池。
3. **写假设**：明确 → 改什么、预期哪个指标动多少、哪个精度指标可能退化、按什么口径对比（同 N 同并发同轮）。**收益估计必须附证据**（profile 输出行 / preflight 微基准数据 / 字节数计算 / 已在外模块验证过的机制）；只有"代码阅读直觉"的估计标记 `confidence=low`，**强制先过 §2.5 preflight**。假设不成立则本轮终止，只记台账。
4. **实现（只动运行时区）**：
   - yaml 实验：`cp 1_code/minicpmo_4_5.yaml 7_runtime/exp/<exp_id>.yaml`，改副本（Edit 工具）。
   - 源码实验：写 patch 到 `7_runtime/exp/<exp_id>.patch`，`git -C $REPO_DIR apply`（可逆：`git apply -R`）；**晋级前不得并入 `6_optimization/patches/`**。
   - 启动：`server_restart.sh` 需支持路径参数（当前实现只接受 `1_code/` 内文件名 —— 执行期加一行 case 分支即可，设计期不动）。
5. **冒烟（N=32，先快后慢）**：WER+perf 一次运行同出 → ASV → 若动 stage0 再 32 VideoMME → 若需确认 DO 未坏，128 条 DO **只作粗回归检测**（128 条 60% vs 全量 78.5% 是样本偏差，不可当绝对口径）。
6. **判定**（见 §3 决策规则）。
6.5 **回滚前展示（2026-08-20 新增，用户指示）**：判定 rejected/falsified 且涉及代码或配置改动（patch 已 apply / 实验 yaml 在用）时，**保持现场不立即回滚**——结束语先给完整证据（本轮假设、改动、冒烟数字表格、同日对照、判定理由），等用户确认后才 `git apply -R` / 还原配置。无人值守（cron）连跑模式下无人在场确认时例外：按判定自动回滚并在台账注明"无人值守自动回滚"。
7. **记账**：`ledger.json` 追加 experiments 条目 + 更新候选 status；F 系列更新 `falsified`；不达标项也记录（防止重试）。
8. **晋级（promotion）**：判定通过 → 复制配置进 `1_code/`（同时更新 `4_perf_report/metrics_summary.md` 实验表）→ 对**受影响**的指标跑全量（TTS 改动：WER 1088 + ASV 1088；stage0 改动：DO 1196 + VideoMME 2700）→ **全量全绿才更新 `champion` 并允许 commit**；32 条冒烟绿但全量红 = 配置回滚，实验记 fail（E9-full 教训：E5/E6/E7/E9 均未跑全量即晋级，全量 WER 3.03% 超标）。

## 2.5 preflight 关卡（2026-08-15 新增，遏制失败率）

**动机**：C30/C31/C32/C33b/C42 连续 5 轮失败的共同根因 = 静态阅读 + 拍脑袋估计直接端到端（±3% 噪声窗把小收益淹没，白烧 30-60min/轮）。成功候选（C27/C29b/C20/E6）全部有实测或已验证机制支撑。

**规则**：任何 perf 候选（TTFT/TTFP/RTF 任一涉及）在端到端冒烟之前，**必须**先写独立微基准脚本（放 `7_runtime/exp/preflight/`，不起服务，≤15min 出结论）直测核心机制；机制不成立 → 直接证伪记 F，**不进入端到端**。精度候选（WER/SIM 口径类）不需 preflight。

**判据**：preflight 实测收益换算成端到端指标，低于采纳门槛（RTF <0.02 / TTFT <10ms / TTFP <10ms）且无放大路径 → 直接证伪。preflight 通过才允许写 patch + 端到端。

**模板**：`7_runtime/exp/preflight/bench_template.py`（独立脚本、真实 shape/dtype、warmup + 循环计时、host 与 device 时间分开测、输出可判定表格）。

**铁律（2026-08-11 危机后新增）**：每 commit 一个举动，必须保证所有测试都是达标的——
32 条小样本只作快速筛选，**任何采纳/晋级/commit 必须以全量（或 ≥256 条并验证与全量趋势一致）验证所有指标达标为前提**。
32 条"达标"曾经是前 32 条简单样本的假象（WER 1.31% vs 全量 3.03%，E9-full 教训），不允许再犯。

**噪声窗**（32 条口径，实测）：WER ±0.5pp；SIM ±0.001；perf ±3%；DO 128 条不可作绝对口径。

| 情况 | 判定 |
|---|---|
| 任一精度指标 32 条跌破准入 | ❌ 直接拒绝，不重试 |
| 精度全绿，perf 任一指标达绝对值门槛 | ✅ 采纳候选（2026-08-13 用户最终指示：**TTFT ≥10ms / TTFP ≥10ms / RTF ≥0.02，任一达标即采纳**，2-3 轮 32 条均值确认，且其他指标不劣化） |
| 未达门槛或方向相反 | ⚠️ 不采纳（记账关闭；其他指标劣化超噪声则拒绝） |
| 精度全绿，perf 在 ±5% 噪声内 | ⚠️ 不采纳（收益不可分辨，记账后关闭候选） |
| WER 相对当前冠军上升 ≥0.5pp 但未破线 | ⚠️ 加跑一轮 32 确认；两轮均值稳定上升则拒绝 |
| 晋级前 | 全量跑受影响指标，全绿才更新 `champion`；全量有红则回滚到上一冠军并记录 |

**红线（违反即作废）**：
- 只改推理/服务侧配置，**不得改模型行为、评测脚本、评测数据、prompt**（baseline.md §4.1 失去可比性）。
- dtype 必须 `float16`（bf16 已证伪 F0）；`--trust-remote-code` + 本地模型路径不可丢。
- 不跨 N、不跨并发、不跨轮次比性能（A/B 必须同批 32 条同会话，或与台账同口径条目比）。
- 禁止重试 falsified 列表中的项。
- 每轮预算 ≤2h 墙钟；全量评测仅在 promotion 意图下启动（VideoMME 全量 10h 更要谨慎）。

## 4. 配置面（旋钮地图，★=已证伪禁试）

```
connector 层（默认 yaml）：
  codec_chunk_frames: 15          # 首块/稳态块帧数；★不可单独缩（F3），须与 ts 配对
  initial_codec_chunk_frames: 8   # ★4 已证伪（F2）；8 是最优平衡点
  token2wav_n_timesteps: 5        # flow 解码步数；C1 候选：4
  token2wav_float16: true         # 已开
  codec_left_context_frames: 3    # 编码器左上下文
  connector_get_sleep_s: 0.01     # 轮询间隔；C4 候选：0.003
stage0（prefill/KV，VideoMME/DO 相关）：
  max_num_seqs: 4 / gpu_memory_utilization: 0.55 / max_num_batched_tokens: 16384
  max_model_len: 32768 / enable_prefix_caching: false（C6 候选）
  cudagraph_mode: PIECEWISE
stage2（Code2Wav）：
  ★cudagraph 已证伪（F4）；enable_chunked_prefill: false
全局：★bf16（F0）、★NZ=2（F1）、★w8a16 未执行（E4，精度余量不足不试）
```

## 5. 记账规范

- `ledger.json` 是唯一状态源；每轮结束必须写回（保持 JSON 合法、字段齐）。
- **代码勘探存档**：本轮看过的代码/平台机制（文件:行号、新结论、修正）增量同步到 `code_survey.md`（同目录）——防重复挖掘，是存档材料。规则：勘察过的新区域或推翻旧结论必须更新；纯 ledger 判定（冒烟数字）不用重复写。
- 实验条目：`{id, date, change, n, rounds, wer, sim, ttft_ms, ttfp_ms, rtf, daily, videomme, verdict, note}`；跑了的字段填数，没跑的置 null。
- 判定值枚举：`promoted / rejected / champion / fail_accuracy / skipped`。
- 全量结果记入 `champion.metrics_full_scale`（带 n/date/yaml）。
- `promotion_history` 追加一行（日期 + 通过的闸口）。

## 8. 平台特性检查表（候选生成/审查必对照；撞上先预判再 preflight）

昇腾 910B3 + CANN 9.0 实测沉淀（F 系列），**不是 CUDA 心智模型**：

| 若候选涉及… | 已知特性（证伪编号） |
|---|---|
| 多流/stream 并行/event 解耦 | DMA（H2D）与 AICore **不并行**；多流 1.74x 仅纯 GEMM 场景（F15） |
| int32/tensor 作 op 参数 | 有额外处理路径，比 list 绑定慢（F14） |
| NPU graph capture | 只支持 aclnn 算子；DiT 的 CausalConv1d 是 aclop → 不可行（F9） |
| torch.compile/inductor | 动态 shape + aclop conv 编译崩溃（F11） |
| QKV 三 Linear 融合 / cat 临时权重 | 无 nn.Linear 优化布局，chunk 路径 GEMM 效率低（F13）；C20 的收益全归 adaLN |
| handoff / 进程间传输通道 | fp16 tolist 已最优；bytes/重建通道负收益（F16） |
| prefix caching | 架构不兼容（F8） |
| 采样链 / top_p / softmax | 见 C43 preflight 实测（每次软 max/sort 的 host 发射主导） |
| fp32 推理路径 | 昇腾算子栈不支持（F7） |
| bf16 | 精度不足（F0）；dtype 恒 float16 |

**收尾纪律**：`status` 输出默认给"下一候选 + preflight 计划"；收尾/提交材料建议**只在候选池耗尽且 grep 源码挖新旋钮失败后提一次**，并附挖掘证据。

## 7. 无人值守连跑模式（可选）

**机制**：durable cron 每 2-3h 触发一次 `/self-optimize once`（新上下文读台账 = 无记忆漂移；
会话意外死亡不影响下轮）。平台限制：recurring cron **7 天自动过期**，到期前重挂一次；
建议节奏 = 冒烟轮隔 2-3h 一跑；启动全量轮（2-12h）的轮次期间不叠跑下一轮。

**每轮启动自检（防残留/挂死，无人值守的核心）**：
1. `npu-smi info` 无残留 `VLLMStageEngi`（有则先 `server_restart.sh` 兜底）；
2. 台账末条实验 verdict 为 fail/hang → 先修再跑；**连续 3 轮 fail → 停止，journal 留痕等人**；
3. `7_runtime/results` 磁盘水位 <85%（`df`）；
4. 每个 bench 命令包 `timeout 4h`（挂死自动 kill，记台账重试一次，重试仍挂 → 停）。

**无人值守的晋级门槛更严**（无人在场看结果）：
- perf 提升需**两轮 32 条均 ≥8%** 才采纳候选；
- 全量全绿才更新 champion；任何精度红 = 记档 + 回滚上一冠军 + 停止本轮；
- 每轮结束在 ledger `experiments` 记一行（含耗时），`candidates` 更新 status —— 下轮全部从台账续。

**预期产出**：冒烟模式 ≈ 8-12 轮/天；混全量 ≈ 2-3 冒烟轮 + 1 全量轮/天。

## 6. 执行期待办（设计→实施切换时）

4. **git 追溯**：`submission/` 已纳入 git（`7_runtime/`、`kernel_meta/` 已 ignore，基线 commit `812926a`）。
   每次晋级把配置落进 `1_code/` 后**必须** `git add -A && git commit -m "E<id> promote: <改动摘要>"`；
   源码 patch 进 `6_optimization/patches/` 时同样 commit。这样每个晋级点在 git 历史里留痕，随时可 diff 回滚。

1. `server_restart.sh` 支持含 `/` 的 yaml 路径（约 3 行）。
2. `2_configs/env.sh` 无改动需求；实验 yaml 统一放 `7_runtime/exp/`（随运行时删除即重置）。
3. skill 驱动方式确认：`/self-optimize status | once | n`（见 `/workspace/.claude/skills/self-optimize/SKILL.md`）。
