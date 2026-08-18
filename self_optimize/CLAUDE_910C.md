# 910C 环境 CLAUDE.md（由 910B 记忆自动生成 2026-08-18）

> 本文件由 910B 上的记忆文件合并生成，供 910C 的 Claude Code 使用。
> 用法：放在工作目录（或子目录）作为 CLAUDE.md，Claude Code 自动加载。
> 原始记忆来源：/root/.claude/projects/-workspace/memory/

## 权威文档（先读再动手）
- baseline.md — 评分规范：精度准入阈值、性能基线（910C F16 档）、提交材料要求
- vllm-omni-demo.md — 操作指南：服务启动、Seed-TTS、Daily-Omni、全双工、官方测试
- self_optimize/SELF_OPT_PROTOCOL.md — 优化规则书；self_optimize/ledger.json — 唯一状态源

## 硬知识速查
- dtype 恒显式 `--dtype float16`（bf16 在昇腾精度不足，WER 3.42% 超标教训）
- 官方 WER 门槛 = ZH（`--seed-tts-locale zh` + paraformer-zh；en 3.03% 是错误参照）
- WER/ASV 用 `--max-concurrency 4`；性能（TTFT/TTFP/RTF）用并发 1
- 评测顺序：先 32 条小样本定性 → 全量（WER zh 2020 ~1.5h + 转写 40min；VideoMME 2700 ~1.5-2h；DO 1196 ~1.5h；ASV 1088 ~1h）
- bench 客户端：`--model` 本地路径 + `--trust-remote-code`（内网 huggingface.co 不可达）
- 服务：`VLLM_WORKER_MULTIPROC_METHOD=spawn`、`--deploy-config` 绝对路径、`--init-timeout 1200`、`--allowed-local-media-path`
- 网络：HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1；github.com 走 ssh.github.com:443
- 评分规则：RTF 优先 → TTFP → TTFT（牺牲 RTF 换 TTFT 是负优化，C48 教训）
- 提交物 = submission/（1_code yaml + 2_configs + 6_optimization patches）；7_runtime 是运行时产物不入库

### 910c-replication-checklist
**910C 机器到手时的复刻清单**（2026-08-18 用户获得 910C 机器，需把 910B 环境与优化完整搬过去）。

**Why**: 比赛评分在官方 910C 环境跑；910C 上要复现 champion 配置并补 910B 上做不到的全量验证。

**How to apply**:

1. **拷贝**：`submission/`（1_code yaml + 2_configs 脚本 + 6_optimization patches）+ `vllm-omni/`（opt/tts-performance 分支，champion 提交状态）+ `local_models/MiniCPM-o-4_5`（本地模型，GlusterFS 规避）+ vllm-ascend（版本以 910C 官方 pip 为准）
2. **环境**：CANN/torch_npu 版本可能不同 → vllm-ascend 匹配官方版本；headroom-venv 重建；`--dtype float16` 保持（官方 F16 档基线，bf16 精度教训）
3. **env.sh 5 变量**：VLLM/MODEL_DIR/REPO_DIR/RAW_DATA_DIR/DATA_DIR → `prepare_data.sh` 幂等生成数据
4. **启动+冒烟四件套 32 条**（WER/ASV/VideoMME/DO 128）→ perf 32×1
5. **910C 专属机会**：①ASV 1088 全量补跑（910B 32G 容器 OOM 欠的账，[[asv-full-32g-container-oom]]）；②C55 enable_static_kernel 试跑（910B 显存不足，910C 显存大可能可行）；③与 910C 官方基线对比（TTFT 333.27/TTFP 986.47/RTF 0.4423）
6. **铁律**：不跨 N 比性能（910C 只与 910C 基线比，不跨设备比）；E5/E6 yaml 配置不受 CANN 版本影响，源码 patch 需重新验证

相关：[[zh-wer-gate]]（WER zh 口径）、[[scoring-rtf-first]]（评分规则）、[[glusterfs-local-model]]（本地模型规避）

### asv-full-32g-container-oom
**本机（32G 容器）跑不了 ASV 1088 全量**（2026-08-17 确认）：容器 cgroup 内存上限 32GB（memory.limit_in_bytes=34359738368），服务权重 page cache ~26G + whisper-large-v3 加载峰值 ~10G = 36G 超限 → OOM kill（oom_kill 计数 8）。swappiness 只读、swap 仅 3G、drop_caches 无权限、taskset 限核无效（mmap 加载不耗 CPU）、cgroup limit 只读——**无解**。服务刚重启后（cache 冷 ~22G）32 条可跑（SIM 0.8478）；服务跑过长评测后必死。

**C49（cudagraph FULL_AND_PIECEWISE）晋级时 ASV 全量未跑**（2026-08-17）：理由 = 上述环境限制；ASV 32 条 SIM 0.8478 达标（准入 0.689，全量历史 0.8524）；WER zh 全量 1.42% / VideoMME 69.48% / DO 78.51% 三项全量坐实 C49 精度零变化（与历史完全一致）。

**Why**: 评分/晋级铁律是全量四件套，ASV 全量是唯一缺口，且缺口原因是环境（官方 910C 环境内存充足无此限制）。

**How to apply**: ①任何需要 ASV 全量的晋级（C49 之后所有 TTS 侧候选），先重启服务再跑（cache 冷贴边可过，但 1088 生成后 usage 仍可能超——目前只有 32 条验证过）；②官方提交材料需注明"ASV 全量在本地容器受限，官方环境补跑"；③C49 之后候选的 ASV 验证默认用 32 条 + 说明，除非换机器/加内存。

### cleanup-conservative
2026-08-11 用户指示：清理只动**确定无用**的东西；凡"以后可能用"的（如 `7_runtime/media/videomme_videos` 102G 转换数据、`7_runtime/exp/` 实验 yaml、`/workspace/eval/` 历史、早期脚本等）一律不删 —— 即使它们可从台账/脚本重建。

**Why**：重建成本可能比磁盘空间更贵（videomme_videos 重转换耗时；exp yaml 记录实验细节，台账只存摘要）。

**How to apply**：给出清理建议前先自问"这个以后还会用吗"——可能用就划进保留区；只建议删除确定垃圾（拼错文件名、空目录、自动再生产物、已回滚残留）。删除大块数据前必须逐项列出让用户拍板，不默认执行。相关：[[debug-patch-hygiene]]。

### debug-patch-hygiene
2026-08-11 教训：WER 危机排查期曾在官方脚本 `vllm_omni/benchmarks/patch/patch.py` 里加 5 行 per-item dump（写死绝对路径 `/workspace/submission/7_runtime/results/wer_items.jsonl`），用于分析前 32 条假象 / 归一化 WER。该路径硬编码 → 换机器必崩，且污染上游代码。

**Why**：官方评测脚本（vllm-omni 仓库）是提交环境的组成部分，任何本地残留都要可审计可回滚；硬编码绝对路径把"运行时行为"错放进"代码"里。

**How to apply**：
1. 调试/取证需要的输出 → 路径走运行时区（`submission/7_runtime/`，经 env.sh 的 `RESULTS_DIR` 等变量），不写死 `/workspace/...`；
2. 取证任务完成、结论定论后，立即 `git checkout` 回滚官方脚本，数据文件留在 7_runtime/ 保留；
3. 评测脚本的改动必须默认假设"换机器 + 别人也在用这个仓库"（参考 [[wer-full-validation-rule]] 同源精神：改动要有全量验证意识）。

### github-mirror-sites
网络无法访问 github.com 时，改用镜像站：
- `git clone https://gitclone.com/github.com/<owner>/<repo>.git`
- `git clone https://ghproxy.net/https://github.com/<owner>/<repo>.git`

例：`git clone https://gitclone.com/github.com/vllm-project/vllm-omni.git`

### glusterfs-local-model
2026-08-11：stage1 fp32 实验崩溃（force-kill）后，GlusterFS 共享盘（`/workspace/shared_assets`，FUSE.GLUSTERFS ro 挂载）checkpoint 加载**系统性卡死**——连续 3 次服务启动都卡在 `weight_utils.py:872`（权重加载）10 分钟直到 orchestrator 超时，与配置无关。

**规避**：模型已拷贝到本地 `/workspace/local_models/MiniCPM-o-4_5`（19G）。服务启动命令用本地路径（`vllm serve /workspace/local_models/MiniCPM-o-4_5`），启动 570s→420s 且稳定。eval 客户端用 `--model $SERVED`（模型名）不依赖路径，无需改评测脚本。

**注意**：env.sh 的 `MODEL_DIR` 仍指共享盘（换机器指引不变）；当前机器服务启动一律用本地路径。若共享盘恢复（另挂/重启），可回退。相关：[[wer-full-validation-rule]]

### perf-adoption-5pct
2026-08-13 用户最终指示（经历 8%→5%→3%→绝对值讨论）：性能采纳门槛改为**绝对值**——**TTFT ≥10ms / TTFP ≥10ms / RTF ≥0.02，任一指标达到即采纳**（2-3 轮 32 条均值确认 + 方向一致），**前提是其他指标不劣化**（WER/SIM 精度不变，其他 perf 指标不在相反方向显著变差）。

**Why**：百分比对量纲不同的指标无意义（TTFT 430ms 的 3% = 13ms 恰在噪声内，而 RTF 0.02 是真实门槛）；且同配置噪声带实测很窄（TTFT ±5-10ms / TTFP ±2-5ms / RTF ±0.005-0.01），10ms/0.02 高于噪声带，多轮均值进一步排除噪声。

**How to apply**：已写入 SELF_OPT_PROTOCOL.md §3。判定流程：3 轮 32 条均值达任一门槛 + 其他指标不劣化 → 采纳；晋级仍须全量铁律（[[wer-full-validation-rule]]）。当前 C20（TTFP 均值 -12ms）未达 10ms 门槛的严格确认（接近但未过）。相关：[[cleanup-conservative]]。

### rtf-eval-metric
2026-08-15 用户确认官方测评标准：
- **主指标 = mean_audio_rtf**（完整链路：`audio_rtf = audio_generation_latency / audio_duration`，latency 从请求发出到音频完成，**含 TTFT/TTFP**，见 `vllm_omni/metrics/definitions.py`）
- 比较优先级：**RTF 优先 → RTF 相同看 TTFP → 再 TTFT**
- 含义：**降 TTFT/TTFP 也直接降 RTF**（双吃）

**RTF 0.62 拆解**（32×1 档，每条音频 ~3.5s）：TTFT ~429ms（RTF 的 ~12%，每降 100ms → RTF -0.029）+ 生成期 ~1741ms（88%，stage1 每 token -1ms → RTF -0.007）+ 结尾。

**优化杠杆排序**（按 RTF 优先）：C33b（stage1 同步解耦，-0.5-1.9ms/token → RTF -0.013~-0.05）> stage2 每块 -2-3ms → RTF -0.014~-0.021 > TTFT 首 token 路径 -40ms → RTF -0.012。

**昇腾多流验证**（2026-08-15）：双流并行 gemm 加速比 1.74x（`torch.npu.Stream` 有效）→ C33b（event 独立流解耦）可行。

相关：[[zh-wer-gate]]（精度口径）、[[perf-adoption-5pct]]（性能门槛）。

### scoring-rtf-first
**比赛评分规则（2026-08-17 用户确认）**：**RTF 优先，RTF 相同看 TTFP**。TTFT 不在优先序（参考指标）。

**Why**: baseline.md §六 只写"综合考虑 RTF/TTFT/TTFP，权重以官方文档为准"——用户提供了确切规则。

**How to apply**:
- 候选评估优先级 = **RTF > TTFP > TTFT**——牺牲 RTF 换 TTFT 的候选（如 C48 DELTA：RTF 0.45→0.47 +4% 换 TTFT -71%）是**负优化**，直接不做
- 任何候选必须先确认对 RTF 无劣化（或 RTF 收益最大）；TTFT 优化只在 RTF/TTFP 不受损时才有价值
- C48（DELTA 流式）已回滚（2026-08-17）：TTFT 108ms 的收益在评分里无效，RTF 劣化 +4% 不可接受；patch 保留在 6_optimization/patches/0006（不启用）
- 与 [[rtf-eval-metric]] 互补：mean_audio_rtf = 完整链路（含 TTFT/TTFP）口径

### wer-full-validation-rule
2026-08-11 WER 危机教训：E5/E6/E7/E9 都基于 32 条小样本（前 32 条恰好是简单样本）判定"达标"并晋级，全量 1088 实测 WER 3.03%——**32 条"达标"是前 32 条简单样本的假象**（1.31% vs 3.03%）。
**2026-08-13 修正**：3.03% 是 **en 口径**；官方门槛是 **zh 口径**（zh 全量 2020 = **1.41% 达标**，与 910C 基线持平，见 [[zh-wer-gate]]）。"全量验证"铁律不变，只是判定口径改为 zh。

**Why**：小样本（尤其数据集按序取前 N 条）有系统性偏差——数据集越往后样本越难；32 条绿 + 全量红 = 假达标。

**How to apply**：任何改动 commit/晋级前，必须全量（WER 1088 / ASV 1088；stage0 改动 DO 1196 + VideoMME 2700）验证所有指标达标；32 条只作快速筛选。已写入 SELF_OPT_PROTOCOL.md §3 铁律。相关：[[seed-tts-wer-crisis]]（910B 全量 WER ~3% 固有，chunk15/initial8 嫌疑未排除）。

### zh-wer-gate
2026-08-13 重大修正：**官方 Seed-TTS WER 达标门槛是 ZH 口径**（`--seed-tts-locale zh`，zh 全量 2020 条，paraformer-zh 转写 —— 代码按 locale 自动选 ASR 模型，whisper 只用于 en）。

**zh 全量 2020 = 1.41%**（准入 ≤1.56，与官方 910C 基线 1.414 持平）→ **达标**。此前 en 全量 3.03% 超线是**口径错误**（参照了错误的 en 阈值），不是 910B 模型问题 —— "910B WER 固有 ~3%" 的结论作废（仅对 en 成立）。

**要点**：
- 评测脚本默认口径已改 zh（`eval_seed_tts_wer.sh` locale 默认 zh，可 `LOCALE=en` 覆盖）
- WER/ASV 用 `--max-concurrency 4`（官方 CI 口径；并发 1/4 精度等价已验证，zh 32 条 0.66% 一致；吞吐 +33%）
- 性能评测（TTFT/TTFP/RTF）仍并发 1（32×1 档）
- zh 转写模型 = funasr paraformer-zh + zhconv（`seed_tts_eval.py` `_ensure_zh_asr`），本地已可用

相关：[[wer-full-validation-rule]]（32 条假象教训仍适用：全量才是有效口径）、[[perf-adoption-5pct]]。
