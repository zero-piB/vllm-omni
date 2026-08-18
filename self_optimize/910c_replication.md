# 910C 机器复刻清单（2026-08-18）

> 用户获得 910C 机器。此清单把 910B 环境与优化完整搬过去，并补 910B 上做不到的验证。

**910C 机器到手时的复刻清单**

**动机**: 比赛评分在官方 910C 环境跑；910C 上要复现 champion 配置并补 910B 上做不到的全量验证。

**步骤**:

1. **拷贝**：`submission/`（1_code yaml + 2_configs 脚本 + 6_optimization patches）+ `vllm-omni/`（opt/tts-performance 分支，champion 提交状态）+ `local_models/MiniCPM-o-4_5`（本地模型，GlusterFS 规避）+ vllm-ascend（版本以 910C 官方 pip 为准）
2. **环境**：CANN/torch_npu 版本可能不同 → vllm-ascend 匹配官方版本；headroom-venv 重建；`--dtype float16` 保持（官方 F16 档基线，bf16 精度教训）
3. **env.sh 5 变量**：VLLM/MODEL_DIR/REPO_DIR/RAW_DATA_DIR/DATA_DIR → `prepare_data.sh` 幂等生成数据
4. **启动+冒烟四件套 32 条**（WER/ASV/VideoMME/DO 128）→ perf 32×1
5. **910C 专属机会**：①ASV 1088 全量补跑（910B 32G 容器 OOM 欠的账，[[asv-full-32g-container-oom]]）；②C55 enable_static_kernel 试跑（910B 显存不足，910C 显存大可能可行）；③与 910C 官方基线对比（TTFT 333.27/TTFP 986.47/RTF 0.4423）
6. **铁律**：不跨 N 比性能（910C 只与 910C 基线比，不跨设备比）；E5/E6 yaml 配置不受 CANN 版本影响，源码 patch 需重新验证

相关：[[zh-wer-gate]]（WER zh 口径）、[[scoring-rtf-first]]（评分规则）、[[glusterfs-local-model]]（本地模型规避）
