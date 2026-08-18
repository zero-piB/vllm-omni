# C20：DiT QKV 融合 + adaLN 预计算（stage2）

## 结论

**TTFP 均值 -11.2ms（三轮 32 条：1022.2 / 1041.5 / 1034.0 vs E9 台账 1043.8），达标 10ms 绝对值门槛采纳**；WER 逐位一致（1.31%×3）、RTF 0.633（改善方向）、TTFT 噪声内。全量验证：WER 1088 + ASV 1088。

## 改动（两个层面，数学等价，WER 逐位一致）

### 1. C15：DiT Attention QKV 三 Linear 惰性融合（decoder_dit.py 双副本）

`Attention.forward_chunk` 的 `to_q/to_k/to_v`（3× Linear 512→512）融合为单次 `F.linear(x, w, b)`（512→1536，权重/bias cat）。
惰性构建（`_qkv_fused`，首次调用时 cat）—— 避开 `load_state_dict(strict=True)` 的键不匹配。
省 2 次 GEMM 发射/block/step = 160 次/块。
> 注意：fp16 累加顺序与三 GEMM 有差异（不同 kernel tiling），32 条 WER 验证在噪声内（1.31% 逐位口径一致）。

### 2. C17：adaLN/t_embedder 全表预计算（batched_token2wav.py + decoder_dit.py 双副本）

`_ensure_adaLN_table()`：首次 decode 前按 `_decode_cfm` 完全一致的时间路径（timeline/cos/dt 累加）计算
5 个 timestep 的 `t_embedder` 输出（5,1,512）与全部 block 的 adaLN 调制（每 block 5×9×512）+ final_layer（5×2×512），~0.7MB。
每步 `_estimator_step` 只做 index/slice（`t_emb_table[step_idx]` / `block_mods[b][step_idx]`）。
省 ~160 次发射/块（16 block × 5 步 × adaLN SiLU+Linear + t_embedder 重复计算）。
decoder_dit 的 `DiTBlock.forward_chunk` / `FinalLayer.forward` / `blocks_forward_chunk` 增加可选参数（`adalaLN_cache` / `block_mods` / `final_mod`），None 时走原路径（向后兼容）。

## 为什么动这两个（profiling 依据）

msprof：stage2 每块 222ms = enc 13 + cfm（DiT）171 + hift 38；cfm 是 ~3400 次小 kernel 发射主导
（12 万个 0-500μs kernel，launch 平均 12.7μs），非算力。QKV 融合 + adaLN 预计算削减发射数。

## 已证伪（同轮次）

- C14：DiT NPU graph 捕获（F4 变体）—— `Cannot run aclop operators during NPU graph capture (Conv2D)`，
  昇腾 graph capture 只支持 aclnn 算子，DiT 的 CausalConv1d 是 aclop。**图化路径在昇腾上不可行**（F9）。
- C16：KV 缓存原地写 —— 实际 att 全长仅 100-300 帧，拷贝收益 ~1-2ms/块，不足；跨块持久 buffer 内存风险大。

## 应用方法（新机器）

```bash
# 1. vllm-omni 仓库
git -C $REPO_DIR apply 6_optimization/patches/0002-c20-adaLN-precompute.patch
# 2. stepaudio2 包（site-packages）—— 两个副本都要
pip show stepaudio2 | grep Location   # 记下 site-packages 路径 SP
cp $SP/stepaudio2/cosyvoice2/flow/decoder_dit.py{,.bak}
cp $SP/cosyvoice2/flow/decoder_dit.py{,.bak}
git -C <临时仓库> apply 6_optimization/patches/0002-c20-decoder-dit-stepaudio2.patch
git -C <临时仓库> apply 6_optimization/patches/0002-c20-decoder-dit-cosyvoice2.patch
# （patch 内路径以 site-packages 为根；或将 site-packages 初始化为 git 仓库后 apply）
# 3. 清 __pycache__（flow 目录下）
find $SP/stepaudio2 $SP/cosyvoice2 -name "__pycache__" -path "*flow*" -exec rm -rf {} +
```

回滚：`git -C $REPO_DIR apply -R` + 从 .bak 恢复（或 patch -R）。
