# RTF ≤0.27 冲刺预案(2026-08-21 晚记录)

现状: champion = RTF 0.28(32×1)/0.29(全量), 提交物内空间已尽(55 条证伪)。
剩余两条机会均在提交物边界外, 需官方环境或自定义算子工程。

## 方案 A: 采样链 Ascend C 融合算子(RTF 上限 -0.017)

**目标**: stage1 的 `_sample_audio_code` 链(实测 1.4ms/token × 81 = 113ms/请求)。
链: head_code Linear(768→6562) → cast/div → rep_penalty(bincount+pow+where) →
top_k_top_p(clone+sort+softmax+cumsum+scatter+masked_fill+topk+masked_fill) →
softmax → multinomial → .item() 同步。

**可融合部分**(数学逐位等价):
1. rep_penalty 的 bincount+pow+where → 1 个 Ascend C kernel
2. top_k_top_p 的 8-op 链 → 1 个 kernel(sort 保留原算法)
3. softmax+multinomial → 1 个 kernel(RNG 用 NPU 随机数接口, 需与 torch.multinomial 逐位一致——
   风险点: RNG 序列不同会改变采样结果 → 需 32 冒烟 + 全量 WER/ASV)

**接口**: aclop 自定义算子(参考 cann-samples dav-2201 单卡样例),
注册进 vllm-omni 的 npu 平台(platforms/npu/), 提交物内调用。

**预期**: 链 1.4ms → ~0.6ms(保留 .item() 同步 0.4ms)→ 省 ~0.8ms/token × 81 = 65ms → RTF -0.017。

**风险/成本**: 工程 ~4-6h(Ascend C + aclop 注册 + 数值验证); RNG 逐位一致难(若不一致
需接受采样输出变化并全量验证精度); 超过单轮 2h 预算 → 需拆两轮或接受跨日。

## 方案 B: stage1 TP2 官方环境验证(RTF -0.02 量级)

**本地失败原因**: HCCL 通信端口冲突(EI0020, IP 192.28.2.197:16666 already bound)
—— stage0 TP2 组已占用, 第二个 TP 组无法绑同一网卡端口。CANN 层限制, 无用户配置。

**官方环境可行性**: 官方 910C 的网卡/端口配置可能不同(多网卡或独立端口分配)。
Talker 为 vllm LlamaModel(o_proj/down_proj RowParallel), TP2 下 hidden 完整、
head_code 采样无需 gather —— 理论上仅需 yaml 加 stage1 tensor_parallel_size: 2,
devices: "0,1"。

**验证步骤**(官方环境):
1. cp champion yaml → stage1 加 `tensor_parallel_size: 2` + `devices: "0,1"`
2. 启动冒烟(服务就绪 + 1 条 seed-tts)
3. 若启动成功 → perf 32×1(预期 stage1 device 2ms→1.1ms/token → RTF 0.28→0.26)
4. 精度: WER zh 32 + SIM 32 → 全量

## 决策建议

- 提交物按当前 champion(0.28, 精度全绿)先提交 —— 分数安全垫
- 方案 A 与 B 均不阻塞提交(提交物内代码已是最终形态)
- 若官方环境可验证 B → 最大杠杆(RTF -0.02); A 作为 B 失效时的后备
