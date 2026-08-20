#!/usr/bin/env python3
"""W8A8 动态 stage0 权重导出脚本（int8 per-channel 对称权重 + per-token 动态激活量化）。

派生自 quant_export_w8a8_static.py：不写 input_scale/input_offset（激活 scale 运行时 per-token 计算），config 的 input_activations 改 dynamic: true → 命中 AscendW8A8DynamicLinearMethod（npu_dynamic_quant + npu_quant_matmul pertoken_scale）。静态版 32 条 WER 3.44% 超线（截断/离群），动态版消除该风险。

Q1（quant_export_w8a16.py）骨架改造。语义对齐 vllm-ascend `W8A8` scheme
（AscendW8A8LinearMethod, w8a8_static.py，compressed-tensors 静态格式）：
    weight_scale   = max|w| / 127                          # per output channel, fp16 (N,1)
    q              = clamp(round(w / weight_scale), -128, 127)  # int8 对称
    input_scale    = act_absmax / 127                      # per-tensor 静态（校准产物，per layer）
    input_offset   = 0
    deq（算子内）    = (xq ⊗ wq) * (input_scale * weight_scale)   # 仅输出反量化，权重零展开

激活校准：先跑 quant_calibrate_act.py 产出 scales.json，本脚本读入嵌入（--scales 参数）。
只量化 llm.* 前缀 Linear；vpm/apm/resampler/tts/embed_tokens/lm_head 保持 fp16。
qkv/gate_up 合并、index/config 重写同 Q1。确定性（纯 RTN + 固定校准产物）。

用法: python3 quant_export_w8a8_dynamic.py [src_dir] [dst_dir] [--skip-layers 0-3,36-39]
--skip-layers: 敏感层回退名单（默认首4+末4，整层保持 fp16，不量化不合并不写 scale）。
               config targets 只枚举非 skip 层 → 这些层不匹配 scheme → 加载时保持原生 nn.Linear。
"""
import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

QUANT_SUFFIXES = ("o_proj", "down_proj")  # 独立 Linear（q/k/v 和 gate/up 合并处理）
SKIP_PREFIXES = ("llm.model.embed_tokens",)
ATTN_RE = re.compile(r"(llm\.model\.layers\.(\d+)\.self_attn)\.(q|k|v)_proj\.weight$")
MLP_RE = re.compile(r"(llm\.model\.layers\.(\d+)\.mlp)\.(gate|up)_proj\.weight$")
LAYER_RE = re.compile(r"llm\.model\.layers\.(\d+)\.")

DEFAULT_SKIP = "0-3,32-35"  # 36 层模型, 首4+末4


def parse_skip(spec: str) -> set[int]:
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = (int(x) for x in part.split("-", 1))
            out.update(range(a, b + 1))
        else:
            out.add(int(part))
    return out


def quantize_weight(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """per-channel symmetric RTN → (q_int8 (N,K), weight_scale fp16 (N,1))"""
    # 全程 fp16 中间量：float32 展开 ~13G/分片 + 服务 page cache 在 32G 容器直接 OOM
    # （2026-08-20 实测 exit 137）。scale 保持 (N,1)（原 unsqueeze(1) 会得 (N,1,1) 形状 bug）。
    wmax = w.abs().amax(dim=1, keepdim=True).float().clamp(min=1e-8)
    scale = (wmax / 127.0).half()  # (N,1) 匹配 get_perchannel_param
    q = torch.round(w / scale).clamp(-128, 127).to(torch.int8)
    return q, scale


def should_quantize(key: str, skip_layers: set[int]) -> bool:
    if not key.startswith("llm.") or not key.endswith(".weight"):
        return False
    if any(key.startswith(p) for p in SKIP_PREFIXES):
        return False
    m = LAYER_RE.search(key)
    if m and int(m.group(1)) in skip_layers:
        return False  # 敏感层整层回退
    if ATTN_RE.match(key):
        return True  # q/k/v → 合并 qkv_proj
    return any(key.rsplit(".weight", 1)[0].endswith(suf) for suf in QUANT_SUFFIXES)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("src_dir", nargs="?", default="/workspace/local_models/MiniCPM-o-4_5")
    parser.add_argument("dst_dir", nargs="?", default="/workspace/local_models/MiniCPM-o-4_5-w8a8")
    parser.add_argument("--skip-layers", default=DEFAULT_SKIP,
                        help="敏感层回退名单, 如 '0-3,36-39' (默认首4+末4)")
    args = parser.parse_args()
    src = Path(args.src_dir)
    dst = Path(args.dst_dir)
    skip_layers = parse_skip(args.skip_layers)
    dst.mkdir(parents=True, exist_ok=True)

    index = json.load(open(src / "model.safetensors.index.json"))
    shards = sorted({v for v in index["weight_map"].values()})

    n_quant = 0
    n_pass = 0
    n_act = 0

    attn_groups: dict[str, dict[str, tuple[str, str]]] = {}
    mlp_groups: dict[str, dict[str, tuple[str, str]]] = {}
    all_keys_by_shard: dict[str, list[str]] = {}
    all_layers: set[int] = set()
    for shard in shards:
        with safe_open(src / shard, framework="pt") as f:
            keys = list(f.keys())
        all_keys_by_shard[shard] = keys
        for key in keys:
            lm = LAYER_RE.search(key)
            if lm:
                all_layers.add(int(lm.group(1)))
            m = ATTN_RE.match(key)
            if m:
                if int(m.group(2)) in skip_layers:
                    continue  # 敏感层: q/k/v 不合并, 保持原样
                attn_groups.setdefault(m.group(1), {})[m.group(3)] = (key, shard)
            m = MLP_RE.match(key)
            if m:
                if int(m.group(2)) in skip_layers:
                    continue  # 敏感层: gate/up 不合并, 保持原样
                mlp_groups.setdefault(m.group(1), {})[m.group(3)] = (key, shard)

    def group_shard(group: dict[str, tuple[str, str]]) -> str:
        return next(iter(group.values()))[1]

    def read_tensor(key: str, shard_name: str) -> torch.Tensor:
        with safe_open(src / shard_name, framework="pt") as f:
            return f.get_tensor(key)

    def act_scale(base: str) -> torch.Tensor | None:
        """动态激活量化：无需校准 input_scale，恒 None（不写 input_scale 张量）。"""
        return None

    def merge_group(group: dict[str, tuple[str, str]], order: tuple[str, ...], out_base: str, tensors: dict):
        qs, ss = [], []
        for part in order:
            key, sh = group[part]
            q, scale = quantize_weight(read_tensor(key, sh))
            qs.append(q); ss.append(scale)
        tensors[f"{out_base}.weight"] = torch.cat(qs, dim=0)
        tensors[f"{out_base}.weight_scale"] = torch.cat(ss, dim=0)
        tensors[f"{out_base}.weight_offset"] = torch.zeros(tensors[f"{out_base}.weight_scale"].shape,
                                                           dtype=torch.float16)
        return act_scale(out_base) is not None

    group_owner = {attn: group_shard(g) for attn, g in attn_groups.items()}
    group_owner.update({mlp: group_shard(g) for mlp, g in mlp_groups.items()})

    for shard in shards:
        tensors: dict[str, torch.Tensor] = {}
        my_attn = {a: g for a, g in attn_groups.items() if group_owner[a] == shard and len(g) == 3}
        my_mlp = {m: g for m, g in mlp_groups.items() if group_owner[m] == shard and len(g) == 2}
        skip_keys = {k for g in my_attn.values() for k, _ in g.values()}
        skip_keys |= {k for g in my_mlp.values() for k, _ in g.values()}

        with safe_open(src / shard, framework="pt") as f:
            for key in all_keys_by_shard[shard]:
                if key in skip_keys:
                    continue
                t = f.get_tensor(key)
                if should_quantize(key, skip_layers):
                    base = key.rsplit(".weight", 1)[0]
                    q, scale = quantize_weight(t)
                    tensors[key] = q
                    tensors[f"{base}.weight_scale"] = scale
                    tensors[f"{base}.weight_offset"] = torch.zeros_like(scale)
                    n_quant += 1
                else:
                    tensors[key] = t
                    n_pass += 1
            for attn, g in my_attn.items():
                if merge_group(g, ("q", "k", "v"), f"{attn}.qkv_proj", tensors):
                    n_act += 1
                n_quant += 1
            for mlp, g in my_mlp.items():
                if merge_group(g, ("gate", "up"), f"{mlp}.gate_up_proj", tensors):
                    n_act += 1
                n_quant += 1
        save_file(tensors, dst / shard, metadata={"format": "pt"})
        print(f"  {shard}: {len(tensors)} keys -> {dst / shard}")

    weight_map = dict(index["weight_map"])
    extra: dict[str, str] = {}
    drop: list[str] = []
    for key, shard_name in weight_map.items():
        m = ATTN_RE.match(key)
        if m:
            if int(m.group(2)) in skip_layers:
                continue  # 敏感层: 保持原始 q/k/v 映射
            attn = m.group(1)
            for suffix in (".weight", ".weight_scale", ".weight_offset"):
                extra[f"{attn}.qkv_proj{suffix}"] = shard_name
            drop.append(key)
        m = MLP_RE.match(key)
        if m:
            if int(m.group(2)) in skip_layers:
                continue  # 敏感层: 保持原始 gate/up 映射
            mlp = m.group(1)
            for suffix in (".weight", ".weight_scale", ".weight_offset"):
                extra[f"{mlp}.gate_up_proj{suffix}"] = shard_name
            drop.append(key)
        elif key.endswith(".weight") and should_quantize(key, skip_layers):
            base = key.rsplit(".weight", 1)[0]
            for suffix in (".weight_scale", ".weight_offset"):
                extra[f"{base}{suffix}"] = shard_name
    for k in drop:
        weight_map.pop(k)
    weight_map.update(extra)
    index_out = {"metadata": index["metadata"], "weight_map": weight_map}
    json.dump(index_out, open(dst / "model.safetensors.index.json", "w"), indent=1)

    cfg = json.load(open(src / "config.json"))
    # targets 枚举非 skip 层 → find_matched_target 用完整层路径等值/正则匹配,
    # 不匹配的层保持 UnquantizedLinearMethod(原生 fp16 Linear)。
    quant_layers = sorted(all_layers - skip_layers)
    target_re = r"re:^thinker\.llm\.model\.layers\.(" + "|".join(map(str, quant_layers)) + r")\."
    cfg["quantization_config"] = {
        "quant_method": "compressed-tensors",
        # int-quantized 才能命中 _detect_quant_type 的 W8A8 分支：
        # is_activation_quantization_format 只认 naive/int/float-quantized（"int8" 不在名单，实测 NotImplementedError）
        "format": "int-quantized",
        "block_name_to_quantize": ["thinker.llm.model.layers.*"],
        "config_groups": {
            "group_1": {
                "targets": [target_re],
                "weights": {"num_bits": 8, "type": "int", "strategy": "channel", "symmetric": True},
                "input_activations": {
                    "num_bits": 8,
                    "type": "int",
                    "strategy": "token",
                    "symmetric": True,
                    "dynamic": True,
                },
            }
        },
    }
    json.dump(cfg, open(dst / "config.json", "w"), indent=1)

    for item in src.iterdir():
        if item.name in ("model-00001-of-00004.safetensors", "model-00002-of-00004.safetensors",
                         "model-00003-of-00004.safetensors", "model-00004-of-00004.safetensors",
                         "model.safetensors.index.json", "config.json"):
            continue
        if item.is_dir():
            shutil.copytree(item, dst / item.name, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dst / item.name)

    missing_act = n_quant - n_act
    print(f"\n完成: quantized={n_quant} (含激活静态 scale {n_act}) passed={n_pass} -> {dst}")
    print(f"敏感层回退: {len(skip_layers)} 层保持 fp16 -> {sorted(skip_layers)}; "
          f"量化层: {len(quant_layers)} -> targets={target_re}")
    if missing_act > 0:
        print(f"⚠  {missing_act} 层无校准 input_scale——这些层 weights 已量化但激活无 scale，"
              f"会导致 scheme 拒绝（W8A8 要求 input_activations）或加载错位；请补齐校准。")


if __name__ == "__main__":
    main()
