#!/usr/bin/env python3
"""Q1: MiniCPM-o stage0（llm.* 前缀）W8A16 权重导出脚本（一次性，确定性 RTN）。

语义（已 NPU 闭环验证 q1_antiquant_semantics.py）：
    scale = (max - min) / 255                       # per output channel, fp16 存储
    offset = min + 128 * scale                      # fp16 存储（算子内置 deq = q*scale + offset）
    q = clamp(round((w - min)/scale) - 128, -128, 127)  # int8 存储

只量化 llm.* 前缀的 Linear 权重；vpm/apm/resampler/tts/embed_tokens/lm_head 保持 fp16。
vLLM V1 的 Qwen3Attention 用 fused qkv_proj（QKVParallelLinear）→ q/k/v 三段分别量化后
合并为 qkv_proj.weight（int8 concat）+ qkv_proj.weight_scale/offset（concat）。
确定性（无校准、无随机）。

用法: python3 quant_export_w8a16.py [src_dir] [dst_dir]
"""
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
ATTN_RE = re.compile(r"(llm\.model\.layers\.\d+\.self_attn)\.(q|k|v)_proj\.weight$")
MLP_RE = re.compile(r"(llm\.model\.layers\.\d+\.mlp)\.(gate|up)_proj\.weight$")


def quantize_weight(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """per-channel asymmetric RTN → (q_int8 (out,in), scale fp16 (out,), offset fp16 (out,))"""
    w = w.float()
    wmin = w.min(dim=1).values
    wmax = w.max(dim=1).values
    scale = (wmax - wmin) / 255.0
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    q = torch.round((w - wmin.unsqueeze(1)) / scale.unsqueeze(1)) - 128
    q = q.clamp(-128, 127).to(torch.int8)
    offset = wmin + 128 * scale
    return q, scale.half().unsqueeze(1), offset.half().unsqueeze(1)  # (out,1) 匹配 get_perchannel_param


def should_quantize(key: str) -> bool:
    if not key.startswith("llm.") or not key.endswith(".weight"):
        return False
    if any(key.startswith(p) for p in SKIP_PREFIXES):
        return False
    if ATTN_RE.match(key):
        return True  # q/k/v → 合并 qkv_proj
    return any(key.rsplit(".weight", 1)[0].endswith(suf) for suf in QUANT_SUFFIXES)


def main() -> None:
    src = Path(sys.argv[1] if len(sys.argv) > 1 else "/workspace/shared_assets/models/OpenBMB/MiniCPM-o-4_5")
    dst = Path(sys.argv[2] if len(sys.argv) > 2 else "/workspace/local_models/MiniCPM-o-4_5-w8a16")
    dst.mkdir(parents=True, exist_ok=True)

    index = json.load(open(src / "model.safetensors.index.json"))
    shards = sorted({v for v in index["weight_map"].values()})

    n_quant = 0
    n_pass = 0

    # 全模型级组收集（支持跨分片组）
    attn_groups: dict[str, dict[str, tuple[str, str]]] = {}
    mlp_groups: dict[str, dict[str, tuple[str, str]]] = {}
    all_keys_by_shard: dict[str, list[str]] = {}
    for shard in shards:
        with safe_open(src / shard, framework="pt") as f:
            keys = list(f.keys())
        all_keys_by_shard[shard] = keys
        for key in keys:
            m = ATTN_RE.match(key)
            if m:
                attn_groups.setdefault(m.group(1), {})[m.group(2)] = (key, shard)
            m = MLP_RE.match(key)
            if m:
                mlp_groups.setdefault(m.group(1), {})[m.group(2)] = (key, shard)

    def group_shard(group: dict[str, tuple[str, str]]) -> str:
        return next(iter(group.values()))[1]

    def read_tensor(key: str, shard_name: str) -> torch.Tensor:
        with safe_open(src / shard_name, framework="pt") as f:
            return f.get_tensor(key)

    def merge_group(group: dict[str, tuple[str, str]], order: tuple[str, ...], out_base: str, tensors: dict):
        """per-channel 量化各段后按 vLLM 顺序合并为单权重 + scale/offset"""
        qs, ss, os_ = [], [], []
        for part in order:
            key, sh = group[part]
            q, scale, offset = quantize_weight(read_tensor(key, sh))
            qs.append(q); ss.append(scale); os_.append(offset)
        tensors[f"{out_base}.weight"] = torch.cat(qs, dim=0)
        tensors[f"{out_base}.weight_scale"] = torch.cat(ss, dim=0)
        tensors[f"{out_base}.weight_offset"] = torch.cat(os_, dim=0)

    # 组的归属片（第一键所在片）
    group_owner = {attn: group_shard(g) for attn, g in attn_groups.items()}
    group_owner.update({mlp: group_shard(g) for mlp, g in mlp_groups.items()})

    for shard in shards:
        tensors: dict[str, torch.Tensor] = {}
        # 本片负责的组（归属片 == 本片 且 完整）
        my_attn = {a: g for a, g in attn_groups.items() if group_owner[a] == shard and len(g) == 3}
        my_mlp = {m: g for m, g in mlp_groups.items() if group_owner[m] == shard and len(g) == 2}
        skip_keys = {k for g in my_attn.values() for k, _ in g.values()}
        skip_keys |= {k for g in my_mlp.values() for k, _ in g.values()}

        with safe_open(src / shard, framework="pt") as f:
            for key in all_keys_by_shard[shard]:
                if key in skip_keys:
                    continue
                t = f.get_tensor(key)
                if should_quantize(key):
                    q, scale, offset = quantize_weight(t)
                    base = key.rsplit(".weight", 1)[0]
                    tensors[key] = q
                    tensors[f"{base}.weight_scale"] = scale
                    tensors[f"{base}.weight_offset"] = offset
                    n_quant += 1
                else:
                    tensors[key] = t
                    n_pass += 1
            # 本片负责的合并组
            for attn, g in my_attn.items():
                merge_group(g, ("q", "k", "v"), f"{attn}.qkv_proj", tensors)
                n_quant += 1
            for mlp, g in my_mlp.items():
                merge_group(g, ("gate", "up"), f"{mlp}.gate_up_proj", tensors)
                n_quant += 1
        save_file(tensors, dst / shard, metadata={"format": "pt"})
        print(f"  {shard}: {len(tensors)} keys -> {dst / shard}")

    # index.json：qkv 合并键 + scale/offset 增补
    weight_map = dict(index["weight_map"])
    extra: dict[str, str] = {}
    drop: list[str] = []
    for key, shard_name in weight_map.items():
        m = ATTN_RE.match(key)
        if m:
            attn = m.group(1)
            extra[f"{attn}.qkv_proj.weight"] = shard_name
            extra[f"{attn}.qkv_proj.weight_scale"] = shard_name
            extra[f"{attn}.qkv_proj.weight_offset"] = shard_name
            drop.append(key)
        m = MLP_RE.match(key)
        if m:
            mlp = m.group(1)
            extra[f"{mlp}.gate_up_proj.weight"] = shard_name
            extra[f"{mlp}.gate_up_proj.weight_scale"] = shard_name
            extra[f"{mlp}.gate_up_proj.weight_offset"] = shard_name
            drop.append(key)
        elif key.endswith(".weight") and should_quantize(key):
            base = key.rsplit(".weight", 1)[0]
            extra[f"{base}.weight_scale"] = shard_name
            extra[f"{base}.weight_offset"] = shard_name
    for k in drop:
        weight_map.pop(k)
    weight_map.update(extra)
    index_out = {"metadata": index["metadata"], "weight_map": weight_map}
    json.dump(index_out, open(dst / "model.safetensors.index.json", "w"), indent=1)

    # config.json：加 quantization_config（block_name_to_quantize 挡 stage1 误伤；targets 用 re: 正则）
    cfg = json.load(open(src / "config.json"))
    cfg["quantization_config"] = {
        "quant_method": "compressed-tensors",
        "format": "int8",
        "block_name_to_quantize": ["thinker.llm.model.layers.*"],
        "config_groups": {
            "group_1": {
                "targets": ["re:^thinker\\.llm\\.model\\.layers\\..*"],
                "weights": {"num_bits": 8, "type": "int", "strategy": "channel", "symmetric": False},
                "input_activations": None,
            }
        },
    }
    json.dump(cfg, open(dst / "config.json", "w"), indent=1)

    # 复制其余文件（assets/tokenizer 等）
    for item in src.iterdir():
        if item.name in ("model-00001-of-00004.safetensors", "model-00002-of-00004.safetensors",
                         "model-00003-of-00004.safetensors", "model-00004-of-00004.safetensors",
                         "model.safetensors.index.json", "config.json"):
            continue
        if item.is_dir():
            shutil.copytree(item, dst / item.name, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dst / item.name)

    print(f"\n完成: quantized={n_quant} passed={n_pass} -> {dst}")


if __name__ == "__main__":
    main()
