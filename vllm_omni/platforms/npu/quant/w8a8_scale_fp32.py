# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NPU patches for W8A8 quantized checkpoints (910C dtype fixes).

vllm-ascend 0.19.1rc2 ships two bugs hit on 910C (2026-08-20):

1. ``npu_quant_matmul`` (aclnnQuantMatmulWeightNz) scale tensors only
   accept [UINT64, BFLOAT16, INT64, FLOAT] — fp16 per-channel weight
   scales raise ``DT_FLOAT16 not implemented``.
2. Layers matching no ``config_groups`` target (embed_tokens, non-llm
   prefixes, sensitive layers excluded from quantization) hit a KeyError
   in ``get_scheme_dict`` instead of falling through to
   ``UnquantizedLinearMethod``.

Both fixes are re-implemented here so quantized MiniCPM checkpoints run
against the *official* vllm-ascend pip package:

1. Wrap ``AscendW8A8DynamicLinearMethod.process_weights_after_loading``
   and upgrade the per-channel scale tensors to fp32 afterwards. The
   official ``apply`` reads ``layer.weight_scale`` (or ``weight_1_scale``
   / ``weight_2_scale`` for chunked weights), so upgrading those tensors
   in place keeps the official apply() path untouched.
2. Wrap ``AscendCompressedTensorsConfig.get_scheme_dict`` and return
   None for unmatched layers before the upstream KeyError site.
"""

from __future__ import annotations

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)

_PATCHED = False
_original_process_weights_after_loading = None
_original_get_scheme_dict = None

_FP32_SCALE_ATTRS = ("weight_scale", "weight_1_scale", "weight_2_scale")


def _patched_process_weights_after_loading(self, layer: torch.nn.Module) -> None:
    assert _original_process_weights_after_loading is not None
    _original_process_weights_after_loading(self, layer)
    # aclnnQuantMatmulWeightNz scale only accepts [UINT64, BFLOAT16, INT64, FLOAT];
    # fp16 per-channel scales raise DT_FLOAT16 not implemented (910C, 2026-08-20).
    # Keep the official apply() reading the same attribute names.
    for attr in _FP32_SCALE_ATTRS:
        w = getattr(layer, attr, None)
        if w is not None and w.dtype != torch.float32:
            if isinstance(w, torch.nn.Parameter):
                # Parameter.to() yields a plain tensor on torch_npu; assign
                # .data in place to keep the parameter registered.
                w.data = w.data.to(torch.float32)
            else:
                setattr(layer, attr, w.to(torch.float32))


def _patched_get_scheme_dict(self, layer, layer_name=None):
    # Layers outside config_groups targets (embed_tokens, non-llm prefixes,
    # skipped sensitive layers) must fall through to UnquantizedLinearMethod;
    # upstream indexes target_scheme_map with None and raises KeyError.
    if self.target_scheme_map:
        from vllm_ascend.quantization.compressed_tensors_config import find_matched_target

        matched_target = find_matched_target(
            layer_name=layer_name,
            module=layer,
            targets=self.target_scheme_map.keys(),
            fused_mapping=self.packed_modules_mapping,
        )
        if matched_target is None:
            return None
    assert _original_get_scheme_dict is not None
    return _original_get_scheme_dict(self, layer, layer_name)


def apply_w8a8_scale_fp32_patch() -> None:
    """Monkey-patch vllm-ascend W8A8 dtype bugs (official-package safe)."""
    global _PATCHED, _original_process_weights_after_loading, _original_get_scheme_dict
    if _PATCHED:
        return

    try:
        from vllm_ascend.quantization.compressed_tensors_config import (
            AscendCompressedTensorsConfig,
        )
        from vllm_ascend.quantization.methods.w8a8_dynamic import (
            AscendW8A8DynamicLinearMethod,
        )
    except ImportError as e:
        logger.debug("vllm-ascend quantization unavailable; skip W8A8 fp32 patch: %s", e)
        return

    _original_process_weights_after_loading = (
        AscendW8A8DynamicLinearMethod.process_weights_after_loading
    )
    _original_get_scheme_dict = AscendCompressedTensorsConfig.get_scheme_dict

    AscendW8A8DynamicLinearMethod.process_weights_after_loading = (  # type: ignore[method-assign]
        _patched_process_weights_after_loading
    )
    AscendCompressedTensorsConfig.get_scheme_dict = (  # type: ignore[method-assign]
        _patched_get_scheme_dict
    )

    _PATCHED = True
    logger.info("Applied W8A8 fp32-scale + get_scheme_dict fallback patches (NPU)")
