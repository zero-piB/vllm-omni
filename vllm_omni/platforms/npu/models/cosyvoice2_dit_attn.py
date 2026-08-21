# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NPU patches for CosyVoice2 / Token2Wav DiT attention.

CosyVoice2's DiT ``Attention.forward`` builds a key-padding style mask
``(B, 1, 1, S)`` via ``mask.unsqueeze(1)`` and feeds it to
``F.scaled_dot_product_attention``. On Ascend that call is routed to
``npu_fusion_attention`` / ``aclnnFlashAttentionScore``, which only accepts
mask shapes ``[B,N,S,S] / [B,1,S,S] / [1,1,S,S] / [S,S]`` and fails with
error **161001** (tiling / parameter invalid) for ``[B,1,1,S]``.

This module:
1. Expands DiT attention masks to ``[B, 1, S, S]`` before SDPA.
2. Provides a MATH-backend SDPA context so inference can avoid the fused FA
   kernel entirely when the platform still incorrectly routes SDPA.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager, nullcontext

import torch
import torch.nn.functional as F
from vllm.logger import init_logger

logger = init_logger(__name__)

_PATCHED = False


def _expand_attn_mask_for_npu(
    attn_mask: torch.Tensor | None,
    q_len: int,
    kv_len: int | None = None,
) -> torch.Tensor | None:
    """Expand CosyVoice key-padding masks to Ascend FA-compatible shapes."""
    if attn_mask is None:
        return None
    kv_len = kv_len if kv_len is not None else q_len

    # (B, 1, S) key-padding -> (B, S_q, S_kv) then unsqueeze heads below.
    if attn_mask.dim() == 3 and attn_mask.shape[-2] == 1:
        attn_mask = attn_mask.expand(-1, q_len, -1)
    if attn_mask.dim() == 3:
        # (B, S_q, S_kv) -> (B, 1, S_q, S_kv)
        attn_mask = attn_mask.unsqueeze(1)
    if attn_mask.dim() == 4 and attn_mask.shape[-2] == 1 and kv_len > 1:
        # (B, 1, 1, S) -> (B, 1, S_q, S)
        attn_mask = attn_mask.expand(-1, -1, q_len, -1)
    return attn_mask


def _patched_attention_forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
    b, t, c = x.shape

    q = self.to_heads(self.to_q(x))
    k = self.to_heads(self.to_k(x))
    v = self.to_heads(self.to_v(x))

    q = self.q_norm(q)
    k = self.k_norm(k)

    attn_mask = _expand_attn_mask_for_npu(attn_mask, q_len=t, kv_len=t)
    x = F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=attn_mask,
        dropout_p=self.attn_drop.p if self.training else 0.0,
    )
    x = x.transpose(1, 2).reshape(b, t, -1)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x


def _patched_attention_forward_chunk(
    self,
    x: torch.Tensor,
    att_cache: torch.Tensor | None = None,
    attn_mask: torch.Tensor | None = None,
):
    b, t, c = x.shape

    q = self.to_heads(self.to_q(x))
    k = self.to_heads(self.to_k(x))
    v = self.to_heads(self.to_v(x))

    q = self.q_norm(q)
    k = self.k_norm(k)

    if att_cache is not None:
        k_cache, v_cache = att_cache.chunk(2, dim=3)
        k = torch.cat([k, k_cache], dim=2)
        v = torch.cat([v, v_cache], dim=2)

    new_att_cache = torch.cat([k, v], dim=3)
    kv_len = k.shape[2]
    if attn_mask is not None:
        attn_mask = _expand_attn_mask_for_npu(attn_mask, q_len=t, kv_len=kv_len)
    x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
    x = x.transpose(1, 2).reshape(b, t, -1)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x, new_att_cache


@contextmanager
def npu_math_sdpa_context() -> Iterator[None]:
    """Force SDPA MATH backend so Ascend does not call fused FA."""
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        with sdpa_kernel(SDPBackend.MATH):
            yield
    except Exception:
        # Older torch / missing backend enum — just run as-is.
        with nullcontext():
            yield


def _disable_upsample_encoder_compile() -> None:
    """Run CosyVoice2 UpsampleConformerEncoderV2 eagerly on Ascend.

    ``forward_chunk`` is decorated with ``@torch.compile(dynamic=True,
    backend="eager")`` upstream. The backend gives no speedup (still eager),
    but Dynamo tracing of the relative-position attention with a KV cache
    creates a symbolic add ``matrix_ac(..., chunk+cache) + matrix_bd(...,
    (pos//2)+1)`` whose sizes Ascend torch's fake-tensor ``infer_size``
    cannot reconcile, crashing on the 2nd+ streaming chunk. Unwrapping the
    compile and disabling Dynamo keeps concrete shapes (always equal).
    """
    try:
        from cosyvoice2.transformer import upsample_encoder_v2
    except ImportError:
        return

    enc_cls = getattr(upsample_encoder_v2, "UpsampleConformerEncoderV2", None)
    if enc_cls is None:
        return

    fn = enc_cls.forward_chunk
    orig = getattr(fn, "_torchdynamo_orig_callable", None) or getattr(fn, "__wrapped__", fn)
    enc_cls.forward_chunk = torch._dynamo.disable(orig)  # type: ignore[method-assign]
    logger.info("Disabled torch.compile on CosyVoice2 UpsampleConformerEncoderV2.forward_chunk (Ascend eager)")


def _npu_modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Fuse ``x * (1 + scale) + shift`` into a single addcmul kernel."""
    return torch.addcmul(shift, x, 1 + scale)


def apply_c67_modulate_fusion() -> None:
    """Fuse the DiT adaLN ``modulate`` path into one addcmul kernel.

    ``modulate`` is the single entry point for all DiT adaLN paths (7 call
    sites: 3 in DiTBlock.forward_chunk, 3 in DiTBlock.forward, 1 in
    FinalLayer.forward); replacing the module-level function covers all of
    them. ``1 + scale`` is computed inline instead of being folded into the
    adaLN table: folding would require table-semantics changes plus patching
    the non-streaming DiTBlock.forward and FinalLayer.forward, with
    silent-miscompute risk.

    NOTE: must run after ``cosyvoice2.flow.decoder_dit`` is imported
    (i.e. after model load); MiniCPM-o's BatchedToken2Wav bypasses
    ``npu_token2wav_sdpa_context``, so this is wired into
    ``_patched_ensure_models_loaded`` instead of ``apply_cosyvoice2_dit_attn_npu_patch``.
    """
    try:
        from cosyvoice2.flow import decoder_dit
    except ImportError:
        logger.debug("cosyvoice2 not installed; skip modulate fusion")
        return

    decoder_dit.modulate = _npu_modulate  # type: ignore[method-assign]
    logger.info("Applied DiT modulate addcmul fusion (NPU)")


def _patched_block_forward_chunk(
    self,
    x: torch.Tensor,
    c: torch.Tensor,
    cnn_cache: torch.Tensor | None = None,
    att_cache: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
    adalaLN_cache: torch.Tensor | None = None,
):
    """DiTBlock.forward_chunk with gate-residual fused via addcmul.

    ``x = x + gate * out`` (3 sites per block) is two kernels (Mul + Add);
    ``torch.addcmul(x, out, gate)`` is one, numerically ``x + gate*out``.
    The body is verbatim upstream except those three lines; ``modulate``
    resolves to the C67-patched module function at call time.
    """
    from cosyvoice2.flow import decoder_dit

    if adalaLN_cache is not None:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp, shift_conv, scale_conv, gate_conv = adalaLN_cache
    else:
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
            shift_conv,
            scale_conv,
            gate_conv,
        ) = self.adaLN_modulation(c).chunk(9, dim=-1)
    mod = decoder_dit.modulate
    x_att, new_att_cache = self.attn.forward_chunk(mod(self.norm1(x), shift_msa, scale_msa), att_cache, mask)
    x = torch.addcmul(x, x_att, gate_msa)
    x_conv, new_cnn_cache = self.conv.forward_chunk(mod(self.norm3(x), shift_conv, scale_conv), cnn_cache)
    x = torch.addcmul(x, x_conv, gate_conv)
    x = torch.addcmul(x, self.mlp(mod(self.norm2(x), shift_mlp, scale_mlp)), gate_mlp)
    return x, new_cnn_cache, new_att_cache


def apply_c74_gate_residual_fusion() -> None:
    """Fuse DiT gate-residual ``x + gate*out`` into addcmul (3 per block).

    Every DiTBlock (7 per step, 3 steps per chunk, ~24 chunks/req -> ~500
    block executions) does three ``x = x + gate*out`` pairs (attention/conv/
    mlp residuals) = 6 elementwise kernels -> 3 addcmul kernels. Saves ~1500
    kernels/request of host launch + per-call overhead. Must run after model
    load like ``apply_c67_modulate_fusion`` (wired into
    ``_patched_ensure_models_loaded``).
    """
    try:
        from cosyvoice2.flow import decoder_dit
    except ImportError:
        logger.debug("cosyvoice2 not installed; skip gate-residual fusion")
        return

    block_cls = getattr(decoder_dit, "DiTBlock", None)
    if block_cls is None:
        return
    block_cls.forward_chunk = _patched_block_forward_chunk  # type: ignore[method-assign]
    logger.info("Applied DiT gate-residual addcmul fusion (NPU)")


def apply_cosyvoice2_dit_attn_npu_patch() -> None:
    """Monkey-patch CosyVoice2 DiT Attention for Ascend FA mask constraints."""
    global _PATCHED
    if _PATCHED:
        return

    try:
        from cosyvoice2.flow import decoder_dit
    except ImportError:
        logger.debug("cosyvoice2 not installed; skip DiT attn NPU patch")
        return

    attn_cls = getattr(decoder_dit, "Attention", None)
    if attn_cls is None:
        return

    attn_cls.forward = _patched_attention_forward  # type: ignore[method-assign]
    attn_cls.forward_chunk = _patched_attention_forward_chunk  # type: ignore[method-assign]
    _disable_upsample_encoder_compile()
    _PATCHED = True
    logger.info("Applied CosyVoice2 DiT Attention NPU patch (expand attn_mask to Bx1xSxS for aclnnFlashAttentionScore)")
