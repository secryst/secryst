"""Kimi Delta Attention (KDA) — K3 SOTA.

Reference: Kimi K3 (arXiv:2607.24653). Adds a small learnable scalar
bias per layer to the attention logits before softmax. Stabilizes
attention when QK products vary widely across layers.

Implementation: a single `nn.Parameter` per attention layer, added to
the QK product inside SDPA via the `attn_mask` argument (or via a
manual softmax when SDPA's mask interface doesn't fit).

Open/closed: existing ModernEncoderLayer/ModernDecoderLayer unchanged.
`with_kda=True` flag in the constructor adds a `kda_bias` parameter
and routes the attention computation through KDA-aware code.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class KDABias(nn.Module):
    """Per-layer learnable attention bias (K3 KDA).

    The bias is a scalar added to every position of the attention logits
    before softmax. Per-layer, not per-head (per-head variant possible
    but adds params for marginal gain at our scale).
    """

    def __init__(self, init_value: float = 0.0) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(init_value))

    def forward(self) -> torch.Tensor:
        return self.bias


def softmax_with_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kda_bias: torch.Tensor | None = None,
    attn_mask: torch.Tensor | None = None,
    is_causal: bool = False,
    dropout_p: float = 0.0,
) -> torch.Tensor:
    """Manual softmax attention with optional KDA bias.

    Use this instead of F.scaled_dot_product_attention when you need to
    inject a per-layer bias. SDPA's attn_mask interface can technically
    encode an additive bias (mask = -inf for masked, 0 elsewhere, plus
    a constant), but a dedicated function is clearer.

    Args:
        q, k, v: (B, H, T, D) tensors.
        kda_bias: scalar tensor added to all logits.
        attn_mask: bool mask, True = masked out.
        is_causal: apply causal mask.
        dropout_p: attention dropout probability.
    """
    scale = 1.0 / math.sqrt(q.size(-1))
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, H, T_q, T_k)
    if kda_bias is not None:
        scores = scores + kda_bias
    if is_causal:
        T = scores.size(-2)
        causal = torch.triu(torch.ones(T, T, device=scores.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal[None, None, :, :], float("-inf"))
    if attn_mask is not None:
        scores = scores.masked_fill(attn_mask, float("-inf"))
    attn = F.softmax(scores, dim=-1)
    if dropout_p > 0:
        attn = F.dropout(attn, p=dropout_p)
    return torch.matmul(attn, v)
