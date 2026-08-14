"""Multi-Token Prediction (MTP) — DeepSeek V4 pretraining objective.

Reference: DeepSeek V4 (arXiv:2606.19348). Standard MLM predicts one
token per masked position. MTP predicts N tokens per position: the
current token + N-1 future tokens. Increases sample efficiency by
extracting more signal per forward pass.

Loss: sum of per-token CE with geometric weights.
  weight_i = 1 / sqrt(i + 1) for i in 0..N-1

Use during PRETRAINING only. At fine-tune time, drop the MTP heads.

Implementation: a single shared trunk + N parallel prediction heads.
The shared trunk is the encoder; the heads are independent linear layers
with tied input embedding (output weights = embedding weights).

Open/closed: standalone module. The existing MLM pretrain can dispatch
to MTP via a config flag (`pretrain_method: mtp`).
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


class MTPHead(nn.Module):
    """Multi-Token Prediction head: N parallel linear projections.

    Each head i predicts the token at position+i (i in 0..N-1).
    All heads share the same input (encoder hidden state).

    Args:
        dim: hidden dim of input.
        vocab_size: output vocab.
        n_predict: number of tokens to predict per position.
        tie_to_embedding: if True, all head output weights are tied to
            a single shared embedding (saves params + helps generalization).
    """

    def __init__(
        self,
        dim: int,
        vocab_size: int,
        n_predict: int = 2,
        tie_to_embedding: bool = True,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.vocab_size = vocab_size
        self.n_predict = n_predict
        self.tie_to_embedding = tie_to_embedding
        # Per-head small projection (RMSNorm + Linear).
        self.head_norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(n_predict)])
        if tie_to_embedding:
            # Single shared weight, used by all heads.
            self.shared_weight = nn.Parameter(torch.empty(vocab_size, dim))
            nn.init.normal_(self.shared_weight, std=0.02)
        else:
            self.head_linears = nn.ModuleList([
                nn.Linear(dim, vocab_size, bias=False) for _ in range(n_predict)
            ])
        self.biases = nn.ParameterList([
            nn.Parameter(torch.zeros(vocab_size)) for _ in range(n_predict)
        ])

    def forward(self, hidden: torch.Tensor) -> list[torch.Tensor]:
        """Return N logit tensors, one per prediction head.

        Args:
            hidden: (B, T, dim) from the encoder.

        Returns: list of N tensors, each (B, T, vocab_size).
        """
        out: list[torch.Tensor] = []
        for i in range(self.n_predict):
            h = self.head_norms[i](hidden)
            if self.tie_to_embedding:
                logits = F.linear(h, self.shared_weight, self.biases[i])
            else:
                logits = self.head_linears[i](h) + self.biases[i]
            out.append(logits)
        return out


def mtp_loss(
    logits_list: list[torch.Tensor],
    target: torch.Tensor,
    ignore_index: int = 0,
    geometric_weights: bool = True,
) -> torch.Tensor:
    """Sum of per-token CE with optional geometric (1/sqrt(i+1)) weighting.

    Args:
        logits_list: N tensors of (B, T, V).
        target: (B, T + N - 1) — for head i, we compare logits_list[i][:, :, :]
            against target shifted by i. Practically: target contains the
            original sequence + N-1 future tokens.
        ignore_index: skip these positions (PAD).
        geometric_weights: if True, weight head i by 1/sqrt(i+1).
    """
    n_predict = len(logits_list)
    total = torch.tensor(0.0, device=logits_list[0].device, dtype=logits_list[0].dtype)
    B, T, V = logits_list[0].shape
    for i in range(n_predict):
        # Head i predicts token at position i (relative to current).
        # We need target[:, i:i+T] for comparison.
        target_slice = target[:, i:i + T]
        flat_logits = logits_list[i].reshape(-1, V)
        flat_target = target_slice.reshape(-1)
        ce = F.cross_entropy(flat_logits, flat_target, ignore_index=ignore_index)
        if geometric_weights:
            weight = 1.0 / math.sqrt(i + 1)
        else:
            weight = 1.0
        total = total + weight * ce
    return total
