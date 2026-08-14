"""Zero-centered RMSNorm — Qwen3.5 SOTA.

Reference: Qwen3.5 (2026). Standard RMSNorm initializes gamma=1 (identity
at start). Zero-centered RMSNorm initializes gamma=0 and shifts the
post-norm output by +1 (effectively identity at init, but with gamma
starting at 0 the gradient flow is better for deep networks).

Mathematically:
  standard RMSNorm:  out = (x / ||x||) * (1 + gamma)  # gamma init = 0
  zero-centered:     out = (x / ||x||) * gamma + x     # gamma init = 1, but
                                                       # represents deviation
                                                       # from identity

Qwen3.5 paper finds the zero-centered form gives better gradient signal
in deep networks (60+ layers).

Implementation: subclass RMSNorm with a different init and forward.
Drop-in replacement.
"""

from __future__ import annotations

import torch
from torch import nn


class ZeroCenteredRMSNorm(nn.Module):
    """RMSNorm with zero-centered init (Qwen3.5).

    `out = x * rsqrt(mean(x²) + eps) * gamma + x`

    The `+ x` term makes the operation identity at init when gamma=0.
    Training then learns gamma as deviations from identity.

    Args:
        dim: feature dimension.
        eps: numerical stability.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.gamma = nn.Parameter(torch.zeros(dim))  # init at 0 (identity)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.gamma + x
