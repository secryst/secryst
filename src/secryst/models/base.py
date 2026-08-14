"""Model dispatch — selects architecture by cfg.model.arch."""

from __future__ import annotations

from typing import Any

from torch import nn


def build_model(cfg: dict[str, Any]) -> nn.Module:
    """Dispatch on cfg.model.arch. Returns a model instance.

    Supported architectures:
      - `modern_seq2seq` (default): encoder-decoder Transformer for G2P.
      - `ctc`: encoder-only + CTC loss; escapes seq2seq mode collapse.
    """
    arch = cfg.get("model", {}).get("arch", "modern_seq2seq")
    if arch == "modern_seq2seq":
        from .seq2seq import build_seq2seq
        return build_seq2seq(cfg)
    if arch == "ctc":
        from .ctc import build_ctc_model
        return build_ctc_model(cfg)
    raise ValueError(f"unknown model arch: {arch!r}")
