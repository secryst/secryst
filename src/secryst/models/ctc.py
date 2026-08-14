"""CTCModel — Connectionist Temporal Classification for Thai→IPA.

Reference: Graves et al. 2006 (ICML).

Alternative to seq2seq + teacher forcing. CTC has:

  - **No decoder LM prior** can form — output is purely conditional on input.
  - **No exposure bias** — single forward pass, no teacher forcing.
  - **No EOS prediction needed** — output length derived from input.
  - **Monotonic alignment** built-in.

These properties directly address the mode collapse that all seq2seq variants
of secryst hit (see `secryst-mode-collapse.md` memory). With CTC there is no
place for a "language model prior" to form — every output token MUST be
explainable by some input position.

Architecture:
  encoder: ModernEncoder (RoPE + mHC + SwiGLU + RMSNorm)  ← reuse existing
  head: Linear(dim, output_vocab + 1)  ← +1 for CTC blank token
  ctc_loss: torch.nn.CTCLoss(blank=output_vocab)

The CTC blank token is the "no output at this position" placeholder. CTC
decoders collapse consecutive identical tokens and drop blanks.

Decoding:
  - Greedy: argmax + collapse repeats + drop blanks.
  - Beam search (optional, more complex): prefix beam search over the
    CTC output distribution.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from ..constants import EOS_ID, PAD_ID
from .modern import ModernEncoder


class CTCModel(nn.Module):
    """Encoder + linear head, trained with CTC loss.

    Args:
        input_vocab_size: source vocabulary size (Thai chars).
        output_vocab_size: target vocabulary size (IPA tokens). A blank
            token with ID = output_vocab_size is added internally.
        dim: model hidden dim.
        layers: encoder layer count.
        heads: attention head count.
        ff_dim: FFN intermediate size.
        dropout: dropout rate.
        max_len: maximum input length.
        pad_id: source PAD token ID (for masking).
        rope_base: RoPE base frequency.
        sk_iters: Sinkhorn-Knopp iterations for mHC.
    """

    def __init__(
        self,
        input_vocab_size: int,
        output_vocab_size: int,
        dim: int = 256,
        layers: int = 8,
        heads: int = 8,
        ff_dim: int = 1024,
        dropout: float = 0.1,
        max_len: int = 128,
        pad_id: int = PAD_ID,
        rope_base: float = 10000.0,
        sk_iters: int = 20,
        resformer: dict | None = None,
    ) -> None:
        super().__init__()
        self.input_vocab_size = input_vocab_size
        self.output_vocab_size = output_vocab_size
        # Blank token is the LAST id (= output_vocab_size). Real tokens occupy
        # [0, output_vocab_size). CTC convention: blank id is a separate slot.
        self.blank_id = output_vocab_size
        self.pad_id = pad_id
        self.dim = dim
        self.max_len = max_len

        self.encoder = ModernEncoder(
            input_vocab_size, dim, layers, heads, ff_dim,
            dropout=dropout, max_len=max_len, pad_id=pad_id,
            rope_base=rope_base, sk_iters=sk_iters,
            resformer=resformer,
        )
        # Output projection: dim → (vocab + 1 blank).
        self.head = nn.Linear(dim, output_vocab_size + 1)
        # LogSoftmax is built into CTCLoss but we use it explicitly for inference.
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.ctc_loss = nn.CTCLoss(blank=self.blank_id, reduction="mean", zero_infinity=True)

    def forward(
        self,
        src: torch.Tensor,
        src_lengths: torch.Tensor,
        target: torch.Tensor | None = None,
        target_lengths: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            src: (B, T_src) input IDs.
            src_lengths: (B,) true source lengths.
            target: (B, T_tgt) target IDs concatenated (no blanks, no repeats).
                Required for training; not used at inference.
            target_lengths: (B,) lengths of each target.

        Returns: dict with:
          - 'log_probs': (T_src, B, V+1) — CTC output distribution.
          - 'loss': scalar CTC loss (NaN if target is None).
        """
        hidden, src_kpm = self.encoder(src)
        logits = self.head(hidden)  # (B, T_src, V+1)
        log_probs = self.log_softmax(logits)
        # CTCLoss expects (T, B, V) — transpose.
        log_probs_t = log_probs.transpose(0, 1)  # (T_src, B, V+1)

        if target is not None and target_lengths is not None:
            loss = self.ctc_loss(log_probs_t, target, src_lengths, target_lengths)
        else:
            loss = torch.tensor(float("nan"), device=log_probs.device)

        return {"log_probs": log_probs_t, "loss": loss}

    @torch.no_grad()
    def greedy_decode(self, src: torch.Tensor, src_lengths: torch.Tensor) -> list[list[int]]:
        """Greedy CTC decode: argmax, collapse repeats, drop blanks.

        Args:
            src: (B, T_src)
            src_lengths: (B,)

        Returns: list of B lists of token IDs (no blanks, no PAD/BOS/EOS).
        """
        self.eval()
        out = self.forward(src, src_lengths)
        log_probs = out["log_probs"]  # (T, B, V+1)
        preds = log_probs.argmax(dim=-1).transpose(0, 1)  # (B, T)
        B, T = preds.shape

        decoded: list[list[int]] = []
        for b in range(B):
            real_len = int(src_lengths[b].item())
            tokens: list[int] = []
            prev = -1
            for t in range(real_len):
                tok = int(preds[b, t].item())
                # CTC collapsing: skip repeats and blanks.
                if tok != prev and tok != self.blank_id:
                    tokens.append(tok)
                prev = tok
            # Strip special tokens (PAD/BOS/EOS shouldn't appear but defense).
            tokens = [t for t in tokens if t not in (PAD_ID, EOS_ID)]
            decoded.append(tokens)
        return decoded


def build_ctc_model(cfg: dict[str, Any]) -> CTCModel:
    """Factory: build CTCModel from a config dict."""
    from ..constants import INPUT_VOCAB_SIZE, OUTPUT_VOCAB_SIZE
    m = cfg.get("model", {})
    return CTCModel(
        input_vocab_size=m.get("input_vocab_size", INPUT_VOCAB_SIZE),
        output_vocab_size=m.get("output_vocab_size", OUTPUT_VOCAB_SIZE),
        dim=m.get("dim", 256),
        layers=m.get("layers", 8),
        heads=m.get("heads", 8),
        ff_dim=m.get("ff_dim", 1024),
        dropout=m.get("dropout", 0.1),
        max_len=m.get("max_len", 128),
        rope_base=m.get("rope_base", 10000.0),
        sk_iters=m.get("sk_iters", 20),
        resformer=m.get("resformer", None),
    )
