"""Modern seq2seq Transformer for Thai→IPA G2P.

Wraps ModernEncoder + ModernDecoder into a single module that the
training loop and ONNX exporter can call.

Training: teacher forcing — decoder input is `[BOS] + target[:-1]`,
target is `target[1:] + [EOS]` shifted by one.

Inference: autoregressive decoding with optional beam search (see
`decoding/beam.py`).
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from ..constants import INPUT_VOCAB_SIZE, OUTPUT_VOCAB_SIZE, PAD_ID
from .modern import ModernDecoder, ModernEncoder


class ModernSeq2Seq(nn.Module):
    """Encoder-decoder Transformer for Thai→IPA."""

    def __init__(
        self,
        input_vocab_size: int = INPUT_VOCAB_SIZE,
        output_vocab_size: int = OUTPUT_VOCAB_SIZE,
        dim: int = 256,
        enc_layers: int = 6,
        dec_layers: int = 6,
        heads: int = 8,
        ff_dim: int = 1024,
        dropout: float = 0.1,
        max_len: int = 128,
        pad_id: int = PAD_ID,
        rope_base: float = 10000.0,
        sk_iters: int = 20,
        memory_dropout: float = 0.0,
        use_mhc_cross: bool = True,
        resformer: dict | None = None,
    ) -> None:
        super().__init__()
        self.pad_id = pad_id
        self.dim = dim
        self.max_len = max_len
        self.input_vocab_size = input_vocab_size
        self.output_vocab_size = output_vocab_size
        # Memory dropout: fraction of encoder-output positions zeroed before
        # cross-attention. Prevents the decoder from overfitting to specific
        # encoder patterns and forces it to use multiple positions. Helps
        # escape the "ignore encoder → mode collapse" attractor.
        self.memory_dropout = memory_dropout

        self.encoder = ModernEncoder(
            input_vocab_size, dim, enc_layers, heads, ff_dim,
            dropout=dropout, max_len=max_len, pad_id=pad_id,
            rope_base=rope_base, sk_iters=sk_iters,
            resformer=resformer,
        )
        self.decoder = ModernDecoder(
            output_vocab_size, dim, dec_layers, heads, ff_dim,
            dropout=dropout, max_len=max_len, pad_id=pad_id,
            rope_base=rope_base, sk_iters=sk_iters,
            use_mhc_cross=use_mhc_cross,
        )
        if memory_dropout > 0:
            self.memory_drop = nn.Dropout(memory_dropout, inplace=False)
        else:
            self.memory_drop = None

    def encode(self, src: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode source. Returns (memory, src_key_padding_mask)."""
        return self.encoder(src)

    def forward(
        self,
        src: torch.Tensor,
        tgt_in: torch.Tensor,
        src_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Teacher-forcing forward. Returns output logits.

        Args:
            src: (B, T_src) input token IDs.
            tgt_in: (B, T_tgt) decoder input (typically [BOS] + target[:-1]).
            src_lengths: kept for API compatibility with rababa's training
              loop — not used directly (we derive the mask from PAD_ID).

        Returns:
            logits: (B, T_tgt, output_vocab_size)
        """
        memory, src_kpm = self.encode(src)
        if self.memory_drop is not None and self.training:
            memory = self.memory_drop(memory)
        return self.decoder(tgt_in, memory, src_kpm)

    def forward_heads(self, src: torch.Tensor, lengths: torch.Tensor) -> list[torch.Tensor]:
        """Diacritizer-protocol-shaped API — but seq2seq can't run
        encoder-only inference. This is here so the unified training
        loop's type-check passes for secryst. The actual training loop
        uses `forward_teacher_forced` (below), not this method.
        """
        raise NotImplementedError(
            "ModernSeq2Seq is encoder-decoder; use forward_teacher_forced "
            "or decoding.beam_search, not forward_heads."
        )

    def head_names(self) -> list[str]:
        return ["ipa"]

    def forward_teacher_forced(
        self,
        src: torch.Tensor,
        tgt_in: torch.Tensor,
    ) -> torch.Tensor:
        """Alias matching the seq2seq convention. Identical to forward()."""
        return self.forward(src, tgt_in)


def build_seq2seq(cfg: dict[str, Any]) -> ModernSeq2Seq:
    """Factory: build ModernSeq2Seq from a config dict."""
    m = cfg.get("model", {})
    return ModernSeq2Seq(
        input_vocab_size=m.get("input_vocab_size", INPUT_VOCAB_SIZE),
        output_vocab_size=m.get("output_vocab_size", OUTPUT_VOCAB_SIZE),
        dim=m.get("dim", 256),
        enc_layers=m.get("enc_layers", m.get("layers", 6)),
        dec_layers=m.get("dec_layers", m.get("layers", 6)),
        heads=m.get("heads", 8),
        ff_dim=m.get("ff_dim", 1024),
        dropout=m.get("dropout", 0.1),
        max_len=m.get("max_len", 128),
        rope_base=m.get("rope_base", 10000.0),
        sk_iters=m.get("sk_iters", 20),
        memory_dropout=m.get("memory_dropout", 0.0),
        use_mhc_cross=m.get("use_mhc_cross", True),
        resformer=m.get("resformer", None),
    )


def build_pretrain_model(cfg: dict[str, Any]) -> ModernEncoderOnly:
    """Factory: build encoder-only model for MLM pretraining."""
    from .modern import ModernEncoderOnly
    m = cfg.get("model", {})
    return ModernEncoderOnly(
        vocab_size=m.get("input_vocab_size", INPUT_VOCAB_SIZE),
        dim=m.get("dim", 256),
        layers=m.get("layers", 6),
        heads=m.get("heads", 8),
        ff_dim=m.get("ff_dim", 1024),
        dropout=m.get("dropout", 0.1),
        max_len=m.get("max_len", 128),
        rope_base=m.get("rope_base", 10000.0),
        sk_iters=m.get("sk_iters", 20),
        resformer=m.get("resformer", None),
    )


def extract_pretrained_encoder(model: ModernEncoderOnly) -> dict[str, Any]:
    """Return encoder state_dict for fine-tune loading.

    Strips the MLM head — only the encoder body is reused. Loaded into
    a fresh ModernSeq2Seq with `strict=False` so the decoder starts
    fresh-init.
    """
    out: dict[str, Any] = {}
    prefix = "encoder."
    for k, v in model.state_dict().items():
        if k.startswith(prefix):
            out[k[len(prefix):]] = v
    return out
