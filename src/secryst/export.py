"""ONNX export — fp32 only (no quantization per user request).

Exports a ModernSeq2Seq with fixed batch + length. Because the model
is encoder-decoder, the exported graph has:
  Inputs:
    - src: (batch, T_src) int64
    - tgt_in: (batch, T_tgt) int64 — pre-shifted target for teacher forcing
  Outputs:
    - ipa: (batch, T_tgt, output_vocab) float32 — per-position logits

For browser inference, the runtime does iterative decode by repeatedly
invoking the ONNX model with the current prefix. A future v0.5.0
export could include an "autoregressive" wrapper, but the simplest
inference loop lives in TypeScript and calls the per-step model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .models.base import build_model


def export_seq2seq_onnx(
    model_state_path: Path,
    cfg: dict[str, Any],
    out_path: Path,
    batch_size: int = 1,
    src_max_len: int = 64,
    tgt_max_len: int = 64,
) -> None:
    """Export trained seq2seq → ONNX fp32.

    Args:
        model_state_path: path to a .pt file containing model.state_dict().
        cfg: model config dict.
        out_path: destination .onnx path.
        batch_size: fixed batch dim.
        src_max_len: fixed source length.
        tgt_max_len: fixed target length.
    """
    model = build_model(cfg).eval()
    state = torch.load(model_state_path, map_location="cpu", weights_only=True)
    # Tolerate both raw state_dict and wrapped {"model": ...} formats.
    if isinstance(state, dict) and "model" in state and not any(
        isinstance(v, torch.Tensor) for k, v in state.items() if k != "model"
    ):
        # Wrapped checkpoint from save_resumable_checkpoint.
        state = state["model"]
    elif isinstance(state, dict) and any(
        isinstance(v, torch.Tensor) for v in state.values()
    ) and "model" not in state:
        # Raw state_dict.
        pass
    model.load_state_dict(state)

    src = torch.randint(4, 50, (batch_size, src_max_len), dtype=torch.long)
    tgt_in = torch.randint(4, 50, (batch_size, tgt_max_len), dtype=torch.long)
    src_lengths = torch.full((batch_size,), src_max_len, dtype=torch.long)

    with torch.no_grad():
        torch.onnx.export(
            model,
            (src, tgt_in, src_lengths),
            str(out_path),
            opset_version=17,
            input_names=["src", "tgt_in", "src_lengths"],
            output_names=["ipa"],
            dynamic_axes={},
        )


# Note: NO quantize_dynamic_int8 here. User explicitly requested fp32 only.
