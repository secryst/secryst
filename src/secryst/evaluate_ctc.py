"""CTC-specific evaluation for secryst Thai→IPA.

Greedy CTC decode + PER computation. Mirrors `evaluate.evaluate_per` API
but uses CTCModel.greedy_decode instead of beam search.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from .constants import detokenize_ipa
from .evaluate import _edit_distance


@torch.no_grad()
def evaluate_ctc_per(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Compute PER + WER + exact-match on a test loader using CTC greedy decode.

    Returns dict with keys: 'per', 'wer', 'exact_match', 'n_examples', 'arch'.
    """
    model.eval()
    total_ed = 0
    total_gold_len = 0
    exact_match = 0
    n = 0
    for batch in loader:
        src = batch.src.to(device)
        src_lengths = batch.src_lengths.to(device)
        tgt_out = batch.tgt_out
        preds = model.greedy_decode(src, src_lengths)
        for i in range(src.size(0)):
            pred = preds[i]
            gold_full = tgt_out[i].tolist()
            gold = [t for t in gold_full if t not in (0, 1, 2)]  # strip PAD/BOS/EOS
            ed = _edit_distance(pred, gold)
            total_ed += ed
            total_gold_len += max(1, len(gold))
            if ed == 0:
                exact_match += 1
            n += 1
    per = total_ed / max(1, total_gold_len)
    wer = 1.0 - (exact_match / max(1, n))
    return {
        "per": per,
        "wer": wer,
        "exact_match": exact_match / max(1, n),
        "n_examples": n,
        "arch": "ctc",
    }
