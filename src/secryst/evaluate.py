"""Evaluation — Phoneme Error Rate (PER), Word Error Rate (WER), accuracy.

PER: edit distance per slot, averaged across the test set. Standard
G2P metric. Lower is better.

WER: fraction of examples where the predicted IPA does not exactly
match the gold (case-sensitive). More stringent than PER.

Both metrics use beam-search decoding (greedy if beam_width=1).
"""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader

from .constants import detokenize_ipa
from .decoding.beam import beam_search


def _edit_distance(a: list[int], b: list[int]) -> int:
    """Levenshtein on integer sequences."""
    m, n = len(a), len(b)
    if m == 0:
        return n
    if n == 0:
        return m
    prev = list(range(n + 1))
    cur = [0] * (n + 1)
    for i in range(1, m + 1):
        cur[0] = i
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                cur[j] = prev[j - 1]
            else:
                cur[j] = 1 + min(prev[j], cur[j - 1], prev[j - 1])
        prev, cur = cur, prev
    return prev[n]


@torch.no_grad()
def evaluate_per(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    beam_width: int = 4,
    max_len: int = 64,
    eos_bias: float = 0.0,
) -> dict[str, float]:
    """Compute PER + WER + exact-match on a test loader.

    Returns dict with keys: 'per', 'wer', 'exact_match', 'n_examples'.
    """
    model.eval()
    total_ed = 0
    total_gold_len = 0
    exact_match = 0
    n = 0
    for batch in loader:
        src = batch.src.to(device)
        src_lengths = batch.src_lengths.to(device)
        # tgt_out is (B, T_tgt) padded; mask out PAD/EOS for edit distance.
        tgt_out = batch.tgt_out
        preds = beam_search(model, src, src_lengths, max_len=max_len, beam_width=beam_width, device=device, eos_bias=eos_bias)
        for i in range(src.size(0)):
            pred = preds[i]
            gold_full = tgt_out[i].tolist()
            gold = [t for t in gold_full if t not in (0, 1, 2)]  # strip PAD/BOS/EOS
            # Edit distance at the token level.
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
    }


def transliterate_all(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    beam_width: int = 4,
    max_len: int = 64,
) -> list[tuple[str, str, str]]:
    """Return list of (src_raw, gold_raw, pred_raw) per example."""
    model.eval()
    out: list[tuple[str, str, str]] = []
    for batch in loader:
        src = batch.src.to(device)
        src_lengths = batch.src_lengths.to(device)
        preds = beam_search(model, src, src_lengths, max_len=max_len, beam_width=beam_width, device=device)
        for i in range(src.size(0)):
            pred_str = detokenize_ipa(preds[i])
            out.append((batch.raw_src[i], batch.raw_tgt[i], pred_str))
    return out
