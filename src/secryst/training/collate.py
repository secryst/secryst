"""Collation — pad sequences to batch max, build teacher-forced targets.

For seq2seq we need:
  - src:        (B, T_src) — input token IDs padded with PAD_ID.
  - src_lengths: (B,)      — true lengths (for batch mask if needed).
  - tgt_in:     (B, T_tgt) — [BOS] + target[:-1] padded with PAD_ID.
  - tgt_out:    (B, T_tgt) — target + [EOS] padded with PAD_ID (-100 for loss).

PAD/BOS/EOS IDs come from `secryst.constants`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..constants import BOS_ID, EOS_ID, PAD_ID
from ..datasets import IPAPair, MLMExample


@dataclass
class Batch:
    """Collated seq2seq batch."""
    src: torch.Tensor           # (B, T_src) int64
    src_lengths: torch.Tensor   # (B,) int64
    tgt_in: torch.Tensor        # (B, T_tgt) int64 — decoder input
    tgt_out: torch.Tensor       # (B, T_tgt) int64 — decoder target
    raw_src: list[str]
    raw_tgt: list[str]


@dataclass
class MLMBatch:
    src: torch.Tensor
    lengths: torch.Tensor
    targets: list[torch.Tensor]
    raw: list[str]


def _truncate_src(pair: IPAPair, max_len: int) -> IPAPair:
    if len(pair.src_ids) > max_len:
        return IPAPair(src_ids=pair.src_ids[:max_len], tgt_ids=pair.tgt_ids, raw_src=pair.raw_src, raw_tgt=pair.raw_tgt)
    return pair


def _truncate_tgt(pair: IPAPair, max_len: int) -> IPAPair:
    if len(pair.tgt_ids) > max_len - 2:  # reserve room for BOS+EOS
        return IPAPair(
            src_ids=pair.src_ids,
            tgt_ids=pair.tgt_ids[: max_len - 2],
            raw_src=pair.raw_src, raw_tgt=pair.raw_tgt,
        )
    return pair


def ipa_collate_batch(batch: list[IPAPair], max_len: int = 128) -> Batch:
    """Pad src + tgt and add BOS/EOS for teacher forcing."""
    batch = [_truncate_tgt(_truncate_src(ex, max_len), max_len) for ex in batch]
    B = len(batch)
    src_max = max(len(ex.src_ids) for ex in batch)
    # Tgt needs +2 for BOS/EOS; align all to batch max.
    tgt_max = max(len(ex.tgt_ids) for ex in batch) + 2

    src = torch.full((B, src_max), PAD_ID, dtype=torch.long)
    src_lengths = torch.zeros((B,), dtype=torch.long)
    tgt_in = torch.full((B, tgt_max), PAD_ID, dtype=torch.long)
    tgt_out = torch.full((B, tgt_max), PAD_ID, dtype=torch.long)
    for i, ex in enumerate(batch):
        n_src = len(ex.src_ids)
        src[i, :n_src] = torch.tensor(ex.src_ids, dtype=torch.long)
        src_lengths[i] = n_src

        # tgt_in = [BOS] + target
        # tgt_out = target + [EOS]
        tgt_full = list(ex.tgt_ids)
        n_tgt = len(tgt_full)
        tgt_in[i, 0] = BOS_ID
        if n_tgt > 0:
            tgt_in[i, 1 : 1 + n_tgt] = torch.tensor(tgt_full, dtype=torch.long)
        tgt_out[i, :n_tgt] = torch.tensor(tgt_full, dtype=torch.long)
        tgt_out[i, n_tgt] = EOS_ID

    return Batch(
        src=src, src_lengths=src_lengths,
        tgt_in=tgt_in, tgt_out=tgt_out,
        raw_src=[ex.raw_src for ex in batch],
        raw_tgt=[ex.raw_tgt for ex in batch],
    )


def make_ipa_collate_fn(max_len: int = 128):
    def _collate(batch: list[IPAPair]) -> Batch:
        return ipa_collate_batch(batch, max_len=max_len)
    return _collate


# ---- MLM collate ------------------------------------------------------


def mlm_collate_batch(batch: list[MLMExample], max_len: int = 128) -> MLMBatch:
    truncated: list[MLMExample] = []
    for ex in batch:
        if len(ex.input_ids) > max_len:
            truncated.append(MLMExample(
                input_ids=ex.input_ids[:max_len],
                target_ids=ex.target_ids[:max_len],
                raw=ex.raw,
            ))
        else:
            truncated.append(ex)
    batch = truncated
    max_actual = max(len(ex.input_ids) for ex in batch)
    src = torch.full((len(batch), max_actual), PAD_ID, dtype=torch.long)
    target = torch.full((len(batch), max_actual), PAD_ID, dtype=torch.long)
    lengths = torch.zeros((len(batch),), dtype=torch.long)
    for i, ex in enumerate(batch):
        n = len(ex.input_ids)
        src[i, :n] = torch.tensor(ex.input_ids, dtype=torch.long)
        target[i, :n] = torch.tensor(ex.target_ids, dtype=torch.long)
        lengths[i] = n
    return MLMBatch(src=src, lengths=lengths, targets=[target], raw=[ex.raw for ex in batch])


def make_mlm_collate_fn(max_len: int = 128):
    def _collate(batch: list[MLMExample]) -> MLMBatch:
        return mlm_collate_batch(batch, max_len=max_len)
    return _collate
