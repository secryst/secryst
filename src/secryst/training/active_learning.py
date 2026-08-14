"""Active learning harness for secryst Thai.

Mirror of rababa's pattern. Mine high-loss IPA examples, cluster by
pattern (tone, syllable count, consonant class), report actionable
insights for targeted data collection.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch.utils.data import DataLoader


@dataclass
class HardExampleReport:
    """Result of an active-learning mine pass."""
    n_examples: int
    mean_loss: float
    hard_examples: list[tuple[str, str, str, float]] = field(default_factory=list)
    pattern_clusters: dict[str, int] = field(default_factory=dict)
    recommendations: list[str] = field(default_factory=list)


@torch.no_grad()
def mine_hard_examples(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    top_k: int = 1000,
) -> HardExampleReport:
    """Mine high-loss examples from a data loader.

    Returns a HardExampleReport with the top-K highest-loss examples
    and pattern analysis.
    """
    from .evaluate import _edit_distance  # type: ignore[attr-defined]
    from ..constants import detokenize_ipa, PAD_ID
    from .beam import beam_search

    model.eval()
    losses: list[tuple[str, str, str, float]] = []
    total_loss = 0.0
    n = 0
    for batch in loader:
        src = batch.src.to(device)
        src_lengths = batch.src_lengths.to(device)
        tgt_out = batch.tgt_out
        preds = beam_search(model, src, src_lengths, max_len=64, beam_width=4, device=device)
        for i in range(src.size(0)):
            pred = preds[i]
            gold_full = tgt_out[i].tolist()
            gold = [t for t in gold_full if t not in (0, 1, 2)]
            ed = _edit_distance(pred, gold)
            loss = ed / max(1, len(gold))
            pred_str = detokenize_ipa(pred)
            losses.append((batch.raw_src[i], batch.raw_tgt[i], pred_str, loss))
            total_loss += loss
            n += 1
    # Sort by loss desc, take top-K.
    losses.sort(key=lambda x: x[3], reverse=True)
    hard = losses[:top_k]
    # Pattern clusters: bucket by edit-distance ranges.
    pattern_clusters = Counter()
    for _, _, _, loss in hard:
        bucket = "0-0.25" if loss < 0.25 else "0.25-0.5" if loss < 0.5 else "0.5-0.75" if loss < 0.75 else "0.75-1.0" if loss < 1.0 else ">1.0"
        pattern_clusters[bucket] += 1
    # Recommendations (heuristic).
    recommendations: list[str] = []
    if pattern_clusters.get(">1.0", 0) > top_k * 0.3:
        recommendations.append("More than 30% of hard examples have loss > 1.0 — consider augmented training data")
    return HardExampleReport(
        n_examples=n,
        mean_loss=total_loss / max(1, n),
        hard_examples=hard,
        pattern_clusters=dict(pattern_clusters),
        recommendations=recommendations,
    )


def write_report(report: HardExampleReport, out_path: Path) -> None:
    """Write a HardExampleReport to JSON."""
    import json
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_examples": report.n_examples,
        "mean_loss": report.mean_loss,
        "pattern_clusters": report.pattern_clusters,
        "recommendations": report.recommendations,
        "hard_examples": [
            {"src": s, "gold": g, "pred": p, "loss": l}
            for s, g, p, l in report.hard_examples[:50]  # cap for size
        ],
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
