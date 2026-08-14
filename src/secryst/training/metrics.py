"""Simple per-epoch JSONL metrics logger for secryst.

Standalone version of rababa.training.metrics.MetricsLogger. Same JSONL
format (one EpochMetrics row per line). Kept separate to avoid coupling
secryst to rababa.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    val_loss: float
    learning_rate: float
    ts: float

    @classmethod
    def from_train_metrics(cls, m: Any) -> "EpochMetrics":
        return cls(
            epoch=int(m.epoch),
            train_loss=float(m.train_loss),
            val_loss=float(m.val_loss),
            learning_rate=float(m.learning_rate),
            ts=time.time(),
        )

    def to_json_line(self) -> str:
        return json.dumps(asdict(self))


class SimpleMetricsLogger:
    """Append-only JSONL logger (mirror of rababa.training.metrics.MetricsLogger)."""

    def __init__(self, volume_path: Path) -> None:
        self.volume_path = volume_path
        if str(volume_path).startswith("/tmp") or volume_path.parent.is_dir():
            self.local_path = volume_path
        else:
            self.local_path = Path("/tmp") / volume_path.name
        self.local_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.local_path.is_file():
            self.local_path.touch()

    def log(self, metrics: Any) -> None:
        if isinstance(metrics, EpochMetrics):
            row = metrics
        else:
            row = EpochMetrics.from_train_metrics(metrics)
        with self.local_path.open("a", encoding="utf-8") as fh:
            fh.write(row.to_json_line() + "\n")
        self.sync_to_volume()

    def sync_to_volume(self) -> None:
        self.volume_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.volume_path.write_text(
                self.local_path.read_text(encoding="utf-8"),
                encoding="utf-8",
            )
        except OSError:
            pass

    def close(self) -> None:
        self.sync_to_volume()

    def read_all(self) -> list[dict[str, Any]]:
        if not self.local_path.is_file():
            return []
        rows: list[dict[str, Any]] = []
        for line in self.local_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return rows
