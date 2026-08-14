"""Resume + status helpers — direct port of rababa's resume.py.

Three concerns when the local session disconnects mid-sprint:
  1. Mid-training resume: find latest checkpoint, continue from next epoch.
  2. Stage status tracking: JSON index on the checkpoints volume.
  3. Log persistence: tee writes to stdout + a volume file.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

EPOCH_CKPT_RE = re.compile(r"checkpoint-epoch-(\d+)\.pt$")


def latest_resume_checkpoint(ckpt_root: Path) -> tuple[Path, int] | None:
    if not ckpt_root.is_dir():
        return None
    epoch_ckpts = []
    for p in ckpt_root.glob("checkpoint-epoch-*.pt"):
        m = EPOCH_CKPT_RE.search(p.name)
        if m:
            epoch_ckpts.append((int(m.group(1)), p))
    if epoch_ckpts:
        epoch_ckpts.sort()
        return epoch_ckpts[-1][1], epoch_ckpts[-1][0]
    best = ckpt_root / "best.pt"
    if best.is_file():
        return best, -1
    return None


def load_resume_state(model, optimizer, scheduler, path: Path, device: str | None = None) -> dict[str, Any]:
    state = (
        torch.load(path, map_location=device, weights_only=False)
        if device
        else torch.load(path, weights_only=False)
    )
    model.load_state_dict(state["model"])
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and "scheduler" in state:
        try:
            scheduler.load_state_dict(state["scheduler"])
        except Exception:
            pass
    return state


def save_resumable_checkpoint(
    path: Path,
    model,
    optimizer=None,
    scheduler=None,
    epoch: int = 0,
    best_val_loss: float | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    state: dict[str, Any] = {
        "model": model.state_dict(),
        "epoch": epoch,
        "best_val_loss": best_val_loss,
    }
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()
    if extra:
        state.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


# ---- Stage status index ----------------------------------------------


def _status_path(volume_root: Path) -> Path:
    return volume_root / "_status.json"


def read_status(volume_root: Path) -> dict[str, Any]:
    p = _status_path(volume_root)
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def is_stage_done(volume_root: Path, stage: str) -> bool:
    return bool(read_status(volume_root).get("stages", {}).get(stage, {}).get("done"))


def mark_stage_done(volume_root: Path, stage: str, extra: dict[str, Any] | None = None) -> None:
    status = read_status(volume_root)
    status.setdefault("stages", {})
    entry: dict[str, Any] = {"done": True, "ts": time.time()}
    if extra:
        entry.update(extra)
    status["stages"][stage] = entry
    _status_path(volume_root).parent.mkdir(parents=True, exist_ok=True)
    _status_path(volume_root).write_text(json.dumps(status, indent=2), encoding="utf-8")


def mark_stage_failed(volume_root: Path, stage: str, error: str) -> None:
    status = read_status(volume_root)
    status.setdefault("stages", {})
    status["stages"][stage] = {
        "done": False,
        "ts": time.time(),
        "error": error[:500],
    }
    _status_path(volume_root).parent.mkdir(parents=True, exist_ok=True)
    _status_path(volume_root).write_text(json.dumps(status, indent=2), encoding="utf-8")


def config_hash(cfg: dict[str, Any]) -> str:
    """Stable hash of a config dict for change detection.

    Used to decide whether a stage's existing checkpoint matches the
    current config (skip) or differs (retrain). Returns a short hex
    string derived from the canonical JSON of the config.
    """
    import hashlib
    import json
    # Sort keys for determinism. json.dumps handles nested dicts.
    canonical = json.dumps(cfg, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def is_stage_done_with_config(
    volume_root: Path,
    stage: str,
    cfg: dict[str, Any],
) -> bool:
    """True iff stage is done AND its stored config_hash matches current cfg.

    Used by the orchestrator to skip stages whose config hasn't changed
    (avoiding wipes when re-running an unchanged pipeline).
    """
    s = read_status(volume_root).get("stages", {}).get(stage, {})
    if not s.get("done"):
        return False
    stored_hash = s.get("config_hash")
    if stored_hash is None:
        # Stage was marked done before config_hash tracking was added.
        # Be conservative: treat as not-done (re-run).
        return False
    return stored_hash == config_hash(cfg)


# ---- Volume log file --------------------------------------------------


class VolumeLogger:
    """Tee writes to stdout + a local file (mirrored to volume on sync).

    Writes go to /tmp to avoid blocking Modal's volume.reload(). Call
    `sync_to_volume()` between stages to persist the log.
    """

    def __init__(self, log_path: Path) -> None:
        self.volume_path = log_path
        self.local_path = Path("/tmp") / log_path.name
        self.local_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.local_path.is_file():
            self.local_path.touch()

    def log(self, msg: str) -> None:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        with self.local_path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def sync_to_volume(self) -> None:
        self.volume_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.volume_path.write_text(self.local_path.read_text(encoding="utf-8"), encoding="utf-8")
        except OSError:
            pass

    def close(self) -> None:
        self.sync_to_volume()


import torch  # noqa: E402  (intentional late import)
