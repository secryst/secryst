"""OmegaConf-based config loader — same pattern as rababa.

Task configs live in `configs/<task>.yaml`. They inherit from
`configs/base.yaml` for shared defaults.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

CONFIGS_DIR = Path(__file__).resolve().parent.parent.parent / "configs"


def load_task_config(task: str, configs_dir: Path | None = None) -> DictConfig:
    base = configs_dir or CONFIGS_DIR
    base_cfg = OmegaConf.load(base / "base.yaml")
    task_path = base / f"{task}.yaml"
    if not task_path.is_file():
        raise FileNotFoundError(f"No config for task '{task}' at {task_path}")
    task_cfg = OmegaConf.load(task_path)
    return OmegaConf.merge(base_cfg, task_cfg)


def to_dict(cfg: DictConfig) -> dict[str, Any]:
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[no-any-return]
