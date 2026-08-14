"""Task dispatch — single source of truth for cfg → dataset + collate."""

from __future__ import annotations

from typing import Any

from torch.utils.data import DataLoader

from .config import load_task_config
from .datasets import load_thai_ipa, load_thai_mlm
from .training.collate import ipa_collate_batch, mlm_collate_batch, make_ipa_collate_fn, make_mlm_collate_fn


SUPERVISED_DATASETS = {
    "secryst": load_thai_ipa,
}
MLM_DATASETS = {
    "secryst_mlm": load_thai_mlm,
}


def _get_cleaner(cfg: Any) -> str:
    return cfg.data.get("cleaner", "thai") if hasattr(cfg.data, "get") else "thai"


def _get_max_len(cfg: Any) -> int:
    return int(cfg.model.get("max_len", 128))


def _get_data_root(cfg: Any) -> str | None:
    raw = cfg.data.get("root") if hasattr(cfg.data, "get") else None
    return raw or None


def build_supervised_loaders(
    cfg: Any,
    batch_size: int | None = None,
    num_workers: int | None = None,
) -> tuple[DataLoader, DataLoader]:
    kind = cfg.kind
    if kind not in SUPERVISED_DATASETS:
        raise ValueError(f"unknown supervised task kind: {kind!r}")
    loader_fn = SUPERVISED_DATASETS[kind]
    cleaner = _get_cleaner(cfg)
    max_len = _get_max_len(cfg)
    root = _get_data_root(cfg)
    bs = batch_size or int(cfg.train.get("batch_size", 64))
    if num_workers is None:
        num_workers = int(cfg.train.get("num_workers", 8)) if hasattr(cfg.train, "get") else 8

    train_ds = loader_fn("train", root=root, cleaner=cleaner, max_len=max_len)
    val_ds = loader_fn("val", root=root, cleaner=cleaner, max_len=max_len)
    collate = make_ipa_collate_fn(max_len)
    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True, num_workers=num_workers,
        collate_fn=collate, persistent_workers=(num_workers > 0), pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=bs, shuffle=False, num_workers=num_workers,
        collate_fn=collate, persistent_workers=(num_workers > 0), pin_memory=True,
    )
    return train_loader, val_loader


def build_mlm_loaders(
    cfg: Any,
    batch_size: int | None = None,
    num_workers: int | None = None,
) -> tuple[DataLoader, DataLoader]:
    kind = cfg.kind
    if kind not in MLM_DATASETS:
        raise ValueError(f"unknown MLM task kind: {kind!r}")
    loader_fn = MLM_DATASETS[kind]
    cleaner = _get_cleaner(cfg)
    max_len = _get_max_len(cfg)
    root = _get_data_root(cfg)
    bs = batch_size or int(cfg.train.get("batch_size", 64))
    mask_prob = float(cfg.data.get("mask_prob", 0.15))
    if num_workers is None:
        num_workers = int(cfg.train.get("num_workers", 8)) if hasattr(cfg.train, "get") else 8

    train_ds = loader_fn("train", root=root, cleaner=cleaner, mask_prob=mask_prob, max_len=max_len)
    val_ds = loader_fn("val", root=root, cleaner=cleaner, mask_prob=mask_prob, max_len=max_len)
    collate = make_mlm_collate_fn(max_len)
    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True, num_workers=num_workers,
        collate_fn=collate, persistent_workers=(num_workers > 0), pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=bs, shuffle=False, num_workers=num_workers,
        collate_fn=collate, persistent_workers=(num_workers > 0), pin_memory=True,
    )
    return train_loader, val_loader


def build_test_loader(
    task: str,
    batch_size: int = 32,
    cleaner: str | None = None,
    max_len: int = 128,
    num_workers: int = 0,
) -> DataLoader:
    cfg = load_task_config(task)
    kind = cfg.kind
    if kind not in SUPERVISED_DATASETS:
        raise ValueError(f"unknown task kind: {kind!r}")
    loader_fn = SUPERVISED_DATASETS[kind]
    cleaner = cleaner or _get_cleaner(cfg)
    root = _get_data_root(cfg)
    ds = loader_fn("test", root=root, cleaner=cleaner, max_len=max_len)
    collate = make_ipa_collate_fn(max_len)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate)
