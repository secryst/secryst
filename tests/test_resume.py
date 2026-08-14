"""Specs for secryst pipeline resume + config_hash helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from secryst.training.resume import (
    config_hash,
    is_stage_done_with_config,
    mark_stage_done,
    read_status,
)


def test_config_hash_is_deterministic():
    cfg = {"a": 1, "b": [2, 3]}
    assert config_hash(cfg) == config_hash(cfg)


def test_config_hash_changes_on_value_change():
    cfg1 = {"a": 1, "b": 2}
    cfg2 = {"a": 1, "b": 3}
    assert config_hash(cfg1) != config_hash(cfg2)


def test_config_hash_invariant_to_key_order():
    """JSON canonical form should make key order irrelevant."""
    cfg1 = {"a": 1, "b": 2}
    cfg2 = {"b": 2, "a": 1}
    assert config_hash(cfg1) == config_hash(cfg2)


def test_config_hash_is_short_hex():
    h = config_hash({"x": 1})
    assert len(h) == 16
    int(h, 16)  # raises if not hex


def test_is_stage_done_with_config_returns_false_when_no_stage(tmp_path: Path):
    assert not is_stage_done_with_config(tmp_path, "train", {"x": 1})


def test_is_stage_done_with_config_returns_false_when_no_hash(tmp_path: Path):
    """Old status entries without config_hash → treat as not-done."""
    mark_stage_done(tmp_path, "train")
    assert not is_stage_done_with_config(tmp_path, "train", {"x": 1})


def test_is_stage_done_with_config_returns_true_when_hash_matches(tmp_path: Path):
    cfg = {"model": {"dim": 256}, "train": {"epochs": 30}}
    mark_stage_done(tmp_path, "train", extra={"config_hash": config_hash(cfg)})
    assert is_stage_done_with_config(tmp_path, "train", cfg)


def test_is_stage_done_with_config_returns_false_when_hash_differs(tmp_path: Path):
    """If config changed since stage was marked done, re-run."""
    cfg_v1 = {"model": {"dim": 256}}
    cfg_v2 = {"model": {"dim": 512}}
    mark_stage_done(tmp_path, "train", extra={"config_hash": config_hash(cfg_v1)})
    assert not is_stage_done_with_config(tmp_path, "train", cfg_v2)
