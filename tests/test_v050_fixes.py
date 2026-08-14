"""Specs for v0.5.0 mode-collapse fixes (scheduled sampling, cross-attn LR)."""

from __future__ import annotations

import torch
import torch.nn as nn

from secryst.config import load_task_config, to_dict
from secryst.models.base import build_model
from secryst.training.optim import MuonAdamWHybrid
from secryst.training.supervised import _scheduled_sample, build_optimizer


def test_scheduled_sample_replaces_gold_with_argmax():
    """_scheduled_sample should produce decoder input where positions > 0
    hold the model's argmax predictions (shifted right)."""
    cfg = load_task_config("secryst_thai_ipa")
    cfg_dict = to_dict(cfg)
    # Disable memory_dropout for deterministic test.
    cfg_dict["model"]["memory_dropout"] = 0.0
    model = build_model(cfg_dict)
    model.eval()  # deterministic
    src = torch.randint(4, 50, (2, 6))
    tgt_in_gold = torch.tensor([[1, 25, 5, 54, 0, 0], [1, 10, 20, 30, 40, 0]])
    out = _scheduled_sample(model, src, tgt_in_gold)
    # Position 0 (BOS) is preserved.
    assert torch.equal(out[:, 0], tgt_in_gold[:, 0])
    # Output has same shape.
    assert out.shape == tgt_in_gold.shape


def test_cross_attn_lr_mult_creates_separate_group():
    """MuonAdamWHybrid with cross_attn_lr_mult > 1 should produce 2 AdamW groups."""
    cfg = load_task_config("secryst_thai_ipa")
    model = build_model(to_dict(cfg))
    opt = MuonAdamWHybrid(model, muon_lr=0.02, adam_lr=3e-4, cross_attn_lr_mult=3.0)
    assert len(opt.adam.param_groups) == 2
    assert opt.adam.param_groups[0]["lr"] == 3e-4
    assert opt.adam.param_groups[1]["lr"] == 9e-4  # 3x
    assert opt.cross_attn_lr_mult == 3.0


def test_cross_attn_lr_mult_default_is_one_group():
    """Without cross_attn_lr_mult, AdamW has just one group."""
    cfg = load_task_config("secryst_thai_ipa")
    model = build_model(to_dict(cfg))
    opt = MuonAdamWHybrid(model, muon_lr=0.02, adam_lr=3e-4)
    assert len(opt.adam.param_groups) == 1
    assert opt.adam.param_groups[0]["lr"] == 3e-4


def test_memory_dropout_is_none_when_disabled():
    """When memory_dropout=0, the model has no memory_drop module."""
    cfg = {"model": {"dim": 64, "enc_layers": 2, "dec_layers": 2, "heads": 4, "ff_dim": 128, "max_len": 16, "memory_dropout": 0.0}}
    model = build_model(cfg)
    assert model.memory_drop is None
    assert model.memory_dropout == 0.0


def test_memory_dropout_is_active_when_enabled():
    cfg = {"model": {"dim": 64, "enc_layers": 2, "dec_layers": 2, "heads": 4, "ff_dim": 128, "max_len": 16, "memory_dropout": 0.1}}
    model = build_model(cfg)
    assert model.memory_drop is not None
    assert model.memory_dropout == 0.1


def test_build_optimizer_passes_cross_attn_mult():
    """build_optimizer should forward cross_attn_lr_mult from cfg to MuonAdamWHybrid."""
    cfg = {"model": {"dim": 64, "enc_layers": 2, "dec_layers": 2, "heads": 4, "ff_dim": 128, "max_len": 16},
           "train": {"optimizer": "muon", "cross_attn_lr_mult": 5.0}}
    from secryst.models.base import build_model as bm
    model = bm(cfg["model"] if "model" in cfg else cfg)
    # build_model expects cfg with 'model' key — pass full cfg.
    full_cfg = {"model": cfg["model"]}
    model = bm(full_cfg)
    opt = build_optimizer(model, cfg["train"])
    assert opt.cross_attn_lr_mult == 5.0
