"""Specs for secryst ResFormer + 2026 Muon variants port.

Covers:
  - ResFormer on ModernEncoderLayer / ModernEncoder (encoder side only).
  - ModernSeq2Seq ResFormer config plumbing.
  - Spectral Cap + HTMuon + AdaMuon + NorMuon via MuonAdamWHybrid.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from secryst.models.modern import ModernEncoderLayer, ModernEncoder
from secryst.models.seq2seq import build_seq2seq, build_pretrain_model
from secryst.training.optim import Muon, MuonAdamWHybrid


# ---- ResFormer on encoder layer ----------------------------------------


def test_resformer_layer_off_by_default():
    """Layer without resformer_lambda1 should not enable ResFormer."""
    layer = ModernEncoderLayer(dim=64, heads=4, ff_dim=128)
    assert layer.use_resformer is False
    assert not hasattr(layer, "resformer_lambda1")


def test_resformer_layer_on_when_lambda1_set():
    """Setting resformer_lambda1 enables ResFormer and creates λ parameters."""
    layer = ModernEncoderLayer(
        dim=64, heads=4, ff_dim=128,
        resformer_lambda1=0.5,
    )
    assert layer.use_resformer is True
    assert isinstance(layer.resformer_lambda1, nn.Parameter)
    assert layer.resformer_lambda1.item() == pytest.approx(0.5)
    assert layer.resformer_lambda2.item() == pytest.approx(0.5)


def test_resformer_encoder_sparse_mode():
    """ModernEncoder sparse mode: only last n layers get V_1."""
    enc = ModernEncoder(
        vocab_size=30, dim=64, layers=6, heads=4, ff_dim=128,
        resformer={"mode": "sparse", "n_last_layers": 2, "lambda1": 5.0},
    )
    expected = [False, False, False, False, True, True]
    actual = [layer.use_resformer for layer in enc.layers]
    assert actual == expected
    assert enc.layers[-1].resformer_lambda1.item() == pytest.approx(5.0)


def test_resformer_encoder_all_mode():
    """ModernEncoder all mode: every layer ≥1 gets V_1."""
    enc = ModernEncoder(
        vocab_size=30, dim=64, layers=4, heads=4, ff_dim=128,
        resformer={"mode": "all"},
    )
    expected = [False, True, True, True]
    actual = [layer.use_resformer for layer in enc.layers]
    assert actual == expected


def test_resformer_forward_preserves_shape():
    """ModernEncoder forward should produce same shape with/without ResFormer."""
    enc_no = ModernEncoder(vocab_size=30, dim=64, layers=4, heads=4, ff_dim=128)
    enc_rf = ModernEncoder(
        vocab_size=30, dim=64, layers=4, heads=4, ff_dim=128,
        resformer={"mode": "all"},
    )
    src = torch.randint(1, 30, (2, 16))
    h_no, _ = enc_no(src)
    h_rf, _ = enc_rf(src)
    assert h_no.shape == h_rf.shape


def test_resformer_encoder_caches_v_first():
    """After forward, layer 0 should have _v_first attribute."""
    enc = ModernEncoder(
        vocab_size=30, dim=64, layers=4, heads=4, ff_dim=128,
        resformer={"mode": "all"},
    )
    src = torch.randint(1, 30, (2, 16))
    enc(src)
    assert hasattr(enc.layers[0], "_v_first")
    assert enc.layers[0]._v_first is not None


def test_resformer_backward_updates_lambda():
    """λ parameters should receive gradients after backward."""
    enc = ModernEncoder(
        vocab_size=30, dim=64, layers=4, heads=4, ff_dim=128,
        resformer={"mode": "all"},
    )
    src = torch.randint(1, 30, (2, 16))
    h, _ = enc(src)
    h.sum().backward()
    for i in range(1, 4):
        assert enc.layers[i].resformer_lambda1.requires_grad


def test_resformer_seq2seq_factory():
    """build_seq2seq should propagate resformer config to encoder."""
    m = build_seq2seq({
        "model": {
            "dim": 64, "enc_layers": 4, "dec_layers": 4, "heads": 4, "ff_dim": 128,
            "max_len": 32,
            "resformer": {"mode": "sparse", "n_last_layers": 2, "lambda1": 5.0},
        }
    })
    # Encoder has ResFormer on last 2 of 4 layers.
    actual = [layer.use_resformer for layer in m.encoder.layers]
    assert actual == [False, False, True, True]


def test_resformer_pretrain_factory():
    """build_pretrain_model should propagate resformer config."""
    m = build_pretrain_model({
        "model": {
            "dim": 64, "layers": 4, "heads": 4, "ff_dim": 128, "max_len": 32,
            "resformer": {"mode": "all"},
        }
    })
    actual = [layer.use_resformer for layer in m.encoder.layers]
    assert actual == [False, True, True, True]


# ---- Muon variants -----------------------------------------------------


def test_muon_spectral_cap():
    """Muon with spectral_cap should step without error."""
    linear = nn.Linear(20, 20)
    opt = Muon([linear.weight], lr=0.01, spectral_cap=1.5)
    out = linear(torch.randn(4, 20))
    out.sum().backward()
    opt.step()
    assert torch.isfinite(linear.weight).all()


def test_muon_heavy_tail():
    """Muon with heavy_tail_alpha should step without error."""
    linear = nn.Linear(20, 20)
    opt = Muon([linear.weight], lr=0.01, heavy_tail_alpha=0.05)
    out = linear(torch.randn(4, 20))
    out.sum().backward()
    opt.step()
    assert torch.isfinite(linear.weight).all()


def test_muon_adamuon_creates_v_buffer():
    """AdaMuon should create v_buffer after first step."""
    linear = nn.Linear(20, 20)
    opt = Muon([linear.weight], lr=0.01, adamuon_beta=0.99)
    out = linear(torch.randn(4, 20))
    out.sum().backward()
    opt.step()
    assert "v_buffer" in opt.state[linear.weight]


def test_muon_normuon_step_runs():
    """NorMuon should step without error."""
    linear = nn.Linear(20, 20)
    opt = Muon([linear.weight], lr=0.01, normuon_enabled=True)
    out = linear(torch.randn(4, 20))
    out.sum().backward()
    opt.step()
    assert torch.isfinite(linear.weight).all()


def test_hybrid_optimizer_accepts_all_variants():
    """MuonAdamWHybrid should accept all 4 new variants."""
    class ToyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(20, 20)
            self.norm = nn.LayerNorm(20)

        def forward(self, x):
            return self.norm(self.lin(x))

    model = ToyModel()
    opt = MuonAdamWHybrid(
        model, muon_lr=0.01, adam_lr=3e-4,
        spectral_cap=1.2, heavy_tail_alpha=0.05,
        adamuon_beta=0.99, normuon_enabled=True,
    )
    out = model(torch.randn(4, 20))
    out.sum().backward()
    opt.step()
    assert torch.isfinite(model.lin.weight).all()


def test_all_variants_combined():
    """Spectral Cap + HTMuon + AdaMuon + NorMuon should compose."""
    linear = nn.Linear(20, 20)
    opt = Muon(
        [linear.weight], lr=0.01,
        spectral_cap=1.5, heavy_tail_alpha=0.05,
        adamuon_beta=0.99, normuon_enabled=True,
    )
    for _ in range(3):
        linear.zero_grad()
        out = linear(torch.randn(4, 20))
        out.sum().backward()
        opt.step()
    assert torch.isfinite(linear.weight).all()
