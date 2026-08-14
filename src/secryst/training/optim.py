"""K3/DS4 optimizer stack — Muon + AdamW hybrid + QK-Clip.

Direct port of rababa's `training/optim.py` (same recipes; secryst is
a separate package so we keep its optimizer self-contained).

References:
  - Muon (Keller Jordan, 2024): github.com/KellerJordan/muon
  - Kimi K2 MuonClip + QK-Clip: arXiv:2507.20534
  - Per-Head Muon (Kimi K3): arXiv:2607.24653 (deferred)

Design:
  - Muon for 2D weights (linear/QKV/FFN).
  - AdamW for 1D params, embeddings, RMSNorm weights.
  - QK-Clip: rescale Q projection when ||w_q||·||w_k|| > τ. Anneals
    τ from 8 → 1 over training. Prevents attention-logit explosion.
"""

from __future__ import annotations

import math
from typing import Iterable

import torch
from torch import nn


# ---- Newton-Schulz orthogonalization ----------------------------------


# DS-V4-Flash §2.4 hybrid NS: two-stage coefficients.
_NS_COEFFS_AGGRESSIVE = (3.4445, -4.7750, 2.0315)
_NS_COEFFS_STABLE = (2.0, -1.5, 0.5)


@torch.no_grad()
def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Newton-Schulz iteration with DS-V4-Flash §2.4 hybrid coefficients.

    First ~80% of iterations use aggressive (3.4445, -4.7750, 2.0315),
    last ~20% use stable (2, -1.5, 0.5) to land singular values at 1.
    """
    assert G.ndim == 2
    aggressive_steps = max(1, int(0.8 * steps))
    X = G.to(torch.bfloat16)
    X = X / (X.norm() + eps)
    for i in range(steps):
        a, b, c = _NS_COEFFS_AGGRESSIVE if i < aggressive_steps else _NS_COEFFS_STABLE
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    return X.to(G.dtype)


# ---- Muon optimizer (2D weights only) ---------------------------------


class Muon(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterable[nn.Parameter],
        lr: float = 0.02,
        momentum: float = 0.95,
        ns_steps: int = 5,
        weight_decay: float = 0.0,
        update_rms_rescale: float | None = None,
        spectral_cap: float | None = None,
        heavy_tail_alpha: float | None = None,
        adamuon_beta: float | None = None,
        normuon_enabled: bool = False,
    ) -> None:
        defaults = dict(
            lr=lr, momentum=momentum, ns_steps=ns_steps, weight_decay=weight_decay,
            update_rms_rescale=update_rms_rescale,
            spectral_cap=spectral_cap, heavy_tail_alpha=heavy_tail_alpha,
            adamuon_beta=adamuon_beta, normuon_enabled=normuon_enabled,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None) -> float | None:
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = group["lr"]
            mom = group["momentum"]
            ns_steps = group["ns_steps"]
            wd = group["weight_decay"]
            rms_rescale = group.get("update_rms_rescale")
            spectral_cap = group.get("spectral_cap")
            heavy_tail_alpha = group.get("heavy_tail_alpha")
            adamuon_beta = group.get("adamuon_beta")
            normuon_enabled = group.get("normuon_enabled", False)
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                # Skip bad grads (NaN/Inf from numerical instability). This
                # prevents the bad update from poisoning the param weights —
                # training continues with the next clean batch.
                if not torch.isfinite(g).all():
                    continue
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(mom).add_(g)
                if wd > 0:
                    buf.add_(p, alpha=wd)
                if g.ndim == 2 and min(g.shape) >= 2:
                    update = zeropower_via_newtonschulz5(buf, steps=ns_steps)
                    # Sanity check: NS can diverge in bf16 → skip if not finite.
                    if not torch.isfinite(update).all():
                        continue
                    # Spectral Cap (2026): cap Frobenius norm to prevent NaN explosions.
                    if spectral_cap is not None:
                        max_frob = spectral_cap * math.sqrt(min(g.shape))
                        frob = update.norm() + 1e-8
                        scale = torch.clamp(max_frob / frob, max=1.0)
                        update = update * scale
                    # HTMuon (arXiv:2603.10067): α-blend orthogonalized with raw momentum.
                    if heavy_tail_alpha is not None and heavy_tail_alpha > 0:
                        update = (1 - heavy_tail_alpha) * update + heavy_tail_alpha * buf
                    # AdaMuon (arXiv:2507.11005): element-wise second-moment on orthogonalized update.
                    if adamuon_beta is not None:
                        if "v_buffer" not in state:
                            state["v_buffer"] = torch.zeros_like(update)
                            state["step"] = 0
                        state["step"] += 1
                        v_buf = state["v_buffer"]
                        v_buf.mul_(adamuon_beta).addcmul_(update, update, value=1 - adamuon_beta)
                        # Bias correction: without this, v_buffer is biased toward
                        # 0 at startup → division amplifies updates ~10x.
                        bias_corr = 1.0 - adamuon_beta ** state["step"]
                        v_hat = v_buf / bias_corr
                        denom = v_hat.sqrt().add_(1e-8)
                        update = update / denom
                    # NorMuon (arXiv:2510.05491): neuron-wise adaptive scaling.
                    if normuon_enabled:
                        row_norms = update.norm(dim=-1, keepdim=True) + 1e-8
                        mean_norm = row_norms.mean().clamp_min(1e-8)
                        update = update * (mean_norm / row_norms)
                    if rms_rescale is not None:
                        scale = math.sqrt(max(g.shape)) * rms_rescale
                    else:
                        scale = max(1.0, math.sqrt(max(g.shape) / min(g.shape)))
                    p.add_(update, alpha=-lr * scale)
                else:
                    p.add_(buf, alpha=-lr)
        return loss


# ---- Hybrid Muon + AdamW ----------------------------------------------


class MuonAdamWHybrid:
    """Routes 2D non-embedding non-norm params to Muon, the rest to AdamW.

    Optional `cross_attn_lr_mult` lets the cross-attention projections
    (q_cross, kv_cross, out_cross) get a higher LR. Useful for seq2seq
    training where the cross-attention gradient signal is weak and the
    decoder collapses to ignoring the encoder (mode collapse).
    """

    def __init__(
        self,
        model: nn.Module,
        muon_lr: float = 0.02,
        adam_lr: float = 3e-4,
        muon_momentum: float = 0.95,
        adam_weight_decay: float = 0.01,
        ns_steps: int = 5,
        cross_attn_lr_mult: float = 1.0,
        spectral_cap: float | None = None,
        heavy_tail_alpha: float | None = None,
        adamuon_beta: float | None = None,
        normuon_enabled: bool = False,
    ) -> None:
        muon_params: list[nn.Parameter] = []
        adam_params: list[nn.Parameter] = []
        cross_attn_params: list[nn.Parameter] = []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            is_cross_attn = (
                cross_attn_lr_mult != 1.0
                and ("q_cross" in name or "kv_cross" in name or "out_cross" in name)
            )
            if is_cross_attn:
                cross_attn_params.append(p)
            elif p.ndim == 2 and "embedding" not in name and "norm" not in name:
                muon_params.append(p)
            else:
                adam_params.append(p)
        # Build param groups. Muon gets its own optimizer (one group).
        self.muon = Muon(
            muon_params,
            lr=muon_lr,
            momentum=muon_momentum,
            ns_steps=ns_steps,
            spectral_cap=spectral_cap,
            heavy_tail_alpha=heavy_tail_alpha,
            adamuon_beta=adamuon_beta,
            normuon_enabled=normuon_enabled,
        )
        # AdamW gets default + cross-attn (with mult) groups.
        adam_groups = [
            {"params": adam_params, "lr": adam_lr, "weight_decay": adam_weight_decay},
        ]
        if cross_attn_params:
            adam_groups.append({
                "params": cross_attn_params,
                "lr": adam_lr * cross_attn_lr_mult,
                "weight_decay": adam_weight_decay,
            })
        self.adam = torch.optim.AdamW(adam_groups)
        self._muon_param_ids = {id(p) for p in muon_params}
        self._adam_param_ids = {id(p) for p in adam_params}
        self._cross_attn_param_ids = {id(p) for p in cross_attn_params}
        self.cross_attn_lr_mult = cross_attn_lr_mult

    @property
    def param_groups(self) -> list[dict]:
        return list(self.muon.param_groups) + list(self.adam.param_groups)

    def step(self, closure=None) -> float | None:
        muon_loss = self.muon.step(closure)
        adam_loss = self.adam.step()
        return muon_loss if muon_loss is not None else adam_loss

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.muon.zero_grad(set_to_none=set_to_none)
        self.adam.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict:
        return {"muon": self.muon.state_dict(), "adam": self.adam.state_dict()}

    def load_state_dict(self, state_dict: dict) -> None:
        self.muon.load_state_dict(state_dict["muon"])
        self.adam.load_state_dict(state_dict["adam"])

    @property
    def defaults(self) -> dict:
        return {**self.muon.defaults, **self.adam.defaults}


# ---- QK-Clip (Kimi K2 MuonClip) ---------------------------------------


@torch.no_grad()
def qk_clip_(model: nn.Module, tau: float = 8.0) -> dict[str, float]:
    """Rescale Q output projections so max(QK^T) ≤ tau (Frobenius bound).

    Scans encoder + decoder self-attn and decoder cross-attn. Each QKV
    layer's Q slice is uniformly rescaled if the product of Frobenius
    norms of Q and K exceeds tau.
    """
    out: dict[str, float] = {}

    def _scan_qkv(layer: nn.Module, prefix: str) -> None:
        for attr in ("qkv", "qkv_self"):
            qkv = getattr(layer, attr, None)
            if qkv is None or not isinstance(qkv, nn.Linear):
                continue
            w = qkv.weight
            D = w.shape[1]
            w_q = w[:D]
            w_k = w[D:2 * D]
            norm_product = w_q.norm() * w_k.norm()
            key = f"{prefix}.{attr}"
            if norm_product.item() > tau:
                scale = math.sqrt(tau / max(norm_product.item(), 1e-8))
                w_q_scaled = w_q * scale
                new_w = torch.cat([w_q_scaled, w[D:]], dim=0)
                qkv.weight.copy_(new_w)
                out[key] = scale
            else:
                out[key] = 1.0
        # Cross-attn: separate Q and KV projections.
        q_cross = getattr(layer, "q_cross", None)
        kv_cross = getattr(layer, "kv_cross", None)
        if isinstance(q_cross, nn.Linear) and isinstance(kv_cross, nn.Linear):
            # kv_cross has shape (2*dim, dim). K slice = rows [0:dim].
            w_q = q_cross.weight
            w_k = kv_cross.weight[: w_q.shape[1]]
            norm_product = w_q.norm() * w_k.norm()
            key = f"{prefix}.cross_qk"
            if norm_product.item() > tau:
                scale = math.sqrt(tau / max(norm_product.item(), 1e-8))
                q_cross.weight.copy_(w_q * scale)
                out[key] = scale
            else:
                out[key] = 1.0

    # Encoder layers.
    enc = getattr(model, "encoder", None)
    if enc is not None and hasattr(enc, "layers"):
        for i, layer in enumerate(enc.layers):
            _scan_qkv(layer, f"enc.{i}")
    # Decoder layers.
    dec = getattr(model, "decoder", None)
    if dec is not None and hasattr(dec, "layers"):
        for i, layer in enumerate(dec.layers):
            _scan_qkv(layer, f"dec.{i}")
    return out


def qk_clip_schedule(step: int, total_steps: int, tau_init: float = 8.0, tau_final: float = 1.0) -> float:
    """Linear anneal of QK-Clip threshold tau_init → tau_final over training."""
    if total_steps <= 0:
        return tau_final
    progress = min(1.0, step / total_steps)
    return tau_init + (tau_final - tau_init) * progress
