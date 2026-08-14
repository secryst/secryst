"""Supervised training loop for the seq2seq Thai→IPA model.

Teacher forcing: decoder input is `[BOS] + target[:-1]`, decoder target
is `target + [EOS]`. Loss = per-position CE over output vocab, masked
to ignore PAD positions.

Loss ignores:
  - PAD positions in tgt_out (ignore_index=PAD_ID).
  - The first decoder input position ([BOS] itself, which has no real
    preceding target). The collate already pads tgt_out so [BOS] is at
    tgt_in[0] and the first real target is at tgt_out[0].
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from ..constants import PAD_ID
from ..models.base import build_model
from .collate import Batch


@dataclass
class TrainMetrics:
    epoch: int
    train_loss: float
    val_loss: float
    learning_rate: float


def build_optimizer(model: nn.Module, cfg: dict[str, Any]) -> Any:
    name = cfg.get("optimizer", "adamw")
    lr = cfg.get("learning_rate", 3e-4)
    weight_decay = cfg.get("weight_decay", 0.01)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "muon":
        from .optim import MuonAdamWHybrid
        return MuonAdamWHybrid(
            model,
            muon_lr=cfg.get("muon_lr", 0.02),
            adam_lr=lr,
            adam_weight_decay=weight_decay,
            muon_momentum=cfg.get("muon_momentum", 0.95),
            ns_steps=cfg.get("ns_steps", 5),
            cross_attn_lr_mult=cfg.get("cross_attn_lr_mult", 1.0),
            spectral_cap=cfg.get("spectral_cap"),
            heavy_tail_alpha=cfg.get("heavy_tail_alpha"),
            adamuon_beta=cfg.get("adamuon_beta"),
            normuon_enabled=cfg.get("normuon_enabled", False),
        )
    raise ValueError(f"unknown optimizer: {name}")


@torch.no_grad()
def _scheduled_sample(model: nn.Module, src: torch.Tensor, tgt_in_gold: torch.Tensor) -> torch.Tensor:
    """Replace gold target prefix with model's own greedy predictions.

    Used for scheduled sampling during training: with some probability,
    feed the model its own argmax prediction at each step instead of the
    gold token. Reduces exposure bias (model trained on gold prefixes,
    evaluated on own).

    Args:
        model: trained seq2seq (eval mode is set internally so dropout is off).
        src: (B, T_src) source IDs.
        tgt_in_gold: (B, T_tgt) gold decoder input.

    Returns: (B, T_tgt) decoder input with positions t > 0 replaced by
        model's argmax prediction from position t-1.
    """
    was_training = model.training
    model.eval()
    B, T = tgt_in_gold.shape
    # Run forward with gold prefix to get per-step predictions.
    logits = model(src, tgt_in_gold)  # (B, T, V)
    preds = logits.argmax(dim=-1)     # (B, T)
    # Shift predictions: pred at position t was for input position t,
    # so it becomes the "own" input at position t+1.
    own_input = torch.full_like(tgt_in_gold, PAD_ID)
    own_input[:, 0] = tgt_in_gold[:, 0]  # keep BOS at position 0
    own_input[:, 1:] = preds[:, :-1]     # shift predictions right
    if was_training:
        model.train()
    return own_input


def build_scheduler(optimizer: Any, cfg: dict[str, Any], total_steps: int) -> Any:
    name = cfg.get("scheduler", "cosine")
    warmup_steps = cfg.get("warmup_steps", 200)
    if name == "cosine":

        class WarmupCosine:
            def __init__(self, optimizer, warmup, total):
                self.optimizer = optimizer
                self.warmup = warmup
                self.total = total
                self.last_epoch = 0
                self.base_lrs = [float(g["lr"]) for g in optimizer.param_groups]

            def get_lr(self) -> list[float]:
                step = self.last_epoch
                if step < self.warmup:
                    return [base * step / max(1, self.warmup) for base in self.base_lrs]
                progress = (step - self.warmup) / max(1, self.total - self.warmup)
                return [base * 0.5 * (1 + math.cos(math.pi * progress)) for base in self.base_lrs]

            def step(self) -> None:
                self.last_epoch += 1
                for g, lr in zip(self.optimizer.param_groups, self.get_lr()):
                    g["lr"] = lr

            def state_dict(self) -> dict[str, Any]:
                return {"last_epoch": self.last_epoch, "base_lrs": list(self.base_lrs)}

            def load_state_dict(self, state: dict[str, Any]) -> None:
                self.last_epoch = int(state.get("last_epoch", 0))
                if "base_lrs" in state:
                    self.base_lrs = list(state["base_lrs"])

        return WarmupCosine(optimizer, warmup_steps, total_steps)
    if name == "constant":

        class _NoOp:
            def step(self): pass
            def state_dict(self): return {}
            def load_state_dict(self, s): pass

        return _NoOp()
    raise ValueError(f"unknown scheduler: {name}")


def masked_label_smoothing_ce(
    logits: torch.Tensor,
    target: torch.Tensor,
    label_smoothing: float = 0.0,
    ignore_index: int = PAD_ID,
    class_weights: torch.Tensor | None = None,
    focal_gamma: float = 0.0,
) -> torch.Tensor:
    """Per-position CE over output vocab, ignoring PAD positions.

    Args:
        logits: (B, T, V) — decoder output at each tgt position.
        target: (B, T) — tgt_out (target tokens + EOS, padded).
        class_weights: per-class weights (shape [V]) for imbalanced classes.
        focal_gamma: focal loss gamma (0 = standard CE).
    """
    flat_logits = logits.reshape(-1, logits.size(-1))
    flat_target = target.reshape(-1)
    if focal_gamma and focal_gamma > 0:
        ce = nn.functional.cross_entropy(
            flat_logits, flat_target,
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
            weight=class_weights,
            reduction="none",
        )
        with torch.no_grad():
            p_t = torch.gather(flat_logits.softmax(-1), -1, flat_target.clamp_min(0).unsqueeze(-1)).squeeze(-1)
            p_t = p_t.where(flat_target != ignore_index, torch.ones_like(p_t))
        loss = ((1 - p_t) ** focal_gamma) * ce
        mask = flat_target != ignore_index
        return loss.sum() / mask.sum().clamp_min(1)
    return nn.functional.cross_entropy(
        flat_logits,
        flat_target,
        ignore_index=ignore_index,
        label_smoothing=label_smoothing,
        weight=class_weights,
    )


def entropy_regularizer(logits: torch.Tensor, weight: float = 0.0) -> torch.Tensor:
    """Entropy regularization: -weight * E[H(p)]. Encourages less confident predictions."""
    if weight <= 0:
        return torch.tensor(0.0, device=logits.device)
    flat = logits.reshape(-1, logits.size(-1))
    probs = flat.softmax(-1)
    log_probs = flat.log_softmax(-1)
    entropy = -(probs * log_probs).sum(-1).mean()
    return -weight * entropy


def compute_class_weights_from_loader(
    train_loader: DataLoader,
    vocab_size: int,
    device: torch.device,
    smoothing: float = 0.1,
) -> torch.Tensor:
    """Compute inverse-frequency class weights from training targets.

    For seq2seq: targets are tgt_out (decoder output tokens).
    """
    counts = torch.zeros(vocab_size)
    for batch in train_loader:
        tgt = batch.tgt_out if hasattr(batch, "tgt_out") else batch.targets[0]
        flat = tgt.reshape(-1)
        flat = flat[flat != PAD_ID]
        counts += torch.bincount(flat, minlength=vocab_size).cpu()
    counts = counts + smoothing * counts.max()
    w = counts.sum() / (vocab_size * counts)
    w = w / w.mean()
    return w.to(device)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, label_smoothing: float = 0.0) -> float:
    model.eval()
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for batch in loader:
            src = batch.src.to(device)
            tgt_in = batch.tgt_in.to(device)
            tgt_out = batch.tgt_out.to(device)
            logits = model(src, tgt_in)
            loss = masked_label_smoothing_ce(logits, tgt_out, label_smoothing=label_smoothing)
            total_loss += loss.item() * src.size(0)
            total_count += src.size(0)
    return total_loss / max(1, total_count)


def train_supervised(
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: dict[str, Any],
    device: torch.device,
    ckpt_root: Path,
    log_fn: Callable[[TrainMetrics], None] | None = None,
    metrics_path: Path | None = None,
) -> nn.Module:
    """Run seq2seq teacher-forcing training. Returns the trained model."""
    from .resume import (
        latest_resume_checkpoint,
        load_resume_state,
        save_resumable_checkpoint,
    )

    metrics_logger = None
    if metrics_path is not None:
        from .metrics import SimpleMetricsLogger
        metrics_logger = SimpleMetricsLogger(metrics_path)

    cfg_train = cfg.get("train", {})
    epochs = cfg_train.get("epochs", 20)
    fp16 = cfg_train.get("fp16", True)
    grad_clip = cfg_train.get("grad_clip", 1.0)
    label_smoothing = cfg_train.get("label_smoothing", 0.1)
    init_from_pretrain = cfg_train.get("init_from_pretrain")
    qk_clip_every = cfg_train.get("qk_clip_every", 0)  # 0 = disabled
    qk_clip_tau_init = cfg_train.get("qk_clip_tau_init", 8.0)
    qk_clip_tau_final = cfg_train.get("qk_clip_tau_final", 1.0)

    model = build_model(cfg).to(device)
    if init_from_pretrain:
        from .pretrain import load_pretrained_encoder
        load_pretrained_encoder(Path(init_from_pretrain), model)
        model.to(device)
    total_steps = epochs * len(train_loader)
    optimizer = build_optimizer(model, cfg_train)
    scheduler = build_scheduler(optimizer, cfg_train, total_steps)

    from .optim import MuonAdamWHybrid, qk_clip_, qk_clip_schedule
    use_scaler = fp16 and device.type == "cuda" and not isinstance(optimizer, MuonAdamWHybrid)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    best_val = float("inf")
    start_epoch = 0
    epochs_since_best = 0
    ckpt_root.mkdir(parents=True, exist_ok=True)

    resume = latest_resume_checkpoint(ckpt_root)
    if resume is not None:
        resume_path, last_epoch = resume
        if last_epoch >= 0:
            state = load_resume_state(model, optimizer, scheduler, resume_path, device=str(device))
            best_val = state.get("best_val_loss", float("inf"))
            start_epoch = last_epoch + 1
            if log_fn is not None:
                log_fn(TrainMetrics(
                    epoch=last_epoch, train_loss=0.0, val_loss=best_val,
                    learning_rate=optimizer.param_groups[0]["lr"],
                ))
            print(f"[resume] continued from {resume_path.name} at epoch {start_epoch}/{epochs}")

    global_step = start_epoch * len(train_loader)
    # Scheduled sampling: probability of feeding model's own prediction
    # instead of gold token. Linearly anneal from 0 → ss_final over epochs.
    # Addresses exposure bias (model trained on gold prefixes, evaluated on own).
    ss_final = cfg_train.get("scheduled_sampling_final", 0.0)
    ss_decay = cfg_train.get("scheduled_sampling_decay", "linear")
    label_smoothing = cfg_train.get("label_smoothing", 0.1)

    # SOTA options.
    focal_gamma = float(cfg_train.get("focal_gamma", 0.0))
    use_class_weights = bool(cfg_train.get("class_weights", False))
    entropy_weight = float(cfg_train.get("entropy_weight", 0.0))
    ema_decay = float(cfg_train.get("ema_decay", 0.0))

    # Compute class weights from training data (one-shot, before training).
    class_weights = None
    if use_class_weights:
        output_vocab = model.output_vocab_size
        class_weights = compute_class_weights_from_loader(train_loader, output_vocab, device)
        print(f"[train] class weights computed: min={class_weights.min().item():.3f} "
              f"max={class_weights.max().item():.3f} mean={class_weights.mean().item():.3f}", flush=True)

    # Initialize EMA after model is loaded.
    ema = None
    if ema_decay > 0:
        from .ema import ModelEMA
        ema = ModelEMA(model, decay=ema_decay)
        print(f"[train] EMA enabled, decay={ema_decay}", flush=True)

    for epoch in range(start_epoch, epochs):
        # Compute ss probability for this epoch.
        if ss_final > 0 and epochs > 1:
            progress = epoch / max(1, epochs - 1)
            ss_prob = ss_final * progress if ss_decay == "linear" else ss_final * (progress ** 2)
        else:
            ss_prob = 0.0
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            src = batch.src.to(device)
            tgt_in = batch.tgt_in.to(device)
            tgt_out = batch.tgt_out.to(device)
            # Scheduled sampling: with prob ss_prob per step, replace
            # tgt_in[t] with argmax(model_logits[t-1]) for t > 0.
            if ss_prob > 0 and global_step > 0 and random.random() < ss_prob:
                tgt_in_scheduled = _scheduled_sample(model, src, tgt_in)
            else:
                tgt_in_scheduled = tgt_in
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=fp16):
                logits = model(src, tgt_in_scheduled)
                loss = masked_label_smoothing_ce(
                    logits, tgt_out,
                    label_smoothing=label_smoothing,
                    class_weights=class_weights,
                    focal_gamma=focal_gamma,
                )
                if entropy_weight > 0:
                    loss = loss + entropy_regularizer(logits, weight=entropy_weight)
            # Skip NaN/Inf loss — protects weights from poisoning. Just zero
            # grads and continue with the next batch.
            if not torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                continue
            if use_scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                # Skip step if any grad went bad (NaN/Inf from bf16 instability).
                all_finite = all(
                    p.grad is None or torch.isfinite(p.grad).all().item()
                    for p in model.parameters()
                )
                if not all_finite:
                    optimizer.zero_grad(set_to_none=True)
                    continue
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                if ema is not None:
                    ema.update(model)
            scheduler.step()

            # QK-Clip (optional) — anneal tau from tau_init → tau_final.
            if qk_clip_every > 0 and global_step % qk_clip_every == 0:
                tau = qk_clip_schedule(global_step, total_steps, qk_clip_tau_init, qk_clip_tau_final)
                qk_clip_(model, tau=tau)

            running_loss += loss.item() * src.size(0)
            global_step += 1

        train_loss = running_loss / max(1, len(train_loader.dataset))
        # Use EMA copy for evaluation if available (smoothed predictions).
        if ema is not None:
            with ema.swap(model):
                val_loss = evaluate(model, val_loader, device, label_smoothing=0.0)
        else:
            val_loss = evaluate(model, val_loader, device, label_smoothing=0.0)
        metrics = TrainMetrics(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            learning_rate=optimizer.param_groups[0]["lr"],
        )
        if log_fn is not None:
            log_fn(metrics)
        if metrics_logger is not None:
            metrics_logger.log(metrics)
        print(f"[epoch {epoch}] train={train_loss:.4f} val={val_loss:.4f} lr={metrics.learning_rate:.2e}")

        save_resumable_checkpoint(
            ckpt_root / f"checkpoint-epoch-{epoch}.pt",
            model, optimizer, scheduler,
            epoch=epoch, best_val_loss=best_val,
        )
        # Update best.pt. Skip NaN val_loss; always save on first epoch
        # if best doesn't exist yet so downstream can find a checkpoint.
        import math as _math
        best_path = ckpt_root / "best.pt"
        val_is_better = (not _math.isnan(val_loss)) and (val_loss < best_val)
        if val_is_better or (epoch == start_epoch and not best_path.is_file()):
            if val_is_better:
                best_val = val_loss
                epochs_since_best = 0
            # Save EMA weights as best.pt (better generalization at inference).
            if ema is not None:
                with ema.swap(model):
                    torch.save(model.state_dict(), best_path)
            else:
                torch.save(model.state_dict(), best_path)
        else:
            epochs_since_best += 1

        # Early stopping: break if val_loss hasn't improved for `patience` epochs.
        es_patience = int(cfg_train.get("early_stopping_patience", 0))
        if (
            es_patience > 0
            and epochs_since_best >= es_patience
            and epoch < epochs - 1
        ):
            print(
                f"[train] early stopping at epoch {epoch}: no val_loss improvement "
                f"for {epochs_since_best} epochs (patience={es_patience})",
                flush=True,
            )
            break

    if metrics_logger is not None:
        metrics_logger.close()
    return model
