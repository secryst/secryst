"""MLM pretraining loop for the encoder-only Thai model."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from ..constants import PAD_ID
from ..models.seq2seq import build_pretrain_model, extract_pretrained_encoder
from .collate import MLMBatch
from .supervised import TrainMetrics, build_optimizer, build_scheduler


def mlm_collate_to_batch(batch_obj: MLMBatch, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    src = batch_obj.src.to(device)
    lengths = batch_obj.lengths.to(device)
    target = batch_obj.targets[0].to(device)
    return src, lengths, target


def masked_lm_ce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    flat_logits = logits.reshape(-1, logits.size(-1))
    flat_target = target.reshape(-1)
    return nn.functional.cross_entropy(flat_logits, flat_target, ignore_index=PAD_ID)


def evaluate_mlm(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for batch in loader:
            src, lengths, target = mlm_collate_to_batch(batch, device)
            logits = model(src, lengths)
            loss = masked_lm_ce(logits, target)
            total_loss += loss.item() * src.size(0)
            total_count += src.size(0)
    return total_loss / max(1, total_count)


def pretrain_mlm(
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: dict[str, Any],
    device: torch.device,
    ckpt_root: Path,
    log_fn: Callable[[TrainMetrics], None] | None = None,
) -> tuple[nn.Module, Path]:
    """Run MLM pretraining on the encoder-only model."""
    cfg_train = cfg.get("train", {})
    epochs = cfg_train.get("epochs", 6)
    fp16 = cfg_train.get("fp16", True)
    grad_clip = cfg_train.get("grad_clip", 1.0)

    from .resume import latest_resume_checkpoint

    model = build_pretrain_model(cfg).to(device)
    total_steps = epochs * len(train_loader)
    optimizer = build_optimizer(model, cfg_train)
    scheduler = build_scheduler(optimizer, cfg_train, total_steps)

    from .optim import MuonAdamWHybrid
    use_scaler = fp16 and device.type == "cuda" and not isinstance(optimizer, MuonAdamWHybrid)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    best_val = float("inf")
    start_epoch = 0
    ckpt_root.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_root / "best.pt"

    resume = latest_resume_checkpoint(ckpt_root)
    if resume is not None:
        resume_path, last_epoch = resume
        if last_epoch >= 0:
            state = torch.load(resume_path, map_location=str(device), weights_only=False)
            if "model" in state:
                model.load_state_dict(state["model"])
            if "optimizer" in state:
                optimizer.load_state_dict(state["optimizer"])
            if "scheduler" in state:
                try:
                    scheduler.load_state_dict(state["scheduler"])
                except Exception:
                    pass
            best_val = state.get("val_loss", float("inf"))
            start_epoch = last_epoch + 1
            print(f"[resume] continued from {resume_path.name} at epoch {start_epoch}/{epochs}")

    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            src, lengths, target = mlm_collate_to_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=fp16):
                logits = model(src, lengths)
                loss = masked_lm_ce(logits, target)
            # Skip NaN/Inf loss — protects weights from poisoning.
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
                all_finite = all(
                    p.grad is None or torch.isfinite(p.grad).all().item()
                    for p in model.parameters()
                )
                if not all_finite:
                    optimizer.zero_grad(set_to_none=True)
                    continue
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            scheduler.step()
            running_loss += loss.item() * src.size(0)

        train_loss = running_loss / max(1, len(train_loader.dataset))
        val_loss = evaluate_mlm(model, val_loader, device)
        metrics = TrainMetrics(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            learning_rate=optimizer.param_groups[0]["lr"],
        )
        if log_fn is not None:
            log_fn(metrics)
        print(f"[pretrain epoch {epoch}] train={train_loss:.4f} val={val_loss:.4f}")

        full_ckpt_path = ckpt_root / f"checkpoint-epoch-{epoch}.pt"
        torch.save(
            {
                "epoch": epoch,
                "val_loss": val_loss,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "encoder_state_dict": extract_pretrained_encoder(model),
            },
            full_ckpt_path,
        )
        # Update best.pt. Skip NaN val_loss; always save on first epoch
        # if best doesn't exist yet so downstream can find a checkpoint.
        import math as _math
        val_is_better = (not _math.isnan(val_loss)) and (val_loss < best_val)
        if val_is_better or (epoch == start_epoch and not best_path.is_file()):
            if val_is_better:
                best_val = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "encoder_state_dict": extract_pretrained_encoder(model),
                },
                best_path,
            )

    return model, best_path


def load_pretrained_encoder(checkpoint_path: Path, model: nn.Module) -> None:
    """Load encoder weights from a pretrain checkpoint into a seq2seq model.

    The seq2seq model's encoder is at `model.encoder.*`; the pretrain
    checkpoint saved it as `encoder_state_dict` keyed by encoder-submodule
    names. We load with strict=False so the decoder stays at fresh init.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    encoder_state = ckpt.get("encoder_state_dict", ckpt)
    # Load into the seq2seq's encoder submodule directly.
    target = model.encoder if hasattr(model, "encoder") else model
    target.load_state_dict(encoder_state, strict=False)
