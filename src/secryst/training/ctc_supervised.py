"""CTC training loop for secryst Thai→IPA.

Replaces seq2seq teacher-forcing for cases where the decoder collapses to
ignoring encoder memory. CTC has no decoder at all — the encoder's per-
position output directly determines the target token (via collapse +
blank removal), so mode collapse is structurally impossible.

Pipeline integration: `pretrain_method: ctc` would dispatch here. For now,
this is a standalone module that can be called from a custom pipeline.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from ..models.ctc import CTCModel, build_ctc_model
from .resume import latest_resume_checkpoint, load_resume_state, save_resumable_checkpoint
from .supervised import TrainMetrics, build_optimizer, build_scheduler


def _batch_to_ctc(batch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert a collated Batch (from ipa_collate_batch) to CTC-ready tensors.

    Returns:
        src: (B, T_src) padded source IDs.
        src_lengths: (B,) true source lengths.
        target_flat: (sum(target_lengths),) — all targets concatenated
            (CTCLoss expects flat target tensor, not padded).
        target_lengths: (B,) lengths of each target.
    """
    from ..constants import EOS_ID, PAD_ID

    src = batch.src
    src_lengths = batch.src_lengths
    tgt_out = batch.tgt_out

    target_list = []
    target_lengths = []
    for i in range(tgt_out.size(0)):
        row = tgt_out[i]
        eos_positions = (row == EOS_ID).nonzero(as_tuple=True)[0]
        if len(eos_positions) > 0:
            target_ids = row[: eos_positions[0].item()]
        else:
            target_ids = row[row != PAD_ID]
        target_list.append(target_ids)
        target_lengths.append(len(target_ids))

    target_flat = torch.cat(target_list) if target_list else torch.zeros(0, dtype=torch.long)
    target_lengths = torch.tensor(target_lengths, dtype=torch.long)
    return src, src_lengths, target_flat, target_lengths


def evaluate_ctc(model: CTCModel, loader: DataLoader, device: torch.device) -> float:
    """Compute average CTC loss on a loader."""
    model.eval()
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for batch in loader:
            src, src_lengths, target_flat, target_lengths = _batch_to_ctc(batch)
            src = src.to(device)
            src_lengths = src_lengths.to(device)
            target_flat = target_flat.to(device)
            target_lengths = target_lengths.to(device)
            out = model(src, src_lengths, target_flat, target_lengths)
            loss = out["loss"]
            if torch.isfinite(loss):
                total_loss += loss.item() * src.size(0)
                total_count += src.size(0)
    return total_loss / max(1, total_count)


def train_ctc(
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: dict[str, Any],
    device: torch.device,
    ckpt_root: Path,
    log_fn: Callable[[TrainMetrics], None] | None = None,
    metrics_path: Path | None = None,
) -> CTCModel:
    """Run CTC training. Returns the trained model.

    The model is checkpointed to `ckpt_root/checkpoint-epoch-{N}.pt` and
    `ckpt_root/best.pt` (lowest val_loss).
    """
    cfg_train = cfg.get("train", {})
    epochs = cfg_train.get("epochs", 30)
    fp16 = cfg_train.get("fp16", True)
    grad_clip = cfg_train.get("grad_clip", 1.0)

    metrics_logger = None
    if metrics_path is not None:
        from .metrics import SimpleMetricsLogger
        metrics_logger = SimpleMetricsLogger(metrics_path)

    model = build_ctc_model(cfg).to(device)
    total_steps = epochs * len(train_loader)
    optimizer = build_optimizer(model, cfg_train)
    scheduler = build_scheduler(optimizer, cfg_train, total_steps)

    from .optim import MuonAdamWHybrid
    use_scaler = fp16 and device.type == "cuda" and not isinstance(optimizer, MuonAdamWHybrid)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    best_val = float("inf")
    start_epoch = 0
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

    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            src, src_lengths, target_flat, target_lengths = _batch_to_ctc(batch)
            src = src.to(device)
            src_lengths = src_lengths.to(device)
            target_flat = target_flat.to(device)
            target_lengths = target_lengths.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=fp16):
                out = model(src, src_lengths, target_flat, target_lengths)
                loss = out["loss"]
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
        val_loss = evaluate_ctc(model, val_loader, device)
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

        save_resumable_checkpoint(
            ckpt_root / f"checkpoint-epoch-{epoch}.pt",
            model, optimizer, scheduler,
            epoch=epoch, best_val_loss=best_val,
        )
        import math as _math
        best_path = ckpt_root / "best.pt"
        val_is_better = (not _math.isnan(val_loss)) and (val_loss < best_val)
        if val_is_better or (epoch == start_epoch and not best_path.is_file()):
            if val_is_better:
                best_val = val_loss
            torch.save(model.state_dict(), best_path)

    if metrics_logger is not None:
        metrics_logger.close()
    return model
