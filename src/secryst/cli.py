"""CLI entry points — thin wrappers around modal_app.py functions.

For local usage without Modal. Most training runs go through Modal
(`modal run modal_app.py::pretrain` etc.) — this CLI exists for
local-dev smoke tests and unit-test fixtures.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


def pretrain_main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run MLM pretraining locally")
    p.add_argument("--task", required=True)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--ckpt-root", default="./checkpoints")
    args = p.parse_args(argv)

    from .config import load_task_config, to_dict
    from .tasks import build_mlm_loaders
    from .training import pretrain_mlm

    cfg = load_task_config(args.task)
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    train_loader, val_loader = build_mlm_loaders(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrain_mlm(
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=to_dict(cfg),
        device=device,
        ckpt_root=Path(args.ckpt_root) / args.task / "run-001",
    )
    return 0


def train_main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run supervised training locally")
    p.add_argument("--task", required=True)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--init-from-pretrain", default=None)
    p.add_argument("--ckpt-root", default="./checkpoints")
    args = p.parse_args(argv)

    from .config import load_task_config, to_dict
    from .tasks import build_supervised_loaders
    from .training import train_supervised

    cfg = load_task_config(args.task)
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    if args.init_from_pretrain:
        cfg.train.init_from_pretrain = args.init_from_pretrain
    train_loader, val_loader = build_supervised_loaders(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_supervised(
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=to_dict(cfg),
        device=device,
        ckpt_root=Path(args.ckpt_root) / args.task / "run-001",
    )
    return 0


def export_main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Export model → ONNX fp32")
    p.add_argument("--task", required=True)
    p.add_argument("--version", default="v0.1.0")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    from .config import load_task_config, to_dict
    from .export import export_seq2seq_onnx

    cfg = load_task_config(args.task)
    cfg_dict = to_dict(cfg)
    checkpoint = args.checkpoint or f"./checkpoints/{args.task}/run-001/best.pt"
    out_path = Path(args.out or f"./models/{args.task}/{args.task}-{args.version}-fp32.onnx")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    export_seq2seq_onnx(Path(checkpoint), cfg_dict, out_path)
    print(f"Wrote {out_path}")
    return 0


def evaluate_main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Evaluate PER on test split")
    p.add_argument("--task", required=True)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--beam-width", type=int, default=4)
    args = p.parse_args(argv)

    import json

    from .config import load_task_config, to_dict
    from .evaluate import evaluate_per
    from .models.base import build_model
    from .tasks import build_test_loader

    cfg = load_task_config(args.task)
    cfg_dict = to_dict(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = args.checkpoint or f"./checkpoints/{args.task}/run-001/best.pt"

    model = build_model(cfg_dict).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    loader = build_test_loader(args.task)
    result = evaluate_per(model, loader, device, beam_width=args.beam_width)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(pretrain_main())
