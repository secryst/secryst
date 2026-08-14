"""Modal app for secryst training + export + evaluation (Thai → IPA).

Usage:
    modal token new

    # Test data fetch:
    modal run modal_app.py::fetch_data --task secryst_thai_ipa

    # MLM pretrain (A100, ~2h):
    modal run modal_app.py::pretrain --task secryst_thai_ipa_pretrain

    # Train (A100, ~3h):
    modal run modal_app.py::train --task secryst_thai_ipa \\
        --init-from-pretrain /checkpoints/secryst_thai_ipa_pretrain/run-001/best.pt

    # Export to ONNX fp32 only (no quantization):
    modal run modal_app.py::export_onnx --task secryst_thai_ipa --version v0.1.0

    # Evaluate (A10G):
    modal run modal_app.py::evaluate --task secryst_thai_ipa

    # Full chain (disconnect-safe):
    modal run --detach modal_app.py::sota_pipeline

Volumes:
    datasets     — kaikki.org Thai-IPA pairs + Thai MLM corpus.
    checkpoints  — per-epoch + best.pt model weights.
    models       — final ONNX fp32 artifacts.
"""

from __future__ import annotations

import modal
from pathlib import Path

APP_NAME = "secryst"
PYTHON_VERSION = "3.11"

datasets_volume = modal.Volume.from_name(f"{APP_NAME}-datasets", create_if_missing=True)
checkpoints_volume = modal.Volume.from_name(f"{APP_NAME}-checkpoints", create_if_missing=True)
models_volume = modal.Volume.from_name(f"{APP_NAME}-models", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version=PYTHON_VERSION)
    .apt_install("build-essential", "git", "curl", "zstd")
    .pip_install(
        "torch>=2.4,<3",
        "numpy>=1.26,<3",
        "omegaconf>=2.3,<3",
        "onnx>=1.17",
        "onnxscript>=0.1",
        "onnxruntime>=1.20",
        "tqdm>=4.66",
        "pyyaml>=6.0",
        "transformers>=4.46",
        "accelerate>=1.1",
        "sentencepiece>=0.2",
    )
    .add_local_dir("src", "/opt/secryst/src", copy=True)
    .add_local_dir("configs", "/opt/secryst/configs", copy=True)
    .add_local_file("pyproject.toml", "/opt/secryst/pyproject.toml", copy=True)
    .workdir("/opt/secryst")
    .env({"PYTHONPATH": "/opt/secryst/src"})
)

app = modal.App(name=APP_NAME, image=image)


# ---- Data: Kaikki Thai-IPA -------------------------------------------------


KAIKKI_THAI_URL = "https://kaikki.org/dictionary/Thai/kaikki.org-dictionary-Thai.jsonl"
KAIKLI_THAI_FALLBACK_URLS = [
    # Kaikki sometimes distributes via mirror / yomtanbleasts; both work.
    "https://kaikki.org/dictionary/Thai/kaikki.org-dictionary-Thai.jsonl.zst",
]


@app.function(
    cpu=2,
    timeout=2 * 60 * 60,
    volumes={"/datasets": datasets_volume},
)
def fetch_data(task: str) -> dict[str, object]:
    """Fetch and preprocess Kaikki Thai dictionary.

    Output: two datasets on /datasets volume.

    1. `/datasets/thai-ipa/{train,val,test}.jsonl` — supervised IPA pairs.
       Format: `{"src": "ภาษา", "tgt": "pʰaː˥"}` per line.

    2. `/datasets/thai-text/train.txt` — undiacritized Thai for MLM pretrain.
       Pulled from Kaikki `examples` field (headword + example sentences).

    Idempotent: if outputs already exist with non-zero size, skip download.
    """
    import hashlib
    import json
    import os
    import random
    import subprocess
    from pathlib import Path

    from secryst.datasets import load_kaikki_pairs
    from secryst.encoder import ThaiEncoder

    summary: dict[str, object] = {"task": task, "files": {}}

    ipa_root = Path("/datasets/thai-ipa")
    text_root = Path("/datasets/thai-text")
    ipa_root.mkdir(parents=True, exist_ok=True)
    text_root.mkdir(parents=True, exist_ok=True)

    # Skip download if splits already exist with content.
    if all((ipa_root / f"{s}.jsonl").is_file() and (ipa_root / f"{s}.jsonl").stat().st_size > 0
           for s in ("train", "val", "test")):
        print(f"[fetch] thai-ipa splits already present at {ipa_root}")
    else:
        # Download raw Kaikki dump.
        raw_path = Path("/tmp/kaikki-thai.jsonl")
        if not raw_path.is_file() or raw_path.stat().st_size == 0:
            print(f"[fetch] downloading {KAIKKI_THAI_URL} ...")
            for url in [KAIKKI_THAI_URL, *KAIKLI_THAI_FALLBACK_URLS]:
                try:
                    subprocess.run(
                        ["curl", "-fsSL", "-o", str(raw_path), url],
                        check=True, timeout=30 * 60,
                    )
                    break
                except subprocess.CalledProcessError as e:
                    print(f"[fetch] curl {url} failed: {e}")
                    continue
            else:
                raise RuntimeError(f"all Kaikki download URLs failed")
            # Decompress if .zst
            if raw_path.stat().st_size < 1000 or raw_path.read_bytes()[:4] == b"\x28\xb5\x2f\xfd":
                zst_path = raw_path.with_suffix(".jsonl.zst")
                raw_path.rename(zst_path)
                subprocess.run(["zstd", "-d", str(zst_path), "-o", str(raw_path)], check=True)
            print(f"[fetch] downloaded {raw_path.stat().st_size:,} bytes")

        # Parse pairs.
        enc = ThaiEncoder(cleaner="thai")
        all_pairs = load_kaikki_pairs(raw_path, enc, max_pairs=None)
        print(f"[fetch] loaded {len(all_pairs):,} (Thai, IPA) pairs")

        # Shuffle + 80/10/10 split.
        rng = random.Random(42)
        rng.shuffle(all_pairs)
        n = len(all_pairs)
        n_train = int(n * 0.8)
        n_val = int(n * 0.1)
        splits = {
            "train": all_pairs[:n_train],
            "val": all_pairs[n_train : n_train + n_val],
            "test": all_pairs[n_train + n_val :],
        }

        for name, items in splits.items():
            out = ipa_root / f"{name}.jsonl"
            with out.open("w", encoding="utf-8") as f:
                for ex in items:
                    f.write(json.dumps({"src": ex.raw_src, "tgt": ex.raw_tgt}, ensure_ascii=False) + "\n")
            line_count = sum(1 for _ in out.open(encoding="utf-8"))
            sha = hashlib.sha256(out.read_bytes()).hexdigest()[:16]
            summary["files"].setdefault("ipa", {})[name] = {
                "path": str(out), "lines": line_count, "sha256": sha,
            }
            print(f"[fetch]   {name}.jsonl: {line_count:,} lines")

    # MLM corpus: pull headwords + example sentences from the same dump.
    if (text_root / "train.txt").is_file() and (text_root / "train.txt").stat().st_size > 0:
        print(f"[fetch] thai-text/train.txt already present")
    else:
        raw_path = Path("/tmp/kaikki-thai.jsonl")
        if not raw_path.is_file() or raw_path.stat().st_size == 0:
            print(f"[fetch] raw Kaikki dump missing — skipping MLM corpus build")
        else:
            # Extract headword + any example sentences.
            sentences: list[str] = []
            seen: set[str] = set()
            with raw_path.open(encoding="utf-8") as f:
                for ln in f:
                    ln = ln.strip()
                    if not ln:
                        continue
                    try:
                        entry = json.loads(ln)
                    except json.JSONDecodeError:
                        continue
                    word = (entry.get("word") or "").strip()
                    if word and word not in seen and len(word) >= 2:
                        seen.add(word)
                        sentences.append(word)
                    # Example sentences live under senses[].examples[].text
                    for sense in entry.get("senses") or []:
                        for ex in sense.get("examples") or []:
                            text = ""
                            if isinstance(ex, dict):
                                text = (ex.get("text") or "").strip()
                            elif isinstance(ex, str):
                                text = ex.strip()
                            if text and text not in seen:
                                seen.add(text)
                                sentences.append(text)

            rng = random.Random(42)
            rng.shuffle(sentences)
            n = len(sentences)
            n_train = int(n * 0.9)
            n_val = n - n_train
            (text_root / "train.txt").write_text("\n".join(sentences[:n_train]) + "\n", encoding="utf-8")
            (text_root / "val.txt").write_text("\n".join(sentences[n_train:]) + "\n", encoding="utf-8")
            summary["files"]["thai-text"] = {
                "train_lines": n_train,
                "val_lines": n_val,
            }
            print(f"[fetch] thai-text: train={n_train:,} val={n_val:,} lines")

    datasets_volume.commit()
    return summary


# ---- Training stages -------------------------------------------------------


@app.function(
    gpu="A100",
    timeout=6 * 60 * 60,
    volumes={"/checkpoints": checkpoints_volume, "/datasets": datasets_volume},
)
def train(
    task: str,
    epochs: int | None = None,
    init_from_pretrain: str | None = None,
) -> dict[str, object]:
    """Run supervised training. Returns path to best checkpoint.

    Dispatches to:
      - `train_ctc` if cfg.model.arch == "ctc"
      - `train_supervised` otherwise (seq2seq)
    """
    import torch

    # Explicit reload: ensure we see files written by fetch_data's volume commit.
    datasets_volume.reload()

    from secryst.config import load_task_config, to_dict
    from secryst.tasks import build_supervised_loaders

    cfg = load_task_config(task)
    arch = cfg.get("model", {}).get("arch", "modern_seq2seq") if hasattr(cfg, "get") else "modern_seq2seq"
    if epochs is not None:
        cfg.train.epochs = epochs
    if init_from_pretrain is not None:
        cfg.train.init_from_pretrain = init_from_pretrain

    device = torch.device("cuda")
    ckpt_root = Path("/checkpoints") / task / "run-001"
    metrics_path = Path("/checkpoints") / "metrics" / f"metrics-{task}-train.jsonl"

    if arch == "byt5_thai":
        from secryst.models.byt5_thai import train_byt5
        data_root = Path("/datasets/thai-ipa")
        best_path = train_byt5(
            cfg=to_dict(cfg),
            train_path=data_root / "train.jsonl",
            val_path=data_root / "val.jsonl",
            ckpt_root=ckpt_root,
            metrics_path=metrics_path,
        )
        checkpoints_volume.commit()
        datasets_volume.commit()
        return {"checkpoint_root": str(ckpt_root), "best": best_path}

    train_loader, val_loader = build_supervised_loaders(cfg)

    if arch == "ctc":
        from secryst.training.ctc_supervised import train_ctc
        train_ctc(
            train_loader=train_loader,
            val_loader=val_loader,
            cfg=to_dict(cfg),
            device=device,
            ckpt_root=ckpt_root,
            metrics_path=metrics_path,
        )
    else:
        from secryst.training import train_supervised
        train_supervised(
            train_loader=train_loader,
            val_loader=val_loader,
            cfg=to_dict(cfg),
            device=device,
            ckpt_root=ckpt_root,
        )
    checkpoints_volume.commit()
    return {"checkpoint_root": str(ckpt_root), "best": str(ckpt_root / "best.pt")}


@app.function(
    gpu="A100",
    timeout=6 * 60 * 60,
    volumes={"/checkpoints": checkpoints_volume, "/datasets": datasets_volume},
)
def pretrain(task: str, epochs: int | None = None) -> dict[str, object]:
    """Run pretraining (MLM or ELECTRA). Returns path to best encoder checkpoint."""
    import torch

    # Explicit reload: ensure we see files written by fetch_data's volume commit.
    datasets_volume.reload()

    from secryst.config import load_task_config, to_dict
    from secryst.tasks import build_mlm_loaders

    cfg = load_task_config(task)
    if epochs is not None:
        cfg.train.epochs = epochs

    train_loader, val_loader = build_mlm_loaders(cfg)
    method = cfg.train.get("pretrain_method", "mlm") if hasattr(cfg.train, "get") else "mlm"
    device = torch.device("cuda")
    ckpt_root = Path("/checkpoints") / task / "run-001"
    if method == "electra":
        try:
            from secryst.training.electra import pretrain_electra
            pretrain_electra(
                train_loader=train_loader, val_loader=val_loader,
                cfg=to_dict(cfg), device=device, ckpt_root=ckpt_root,
            )
        except ImportError:
            from secryst.training import pretrain_mlm
            pretrain_mlm(
                train_loader=train_loader, val_loader=val_loader,
                cfg=to_dict(cfg), device=device, ckpt_root=ckpt_root,
            )
    else:
        from secryst.training import pretrain_mlm
        pretrain_mlm(
            train_loader=train_loader, val_loader=val_loader,
            cfg=to_dict(cfg), device=device, ckpt_root=ckpt_root,
        )
    checkpoints_volume.commit()
    return {"checkpoint_root": str(ckpt_root), "best": str(ckpt_root / "best.pt")}


@app.function(
    gpu="A10G",
    timeout=30 * 60,
    volumes={"/checkpoints": checkpoints_volume, "/models": models_volume},
)
def export_onnx(task: str, version: str, checkpoint: str | None = None) -> dict[str, object]:
    """Export checkpoint → ONNX fp32 (NO quantization)."""
    from secryst.config import load_task_config, to_dict
    from secryst.export import export_seq2seq_onnx

    cfg = load_task_config(task)
    cfg_dict = to_dict(cfg)
    src_max_len = int(cfg.model.get("max_len", 64))
    tgt_max_len = int(cfg.model.get("max_len", 64))

    if checkpoint is None:
        checkpoint = str(Path("/checkpoints") / task / "run-001" / "best.pt")

    out_dir = Path("/models") / task
    out_dir.mkdir(parents=True, exist_ok=True)
    fp32_path = out_dir / f"{task}-{version}-fp32.onnx"

    export_seq2seq_onnx(Path(checkpoint), cfg_dict, fp32_path, batch_size=1, src_max_len=src_max_len, tgt_max_len=tgt_max_len)

    models_volume.commit()
    return {"fp32": str(fp32_path)}


@app.function(
    gpu="A10G",
    timeout=30 * 60,
    volumes={"/checkpoints": checkpoints_volume, "/datasets": datasets_volume},
)
def evaluate(
    task: str,
    checkpoint: str | None = None,
    beam_width: int = 4,
    eos_bias: float = 0.0,
) -> dict[str, object]:
    """Compute PER on test split using beam search.

    `eos_bias`: additive boost to the EOS logit at every step. Useful when
    the model under-predicts EOS (PER > 1.0 from runaway generation).
    """
    import json
    import torch

    # Explicit reload: ensure we see files written by fetch_data's volume commit.
    datasets_volume.reload()

    from secryst.config import load_task_config, to_dict
    from secryst.evaluate import evaluate_per, transliterate_all
    from secryst.models.base import build_model
    from secryst.tasks import build_test_loader

    cfg = load_task_config(task)
    cfg_dict = to_dict(cfg)
    device = torch.device("cuda")

    if checkpoint is None:
        checkpoint = str(Path("/checkpoints") / task / "run-001" / "best.pt")

    arch = cfg_dict.get("model", {}).get("arch", "modern_seq2seq")

    if arch == "byt5_thai":
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        from secryst.models.byt5_thai import evaluate_byt5
        from pathlib import Path as _P

        ckpt_dir = checkpoint
        if not _P(ckpt_dir).is_dir():
            ckpt_dir = str(_P(checkpoint).parent / "best")
        model = AutoModelForSeq2SeqLM.from_pretrained(ckpt_dir).to(device)
        tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
        result = evaluate_byt5(model, tokenizer, _P("/datasets/thai-ipa/test.jsonl"), device)
        result["task"] = task
        result["checkpoint"] = checkpoint
        result["arch"] = "byt5_thai"
        print("=== evaluate result (byt5) ===")
        print(json.dumps(result, indent=2, default=str))
        return result

    model = build_model(cfg_dict).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    # Tolerate both raw state_dict and wrapped {"model": ...} formats.
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    loader = build_test_loader(task)

    # CTC arch uses greedy decode instead of beam search.
    arch = cfg_dict.get("model", {}).get("arch", "modern_seq2seq")
    if arch == "ctc":
        from secryst.evaluate_ctc import evaluate_ctc_per
        metrics = evaluate_ctc_per(model, loader, device)
    else:
        metrics = evaluate_per(model, loader, device, beam_width=beam_width, eos_bias=eos_bias)
    result = {
        "task": task,
        "checkpoint": checkpoint,
        "beam_width": beam_width if arch != "ctc" else 0,
        "arch": arch,
        **metrics,
    }
    print("=== evaluate result ===")
    print(json.dumps(result, indent=2, default=str))

    # Dump 20 sample predictions for inspection (skip for CTC — no beam search).
    if arch != "ctc":
        samples = transliterate_all(model, loader, device, beam_width=beam_width)[:20]
        print("\n=== sample predictions ===")
        for src, gold, pred in samples:
            print(f"  {src!r}\n    gold: {gold}\n    pred: {pred}")

    return result


# ---- Full SOTA pipeline (server-side chain, survives disconnect) --------


@app.function(
    timeout=24 * 60 * 60,  # Modal max
    volumes={
        "/checkpoints": checkpoints_volume,
        "/datasets": datasets_volume,
        "/models": models_volume,
    },
)
def run_sota_pipeline(
    task: str = "secryst_thai_ipa",
    version: str = "v0.1.0",
    skip_fetch: bool = False,
    skip_pretrain: bool = False,
    skip_train: bool = False,
    skip_export: bool = False,
    force: bool = False,
) -> dict[str, object]:
    """Server-side chain: fetch → pretrain → train → export → evaluate."""
    from pathlib import Path as _Path

    from secryst.training.resume import (
        VolumeLogger,
        config_hash,
        is_stage_done,
        is_stage_done_with_config,
        mark_stage_done,
        mark_stage_failed,
    )
    from secryst.config import load_task_config, to_dict as _to_dict

    status_root = _Path("/checkpoints")
    log = VolumeLogger(status_root / "logs" / f"sota_pipeline-{task}.log")
    summary: dict[str, object] = {"task": task, "version": version, "stages": {}}

    # Snapshot current config for change detection. If the config hasn't
    # changed since a stage was marked done, we can skip even with --force
    # absent (saves work when re-running an unchanged pipeline).
    _cfg_snapshot = _to_dict(load_task_config(task))
    _cfg_hash = config_hash(_cfg_snapshot)

    # Pull latest committed state into our container's view so file checks
    # see artifacts from previous runs.
    checkpoints_volume.reload()
    datasets_volume.reload()
    models_volume.reload()

    def _stage_key(stage_name: str) -> str:
        return f"{task}:{stage_name}"

    def _done(stage_name: str) -> bool:
        if force:
            return False
        # Skip iff: stage marked done AND config hash matches current cfg.
        return is_stage_done_with_config(status_root, _stage_key(stage_name), _cfg_snapshot)

    def _mark_done(stage_name: str, extra: dict | None = None) -> None:
        """Wrap mark_stage_done to always include the current config hash."""
        merged = {"config_hash": _cfg_hash}
        if extra:
            merged.update(extra)
        mark_stage_done(status_root, _stage_key(stage_name), extra=merged)

    pretrain_task = f"{task}_pretrain"
    pretrain_best = _Path("/checkpoints") / pretrain_task / "run-001" / "best.pt"
    pretrain_latest_dir = _Path("/checkpoints") / pretrain_task / "run-001"
    train_best = _Path("/checkpoints") / task / "run-001" / "best.pt"
    onnx_fp32 = _Path("/models") / task / f"{task}-{version}-fp32.onnx"

    def _resolve_pretrain_ckpt() -> _Path | None:
        """Return best.pt if present, else the highest-epoch checkpoint."""
        if pretrain_best.is_file():
            return pretrain_best
        if pretrain_latest_dir.is_dir():
            # Sort by EPOCH NUMBER not lexicographically (so epoch-19 > epoch-9).
            import re as _re
            def _epoch_num(p: _Path) -> int:
                m = _re.search(r"checkpoint-epoch-(\d+)\.pt$", p.name)
                return int(m.group(1)) if m else -1
            epoch_ckpts = [p for p in pretrain_latest_dir.glob("checkpoint-epoch-*.pt") if _epoch_num(p) >= 0]
            if epoch_ckpts:
                return sorted(epoch_ckpts, key=_epoch_num)[-1]
        return None

    # ---- 1. fetch_data -------------------------------------------------
    stage = "fetch"
    if skip_fetch or _done(stage):
        log.log(f"[{stage}] skipped")
        summary["stages"][stage] = {"skipped": True}
    else:
        if force:
            import shutil
            for d in ("/datasets/thai-ipa", "/datasets/thai-text"):
                p = _Path(d)
                if p.exists():
                    shutil.rmtree(p)
                    log.log(f"[{stage}] force: wiped {p}")
            datasets_volume.commit()
        log.log(f"[{stage}] starting fetch_data({task})")
        try:
            result = fetch_data.remote(task)
            _mark_done(stage, extra=result if isinstance(result, dict) else {})
            checkpoints_volume.commit()
            datasets_volume.commit()
            summary["stages"][stage] = result
            log.log(f"[{stage}] done: {result}")
        except Exception as e:
            mark_stage_failed(status_root, _stage_key(stage), str(e))
            checkpoints_volume.commit()
            log.log(f"[{stage}] FAILED: {e}")
            raise

    # ---- 2. pretrain ---------------------------------------------------
    stage = "pretrain"
    if skip_pretrain or _done(stage) or (pretrain_best.is_file() and not force):
        log.log(f"[{stage}] skipped (best exists={pretrain_best.is_file()})")
        summary["stages"][stage] = {"skipped": True, "best": str(pretrain_best)}
    else:
        if force:
            import shutil
            pretrain_run = _Path("/checkpoints") / pretrain_task / "run-001"
            if pretrain_run.exists():
                shutil.rmtree(pretrain_run)
                log.log(f"[{stage}] force: wiped {pretrain_run}")
            checkpoints_volume.commit()
        log.log(f"[{stage}] starting pretrain({pretrain_task})")
        try:
            result = pretrain.remote(pretrain_task)
            # Reload volume view to see pretrain's writes (best.pt).
            checkpoints_volume.reload()
            _mark_done(stage, extra=result if isinstance(result, dict) else {})
            checkpoints_volume.commit()
            summary["stages"][stage] = result
            log.log(f"[{stage}] done: {result}")
        except Exception as e:
            mark_stage_failed(status_root, _stage_key(stage), str(e))
            checkpoints_volume.commit()
            log.log(f"[{stage}] FAILED: {e}")
            raise

    # ---- 3. supervised train -------------------------------------------
    stage = "train"
    pretrain_ckpt = _resolve_pretrain_ckpt()
    init_from = str(pretrain_ckpt) if pretrain_ckpt else None
    if skip_train or _done(stage) or (train_best.is_file() and not force):
        log.log(f"[{stage}] skipped (best exists={train_best.is_file()})")
        summary["stages"][stage] = {"skipped": True, "best": str(train_best)}
    else:
        if force:
            import shutil
            train_run = _Path("/checkpoints") / task / "run-001"
            if train_run.exists():
                shutil.rmtree(train_run)
                log.log(f"[{stage}] force: wiped {train_run}")
            checkpoints_volume.commit()
        if pretrain_ckpt is None:
            log.log(f"[{stage}] WARNING: no pretrain checkpoint — training from scratch")
            init_from = None
        else:
            log.log(f"[{stage}] using pretrain checkpoint: {pretrain_ckpt}")
        log.log(f"[{stage}] starting train({task}) init_from={init_from}")
        try:
            result = train.remote(task, init_from_pretrain=init_from)
            checkpoints_volume.reload()
            _mark_done(stage, extra=result if isinstance(result, dict) else {})
            checkpoints_volume.commit()
            summary["stages"][stage] = result
            log.log(f"[{stage}] done: {result}")
        except Exception as e:
            mark_stage_failed(status_root, _stage_key(stage), str(e))
            checkpoints_volume.commit()
            log.log(f"[{stage}] FAILED: {e}")
            raise

    # ---- 4. export ONNX fp32 (NO quantization) -------------------------
    stage = "export"
    # Fallback to latest checkpoint if best.pt missing.
    train_ckpt = train_best
    if not train_ckpt.is_file():
        train_run_dir = _Path("/checkpoints") / task / "run-001"
        if train_run_dir.is_dir():
            import re as _re
            def _epoch_num(p: _Path) -> int:
                m = _re.search(r"checkpoint-epoch-(\d+)\.pt$", p.name)
                return int(m.group(1)) if m else -1
            epoch_ckpts = [p for p in train_run_dir.glob("checkpoint-epoch-*.pt") if _epoch_num(p) >= 0]
            if epoch_ckpts:
                train_ckpt = sorted(epoch_ckpts, key=_epoch_num)[-1]
                log.log(f"[{stage}] best.pt missing — using {train_ckpt.name}")
    if skip_export or _done(stage) or (onnx_fp32.is_file() and not force):
        log.log(f"[{stage}] skipped (artifacts exist)")
        summary["stages"][stage] = {"skipped": True, "onnx": str(onnx_fp32)}
    else:
        if force:
            import shutil
            models_dir = _Path("/models") / task
            if models_dir.exists():
                shutil.rmtree(models_dir)
                log.log(f"[{stage}] force: wiped {models_dir}")
            models_volume.commit()
        if not train_ckpt.is_file():
            raise FileNotFoundError(f"train checkpoint missing at {train_ckpt} — cannot export")
        log.log(f"[{stage}] starting export_onnx (fp32 only) from {train_ckpt}")
        try:
            onnx_result = export_onnx.remote(task, version, checkpoint=str(train_ckpt))
            _mark_done(stage, extra=onnx_result)
            checkpoints_volume.commit()
            models_volume.commit()
            summary["stages"][stage] = onnx_result
            log.log(f"[{stage}] done: {onnx_result}")
        except Exception as e:
            mark_stage_failed(status_root, _stage_key(stage), str(e))
            checkpoints_volume.commit()
            log.log(f"[{stage}] FAILED: {e}")
            raise

    # ---- 5. evaluate PER -----------------------------------------------
    stage = "evaluate"
    # Use the same fallback as export: best.pt or latest checkpoint.
    eval_ckpt = train_ckpt if train_ckpt.is_file() else train_best
    if _done(stage):
        log.log(f"[{stage}] skipped")
        summary["stages"][stage] = {"skipped": True}
    else:
        log.log(f"[{stage}] starting evaluate({task}) from {eval_ckpt}")
        try:
            eval_result = evaluate.remote(task, checkpoint=str(eval_ckpt), beam_width=4)
            _mark_done(stage, extra=eval_result if isinstance(eval_result, dict) else {})
            checkpoints_volume.commit()
            summary["stages"][stage] = eval_result
            log.log(f"[{stage}] done: {eval_result}")
        except Exception as e:
            mark_stage_failed(status_root, _stage_key(stage), str(e))
            checkpoints_volume.commit()
            log.log(f"[{stage}] FAILED: {e}")
            # Evaluation failure is non-fatal — pipeline still succeeded.

    log.log(f"PIPELINE COMPLETE: {summary}")
    log.close()
    checkpoints_volume.commit()
    return summary


@app.local_entrypoint()
def sota_pipeline(
    task: str = "secryst_thai_ipa",
    version: str = "v0.1.0",
    skip_fetch: bool = False,
    skip_pretrain: bool = False,
    skip_train: bool = False,
    skip_export: bool = False,
    force: bool = False,
):
    """Fire-and-forget full Thai→IPA SOTA pipeline.

    Usage (disconnect-safe):

        modal run --detach modal_app.py::sota_pipeline

    Optional flags:
        modal run --detach modal_app.py::sota_pipeline --force
        modal run --detach modal_app.py::sota_pipeline --skip-pretrain

    Monitor via:
        modal app list
        python scripts/status.py
        modal volume ls secryst-checkpoints /checkpoints
    """
    print(f"Submitting SOTA pipeline: task={task} version={version}")
    print("  (orchestrator runs fully on Modal — safe to disconnect after submit)")
    result = run_sota_pipeline.remote(
        task=task,
        version=version,
        skip_fetch=skip_fetch,
        skip_pretrain=skip_pretrain,
        skip_train=skip_train,
        skip_export=skip_export,
        force=force,
    )
    print(f"Pipeline result: {result}")
    return result
