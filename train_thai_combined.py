"""Retrain Thai G2P with combined Kaikki + epitran-augmented data.

Approach: continued training from s789 checkpoint (3.24% PER baseline)
on combined ~60K examples (9.7K Kaikki original + 50K epitran augmented).

If format mismatch (Kaikki vs epitran) hurts, we can fall back to
training only on epitran data (sacrificing test-set format match but
gaining 5x more data).

Usage:
    modal run train_thai_combined.py
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

APP_NAME = "secryst"
checkpoints_volume = modal.Volume.from_name(f"{APP_NAME}-checkpoints", create_if_missing=True)
datasets_volume = modal.Volume.from_name(f"{APP_NAME}-datasets", create_if_missing=True)

START_CKPT = "/ckpts/secryst_thai_ipa_umt5_s789/run-001/best"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("build-essential", "git", "curl")
    .pip_install(
        "torch>=2.4,<3",
        "transformers>=4.46",
        "accelerate>=1.1.0",
        "sentencepiece",
        "numpy>=1.26,<3",
        "tqdm>=4.66",
        "pyyaml>=6.0",
    )
)

app = modal.App(name=f"{APP_NAME}-combined", image=image)


@app.function(
    gpu="A100",
    timeout=4 * 60 * 60,
    volumes={"/ckpts": checkpoints_volume, "/datasets": datasets_volume},
)
def train(use_epitran_only: bool = False) -> dict:
    """Train on combined or epitran-only data."""
    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForSeq2SeqLM,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        DataCollatorForSeq2Seq,
    )
    from torch.utils.data import Dataset

    checkpoints_volume.reload()
    datasets_volume.reload()

    # Load existing Kaikki training data
    kanssa_path = Path("/datasets/thai-ipa/train.jsonl")
    epitran_path = Path("/datasets/thai-ipa/augmented_epitran.jsonl")

    def load_jsonl(p):
        out = []
        if not p.is_file():
            return out
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            src = (r.get("src") or "").strip()
            tgt = (r.get("tgt") or "").strip()
            if src and tgt:
                out.append({"src": src, "tgt": tgt})
        return out

    kanssa = load_jsonl(kanssa_path)
    epitran = load_jsonl(epitran_path)
    print(f"Kaikki: {len(kanssa)}, Epitran: {len(epitran)}", flush=True)

    if use_epitran_only:
        train_data = epitran
        out_name = "thai_combined_epitran_only"
    else:
        train_data = kanssa + epitran
        out_name = "thai_combined_mixed"

    print(f"Training data: {len(train_data)} ({out_name})", flush=True)

    val_path = Path("/datasets/thai-ipa/val.jsonl")
    val_data = load_jsonl(val_path)

    # Write to temp file
    train_path = Path(f"/tmp/{out_name}_train.jsonl")
    with train_path.open("w", encoding="utf-8") as f:
        for ex in train_data:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    val_path_tmp = Path(f"/tmp/{out_name}_val.jsonl")
    with val_path_tmp.open("w", encoding="utf-8") as f:
        for ex in val_data:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Loading {START_CKPT}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(START_CKPT)
    model = AutoModelForSeq2SeqLM.from_pretrained(START_CKPT).to("cuda")

    class JsonlDataset(Dataset):
        def __init__(self, path, tok, max_len=256):
            self.examples = []
            for ln in Path(path).read_text(encoding="utf-8").splitlines():
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    r = json.loads(ln)
                except Exception:
                    continue
                s = (r.get("src") or "").strip()
                t = (r.get("tgt") or "").strip()
                if s and t:
                    self.examples.append((s, t))
            self.tok = tok
            self.max_len = max_len

        def __len__(self):
            return len(self.examples)

        def __getitem__(self, idx):
            s, t = self.examples[idx]
            mi = self.tok(s, truncation=True, max_length=self.max_len)
            lab = self.tok(t, truncation=True, max_length=self.max_len)
            mi["labels"] = lab["input_ids"]
            return mi

    train_ds = JsonlDataset(str(train_path), tokenizer)
    val_ds = JsonlDataset(str(val_path_tmp), tokenizer)
    print(f"train={len(train_ds)}, val={len(val_ds)}", flush=True)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, label_pad_token_id=-100)

    out_dir = Path(f"/ckpts/secryst_thai_ipa_{out_name}/run-001")
    out_dir.mkdir(parents=True, exist_ok=True)

    args = Seq2SeqTrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=2,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        warmup_steps=100,
        weight_decay=0.01,
        max_grad_norm=1.0,
        label_smoothing_factor=0.0,
        seed=42,
        save_strategy="epoch",
        eval_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        predict_with_generate=False,
        logging_steps=50,
        report_to=[],
        dataloader_num_workers=2,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

    best_path = out_dir / "best"
    trainer.save_model(str(best_path))
    tokenizer.save_pretrained(str(best_path))

    checkpoints_volume.commit()
    return {"best": str(best_path), "n_train": len(train_ds)}


@app.function(
    gpu="A10G",
    timeout=30 * 60,
    volumes={"/ckpts": checkpoints_volume, "/datasets": datasets_volume},
)
def evaluate(use_epitran_only: bool = False) -> dict:
    """Evaluate on test set."""
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    checkpoints_volume.reload()
    datasets_volume.reload()

    out_name = "thai_combined_epitran_only" if use_epitran_only else "thai_combined_mixed"
    ckpt = Path(f"/ckpts/secryst_thai_ipa_{out_name}/run-001/best")
    if not ckpt.is_dir():
        return {"error": f"{ckpt} not found"}

    tokenizer = AutoTokenizer.from_pretrained(str(ckpt))
    model = AutoModelForSeq2SeqLM.from_pretrained(str(ckpt)).to("cuda")
    model.eval()

    test_path = Path("/datasets/thai-ipa/test.jsonl")
    examples = []
    for line in test_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        src = (row.get("src") or "").strip()
        tgt = (row.get("tgt") or "").strip()
        if src and tgt:
            examples.append((src, tgt))

    def _ed(a, b):
        m, n = len(a), len(b)
        if m == 0:
            return n
        if n == 0:
            return m
        prev = list(range(n + 1))
        for i in range(1, m + 1):
            curr = [i] + [0] * n
            for j in range(1, n + 1):
                cost = 0 if a[i - 1] == b[j - 1] else 1
                curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
            prev = curr
        return prev[n]

    total_ed = 0
    total_gold = 0
    exact = 0
    n = 0
    batch_size = 32

    with torch.no_grad():
        for i in range(0, len(examples), batch_size):
            batch = examples[i : i + batch_size]
            inputs = [src for src, _ in batch]
            enc = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True, max_length=256).to("cuda")
            out = model.generate(**enc, max_new_tokens=256, num_beams=4)
            preds = tokenizer.batch_decode(out, skip_special_tokens=True)
            for j, (_, gold) in enumerate(batch):
                pred = preds[j].strip().split()
                gt = gold.strip().split()
                e = _ed(pred, gt)
                total_ed += e
                total_gold += max(1, len(gt))
                if e == 0:
                    exact += 1
                n += 1
            if i % 240 == 0 and i > 0:
                print(f"  [{i}/{len(examples)}] PER={total_ed/max(1,total_gold):.4f}", flush=True)

    per = total_ed / max(1, total_gold)
    result = {
        "ckpt": str(ckpt),
        "per": per,
        "wer": 1.0 - (exact / max(1, n)),
        "exact_match": exact / max(1, n),
        "n_examples": n,
    }
    print(f"=== Combined ({out_name}) PER: {per:.4f} ===", flush=True)
    return result


@app.local_entrypoint()
def main(use_epitran_only: bool = False):
    train_result = train.remote(use_epitran_only=use_epitran_only)
    print(f"Train: {json.dumps(train_result, indent=2)}")

    eval_result = evaluate.remote(use_epitran_only=use_epitran_only)
    print(f"Evaluate: {json.dumps(eval_result, indent=2)}")
