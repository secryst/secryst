"""Thai G2P scale-up: continue from thai_combined_mixed (2.32% PER) on
Kaikki + 600K epitran-augmented sentences (12x the previous corpus).

Init: /ckpts/secryst_thai_ipa_thai_combined_mixed/run-001/best
Data: /datasets/thai-ipa/train.jsonl + augmented_epitran_600k.jsonl
Eval: fixed test.jsonl with beam-4 PER (same harness as before).

Usage:
    modal run --detach train_thai_scaleup.py
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

APP_NAME = "secryst"
checkpoints_volume = modal.Volume.from_name(f"{APP_NAME}-checkpoints", create_if_missing=True)
datasets_volume = modal.Volume.from_name(f"{APP_NAME}-datasets", create_if_missing=True)

START_CKPT = "/ckpts/secryst_thai_ipa_thai_combined_mixed/run-001/best"
RUN = "secryst_thai_ipa_scaleup600k/run-001"

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

app = modal.App(name=f"{APP_NAME}-scaleup", image=image)


def _load_jsonl(p: Path) -> list[dict]:
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


@app.function(
    gpu="A100",
    timeout=24 * 60 * 60,
    volumes={"/ckpts": checkpoints_volume, "/datasets": datasets_volume},
)
def train() -> dict:
    import torch
    from torch.utils.data import Dataset
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        TrainerCallback,
    )

    checkpoints_volume.reload()
    datasets_volume.reload()

    kanssa = _load_jsonl(Path("/datasets/thai-ipa/train.jsonl"))
    epitran = _load_jsonl(Path("/datasets/thai-ipa/augmented_epitran_600k.jsonl"))
    val_data = _load_jsonl(Path("/datasets/thai-ipa/val.jsonl"))
    print(f"Kaikki: {len(kanssa)}, Epitran-600k: {len(epitran)}", flush=True)
    if len(epitran) < 300_000:
        return {"error": "augmented_epitran_600k.jsonl missing or incomplete; run augment first"}
    train_data = kanssa + epitran

    done_marker = Path("/ckpts") / RUN / "EVAL_DONE"
    if done_marker.exists():
        return {"run": RUN, "status": "already-done"}

    tokenizer = AutoTokenizer.from_pretrained(START_CKPT)
    model = AutoModelForSeq2SeqLM.from_pretrained(START_CKPT).to("cuda")

    class JsonlDataset(Dataset):
        def __init__(self, rows: list[dict], tok, max_len: int = 256) -> None:
            self.rows = rows
            self.tok = tok
            self.max_len = max_len

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, idx: int) -> dict:
            s, t = self.rows[idx]["src"], self.rows[idx]["tgt"]
            mi = self.tok(s, truncation=True, max_length=self.max_len)
            lab = self.tok(t, truncation=True, max_length=self.max_len)
            mi["labels"] = lab["input_ids"]
            return mi

    class VolumeCommitCallback(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            try:
                checkpoints_volume.commit()
                print(f"[volume] committed at step {state.global_step}", flush=True)
            except Exception as e:
                print(f"[volume] commit failed at step {state.global_step}: {e}", flush=True)

    out_dir = Path("/ckpts") / RUN
    out_dir.mkdir(parents=True, exist_ok=True)

    args = Seq2SeqTrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=1,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        warmup_steps=200,
        weight_decay=0.01,
        max_grad_norm=1.0,
        label_smoothing_factor=0.0,
        seed=42,
        save_strategy="steps",
        save_steps=2000,
        save_total_limit=1,
        eval_strategy="epoch",
        bf16=True,
        predict_with_generate=False,
        logging_steps=100,
        report_to=[],
        dataloader_num_workers=4,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=JsonlDataset(train_data, tokenizer),
        eval_dataset=JsonlDataset(val_data, tokenizer),
        processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, label_pad_token_id=-100),
        callbacks=[VolumeCommitCallback()],
    )

    import glob

    latest = sorted(glob.glob(f"{out_dir}/checkpoint-*"), key=lambda p: int(p.rsplit("-", 1)[1]))
    resume = latest[-1] if latest else None
    print(f"[resume] {resume}", flush=True)
    trainer.train(resume_from_checkpoint=resume)

    best = out_dir / "best"
    best.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(best))
    tokenizer.save_pretrained(str(best))
    checkpoints_volume.commit()
    return {"run": RUN, "n_train": len(train_data), "best": str(best)}


@app.function(
    gpu="A10G",
    timeout=60 * 60,
    volumes={"/ckpts": checkpoints_volume, "/datasets": datasets_volume},
)
def evaluate() -> dict:
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    checkpoints_volume.reload()
    datasets_volume.reload()

    ckpt = Path("/ckpts") / RUN / "best"
    tokenizer = AutoTokenizer.from_pretrained(str(ckpt))
    model = AutoModelForSeq2SeqLM.from_pretrained(str(ckpt)).to("cuda")
    model.eval()

    test = _load_jsonl(Path("/datasets/thai-ipa/test.jsonl"))
    print(f"test: {len(test)}", flush=True)

    def _ed(a: list[str], b: list[str]) -> int:
        m, n = len(a), len(b)
        prev = list(range(n + 1))
        for i in range(1, m + 1):
            curr = [i] + [0] * n
            for j in range(1, n + 1):
                cost = 0 if a[i - 1] == b[j - 1] else 1
                curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
            prev = curr
        return prev[n]

    total_ed = total_gold = exact = n = 0
    with torch.no_grad():
        for i in range(0, len(test), 32):
            batch = test[i : i + 32]
            enc = tokenizer(
                [r["src"] for r in batch], return_tensors="pt", padding=True,
                truncation=True, max_length=256,
            ).to("cuda")
            out = model.generate(**enc, max_new_tokens=256, num_beams=4)
            preds = tokenizer.batch_decode(out, skip_special_tokens=True)
            for r, p in zip(batch, preds):
                pred = p.strip().split()
                gt = r["tgt"].strip().split()
                e = _ed(pred, gt)
                total_ed += e
                total_gold += max(1, len(gt))
                exact += e == 0
                n += 1
            if i % 320 == 0 and i > 0:
                print(f"  [{i}/{len(test)}] PER={total_ed/max(1,total_gold):.4f}", flush=True)

    per = total_ed / max(1, total_gold)
    print(f"=== scaleup600k PER: {per:.4f} (WER {1 - exact/max(1,n):.4f}) ===", flush=True)
    (Path("/ckpts") / RUN / "metrics.json").write_text(
        json.dumps({"per": per, "wer": 1 - exact / max(1, n), "n": n}), encoding="utf-8")
    checkpoints_volume.commit()
    return {"per": per, "wer": 1 - exact / max(1, n), "n": n}


@app.local_entrypoint()
def main():
    print(json.dumps(train.remote(), indent=2))
    print(json.dumps(evaluate.remote(), indent=2))
