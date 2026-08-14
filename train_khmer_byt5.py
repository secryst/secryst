"""Khmer word transliteration with ByT5-small.

Data: secryst/data-khmer-translit (GitHub) — 17,911 word pairs in two
parallel single-column CSVs (data_kh.csv Khmer script, data_rom.csv
romanization), originally used to train the legacy seq2seq net
(net-500-epochs.pth). We retrain with the modern ByT5 recipe and report
exact match + CER on a held-out test split.

Usage:
    modal run train_khmer_byt5.py
"""

from __future__ import annotations

import json
import random
import urllib.request
from pathlib import Path

import modal

APP_NAME = "secryst"
checkpoints_volume = modal.Volume.from_name(f"{APP_NAME}-checkpoints", create_if_missing=True)
datasets_volume = modal.Volume.from_name(f"{APP_NAME}-datasets", create_if_missing=True)

DATA_URLS = {
    "data_kh.csv": "https://raw.githubusercontent.com/secryst/data-khmer-translit/master/data_kh.csv",
    "data_rom.csv": "https://raw.githubusercontent.com/secryst/data-khmer-translit/master/data_rom.csv",
}
RUN_NAME = "khmer_byt5/run-001"
SEED = 42

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "transformers==4.46.3",
        "accelerate>=1.1.0",
        "editdistance",
    )
)

app = modal.App("secryst-khmer-byt5", image=image)


def edit_distance(a: str, b: str) -> int:
    import editdistance

    return editdistance.eval(a, b)


@app.function(
    volumes={"/ckpts": checkpoints_volume, "/datasets": datasets_volume},
    gpu="A10G",
    timeout=2 * 60 * 60,
)
def train() -> None:
    import torch
    from torch.utils.data import Dataset
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        EarlyStoppingCallback,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
    )

    raw_dir = Path("/datasets/khmer-translit")
    raw_dir.mkdir(parents=True, exist_ok=True)
    for fname, url in DATA_URLS.items():
        dest = raw_dir / fname
        if not dest.exists():
            print(f"[data] fetching {fname}", flush=True)
            urllib.request.urlretrieve(url, dest)

    kh_lines = [l.strip() for l in (raw_dir / "data_kh.csv").read_text(encoding="utf-8").splitlines()]
    rom_lines = [l.strip() for l in (raw_dir / "data_rom.csv").read_text(encoding="utf-8").splitlines()]
    assert len(kh_lines) == len(rom_lines), f"parallel mismatch: {len(kh_lines)} vs {len(rom_lines)}"

    pairs, conflicts = {}, 0
    for src, tgt in zip(kh_lines, rom_lines):
        if not src or not tgt:
            continue
        if src in pairs and pairs[src] != tgt:
            conflicts += 1
            continue
        pairs[src] = tgt
    items = sorted(pairs.items())
    print(f"[data] {len(kh_lines)} raw lines, {len(items)} unique pairs, {conflicts} conflicting dupes", flush=True)

    rng = random.Random(SEED)
    rng.shuffle(items)
    n_test = max(1, int(len(items) * 0.05))
    n_val = max(1, int(len(items) * 0.05))
    splits = {
        "test": items[:n_test],
        "val": items[n_test : n_test + n_val],
        "train": items[n_test + n_val :],
    }
    for name, rows in splits.items():
        path = raw_dir / f"{name}.jsonl"
        path.write_text(
            "".join(json.dumps({"src": s, "tgt": t}, ensure_ascii=False) + "\n" for s, t in rows),
            encoding="utf-8",
        )
        print(f"[data] {name}: {len(rows)} -> {path}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained("google/byt5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/byt5-small")

    class PairDataset(Dataset):
        def __init__(self, path: Path) -> None:
            self.rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, idx: int) -> dict:
            row = self.rows[idx]
            inputs = tokenizer(row["src"], truncation=True, max_length=128)
            labels = tokenizer(row["tgt"], truncation=True, max_length=128)
            inputs["labels"] = labels["input_ids"]
            return inputs

    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, label_pad_token_id=-100)

    args = Seq2SeqTrainingArguments(
        output_dir="/ckpts/" + RUN_NAME,
        num_train_epochs=30,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        learning_rate=3e-4,
        warmup_steps=200,
        weight_decay=0.01,
        max_grad_norm=1.0,
        label_smoothing_factor=0.1,
        seed=SEED,
        save_strategy="epoch",
        eval_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=50,
        report_to=[],
        predict_with_generate=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=PairDataset(raw_dir / "train.jsonl"),
        eval_dataset=PairDataset(raw_dir / "val.jsonl"),
        data_collator=collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )
    trainer.train()

    best_dir = Path("/ckpts") / RUN_NAME / "best"
    best_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))
    checkpoints_volume.commit()

    model = trainer.model.to("cuda")
    model.eval()
    test_rows = [json.loads(l) for l in (raw_dir / "test.jsonl").read_text(encoding="utf-8").splitlines()]
    for beam in (1, 4):
        em, cer, total_chars, n = 0, 0, 0, 0
        with torch.no_grad():
            for i in range(0, len(test_rows), 64):
                batch = test_rows[i : i + 64]
                enc = tokenizer([r["src"] for r in batch], return_tensors="pt", padding=True, truncation=True, max_length=128).to("cuda")
                out = model.generate(**enc, max_new_tokens=128, num_beams=beam)
                preds = tokenizer.batch_decode(out, skip_special_tokens=True)
                for r, p in zip(batch, preds):
                    p = p.strip()
                    if p == r["tgt"]:
                        em += 1
                    cer += edit_distance(p, r["tgt"])
                    total_chars += max(1, len(r["tgt"]))
                    n += 1
        print(
            f"[eval] beam={beam} n={n} EM={em / n:.4%} CER={cer / total_chars:.4%}",
            flush=True,
        )
    checkpoints_volume.commit()


if __name__ == "__main__":
    with app.run():
        train()
