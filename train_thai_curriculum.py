"""Thai G2P curriculum training: continue from best umt5 checkpoint with
difficulty-ordered data + replay anchoring (HEBATRON pattern).

Strategy:
1. Rank training examples by difficulty (length + rare-char density)
2. Phase 1: train on easy 1/3 + replay 10% random
3. Phase 2: train on easy+medium 2/3 + replay 10% easy
4. Phase 3: train on all data + replay 10% easy+medium
5. Continue from best checkpoint (seed 789, PER 3.24%)
6. Target: <3% PER

Usage:
    modal run train_thai_curriculum.py
"""

from __future__ import annotations

import json
import random
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
        "sentencepiece",
        "protobuf",
        "accelerate",
        "numpy>=1.26,<3",
        "tqdm>=4.66",
        "pyyaml>=6.0",
    )
)

app = modal.App(name=f"{APP_NAME}-curriculum", image=image)


def score_difficulty(src: str, tgt: str) -> float:
    """Higher = harder. Length + rare-char density."""
    src_len = len(src)
    tgt_len = len(tgt)
    # rare = non-Thai chars in source (digits, latin, rare symbols)
    rare_chars = sum(1 for c in src if not ("฀" <= c <= "๿" or c == " "))
    rare_density = rare_chars / max(1, src_len)
    return src_len + tgt_len * 0.5 + rare_density * 20.0


def split_by_difficulty(examples: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    scored = [(score_difficulty(ex["src"], ex["tgt"]), ex) for ex in examples]
    scored.sort(key=lambda x: x[0])
    n = len(scored)
    easy = [ex for _, ex in scored[: n // 3]]
    medium = [ex for _, ex in scored[n // 3 : 2 * n // 3]]
    hard = [ex for _, ex in scored[2 * n // 3 :]]
    return easy, medium, hard


@app.function(
    gpu="A100",
    timeout=8 * 60 * 60,
    volumes={"/ckpts": checkpoints_volume, "/datasets": datasets_volume},
)
def train_curriculum(phase: int = 1, replay_frac: float = 0.1) -> dict:
    """Run one phase of curriculum training. phase=1, 2, or 3."""
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

    print(f"[curriculum] starting phase {phase}", flush=True)

    train_path = Path("/datasets/thai-ipa/train.jsonl")
    val_path = Path("/datasets/thai-ipa/val.jsonl")

    all_train = []
    for line in train_path.read_text(encoding="utf-8").splitlines():
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
            all_train.append({"src": src, "tgt": tgt})
    print(f"[curriculum] total train: {len(all_train)}", flush=True)

    easy, medium, hard = split_by_difficulty(all_train)
    print(f"[curriculum] easy={len(easy)}, medium={len(medium)}, hard={len(hard)}", flush=True)

    rng = random.Random(42)

    if phase == 1:
        phase_data = list(easy)
        replay_pool = all_train
    elif phase == 2:
        phase_data = list(easy) + list(medium)
        replay_pool = easy
    else:  # phase 3
        phase_data = list(all_train)
        replay_pool = easy + medium

    # Add replay samples
    n_replay = int(len(phase_data) * replay_frac)
    if n_replay > 0 and replay_pool:
        replay = rng.sample(replay_pool, min(n_replay, len(replay_pool)))
        phase_data.extend(replay)
    rng.shuffle(phase_data)
    print(f"[curriculum] phase {phase}: {len(phase_data)} examples", flush=True)

    # Write phase data to a temp file
    phase_path = Path(f"/tmp/phase_{phase}.jsonl")
    with phase_path.open("w", encoding="utf-8") as f:
        for ex in phase_data:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # Load model from previous phase or starting checkpoint
    if phase == 1:
        ckpt = START_CKPT
    else:
        ckpt = f"/ckpts/secryst_thai_ipa_curriculum/phase-{phase-1}/best"
        if not Path(ckpt).is_dir():
            ckpt = START_CKPT
            print(f"[curriculum] WARNING: previous phase not found, using start ckpt", flush=True)
    print(f"[curriculum] loading {ckpt}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModelForSeq2SeqLM.from_pretrained(ckpt).to("cuda")

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

    train_ds = JsonlDataset(str(phase_path), tokenizer)
    val_ds = JsonlDataset(str(val_path), tokenizer)
    print(f"[curriculum] train={len(train_ds)}, val={len(val_ds)}", flush=True)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, label_pad_token_id=-100)

    out_dir = Path(f"/ckpts/secryst_thai_ipa_curriculum/phase-{phase}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Phase-specific LR: warm restart with cosine
    base_lr = 3e-5 if phase == 1 else 2e-5 if phase == 2 else 1e-5
    epochs = 4 if phase == 1 else 3 if phase == 2 else 2

    args = Seq2SeqTrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=base_lr,
        warmup_steps=50,
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
    return {"phase": phase, "best": str(best_path), "n_train": len(train_ds)}


@app.function(
    gpu="A10G",
    timeout=30 * 60,
    volumes={"/ckpts": checkpoints_volume, "/datasets": datasets_volume},
)
def evaluate_curriculum(phase: int = 1) -> dict:
    """Evaluate curriculum phase checkpoint."""
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    checkpoints_volume.reload()
    datasets_volume.reload()

    ckpt = Path(f"/ckpts/secryst_thai_ipa_curriculum/phase-{phase}/best")
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
        "phase": phase,
        "per": per,
        "wer": 1.0 - (exact / max(1, n)),
        "exact_match": exact / max(1, n),
        "n_examples": n,
    }
    print(f"=== Phase {phase} PER: {per:.4f} ===", flush=True)
    return result


@app.local_entrypoint()
def main():
    """Run all 3 curriculum phases sequentially + evaluate each."""
    for phase in (1, 2, 3):
        train_result = train_curriculum.remote(phase=phase)
        print(f"Phase {phase} train: {json.dumps(train_result, indent=2)}")

        eval_result = evaluate_curriculum.remote(phase=phase)
        print(f"Phase {phase} eval: {json.dumps(eval_result, indent=2)}")
