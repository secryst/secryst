"""Evaluate Gregniuki/thai-g2p-byt5-finetuned-v2 on our Thai test set.

This is a ByT5-small model fine-tuned for Thai G2P. We want to know if it
beats our umt5-based model (3.24% PER) on our test set.

Usage:
    modal run eval_gregniuki_thai.py
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

APP_NAME = "secryst"
datasets_volume = modal.Volume.from_name(f"{APP_NAME}-datasets", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.4,<3",
        "transformers>=4.46",
        "sentencepiece",
        "numpy>=1.26,<3",
        "tqdm>=4.66",
    )
)

app = modal.App(name=f"{APP_NAME}-eval-greg", image=image)


@app.function(
    gpu="A10G",
    timeout=30 * 60,
    volumes={"/datasets": datasets_volume},
    secrets=[modal.Secret.from_name("huggingface")],
)
def evaluate() -> dict:
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    datasets_volume.reload()

    model_name = "Gregniuki/thai-g2p-byt5-finetuned-v2"
    print(f"Loading {model_name}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to("cuda")
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
    print(f"Test examples: {len(examples)}", flush=True)

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

            if i < 6:
                for j in range(min(2, len(batch))):
                    print(f"\n--- Example {i+j} ---", flush=True)
                    print(f"  input: {batch[j][0][:80]}", flush=True)
                    print(f"  pred:  {preds[j][:120]}", flush=True)
                    print(f"  gold:  {batch[j][1][:120]}", flush=True)

            if i % 240 == 0 and i > 0:
                per = total_ed / max(1, total_gold)
                print(f"  [{i}/{len(examples)}] PER={per:.4f}", flush=True)

    per = total_ed / max(1, total_gold)
    result = {
        "model": model_name,
        "per": per,
        "wer": 1.0 - (exact / max(1, n)),
        "exact_match": exact / max(1, n),
        "n_examples": n,
    }
    print(f"\n=== Gregniuki ByT5 PER: {per:.4f} ===", flush=True)
    return result


@app.local_entrypoint()
def main():
    result = evaluate.remote()
    print(json.dumps(result, indent=2))
