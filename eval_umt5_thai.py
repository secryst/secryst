"""Evaluate B-K/umt5-thai-g2p-v2-0.5k on our Thai test set.

This is a ready-made Thai G2P model on HuggingFace that achieves CER ≈ 3.7%.
If it works on our test set, we get SOTA Thai G2P with zero training.
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

APP_NAME = "secryst"
checkpoints_volume = modal.Volume.from_name(f"{APP_NAME}-checkpoints", create_if_missing=True)
datasets_volume = modal.Volume.from_name(f"{APP_NAME}-datasets", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("build-essential", "git", "curl")
    .pip_install(
        "torch>=2.4,<3",
        "transformers>=4.46",
        "huggingface_hub>=0.26",
        "sentencepiece>=0.2",
        "numpy>=1.26,<3",
        "tqdm>=4.66",
        "pyyaml>=6.0",
    )
    .add_local_dir("src", "/opt/secryst/src", copy=True)
    .add_local_dir("configs", "/opt/secryst/configs", copy=True)
    .workdir("/opt/secryst")
    .env({"PYTHONPATH": "/opt/secryst/src"})
)

app = modal.App(name=APP_NAME, image=image)


@app.function(
    gpu="A10G",
    timeout=30 * 60,
    volumes={"/datasets": datasets_volume},
    secrets=[modal.Secret.from_name("huggingface")],
)
def evaluate_umt5_thai() -> dict:
    """Load umt5-thai-g2p and evaluate PER on our Thai test set."""
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    model_name = "B-K/umt5-thai-g2p-v2-0.5k"
    print(f"Loading {model_name}...", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to("cuda")
    model.eval()
    print(f"Model loaded.", flush=True)

    # Smoke test
    test_input = "สวัสดี"
    inputs = tokenizer(test_input, return_tensors="pt").to("cuda")
    output = model.generate(**inputs, max_new_tokens=128)
    result = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"\n=== Smoke test ===", flush=True)
    print(f"Input:  {test_input}", flush=True)
    print(f"Output: {result}", flush=True)

    # Load test data
    datasets_volume.reload()
    import json as _json
    test_path = Path("/datasets/thai-ipa/test.jsonl")
    examples = []
    for line in test_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = _json.loads(line)
        except _json.JSONDecodeError:
            continue
        src = (row.get("src") or "").strip()
        tgt = (row.get("tgt") or "").strip()
        if src and tgt:
            examples.append((src, tgt))

    print(f"\nTest examples: {len(examples)}", flush=True)

    def _edit_distance(a, b):
        m, n = len(a), len(b)
        if m == 0: return n
        if n == 0: return m
        prev = list(range(n + 1))
        for i in range(1, m + 1):
            curr = [i] + [0] * n
            for j in range(1, n + 1):
                cost = 0 if a[i-1] == b[j-1] else 1
                curr[j] = min(curr[j-1]+1, prev[j]+1, prev[j-1]+cost)
            prev = curr
        return prev[n]

    total_ed = 0
    total_gold_len = 0
    exact_match = 0
    total_n = 0
    batch_size = 32

    with torch.no_grad():
        for i in range(0, len(examples), batch_size):
            batch = examples[i:i + batch_size]
            inputs = [src for src, _ in batch]
            encoded = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True, max_length=128).to("cuda")
            output = model.generate(**encoded, max_new_tokens=128, num_beams=1)
            predictions = tokenizer.batch_decode(output, skip_special_tokens=True)

            for j, (_, gold) in enumerate(batch):
                pred = predictions[j].strip().split()
                gold_tokens = gold.strip().split()
                ed = _edit_distance(pred, gold_tokens)
                total_ed += ed
                total_gold_len += max(1, len(gold_tokens))
                if ed == 0:
                    exact_match += 1
                total_n += 1

            if i < 3:
                for j in range(min(3, len(batch))):
                    src, gold = batch[j]
                    print(f"\n--- Example {i+j} ---", flush=True)
                    print(f"  input: {src}", flush=True)
                    print(f"  pred:  {predictions[j]}", flush=True)
                    print(f"  gold:  {gold}", flush=True)

            if i % 200 == 0 and i > 0:
                per = total_ed / max(1, total_gold_len)
                print(f"  [{i}/{len(examples)}] PER={per:.4f}", flush=True)

    per = total_ed / max(1, total_gold_len)
    result = {
        "model": model_name,
        "per": per,
        "wer": 1.0 - (exact_match / max(1, total_n)),
        "exact_match": exact_match / max(1, total_n),
        "n_examples": total_n,
    }
    print(f"\n=== umt5-thai-g2p PER: {per:.4f} ({total_n} examples) ===", flush=True)
    return result


@app.local_entrypoint()
def main():
    result = evaluate_umt5_thai.remote()
    print(json.dumps(result, indent=2))
