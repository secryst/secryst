"""Expand Thai G2P training data by combining multiple HuggingFace datasets.

Sources:
1. pythainlp/thai-g2p-v4-dataset (PyThaiNLP, has train/test/val parquets)
2. B-K/thai-w2p-ipa (word-to-phoneme with IPA, CSV)
3. Our existing Kaikki data (already on volume)

Combines all into one expanded corpus for umt5 fine-tuning.
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

datasets_volume = modal.Volume.from_name("secryst-datasets", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("build-essential", "git")
    .pip_install(
        "torch>=2.4,<3", "numpy>=1.26,<3", "transformers>=4.46",
        "huggingface_hub>=0.26", "datasets>=2.20", "pyarrow>=15.0", "pandas>=2.0",
        "omegaconf>=2.3,<3", "tqdm>=4.66", "pyyaml>=6.0",
    )
    .add_local_dir("src", "/opt/secryst/src", copy=True)
    .add_local_dir("configs", "/opt/secryst/configs", copy=True)
    .workdir("/opt/secryst")
    .env({"PYTHONPATH": "/opt/secryst/src"})
)

app = modal.App(name="secryst", image=image)


@app.function(
    cpu=4,
    timeout=30 * 60,
    volumes={"/datasets": datasets_volume},
    secrets=[modal.Secret.from_name("huggingface")],
)
def expand_thai_data() -> dict:
    """Download and combine Thai G2P datasets."""
    import pandas as pd
    from pathlib import Path as _P

    datasets_volume.reload()
    all_pairs: set[tuple[str, str]] = set()

    # 1. Existing Kaikki data
    kaikki_path = _P("/datasets/thai-ipa/train.jsonl")
    if kaikki_path.is_file():
        count = 0
        for line in kaikki_path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
                src = row.get("src", "").strip()
                tgt = row.get("tgt", "").strip()
                if src and tgt:
                    all_pairs.add((src, tgt))
                    count += 1
            except Exception:
                continue
        print(f"[1] Kaikki: {count} pairs", flush=True)

    # 2. pythainlp/thai-g2p-v4-dataset
    try:
        from datasets import load_dataset
        ds = load_dataset("pythainlp/thai-g2p-v4-dataset", split="train")
        count = 0
        for ex in ds:
            # Check column names
            src = str(ex.get("grapheme", ex.get("text", ex.get("src", "")))).strip()
            tgt = str(ex.get("phoneme", ex.get("ipa", ex.get("tgt", "")))).strip()
            if src and tgt:
                all_pairs.add((src, tgt))
                count += 1
        print(f"[2] pythainlp/thai-g2p-v4: +{count} pairs", flush=True)
    except Exception as e:
        print(f"[2] pythainlp error: {e}", flush=True)

    # 3. B-K/thai-w2p-ipa
    try:
        from datasets import load_dataset
        ds2 = load_dataset("B-K/thai-w2p-ipa", split="train")
        count = 0
        for ex in ds2:
            src = str(ex.get("word", ex.get("grapheme", ex.get("src", "")))).strip()
            tgt = str(ex.get("ipa", ex.get("phoneme", ex.get("tgt", "")))).strip()
            if src and tgt:
                all_pairs.add((src, tgt))
                count += 1
        print(f"[3] B-K/thai-w2p-ipa: +{count} pairs", flush=True)
    except Exception as e:
        print(f"[3] B-K error: {e}", flush=True)

    # 4. B-K/thai-g2p (sentence-level)
    try:
        ds3 = load_dataset("B-K/thai-g2p", split="train")
        count = 0
        for ex in ds3:
            src = str(ex.get("grapheme", ex.get("text", ex.get("src", "")))).strip()
            tgt = str(ex.get("phoneme", ex.get("ipa", ex.get("tgt", "")))).strip()
            if src and tgt:
                all_pairs.add((src, tgt))
                count += 1
        print(f"[4] B-K/thai-g2p: +{count} pairs", flush=True)
    except Exception as e:
        print(f"[4] B-K/thai-g2p error: {e}", flush=True)

    total = len(all_pairs)
    print(f"\n=== Total unique pairs: {total} ===", flush=True)

    # Split and write
    import random
    pairs_list = sorted(all_pairs)
    random.seed(42)
    random.shuffle(pairs_list)

    n_test = max(500, total // 20)
    n_val = max(500, total // 20)

    out_dir = _P("/datasets/thai-ipa-expanded")
    out_dir.mkdir(parents=True, exist_ok=True)

    for split, data in [
        ("test", pairs_list[:n_test]),
        ("val", pairs_list[n_test:n_test+n_val]),
        ("train", pairs_list[n_test+n_val:]),
    ]:
        out_path = out_dir / f"{split}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for src, tgt in data:
                f.write(json.dumps({"src": src, "tgt": tgt}) + "\n")
        print(f"  {split}: {len(data)} pairs → {out_path}", flush=True)

    datasets_volume.commit()
    return {"total": total, "train": len(pairs_list) - n_test - n_val}


@app.local_entrypoint()
def main():
    result = expand_thai_data.remote()
    print(json.dumps(result, indent=2, default=str))
