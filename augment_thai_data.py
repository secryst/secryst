"""Augment Thai G2P training data with pythainlp thai2rom (neural model).

Source: pythainlp/thai_wikipedia_clean_20230101 (718K Thai text rows)
Phonemizer: pythainlp.transliterate.romanize(engine='thai2rom')
  - thai2rom is a PyTorch attention model trained on Thai G2P data
  - Outputs clean Latin text matching our test format
  - Not a general LLM — narrow purpose, no hallucination of phonology
  - Still a learned model but well-validated for Thai

Strategy:
1. Sample ~50K sentences from Thai Wikipedia
2. Word-tokenize with pythainlp.tokenize.word_tokenize(engine='newmm')
3. Romanize each word with thai2rom
4. Filter: keep only sentences where >80% of words romanize cleanly
5. Upload to secryst volume
6. Retrain umt5 on combined ~60K (9.7K Kaikki + 50K augmented) pairs

Usage:
    modal run augment_thai_data.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import modal

APP_NAME = "secryst"
datasets_volume = modal.Volume.from_name(f"{APP_NAME}-datasets", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("build-essential", "git", "curl")
    .pip_install(
        "torch>=2.4,<3",
        "transformers>=4.46",
        "pythainlp>=5.0",
        "sentencepiece",
        "numpy>=1.26,<3",
        "tqdm>=4.66",
        "pyarrow",
        "huggingface_hub",
    )
)

app = modal.App(name=f"{APP_NAME}-augment-thai", image=image)


@app.function(
    gpu="A10G",
    timeout=60 * 60,
    volumes={"/datasets": datasets_volume},
)
def augment(max_sentences: int = 50000) -> dict:
    """Run pythainlp thai2rom on Wikipedia text to generate new training pairs."""
    import pyarrow.parquet as pq
    from pythainlp.transliterate import romanize
    from pythainlp.tokenize import word_tokenize

    datasets_volume.reload()

    import urllib.request
    parquet_path = "/tmp/thai_wiki.parquet"
    if not Path(parquet_path).exists():
        url = "https://huggingface.co/datasets/pythainlp/thai_wikipedia_clean_20230101/resolve/main/data/train-00000-of-00002-97bd8a2b732f13e6.parquet"
        print("Downloading Thai Wikipedia...", flush=True)
        urllib.request.urlretrieve(url, parquet_path)

    print("Reading parquet...", flush=True)
    table = pq.read_table(parquet_path)
    all_text = table.column("text").to_pylist()
    print(f"Total Wikipedia rows: {len(all_text)}", flush=True)

    rng = random.Random(42)
    rng.shuffle(all_text)

    output_path = Path("/datasets/thai-ipa/augmented_train.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Warming up thai2rom model...", flush=True)
    _ = romanize("ทดสอบ", engine="thai2rom")

    def is_thai_char(c: str) -> bool:
        code = ord(c)
        return 0x0E00 <= code <= 0x0E7F

    count = 0
    skipped = 0
    with output_path.open("w", encoding="utf-8") as out_f:
        for text in all_text:
            if count >= max_sentences:
                break

            text = text.strip()
            if not text or len(text) < 20 or len(text) > 500:
                skipped += 1
                continue

            try:
                words = word_tokenize(text, engine="newmm")
            except Exception:
                skipped += 1
                continue

            if len(words) < 3 or len(words) > 60:
                skipped += 1
                continue

            roman_words = []
            n_valid = 0
            for w in words:
                w = w.strip()
                if not w:
                    continue
                if not any(is_thai_char(c) for c in w):
                    roman_words.append(w)
                    n_valid += 1
                    continue
                try:
                    r = romanize(w, engine="thai2rom")
                    if r:
                        roman_words.append(r)
                        n_valid += 1
                except Exception:
                    continue

            if n_valid < 0.8 * len(words):
                skipped += 1
                continue

            if not roman_words:
                skipped += 1
                continue

            src = "".join(words)
            tgt = " ".join(roman_words)
            out_f.write(json.dumps({"src": src, "tgt": tgt}, ensure_ascii=False) + "\n")
            count += 1

            if count % 5000 == 0:
                print(f"  [augment] {count}/{max_sentences} (skipped {skipped})", flush=True)

    datasets_volume.commit()
    print(f"\n=== Augmented: {count} (skipped {skipped}) -> {output_path} ===", flush=True)
    return {"n_augmented": count, "n_skipped": skipped, "output": str(output_path)}


@app.local_entrypoint()
def main(max_sentences: int = 50000):
    result = augment.remote(max_sentences=max_sentences)
    print(json.dumps(result, indent=2))
