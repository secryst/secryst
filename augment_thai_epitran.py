"""Augment Thai G2P training data with epitran (deterministic rule-based).

Source: pythainlp/thai_wikipedia_clean_20230101 (718K Thai text rows)
Phonemizer: epitran (Thai) — fully deterministic, no neural model, no LLM
  - epitran is a rule-based G2P library covering 130+ languages
  - Outputs IPA with long vowel marks (ː) and aspirated consonants
  - Format differs from our Kaikki test format but is internally consistent

Strategy:
1. Sample ~50K sentences from Thai Wikipedia
2. Word-tokenize with pythainlp.word_tokenize
3. Epitran each Thai word, keep non-Thai tokens as-is
4. Output format: concatenated epitran output (NO tones, NO syllable split)
5. Upload to secryst volume

Note: Our test set uses Kaikki format (space-separated IPA with syllable
boundaries). Training on epitran format won't perfectly match test format,
but adds data diversity that should help generalization.

Usage:
    modal run augment_thai_epitran.py
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
        "epitran>=1.25",
        "pythainlp>=5.0",
        "pyarrow",
        "huggingface_hub",
        "numpy>=1.26,<3",
        "tqdm>=4.66",
    )
)

app = modal.App(name=f"{APP_NAME}-augment-epitran", image=image)


@app.function(
    cpu=4,
    timeout=30 * 60,
    volumes={"/datasets": datasets_volume},
)
def augment(max_sentences: int = 50000) -> dict:
    """Run epitran on Wikipedia text."""
    import epitran
    from pythainlp.tokenize import word_tokenize
    import pyarrow.parquet as pq

    datasets_volume.reload()

    epi = epitran.Epitran("tha-Thai")

    # Download Wikipedia
    import urllib.request
    parquet_path = "/tmp/thai_wiki.parquet"
    if not Path(parquet_path).exists():
        url = "https://huggingface.co/datasets/pythainlp/thai_wikipedia_clean_20230101/resolve/main/data/train-00000-of-00002-97bd8a2b732f13e6.parquet"
        print("Downloading Thai Wikipedia...", flush=True)
        urllib.request.urlretrieve(url, parquet_path)

    table = pq.read_table(parquet_path)
    all_text = table.column("text").to_pylist()
    print(f"Total Wikipedia rows: {len(all_text)}", flush=True)

    rng = random.Random(42)
    rng.shuffle(all_text)

    output_path = Path("/datasets/thai-ipa/augmented_epitran.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

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

            ipa_words = []
            n_valid = 0
            for w in words:
                w = w.strip()
                if not w:
                    continue
                if not any(is_thai_char(c) for c in w):
                    ipa_words.append(w)
                    n_valid += 1
                    continue
                try:
                    ipa = epi.transliterate(w)
                    if ipa:
                        ipa_words.append(ipa)
                        n_valid += 1
                except Exception:
                    continue

            if n_valid < 0.8 * len(words):
                skipped += 1
                continue

            if not ipa_words:
                skipped += 1
                continue

            src = "".join(words)
            tgt = " ".join(ipa_words)
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
