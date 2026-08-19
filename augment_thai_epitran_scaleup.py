"""Scale up Thai epitran augmentation: 50K -> 600K sentences, both shards.

Differences vs augment_thai_epitran.py (which produced the original 50K,
kept untouched on the volume):
- downloads BOTH parquet shards of thai_wikipedia_clean_20230101
- ProcessPoolExecutor (8 workers) so 600K sentences fit in hours
- writes to a NEW file: /datasets/thai-ipa/augmented_epitran_600k.jsonl

Usage:
    modal run --detach augment_thai_epitran_scaleup.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import modal

APP_NAME = "secryst"
datasets_volume = modal.Volume.from_name(f"{APP_NAME}-datasets", create_if_missing=True)

SHARDS = [
    "https://huggingface.co/datasets/pythainlp/thai_wikipedia_clean_20230101/resolve/main/data/train-00000-of-00002-97bd8a2b732f13e6.parquet",
    "https://huggingface.co/datasets/pythainlp/thai_wikipedia_clean_20230101/resolve/main/data/train-00001-of-00002-c1f07f6eafaa703a.parquet",
]

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("build-essential", "git", "curl")
    .pip_install(
        "epitran>=1.25",
        "pythainlp>=5.0",
        "pyarrow",
        "numpy>=1.26,<3",
        "tqdm>=4.66",
    )
)

app = modal.App(name=f"{APP_NAME}-augment-scaleup", image=image)

_epi = None


def _init_worker():
    global _epi
    import epitran

    _epi = epitran.Epitran("tha-Thai")


def _is_thai_char(c: str) -> bool:
    return 0x0E00 <= ord(c) <= 0x0E7F


def _process(text: str) -> dict | None:
    from pythainlp.tokenize import word_tokenize

    text = text.strip()
    if not text or len(text) < 20 or len(text) > 500:
        return None
    try:
        words = word_tokenize(text, engine="newmm")
    except Exception:
        return None
    if len(words) < 3 or len(words) > 60:
        return None

    ipa_words = []
    n_valid = 0
    for w in words:
        w = w.strip()
        if not w:
            continue
        if not any(_is_thai_char(c) for c in w):
            ipa_words.append(w)
            n_valid += 1
            continue
        try:
            ipa = _epi.transliterate(w)
            if ipa:
                ipa_words.append(ipa)
                n_valid += 1
        except Exception:
            continue

    if n_valid < 0.8 * len(words) or not ipa_words:
        return None
    return {"src": "".join(words), "tgt": " ".join(ipa_words)}


@app.function(
    cpu=8,
    timeout=3 * 60 * 60,
    volumes={"/datasets": datasets_volume},
)
def augment(max_sentences: int = 600_000) -> dict:
    import pyarrow.parquet as pq
    import urllib.request
    from concurrent.futures import ProcessPoolExecutor

    datasets_volume.reload()

    all_text: list[str] = []
    for url in SHARDS:
        parquet_path = Path("/tmp") / url.rsplit("/", 1)[1]
        if not parquet_path.exists():
            print(f"[download] {url.rsplit('/', 1)[1]}", flush=True)
            urllib.request.urlretrieve(url, parquet_path)
        texts = pq.read_table(parquet_path).column("text").to_pylist()
        print(f"[download] {parquet_path.name}: {len(texts)} rows", flush=True)
        all_text.extend(texts)
    print(f"[data] total rows: {len(all_text)}", flush=True)

    random.Random(42).shuffle(all_text)

    output_path = Path("/datasets/thai-ipa/augmented_epitran_600k.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    skipped = 0
    CHUNK = 4000
    with output_path.open("w", encoding="utf-8") as out_f:
        with ProcessPoolExecutor(max_workers=8, initializer=_init_worker) as ex:
            for i in range(0, len(all_text), CHUNK):
                for row in ex.map(_process, all_text[i : i + CHUNK], chunksize=64):
                    if row is None:
                        skipped += 1
                        continue
                    out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    count += 1
                    if count >= max_sentences:
                        break
                if count % 50_000 < CHUNK:
                    print(f"  [augment] {count}/{max_sentences} (skipped {skipped})", flush=True)
                if count >= max_sentences:
                    break

    datasets_volume.commit()
    print(f"\n=== Augmented: {count} (skipped {skipped}) -> {output_path} ===", flush=True)
    return {"n_augmented": count, "n_skipped": skipped, "output": str(output_path)}


@app.local_entrypoint()
def main(max_sentences: int = 600_000):
    print(json.dumps(augment.remote(max_sentences=max_sentences), indent=2))
