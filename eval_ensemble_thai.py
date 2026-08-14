"""Evaluate Thai G2P ensemble: 4 umt5 seed models with medoid voting.

Strategy: for each test example, generate N candidates (one per seed model),
then pick the candidate that minimizes total edit distance to the others
(the "medoid"). This is the standard CCV (Consensus Candidate Voting)
method for seq2seq ensembles.

Checkpoints (on Modal volume):
  /ckpts/secryst_thai_ipa_umt5_v3/run-001/best  (seed 42)
  /ckpts/secryst_thai_ipa_umt5_s123/run-001/best (seed 123)
  /ckpts/secryst_thai_ipa_umt5_s456/run-001/best (seed 456)
  /ckpts/secryst_thai_ipa_umt5_s789/run-001/best (seed 789)

Usage:
    modal run eval_ensemble_thai.py
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

APP_NAME = "secryst"
checkpoints_volume = modal.Volume.from_name(f"{APP_NAME}-checkpoints", create_if_missing=True)
datasets_volume = modal.Volume.from_name(f"{APP_NAME}-datasets", create_if_missing=True)

CKPTS = [
    "/ckpts/secryst_thai_ipa_umt5_v3/run-001/best",
    "/ckpts/secryst_thai_ipa_umt5_s123/run-001/best",
    "/ckpts/secryst_thai_ipa_umt5_s456/run-001/best",
    "/ckpts/secryst_thai_ipa_umt5_s789/run-001/best",
]

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("build-essential", "git", "curl")
    .pip_install(
        "torch>=2.4,<3",
        "transformers>=4.46",
        "sentencepiece>=0.2",
        "numpy>=1.26,<3",
        "tqdm>=4.66",
        "pyyaml>=6.0",
    )
)

app = modal.App(name=f"{APP_NAME}-ensemble", image=image)


def _edit_distance(a: list, b: list) -> int:
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


@app.function(
    gpu="A10G",
    timeout=90 * 60,
    volumes={"/ckpts": checkpoints_volume, "/datasets": datasets_volume},
)
def evaluate_ensemble(num_beams: int = 4) -> dict:
    """Load all 4 seed models, run ensemble voting on test set."""
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    checkpoints_volume.reload()
    datasets_volume.reload()

    device = torch.device("cuda")

    models = []
    tokenizer = None
    for ckpt_path in CKPTS:
        p = Path(ckpt_path)
        if not p.is_dir():
            print(f"[ensemble] WARNING: {p} not found, skipping", flush=True)
            continue
        print(f"[ensemble] loading {p}", flush=True)
        tok = AutoTokenizer.from_pretrained(str(p))
        model = AutoModelForSeq2SeqLM.from_pretrained(str(p)).to(device)
        model.eval()
        if tokenizer is None:
            tokenizer = tok
        models.append(model)
    if len(models) < 2:
        return {"error": f"Only {len(models)} models loaded, need >=2"}

    print(f"[ensemble] {len(models)} models loaded", flush=True)

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
    print(f"[ensemble] test examples: {len(examples)}", flush=True)

    total_ed_ensemble = 0
    total_gold_len = 0
    exact_match_ensemble = 0
    total_n = 0

    per_model_stats = [{"ed": 0, "gold_len": 0, "exact": 0} for _ in models]

    batch_size = 32

    with torch.no_grad():
        for i in range(0, len(examples), batch_size):
            batch = examples[i : i + batch_size]
            inputs = [src for src, _ in batch]
            encoded = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)

            all_preds = []
            for m_idx, model in enumerate(models):
                out = model.generate(
                    **encoded,
                    max_new_tokens=256,
                    num_beams=num_beams,
                )
                preds = tokenizer.batch_decode(out, skip_special_tokens=True)
                all_preds.append([p.strip() for p in preds])

            for j, (src, gold) in enumerate(batch):
                gold_tokens = gold.strip().split()
                len_gold = max(1, len(gold_tokens))

                candidate_outputs = [all_preds[m_idx][j] for m_idx in range(len(models))]
                candidate_tokens = [c.split() for c in candidate_outputs]

                per_model_ed = []
                for m_idx, toks in enumerate(candidate_tokens):
                    ed = _edit_distance(toks, gold_tokens)
                    per_model_stats[m_idx]["ed"] += ed
                    per_model_stats[m_idx]["gold_len"] += len_gold
                    if ed == 0:
                        per_model_stats[m_idx]["exact"] += 1
                    per_model_ed.append(ed)

                # Medoid: pick the candidate with min total edit distance to others
                best_idx = 0
                best_cost = float("inf")
                for a_idx in range(len(candidate_tokens)):
                    cost = 0
                    for b_idx in range(len(candidate_tokens)):
                        if a_idx == b_idx:
                            continue
                        cost += _edit_distance(candidate_tokens[a_idx], candidate_tokens[b_idx])
                    if cost < best_cost:
                        best_cost = cost
                        best_idx = a_idx

                ensemble_ed = per_model_ed[best_idx]
                total_ed_ensemble += ensemble_ed
                total_gold_len += len_gold
                if ensemble_ed == 0:
                    exact_match_ensemble += 1
                total_n += 1

            if i % 320 == 0 and i > 0:
                per = total_ed_ensemble / max(1, total_gold_len)
                print(f"  [{i}/{len(examples)}] ensemble PER={per:.4f}", flush=True)

    ensemble_per = total_ed_ensemble / max(1, total_gold_len)
    ensemble_wer = 1.0 - (exact_match_ensemble / max(1, total_n))

    result = {
        "ensemble_per": ensemble_per,
        "ensemble_wer": ensemble_wer,
        "ensemble_exact_match": exact_match_ensemble / max(1, total_n),
        "n_models": len(models),
        "n_examples": total_n,
        "per_model": [
            {
                "seed_label": Path(CKPTS[i]).parents[1].name,
                "per": s["ed"] / max(1, s["gold_len"]),
                "wer": 1.0 - (s["exact"] / max(1, total_n)),
                "exact_match": s["exact"] / max(1, total_n),
            }
            for i, s in enumerate(per_model_stats)
        ],
    }

    print(f"\n=== Ensemble PER: {ensemble_per:.4f} (WER={ensemble_wer:.4f}) ===", flush=True)
    for pm in result["per_model"]:
        print(f"  {pm['seed_label']}: PER={pm['per']:.4f}", flush=True)

    return result


@app.local_entrypoint()
def main(num_beams: int = 4):
    result = evaluate_ensemble.remote(num_beams=num_beams)
    print(json.dumps(result, indent=2, ensure_ascii=False))
