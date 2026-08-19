# REPORT: transformers 5.15 save corrupts UMT5 checkpoints — scaleup600k is at risk

2026-08-19. From the interscript/ml-models distillation campaign
(rababa TODO.runtime-arch WO07). Action needed BEFORE the scaleup600k
run finishes and saves.

## TL;DR

Every secryst-saved umt5 checkpoint on `secryst-checkpoints` is
**unusable at inference** — including `secryst_thai_ipa_thai_combined_mixed`
(the published 2.32% PER teacher). The published numbers were live-eval
results; they cannot be reproduced from the saved artifacts. The same
save path will corrupt **scaleup600k** when it saves `best/`, unless
the training image pins `transformers==5.14.1` (or the save is patched)
before the run completes.

## Evidence (probes, 2026-08-18/19)

1. `secryst_thai_ipa_thai_combined_mixed/run-001/best` and
   `secryst_thai_ipa_umt5_s789/run-001/best` under their own training
   environment (unpinned image, transformers 5.15.0, fp32, beam-4,
   correct held-out test file): degenerate outputs — single tokens or
   character soup. PER 218–231% vs published 2.32/3.24%.
2. The saved `lm_head.weight` diverges from `shared.weight` by
   max|Δ| ≈ 94.8 (norms 205 vs 4377). The trained output layer is not
   in the file; what is there is untrained. Loading untied (as the
   file dictates) → garbage. Force-tying lm_head := shared → repetition
   loops (umt5 is architecturally untied — tying is wrong too).
3. `generation_config.json` in all these dirs says
   `"transformers_version": "5.15.0"` — the save-time version.
   The B-K/umt5-thai-g2p-v2-0.5k HUB artifact (saved by its author
   under an older stack) loads and generates perfectly under BOTH 5.14.1
   and 5.15. The corruption is in the 5.15 save path for untied-umt5,
   not in loading.
4. Second hazard: transformers 5.x `batch_decode` inserts spurious
   spaces between sentencepiece pieces. Any eval of these models must
   decode via `"".join(convert_ids_to_tokens(ids))` (skip pad/eos/bos).
   With that fix the hub base reproduces exact matches; the secryst
   artifacts stay dead (see 2).

## Additional finding: the epitran corpus is tone-less

`thai-ipa/augmented_epitran.jsonl` (50K) on `secryst-datasets` is
**0% tone marks / 100% space-separated**. Kaikki (`train.jsonl`) is
100% tones. Training on the current mix destroyed tone production in
our recovery attempt (stage-1 Kaikki-only: 8.7% PER, tone-faithful;
adding the epitran file: 50.7% PER, tones gone). The published 2.32%
recipe cannot be reproduced with the file as it stands — epitran
labels must be regenerated WITH tone marks (and unspaced).

## What to do

1. **Pin `transformers==5.14.1` in the scaleup600k training image NOW**
   (if the run is mid-flight and cannot be pinned: prepare to re-run —
   a corrupted 367K-corpus artifact is worth nothing). 5.14.1 saves are
   verified clean: all interscript-ml ByT5 exports (khm/urd/heb) were
   produced under 5.14.1 and load correctly everywhere.
   Alternatively patch the save: set `config.tie_word_embeddings=False`
   and save the true lm_head state.
2. Regenerate epitran augmentation with tone marks (the
   augment_thai_epitran.py output currently on the volume is
   tone-less).
3. After a verified eval lands, update rababa
   `docs/DISTILL-SOURCE-PROMPT.md` section 4 — the ml-models
   distillation pipeline will pick the new teacher up from there.

## Reference artifacts (ml-models, all verified)

- Recovery harness: `src/gpu/modal_teacher_thai.py` (+ run logs).
- Healthy 5.14.1-saved umt5: `secryst_thai_ipa_teacher_recovery/run-003/stage1`
  (Kaikki-only, 8.7% PER, tone-faithful, exact-match verified on CPU).
- Shipped despite all this: `tha-g2p-base-1.0` — ByT5-base student
  distilled from the B-K hub teacher (4.43% PER on the fixed harness),
  gate +4.76pp ≤ +5pp. It will be superseded by a scaleup600k-tier
  teacher once one verifies.
