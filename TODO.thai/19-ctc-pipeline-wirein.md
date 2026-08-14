# 19 — Wire CTC into secryst SOTA pipeline

## Problem

`CTCModel` is built and spec'd (TODO 17, 2026-08-08) but not wired into
secryst's `run_sota_pipeline`. The pipeline still only knows how to run
`modern_seq2seq` arch.

## Fix

Add CTC dispatch to secryst pipeline:

1. `secryst/modal_app.py:train` — if `cfg.model.arch == "ctc"`, dispatch to
   `train_ctc` instead of `train_supervised`.
2. `secryst/modal_app.py:evaluate` — if `cfg.model.arch == "ctc"`, use
   CTC greedy decode instead of beam search.
3. `secryst/modal_app.py:export_onnx` — handle CTCModel's forward signature.

## Files

- `secryst/modal_app.py` — add arch dispatch in train/eval/export.
- `secryst/configs/secryst_thai_ipa_ctc.yaml` — already exists.
- `tests/test_ctc_pipeline.py` (NEW) — spec: arch=ctc dispatches correctly.

## Acceptance

- `modal run modal_app.py::sota_pipeline --task secryst_thai_ipa_ctc` works.
- CTC training reaches PER < 0.50 (vs seq2seq's PER > 1.0).
- All existing secryst specs pass.
