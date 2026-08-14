# 05 — Multi-seed ensemble (Thai)

Same recipe as Arabic 02 / Hebrew 03.

## Tasks

### 5.1 Launch 3-seed parallel train (after v0.5.0 mode-collapse fix lands)
- `python scripts/train_seeds.py --task secryst_thai_ipa --n-seeds 3`

### 5.2 Distill
- Load 3 teachers, distill into single student via `training/distill.py`
- Note: distill.py currently targets rababa — needs adaptation for secryst
  (multi-head vs single IPA head)

### 5.3 Re-export fp32 ONNX

## Acceptance
- [ ] 3 seeds train without mode collapse (each produces varied outputs)
- [ ] Distilled PER < best single-seed PER by ≥ 2%

## Files
- `src/secryst/training/distill.py` (port from rababa, adapt for seq2seq)
