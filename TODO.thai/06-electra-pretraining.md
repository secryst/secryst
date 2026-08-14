# 06 — ELECTRA pretraining (Thai)

Same as rababa's ELECTRA TODO. Replaces MLM with RTD for ~2x sample-efficiency.

## Tasks

### 6.1 Port ELECTRA to secryst
- `src/secryst/training/electra.py` (port from rababa)
- Generator: half-width encoder; discriminator: full encoder.

### 6.2 Wire into modal_app.py
- `pretrain_method: mlm | electra` dispatch

### 6.3 Benchmark vs MLM
- Same compute budget, compare val loss + downstream PER

## Acceptance
- [ ] ELECTRA val loss < MLM val loss at same epoch
- [ ] ELECTRA-pretrained Thai PER ≤ MLM-pretrained PER

## Files
- `src/secryst/training/electra.py` (port from rababa)
- `configs/secryst_thai_ipa_pretrain.yaml` (add `pretrain_method: electra`)
