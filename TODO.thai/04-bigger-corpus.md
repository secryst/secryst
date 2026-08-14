# 04 — Bigger corpus (Thai, PyThaiNLP silver)

## Why
Secryst Thai v0.1.0 trained on 9,663 Wiktionary pairs. SOTA G2P systems
typically use 50K+ pairs. PyThaiNLP's rule-based g2p can produce silver
labels for any Thai text, giving us effectively unlimited training data.

## Tasks

### 4.1 Install PyThaiNLP in Modal image
- Add `pythainlp>=4.0` to image pip_install

### 4.2 Silver-label hewiki-equivalent (Thai Wikipedia)
- Clone Thai Wikipedia dump
- Run PyThaiNLP g2p on each word
- Filter to words where g2p succeeds without warnings
- Output: `/datasets/thai-ipa-silver/{train,val,test}.jsonl`

### 4.3 Combine gold + silver with confidence weights
- Gold Wiktionary: weight 1.0
- Silver PyThaiNLP: weight 0.5 (lower confidence)
- Sample weighted in DataLoader

## Acceptance
- [ ] Combined corpus ≥ 50K Thai-IPA pairs
- [ ] PER improves vs v0.1.0 baseline (9K only)

## Files
- `modal_app.py` (extend fetch_data to build silver)
- `configs/secryst_thai_ipa_v0.5.0.yaml` (new, uses combined corpus)
