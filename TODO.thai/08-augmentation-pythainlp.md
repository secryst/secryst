# 08 — PyThaiNLP augmentation (Thai silver)

## Why
PyThaiNLP's g2p produces rule-based IPA. When the model's prediction
DISAGREES with PyThaiNLP on a word, that's a high-information signal —
either the model is wrong (and should learn PyThaiNLP's answer), or
PyThaiNLP is wrong (and we should flag for review).

## Tasks

### 8.1 PyThaiNLP disagreement miner
- For each train example, run PyThaiNLP g2p
- Compare with gold IPA
- Bucket: agree / disagree (PyThaiNLP wrong) / disagree (gold wrong)

### 8.2 Augmentation policy
- "Agree" examples: weight 1.0
- "Disagree (PyThaiNLP wrong)": weight 1.5 (model should learn gold over rule)
- "Disagree (gold wrong)": drop (low-quality gold)

### 8.3 Integration
- Pre-compute bucket per train example, store as JSON
- DataLoader samples with weights

## Acceptance
- [ ] At least 70% of train examples agree with PyThaiNLP
- [ ] PER improves by ≥ 1% after augmentation

## Files
- `src/secryst/augment_pythainlp.py` (new)
- `scripts/bucket_thai_examples.py` (new)
