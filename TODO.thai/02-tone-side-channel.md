# 02 — Tone-class side channel for Thai

## Why
Thai has 5 tones (mid, low, falling, high, rising). The tone of a
syllable depends on:
- Initial consonant class (low/mid/high)
- Vowel length
- Tone mark (if any)
- Syllable type (open/closed/dead/live)

The model can learn these implicitly, but exposing them as input
features gives free phonological signal without arch changes.

## Tasks

### 2.1 Thai feature extractor (`src/secryst/features.py`)
- `compute_thai_features(text: str) -> list[FeatureVector]` per char.
- FeatureVector: `{consonant_class: int, vowel_length: int, has_tone_mark: bool}`.

### 2.2 ModernSeq2Seq: feature embedding
- Add `feature_vocab_size` constructor param.
- New `nn.Embedding(feature_vocab_size, dim)` added to char embedding.
- When feature_vocab_size=0 (default), no change — backward compatible.

### 2.3 Wire into datasets + collate
- Dataset returns `(src_ids, tgt_ids, feature_ids)`.
- Collate pads feature_ids alongside src.

### 2.4 Config flag
- `cfg.model.features: ["consonant_class", "vowel_length"]` (default empty list).

## Acceptance
- [ ] `compute_thai_features` returns expected classes for test inputs.
- [ ] Training with features reduces PER vs baseline by ≥ 1%.

## Files
- `src/secryst/features.py` (new)
- `src/secryst/models/seq2seq.py` (add feature embedding)
- `src/secryst/datasets.py` (extract features)
- `src/secryst/training/collate.py` (collate features)
- `tests/test_features.py` (new)
