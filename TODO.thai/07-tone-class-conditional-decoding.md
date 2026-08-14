# 07 — Tone-class conditional decoding (Thai)

## Why
Thai has 5 lexical tones (mid, low, falling, high, rising). The tone of
a syllable depends on consonant class + vowel length + tone mark. At
inference, we can constrain beam search to only consider tone-valid paths.

## Tasks

### 7.1 Tone-class detector
- `src/secryst/tone.py` (new)
- `detect_tone_class(thai_syllable: str) -> str` returns one of
  {mid, low, falling, high, rising, unknown}

### 7.2 Tone-constrained beam search
- Extend `decoding/beam.py::beam_search` with optional `tone_valid: set[int]`
- Skip beam candidates whose predicted tone isn't in the valid set

### 7.3 Wire into evaluate
- `evaluate.py` calls `detect_tone_class` per source word, builds valid set

## Acceptance
- [ ] Tone errors eliminated (every prediction has a valid tone for the source)
- [ ] PER drops by ≥ 1% vs unconstrained beam search

## Files
- `src/secryst/tone.py` (new)
- `src/secryst/decoding/beam.py` (extend beam_search signature)
- `src/secryst/evaluate.py` (call tone detector)
- `tests/test_tone.py` (new)
