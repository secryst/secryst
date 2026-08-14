# 09 — Active learning (Thai)

Same as Arabic 11. Surface hard examples, target data collection.

## Tasks

### 9.1 Hardness miner
- `src/secryst/training/active_learning.py` (new, port from rababa pattern)
- `mine_hard_examples(model, loader, top_k=1000)`

### 9.2 Pattern analysis for Thai
- Cluster hard examples by tone pattern, syllable count, consonant class
- Report actionable patterns

## Acceptance
- [ ] `mine_hard_examples` returns examples with mean loss > 2x val mean
- [ ] Pattern report identifies ≥ 3 actionable patterns

## Files
- `src/secryst/training/active_learning.py` (new)
