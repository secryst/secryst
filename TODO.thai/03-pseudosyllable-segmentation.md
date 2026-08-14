# 03 — Pseudo-syllable segmentation + cross-attention fix

## Why
v0.1.0 secryst Thai→IPA exhibits mode collapse: the decoder outputs
the same IPA sequence (`kaːn˧.kʰa˨˩.naʔ˨˩`) for every input. The
cross-attention gradient signal is too weak relative to the decoder's
self-attention, so the model learns to ignore the encoder and produce
a fixed "safe" output.

Two fixes packaged together because they address the same root cause
(decoder not conditioning on input):

## Tasks

### 3.1 Cross-attention LR boost
- The cross-attention projections (`q_cross`, `kv_cross`, `out_cross`)
  get the same LR as everything else. For from-scratch training, they
  need a higher LR to escape the "ignore encoder" attractor.
- Implementation: parameter group with separate LR in MuonAdamWHybrid.
- Config: `train.cross_attn_lr_mult: 3.0` (default 1.0).

### 3.2 Scheduled sampling
- During training, mix teacher-forced and own-prediction steps.
- Schedule: start with 100% teacher forcing, linearly decay to 50%
  own-predictions by end of training.
- Implementation: in the training loop, with probability `p_use_own`,
  replace `tgt_in[t]` with `argmax(logits[t-1])` for t > 0.

### 3.3 Encoder output regularization
- Add a small dropout on the encoder output (memory) before cross-attn.
- Prevents the decoder from overfitting to specific encoder patterns.
- Config: `model.memory_dropout: 0.1` (default 0).

### 3.4 Multi-task syllable segmentation head
- Thai has no inter-word spaces. A side head that predicts syllable
  boundaries gives the model extra signal about input structure.
- Mirrors rababa's seg head architecture.
- Labels derived from IPA output (`.` syllable boundary → boundary in input).

## Acceptance
- [ ] Different inputs produce different outputs (no mode collapse).
- [ ] PER drops below 0.50 (from current 1.10).
- [ ] Beam search top-1 predictions vary across test examples.

## Files
- `src/secryst/models/seq2seq.py` (add memory_dropout)
- `src/secryst/training/supervised.py` (scheduled sampling + cross-attn LR)
- `src/secryst/training/optim.py` (per-param-group LR routing)
- `configs/secryst_thai_ipa.yaml` (new config flags)
