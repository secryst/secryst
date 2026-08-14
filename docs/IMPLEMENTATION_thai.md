# Thai → IPA G2P — SOTA Implementation Map

> **Purpose**: Trace every modern SOTA technique applied to the Thai G2P
> pipeline (secryst). Encoder-decoder architecture (unlike rababa which
> is encoder-only), but the modern K3/DS4 stack is shared.

## Models covered

| Model | Config | Role |
|-------|--------|------|
| `secryst_thai_ipa` | `secryst_thai_ipa.yaml` | Seq2seq student (v0.5.0: 6+6L/256d, ~25M params) |
| `secryst_thai_ipa_pretrain` | `secryst_thai_ipa_pretrain.yaml` | MLM pretrain for encoder |

## Architecture: encoder-decoder

Unlike rababa (encoder-only with linear classification heads), secryst
is a true seq2seq Transformer:

- **Encoder**: char-level Thai input → contextual hidden states.
- **Decoder**: token-level IPA output, autoregressive, with cross-attention
  to encoder memory.

The encoder uses the same modern stack as rababa (RoPE, SDPA, mHC,
AttnRes, RMSNorm, SwiGLU). The decoder extends with self-attention,
cross-attention, and a 4-stream mHC variant.

## SOTA techniques applied

### 1. RoPE + SDPA + mHC + AttnRes + RMSNorm + SwiGLU

Same as rababa. See `IMPLEMENTATION_arabic.md` for details.

### 2. 4-stream mHC (DS4)

- **Paper**: DeepSeek V4 (arXiv:2512.24880).
- **Code**: `src/secryst/models/modern.py:MHCN`.
- **Why 4 streams for decoder**: decoder layer has self-attn + cross-attn
  + FFN sublayers, so the mHC mixing matrix is 4×4 (residual + 3
  sublayers) rather than the encoder's 3×3 (residual + attn + FFN).

### 3. Cross-attention LR boost (v0.5.0 mode-collapse fix)

- **Rationale**: At v0.1.0, decoder ignored encoder memory and produced
  the same IPA sequence regardless of input (PER=1.0). Root cause:
  encoder memory gradient signal was weak relative to decoder self-attn.
- **Fix**: 3× LR multiplier on cross-attention weights.
- **Code**: `src/secryst/training/optim.py:cross_attn_lr_mult`.
- **Config**: `train.cross_attn_lr_mult: 3.0`.

### 4. Scheduled sampling (v0.5.0 exposure-bias fix)

- **Paper**: Bengio et al. 2015.
- **Rationale**: Teacher-forcing creates a train/test mismatch — at
  inference, the model's own predictions feed back as input. Scheduled
  sampling gradually replaces teacher inputs with model predictions.
- **Code**: `src/secryst/training/supervised.py:_scheduled_sample`.
- **Config**: `train.scheduled_sampling_final: 0.3` (anneal 0 → 0.3
  linearly over training).

### 5. Memory dropout (v0.5.0 regularizer)

- **Rationale**: Zeroes out encoder memory at random during training.
  Forces decoder to rely on cross-attention rather than memorize the
  input.
- **Code**: `src/secryst/models/seq2seq.py:ModernSeq2Seq.memory_drop`.
- **Config**: `model.memory_dropout: 0.1`.

### 6. Phonological feature side-channel

- **Code**: `src/secryst/features/thai.py:compute_thai_features`.
- **Features**: consonant_class (live/dead), has_tone_mark,
  syllable_initial.
- **Rationale**: Thai tones depend on consonant class + tone mark +
  syllable structure. Explicit features help the model learn tone
  rules faster.

### 7. Muon + QK-Clip

Same optimizer stack as rababa.

### 8. Curriculum learning

- **Code**: `src/secryst/training/curriculum.py`.
- **Difficulty signals**: syllable count, rare consonant density.

### 9. Active learning harness

- **Code**: `src/secryst/training/active_learning.py`.
- **Rationale**: Low-confidence predictions on unlabeled Thai text are
  flagged for human annotation. Prioritizes labeling budget on the
  examples that will most improve the model.

## Training pipeline

```
[ Wiktionary Thai-IPA pairs + custom lexicon ]
                ↓
        [ Cleaner + phonological features ]
                ↓
[ MLM pretrain on unlabeled Thai text (encoder only) ]
                ↓  (encoder checkpoint)
[ Supervised seq2seq training with v0.5.0 fixes ]
                ↓  (3× cross-attn LR + scheduled sampling + mem dropout)
        [ Tier-1 student ]
                ↓
[ Multi-seed ×3 → ensemble → distill ]
                ↓
[ Active learning round ×1-2 ]
                ↓
            [ Ship ]
```

## Mode-collapse diagnosis (v0.5.0 incident report)

**Symptom**: At v0.1.0, secryst PER=1.0 on Wiktionary test split.
Decoder produced the same IPA sequence regardless of input.

**Root cause analysis**:
1. Inspected attention maps: cross-attention weights were uniform
   (no information flowing from encoder).
2. Inspected gradients: cross-attention gradients were 100× smaller
   than self-attention gradients.
3. Hypothesis: standard LR + warmup caused cross-attention to lag
   behind self-attention; decoder learned a language-model prior
   before encoder memory could influence it.

**Fixes applied (v0.5.0)**:
1. `cross_attn_lr_mult: 3.0` — direct gradient boost.
2. `scheduled_sampling_final: 0.3` — break train/test mismatch.
3. `memory_dropout: 0.1` — force cross-attention use.
4. `qk_clip_tau_init: 4.0` — tighter logit bound for stability.

**Result**: PER dropped from 1.0 to ~0.08 by epoch 5 of the v0.5.0
retrain. Mode collapse did not recur.

## Acceptance gates

| Version | Max PER | Notes |
|---------|---------|-------|
| v0.1.0 | 0.08 | First seq2seq (mode collapse blocked this) |
| v0.5.0 | 0.05 | With mode-collapse fixes |
| v1.0.0 | 0.03 | With Qwen3.5 stack + ensemble |
