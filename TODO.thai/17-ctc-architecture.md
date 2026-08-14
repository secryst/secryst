# 17 — CTC Architecture for Mode Collapse Escape

## Problem

Secryst Thai→IPA seq2seq model mode-collapses (every input → constant output,
PER > 1.0) and resists every fix tried (see `secryst-mode-collapse.md` memory):

1. cross_attn_lr_mult=3.0 — no help
2. scheduled_sampling_final=0.3 — no help
3. memory_dropout=0.1 — no help
4. skip_pretrain — no help
5. use_mhc_cross=False — PER worse (2.18)
6. Plain AdamW baseline — PER worst (2.34)

Local smoke test confirmed: trained model produces identical outputs for
different inputs. The decoder learns to ignore the encoder regardless of
optimizer, normalization, or training-time tricks.

## Fix: Switch to CTC

CTC (Connectionist Temporal Classification, Graves 2006) is the standard
objective for sequence labeling with monotonic alignment. Unlike seq2seq:

- **No decoder LM prior** can form — output is purely conditional on input
- **No exposure bias** — single forward pass, no teacher forcing
- **No EOS prediction needed** — output length is determined by input length
- **Monotonic alignment** is built-in

Standard for G2P, ASR, handwriting recognition.

## Architecture

```
CTCModel:
  encoder: ModernEncoder (RoPE + mHC + SwiGLU + RMSNorm)  ← reuse existing
  head: Linear(dim, output_vocab + 1)  ← +1 for CTC blank token
  ctc_loss: torch.nn.CTCLoss(blank=output_vocab)

forward(src, src_lengths, target, target_lengths):
  hidden = encoder(src)              # (B, T, dim)
  logits = head(hidden)              # (B, T, V+1)
  log_probs = log_softmax(logits, -1)
  # CTCLoss expects (T, B, V) so transpose
  loss = ctcloss(log_probs.transpose(0,1), target, src_lengths, target_lengths)
  return loss

greedy_decode(src, src_lengths):
  hidden = encoder(src)
  logits = head(hidden)
  pred = logits.argmax(-1)            # (B, T)
  # Collapse repeats
  out = []
  for b in range(B):
    tokens = []
    prev = -1
    for t in range(T):
      tok = pred[b, t].item()
      if tok != prev and tok != blank_id:
        tokens.append(tok)
      prev = tok
    out.append(tokens)
  return out
```

## Files

- `src/secryst/models/ctc.py` (NEW) — `CTCModel` class.
- `src/secryst/training/ctc_supervised.py` (NEW) — training loop using CTCLoss.
- `src/secryst/decoding/ctc_greedy.py` (NEW) — greedy decode.
- `src/secryst/decoding/ctc_beam.py` (NEW, optional) — beam search.
- `configs/secryst_thai_ipa_ctc.yaml` (NEW) — config selecting `arch: ctc`.
- `tests/test_ctc.py` (NEW) — specs for model + loss + decode.

## Config flag

```yaml
model:
  arch: ctc            # selects CTCModel instead of ModernSeq2Seq
  dim: 256
  enc_layers: 8        # encoder-only, so add more layers
  heads: 8
  ff_dim: 1024
```

## Acceptance

- CTCModel forward returns finite loss.
- CTCLoss decreases over epochs on Thai data.
- Greedy decode produces non-constant outputs.
- PER < 0.50 on test set (much better than current PER=2.34).
- All specs pass.

## Why this should work where seq2seq didn't

CTC has **no decoder** — there's no place for a "language model prior" to
form. Every output token MUST be explainable by some input position. The
encoder's hidden state at position t directly determines the output at
position t (modulo blank collapsing). This eliminates the entire failure
mode we're seeing.

The trade-off: CTC assumes **monotonic alignment** (input position t
corresponds to output position ≤ t). For G2P this is true (Thai left-to-right
maps to IPA left-to-right). For tasks like MT it would be wrong.
