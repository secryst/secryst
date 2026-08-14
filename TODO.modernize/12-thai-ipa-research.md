# Thai→IPA SOTA research synthesis

## Task

Thai→IPA grapheme-to-phoneme (G2P) transliteration. Input: Thai
script (44 consonants, 15 vowel symbols, 4 tone marks, digits,
punct). Output: International Phonetic Alphabet string with tones,
length, and stress markers.

This is **not** a diacritization task. Output length differs from
input length. Tones are conditioned on syllable-class features
(consonant class + vowel length + tone mark) and cannot be predicted
by per-position classification alone. → use **encoder-decoder**
Transformer with cross-attention, not rababa's encoder-only stack.

## Latest research (2022–2026)

### Yamasaki et al., NAACL 2022 — Neural regression for Thai G2P
- Encoder-decoder Transformer, ~20M params.
- 80/10/10 train/val/test split on Wiktionary.
- PER ≈ 4.5% on test. Strong baseline.

### Rugchatjaroen 2019 — Two-stage joint sequence model
- Stage 1: pseudo-syllable segmentation (Thai has no spaces).
- Stage 2: per-syllable G2P with joint sequence model.
- Foundational insight: Thai has no inter-word spaces, so word and
  syllable boundaries are themselves signals to learn.

### Saychum et al. 2016 — Joint sequence model for Thai G2P
- CRF + joint n-gram over (grapheme, phoneme) pairs.
- Established the Wiktionary Thai-IPA test corpus.

### AyutthayaAlpha (arXiv:2412.03877, Dec 2024) — Thai-Latin transliteration
- Transformer seq2seq for Thai→Latin (not IPA, but same architecture).
- Introduced tone-aware positional encoding. Validated that char-level
  Transformer with appropriate positional encoding handles Thai tonal
  structure.

### Phoneme-Tone Adaptive G2P (arXiv:2504.07858, April 2025)
- **Key paper for our sprint.** Adapter modules stacked on a
  pretrained multilingual G2P that route tone-class features into
  per-head Q/K/V. Achieves SOTA on Thai with <1M fine-tuning params.
- We can adopt the *spirit* (tone-aware attention) without the
  adapter overhead by exposing tone-class as auxiliary input. For
  v0.1.0 we keep it simple: char-level encoder learns tones implicitly.
  v0.5.0 may add a tone-class side channel if DER stalls.

### PyThaiNLP g2p (rule-based, 2024)
- Used for synthetic silver augmentation when gold IPA is scarce.
- Provides deterministic fallback for low-frequency words.

## Architecture (secryst v0.1.0)

```
Encoder (Thai chars → hidden)
  └─ ModernEncoderLayer × N  (RoPE + SDPA + mHC + AttnRes + RMSNorm + SwiGLU)

Decoder (IPA tokens → IPA tokens, autoregressive)
  └─ ModernDecoderLayer × N
       ├─ self-attn (causal mask + RoPE + mHC)
       ├─ cross-attn (Q from decoder, KV from encoder, no RoPE)
       └─ ffn (SwiGLU + mHC)

Optimizer: Muon (2D weights) + AdamW (1D + embeddings) — same as rababa.
QK-Clip: applied to encoder self-attn + decoder self-attn + cross-attn.

Sizes (target ≤30M params for single-A100 train in hours):
  dim=256, layers=6 enc + 6 dec, heads=8, ff_dim=1024, max_len=128.
```

## Data

**Wiktionary Thai-IPA** via `kaikki.org`:
- URL: `https://kaikki.org/dictionary/Thai/kaikki.org-dictionary-Thai.jsonl`
- ~10K entries with IPA. Filter to single-IPA entries, validate IPA
  char set, deduplicate.

**Augmentation** (for v0.5.0): run PyThaiNLP g2p on unlabeled Thai
text from Wikipedia; treat as silver; mix in with confidence weight.
Not in v0.1.0 — keep training set clean for baseline PER.

## Acceptance

| Version | PER on test | Notes |
|---------|------------|-------|
| v0.1.0  | ≤ 8%       | Direct supervised, no aug |
| v0.5.0  | ≤ 5%       | + augmentation + tuning |
| v1.0.0  | ≤ 3%       | + tone-class side channel, SOTA territory |

## "No quantized model" — what it means

User explicitly requested: ship fp32 only, no int8 quantization.
- ONNX export: fp32 only, skip `quantize_dynamic_int8`.
- TFLite export: fp32 only, skip PT2E.
- ~50–80MB ONNX at fp32 (well within browser-loadable range).

## Tasks tracked

- #193 secryst Thai-IPA SOTA pipeline (in_progress)
