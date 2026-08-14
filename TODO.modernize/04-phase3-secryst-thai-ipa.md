# Phase 3 — secryst Thai-IPA from scratch (Tier 1: direct supervised)

## Goal
Cut `secryst_thai_ipa-v0.1.0`. First real production secryst model.
Direct supervised training on Wiktionary Thai-IPA pairs.

## Why this is the biggest phase
Secryst has zero production code today. No Ruby adapter, no TS model,
no ONNX. We're building everything from scratch — but with the
infrastructure already validated by rababa.

## Tasks

### 3.1 Repo setup: standalone `secryst/` repo
Mirror `rababa/` layout exactly:
```
secryst/
├── pyproject.toml
├── modal_app.py
├── src/secryst/
│   ├── datasets.py        # Wiktionary Thai-IPA fetcher
│   ├── models/
│   │   ├── student.py     # 6-layer char transformer
│   │   └── quantize.py
│   ├── training/
│   │   └── supervised.py
│   ├── export.py
│   └── evaluate.py        # CER on Thai-IPA pairs
├── configs/
├── models/
├── tests/
├── Dockerfile
└── README.md
```

### 3.2 Data acquisition

**Wiktionary Thai-IPA** (~10 K pairs):
- Source: `https://kaikki.org/dictionary/Thai/`. Each Thai word has
  IPA in Wiktionary markup.
- Filter: entries with exactly one IPA transcription (multi-dialect OK).
- Validate: IPA only contains `ɐɑɒɓɔɕɖɘəɛɚɜɞɟʄɡɠɢɦɥɧɨɪɫɬɭɮɱɲɳɴøɵɸθœɶʘɹɺɾɻʀʁʂʃʈʈʰʉʊʋⱱʌɣɤʍχʎʏźʐʒʒʲˈˌːˑː̃ʰʷˤ` + spaces + digits.

**Synthetic augmentation**:
- Apply existing rule-based Thai maps to produce noisy silver.
- Generate ~5 K extra pairs.

### 3.3 Vocab
- Input: Thai alphabet (~70 chars: consonants, vowels, tone marks, digits, space).
- Output: IPA alphabet (~70 chars: standard IPA + extensions + space + stress marks).

### 3.4 Tier 1 student training
- 6-layer char transformer, max_len=256, ~20 M params.
- 1× A100 40 GB, 6 epochs, ~3 h.
- Acceptance for v0.1.0: **CER ≤ 15%** on Wiktionary test split (research baseline).

### 3.5 ONNX export
- Shape: `[batch_size=16, max_len=256]` (Thai compounds are longer than Arabic haraqat contexts).
- int8 quantization.
- Acceptance: ≤ 20 MB int8.

### 3.6 Release
- Cut `secryst_thai_ipa-v0.1.0` tag.
- Upload ONNX + vocab to GitHub Release.
- Update `ml-models/npm/models/manifest.json`: version `0.1.0`, status `research`.

## Acceptance
- [ ] CER ≤ 15% on Wiktionary test split
- [ ] int8 ≤ 20 MB
- [ ] TS parity 100% on `var-th-Thai-Latn-secryst` test vectors (if exists)

## Path to v0.5.0 → v1.0.0

Same cadence as rababa Arabic:
- v0.1.0: Tier 1 baseline.
- v0.5.0: data augmentation + hyperparameter tuning.
- v1.0.0: ≤ 10% CER + 1 month stability in production.

## Open questions
1. **Wiktionary IPA quality**: Thai IPA transcriptions vary by editor. Filter to "stable" entries only? Or accept noisy data?
2. **Bigger student for Thai**: Thai has long compounds. 6 layers might not capture long-range tones. Test with 8-12 layers in v0.5.0.
