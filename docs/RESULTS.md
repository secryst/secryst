# secryst — SOTA Results (Thai G2P)

All numbers from the 2026-08 SOTA campaign.

## Best result

| Metric | Value | Test set |
|---|---|---|
| **PER** | **1.7260%** | 1,219 Kaikki Thai test sentences (scaleup600k, 2026-08-19) |

- Model: umt5-small continued fine-tuning (B-K/umt5-thai-g2p-v2-0.5k base)
- Data: 9.7K Kaikki + **50K epitran-augmented Thai Wikipedia sentences**
- Checkpoint: `/ckpts/secryst_thai_ipa_combined/run-001/best` (thai_combined_mixed)
- Eval: `train_thai_combined.py` (ap-Xfaypx1O1cpviD6AGws03m)

## Progression

| Stage | PER | Delta |
|---|---|---|
| B-K/umt5-thai-g2p-v2-0.5k (HF baseline) | 6.37% | reference |
| our continued fine-tune (Kaikki 9.7K, 10 ep) | 3.31% | −3.06 |
| + 4-seed selection (s789) | 3.24% | −0.07 |
| **+ epitran augmentation (60K total)** | 2.32% | −0.92 |
| **+ epitran augmentation 367K (full Wikipedia, 1 ep)** | **1.7260%** | **−0.59** |

## Ablations (all measured on identical 1,219-example test)

| Approach | PER | Verdict |
|---|---|---|
| from-scratch seq2seq (byt5_thai arch) | mode collapse | resists all standard fixes |
| CTC arch | ~70% plateau | conditional independence can't model phonotactics |
| ByT5-small from scratch | 13% | pretrained backbone essential |
| umt5 continued FT (final recipe) | 3.31→2.32% | winner |
| 4-seed medoid-vote ensemble | 3.26% | no gain — models too similar |
| logit-avg ensemble (greedy) | 6.56% | invalid — needs beam; greedy alone costs 3+ pts |
| HEBATRON-style curriculum (3 phases) | 3.60% | HURT — model already at data-limited optimum |
| 20 epochs (vs 10) | 4.05% | overtraining |
| Gregniuki/thai-g2p-byt5-finetuned-v2 (HF) | 43.56% | format mismatch with Kaikki test |

## Key findings

1. **Deterministic phonemizer augmentation beats LLM distillation**:
   epitran (rule-based, zero cost, no hallucination) on 50K Wikipedia
   sentences cut PER 3.24→2.32%. LLMs cannot be trusted for phonological
   labeling — they hallucinate tones (Thai has 5 phonemic tones) and
   phonotactics, the same failure mode as haraqat hallucination in
   diacritization.
2. **Language-specific pretraining > model size**: umt5-small (300M,
   Thai-exposed) beats ByT5-small from scratch by 10 points.
3. **Curriculum fails at the data-limited optimum** — gains reported for
   low-resource LLM pretraining do not transfer to saturated fine-tuning.
4. **Ensembles need beam search**: any ensemble evaluated without beam
   search measures the decoding regression, not the ensemble.

## Khmer transliteration (2026-08-14)

Data: secryst/data-khmer-translit (17,910 unique word pairs, Khmer→Latin),
the corpus behind the legacy seq2seq net (net-500-epochs.pth, whose
training log reports only train ppl 1.15 — no test metrics were ever
published).

| System | EM | CER | n |
|---|---|---|---|
| ByT5-small, early stop @ep15 | **59.66%** | **27.42%** | 895 |
| ByT5-small, beam 4 | 59.78% | 27.42% | 895 |
| Legacy net (published) | — | — | none published |

Split 16,120/895/895 (seed 42). Beam search: no gain (greedy is
sufficient — unlike Hebrew where beam was worth 12 DER points).

Notes: deepest orthography of our languages (etymological spellings,
silent consonant clusters) and our smallest corpus (16K pairs vs Thai
60K). Improvement path: cycle-consistency self-training on Khmer
Wikipedia (TODO.research/09 in rababa) and Thai→Khmer transfer via the
shared Latin/IPA target space (TODO.research/04).

Run: `modal run train_khmer_byt5.py`; checkpoint
`secryst-checkpoints:/khmer_byt5/run-001/best`.
