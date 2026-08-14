# Secryst Thai→IPA — remaining work

First run launched (ap-ovDv54b9BPk5EPOYuZH57v). After v0.1.0 ships:

## Tier 1 — v0.1.0 ship
- (Pipeline already covers: fetch → pretrain → train → export fp32 → eval)
- [02-tone-side-channel](02-tone-side-channel.md) — expose Thai consonant class as input feature.

## Tier 2 — v0.5.0
- [03-pseudosyllable-segmentation](03-pseudosyllable-segmentation.md) — multi-task head for syllable boundaries (Thai has no spaces).
- [04-bigger-corpus](04-bigger-corpus.md) — augment Wiktionary with PyThaiNLP g2p silver.
- [05-multi-seed-ensemble](05-multi-seed-ensemble.md) — same recipe as rababa.
- [06-electra-pretraining](06-electra-pretraining.md) — RTD replaces MLM.

## Tier 3 — v1.0.0+
- [07-tone-class-conditional-decoding](07-tone-class-conditional-decoding.md) — beam search biased by tone rules.
- [08-augmentation-pythainlp](08-augmentation-pythainlp.md) — silver augmentation via PyThaiNLP g2p.
- [09-active-learning](09-active-learning.md) — surface hard IPA patterns.
- [10-benchmark-harness](10-benchmark-harness.md) — standard Thai G2P test sets.
