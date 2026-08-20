# 12 — Website redesign: technical primer + documentation site

Replace the candlelit single-pager with a modern, professional,
AI/ML-and-standards-themed primer/documentation site at
~/src/secryst/secryst.github.io (deployed as secryst.github.io).

Identity: the scrying crystal re-expressed as a PRISM — opaque script
in, spectral reading out. Light, instrument-panel theme; Space
Grotesk / IBM Plex Sans / IBM Plex Mono; one spectral-gradient accent
used sparingly (hero beam); everything else strict grid + code-forward
typography.

Pages (static, no build chain, no-JS navigation):
- index.html — landing: prism hero, what/why, quickstart per crystal,
  contract summary card, models table, status honesty.
- primer.html — concepts: the hidden reading; diacritization /
  vocalization / phonemization; local deterministic inference; the
  no-LLM-teaching rule.
- spec.html — the standards core: models.yaml index schema +
  resolution algorithm; IMF v1 (metadata.yaml fields, member sha256,
  byte-tokenizer table byte+3/pad0/eos1/unk2, decoder plain/kv, parts
  contract); RFC-2119-style conformance statements.
- crystals.html — per-language API reference (Ruby/Python/TS) with
  install, snippets, env vars, status.
- models.html — model zoo with metrics + conformance methodology
  (golden sets, three-way diff).

Quality floor: responsive, visible focus, prefers-reduced-motion,
no-JS nav via per-page body classes, semantic HTML.
Acceptance: deployed at https://secryst.github.io, all pages 200,
no console-JS dependency.
Status: DONE (deployed)
