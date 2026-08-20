# 15 — Separation docs audit (interscript vs secryst)

Audit + fix of every surface that still described the old rababa layout
or failed to state the dependency directions.

## Done
- interscript-ml README rewritten (contract owner, system map) — PR #18 merged
- interscript-ml/SPEC.md normative v1 added — PR #18
- runtime/README sample: from secryst import Model — PR #18
- interscript-ruby README phonological-layer section — PR #761 merged
- rababa/rababa-farsi/rababa-urdu descriptions → archived origins
- interscript + secryst org descriptions updated
- secryst.org spec page points at canonical SPEC.md
- interscript.org (astro-migration): about ecosystem, /docs/phonological-layer,
  footer links — PR #122 open against astro-migration
- gem release prep 1.0.0 (gemspec + trusted publish workflow) — PR #49

## Remaining user actions
- Merge interscript.org PR #122 (astro-migration)
- Register RubyGems trusted publisher for secryst/secryst (workflow release.yml)
- Cut gem 1.0.0: gh release create v1.0.0 -R secryst/secryst
- r6 verdict when training completes (~step 8700/24263 as of write-up)
