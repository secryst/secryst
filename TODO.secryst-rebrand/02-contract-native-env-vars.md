# 02 — Native env vars (SECRYST_INDEX / SECRYST_CACHE)

Decision: crystals are secryst-branded, so their env vars are SECRYST_*
from scratch. Nothing is published/deployed yet (verified: interscript-ml
0.1.0 unpublished, npm names 404) — so rename outright, no aliases.
The CONTRACT itself stays interscript-ml (models.yaml, IMF v1, model IDs).

Repo: ml-models (branch off current HEAD; user's uncommitted fas-g2p work
stays untouched — explicit-path staging only).

Steps:
1. models.yaml header: INTERSCRIPT_ML_INDEX/CACHE -> SECRYST_INDEX/CACHE.
2. runtime/ code + tests: env var reads.
3. Docs mentioning the env vars (README, runtime README).
4. Branch feat/secryst-native-env, push, PR.

Acceptance: grep INTERSCRIPT_ML_ returns only historical/changelog hits.
Status: DONE (PR opened)
