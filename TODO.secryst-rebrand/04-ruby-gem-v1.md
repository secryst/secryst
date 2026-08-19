# 04 — Ruby gem `secryst` v1.0 line (IMF v1)

The modernized gem lives on secryst/secryst feat/imf-runtime (gem lib +
training playground). Target: secryst 1.0 == IMF v1 conformance.

Steps:
1. Examine lib/ + spec/ on feat/imf-runtime (post-01 fetch).
2. Align to SECRYST_INDEX/SECRYST_CACHE (per 02).
3. README.adoc: scrying etymology, IMF v1 conformance statement,
   install (gem 'secryst'), models.yaml contract reference.
4. Branch feat/secryst-v1 off feat/imf-runtime; push; PR to master.
5. Full v1.0 release gated on golden-set CI parity vs Python crystal
   (tracked separately; do NOT cut gem 1.0 from this TODO alone).

Acceptance: PR open with naming/env/README alignment; specs green if
runnable locally (bundle may be needed).
Status: DONE (PR opened; v1.0 release gated on parity CI)
