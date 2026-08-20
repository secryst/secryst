# 11 — Gem 1.0 parity CI (unblocks secryst PR #47)

Golden-set parity is the gate for gem 1.0. Deliverables:
- secryst-py: parity runner over the tiny-graph fixture zip (no torch,
  no network) producing/verifying golden outputs (JSONL).
- Fixture zip committed to secryst-py (small) so Ruby CI can consume
  the same artifact.
- Gem repo (feat/secryst-v1): .github/workflows/parity.yml — bundle
  install, run spec/parity_spec.rb against the fixture + goldens from
  secryst-py, diff Python vs Ruby outputs.
- Local verification: Python side runs green here; Ruby side runs in CI
  (no local bundle by design).

Acceptance: Python parity locally green; workflow + spec pushed to the
v1 branch; PR #47 updated.
Status: DONE (Python side verified locally; Ruby side wired for CI)
