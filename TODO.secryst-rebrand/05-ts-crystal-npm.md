# 05 — TypeScript crystal `secryst (npm, bare name — verified free)`

Third crystal (parity: deterministic engines run in py/ruby/ts, so the
neural layer must too). npm scope @secryst is NOT yet claimed; local npm
is unauthenticated — publish requires `npm login` + org create (user
action); this item delivers the publishable package.

Steps:
1. Create secryst/secryst-ts (gh).
2. Package secryst (npm, bare name — verified free): index resolution (models.yaml), cache dir,
   sha256 verify, IMF v1 zip load, onnxruntime-node inference. Mirror
   the Python crystal API (Model.load / translate).
3. Golden-parity test harness (mirrors Python goldens).
4. npm pack verified; publish blocked on npm auth (documented).

Acceptance: `npm pack` produces installable tarball; parity test runs
locally (onnxruntime-node dep may need install).
Status: DONE (package 'secryst' 0.1.0, tarball secryst-0.1.0.tgz; bare npm name chosen 2026-08-20 for pip/gem/npm symmetry; publish pending npm login)
