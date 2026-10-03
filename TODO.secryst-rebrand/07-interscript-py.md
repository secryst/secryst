# 07 — interscript-py: deterministic engine in Python

interscript-py does not exist (verified) but parity requires py/ruby/ts
engines AND crystals. MVP scope for this item: a real Python engine for
ISC (YAML) maps, not a stub.

Steps:
1. Create interscript/interscript-py (gh) + local ~/src/interscript/interscript-py.
2. ISC parser: YAML -> document tree (stage lists, subst/run rules).
3. Executor: compose/decompose/case ops, subst (regex), run, parse,
   int_class/int_base, secryst/rababa adapters (neural stages).
4. Parity harness: run maps vs golden fixtures; spec suite.
5. Publish prep: pyproject interscript-py (PyPI publish after review).

Acceptance: engine executes a representative map subset and matches
golden fixtures; specs pass; README states scope honestly.
Status: PARTIAL

Coverage notes (verified 2026-10-03 against the interscript/interscript-py
checkout, 52 tests green locally):
- DONE: repo + ISC parser (.imp parsed directly, v0.2.0 — no
  pre-compiled map modules); executor ops (compose/decompose/case,
  subst/run/parse, int_class/int_base); expression layer advanced
  subs per TODO 13 (any/range/list-any/maybe/anchors); engine + ISC +
  gallery-parity spec suites.
- REMAINING: (a) neural stages are PARSED but not EXECUTED — the
  grammar accepts secryst/rababa funcalls yet isc.py only consumes
  'rababa'; wiring the secryst stage means a soft dependency on the
  crystal package and a parity harness across py/ruby/ts neural
  stages; (b) PyPI publish prep done but NOT published
  (pypi.org/pypi/interscript-py 404 as of 2026-10-03) — publish after
  the neural-stage wiring review.
