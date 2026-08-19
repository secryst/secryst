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
Status: PARTIAL (see file for coverage notes)
