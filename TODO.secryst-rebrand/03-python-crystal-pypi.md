# 03 — Python crystal: PyPI package `secryst`

The existing Python runtime (ml-models/runtime, pkg interscript-ml 0.1.0,
unpublished) IS the Python crystal — rename to `secryst` and give it the
golden-owner role: it generates the goldens, Ruby/TS diff against it.

Steps:
1. Create secryst/secryst-py (gh). Copy runtime content (history note in
   README: originated in ml-models/runtime). ml-models/runtime stays
   frozen as provenance; add pointer note (no deletions).
2. Rename module interscript_ml -> secryst; pyproject name=secryst
   v0.1.0; README: scrying etymology + "implements the interscript-ml
   contract" + install/usage.
3. SECRYST_INDEX/SECRYST_CACHE env vars (per 02).
4. Build sdist+wheel; twine upload (token in ~/.pypirc, never printed).
5. Smoke: pip install secryst in venv; resolve index; Model.load smoke.

Acceptance: pypi.org/pypi/secryst/json 200; pip install works.
Status: DONE (published 0.1.0)
