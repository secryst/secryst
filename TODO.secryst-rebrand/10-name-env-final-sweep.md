# 10 — Name/env final sweep (native from scratch)

Anything still carrying INTERSCRIPT_ML_* or interscript-ml-branded
implementation names must be renamed. The CONTRACT (models.yaml,
"interscript-ml model index", model IDs) keeps its descriptive name by
design — implementations are secryst.

Targets found:
- ml-models/npm/models: package "@interscript/models" (v0.0.1,
  unpublished) -> "@secryst/models" (the npm face of the index
  manifest).
- Any remaining INTERSCRIPT_ML_/interscript_ml references in active
  code/docs (grep-driven; historical RESULTS/papers stay as history).

Acceptance: grep sweep shows only contract-proper uses; commit rides
ml-models PR #14 branch.
Status: DONE
