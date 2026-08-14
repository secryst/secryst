# 18 — Secryst Pipeline Resume Support

## Problem

Secryst pipeline orchestrator uses `--force` to wipe BOTH pretrain and train
checkpoint dirs. The rababa pipeline has the same pattern but with smarter
skip logic (only wipe if config changed). When secryst crashes mid-train
(e.g., workspace disabled), restarting loses all progress.

The supervised.py training loop DOES call `latest_resume_checkpoint`, so
resume works WITHIN a single train stage. The issue is the orchestrator
wiping between pipeline runs.

## Fix

### Part A: Smarter --force in secryst orchestrator

Match rababa's pattern: when --force is passed, only wipe if the existing
checkpoint's config differs from the current config. Otherwise resume.

### Part B: Stage-level skip if config unchanged

Add a `_config_hash` to the status file. If the current config's hash
matches the saved one AND `_done(stage)` is True, skip the stage.

## Files

- `secryst/modal_app.py:run_sota_pipeline` — add config hash check.
- `secryst/src/secryst/training/resume.py` — `config_hash()` helper.

## Acceptance

- Re-run pipeline after crash → resumes from last epoch, doesn't restart.
- Re-run with changed config → wipes and retrains.
- Re-run with --force → always wipes (current behavior preserved).
