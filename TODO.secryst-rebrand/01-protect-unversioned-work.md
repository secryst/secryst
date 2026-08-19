# 01 — Protect unversioned secryst work (git reconnect)

The local working copy at ~/src/interscript/secryst has NO .git. Today's
Thai scale-up scripts exist only on local disk. Remote base:
secryst/secryst branch feat/imf-runtime (contains the gem + training
playground; closest ancestor of the local tree).

Steps:
1. git init; remote add origin (ssh); fetch origin.
2. Mark base: read-tree feat/imf-runtime into the index (working tree
   untouched) to see the true local-vs-remote diff.
3. Stage ONLY explicit changed/new files (never stage-all).
4. Commit on branch wip/training-playground (HEAD switched via
   symbolic-ref, no checkout), push to secryst/secryst.

Acceptance: `git ls-remote origin wip/training-playground` resolves;
`git status` clean for tracked files; remote branches untouched.
Status: DONE (see commit on branch)
