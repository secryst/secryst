# 06 — Website: secryst.github.io

Static site at ~/src/secryst/secryst.github.io, deployed via GitHub
Pages on the secryst org. Identity: scrying + crystal — "reveal the
hidden reading of a script". NO generic template look.

Sections: story (etymology), what it does (vocalize/phonemize/diacritize
examples Arabic/Hebrew/Khmer), the crystal family (Ruby/Python/TS
installs), the contract diagram (interscript-ml models.yaml/IMF v1 +
golden parity), models table (from models.yaml), relationship to
Interscript, BSD-3/MIT license notes.

Steps:
1. Build the site (hand-written HTML/CSS, no build chain — deterministic
   deploy). 2. gh repo create secryst/secryst.github.io --public.
3. Push master. 4. Enable Pages (gh api .../pages, source master,/).
5. Verify https://secryst.github.io renders.

Acceptance: live URL returns the site; validates on no-JS.
Status: DONE (deployed)
