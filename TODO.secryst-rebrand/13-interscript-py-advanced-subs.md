# 13 — interscript-py: advanced sub forms + library runs

Extend the .imp expression layer so the prs-Arab map class loads and
the ell-Grek map class runs:
- Expression evaluator for sub patterns: string literals, any("...") ->
  character class, space, boundary -> \b, concatenation with +.
- Context guards before:/after: -> lookbehind/lookahead.
- Dotted run targets (map.greklatn.stage.main) -> resolve in the
  maps/libs library space.
Acceptance: bgnpcgn-prs-Arab-Latn-2007 embedded tests improve from
0/48 (currently xfail); ell map resolves library runs; suite green.
Status: see final report
