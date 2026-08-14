# 10 — Benchmark harness (Thai)

Standard Thai G2P benchmarks:
- **Wiktionary held-out test** (1,209 examples) — current default
- **BEST corpus** (Thai NLP standard)
- **Wisesight-1000** (modern Thai text)

## Tasks

### 10.1 Register benchmarks in secryst
- `src/secryst/benchmarks/registry.py` (new, port pattern from rababa)

### 10.2 Download + format
- BEST corpus: download from PyThaiNLP mirror
- Wisesight: from Wisesight repo

### 10.3 Benchmark entrypoint
- `modal_app.py::benchmark` (mirror rababa pattern)

## Acceptance
- [ ] All 3 benchmarks run end-to-end
- [ ] Results JSON written to `/models/secryst_thai_ipa/benchmark-{version}.json`

## Files
- `src/secryst/benchmarks/` (new subpackage)
- `modal_app.py` (add benchmark entrypoint)
