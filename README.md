# MemPoison

MemPoison is a benchmark for persistent memory poisoning in LLM agents.
This  repository contains:

- a lightweight Python package for benchmark loading, memory-surface simulation,
  defense configuration, live evaluation, and targeted blind-spot analysis
- a small public benchmark sample for local smoke testing

## Repository Layout

The main directories are:

- `src/mempoison/`
  Core reusable code:
  - benchmark loaders and schema helpers
  - memory substrate simulators
  - defense configuration, metrics, and metric formula helpers
  - live evaluation harness
  - targeted blind-spot helpers
- `data/benchmark/`
  Small benchmark sample and benchmark documentation.
- `scripts/`
  Small repository utilities for manifest building and retrieval simulation.
- `tests/`
  Lightweight smoke tests for the public package surface.

If you are reading this repository for the first time, start with the quick
start below and then inspect `src/mempoison/`.

## Install

The core `mempoison` package is installable in editable mode.

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e .
```

The core package uses the Python standard library only.

## Quick Start

### 1. Inspect a benchmark pack

```powershell
@'
from mempoison.benchmark_schema import load_cases, summarize_cases, validate_cases
cases = load_cases("data/benchmark/sample_cases.json")
errors = validate_cases(cases)
if errors:
    raise SystemExit("\n".join(errors))
print(summarize_cases(cases))
'@ | python -
```

This validates the benchmark pack and prints a compact summary.

### 2. Build a small manifest

```powershell
python scripts/build_manifest.py `
  --input data/benchmark/sample_cases.json `
  --output data/manifests/demo_manifest.jsonl `
  --summary-output data/manifests/demo_manifest_summary.json
```

### 3. Run the live harness locally

```powershell
mempoison-live-eval `
  --input data/manifests/demo_manifest.jsonl `
  --output-dir data/results/demo_eval `
  --backend heuristic `
  --defense-config d2_consistency `
  --limit 12
```

For a simple retrieval-only sanity check, you can also run:

```powershell
python scripts/simulate_retrieval.py `
  --input data/benchmark/sample_cases.json `
  --output-dir data/results/demo_retrieval
```

## Tests

Run the public smoke tests with:

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests
```

## Public Code Surface

If you want the smallest supported surface for reuse, focus on:

- `src/mempoison/benchmark_schema.py`
- `src/mempoison/substrates.py`
- `src/mempoison/metrics.py`
- `src/mempoison/metric_formulas.py`
- `src/mempoison/live_eval.py`
- `src/mempoison/targeted_blindspot.py`
- `data/benchmark/sample_cases.json`

## Open-Source Notes

The code is released under the MIT license. Benchmark artifacts may have
separate publication or citation requirements; see `data/benchmark/` before
redistribution.
