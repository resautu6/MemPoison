# MemPoison Benchmark Sample

This directory contains a small public benchmark sample for local smoke testing
and schema inspection. Larger reviewed benchmark releases should be published as
separately versioned artifacts.

## Current Files

- `sample_cases.json`
  - deterministic public sample from the final `1,227`-case release set
  - `30` cases and `90` trigger tasks
  - compact demonstration subset with `10` cases each for L1, L2, and L3
  - channel mix: `9` cross-agent, `12` tool-return, and `9` user-input cases
- `../../src/mempoison/benchmark_schema.py`
  - JSON-array validation logic used by the sample pack

## Validation Contract

Base schema requirements are:

- `case_id`, `title`, `tier`
- `attack_type`, `difficulty_level`, `injection_channel`
- `scenario_domain`, `context`, `injection_turns`, `trigger_tasks`

Additional release fields such as `family_id`, `seed_family_id`,
`domain_cluster`, `multidomain`, and L3 `natural_trigger_condition` metadata
may be present.

Run the smoke validation from the repository root with:

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests
```
