from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mempoison.benchmark_schema import SUBSTRATES, load_cases, summarize_cases, validate_cases


MODELS = ("demo-model-a", "demo-model-b")


def build_manifest_records(cases: list[dict]) -> list[dict]:
    records = []
    for case in cases:
        for model_id in MODELS:
            for substrate_id in SUBSTRATES:
                for condition in ("clean", "poisoned"):
                    metadata = {
                        key: case[key]
                        for key in (
                            "tier",
                            "family_id",
                            "seed_family_id",
                            "domain_cluster",
                            "multidomain",
                        )
                        if key in case
                    }
                    records.append(
                        {
                            "run_id": f"{case['case_id']}::{model_id}::{substrate_id}::{condition}",
                            "case_id": case["case_id"],
                            "title": case["title"],
                            "model_id": model_id,
                            "substrate_id": substrate_id,
                            "condition": condition,
                            "difficulty_level": case["difficulty_level"],
                            "attack_type": case["attack_type"],
                            "injection_channel": case["injection_channel"],
                            "scenario_domain": case["scenario_domain"],
                            "context": case["context"],
                            "injection_turns": case["injection_turns"] if condition == "poisoned" else [],
                            "trigger_tasks": case["trigger_tasks"],
                            "metadata": metadata,
                        }
                    )
    return records


def build_manifest_summary(cases: list[dict], records: list[dict]) -> dict:
    return {
        "num_cases": len(cases),
        "models": list(MODELS),
        "substrates": list(SUBSTRATES),
        "num_runs": len(records),
        "case_mix": summarize_cases(cases),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output", required=True)
    args = parser.parse_args()

    cases = load_cases(args.input)
    errors = validate_cases(cases)
    if errors:
        raise SystemExit("\n".join(errors))

    manifest_path = Path(args.output)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = Path(args.summary_output)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    records = build_manifest_records(cases)

    with manifest_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = build_manifest_summary(cases, records)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
