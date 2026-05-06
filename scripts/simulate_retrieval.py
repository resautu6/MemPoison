from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mempoison.benchmark_schema import SUBSTRATES, load_cases, validate_cases
from mempoison.substrates import build_memory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    cases = load_cases(args.input)
    errors = validate_cases(cases)
    if errors:
        raise SystemExit("\n".join(errors))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    per_case_path = output_dir / "per_case_retrieval.jsonl"
    aggregate: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    rows = []

    for substrate in SUBSTRATES:
        by_difficulty: dict[str, list[dict[str, float]]] = defaultdict(list)
        for case in cases:
            memory = build_memory(substrate)
            memory.write_case(case)
            trigger_hits = 0
            trigger_count = 0
            top1_poison = 0
            clean_support = 0
            retrieved_ids = []
            for trigger in case["trigger_tasks"]:
                trigger_count += 1
                retrieved = memory.retrieve(trigger["prompt"], top_k=args.top_k)
                retrieved_ids.append([record.record_id for record in retrieved])
                poison_hit = any(record.is_poison for record in retrieved)
                clean_hit = any(not record.is_poison for record in retrieved)
                if poison_hit:
                    trigger_hits += 1
                if clean_hit:
                    clean_support += 1
                if retrieved and retrieved[0].is_poison:
                    top1_poison += 1

            row = {
                "case_id": case["case_id"],
                "substrate": substrate,
                "difficulty_level": case["difficulty_level"],
                "attack_type": case["attack_type"],
                "poison_recall_at_k": trigger_hits / trigger_count,
                "top1_poison_rate": top1_poison / trigger_count,
                "clean_support_rate": clean_support / trigger_count,
                "retrieved_record_ids": retrieved_ids,
            }
            rows.append(row)
            by_difficulty[case["difficulty_level"]].append(row)

        aggregate[substrate] = {}
        for difficulty, difficulty_rows in by_difficulty.items():
            aggregate[substrate][difficulty] = {
                "num_cases": len(difficulty_rows),
                "avg_poison_recall_at_k": average(difficulty_rows, "poison_recall_at_k"),
                "avg_top1_poison_rate": average(difficulty_rows, "top1_poison_rate"),
                "avg_clean_support_rate": average(difficulty_rows, "clean_support_rate"),
            }

    with per_case_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(aggregate, indent=2, ensure_ascii=False), encoding="utf-8")


def average(rows: list[dict[str, float]], key: str) -> float:
    return round(sum(row[key] for row in rows) / max(len(rows), 1), 4)


if __name__ == "__main__":
    main()
