from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ATTACK_TYPES = {"fact", "preference", "instruction", "state"}
DIFFICULTY_LEVELS = {"L1_naive", "L2_compositional", "L3_sleeper"}
INJECTION_CHANNELS = {"user_input", "tool_return", "cross_agent"}
SUBSTRATES = ("flat_chunk", "fact_store", "hierarchical_notes")


def load_cases(path: str | Path) -> list[dict[str, Any]]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def validate_case(case: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required = {
        "case_id",
        "title",
        "tier",
        "attack_type",
        "difficulty_level",
        "injection_channel",
        "scenario_domain",
        "context",
        "injection_turns",
        "trigger_tasks",
    }
    missing = sorted(required - set(case))
    if missing:
        errors.append(f"missing required keys: {missing}")

    if case.get("attack_type") not in ATTACK_TYPES:
        errors.append(f"invalid attack_type: {case.get('attack_type')}")
    if case.get("difficulty_level") not in DIFFICULTY_LEVELS:
        errors.append(f"invalid difficulty_level: {case.get('difficulty_level')}")
    if case.get("injection_channel") not in INJECTION_CHANNELS:
        errors.append(f"invalid injection_channel: {case.get('injection_channel')}")

    context = case.get("context", [])
    if not isinstance(context, list) or not context:
        errors.append("context must be a non-empty list")

    turns = case.get("injection_turns", [])
    if not isinstance(turns, list) or not turns:
        errors.append("injection_turns must be a non-empty list")
    difficulty = case.get("difficulty_level")
    if difficulty == "L1_naive" and len(turns) != 1:
        errors.append("L1 cases must have exactly one injection turn")
    if difficulty == "L2_compositional" and len(turns) < 2:
        errors.append("L2 cases must have at least two injection turns")

    triggers = case.get("trigger_tasks", [])
    if not isinstance(triggers, list) or not triggers:
        errors.append("trigger_tasks must be a non-empty list")
    for trigger in triggers:
        if "prompt" not in trigger:
            errors.append("every trigger task needs a prompt")
        if "clean_behavior" not in trigger or "poisoned_behavior" not in trigger:
            errors.append("every trigger task needs clean_behavior and poisoned_behavior")
        if (
            difficulty == "L3_sleeper"
            and "natural_trigger_condition" in trigger
            and not trigger.get("natural_trigger_condition")
        ):
            errors.append("L3 natural_trigger_condition must be non-empty when provided")

    return errors


def summarize_cases(cases: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    summary = {
        "by_attack_type": {name: 0 for name in sorted(ATTACK_TYPES)},
        "by_difficulty": {name: 0 for name in sorted(DIFFICULTY_LEVELS)},
        "by_channel": {name: 0 for name in sorted(INJECTION_CHANNELS)},
    }
    for case in cases:
        summary["by_attack_type"][case["attack_type"]] += 1
        summary["by_difficulty"][case["difficulty_level"]] += 1
        summary["by_channel"][case["injection_channel"]] += 1
    return summary


def validate_cases(cases: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    seen_ids: set[str] = set()
    for case in cases:
        case_id = case.get("case_id", "<missing>")
        if case_id in seen_ids:
            errors.append(f"duplicate case_id: {case_id}")
        seen_ids.add(case_id)
        for error in validate_case(case):
            errors.append(f"{case_id}: {error}")
    return errors


