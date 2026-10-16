from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable


TOKEN_RE = re.compile(r"[a-z0-9@._/-]+")
FACT_SPLIT_RE = re.compile(r"(?<=[!?;])\s+|(?<=\.)\s+(?=[A-Z0-9])")
EXTERNAL_SUFFIXES = (".io", ".xyz", ".biz", ".ru", ".cn", ".top")
CHANGE_HINT_TOKENS = {
    "changed",
    "future",
    "migrate",
    "migrated",
    "move",
    "moved",
    "now",
    "remember",
    "replace",
    "reroute",
    "switch",
    "update",
    "use",
}
DEFAULT_ANOMALY_WEIGHT_PRESETS = {
    "legacy": {
        "novel_ratio": 0.55,
        "special_novelty": 0.25,
        "external_marker": 0.2,
    },
    "anchored_update": {
        "anchored_novelty": 0.45,
        "change_hint": 0.35,
        "special_novelty": 0.1,
        "external_marker": 0.1,
    },
}


@dataclass(slots=True)
class MemoryDefenseConfig:
    name: str = "none"
    base_name: str = "none"
    apply_provenance: bool = False
    apply_consistency: bool = False
    apply_anomaly_filter: bool = False
    trust_weights: dict[str, float] = field(
        default_factory=lambda: {
            "context": 1.0,
            "episode_summary": 0.9,
            "user_input": 0.7,
            "tool_return": 0.4,
            "cross_agent": 0.3,
        }
    )
    consistency_threshold: float = 0.58
    anomaly_threshold: float = 0.68
    anomaly_formula: str = "legacy"
    anomaly_component_weights: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_ANOMALY_WEIGHT_PRESETS["legacy"])
    )
    defense_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MemoryRecord:
    record_id: str
    text: str
    source: str
    stage: str
    is_poison: bool
    trust_weight: float = 1.0


def normalize_tokens(text: str) -> set[str]:
    return set(TOKEN_RE.findall(text.lower()))


def overlap_score(query: str, text: str) -> float:
    query_tokens = normalize_tokens(query)
    text_tokens = normalize_tokens(text)
    if not query_tokens or not text_tokens:
        return 0.0
    return len(query_tokens & text_tokens) / len(query_tokens | text_tokens)


def special_tokens(text: str) -> set[str]:
    return {token for token in normalize_tokens(text) if "." in token or "@" in token or "/" in token}


def consistency_conflict_score(text: str, records: list[MemoryRecord]) -> float:
    baseline_records = records
    if not baseline_records:
        return 0.0

    incoming_tokens = normalize_tokens(text)
    incoming_special = special_tokens(text)
    if not incoming_tokens:
        return 0.0

    update_hint = 1.0 if incoming_tokens & CHANGE_HINT_TOKENS else 0.0
    best_score = 0.0

    for record in baseline_records:
        existing_tokens = normalize_tokens(record.text)
        if not existing_tokens:
            continue
        shared_tokens = incoming_tokens & existing_tokens
        if not shared_tokens:
            continue
        existing_special = special_tokens(record.text)
        anchor_score = len(shared_tokens) / max(min(len(incoming_tokens), len(existing_tokens)), 1)
        special_mismatch = 1.0 if incoming_special and existing_special and incoming_special.isdisjoint(existing_special) else 0.0
        score = min(1.0, 0.45 * anchor_score + 0.35 * special_mismatch + 0.2 * update_hint)
        best_score = max(best_score, score)

    return round(best_score, 4)


def normalize_anomaly_formula(formula: str | None = None) -> str:
    normalized = (formula or "legacy").strip().lower()
    if normalized not in DEFAULT_ANOMALY_WEIGHT_PRESETS:
        raise ValueError(f"unknown anomaly formula: {formula}")
    return normalized


def normalize_anomaly_component_weights(
    overrides: dict[str, float] | None = None,
    *,
    formula: str = "legacy",
) -> dict[str, float]:
    normalized_formula = normalize_anomaly_formula(formula)
    weights = dict(DEFAULT_ANOMALY_WEIGHT_PRESETS[normalized_formula])
    if not overrides:
        return weights

    for raw_key, raw_value in overrides.items():
        key = str(raw_key).strip().lower()
        if key not in weights:
            raise ValueError(f"unknown anomaly component for {normalized_formula}: {raw_key}")
        value = float(raw_value)
        if value < 0:
            raise ValueError(f"anomaly component weights must be non-negative: {raw_key}={raw_value}")
        weights[key] = value
    return weights


def _format_defense_float(value: float) -> str:
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    return text.replace("-", "m").replace(".", "p")


def make_defense_label(
    base_name: str,
    *,
    defense_label: str | None = None,
    anomaly_formula: str | None = None,
    anomaly_threshold: float | None = None,
    anomaly_component_weights: dict[str, float] | None = None,
) -> str:
    if defense_label:
        return defense_label

    normalized_formula = normalize_anomaly_formula(anomaly_formula)
    parts: list[str] = []
    if normalized_formula != "legacy":
        parts.append(f"f{normalized_formula}")
    if anomaly_threshold is not None:
        parts.append(f"t{_format_defense_float(anomaly_threshold)}")
    if anomaly_component_weights:
        normalized = normalize_anomaly_component_weights(anomaly_component_weights, formula=normalized_formula)
        for component in sorted(normalized):
            parts.append(f"{component[0]}{_format_defense_float(normalized[component])}")
    if not parts:
        return base_name
    return f"{base_name}@{'_'.join(parts)}"


def anomaly_score_breakdown(
    text: str,
    records: list[MemoryRecord],
    weights: dict[str, float] | None = None,
    *,
    formula: str = "legacy",
) -> dict[str, float]:
    normalized_formula = normalize_anomaly_formula(formula)
    normalized_weights = normalize_anomaly_component_weights(weights, formula=normalized_formula)
    baseline_records = records
    if not baseline_records:
        raw_components = {
            "novel_ratio": 0.0,
            "anchored_novelty": 0.0,
            "special_novelty": 0.0,
            "external_marker": 0.0,
            "change_hint": 0.0,
        }
    else:
        incoming_tokens = normalize_tokens(text)
        if not incoming_tokens:
            raw_components = {
                "novel_ratio": 0.0,
                "anchored_novelty": 0.0,
                "special_novelty": 0.0,
                "external_marker": 0.0,
                "change_hint": 0.0,
            }
        else:
            baseline_tokens: set[str] = set()
            baseline_special: set[str] = set()
            for record in baseline_records:
                baseline_tokens |= normalize_tokens(record.text)
                baseline_special |= special_tokens(record.text)

            incoming_special = special_tokens(text)
            novel_tokens = incoming_tokens - baseline_tokens
            novel_ratio = len(novel_tokens) / max(len(incoming_tokens), 1)
            special_novelty = len(incoming_special - baseline_special) / len(incoming_special) if incoming_special else 0.0
            external_marker = 1.0 if any(token.endswith(EXTERNAL_SUFFIXES) for token in incoming_special) else 0.0
            change_hint = 1.0 if incoming_tokens & CHANGE_HINT_TOKENS else 0.0
            raw_components = {
                "novel_ratio": novel_ratio,
                "anchored_novelty": novel_ratio * change_hint,
                "special_novelty": special_novelty,
                "external_marker": external_marker,
                "change_hint": change_hint,
            }

    contributions = {
        "novel_ratio_contribution": 0.0,
        "anchored_novelty_contribution": 0.0,
        "special_novelty_contribution": 0.0,
        "external_marker_contribution": 0.0,
        "change_hint_contribution": 0.0,
    }
    score = 0.0
    for component, weight in normalized_weights.items():
        contribution = raw_components[component] * weight
        contributions[f"{component}_contribution"] = round(contribution, 4)
        score += contribution

    return {
        "novel_ratio": round(raw_components["novel_ratio"], 4),
        "anchored_novelty": round(raw_components["anchored_novelty"], 4),
        "special_novelty": round(raw_components["special_novelty"], 4),
        "external_marker": round(raw_components["external_marker"], 4),
        "change_hint": round(raw_components["change_hint"], 4),
        **contributions,
        "score": round(min(1.0, score), 4),
    }


def anomaly_score(
    text: str,
    records: list[MemoryRecord],
    weights: dict[str, float] | None = None,
    *,
    formula: str = "legacy",
) -> float:
    return anomaly_score_breakdown(text, records, weights, formula=formula)["score"]


def build_defense_config(
    name: str | None = None,
    *,
    defense_label: str | None = None,
    anomaly_threshold: float | None = None,
    anomaly_component_weights: dict[str, float] | None = None,
    anomaly_formula: str | None = None,
) -> MemoryDefenseConfig:
    normalized_formula = normalize_anomaly_formula(anomaly_formula)
    normalized = (name or "none").strip().lower()
    if normalized in {"", "none", "baseline", "no_defense"}:
        config = MemoryDefenseConfig(name="none", base_name="none")
    elif normalized in {"d1", "d1_provenance", "provenance"}:
        config = MemoryDefenseConfig(name="d1_provenance", base_name="d1_provenance", apply_provenance=True)
    elif normalized in {"d2", "d2_consistency", "consistency"}:
        config = MemoryDefenseConfig(name="d2_consistency", base_name="d2_consistency", apply_consistency=True)
    elif normalized in {"d3", "d3_anomaly", "anomaly"}:
        config = MemoryDefenseConfig(name="d3_anomaly", base_name="d3_anomaly", apply_anomaly_filter=True)
    elif normalized in {"d1_d2_d3", "all", "full"}:
        config = MemoryDefenseConfig(
            name="d1_d2_d3",
            base_name="d1_d2_d3",
            apply_provenance=True,
            apply_consistency=True,
            apply_anomaly_filter=True,
        )
    else:
        raise ValueError(f"unknown defense config: {name}")

    config.anomaly_formula = normalized_formula
    if anomaly_threshold is not None:
        config.anomaly_threshold = float(anomaly_threshold)
    config.anomaly_component_weights = normalize_anomaly_component_weights(
        anomaly_component_weights,
        formula=normalized_formula,
    )
    config.name = make_defense_label(
        config.base_name,
        defense_label=defense_label,
        anomaly_formula=normalized_formula,
        anomaly_threshold=anomaly_threshold,
        anomaly_component_weights=anomaly_component_weights,
    )
    config.defense_metadata = {
        "base_name": config.base_name,
        "anomaly_formula": config.anomaly_formula,
        "anomaly_threshold": round(config.anomaly_threshold, 4),
        "anomaly_component_weights": dict(config.anomaly_component_weights),
    }
    return config


class BaseMemory:
    def __init__(self, defense: MemoryDefenseConfig | None = None) -> None:
        self.records: list[MemoryRecord] = []
        self.defense = defense or build_defense_config("none")

    def write_case(self, case: dict) -> dict[str, Any]:
        raise NotImplementedError

    def retrieve(self, query: str, top_k: int = 3) -> list[MemoryRecord]:
        scored: list[tuple[float, float, float, MemoryRecord]] = []
        for record in self.records:
            raw_score = overlap_score(query, record.text)
            if raw_score <= 0:
                continue
            trust_weight = record.trust_weight if self.defense.apply_provenance else 1.0
            scored.append((raw_score * trust_weight, raw_score, trust_weight, record))
        scored.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
        return [record for _, _, _, record in scored[:top_k]]

    def _new_write_report(self) -> dict[str, Any]:
        return {
            "defense_config": self.defense.name,
            "defense_base_config": self.defense.base_name,
            "defense_metadata": self.defense.defense_metadata,
            "attempted_adversarial_writes": 0,
            "blocked_adversarial_writes": 0,
            "attempted_legitimate_writes": 0,
            "blocked_legitimate_writes": 0,
            "admitted_record_ids": [],
            "blocked_details": [],
        }

    def _block_reasons(self, record: MemoryRecord) -> tuple[list[str], dict[str, float]]:
        reasons: list[str] = []
        signals: dict[str, float] = {}
        if record.source in {"context", "episode_summary"}:
            return reasons, signals

        if self.defense.apply_consistency and self.records:
            score = consistency_conflict_score(record.text, self.records)
            signals["consistency_risk"] = score
            if score >= self.defense.consistency_threshold:
                reasons.append("consistency_check")

        if self.defense.apply_anomaly_filter and self.records:
            breakdown = anomaly_score_breakdown(
                record.text,
                self.records,
                self.defense.anomaly_component_weights,
                formula=self.defense.anomaly_formula,
            )
            signals["anomaly_risk"] = breakdown["score"]
            signals["anomaly_novel_ratio"] = breakdown["novel_ratio"]
            signals["anomaly_anchored_novelty"] = breakdown["anchored_novelty"]
            signals["anomaly_change_hint"] = breakdown["change_hint"]
            signals["anomaly_special_novelty"] = breakdown["special_novelty"]
            signals["anomaly_external_marker"] = breakdown["external_marker"]
            signals["anomaly_novel_ratio_contribution"] = breakdown["novel_ratio_contribution"]
            signals["anomaly_anchored_novelty_contribution"] = breakdown["anchored_novelty_contribution"]
            signals["anomaly_change_hint_contribution"] = breakdown["change_hint_contribution"]
            signals["anomaly_special_novelty_contribution"] = breakdown["special_novelty_contribution"]
            signals["anomaly_external_marker_contribution"] = breakdown["external_marker_contribution"]
            if breakdown["score"] >= self.defense.anomaly_threshold:
                reasons.append("anomaly_filter")

        return reasons, signals

    def _store_record(self, record: MemoryRecord, report: dict[str, Any]) -> None:
        is_adversarial = bool(record.is_poison and record.source not in {"context", "episode_summary"})
        if is_adversarial:
            report["attempted_adversarial_writes"] += 1
        else:
            report["attempted_legitimate_writes"] += 1

        reasons, signals = self._block_reasons(record)
        if reasons:
            if is_adversarial:
                report["blocked_adversarial_writes"] += 1
            else:
                report["blocked_legitimate_writes"] += 1
            report["blocked_details"].append(
                {
                    "record_id": record.record_id,
                    "source": record.source,
                    "stage": record.stage,
                    "reason": "+".join(reasons),
                    "signals": signals,
                }
            )
            return

        record.trust_weight = self.defense.trust_weights.get(record.source, 1.0)
        self.records.append(record)
        report["admitted_record_ids"].append(record.record_id)


class FlatChunkMemory(BaseMemory):
    def write_case(self, case: dict) -> dict[str, Any]:
        report = self._new_write_report()
        for index, text in enumerate(case["context"], start=1):
            self._store_record(
                MemoryRecord(
                    record_id=f"{case['case_id']}:context:{index}",
                    text=text,
                    source="context",
                    stage="context",
                    is_poison=False,
                ),
                report,
            )
        for turn in case["injection_turns"]:
            self._store_record(
                MemoryRecord(
                    record_id=f"{case['case_id']}:{turn['turn_id']}",
                    text=turn["content"],
                    source=case["injection_channel"],
                    stage="injection",
                    is_poison=True,
                ),
                report,
            )
        return report


class FactStoreMemory(BaseMemory):
    def write_case(self, case: dict) -> dict[str, Any]:
        report = self._new_write_report()
        for index, text in enumerate(case["context"], start=1):
            for fact_index, fact in enumerate(split_facts(text), start=1):
                self._store_record(
                    MemoryRecord(
                        record_id=f"{case['case_id']}:context:{index}:{fact_index}",
                        text=fact,
                        source="context",
                        stage="context",
                        is_poison=False,
                    ),
                    report,
                )
        for turn in case["injection_turns"]:
            for fact_index, fact in enumerate(split_facts(turn["content"]), start=1):
                self._store_record(
                    MemoryRecord(
                        record_id=f"{case['case_id']}:{turn['turn_id']}:{fact_index}",
                        text=fact,
                        source=case["injection_channel"],
                        stage="injection",
                        is_poison=True,
                    ),
                    report,
                )
        return report


class HierarchicalNoteMemory(BaseMemory):
    def write_case(self, case: dict) -> dict[str, Any]:
        report = self._new_write_report()
        episode_lines: list[str] = []
        poison_seen = False
        for index, text in enumerate(case["context"], start=1):
            self._store_record(
                MemoryRecord(
                    record_id=f"{case['case_id']}:context:{index}",
                    text=text,
                    source="context",
                    stage="context",
                    is_poison=False,
                ),
                report,
            )
            episode_lines.append(text)
        for turn in case["injection_turns"]:
            content = turn["content"]
            raw_record = MemoryRecord(
                record_id=f"{case['case_id']}:{turn['turn_id']}:raw",
                text=content,
                source=case["injection_channel"],
                stage="injection",
                is_poison=True,
            )
            self._store_record(raw_record, report)
            poison_seen = poison_seen or raw_record.record_id in report["admitted_record_ids"]
            note_record = MemoryRecord(
                record_id=f"{case['case_id']}:{turn['turn_id']}:note",
                text=build_note(content),
                source=case["injection_channel"],
                stage="note",
                is_poison=True,
            )
            self._store_record(note_record, report)
            poison_seen = poison_seen or note_record.record_id in report["admitted_record_ids"]
            episode_lines.append(content)
        self._store_record(
            MemoryRecord(
                record_id=f"{case['case_id']}:episode:summary",
                text=build_note(" ".join(episode_lines)),
                source="episode_summary",
                stage="note",
                is_poison=poison_seen,
            ),
            report,
        )
        return report


def split_facts(text: str) -> Iterable[str]:
    # Split sentence-like facts on punctuation followed by whitespace so dotted
    # entities such as hostnames, emails, and URLs stay intact.
    parts = [part.strip() for part in re.split(FACT_SPLIT_RE, text) if part.strip()]
    return parts or [text.strip()]


def build_note(text: str) -> str:
    tokens = list(normalize_tokens(text))
    selected = sorted(tokens)[:16]
    return "note " + " ".join(selected)


def build_memory(substrate: str, defense: MemoryDefenseConfig | None = None) -> BaseMemory:
    if substrate == "flat_chunk":
        return FlatChunkMemory(defense=defense)
    if substrate == "fact_store":
        return FactStoreMemory(defense=defense)
    if substrate == "hierarchical_notes":
        return HierarchicalNoteMemory(defense=defense)
    raise ValueError(f"unknown substrate: {substrate}")


