from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

from mempoison.benchmark_schema import load_cases, summarize_cases, validate_cases
from mempoison.live_eval import DEFENSE_CONFIG_CHOICES, evaluate_manifest
from mempoison.metric_formulas import activation_shift_i, delta_i, omega_ij, pipeline_bucket_rates
from mempoison.substrates import MemoryRecord, build_defense_config, overlap_score


REPO_ROOT = Path(__file__).resolve().parents[1]


class PublicReleaseSmokeTest(unittest.TestCase):
    def test_demo_benchmark_validates(self) -> None:
        cases = load_cases(REPO_ROOT / "data" / "benchmark" / "sample_cases.json")
        self.assertEqual(len(cases), 30)
        self.assertEqual(validate_cases(cases), [])
        summary = summarize_cases(cases)
        self.assertEqual(summary["by_difficulty"]["L1_naive"], 10)
        self.assertEqual(summary["by_difficulty"]["L2_compositional"], 10)
        self.assertEqual(summary["by_difficulty"]["L3_sleeper"], 10)
        self.assertEqual(summary["by_channel"]["cross_agent"], 9)
        self.assertEqual(summary["by_channel"]["tool_return"], 12)
        self.assertEqual(summary["by_channel"]["user_input"], 9)

    def test_substrate_overlap_is_bounded(self) -> None:
        score = overlap_score("use encrypted portal", "Use the encrypted internal portal path.")
        self.assertGreater(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_memory_record_dataclass(self) -> None:
        record = MemoryRecord(
            record_id="demo",
            text="Use the encrypted internal portal path.",
            source="context",
            stage="seed",
            is_poison=False,
        )
        self.assertEqual(record.record_id, "demo")
        self.assertFalse(record.is_poison)

    def test_public_defense_surface_is_limited(self) -> None:
        self.assertEqual(
            DEFENSE_CONFIG_CHOICES,
            ["none", "d1_provenance", "d2_consistency", "d3_anomaly", "d1_d2_d3"],
        )
        self.assertEqual(build_defense_config("d1_d2_d3").base_name, "d1_d2_d3")
        with self.assertRaises(ValueError):
            build_defense_config("not_public_combo")

    def test_manifest_feeds_live_eval(self) -> None:
        tmp_path = REPO_ROOT / "workspace" / "test_eval"
        tmp_path.mkdir(parents=True, exist_ok=True)
        spec = importlib.util.spec_from_file_location(
            "build_manifest",
            REPO_ROOT / "scripts" / "build_manifest.py",
        )
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        build_manifest = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(build_manifest)
        cases = load_cases(REPO_ROOT / "data" / "benchmark" / "sample_cases.json")
        rows = build_manifest.build_manifest_records(cases)
        self.assertIn("context", rows[0])
        self.assertIn("injection_turns", rows[0])
        self.assertIn("trigger_tasks", rows[0])

        outputs, summary = evaluate_manifest(
            manifest_rows=rows[:2],
            output_dir=tmp_path / "eval",
            backend="heuristic",
            model_name=None,
            base_url="http://127.0.0.1:8000/v1",
            api_key="EMPTY",
            max_new_tokens=96,
            temperature=0.0,
            disable_thinking=False,
            top_k=3,
            timeout_seconds=1.0,
            defense_config="none",
            defense_label=None,
            anomaly_threshold=None,
            anomaly_component_weights=None,
            anomaly_formula=None,
        )
        self.assertEqual(summary["num_errors"], 0)
        self.assertGreater(len(outputs), 0)
        self.assertIn("behavior_metrics", summary)
        self.assertIn("pipeline_bucket_rates", summary)

    def test_metric_formula_helpers(self) -> None:
        self.assertAlmostEqual(delta_i(0.8, 0.2), 0.6)
        self.assertAlmostEqual(omega_ij(0.9, 0.8, 0.7, 0.2), -0.4)
        self.assertAlmostEqual(activation_shift_i(0.5, 0.1), 0.4)
        rates = pipeline_bucket_rates(
            [
                "write_blocked",
                "admitted_not_retrieved",
                "retrieved_noncausal",
                "retrieved_residual_causal",
            ]
        )
        self.assertEqual(rates["write_block_rate"], 0.25)


if __name__ == "__main__":
    unittest.main()
