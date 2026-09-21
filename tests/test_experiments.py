from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from hydrowatch_baseline.experiments import (
    build_experiment_plan,
    fold_score_pair_ids,
    parse_ablations,
)
from hydrowatch_baseline.pipeline import select_pairs
from hydrowatch_baseline.training import select_training_rows


class ExperimentPlanTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.root = Path(self.temporary.name)
        pd.DataFrame([
            {"pair_id": "flood_b", "event_id": "flood_b", "event_kind": "rain_flood"},
            {"pair_id": "baseline_a", "event_id": "baseline_a", "event_kind": "baseline"},
            {"pair_id": "flood_a", "event_id": "flood_a", "event_kind": "rain_flood"},
        ]).to_csv(self.root / "pairs.csv", index=False)
        self.manifest = self.root / "patch_manifest.csv"
        pd.DataFrame({"event_id": ["flood_a", "flood_b", "baseline_a"]}).to_csv(
            self.manifest, index=False
        )

    def tearDown(self):
        self.temporary.cleanup()

    def test_plan_crosses_ablations_with_non_baseline_events(self):
        plan = build_experiment_plan(
            self.root, self.manifest, ["sar-only", "sar-ndwi"]
        )
        self.assertEqual(len(plan), 4)
        self.assertEqual(set(plan["validation_event"]), {"flood_a", "flood_b"})
        self.assertEqual(
            plan.loc[plan["ablation"] == "sar_only", "optical_features"].iloc[0], ""
        )

    def test_unknown_ablation_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown ablation"):
            parse_ablations(["sar-only", "optical-magic"])

    def test_fold_score_keeps_controls_and_held_out_event(self):
        self.assertEqual(
            fold_score_pair_ids(self.root, "flood_a"),
            ["baseline_a", "flood_a"],
        )

    def test_inference_pairs_are_limited_to_requested_fold(self):
        pairs = pd.read_csv(self.root / "pairs.csv")
        selected = select_pairs(pairs, ["baseline_a", "flood_a"])
        self.assertEqual(selected["pair_id"].tolist(), ["baseline_a", "flood_a"])
        with self.assertRaisesRegex(ValueError, "Unknown pair_id"):
            select_pairs(pairs, ["missing_pair"])

    def test_controls_are_never_used_for_training(self):
        manifest = pd.DataFrame([
            {
                "pair_id": "flood_a", "event_id": "flood_a", "event_kind": "rain_flood",
                "flood_fraction": 0.2, "water_pre_fraction": 0.1, "water_peak_fraction": 0.3,
            },
            {
                "pair_id": "flood_b", "event_id": "flood_b", "event_kind": "rain_flood",
                "flood_fraction": 0.1, "water_pre_fraction": 0.1, "water_peak_fraction": 0.2,
            },
            {
                "pair_id": "baseline_a", "event_id": "baseline_a", "event_kind": "baseline",
                "flood_fraction": 0.0, "water_pre_fraction": 0.1, "water_peak_fraction": 0.1,
            },
        ])
        training, validation = select_training_rows(manifest, "flood_a")
        self.assertEqual(set(training["pair_id"]), {"flood_b"})
        self.assertEqual(set(validation["pair_id"]), {"flood_a", "baseline_a"})
