from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from hydrowatch_baseline.experiments import (
    build_experiment_plan,
    fold_score_pair_ids,
    parse_ablations,
)


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
