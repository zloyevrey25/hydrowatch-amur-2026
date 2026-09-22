import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
import json

import numpy as np
import pandas as pd

from hydrowatch_baseline.metric import area_convergence, score_submission, write_submission_report
from hydrowatch_baseline.sturm import INPUT_CHANNELS, build_eight_channel_input


class AreaConvergenceTests(unittest.TestCase):
    def test_exact_area_scores_one(self):
        self.assertEqual(area_convergence(250.0, 250.0, 50.0), 1.0)

    def test_small_reference_uses_floor(self):
        self.assertAlmostEqual(area_convergence(0.0, 25.0, 50.0), 0.5)

    def test_large_error_is_clipped_at_zero(self):
        self.assertEqual(area_convergence(1000.0, 100.0, 50.0), 0.0)


class ScoreReportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / "reference_masks").mkdir()
        pd.DataFrame([
            {
                "pair_id": "event_pair",
                "event_id": "event_a",
                "event_kind": "rain_flood",
                "aoi_id": "aoi_1",
                "aoi_km2": 10.0,
            },
            {
                "pair_id": "control_pair",
                "event_id": "baseline_a",
                "event_kind": "baseline",
                "aoi_id": "aoi_1",
                "aoi_km2": 10.0,
            },
        ]).to_csv(self.root / "pairs.csv", index=False)
        for pair_id, stats in {
            "event_pair": {"flood_ha": 100.0, "water_pre_ha": 300.0, "water_peak_ha": 500.0},
            "control_pair": {"flood_ha": 20.0, "water_pre_ha": 300.0, "water_peak_ha": 300.0},
        }.items():
            (self.root / "reference_masks" / f"reference_{pair_id}.json").write_text(
                json.dumps({"stats": stats}), encoding="utf-8"
            )
        self.submission = self.root / "submission.csv"
        pd.DataFrame([
            {
                "pair_id": "event_pair",
                "flood_ha": 75.0,
                "water_pre_ha": 260.0,
                "water_peak_ha": 550.0,
            },
            {
                "pair_id": "control_pair",
                "flood_ha": 25.0,
                "water_pre_ha": 300.0,
                "water_peak_ha": 300.0,
            },
        ]).to_csv(self.submission, index=False)

    def tearDown(self):
        self.temporary.cleanup()

    def test_report_writes_summary_and_pair_diagnostics(self):
        output_dir = self.root / "report"
        outputs = write_submission_report(self.submission, self.root, output_dir)
        self.assertTrue(outputs["summary"].exists())
        self.assertTrue(outputs["details"].exists())
        summary = json.loads(outputs["summary"].read_text(encoding="utf-8"))
        self.assertAlmostEqual(summary["q_flood"], 0.75)
        details = pd.read_csv(outputs["details"])
        event = details[details["pair_id"] == "event_pair"].iloc[0]
        self.assertEqual(event["flood_error_ha"], -25.0)
        self.assertAlmostEqual(event["water_peak_quality"], 0.9)
        control = details[details["pair_id"] == "control_pair"].iloc[0]
        self.assertEqual(control["control_excess_flood_ha"], 5.0)

    def test_unknown_pair_id_is_rejected(self):
        bad = self.root / "bad_submission.csv"
        pd.DataFrame([
            {
                "pair_id": "event_pair",
                "flood_ha": 75.0,
                "water_pre_ha": 260.0,
                "water_peak_ha": 550.0,
            },
            {
                "pair_id": "unknown",
                "flood_ha": 0.0,
                "water_pre_ha": 0.0,
                "water_peak_ha": 0.0,
            },
        ]).to_csv(bad, index=False)
        with self.assertRaisesRegex(ValueError, "unknown pair_id"):
            score_submission(bad, self.root)

    def test_score_can_be_limited_to_held_out_event_and_controls(self):
        score = score_submission(self.submission, self.root, ["event_pair", "control_pair"])
        self.assertAlmostEqual(score.q_flood, 0.75)


class MultimodalInputTests(unittest.TestCase):
    def test_builds_channels_in_documented_order(self):
        values = [np.full((2, 3), value, dtype=np.float32) for value in range(8)]
        config = {
            "normalization": {
                "vv_min_db": 0.0, "vv_max_db": 10.0,
                "vh_min_db": 0.0, "vh_max_db": 10.0,
            },
            "model": {"missing_optical_value": -1.0},
        }
        result = build_eight_channel_input(*values, config)
        self.assertEqual(result.shape, (2, 3, len(INPUT_CHANNELS)))
        np.testing.assert_allclose(result[0, 0, :4], [0.0, 0.1, 0.2, 0.3])
        np.testing.assert_allclose(result[0, 0, 4:], [1.0, 1.0, 1.0, 1.0])

    def test_missing_optical_has_explicit_sentinel_value(self):
        sar = [np.zeros((1, 1), dtype=np.float32) for _ in range(4)]
        optical = [np.full((1, 1), np.nan, dtype=np.float32) for _ in range(4)]
        config = {
            "normalization": {
                "vv_min_db": -30.0, "vv_max_db": 10.0,
                "vh_min_db": -30.0, "vh_max_db": 10.0,
            },
            "model": {"missing_optical_value": -1.0},
        }
        result = build_eight_channel_input(*sar, *optical, config)
        np.testing.assert_array_equal(result[..., 4:], -1.0)


if __name__ == "__main__":
    unittest.main()
