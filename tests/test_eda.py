from pathlib import Path
import tempfile
import unittest

import pandas as pd

from hydrowatch_baseline.eda import write_eda_report


class EdaReportTest(unittest.TestCase):
    def test_writes_readiness_and_patch_balance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            preparation = root / "preparation"
            preparation.mkdir()
            pd.DataFrame(
                [
                    {
                        "pair_id": "flood__a",
                        "event_id": "flood",
                        "event_kind": "flood",
                        "s1_ready": True,
                        "s2_ready": False,
                        "s1_coverage_min": 1.0,
                        "flood_ha": 10.0,
                        "water_pre_ha": 20.0,
                        "water_peak_ha": 30.0,
                        "receded_ha": 2.0,
                        "status": "ready",
                    }
                ]
            ).to_csv(preparation / "data_audit.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "event_id": "flood",
                        "event_kind": "flood",
                        "flood_fraction": 0.25,
                        "water_pre_fraction": 0.1,
                        "water_peak_fraction": 0.3,
                    },
                    {
                        "event_id": "flood",
                        "event_kind": "flood",
                        "flood_fraction": 0.0,
                        "water_pre_fraction": 0.0,
                        "water_peak_fraction": 0.0,
                    },
                ]
            ).to_csv(preparation / "patch_manifest.csv", index=False)

            outputs = write_eda_report(preparation, root / "eda")

            self.assertTrue(all(path.exists() for path in outputs.values()))
            summary = outputs["summary"].read_text(encoding="utf-8")
            self.assertIn('"flood_positive_fraction": 0.5', summary)
            self.assertIn("flood__a", outputs["report"].read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
