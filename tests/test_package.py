from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin

from hydrowatch_baseline.package import validate_submission_package


class PackageValidationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.data_root = self.root / "data"
        self.package_dir = self.root / "package"
        (self.data_root / "reference_masks").mkdir(parents=True)
        (self.package_dir / "predictions").mkdir(parents=True)
        self.pair_id = "event__aoi"
        pd.DataFrame([{
            "pair_id": self.pair_id,
            "aoi_km2": 0.0004,
            "reference_mask": f"reference_masks/reference_{self.pair_id}.tif",
        }]).to_csv(self.data_root / "pairs.csv", index=False)
        profile = {
            "driver": "GTiff", "height": 2, "width": 2, "count": 1,
            "dtype": "uint8", "crs": "EPSG:32652",
            "transform": from_origin(100, 100, 10, 10), "nodata": 0,
        }
        with rasterio.open(
            self.data_root / "reference_masks" / f"reference_{self.pair_id}.tif", "w", **profile
        ) as target:
            target.write(np.zeros((2, 2), dtype="uint8"), 1)
        with rasterio.open(
            self.package_dir / "predictions" / f"{self.pair_id}_flood.tif", "w", **profile
        ) as target:
            target.write(np.array([[1, 0], [0, 0]], dtype="uint8"), 1)
        pd.DataFrame([{
            "pair_id": self.pair_id,
            "flood_ha": 0.01,
            "water_pre_ha": 0.01,
            "water_peak_ha": 0.02,
        }]).to_csv(self.package_dir / "submission.csv", index=False)

    def tearDown(self):
        self.temporary.cleanup()

    def test_accepts_consistent_package(self):
        result = validate_submission_package(self.data_root, self.package_dir)
        self.assertTrue(result.valid, result.errors)
        self.assertEqual(result.masks, 1)

    def test_rejects_semantic_and_raster_errors(self):
        submission = pd.read_csv(self.package_dir / "submission.csv")
        submission.loc[0, "flood_ha"] = 0.04
        submission.loc[0, "water_peak_ha"] = 0.01
        submission.to_csv(self.package_dir / "submission.csv", index=False)
        mask_path = self.package_dir / "predictions" / f"{self.pair_id}_flood.tif"
        with rasterio.open(mask_path, "r+") as target:
            target.write(np.full((2, 2), 2, dtype="uint8"), 1)

        result = validate_submission_package(self.data_root, self.package_dir)

        self.assertFalse(result.valid)
        self.assertTrue(any("flood_ha exceeds" in error for error in result.errors))
        self.assertTrue(any("non-binary" in error for error in result.errors))


if __name__ == "__main__":
    unittest.main()
