from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin

from hydrowatch_baseline.dataset import audit_dataset, build_event_folds, build_patch_manifest


class DatasetPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.root = Path(self.temporary.name)
        references = self.root / "reference_masks"
        references.mkdir()
        pairs = []
        for index, event in enumerate(("event_a", "event_b")):
            pair_id = f"{event}__aoi"
            path = references / f"reference_{pair_id}.tif"
            masks = np.zeros((5, 5, 5), dtype=np.uint8)
            masks[0, 0:2, 0:2] = 1
            masks[1, 0:2, :] = 1
            masks[2, :, 0:2] = 1
            with rasterio.open(
                path, "w", driver="GTiff", width=5, height=5, count=5,
                dtype="uint8", crs="EPSG:32652", transform=from_origin(0, 50, 10, 10),
            ) as target:
                target.write(masks)
            pairs.append({
                "pair_id": pair_id,
                "event_id": event,
                "event_kind": "rain_flood",
                "aoi_id": "aoi",
                "rasters_dir": f"rasters/{pair_id}",
                "reference_mask": str(path.relative_to(self.root)),
            })
        pd.DataFrame(pairs).to_csv(self.root / "pairs.csv", index=False)

    def tearDown(self):
        self.temporary.cleanup()

    def test_event_folds_never_split_one_event(self):
        folds = build_event_folds(self.root)
        self.assertEqual(folds.groupby("event_id")["fold"].nunique().max(), 1)
        self.assertEqual(folds["fold"].nunique(), 2)

    def test_patch_manifest_covers_bottom_and_right_edges(self):
        manifest = build_patch_manifest(self.root, patch_size=2, stride=2)
        for pair_id, rows in manifest.groupby("pair_id"):
            self.assertIn(3, set(rows["row"]), pair_id)
            self.assertIn(3, set(rows["col"]), pair_id)
            self.assertEqual(len(rows), 9)
        first = manifest[
            (manifest["pair_id"] == "event_a__aoi")
            & (manifest["row"] == 0)
            & (manifest["col"] == 0)
        ].iloc[0]
        self.assertEqual(first["flood_fraction"], 1.0)
        self.assertEqual(first["water_pre_fraction"], 1.0)
        self.assertEqual(first["water_peak_fraction"], 1.0)

    def test_audit_rejects_almost_empty_sentinel_exports(self):
        pairs = pd.read_csv(self.root / "pairs.csv")
        profile = {
            "driver": "GTiff", "width": 5, "height": 5, "dtype": "float32",
            "crs": "EPSG:32652", "transform": from_origin(0, 50, 10, 10), "nodata": -9999.0,
        }
        for index, pair in pairs.iterrows():
            directory = self.root / pair["rasters_dir"]
            directory.mkdir(parents=True)
            with rasterio.open(directory / "AUX_terrain_gsw.tif", "w", count=6, **profile) as target:
                target.write(np.zeros((6, 5, 5), dtype="float32"))
            values = np.full((3, 5, 5), -15.0, dtype="float32")
            if index == 0:
                values[:] = -9999.0
                values[:, 0, 0] = -15.0
            for window in ("pre", "peak"):
                with rasterio.open(directory / f"S1_{window}_2021-01-01.tif", "w", count=3, **profile) as target:
                    target.write(values)

        audit = audit_dataset(self.root).set_index("pair_id")

        self.assertFalse(bool(audit.loc["event_a__aoi", "s1_ready"]))
        self.assertIn("coverage", audit.loc["event_a__aoi", "problems"])
        self.assertTrue(bool(audit.loc["event_b__aoi", "s1_ready"]))


if __name__ == "__main__":
    unittest.main()
