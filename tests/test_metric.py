import unittest
import numpy as np

from hydrowatch_baseline.metric import area_convergence
from hydrowatch_baseline.sturm import INPUT_CHANNELS, build_eight_channel_input


class AreaConvergenceTests(unittest.TestCase):
    def test_exact_area_scores_one(self):
        self.assertEqual(area_convergence(250.0, 250.0, 50.0), 1.0)

    def test_small_reference_uses_floor(self):
        self.assertAlmostEqual(area_convergence(0.0, 25.0, 50.0), 0.5)

    def test_large_error_is_clipped_at_zero(self):
        self.assertEqual(area_convergence(1000.0, 100.0, 50.0), 0.0)


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
