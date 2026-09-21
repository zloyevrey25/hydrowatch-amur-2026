import unittest

from hydrowatch_baseline.metric import area_convergence


class AreaConvergenceTests(unittest.TestCase):
    def test_exact_area_scores_one(self):
        self.assertEqual(area_convergence(250.0, 250.0, 50.0), 1.0)

    def test_small_reference_uses_floor(self):
        self.assertAlmostEqual(area_convergence(0.0, 25.0, 50.0), 0.5)

    def test_large_error_is_clipped_at_zero(self):
        self.assertEqual(area_convergence(1000.0, 100.0, 50.0), 0.0)


if __name__ == "__main__":
    unittest.main()

