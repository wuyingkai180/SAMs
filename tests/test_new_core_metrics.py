from __future__ import annotations

import unittest

import numpy as np

from analysis.new_solvent_analysis.core_metrics import (
    fit_anomalous_exponent,
    tamsd,
    unwrap_positions,
)


class CoreMetricTests(unittest.TestCase):
    def test_unwrap_removes_periodic_jump(self):
        wrapped = np.array([[9.5, 0, 0], [0.2, 0, 0], [0.9, 0, 0]])
        got = unwrap_positions(wrapped, np.diag([10.0, 10.0, 10.0]))
        np.testing.assert_allclose(got[:, 0], [9.5, 10.2, 10.9])

    def test_tamsd_matches_linear_track(self):
        pos = np.arange(6.0)[:, None] * np.array([[1.0, 0.0, 0.0]])
        np.testing.assert_allclose(tamsd(pos, 3), [1.0, 4.0, 9.0])

    def test_alpha_for_ballistic_track_is_two(self):
        tau = np.arange(1.0, 11.0)
        fit = fit_anomalous_exponent(tau, tau**2)
        self.assertAlmostEqual(fit.alpha, 2.0, places=10)
        self.assertAlmostEqual(fit.r_squared, 1.0, places=10)

    def test_tamsd_rejects_invalid_lag(self):
        with self.assertRaises(ValueError):
            tamsd(np.zeros((3, 3)), 3)


if __name__ == "__main__":
    unittest.main()
