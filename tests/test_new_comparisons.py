from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from analysis.new_solvent_analysis.comparisons import (
    build_group3_comparison,
    collapse_duplicate_trajectories,
    pearson_spearman,
)


class ComparisonTests(unittest.TestCase):
    def test_monotonic_series_has_unit_rank_correlation(self):
        corr = pearson_spearman(np.arange(5.0), np.arange(5.0) ** 3)
        self.assertGreater(corr.pearson, 0.90)
        self.assertAlmostEqual(corr.spearman, 1.0, places=12)

    def test_group3_core_excludes_pure_and_alternate_base_rows(self):
        frame = pd.DataFrame(
            {
                "system": [
                    "acetone_5_n-heptane_95",
                    "isopropanol_5_n-heptane_95",
                    "thf_n-heptane",
                    "toluene_n-heptane",
                    "thf_toluene",
                    "ethyl_acetate",
                    "propylene_carbonate",
                ],
                "role": ["n-heptane-base-composition"] * 4
                + ["alternate-base-composition", "pure-reference", "pure-reference"],
                "alpha": np.arange(7.0),
            }
        )
        result = build_group3_comparison(frame)
        self.assertEqual(
            set(result.core_rows.system),
            {
                "acetone_5_n-heptane_95",
                "isopropanol_5_n-heptane_95",
                "thf_n-heptane",
                "toluene_n-heptane",
            },
        )

    def test_duplicate_checksum_is_not_counted_as_replicate(self):
        frame = pd.DataFrame(
            {"system": ["copy_a", "copy_b"], "sha256": ["same-hash", "same-hash"]}
        )
        result = collapse_duplicate_trajectories(frame)
        self.assertEqual(result.independent_trajectory.nunique(), 1)


if __name__ == "__main__":
    unittest.main()
