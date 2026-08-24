import tempfile
import unittest
from pathlib import Path

import pandas as pd


class ReportingTests(unittest.TestCase):
    def setUp(self):
        self._temp = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self._temp.name)

    def tearDown(self):
        self._temp.cleanup()

    def test_report_contains_scope_and_uncertainty_language(self):
        from analysis.new_solvent_analysis.reporting import build_combined_markdown

        context = {
            "system_count": 27,
            "headline_findings": [],
            "limitations": ["每个条件只有单条轨迹", "模拟描述预吸附溶剂化"],
        }
        text = build_combined_markdown(context)
        self.assertIn("预吸附", text)
        self.assertIn("单条轨迹", text)
        self.assertNotIn("显著提高 SAM 生长", text)

    def test_all_figure_exports_are_nonempty(self):
        from analysis.new_solvent_analysis.reporting import render_group1_figures

        frame = pd.DataFrame(
            {
                "system": [
                    "acetone_1_n-heptane_99",
                    "acetone_5_n-heptane_95",
                ],
                "cosolvent_fraction": [0.01, 0.05],
                "alpha": [0.8, 0.9],
                "mean_force_eV_A": [0.4, 0.5],
            }
        )
        paths = render_group1_figures(frame, self.tmpdir)
        self.assertEqual({path.suffix for path in paths}, {".png", ".pdf", ".svg"})
        self.assertTrue(all(path.exists() and path.stat().st_size > 0 for path in paths))


if __name__ == "__main__":
    unittest.main()
