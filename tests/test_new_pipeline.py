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

    def test_combined_html_embeds_figures_and_structured_text(self):
        from analysis.new_solvent_analysis.reporting import render_combined_report

        context = {
            "system_count": 27,
            "headline_findings": ["浓度趋势为非单调。"],
            "limitations": ["每个条件只有单条轨迹", "模拟描述预吸附溶剂化"],
        }
        paths = render_combined_report(context, self.tmpdir)
        html_path = next(path for path in paths if path.suffix == ".html")
        text = html_path.read_text(encoding="utf-8")
        self.assertIn("<img", text)
        self.assertIn("../figures/group1_concentration_response.png", text)
        self.assertIn("<h2>主要结果</h2>", text)
        self.assertNotIn("<pre>", text)

    def test_groupwise_index_explicitly_rejects_cross_group_ranking(self):
        from analysis.new_solvent_analysis.reporting import build_groupwise_index_markdown

        text = build_groupwise_index_markdown({"group1": 7, "group2": 13, "group3": 7})
        self.assertIn("不进行跨 Group", text)
        self.assertIn("每组只在自身实验设计范围内", text)
        self.assertIn("Group 1", text)
        self.assertIn("Group 2", text)
        self.assertIn("Group 3", text)

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

    def test_mixture_enrichment_figure_exports(self):
        from analysis.new_solvent_analysis.reporting import render_enrichment_figure

        frame = pd.DataFrame(
            {
                "group": ["group3"],
                "system": ["thf_n-heptane"],
                "cosolvent": ["thf"],
                "cosolvent_fraction": [0.05],
                "bulk_molecule_fraction__thf": [0.085],
                "head_enrichment_4A__thf": [2.56],
                "tail_enrichment_4A__thf": [0.0],
            }
        )
        paths = render_enrichment_figure(frame, self.tmpdir)
        self.assertTrue(all(path.exists() and path.stat().st_size > 0 for path in paths))

    def test_rdf_series_keep_same_named_systems_in_different_groups_separate(self):
        from analysis.new_solvent_analysis.reporting import _iter_rdf_series

        profiles = pd.DataFrame(
            {
                "group": ["group1", "group1", "group3", "group3"],
                "system": ["copy", "copy", "copy", "copy"],
                "species": ["all"] * 4,
                "r_A": [0.1, 0.2, 0.1, 0.2],
                "g_head": [0.0, 1.0, 0.0, 2.0],
                "g_tail": [0.0, 1.0, 0.0, 2.0],
            }
        )
        series = _iter_rdf_series(profiles, max_per_group=2)
        self.assertEqual([len(item[1]) for item in series], [2, 2])

    def test_pure_solvent_svg_contains_class_legend(self):
        from analysis.new_solvent_analysis.reporting import render_group2_figures

        frame = pd.DataFrame(
            {
                "system": ["a", "b", "c"],
                "alpha": [0.7, 0.8, 0.9],
                "mean_force_eV_A": [0.4, 0.5, 0.6],
                "solvent_class": ["aprotic-polar", "protic", "nonpolar"],
            }
        )
        paths = render_group2_figures(frame, self.tmpdir)
        svg = next(path for path in paths if path.suffix == ".svg").read_text(encoding="utf-8")
        self.assertIn("aprotic-polar", svg)
        self.assertIn("protic", svg)
        self.assertIn("nonpolar", svg)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self._temp = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self._temp.name)

    def tearDown(self):
        self._temp.cleanup()

    def test_dry_run_validates_all_inputs_without_writing_results(self):
        from analysis.new_solvent_analysis.pipeline import run_analysis

        output = self.tmpdir / "out"
        summary = run_analysis(
            Path("data"),
            output,
            Path("analysis/literature/sams_solvent_methods.json"),
            dry_run=True,
        )
        self.assertEqual(summary.system_count, 27)
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
