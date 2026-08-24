from pathlib import Path

import pandas as pd

from analysis.new_solvent_analysis.publication_reporting import (
    display_label,
    render_publication_group,
)


def test_display_label_expands_mixture_composition() -> None:
    assert (
        display_label("acetone_20_n-heptane_80")
        == "acetone 20% / n-heptane 80%"
    )
    assert display_label("p-xylene") == "p-xylene"
    assert display_label("thf_n-heptane") == "THF / n-heptane"


def test_publication_group_outputs_labelled_vector_and_html(tmp_path: Path) -> None:
    metrics = pd.DataFrame(
        {
            "group": ["group1", "group1"],
            "system": ["acetone_10_n-heptane_90", "acetone_20_n-heptane_80"],
            "alpha": [0.8, 1.0],
            "alpha_fit_r2": [0.98, 0.99],
            "alpha_prefactor_A2_ps_alpha": [1.2, 1.5],
            "time_span_ps": [10.0, 10.0],
            "n_frames": [101, 101],
            "mean_disp_A": [0.8, 1.2],
            "mean_force_eV_A": [0.4, 0.5],
            "std_force_eV_A": [0.1, 0.1],
            "final_disp_A": [1.2, 2.1],
            "head_coord_number_4A": [12.0, 14.0],
            "head_tail_coord_ratio": [1.1, 1.3],
            "cosolvent_fraction": [0.1, 0.2],
            "structural_data_status": ["matched", "matched"],
        }
    )
    outputs = render_publication_group(metrics, pd.DataFrame(), "group1", tmp_path)

    names = {path.name for path in outputs}
    assert "group1_labelled_comparison.svg" in names
    assert "group1_labelled_comparison.pdf" in names
    assert "group1_labelled_comparison.png" in names
    assert "group1.html" in names

    svg = (tmp_path / "figures" / "group1_labelled_comparison.svg").read_text(
        encoding="utf-8"
    )
    assert "acetone 10% / n-heptane 90%" in svg
    assert "acetone 20% / n-heptane 80%" in svg
    html = (tmp_path / "reports" / "group1.html").read_text(encoding="utf-8")
    assert "group1_labelled_comparison.png" in html
    assert "600 dpi" in html
