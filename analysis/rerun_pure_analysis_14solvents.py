"""
Re-run the pure-solvent OPA analysis (analyze_pure_solvents.py +
analyze_opa_displacement_dynamics.py + analyze_opa_solvation_structure.py +
build_combined_report.py) over all 14 pure-solvent folders now present under
results/ -- the original 10 plus the 4 newly added (acetone, n-heptane, thf,
toluene).

Writes into results/pure_components_analysis_14solvents/, a folder separate
from the original results/pure_components_analysis/ (which is left untouched
as the round-1..3 analysis of the original 10 solvents).

Known gap: acetone, n-heptane, thf, and toluene only have
<solvent>_opa_motion_force.csv (no <solvent>_md_300K.traj), so the RDF/
solvation-structure pass (analyze_opa_solvation_structure.py) and anything
downstream that depends on it (merged_summary.csv, literature_report.html)
can only include solvents that do have a trajectory file. Those 4 solvents
are automatically skipped ("no trajectory") by that pass and dropped from the
merge; they still appear fully in summary.csv, report.html,
displacement_dynamics_summary.csv, and displacement_dynamics_report.html.
"""

from __future__ import annotations

from pathlib import Path

import analyze_pure_solvents as aps
import analyze_opa_displacement_dynamics as add
import analyze_opa_solvation_structure as aos
import build_combined_report as bcr

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
OUT_DIR = RESULTS_DIR / "pure_components_analysis_14solvents"

NEW_SOLVENTS = ["acetone", "n-heptane", "thf", "toluene"]
ALL_SOLVENTS = aps.PURE_SOLVENTS + NEW_SOLVENTS

# Manual chemistry labels for the 4 new solvents, same convention as
# SOLVENT_CLASS in analyze_pure_solvents.py (not derived from the MD data).
NEW_CLASS = {
    "acetone": "aprotic-polar",
    "n-heptane": "nonpolar",
    "thf": "aprotic-polar",
    "toluene": "nonpolar",
}
CLASS_MAP = {**aps.SOLVENT_CLASS, **NEW_CLASS}


def main() -> None:
    aps.run(
        ALL_SOLVENTS, RESULTS_DIR, OUT_DIR, CLASS_MAP,
        title="Pure-component solvent systems: OPA displacement &amp; force (14 solvents)",
        scope_note=(
            "Only single-component systems are included here; mixtures "
            "(e.g. acetone/n-heptane blends, isopropanol/n-heptane) are excluded. "
            "Includes 4 solvents added after the original round-1..3 analysis: "
            "acetone, n-heptane, thf, toluene."
        ),
    )

    add.main(
        names=ALL_SOLVENTS, results_dir=RESULTS_DIR, out_dir=OUT_DIR,
        title="OPA displacement dynamics: time-averaged MSD and anomalous diffusion exponent (14 solvents)",
    )

    aos.main(names=ALL_SOLVENTS, results_dir=RESULTS_DIR, out_dir=OUT_DIR)

    bcr.main(
        out_dir=OUT_DIR, results_dir=RESULTS_DIR,
        page_title="Pure-solvent OPA solvation analysis (literature-informed, 14 solvents)",
        heading="Solvent effect on OPA solvation: literature-informed re-analysis (14 solvents)",
    )


if __name__ == "__main__":
    main()
