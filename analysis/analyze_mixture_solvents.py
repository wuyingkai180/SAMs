"""
Apply the same three-pass analysis used for the pure solvents to a set of
binary mixed-solvent OPA systems:

  1. analyze_pure_solvents.run()      -- displacement/force stats, PCA + KMeans
  2. analyze_opa_solvation_structure  -- head/tail RDF + coordination number
  3. analyze_opa_displacement_dynamics -- TAMSD + anomalous diffusion exponent
  4. build_combined_report            -- merge all three + correlations

"Same method" is taken literally: solvent atoms are still one undifferentiated
group in the RDF/coordination step (no split between e.g. acetone and
n-heptane atoms). A species-resolved RDF (does the polar co-solvent
preferentially solvate the phosphonate head while the alkane diluent sits
near the tail?) would be a genuinely different, more involved analysis and is
left for a future round if wanted -- it is flagged, not silently done.

Systems included: all binary solvent-ratio folders under results/ that have
both a *_opa_motion_force.csv and a *_md_300K.traj (i.e. completed runs).
Excluded: `thf_toluene` (no analysis CSV -- run never completed) and
`wrongacetone_40_n-heptane_60` (the "wrong" prefix suggests a discarded/
duplicate run of acetone_40_n-heptane_60; excluded rather than silently
treated as valid data -- flag to the user if this one should be included).
"""

from pathlib import Path

import analyze_opa_displacement_dynamics as aodd
import analyze_opa_solvation_structure as aoss
import analyze_pure_solvents as aps
import build_combined_report as bcr

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
OUT_DIR = RESULTS_DIR / "mixture_components_analysis"

MIXTURE_NAMES = [
    "acetone_1_n-heptane_99",
    "acetone_10_n-heptane_90",
    "acetone_20_n-heptane_80",
    "acetone_30_n-heptane_70",
    "acetone_50_n-heptane_50",
    "isopropanol_5_n-heptane_95",
]

# Manual label (co-solvent identity + its target volume fraction), the mixture
# analogue of SOLVENT_CLASS in analyze_pure_solvents.py. Not derived from the
# simulation; not used as a clustering feature, only to annotate/read plots.
MIXTURE_LABEL = {
    "acetone_1_n-heptane_99": "acetone 1%",
    "acetone_10_n-heptane_90": "acetone 10%",
    "acetone_20_n-heptane_80": "acetone 20%",
    "acetone_30_n-heptane_70": "acetone 30%",
    "acetone_50_n-heptane_50": "acetone 50%",
    "isopropanol_5_n-heptane_95": "isopropanol 5%",
}


def main() -> None:
    aps.run(
        MIXTURE_NAMES, RESULTS_DIR, OUT_DIR, MIXTURE_LABEL,
        title="Mixed-solvent systems: OPA displacement &amp; force",
        scope_note=(
            "Binary acetone/n-heptane and isopropanol/n-heptane mixtures at "
            "the labeled minority-component volume fraction; pure "
            "single-component systems are analyzed separately "
            "(see ../pure_components_analysis/)."
        ),
    )
    aoss.main(MIXTURE_NAMES, RESULTS_DIR, OUT_DIR)
    aodd.main(
        MIXTURE_NAMES, RESULTS_DIR, OUT_DIR,
        title="Mixed-solvent systems: displacement dynamics (TAMSD / alpha)",
    )
    bcr.main(
        OUT_DIR, RESULTS_DIR, MIXTURE_NAMES,
        page_title="Mixed-solvent OPA solvation analysis (literature-informed)",
        heading="Solvent-mixture effect on OPA solvation: literature-informed re-analysis",
        report_name="literature_report.html",
    )


if __name__ == "__main__":
    main()
