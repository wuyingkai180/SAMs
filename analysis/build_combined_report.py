"""
Merge the mobility/force summary (analyze_pure_solvents.py) with the
solvation-structure summary (analyze_opa_solvation_structure.py) into one
literature-informed report.

Metrics and rationale, mapped to sources (see report footer for full citations):
  - Diffusion proxy D_eff / effective drag gamma_eff: analogue of the
    "diffusion-controlled adsorption kinetics" framing used for phosphonic
    acid adsorption (ITO order paper) and for solvent comparison in ODPA/
    alumina self-organization MD (RSC PCCP 2017).
  - Head-vs-tail solvent coordination number / RDF: structural analogue of
    the RDF-based solvent-adsorbate analysis in the Cu(111) thiol-SAM
    solvent-polarity study (PMC6017570), applied here to the phosphonate
    head vs. terminal alkyl carbon of the same OPA molecule.
  - solvent_class (protic / aprotic-polar / nonpolar): manual chemistry
    label (not derived from the simulation), used only to help *read* the
    plots -- the same qualitative distinction (polar/protic vs. nonpolar)
    that recurs across all of the papers surveyed as the main lens for
    solvent effects on SAM formation.

Scope caveat (repeated here because it matters for interpretation): these
are single-OPA-molecule-in-bulk-solvent runs, with no substrate and no
second OPA molecule, so this cannot reproduce monolayer-level metrics from
the literature (packing density, tilt angle, gauche-defect population). It
characterizes only the pre-adsorption solvation environment.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

from analyze_pure_solvents import (
    CLUSTER_COLORS, SOLVENT_CLASS, fmt, hbar_chart_svg, scatter_svg,
)
from analyze_opa_solvation_structure import compute_all as compute_solvation_profiles

RESULTS_DIR = Path("/Users/internship/Desktop/test/results")
DEFAULT_OUT_DIR = RESULTS_DIR / "pure_components_analysis"


def read_csv_dicts(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as fh:
        return [
            {k: (float(v) if _is_float(v) else v) for k, v in row.items()}
            for row in csv.DictReader(fh)
        ]


def _is_float(v: str) -> bool:
    try:
        float(v)
        return True
    except ValueError:
        return False


def pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    deny = math.sqrt(sum((y - my) ** 2 for y in ys))
    return num / (denx * deny) if denx and deny else 0.0


def main(
    out_dir: Path | None = None,
    results_dir: Path | None = None,
    names: list[str] | None = None,
    page_title: str = "Pure-solvent OPA solvation analysis (literature-informed)",
    heading: str = "Solvent effect on OPA solvation: literature-informed re-analysis",
    report_name: str = "literature_report.html",
) -> None:
    out_dir = DEFAULT_OUT_DIR if out_dir is None else out_dir
    results_dir = RESULTS_DIR if results_dir is None else results_dir
    mobility = {r["solvent"]: r for r in read_csv_dicts(out_dir / "summary.csv")}
    structure = {r["solvent"]: r for r in read_csv_dicts(out_dir / "solvation_structure_summary.csv")}
    dyn_path = out_dir / "displacement_dynamics_summary.csv"
    dynamics = {r["solvent"]: r for r in read_csv_dicts(dyn_path)} if dyn_path.exists() else {}
    names = [n for n in mobility if n in structure]
    merged = [{**mobility[n], **structure[n], **dynamics.get(n, {})} for n in names]

    merged_path = out_dir / "merged_summary.csv"
    field_order = [
        "solvent", "solvent_class", "cluster", "diffusion_d_eff", "gamma_eff",
        "mean_force_eV_A", "std_force_eV_A", "final_disp_A",
        "head_coord_number_4A", "tail_coord_number_4A", "head_tail_coord_ratio",
        "head_first_peak_r_A", "tail_first_peak_r_A",
        "alpha", "alpha_fit_r2", "d_eff_tamsd", "regime",
    ]
    with merged_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=field_order)
        writer.writeheader()
        for row in merged:
            writer.writerow({k: row.get(k) for k in field_order})
    print(f"Wrote {merged_path}")

    def col(key):
        return [row[key] for row in merged]

    correlations = {
        "head_coord_4A vs D_eff": pearson(col("head_coord_number_4A"), col("diffusion_d_eff")),
        "head_coord_4A vs mean_force": pearson(col("head_coord_number_4A"), col("mean_force_eV_A")),
        "tail_coord_4A vs D_eff": pearson(col("tail_coord_number_4A"), col("diffusion_d_eff")),
        "head_tail_ratio vs D_eff": pearson(col("head_tail_coord_ratio"), col("diffusion_d_eff")),
        "head_coord_4A vs final_disp": pearson(col("head_coord_number_4A"), col("final_disp_A")),
        "alpha vs head_coord_4A": pearson(col("alpha"), col("head_coord_number_4A")),
        "alpha vs mean_force": pearson(col("alpha"), col("mean_force_eV_A")),
        "alpha vs diffusion_d_eff (round 1)": pearson(col("alpha"), col("diffusion_d_eff")),
    }
    for k, v in correlations.items():
        print(f"corr({k}) = {v:.3f}")

    by_class: dict[str, list[dict]] = {}
    for row in merged:
        by_class.setdefault(row["solvent_class"], []).append(row)
    class_order = sorted(by_class, key=lambda c: -len(by_class[c]))
    class_color = {c: f"var(--cluster-{i % 3})" for i, c in enumerate(class_order)}
    point_colors = [class_color[row["solvent_class"]] for row in merged]
    class_legend = [(c, class_color[c]) for c in class_order]

    head_bar = hbar_chart_svg(
        [r["solvent"] for r in sorted(merged, key=lambda r: -r["head_coord_number_4A"])],
        [r["head_coord_number_4A"] for r in sorted(merged, key=lambda r: -r["head_coord_number_4A"])],
        "Solvent atoms within 4 A of the phosphonate headgroup (avg. per head atom)",
        "atoms",
        color="var(--series-1)",
    )
    tail_bar = hbar_chart_svg(
        [r["solvent"] for r in sorted(merged, key=lambda r: -r["tail_coord_number_4A"])],
        [r["tail_coord_number_4A"] for r in sorted(merged, key=lambda r: -r["tail_coord_number_4A"])],
        "Solvent atoms within 4 A of the terminal (C18) alkyl carbon",
        "atoms",
        color="var(--series-1)",
    )
    head_vs_deff = scatter_svg(
        col("diffusion_d_eff"), col("head_coord_number_4A"), col("solvent"),
        "Diffusion proxy D_eff (A^2/rec.ps)", "Head coordination @4A (atoms)",
        colors=point_colors, legend=class_legend,
    )
    head_vs_force = scatter_svg(
        col("mean_force_eV_A"), col("head_coord_number_4A"), col("solvent"),
        "Mean |F_OPA| (eV/A)", "Head coordination @4A (atoms)",
        colors=point_colors, legend=class_legend,
    )
    ratio_vs_deff = scatter_svg(
        col("diffusion_d_eff"), col("head_tail_coord_ratio"), col("solvent"),
        "Diffusion proxy D_eff (A^2/rec.ps)", "Head/tail coordination ratio",
        colors=point_colors, legend=class_legend,
    )

    alpha_vs_head = scatter_svg(
        col("head_coord_number_4A"), col("alpha"), col("solvent"),
        "Head coordination @4A (atoms)", "Anomalous exponent alpha",
        colors=point_colors, legend=class_legend,
    )
    alpha_vs_force = scatter_svg(
        col("mean_force_eV_A"), col("alpha"), col("solvent"),
        "Mean |F_OPA| (eV/A)", "Anomalous exponent alpha",
        colors=point_colors, legend=class_legend,
    )

    _topo, _rows, rdf_panels = compute_solvation_profiles(names, results_dir)
    rdf_panels_html = "\n".join(f'<div class="panel">{p}</div>' for p in rdf_panels)

    table_rows = "\n".join(
        f"<tr><td>{r['solvent']}</td><td>{r['solvent_class']}</td>"
        f"<td>{fmt(r['head_coord_number_4A'],2)}</td><td>{fmt(r['tail_coord_number_4A'],2)}</td>"
        f"<td>{fmt(r['head_tail_coord_ratio'],2)}</td>"
        f"<td>{fmt(r['diffusion_d_eff'],4)}</td><td>{fmt(r['mean_force_eV_A'],3)}</td>"
        f"<td>{fmt(r.get('alpha', float('nan')),3)}</td><td>{r.get('regime','n/a')}</td></tr>"
        for r in sorted(merged, key=lambda r: r["solvent"])
    )

    corr_rows = "\n".join(
        f"<tr><td>{k}</td><td>{fmt(v,3)}</td></tr>" for k, v in correlations.items()
    )

    html = f'''<!doctype html>
<html data-theme="light">
<head>
<meta charset="utf-8">
<title>{page_title}</title>
<style>
  :root {{
    --surface-1: #fcfcfb; --page: #f9f9f7; --text-primary: #0b0b0b;
    --text-secondary: #52514e; --border: rgba(11,11,11,0.10);
    --series-1: {CLUSTER_COLORS[0][0]};
    --cluster-0: {CLUSTER_COLORS[0][0]}; --cluster-1: {CLUSTER_COLORS[1][0]}; --cluster-2: {CLUSTER_COLORS[2][0]};
  }}
  :root[data-theme="dark"] {{
    --surface-1: #1a1a19; --page: #0d0d0d; --text-primary: #ffffff;
    --text-secondary: #c3c2b7; --border: rgba(255,255,255,0.10);
    --series-1: {CLUSTER_COLORS[0][1]};
    --cluster-0: {CLUSTER_COLORS[0][1]}; --cluster-1: {CLUSTER_COLORS[1][1]}; --cluster-2: {CLUSTER_COLORS[2][1]};
  }}
  body {{ background: var(--page); color: var(--text-primary);
          font-family: system-ui, -apple-system, "Segoe UI", sans-serif; margin: 0; padding: 24px; }}
  h1 {{ font-size: 18px; }}
  h2 {{ font-size: 14px; color: var(--text-secondary); margin-top: 32px; }}
  .row {{ display: flex; gap: 16px; flex-wrap: wrap; }}
  .grid {{ display: grid; grid-template-columns: repeat(5, 1fr); gap: 8px; }}
  .panel {{ background: var(--surface-1); border: 1px solid var(--border); border-radius: 6px; padding: 4px; }}
  table {{ border-collapse: collapse; font-size: 12px; margin-top: 8px; }}
  th, td {{ padding: 4px 10px; border-bottom: 1px solid var(--border); text-align: right; }}
  th:first-child, td:first-child {{ text-align: left; }}
  #theme-toggle {{ float: right; font-size: 12px; }}
  .caveat {{ font-size: 12px; color: var(--text-secondary); max-width: 760px; }}
  .sources {{ font-size: 11px; color: var(--text-secondary); max-width: 760px; }}
</style>
</head>
<body>
<button id="theme-toggle" onclick="
  const r = document.documentElement;
  r.setAttribute('data-theme', r.getAttribute('data-theme') === 'dark' ? 'light' : 'dark');
">Toggle dark mode</button>
<h1>{heading}</h1>
<p class="caveat">
This adds two literature-standard evaluation angles on top of the original
displacement/force analysis (<a href="report.html">report.html</a>):
(1) <b>head-vs-tail solvent coordination number</b>, the structural analogue
of RDF-based solvent-adsorbate analysis used for alkanethiol/Cu(111) SAM
solvent studies, applied here to the phosphonate headgroup (P + 3 O + 2
acidic H, identified from opa.vasp bond connectivity) vs. the terminal
(C18) alkyl carbon; (2) explicit correlation between that structural
coordination and the diffusion/force metrics from the first pass.
<b>Scope caveat:</b> each run is one OPA molecule free in bulk periodic
solvent, no substrate, no second OPA molecule -- so packing density, tilt
angle, and gauche-defect metrics used for assembled monolayers in the
literature are not measurable from this data; everything here characterizes
the pre-adsorption solvation environment only. solvent_class is a manual
chemistry label, not derived from the simulation.
</p>

<h2>Radial distribution: solvent around headgroup (blue) vs. terminal alkyl carbon (orange)</h2>
<div class="grid">{rdf_panels_html}</div>

<h2>Head-group coordination (sorted)</h2>
{head_bar}

<h2>Tail terminal-carbon coordination (sorted)</h2>
{tail_bar}

<h2>Head coordination vs. diffusion proxy (colored by solvent class)</h2>
<div class="row">{head_vs_deff}</div>

<h2>Head coordination vs. mean force (colored by solvent class)</h2>
<div class="row">{head_vs_force}</div>

<h2>Head/tail coordination ratio vs. diffusion proxy</h2>
<div class="row">{ratio_vs_deff}</div>

<h2>Anomalous diffusion exponent (alpha) vs. head coordination</h2>
<p class="caveat">alpha from time-averaged MSD, see
<a href="displacement_dynamics_report.html">displacement_dynamics_report.html</a>
for the full method and per-solvent log-log fits.</p>
<div class="row">{alpha_vs_head}</div>

<h2>Anomalous diffusion exponent (alpha) vs. mean force</h2>
<div class="row">{alpha_vs_force}</div>

<h2>Pearson correlations</h2>
<table>
<tr><th>pair</th><th>r</th></tr>
{corr_rows}
</table>

<h2>Merged summary table</h2>
<table>
<tr><th>solvent</th><th>class</th><th>head coord@4A</th><th>tail coord@4A</th>
<th>head/tail ratio</th><th>D_eff</th><th>mean |F| (eV/A)</th><th>alpha</th><th>regime</th></tr>
{table_rows}
</table>

<h2>Sources consulted</h2>
<p class="sources">
Theoretical Insights into the Solvent Polarity Effect on the Quality of
Self-Assembled N-Octadecanethiol Monolayers on Cu(111) Surfaces --
<a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC6017570/">PMC6017570</a>.
Molecular dynamics simulations of phosphonic acid-aluminum oxide
self-organization and their evolution into ordered monolayers, Phys. Chem.
Chem. Phys. 2017 --
<a href="https://pubs.rsc.org/en/content/articlelanding/2017/cp/c6cp08681k">c6cp08681k</a>.
Characterizing the Molecular Order of Phosphonic Acid Self-Assembled
Monolayers on Indium Tin Oxide Surfaces, Langmuir --
<a href="https://pubmed.ncbi.nlm.nih.gov/21863828/">PMID 21863828</a>.
Solvent effect on the formation of self-assembled monolayer on DLC surface
between n-hexane and Vertrel XF --
<a href="https://www.sciencedirect.com/science/article/abs/pii/S0169433208006363">ScienceDirect</a>.
</p>
</body>
</html>'''

    report_path = out_dir / report_name
    report_path.write_text(html, encoding="utf-8")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
