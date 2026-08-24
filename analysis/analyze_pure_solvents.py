"""
Analyze OPA motion/force CSVs for single-component (non-mixture) solvent systems
produced by jupyter_copy_cell.py / mix.py.

Reads results/<solvent>/<solvent>_opa_motion_force.csv for each pure-solvent
folder, computes summary statistics, and writes:
  results/pure_components_analysis/summary.csv
  results/pure_components_analysis/report.html
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from _common import GRID, MUTED, TEXT_PRIMARY, TEXT_SECONDARY, pearson

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
OUT_DIR = RESULTS_DIR / "pure_components_analysis"

# Folders under results/ that hold a single solvent component (no mixture ratio
# in the name, and not a comparison artifact from the mixture workflow).
PURE_SOLVENTS = [
    "acetonitrile",
    "propylene_carbonate",
    "isopropanol",
    "CPME",
    "ethyl_acetate",
    "DMC",
    "p-xylene",
    "cyclohexane",
    "ethanol",
    "methanol",
]

# Manually assigned from standard solvent chemistry (H-bond donor ability /
# polarity), NOT derived from the MD data. Used only to annotate/interpret
# clusters, never as a clustering feature itself.
SOLVENT_CLASS = {
    "acetonitrile": "aprotic-polar",
    "propylene_carbonate": "aprotic-polar",
    "isopropanol": "protic",
    "CPME": "aprotic-polar",
    "ethyl_acetate": "aprotic-polar",
    "DMC": "aprotic-polar",
    "p-xylene": "nonpolar",
    "cyclohexane": "nonpolar",
    "ethanol": "protic",
    "methanol": "protic",
}

KB_EV_K = 8.617333262e-5
TEMPERATURE_K = 300.0
KT_EV = KB_EV_K * TEMPERATURE_K

COLOR = "#2a78d6"
COLOR_DARK = "#3987e5"
# (light, dark) pairs for the categorical slots actually used here (<=3 series,
# validated all-pairs by scripts/validate_palette.js in the dataviz skill).
CLUSTER_COLORS = [("#2a78d6", "#3987e5"), ("#eb6834", "#d95926"), ("#1baf7a", "#199e70")]


def read_csv(path: Path) -> dict[str, list[float]]:
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        cols: dict[str, list[float]] = {name: [] for name in reader.fieldnames}
        for row in reader:
            for key, value in row.items():
                cols[key].append(float(value))
    return cols


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def stdev(xs: list[float]) -> float:
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def linear_slope(xs: list[float], ys: list[float]) -> float:
    """Least-squares slope of ys vs xs."""
    mx, my = mean(xs), mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    return num / den if den else 0.0


def compute_metrics(name: str, cols: dict[str, list[float]], class_map: dict[str, str] | None = None) -> dict:
    class_map = SOLVENT_CLASS if class_map is None else class_map
    t = cols["time_ps"]
    disp = cols["OPA_disp_norm_A"]
    force = cols["F_OPA_norm_eV_A"]
    msd = [d * d for d in disp]
    # 3D Einstein relation: MSD(t) = 6*D*t -> D = slope(MSD vs t) / 6.
    # From a single trajectory (no ensemble averaging), this is a noisy,
    # comparative mobility proxy, not a publication-grade diffusion coefficient.
    diffusion_slope = linear_slope(t, msd)
    diffusion_d_eff = diffusion_slope / 6.0
    return {
        "solvent": name,
        "n_frames": len(t),
        "recorded_time_span": t[-1] - t[0],
        "final_disp_A": disp[-1],
        "mean_disp_A": mean(disp),
        "max_disp_A": max(disp),
        "mean_force_eV_A": mean(force),
        "std_force_eV_A": stdev(force),
        "max_force_eV_A": max(force),
        "disp_slope_A_per_unit_time": linear_slope(t, disp),
        "force_disp_correlation": pearson(force, disp),
        "diffusion_d_eff": diffusion_d_eff,
        # Stokes-Einstein-style effective drag: gamma = kT / D. Only meaningful
        # when D_eff > 0 (net diffusive drift, not net caging).
        "gamma_eff": (KT_EV / diffusion_d_eff) if diffusion_d_eff > 0 else float("nan"),
        "solvent_class": class_map.get(name, "unknown"),
        "t": t,
        "disp": disp,
        "force": force,
    }


def fmt(x: float, nd: int = 3) -> str:
    return f"{x:.{nd}f}"


# ---------------------------------------------------------------------------
# SVG helpers
# ---------------------------------------------------------------------------


def line_chart_svg(
    t: list[float],
    y: list[float],
    title: str,
    y_label: str,
    color: str = COLOR,
    width: int = 260,
    height: int = 170,
) -> str:
    pad_l, pad_r, pad_t, pad_b = 46, 14, 26, 24
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b

    y_min = min(0.0, min(y))
    y_max = max(y) * 1.08 if max(y) > 0 else 0.1
    t_min, t_max = t[0], t[-1]

    def xs(v: float) -> float:
        return pad_l + (v - t_min) / (t_max - t_min or 1) * plot_w

    def ys(v: float) -> float:
        return pad_t + plot_h - (v - y_min) / (y_max - y_min or 1) * plot_h

    points = " ".join(f"{xs(a):.1f},{ys(b):.1f}" for a, b in zip(t, y))

    ticks = [y_min, (y_min + y_max) / 2, y_max]
    gridlines = []
    for tick in ticks:
        gy = ys(tick)
        gridlines.append(
            f'<line x1="{pad_l}" y1="{gy:.1f}" x2="{width - pad_r}" y2="{gy:.1f}" '
            f'stroke="{GRID}" stroke-width="1"/>'
        )
        gridlines.append(
            f'<text x="{pad_l - 6}" y="{gy + 3:.1f}" font-size="9" fill="{MUTED}" '
            f'text-anchor="end" font-family="system-ui,sans-serif">{fmt(tick, 1)}</text>'
        )

    end_x, end_y = xs(t[-1]), ys(y[-1])

    return f'''
<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img"
     aria-label="{title}: {y_label} over time">
  <text x="{pad_l}" y="14" font-size="11" fill="{TEXT_PRIMARY}" font-weight="600"
        font-family="system-ui,sans-serif">{title}</text>
  {''.join(gridlines)}
  <polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"
            stroke-linejoin="round" stroke-linecap="round"/>
  <circle cx="{end_x:.1f}" cy="{end_y:.1f}" r="4" fill="{color}" stroke="#fcfcfb" stroke-width="2"/>
  <text x="{end_x - 4:.1f}" y="{end_y - 8:.1f}" font-size="9" fill="{TEXT_SECONDARY}"
        text-anchor="end" font-family="system-ui,sans-serif">{fmt(y[-1], 2)}</text>
</svg>'''


def hbar_chart_svg(
    labels: list[str],
    values: list[float],
    title: str,
    unit: str,
    color: str = COLOR,
    width: int = 520,
) -> str:
    row_h = 26
    pad_l, pad_r, pad_t = 130, 50, 30
    height = pad_t + row_h * len(labels) + 10
    plot_w = width - pad_l - pad_r
    v_max = max(values) * 1.1 if max(values) > 0 else 1.0

    bars = []
    for i, (label, value) in enumerate(zip(labels, values)):
        y = pad_t + i * row_h
        bar_w = max(2.0, value / v_max * plot_w)
        bars.append(
            f'<text x="{pad_l - 8}" y="{y + row_h / 2 + 4:.1f}" font-size="10" '
            f'fill="{TEXT_PRIMARY}" text-anchor="end" '
            f'font-family="system-ui,sans-serif">{label}</text>'
        )
        bars.append(
            f'<rect x="{pad_l}" y="{y + 3:.1f}" width="{bar_w:.1f}" height="{row_h - 8}" '
            f'rx="4" fill="{color}"/>'
        )
        bars.append(
            f'<text x="{pad_l + bar_w + 6:.1f}" y="{y + row_h / 2 + 4:.1f}" font-size="10" '
            f'fill="{TEXT_SECONDARY}" font-family="system-ui,sans-serif">{fmt(value, 2)} {unit}</text>'
        )

    return f'''
<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img"
     aria-label="{title}">
  <text x="{pad_l}" y="16" font-size="12" fill="{TEXT_PRIMARY}" font-weight="600"
        font-family="system-ui,sans-serif">{title}</text>
  {''.join(bars)}
</svg>'''


def scatter_svg(
    xs_v: list[float],
    ys_v: list[float],
    labels: list[str],
    x_title: str,
    y_title: str,
    color: str = COLOR,
    colors: list[str] | None = None,
    width: int = 420,
    height: int = 340,
    legend: list[tuple[str, str]] | None = None,
) -> str:
    pad_l, pad_r, pad_t, pad_b = 50, 20, 16, 34
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    x_lo, x_hi = min(xs_v), max(xs_v)
    y_lo, y_hi = min(ys_v), max(ys_v)
    x_span = (x_hi - x_lo) or 1.0
    y_span = (y_hi - y_lo) or 1.0
    x_min, x_max = min(0.0, x_lo - 0.1 * x_span), x_hi + 0.15 * x_span
    y_min, y_max = min(0.0, y_lo - 0.1 * y_span), y_hi + 0.15 * y_span

    def xs(v: float) -> float:
        return pad_l + (v - x_min) / (x_max - x_min or 1) * plot_w

    def ys(v: float) -> float:
        return pad_t + plot_h - (v - y_min) / (y_max - y_min or 1) * plot_h

    point_colors = colors if colors is not None else [color] * len(xs_v)
    points = []
    for x, y, label, c in zip(xs_v, ys_v, labels, point_colors):
        px, py = xs(x), ys(y)
        points.append(
            f'<circle cx="{px:.1f}" cy="{py:.1f}" r="5" fill="{c}" '
            f'stroke="#fcfcfb" stroke-width="2"/>'
        )
        points.append(
            f'<text x="{px + 7:.1f}" y="{py + 3:.1f}" font-size="9" fill="{TEXT_SECONDARY}" '
            f'font-family="system-ui,sans-serif">{label}</text>'
        )

    legend_html = ""
    if legend:
        items = "".join(
            f'<span style="display:inline-flex;align-items:center;gap:4px;margin-right:12px">'
            f'<span style="width:9px;height:9px;border-radius:50%;background:{c};display:inline-block"></span>'
            f"{name}</span>"
            for name, c in legend
        )
        legend_html = f'<div style="font-size:10px;color:{TEXT_SECONDARY};margin:2px 0 4px">{items}</div>'

    return f'''{legend_html}
<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img"
     aria-label="{x_title} vs {y_title} scatter">
  <line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{pad_t + plot_h}" stroke="{GRID}" stroke-width="1"/>
  <line x1="{pad_l}" y1="{pad_t + plot_h}" x2="{pad_l + plot_w}" y2="{pad_t + plot_h}" stroke="{GRID}" stroke-width="1"/>
  <text x="{pad_l + plot_w / 2:.1f}" y="{height - 6}" font-size="10" fill="{MUTED}"
        text-anchor="middle" font-family="system-ui,sans-serif">{x_title}</text>
  <text x="12" y="{pad_t + plot_h / 2:.1f}" font-size="10" fill="{MUTED}"
        text-anchor="middle" font-family="system-ui,sans-serif"
        transform="rotate(-90 12 {pad_t + plot_h / 2:.1f})">{y_title}</text>
  {''.join(points)}
</svg>'''


def run(
    names: list[str],
    results_dir: Path,
    out_dir: Path,
    class_map: dict[str, str],
    title: str,
    scope_note: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = []
    missing = []
    for name in names:
        csv_path = results_dir / name / f"{name}_opa_motion_force.csv"
        if not csv_path.exists():
            missing.append(name)
            continue
        cols = read_csv(csv_path)
        metrics.append(compute_metrics(name, cols, class_map))

    if missing:
        print("Skipped (no CSV found):", ", ".join(missing))

    # --- cluster solvents by mobility/force pattern (sklearn) ---
    # solvent_class (protic/aprotic-polar/nonpolar) is intentionally excluded
    # from the clustering features: it is a manual chemistry label used only
    # to *interpret* the clusters afterwards, not to derive them.
    feature_names = [
        "mean_disp_A", "final_disp_A", "mean_force_eV_A", "std_force_eV_A",
        "diffusion_d_eff",
    ]
    X = np.array([[m[f] for f in feature_names] for m in metrics])
    X_scaled = StandardScaler().fit_transform(X)
    n_clusters = min(3, len(metrics))
    labels = KMeans(n_clusters=n_clusters, random_state=0, n_init=10).fit_predict(X_scaled)
    coords = PCA(n_components=2, random_state=0).fit_transform(X_scaled)
    for m, cluster, (pc1, pc2) in zip(metrics, labels, coords):
        m["cluster"] = int(cluster)
        m["pca1"] = float(pc1)
        m["pca2"] = float(pc2)

    # --- summary.csv ---
    summary_path = out_dir / "summary.csv"
    field_order = [
        "solvent", "solvent_class", "cluster", "n_frames", "recorded_time_span",
        "final_disp_A", "mean_disp_A", "max_disp_A", "mean_force_eV_A",
        "std_force_eV_A", "max_force_eV_A", "disp_slope_A_per_unit_time",
        "force_disp_correlation", "diffusion_d_eff", "gamma_eff",
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=field_order)
        writer.writeheader()
        for m in metrics:
            writer.writerow({k: m[k] for k in field_order})
    print(f"Wrote {summary_path}")

    # --- charts ---
    metrics_by_disp = sorted(metrics, key=lambda m: m["final_disp_A"], reverse=True)
    metrics_by_force = sorted(metrics, key=lambda m: m["mean_force_eV_A"], reverse=True)

    disp_panels = "\n".join(
        f'<div class="panel">{line_chart_svg(m["t"], m["disp"], m["solvent"], "disp (A)", color="var(--series-1)")}</div>'
        for m in metrics
    )
    force_panels = "\n".join(
        f'<div class="panel">{line_chart_svg(m["t"], m["force"], m["solvent"], "force (eV/A)", color="var(--series-1)")}</div>'
        for m in metrics
    )

    disp_bar = hbar_chart_svg(
        [m["solvent"] for m in metrics_by_disp],
        [m["final_disp_A"] for m in metrics_by_disp],
        "Final OPA displacement by solvent",
        "A",
        color="var(--series-1)",
    )
    force_bar = hbar_chart_svg(
        [m["solvent"] for m in metrics_by_force],
        [m["mean_force_eV_A"] for m in metrics_by_force],
        "Mean |F_OPA| by solvent",
        "eV/A",
        color="var(--series-1)",
    )
    scatter = scatter_svg(
        [m["mean_force_eV_A"] for m in metrics],
        [m["final_disp_A"] for m in metrics],
        [m["solvent"] for m in metrics],
        "Mean |F_OPA| (eV/A)",
        "Final displacement (A)",
        color="var(--series-1)",
    )

    cluster_var = [f"var(--cluster-{m['cluster']})" for m in metrics]
    n_clusters_used = len({m["cluster"] for m in metrics})
    cluster_legend = [
        (f"cluster {i}", f"var(--cluster-{i})") for i in range(n_clusters_used)
    ]
    mobility_scatter = scatter_svg(
        [m["diffusion_d_eff"] for m in metrics],
        [m["mean_force_eV_A"] for m in metrics],
        [m["solvent"] for m in metrics],
        "Diffusion proxy D_eff (A^2 / recorded ps)",
        "Mean |F_OPA| (eV/A)",
        colors=cluster_var,
        legend=cluster_legend,
    )
    pca_scatter = scatter_svg(
        [m["pca1"] for m in metrics],
        [m["pca2"] for m in metrics],
        [m["solvent"] for m in metrics],
        "PCA 1",
        "PCA 2",
        colors=cluster_var,
        legend=cluster_legend,
    )

    table_rows = "\n".join(
        f"<tr><td>{m['solvent']}</td><td>{m['solvent_class']}</td><td>{m['cluster']}</td>"
        f"<td>{fmt(m['final_disp_A'],3)}</td>"
        f"<td>{fmt(m['mean_disp_A'],3)}</td><td>{fmt(m['max_disp_A'],3)}</td>"
        f"<td>{fmt(m['mean_force_eV_A'],3)}</td><td>{fmt(m['max_force_eV_A'],3)}</td>"
        f"<td>{fmt(m['force_disp_correlation'],3)}</td>"
        f"<td>{fmt(m['diffusion_d_eff'],4)}</td>"
        f"<td>{'n/a' if math.isnan(m['gamma_eff']) else fmt(m['gamma_eff'],4)}</td></tr>"
        for m in sorted(metrics, key=lambda m: m["solvent"])
    )

    html = f'''<!doctype html>
<html data-theme="light">
<head>
<meta charset="utf-8">
<title>{title}</title>
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
  .grid {{ display: grid; grid-template-columns: repeat(5, 1fr); gap: 8px; }}
  .panel {{ background: var(--surface-1); border: 1px solid var(--border); border-radius: 6px; padding: 4px; }}
  table {{ border-collapse: collapse; font-size: 12px; margin-top: 8px; }}
  th, td {{ padding: 4px 10px; border-bottom: 1px solid var(--border); text-align: right; }}
  th:first-child, td:first-child {{ text-align: left; }}
  #theme-toggle {{ float: right; font-size: 12px; }}
  .caveat {{ font-size: 12px; color: var(--text-secondary); max-width: 720px; }}
</style>
</head>
<body>
<button id="theme-toggle" onclick="
  const r = document.documentElement;
  r.setAttribute('data-theme', r.getAttribute('data-theme') === 'dark' ? 'light' : 'dark');
">Toggle dark mode</button>
<h1>{title}</h1>
<p class="caveat">
Source: results/&lt;solvent&gt;/&lt;solvent&gt;_opa_motion_force.csv, generated by
jupyter_copy_cell.py / mix.py. {scope_note} <b>Caveat:</b> the analysis script's
<code>time_ps</code> column is computed from <code>AnalysisInput.save_interval</code>
(100 steps) rather than the actual trajectory dump interval
(<code>MD_LOG_INTERVAL</code> = 1000 steps), so the recorded time axis likely
understates real elapsed time by ~10x. Relative comparisons across solvents are
unaffected since all runs share the same step count and dump interval; absolute
D_eff / gamma_eff values below would shift by a constant factor if the time axis
were corrected. <b>solvent_class</b> (protic / aprotic-polar / nonpolar) is a
manually assigned chemistry label, not derived from the simulation, added only to
help interpret the clusters below. <b>cluster</b> is unsupervised (KMeans, k={n_clusters_used})
on standardized [mean_disp_A, final_disp_A, mean_force_eV_A, std_force_eV_A,
diffusion_d_eff]; solvent_class was excluded from clustering features.
</p>

<h2>Final OPA displacement (sorted)</h2>
{disp_bar}

<h2>Mean |F_OPA| (sorted)</h2>
{force_bar}

<h2>Mean force vs. final displacement</h2>
{scatter}

<h2>Mobility (D_eff) vs. mean force, colored by cluster</h2>
{mobility_scatter}

<h2>PCA of standardized mobility/force features, colored by cluster</h2>
{pca_scatter}

<h2>OPA displacement over time, per solvent</h2>
<div class="grid">{disp_panels}</div>

<h2>|F_OPA| over time, per solvent</h2>
<div class="grid">{force_panels}</div>

<h2>Summary table</h2>
<table>
<tr><th>solvent</th><th>class</th><th>cluster</th><th>final disp (A)</th><th>mean disp (A)</th><th>max disp (A)</th>
<th>mean |F| (eV/A)</th><th>max |F| (eV/A)</th><th>corr(F, disp)</th>
<th>D_eff (A^2/rec.ps)</th><th>gamma_eff (eV*rec.ps/A^2)</th></tr>
{table_rows}
</table>
</body>
</html>'''

    report_path = out_dir / "report.html"
    report_path.write_text(html, encoding="utf-8")
    print(f"Wrote {report_path}")


def main() -> None:
    run(
        PURE_SOLVENTS, RESULTS_DIR, OUT_DIR, SOLVENT_CLASS,
        title="Pure-component solvent systems: OPA displacement &amp; force",
        scope_note=(
            "Only single-component systems are included here; mixtures "
            "(e.g. acetone/n-heptane blends, isopropanol/n-heptane) are excluded."
        ),
    )


if __name__ == "__main__":
    main()
