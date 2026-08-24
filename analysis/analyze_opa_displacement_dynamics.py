"""
Displacement/motion analysis for the pure-solvent OPA runs, using the same
class of methods the single-particle-tracking (SPT) and MD self-diffusion
literature uses to analyze a *tracked object's* trajectory -- the direct
counterpart, for displacement data, of the RDF/coordination-number analysis
already done for force/structure in analyze_opa_solvation_structure.py.

Method (see report footer for full citations):
  - Time-averaged MSD (TAMSD) with multiple time origins, not the single
    origin-at-t=0 "OPA_disp_norm_A" column used in the first-pass analysis.
  - Anomalous diffusion exponent alpha from a log-log fit of MSD(tau) vs tau
    (MSD ~ tau^alpha): alpha=1 normal (Fickian) diffusion, alpha<1
    subdiffusive/caged motion, alpha>1 superdiffusive/persistent motion.
    This directly tests the "solvent caging" hypothesis raised from the
    first-pass D_eff numbers (e.g. isopropanol's negative single-origin
    slope) with a proper multi-origin estimator.
  - A normalized velocity autocorrelation function (VACF, from frame-to-frame
    displacement as a coarse velocity proxy) and its 1/e relaxation time, as
    a complementary measure of how quickly the direction of motion
    decorrelates.

Trajectory-length caveat (explicit, not glossed over): each run has only
N=101 saved frames. The SPT literature (Guidelines for the Fitting of
Anomalous Diffusion MSD Graphs from Single Particle Tracking Experiments,
PMC4334513) recommends L >= 300 points for reliable alpha estimates, and a
transport-classification study (PMC4241458) notes "weak ergodicity breaking"
below M=100 points. Our N=101 sits right at that unreliable boundary, so
alpha here should be read as indicative/comparative across solvents, not a
precise, publication-grade exponent.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

from _common import GRID, MUTED, TEXT_PRIMARY, TEXT_SECONDARY

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
OUT_DIR = RESULTS_DIR / "pure_components_analysis"

PURE_SOLVENTS = [
    "acetonitrile", "propylene_carbonate", "isopropanol", "CPME",
    "ethyl_acetate", "DMC", "p-xylene", "cyclohexane", "ethanol", "methanol",
]

K_MAX_FRAC = 3  # use lags up to N // K_MAX_FRAC (short-trajectory-safe fraction)
FIT_MIN_LAG = 2  # skip k=1 in the alpha fit (short-lag/noise dominated)
VACF_MAX_LAG = 20


def read_com(csv_path: Path):
    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        t, x, y, z = [], [], [], []
        for row in reader:
            t.append(float(row["time_ps"]))
            x.append(float(row["OPA_COM_x_A"]))
            y.append(float(row["OPA_COM_y_A"]))
            z.append(float(row["OPA_COM_z_A"]))
    return np.array(t), np.column_stack([x, y, z])


def tamsd(pos: np.ndarray, k_max: int) -> np.ndarray:
    n = len(pos)
    msd = np.empty(k_max)
    for k in range(1, k_max + 1):
        diff = pos[k:] - pos[:-k]
        msd[k - 1] = np.mean(np.sum(diff * diff, axis=1))
    return msd


def fit_alpha(tau: np.ndarray, msd: np.ndarray, fit_min_lag: int):
    mask = tau >= tau[fit_min_lag - 1]
    log_tau = np.log(tau[mask])
    log_msd = np.log(msd[mask])
    slope, intercept = np.polyfit(log_tau, log_msd, 1)
    pred = slope * log_tau + intercept
    ss_res = np.sum((log_msd - pred) ** 2)
    ss_tot = np.sum((log_msd - np.mean(log_msd)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot else float("nan")
    return float(slope), float(intercept), float(r2), mask


def linear_d_eff(tau: np.ndarray, msd: np.ndarray, mask: np.ndarray) -> float:
    slope = np.polyfit(tau[mask], msd[mask], 1)[0]
    return float(slope) / 6.0


def velocity_autocorrelation(pos: np.ndarray, dt: float, max_lag: int):
    v = np.diff(pos, axis=0) / dt
    n = len(v)
    max_lag = min(max_lag, n - 1)
    vacf = np.empty(max_lag + 1)
    for lag in range(max_lag + 1):
        vacf[lag] = np.mean(np.sum(v[: n - lag] * v[lag:], axis=1))
    vacf /= vacf[0]
    return vacf


def relaxation_time(vacf: np.ndarray, dt: float) -> float:
    below = np.where(vacf < 1.0 / math.e)[0]
    if len(below) == 0:
        return float("nan")
    return float(below[0] * dt)


def classify_regime(alpha: float) -> str:
    if alpha < 0.85:
        return "subdiffusive/caged"
    if alpha > 1.15:
        return "superdiffusive/persistent"
    return "near-Brownian"


def fmt(x: float, nd: int = 4) -> str:
    if isinstance(x, float) and math.isnan(x):
        return "n/a"
    return f"{x:.{nd}f}"


def log_log_svg(tau, msd, alpha, intercept, fit_mask, title, width=280, height=180):
    pad_l, pad_r, pad_t, pad_b = 40, 10, 20, 22
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    log_tau = np.log(tau)
    log_msd = np.log(msd)
    x_min, x_max = log_tau.min(), log_tau.max()
    y_min, y_max = log_msd.min(), log_msd.max()
    y_pad = (y_max - y_min) * 0.08 or 0.1

    def xs(v):
        return pad_l + (v - x_min) / (x_max - x_min or 1) * plot_w

    def ys(v):
        return pad_t + plot_h - (v - (y_min - y_pad)) / ((y_max + y_pad) - (y_min - y_pad)) * plot_h

    pts = " ".join(f"{xs(a):.1f},{ys(b):.1f}" for a, b in zip(log_tau, log_msd))
    fit_tau = log_tau[fit_mask]
    fit_line_y = alpha * fit_tau + intercept
    fit_pts = " ".join(f"{xs(a):.1f},{ys(b):.1f}" for a, b in zip(fit_tau, fit_line_y))

    return f'''
<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="{title}">
  <text x="{pad_l}" y="12" font-size="10" fill="{TEXT_PRIMARY}" font-weight="600" font-family="system-ui,sans-serif">{title} (a={alpha:.2f})</text>
  <line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{pad_t + plot_h}" stroke="{GRID}" stroke-width="1"/>
  <line x1="{pad_l}" y1="{pad_t + plot_h}" x2="{width - pad_r}" y2="{pad_t + plot_h}" stroke="{GRID}" stroke-width="1"/>
  <polyline points="{pts}" fill="none" stroke="var(--series-1)" stroke-width="2" stroke-linejoin="round"/>
  <polyline points="{fit_pts}" fill="none" stroke="var(--cluster-1)" stroke-width="1.5" stroke-dasharray="3,2"/>
  <text x="{pad_l}" y="{height - 4}" font-size="8" fill="{MUTED}" font-family="system-ui,sans-serif">log(tau) vs log(MSD); dashed = alpha fit</text>
</svg>'''


def bar_svg(labels, values, title, unit, width=520):
    row_h = 26
    pad_l, pad_r, pad_t = 150, 50, 30
    height = pad_t + row_h * len(labels) + 10
    plot_w = width - pad_l - pad_r
    v_min = min(0.0, min(values))
    v_max = max(values) * 1.1 if max(values) > 0 else 1.0
    span = v_max - v_min or 1.0
    zero_x = pad_l + (0 - v_min) / span * plot_w

    bars = []
    for i, (label, value) in enumerate(zip(labels, values)):
        y = pad_t + i * row_h
        x0 = pad_l + (min(0, value) - v_min) / span * plot_w
        w = abs(value - 0) / span * plot_w
        bars.append(
            f'<text x="{pad_l - 8}" y="{y + row_h / 2 + 4:.1f}" font-size="10" fill="{TEXT_PRIMARY}" '
            f'text-anchor="end" font-family="system-ui,sans-serif">{label}</text>'
        )
        bars.append(f'<rect x="{x0:.1f}" y="{y + 3:.1f}" width="{max(w,2):.1f}" height="{row_h - 8}" rx="4" fill="var(--series-1)"/>')
        bars.append(
            f'<text x="{(x0 + w + 6) if value >= 0 else (x0 - 6):.1f}" y="{y + row_h / 2 + 4:.1f}" font-size="10" '
            f'fill="{TEXT_SECONDARY}" text-anchor="{"start" if value >= 0 else "end"}" '
            f'font-family="system-ui,sans-serif">{value:.2f} {unit}</text>'
        )
    bars.append(f'<line x1="{zero_x:.1f}" y1="{pad_t}" x2="{zero_x:.1f}" y2="{height-6}" stroke="{GRID}" stroke-width="1"/>')

    return f'''
<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="{title}">
  <text x="{pad_l}" y="16" font-size="12" fill="{TEXT_PRIMARY}" font-weight="600" font-family="system-ui,sans-serif">{title}</text>
  {''.join(bars)}
</svg>'''


def main(
    names: list[str] | None = None,
    results_dir: Path | None = None,
    out_dir: Path | None = None,
    title: str = "OPA displacement dynamics: time-averaged MSD and anomalous diffusion exponent",
) -> None:
    names = PURE_SOLVENTS if names is None else names
    results_dir = RESULTS_DIR if results_dir is None else results_dir
    out_dir = OUT_DIR if out_dir is None else out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    msd_panels = []
    for name in names:
        csv_path = results_dir / name / f"{name}_opa_motion_force.csv"
        if not csv_path.exists():
            print(f"Skip {name}: no CSV")
            continue
        t, pos = read_com(csv_path)
        dt = t[1] - t[0]
        n = len(t)
        k_max = max(FIT_MIN_LAG + 1, n // K_MAX_FRAC)
        msd = tamsd(pos, k_max)
        tau = np.arange(1, k_max + 1) * dt

        alpha, intercept, r2, fit_mask = fit_alpha(tau, msd, FIT_MIN_LAG)
        d_eff = linear_d_eff(tau, msd, fit_mask)
        vacf = velocity_autocorrelation(pos, dt, VACF_MAX_LAG)
        tau_v = relaxation_time(vacf, dt)
        regime = classify_regime(alpha)

        rows.append({
            "solvent": name,
            "alpha": alpha,
            "alpha_fit_r2": r2,
            "d_eff_tamsd": d_eff,
            "vacf_relaxation_time": tau_v,
            "regime": regime,
            "n_frames": n,
            "k_max_lag": k_max,
        })
        msd_panels.append(log_log_svg(tau, msd, alpha, intercept, fit_mask, name))
        print(f"{name}: alpha={alpha:.3f} (R2={r2:.2f})  D_eff_tamsd={d_eff:.4f}  "
              f"vacf_tau={tau_v}  regime={regime}")

    summary_path = out_dir / "displacement_dynamics_summary.csv"
    field_order = ["solvent", "alpha", "alpha_fit_r2", "d_eff_tamsd",
                   "vacf_relaxation_time", "regime", "n_frames", "k_max_lag"]
    with summary_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=field_order)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"Wrote {summary_path}")

    alpha_sorted = sorted(rows, key=lambda r: r["alpha"])
    alpha_bar = bar_svg(
        [r["solvent"] for r in alpha_sorted], [r["alpha"] for r in alpha_sorted],
        "Anomalous diffusion exponent alpha (1 = Brownian, <1 = caged, >1 = persistent)", "",
    )

    table_rows = "\n".join(
        f"<tr><td>{r['solvent']}</td><td>{r['regime']}</td><td>{fmt(r['alpha'],3)}</td>"
        f"<td>{fmt(r['alpha_fit_r2'],3)}</td><td>{fmt(r['d_eff_tamsd'],4)}</td>"
        f"<td>{fmt(r['vacf_relaxation_time'],3)}</td></tr>"
        for r in sorted(rows, key=lambda r: r["solvent"])
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
    --series-1: #2a78d6; --cluster-1: #eb6834;
  }}
  :root[data-theme="dark"] {{
    --surface-1: #1a1a19; --page: #0d0d0d; --text-primary: #ffffff;
    --text-secondary: #c3c2b7; --border: rgba(255,255,255,0.10);
    --series-1: #3987e5; --cluster-1: #d95926;
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
  .caveat {{ font-size: 12px; color: var(--text-secondary); max-width: 760px; }}
</style>
</head>
<body>
<button id="theme-toggle" onclick="
  const r = document.documentElement;
  r.setAttribute('data-theme', r.getAttribute('data-theme') === 'dark' ? 'light' : 'dark');
">Toggle dark mode</button>
<h1>{title}</h1>
<p class="caveat">
Displacement-side counterpart to the force/structure RDF analysis. Uses
time-averaged MSD with multiple time origins (not the single-origin
"displacement from t=0" column used in the first pass) and fits the
anomalous diffusion exponent alpha from log(MSD) vs log(tau), alpha=1
normal diffusion, alpha&lt;1 subdiffusive/caged, alpha&gt;1
superdiffusive/persistent. <b>Caveat:</b> each trajectory has only ~101
frames; single-particle-tracking guidelines recommend >=300 points for a
reliable alpha, and note "weak ergodicity breaking" below ~100 points, so
treat alpha as comparative/indicative across solvents, not a precise
exponent. Fit uses lags tau = 2 up to about a third of the trajectory length
(k_max = n_frames // {K_MAX_FRAC}); R^2 of the log-log fit is reported per
solvent.
</p>

<h2>Anomalous diffusion exponent by solvent</h2>
{alpha_bar}

<h2>log(MSD) vs log(tau) per solvent, with fitted alpha</h2>
<div class="grid">{''.join(f'<div class="panel">{p}</div>' for p in msd_panels)}</div>

<h2>Summary table</h2>
<table>
<tr><th>solvent</th><th>regime</th><th>alpha</th><th>fit R^2</th>
<th>D_eff (TAMSD)</th><th>VACF 1/e time (rec. ps)</th></tr>
{table_rows}
</table>

<h2>Sources</h2>
<p class="caveat">
Guidelines for the Fitting of Anomalous Diffusion Mean Square Displacement
Graphs from Single Particle Tracking Experiments --
<a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC4334513/">PMC4334513</a>.
Identifying Transport Behavior of Single-Molecule Trajectories --
<a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC4241458">PMC4241458</a>.
Application of Molecular Dynamics Simulations in Molecular Property
Prediction II: Diffusion Coefficient --
<a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC3193570/">PMC3193570</a>.
Trajectory Analysis in Single-Particle Tracking: From Mean Squared
Displacement to Machine Learning Approaches --
<a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC11354962/">PMC11354962</a>.
</p>
</body>
</html>'''

    report_path = out_dir / "displacement_dynamics_report.html"
    report_path.write_text(html, encoding="utf-8")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
