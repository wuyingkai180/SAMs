"""
Force decomposition analysis: how much of the net force on OPA acts on the
phosphonate head group (P + 3 O + 2 acidic H) versus the rest of the
molecule (alkyl tail), per pure solvent.

The existing `<solvent>_opa_motion_force.csv` (used by analyze_pure_solvents.py)
only has the *whole-molecule* net force (F_OPA_norm_eV_A) -- it was computed
by summing per-atom forces over all OPA atoms before saving, so the
per-atom breakdown is gone from that file. This script re-reads per-atom
forces directly from the saved MD trajectory (`<solvent>_md_300K.traj`,
which ASE caches at write time and returns via `atoms.get_forces()` on
re-read) and re-sums them separately over the head-group vs. tail atom
index subsets already identified by
analyze_opa_solvation_structure.classify_head_tail() (bond connectivity
from structures/opa.vasp, not hardcoded indices).

Scope: only solvents with a saved trajectory file are covered (see
results/README.md); solvents added without a `.traj` (acetone, n-heptane,
thf, toluene) are skipped, same as analyze_opa_solvation_structure.py.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
from ase.io.trajectory import Trajectory

from analyze_pure_solvents import (
    PURE_SOLVENTS, SOLVENT_CLASS, CLUSTER_COLORS, fmt, hbar_chart_svg, scatter_svg,
)
from analyze_opa_solvation_structure import OPA_VASP, read_opa_vasp, classify_head_tail

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
OUT_DIR = RESULTS_DIR / "pure_components_analysis"


def per_frame_forces(traj_path: Path, head_idx: list[int], n_opa: int):
    """Per-frame force norms (eV/A) on head atoms, tail atoms, and whole OPA."""
    traj = Trajectory(traj_path)
    tail_idx = sorted(set(range(n_opa)) - set(head_idx))
    f_head, f_tail, f_opa = [], [], []
    for atoms in traj:
        forces = atoms.get_forces()
        f_head.append(np.linalg.norm(forces[head_idx].sum(axis=0)))
        f_tail.append(np.linalg.norm(forces[tail_idx].sum(axis=0)))
        f_opa.append(np.linalg.norm(forces[:n_opa].sum(axis=0)))
    return np.array(f_head), np.array(f_tail), np.array(f_opa)


def read_reference_f_opa(csv_path: Path) -> np.ndarray:
    """F_OPA_norm_eV_A already stored in <solvent>_opa_motion_force.csv, used
    only as a sanity check that the re-summed whole-molecule force here
    matches the value the MD workflow scripts originally computed."""
    with csv_path.open(newline="", encoding="utf-8") as fh:
        return np.array([float(row["F_OPA_norm_eV_A"]) for row in csv.DictReader(fh)])


def compute_metrics(name: str, f_head: np.ndarray, f_tail: np.ndarray, f_opa: np.ndarray,
                     class_map: dict[str, str]) -> dict:
    mean_head, mean_tail, mean_opa = float(f_head.mean()), float(f_tail.mean()), float(f_opa.mean())
    return {
        "solvent": name,
        "solvent_class": class_map.get(name, "unknown"),
        "n_frames": len(f_head),
        "mean_head_force_eV_A": mean_head,
        "std_head_force_eV_A": float(f_head.std()),
        "max_head_force_eV_A": float(f_head.max()),
        "mean_tail_force_eV_A": mean_tail,
        "std_tail_force_eV_A": float(f_tail.std()),
        "max_tail_force_eV_A": float(f_tail.max()),
        "mean_total_force_eV_A": mean_opa,
        "head_share_of_total": mean_head / mean_opa if mean_opa else float("nan"),
        "head_tail_force_ratio": mean_head / mean_tail if mean_tail else float("nan"),
        "f_head": f_head,
        "f_tail": f_tail,
    }


def run(names: list[str], results_dir: Path, out_dir: Path, class_map: dict[str, str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    symbols, coords = read_opa_vasp(OPA_VASP)
    topo = classify_head_tail(symbols, coords)
    head_idx, n_opa = topo["head"], topo["n_opa_atoms"]
    print(f"Head atoms (n={len(head_idx)}): {head_idx}; tail atoms (n={n_opa - len(head_idx)})")

    metrics, missing = [], []
    for name in names:
        traj_path = results_dir / name / f"{name}_md_300K.traj"
        if not traj_path.exists():
            missing.append(name)
            continue
        f_head, f_tail, f_opa = per_frame_forces(traj_path, head_idx, n_opa)

        ref_csv = results_dir / name / f"{name}_opa_motion_force.csv"
        if ref_csv.exists():
            ref = read_reference_f_opa(ref_csv)
            max_diff = float(np.max(np.abs(ref - f_opa))) if len(ref) == len(f_opa) else float("nan")
            if not (max_diff <= 1e-6):
                print(f"  [check] {name}: recomputed whole-molecule force differs from "
                      f"{ref_csv.name} by up to {max_diff:.2e} eV/A")

        metrics.append(compute_metrics(name, f_head, f_tail, f_opa, class_map))

    if missing:
        print("Skipped (no trajectory found):", ", ".join(missing))

    summary_path = out_dir / "head_tail_force_summary.csv"
    field_order = [
        "solvent", "solvent_class", "n_frames", "mean_head_force_eV_A",
        "std_head_force_eV_A", "max_head_force_eV_A", "mean_tail_force_eV_A",
        "std_tail_force_eV_A", "max_tail_force_eV_A", "mean_total_force_eV_A",
        "head_share_of_total", "head_tail_force_ratio",
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=field_order)
        writer.writeheader()
        for m in metrics:
            writer.writerow({k: m[k] for k in field_order})
    print(f"Wrote {summary_path}")

    head_bar = hbar_chart_svg(
        [m["solvent"] for m in sorted(metrics, key=lambda m: -m["mean_head_force_eV_A"])],
        [m["mean_head_force_eV_A"] for m in sorted(metrics, key=lambda m: -m["mean_head_force_eV_A"])],
        "Mean force on phosphonate head group (P+3O+2 acidic H)", "eV/A",
        color="var(--series-1)",
    )
    tail_bar = hbar_chart_svg(
        [m["solvent"] for m in sorted(metrics, key=lambda m: -m["mean_tail_force_eV_A"])],
        [m["mean_tail_force_eV_A"] for m in sorted(metrics, key=lambda m: -m["mean_tail_force_eV_A"])],
        "Mean force on alkyl tail (remaining 55 atoms)", "eV/A",
        color="var(--series-1)",
    )
    share_scatter = scatter_svg(
        [m["mean_total_force_eV_A"] for m in metrics],
        [m["head_share_of_total"] for m in metrics],
        [m["solvent"] for m in metrics],
        "Mean total |F_OPA| (eV/A)", "Head share of total force",
        color="var(--series-1)",
    )

    table_rows = "\n".join(
        f"<tr><td>{m['solvent']}</td><td>{m['solvent_class']}</td>"
        f"<td>{fmt(m['mean_head_force_eV_A'],3)}</td><td>{fmt(m['mean_tail_force_eV_A'],3)}</td>"
        f"<td>{fmt(m['head_share_of_total'],3)}</td><td>{fmt(m['head_tail_force_ratio'],3)}</td>"
        f"<td>{fmt(m['mean_total_force_eV_A'],3)}</td></tr>"
        for m in sorted(metrics, key=lambda m: m["solvent"])
    )

    html = f'''<!doctype html>
<html data-theme="light">
<head>
<meta charset="utf-8">
<title>OPA head-vs-tail force decomposition</title>
<style>
  :root {{
    --surface-1: #fcfcfb; --page: #f9f9f7; --text-primary: #0b0b0b;
    --text-secondary: #52514e; --border: rgba(11,11,11,0.10);
    --series-1: {CLUSTER_COLORS[0][0]};
  }}
  :root[data-theme="dark"] {{
    --surface-1: #1a1a19; --page: #0d0d0d; --text-primary: #ffffff;
    --text-secondary: #c3c2b7; --border: rgba(255,255,255,0.10);
    --series-1: {CLUSTER_COLORS[0][1]};
  }}
  body {{ background: var(--page); color: var(--text-primary);
          font-family: system-ui, -apple-system, "Segoe UI", sans-serif; margin: 0; padding: 24px; }}
  h1 {{ font-size: 18px; }}
  h2 {{ font-size: 14px; color: var(--text-secondary); margin-top: 32px; }}
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
<h1>OPA head-vs-tail force decomposition</h1>
<p class="caveat">
Re-sums per-atom forces read back from each solvent's saved MD trajectory
over the phosphonate head group (P + 3 O + 2 acidic H, identified from
opa.vasp bond connectivity, same definition as
<a href="literature_report.html">literature_report.html</a>'s head-group RDF)
versus the remaining alkyl-tail atoms. <b>head_share_of_total</b> and
<b>head_tail_force_ratio</b> are computed from per-frame force
<i>magnitudes</i> (norms), so they do not sum/ratio exactly like vector
components would -- read them as "how much of the fluctuating force
magnitude shows up on the head" rather than a strict decomposition of the
net force vector. Only solvents with a saved <code>.traj</code> are
included; see results/README.md for which pure-solvent folders lack one.
</p>

<h2>Mean force on head group (sorted)</h2>
{head_bar}

<h2>Mean force on tail (sorted)</h2>
{tail_bar}

<h2>Head force share of total vs. mean total force</h2>
{share_scatter}

<h2>Summary table</h2>
<table>
<tr><th>solvent</th><th>class</th><th>mean head |F| (eV/A)</th><th>mean tail |F| (eV/A)</th>
<th>head share</th><th>head/tail ratio</th><th>mean total |F| (eV/A)</th></tr>
{table_rows}
</table>
</body>
</html>'''

    report_path = out_dir / "head_tail_force_report.html"
    report_path.write_text(html, encoding="utf-8")
    print(f"Wrote {report_path}")


def main() -> None:
    run(PURE_SOLVENTS, RESULTS_DIR, OUT_DIR, SOLVENT_CLASS)


if __name__ == "__main__":
    main()
