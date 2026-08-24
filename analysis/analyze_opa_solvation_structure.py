"""
Structural solvation analysis for the pure-solvent OPA (octadecylphosphonic
acid, C18H39O3P) MD runs, following the same class of metrics used in the
literature on solvent effects on phosphonic-acid / alkanethiol SAM formation:

  - radial distribution function (RDF) / running coordination number of
    solvent atoms around the phosphonate headgroup vs. around the terminal
    alkyl (tail) carbon -- the structural analogue of the head-vs-tail RDF
    comparison in solvent-SAM studies (e.g. Theoretical Insights into the
    Solvent Polarity Effect on the Quality of Self-Assembled
    N-Octadecanethiol Monolayers on Cu(111), PMC6017570; Molecular dynamics
    simulations of phosphonic acid-aluminum oxide self-organization,
    RSC Phys. Chem. Chem. Phys. 2017, c6cp08681k -- which discriminates
    solvent role by comparing 2-propanol/hexane/vacuum for the same
    octadecylphosphonic acid headgroup studied here).

Head/tail atom membership is derived from bond connectivity in
structures/opa.vasp (not hardcoded indices): P + its 3 O neighbors + any H
bonded to those O's = headgroup; everything else (18 alkyl C + 37 alkyl H)
= tail. The single carbon with the largest bonded-graph distance from P is
used as the "terminal methyl" analogue for the tail-side RDF, mirroring how
the Cu(111) paper focused its hydrophobic-side RDF on the terminal CH3.

Caveat carried over from analyze_pure_solvents.py: there is no substrate and
no second OPA molecule in these runs, so packing density / tilt angle /
gauche-defect metrics from the literature (which describe an assembled
monolayer) do not apply here. This script only characterizes the *pre
-adsorption solvation environment* of a single OPA molecule in bulk solvent.
"""

from __future__ import annotations

import csv
import math
from collections import deque
from pathlib import Path

import numpy as np
from ase.io.trajectory import Trajectory

from _common import GRID, MUTED, TEXT_PRIMARY, TEXT_SECONDARY

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
OUT_DIR = RESULTS_DIR / "pure_components_analysis"
OPA_VASP = Path(__file__).resolve().parent.parent / "structures" / "opa.vasp"

PURE_SOLVENTS = [
    "acetonitrile", "propylene_carbonate", "isopropanol", "CPME",
    "ethyl_acetate", "DMC", "p-xylene", "cyclohexane", "ethanol", "methanol",
]

BOND_CUTOFF = {
    frozenset(["P", "O"]): 1.9,
    frozenset(["O", "H"]): 1.3,
    frozenset(["C", "H"]): 1.3,
    frozenset(["C", "C"]): 1.75,
    frozenset(["P", "C"]): 1.95,
}

RMAX = 12.0
DR = 0.1
NBINS = int(RMAX / DR)
COORD_CUTOFF_A = 4.0  # fixed radius for the head-vs-tail coordination-number comparison


def read_opa_vasp(path: Path) -> tuple[list[str], np.ndarray]:
    lines = path.read_text().splitlines()
    scale = float(lines[1].split()[0])
    cell = np.array([[float(x) for x in lines[i].split()[:3]] for i in range(2, 5)]) * scale
    symbols_line = lines[5].split()
    counts = [int(x) for x in lines[6].split()]
    coord_start = 7
    if lines[coord_start].strip().lower().startswith("s"):
        coord_start += 1
    mode = lines[coord_start].strip().lower()
    coord_start += 1
    symbols = []
    for s, c in zip(symbols_line, counts):
        symbols.extend([s] * c)
    coords = np.array(
        [[float(x) for x in lines[coord_start + i].split()[:3]] for i in range(len(symbols))]
    )
    if mode.startswith(("d", "f")):
        coords = coords @ cell
    return symbols, coords


def classify_head_tail(symbols: list[str], coords: np.ndarray) -> dict:
    n = len(symbols)
    adj = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            cutoff = BOND_CUTOFF.get(frozenset([symbols[i], symbols[j]]))
            if cutoff and np.linalg.norm(coords[i] - coords[j]) < cutoff:
                adj[i].append(j)
                adj[j].append(i)

    p_idx = symbols.index("P")
    dist = [-1] * n
    dist[p_idx] = 0
    q = deque([p_idx])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                q.append(v)
    if any(d == -1 for d in dist):
        raise ValueError("opa.vasp connectivity graph is disconnected; check BOND_CUTOFF")

    o_indices = [i for i in range(n) if symbols[i] == "O"]
    head_h = [j for i in o_indices for j in adj[i] if symbols[j] == "H"]
    head = sorted([p_idx] + o_indices + head_h)
    tail_c = [i for i in range(n) if symbols[i] == "C"]
    terminal_c = max(tail_c, key=lambda i: dist[i])
    return {"head": head, "terminal_c": terminal_c, "n_opa_atoms": n}


def minimum_image(diff: np.ndarray, box: np.ndarray) -> np.ndarray:
    return diff - box * np.round(diff / box)


def solvation_profile(traj_path: Path, head_idx: list[int], terminal_idx: int, n_opa: int):
    traj = Trajectory(traj_path)
    n_frames = len(traj)
    head_hist = np.zeros(NBINS)
    tail_hist = np.zeros(NBINS)
    head_running = np.zeros(NBINS)  # cumulative count within r, per head atom per frame (summed)
    tail_running = np.zeros(NBINS)
    box = None
    n_solvent_atoms = None

    for atoms in traj:
        pos = atoms.get_positions()
        box = atoms.cell.diagonal()
        solvent_pos = pos[n_opa:]
        n_solvent_atoms = len(solvent_pos)

        head_diff = solvent_pos[None, :, :] - pos[head_idx][:, None, :]
        head_dist = np.linalg.norm(minimum_image(head_diff, box), axis=-1)
        h, _ = np.histogram(head_dist.ravel(), bins=NBINS, range=(0, RMAX))
        head_hist += h

        tail_diff = solvent_pos - pos[terminal_idx][None, :]
        tail_dist = np.linalg.norm(minimum_image(tail_diff, box), axis=-1)
        t, _ = np.histogram(tail_dist.ravel(), bins=NBINS, range=(0, RMAX))
        tail_hist += t

    edges = np.linspace(0, RMAX, NBINS + 1)
    r_mid = 0.5 * (edges[:-1] + edges[1:])
    shell_vol = 4.0 / 3.0 * math.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    volume = float(np.prod(box))
    rho = n_solvent_atoms / volume

    n_head = len(head_idx)
    g_head = head_hist / (n_frames * n_head * shell_vol * rho)
    g_tail = tail_hist / (n_frames * 1 * shell_vol * rho)

    running_head = np.cumsum(head_hist) / (n_frames * n_head)
    running_tail = np.cumsum(tail_hist) / (n_frames * 1)

    return {
        "r": r_mid,
        "g_head": g_head,
        "g_tail": g_tail,
        "running_head": running_head,
        "running_tail": running_tail,
        "n_frames": n_frames,
        "n_solvent_atoms": n_solvent_atoms,
        "box_length": float(box[0]),
    }


def coord_number_at(r: np.ndarray, running: np.ndarray, radius: float) -> float:
    return float(np.interp(radius, r, running))


def first_peak(r: np.ndarray, g: np.ndarray) -> tuple[float, float]:
    # first local maximum of g(r) beyond the unphysical near-zero region
    start = int(1.5 / DR)
    idx = start + int(np.argmax(g[start:]))
    return float(r[idx]), float(g[idx])


def line_svg(r, y1, y2, title, y_title, width=280, height=180):
    pad_l, pad_r, pad_t, pad_b = 42, 12, 22, 24
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    y_max = max(max(y1), max(y2)) * 1.1 or 1.0
    x_min, x_max = r[0], r[-1]

    def xs(v):
        return pad_l + (v - x_min) / (x_max - x_min) * plot_w

    def ys(v):
        return pad_t + plot_h - v / y_max * plot_h

    def path(y):
        return " ".join(f"{xs(a):.1f},{ys(b):.1f}" for a, b in zip(r, y))

    ticks = [0, y_max / 2, y_max]
    grid = "".join(
        f'<line x1="{pad_l}" y1="{ys(t):.1f}" x2="{width - pad_r}" y2="{ys(t):.1f}" stroke="{GRID}" stroke-width="1"/>'
        f'<text x="{pad_l - 5}" y="{ys(t) + 3:.1f}" font-size="8" fill="{MUTED}" text-anchor="end" font-family="system-ui,sans-serif">{t:.1f}</text>'
        for t in ticks
    )

    return f'''
<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="{title}">
  <text x="{pad_l}" y="12" font-size="10" fill="{TEXT_PRIMARY}" font-weight="600" font-family="system-ui,sans-serif">{title}</text>
  {grid}
  <polyline points="{path(y1)}" fill="none" stroke="var(--series-1)" stroke-width="2" stroke-linejoin="round"/>
  <polyline points="{path(y2)}" fill="none" stroke="var(--cluster-1)" stroke-width="2" stroke-linejoin="round"/>
  <text x="{width - pad_r}" y="{height - 4}" font-size="8" fill="{MUTED}" text-anchor="end" font-family="system-ui,sans-serif">r (A)</text>
</svg>'''


def compute_all(names: list[str] | None = None, results_dir: Path | None = None):
    names = PURE_SOLVENTS if names is None else names
    results_dir = RESULTS_DIR if results_dir is None else results_dir
    symbols, coords = read_opa_vasp(OPA_VASP)
    topo = classify_head_tail(symbols, coords)
    print(f"Head atoms (n={len(topo['head'])}): {topo['head']}")
    print(f"Terminal tail carbon index: {topo['terminal_c']}")

    rows = []
    panels = []
    for name in names:
        traj_path = results_dir / name / f"{name}_md_300K.traj"
        if not traj_path.exists():
            print(f"Skip {name}: no trajectory")
            continue
        prof = solvation_profile(traj_path, topo["head"], topo["terminal_c"], topo["n_opa_atoms"])
        head_r_peak, head_g_peak = first_peak(prof["r"], prof["g_head"])
        tail_r_peak, tail_g_peak = first_peak(prof["r"], prof["g_tail"])
        head_coord = coord_number_at(prof["r"], prof["running_head"], COORD_CUTOFF_A)
        tail_coord = coord_number_at(prof["r"], prof["running_tail"], COORD_CUTOFF_A)
        rows.append({
            "solvent": name,
            "head_first_peak_r_A": head_r_peak,
            "head_first_peak_g": head_g_peak,
            "head_coord_number_4A": head_coord,
            "tail_first_peak_r_A": tail_r_peak,
            "tail_first_peak_g": tail_g_peak,
            "tail_coord_number_4A": tail_coord,
            "head_tail_coord_ratio": head_coord / tail_coord if tail_coord else float("nan"),
            "n_solvent_atoms": prof["n_solvent_atoms"],
            "box_length_A": prof["box_length"],
        })
        panels.append(
            f'<div class="panel">{line_svg(prof["r"], prof["g_head"], prof["g_tail"], name, "g(r)")}</div>'
        )
        print(f"{name}: head_coord@4A={head_coord:.2f}  tail_coord@4A={tail_coord:.2f}  "
              f"head_peak=({head_r_peak:.1f}A, {head_g_peak:.2f})  tail_peak=({tail_r_peak:.1f}A, {tail_g_peak:.2f})")

    return topo, rows, panels


def main(names: list[str] | None = None, results_dir: Path | None = None, out_dir: Path | None = None) -> None:
    out_dir = OUT_DIR if out_dir is None else out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    topo, rows, panels = compute_all(names, results_dir)

    summary_path = out_dir / "solvation_structure_summary.csv"
    field_order = [
        "solvent", "head_first_peak_r_A", "head_first_peak_g", "head_coord_number_4A",
        "tail_first_peak_r_A", "tail_first_peak_g", "tail_coord_number_4A",
        "head_tail_coord_ratio", "n_solvent_atoms", "box_length_A",
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=field_order)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"Wrote {summary_path}")

    fragment_path = out_dir / "_solvation_rdf_panels.html"
    fragment_path.write_text(
        f'<div class="grid">{"".join(panels)}</div>\n'
        f'<p class="caveat">Blue = solvent g(r) around the phosphonate headgroup '
        f'(P + 3 O + 2 acidic H). Orange = solvent g(r) around the terminal (C18) '
        f'alkyl carbon. Coordination number reported at r={COORD_CUTOFF_A} A.</p>',
        encoding="utf-8",
    )
    print(f"Wrote {fragment_path}")


if __name__ == "__main__":
    main()
