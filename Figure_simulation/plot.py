"""Reproduce the three main OPA/SAM figures from CSV files in this folder.

Run from any working directory:
    python plot_three_figures.py

Outputs are written to the current tables directory as PNG files.
"""

from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


HERE = Path(__file__).resolve().parent
OUT = HERE
GROUPS = ("group1", "group2", "group3")
GROUP_NAMES = {"group1": "Group 1", "group2": "Group 2", "group3": "Group 3"}
GROUP_TITLES = {
    "group1": "Group 1 | Acetone concentration series in n-heptane",
    "group2": "Group 2 | Pure-solvent comparison",
    "group3": "Group 3 | Solvent-composition comparison",
}


def style():
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 8.5, "axes.linewidth": 1.0, "axes.edgecolor": "#202428",
        "svg.fonttype": "none", "pdf.fonttype": 42,
    })


def boxed(ax):
    for spine in ax.spines.values():
        spine.set_visible(True); spine.set_color("#202428"); spine.set_linewidth(1.0)
    ax.grid(True, color="#D7DCE0", linewidth=0.55, alpha=0.65)
    ax.set_axisbelow(True)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.png"
    fig.savefig(path, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def load_tables():
    summary = pd.read_csv(HERE / "three_groups_TAMSD_Kalpha_fit_summary.csv")
    if "alpha_prefactor_A2_ps_alpha" not in summary.columns and "K_alpha_A2_ps_minus_alpha" in summary.columns:
        summary = summary.rename(columns={"K_alpha_A2_ps_minus_alpha": "alpha_prefactor_A2_ps_alpha"})
    paired = pd.read_csv(HERE / "three_groups_force_displacement_dual_axis_points.csv")
    motion = pd.read_csv(HERE / "three_groups_displacement_curves_raw_from_excel.csv")
    return summary, paired, motion


def k_alpha(summary):
    x = summary["alpha_prefactor_A2_ps_alpha"]
    y = summary["alpha"]
    xlim = (max(0, x.min() - 0.08), x.max() + 0.08)
    ylim = (y.min() - 0.04, y.max() + 0.04)
    for group in GROUPS:
        q = summary[summary.group == group]
        fig, ax = plt.subplots(figsize=(7.6, 5.8), constrained_layout=True)
        for marker, p in q.groupby("marker", sort=False):
            ax.scatter(p["alpha_prefactor_A2_ps_alpha"], p["alpha"], c=p.color_hex,
                       marker=marker, s=102, edgecolor="#171A1D", linewidth=0.8, zorder=3)
        for row in q.itertuples(index=False):
            ax.annotate(row.display_label, (row.alpha_prefactor_A2_ps_alpha, row.alpha),
                        xytext=(4, 3), textcoords="offset points", fontsize=7.2,
                        arrowprops={"arrowstyle": "-", "color": "#7A8085", "lw": 0.45})
        ax.axhline(1, color="#656A6F", lw=0.9, ls="--")
        ax.set(xlabel=r"TAMSD prefactor, $K_\alpha$ ($\mathrm{\AA^2\,ps^{-\alpha}}$)",
               ylabel=r"TAMSD exponent, $\alpha$", xlim=xlim, ylim=ylim)
        ax.set_title(GROUP_TITLES[group] + "\nTAMSD amplitude and scaling exponent", fontsize=11, fontweight="bold")
        boxed(ax); save(fig, f"{group}_Kalpha_vs_alpha")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True, constrained_layout=True)
    for letter, ax, group in zip("abc", axes, GROUPS):
        q = summary[summary.group == group]
        for marker, p in q.groupby("marker", sort=False):
            ax.scatter(p["alpha_prefactor_A2_ps_alpha"], p["alpha"], c=p.color_hex,
                       marker=marker, s=102, edgecolor="#171A1D", linewidth=0.8, zorder=3)
        for row in q.itertuples(index=False):
            ax.annotate(row.display_label, (row.alpha_prefactor_A2_ps_alpha, row.alpha),
                        xytext=(3, 2), textcoords="offset points", fontsize=6.2,
                        arrowprops={"arrowstyle": "-", "color": "#7A8085", "lw": 0.4})
        ax.axhline(1, color="#656A6F", lw=0.9, ls="--")
        ax.set(xlabel=r"TAMSD prefactor, $K_\alpha$ ($\mathrm{\AA^2\,ps^{-\alpha}}$)",
               ylabel=r"TAMSD exponent, $\alpha$", xlim=xlim, ylim=ylim)
        ax.set_title(GROUP_NAMES[group], fontsize=10, fontweight="bold")
        ax.text(-0.11, 1.05, letter, transform=ax.transAxes, fontsize=11, fontweight="bold")
        boxed(ax)
    fig.suptitle(r"Three-group comparison of $K_\alpha$ and $\alpha$", fontsize=13, fontweight="bold")
    save(fig, "three_groups_Kalpha_vs_alpha_multipanel")


def paired_points(paired):
    q = paired.sort_values(["group", "display_label"]).reset_index(drop=True)
    y = np.arange(len(q))
    labels = [f"{GROUP_NAMES[g]} | {s}" for g, s in zip(q.group, q.display_label)]
    fig, (left, right) = plt.subplots(1, 2, figsize=(11.2, 11.5), sharey=True,
                                      gridspec_kw={"wspace": 0.06}, constrained_layout=False)
    fig.subplots_adjust(left=0.29, right=0.98, top=0.88, bottom=0.08)
    left.scatter(q.mean_force_eV_A, y, c=q.color_hex, marker="o", s=92,
                 edgecolor="#202428", linewidth=0.8)
    right.scatter(q.max_disp_A, y, c=q.color_hex, marker="D", s=82,
                  edgecolor="#202428", linewidth=0.8)
    left.set_yticks(y); left.tick_params(axis="y", labelleft=False); left.invert_yaxis()
    right.tick_params(axis="y", left=False, labelleft=False)
    left.set_xlim(0.35, 0.80); right.set_xlim(0, 8)
    left.set_xlabel("Mean force magnitude on OPA, |F| (eV Å⁻¹)")
    right.set_xlabel("Maximum OPA displacement (Å)")
    left.set_title("Mean force", fontsize=11, fontweight="bold")
    right.set_title("Maximum displacement", fontsize=11, fontweight="bold")
    for ax in (left, right): boxed(ax)
    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor="#5B6770", markeredgecolor="#202428", markersize=8, label="Mean force"),
               Line2D([0], [0], marker="D", color="w", markerfacecolor="#5B6770", markeredgecolor="#202428", markersize=7, label="Maximum displacement")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.925), ncol=2,
               frameon=True, edgecolor="#202428", fontsize=8)
    fig.suptitle("Three groups | aligned comparisons without a correlation fit", fontsize=13, fontweight="bold", y=0.975)
    save(fig, "three_groups_force_displacement_paired_points")


def displacement(motion, summary):
    colors = dict(summary.drop_duplicates("display_label").set_index("display_label")["color_hex"])
    fig, ax = plt.subplots(figsize=(15, 9), constrained_layout=False)
    for (group, system, label), q in motion.groupby(["group", "system", "display_label"], sort=False):
        ax.plot(q.time_ps, q.OPA_disp_norm_A, color=colors.get(label, "#4C78A8"),
                lw=1.15, label=f"{group} | {label}")
    ax.set(xlabel="Simulation time (ps)", ylabel="OPA displacement from t=0 (Å)")
    ax.set_title("All groups | complete OPA displacement trajectories", fontsize=13, fontweight="bold")
    boxed(ax)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=4,
               fontsize=6.1, frameon=True, edgecolor="#202428", fancybox=False)
    fig.subplots_adjust(bottom=0.23)
    save(fig, "three_groups_all_displacement_curves")


def color_annotation(summary):
    """Create the compact key used to decode marker shapes and colors."""
    data = summary.drop_duplicates("display_label").copy()
    data["_fraction"] = np.nan
    data = data.sort_values(["group", "display_label"]).reset_index(drop=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharex=True, constrained_layout=True)
    for ax, group in zip(axes, GROUPS):
        q = data[data.group == group].reset_index(drop=True)
        y = np.arange(len(q))[::-1] * 0.84
        for ypos, row in zip(y, q.itertuples(index=False)):
            ax.scatter(0.06, ypos, s=230, color=row.color_hex, marker=row.marker,
                       edgecolor="#202428", linewidth=0.7)
            ax.text(0.13, ypos, row.display_label, va="center", ha="left", fontsize=9.5)
            ax.text(0.98, ypos, row.color_hex, va="center", ha="right", fontsize=8.5, family="monospace")
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.75, max(float(y.max()) + 0.75, 1.0))
        ax.set_title(GROUP_NAMES[group], fontsize=10.5, fontweight="bold", pad=5)
        ax.axis("off")
    save(fig, "three_groups_unified_color_annotation")


if __name__ == "__main__":
    style()
    summary, paired, motion = load_tables()
    k_alpha(summary)
    paired_points(paired)
    displacement(motion, summary)
    color_annotation(summary)
    print(f"Wrote PNG figures to {OUT}")
