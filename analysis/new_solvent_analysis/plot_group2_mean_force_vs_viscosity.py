"""Plot Group 2 mean force against log10 solvent viscosity."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress, t


ROOT = Path(__file__).resolve().parents[2]
INPUT = (
    ROOT
    / "results"
    / "result2"
    / "group2_lda_external_properties"
    / "tables"
    / "group2_external_md_merged.csv"
)
OUT_FIGURES = ROOT / "results" / "result2" / "figures"
OUT_TABLES = ROOT / "results" / "result2" / "tables"
STEM = "group2_mean_force_vs_log10_viscosity"


def main() -> None:
    OUT_FIGURES.mkdir(parents=True, exist_ok=True)
    OUT_TABLES.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(INPUT)
    required = {"system", "log10_viscosity_mPa_s", "mean_force_eV_A"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    plot_data = data[
        ["system", "log10_viscosity_mPa_s", "viscosity_mPa_s", "mean_force_eV_A"]
    ].dropna()

    x = plot_data["log10_viscosity_mPa_s"].to_numpy(float)
    y = plot_data["mean_force_eV_A"].to_numpy(float)
    fit = linregress(x, y)
    x_grid = np.linspace(x.min(), x.max(), 300)
    y_fit = fit.intercept + fit.slope * x_grid

    # Pointwise 95% confidence interval for the fitted mean response.
    n = len(x)
    residuals = y - (fit.intercept + fit.slope * x)
    residual_se = np.sqrt(np.sum(residuals**2) / (n - 2))
    sxx = np.sum((x - x.mean()) ** 2)
    critical_t = t.ppf(0.975, df=n - 2)
    fit_se = residual_se * np.sqrt(1 / n + (x_grid - x.mean()) ** 2 / sxx)
    lower = y_fit - critical_t * fit_se
    upper = y_fit + critical_t * fit_se

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 9,
            "axes.linewidth": 0.9,
            "axes.edgecolor": "#202428",
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(89 / 25.4, 78 / 25.4), constrained_layout=True)
    ax.fill_between(x_grid, lower, upper, color="#68737B", alpha=0.16, linewidth=0)
    ax.plot(x_grid, y_fit, color="#4F5960", linewidth=1.4, zorder=2)

    colors = {"acetone": "#D95F02", "n-heptane": "#0072B2"}
    for row in plot_data.itertuples(index=False):
        highlighted = row.system in colors
        ax.scatter(
            row.log10_viscosity_mPa_s,
            row.mean_force_eV_A,
            s=58 if highlighted else 31,
            color=colors.get(row.system, "#A7AEB3"),
            edgecolor="#202428",
            linewidth=0.75,
            zorder=4 if highlighted else 3,
        )
        if highlighted:
            offset = (6, 6) if row.system == "acetone" else (6, -13)
            ax.annotate(
                row.system,
                (row.log10_viscosity_mPa_s, row.mean_force_eV_A),
                xytext=offset,
                textcoords="offset points",
                fontsize=8.2,
                fontweight="bold",
            )

    ax.text(
        0.97,
        0.96,
        f"Pearson r = {fit.rvalue:.2f}\np = {fit.pvalue:.3f}\nn = {n}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.2,
    )
    ax.set_xlabel("log10 viscosity (mPa s)")
    ax.set_ylabel("Mean force (eV Å$^{-1}$)")
    ax.set_title("Mean force versus solvent viscosity", fontsize=10, fontweight="bold")
    ax.grid(True, color="#D8DDE1", linewidth=0.55, alpha=0.65)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.9)
        spine.set_color("#202428")

    base = OUT_FIGURES / STEM
    fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    plot_data.to_csv(OUT_TABLES / f"{STEM}_source_data.csv", index=False)
    pd.DataFrame(
        [
            {
                "n": n,
                "slope": fit.slope,
                "intercept": fit.intercept,
                "pearson_r": fit.rvalue,
                "r_squared": fit.rvalue**2,
                "p_value_two_sided": fit.pvalue,
                "slope_standard_error": fit.stderr,
                "confidence_band": "pointwise 95% CI for fitted mean",
            }
        ]
    ).to_csv(OUT_TABLES / f"{STEM}_linear_fit.csv", index=False)
    print(base)


if __name__ == "__main__":
    main()
