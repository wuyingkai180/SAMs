"""Build publication figures and one tidy raw CSV for three-group TAMSD analysis."""

from __future__ import annotations

import colorsys
import html
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

from publication_reporting import GROUP_TITLES, _boxed_axes, _panel, _save, _style, display_label


GROUPS = ("group1", "group2", "group3")
GROUP_SHORT = {"group1": "Group 1", "group2": "Group 2", "group3": "Group 3"}
GROUP_LINESTYLES = {"group1": "-", "group2": "-", "group3": "-"}
MARKER_CYCLE = ("o", "s", "^", "D", "P", "X", "v", "<", ">", "p", "h", "8", "*")


def _distinct_colors(n: int) -> list[str]:
    """Generate deterministic, non-repeating colors with alternating lightness."""
    golden_ratio = 0.618033988749895
    colors: list[str] = []
    for index in range(n):
        hue = (0.07 + index * golden_ratio) % 1.0
        lightness = (0.42, 0.52, 0.36)[index % 3]
        saturation = (0.78, 0.68)[index % 2]
        colors.append(to_hex(colorsys.hls_to_rgb(hue, lightness, saturation)))
    if len(set(colors)) != n:
        raise AssertionError("color generation did not produce unique colors")
    return colors


def _tamsd(positions: np.ndarray, max_lag: int) -> np.ndarray:
    """Match the existing pipeline: mean squared 3D displacement at every lag."""
    return np.asarray(
        [
            np.mean(np.sum((positions[lag:] - positions[:-lag]) ** 2, axis=1))
            for lag in range(1, max_lag + 1)
        ],
        dtype=float,
    )


def _fit_tamsd(lag_ps: np.ndarray, values: np.ndarray) -> tuple[float, float, float]:
    """Match the existing log-log fit, excluding only the first lag point."""
    mask = (
        (np.arange(1, len(lag_ps) + 1) >= 2)
        & np.isfinite(lag_ps)
        & np.isfinite(values)
        & (lag_ps > 0)
        & (values > 0)
    )
    x = np.log(lag_ps[mask])
    y = np.log(values[mask])
    alpha, intercept = np.polyfit(x, y, 1)
    predicted = alpha * x + intercept
    residual = float(np.sum((y - predicted) ** 2))
    total = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1.0 if total == 0.0 and residual == 0.0 else 1.0 - residual / total
    return float(alpha), float(np.exp(intercept)), float(r_squared)


def _load_data(project_root: Path, result_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics_parts: list[pd.DataFrame] = []
    for group in GROUPS:
        # Keep every declared group condition.  Group 3 intentionally includes
        # acetone 5% / n-heptane 95%, even though its source trajectory is also
        # present in Group 1; it must remain visible as a Group-3 comparison.
        frame = pd.read_csv(result_root / "tables" / f"{group}_metrics.csv").copy()
        metrics_parts.append(frame)
    metrics = pd.concat(metrics_parts, ignore_index=True)
    metrics["display_label"] = metrics["system"].map(display_label)
    metrics["curve_id"] = metrics["group"] + " | " + metrics["display_label"]
    # Encode the chemical condition, not the group, so a repeated composition
    # (e.g. acetone 5% / n-heptane 95%) is visually identical across panels.
    unique_labels = list(dict.fromkeys(metrics["display_label"].tolist()))
    colors = _distinct_colors(len(unique_labels))
    color_by_label = dict(zip(unique_labels, colors))
    marker_by_label = {
        label: MARKER_CYCLE[index % len(MARKER_CYCLE)]
        for index, label in enumerate(unique_labels)
    }
    metrics["color_hex"] = metrics["display_label"].map(color_by_label)
    metrics["marker"] = metrics["display_label"].map(marker_by_label)
    color_map = dict(zip(metrics["curve_id"], metrics["color_hex"]))

    raw_parts: list[pd.DataFrame] = []
    for row in metrics.itertuples(index=False):
        csv_path = Path(str(row.csv_path))
        if not csv_path.is_absolute():
            csv_path = project_root / csv_path
        motion = pd.read_csv(csv_path)
        required = {
            "time_ps", "OPA_COM_x_A", "OPA_COM_y_A", "OPA_COM_z_A", "OPA_disp_norm_A"
        }
        missing = required - set(motion.columns)
        if missing:
            raise ValueError(f"{csv_path} missing columns: {sorted(missing)}")
        positions = motion[["OPA_COM_x_A", "OPA_COM_y_A", "OPA_COM_z_A"]].to_numpy(float)
        max_lag = max(2, len(positions) // 3)
        dt = float(np.median(np.diff(motion["time_ps"].to_numpy(float))))
        lag_ps = np.arange(1, max_lag + 1, dtype=float) * dt
        tamsd_values = _tamsd(positions, max_lag)
        alpha, prefactor, fit_r2 = _fit_tamsd(lag_ps, tamsd_values)
        if not np.isclose(alpha, float(row.alpha), rtol=1e-8, atol=1e-10):
            raise AssertionError(f"alpha mismatch for {row.group}/{row.system}: {alpha} vs {row.alpha}")
        if not np.isclose(prefactor, float(row.alpha_prefactor_A2_ps_alpha), rtol=1e-8, atol=1e-10):
            raise AssertionError(
                f"K_alpha mismatch for {row.group}/{row.system}: "
                f"{prefactor} vs {row.alpha_prefactor_A2_ps_alpha}"
            )
        common = {
            "group": row.group,
            "system": row.system,
            "display_label": row.display_label,
            "curve_id": row.curve_id,
            "color_hex": color_map[row.curve_id],
            "marker": row.marker,
            "alpha": alpha,
            "K_alpha_A2_ps_minus_alpha": prefactor,
            "alpha_fit_r2": fit_r2,
            "fit_tau_min_ps": float(row.alpha_fit_tau_min_ps),
            "fit_tau_max_ps": float(row.alpha_fit_tau_max_ps),
        }
        displacement_part = pd.DataFrame(
            {
                **common,
                "record_type": "displacement",
                "time_ps": motion["time_ps"].to_numpy(float),
                "OPA_COM_x_A": motion["OPA_COM_x_A"].to_numpy(float),
                "OPA_COM_y_A": motion["OPA_COM_y_A"].to_numpy(float),
                "OPA_COM_z_A": motion["OPA_COM_z_A"].to_numpy(float),
                "OPA_displacement_from_t0_A": motion["OPA_disp_norm_A"].to_numpy(float),
                "lag_ps": np.nan,
                "TAMSD_A2": np.nan,
                "TAMSD_powerlaw_fit_A2": np.nan,
                "in_alpha_fit_window": False,
            }
        )
        fit_curve = prefactor * np.power(lag_ps, alpha)
        tamsd_part = pd.DataFrame(
            {
                **common,
                "record_type": "tamsd",
                "time_ps": np.nan,
                "OPA_COM_x_A": np.nan,
                "OPA_COM_y_A": np.nan,
                "OPA_COM_z_A": np.nan,
                "OPA_displacement_from_t0_A": np.nan,
                "lag_ps": lag_ps,
                "TAMSD_A2": tamsd_values,
                "TAMSD_powerlaw_fit_A2": fit_curve,
                "in_alpha_fit_window": (
                    (lag_ps >= float(row.alpha_fit_tau_min_ps) - 1e-12)
                    & (lag_ps <= float(row.alpha_fit_tau_max_ps) + 1e-12)
                ),
            }
        )
        raw_parts.extend([displacement_part, tamsd_part])
    raw = pd.concat(raw_parts, ignore_index=True)
    return metrics, raw


def _scatter_points(ax: plt.Axes, frame: pd.DataFrame, *, annotate: bool = True) -> None:
    x = frame["alpha_prefactor_A2_ps_alpha"].to_numpy(float)
    y = frame["alpha"].to_numpy(float)
    # Draw each marker type separately so every point uses the same shape as
    # the unified annotation key and the trajectory plots.
    for marker, subset in frame.groupby("marker", sort=False):
        ax.scatter(
            subset["alpha_prefactor_A2_ps_alpha"],
            subset["alpha"],
            c=subset["color_hex"].tolist(),
            marker=marker,
            s=102,
            edgecolor="#171A1D",
            linewidth=0.8,
            zorder=3,
        )
    if annotate:
        x_span = max(float(np.ptp(x)), 0.2)
        y_span = max(float(np.ptp(y)), 0.1)
        offsets = ((0.025, 0.050), (0.025, -0.060), (-0.025, 0.070), (-0.025, -0.080))
        for index, row in enumerate(frame.itertuples(index=False)):
            dx, dy = offsets[index % len(offsets)]
            ax.annotate(
                row.display_label,
                (row.alpha_prefactor_A2_ps_alpha, row.alpha),
                xytext=(row.alpha_prefactor_A2_ps_alpha + dx * x_span, row.alpha + dy * y_span),
                fontsize=7.2,
                ha="left" if dx > 0 else "right",
                va="bottom" if dy > 0 else "top",
                arrowprops={"arrowstyle": "-", "color": "#6C7175", "lw": 0.45},
                bbox={"boxstyle": "square,pad=0.10", "fc": "white", "ec": "none", "alpha": 0.80},
                zorder=4,
            )
    ax.axhline(1.0, color="#656A6F", lw=0.9, ls="--", zorder=1)
    ax.set_xlabel(r"TAMSD prefactor, $K_\alpha$ ($\mathrm{\AA^2\,ps^{-\alpha}}$)")
    ax.set_ylabel(r"TAMSD exponent, $\alpha$")
    ax.margins(x=0.17, y=0.15)
    _boxed_axes(ax)


def _plot_kalpha(metrics: pd.DataFrame, figures: Path) -> list[Path]:
    outputs: list[Path] = []
    global_x = metrics["alpha_prefactor_A2_ps_alpha"]
    global_y = metrics["alpha"]
    x_pad = max(float(global_x.max() - global_x.min()) * 0.08, 0.08)
    y_pad = max(float(global_y.max() - global_y.min()) * 0.08, 0.04)
    xlim = (max(0.0, float(global_x.min()) - x_pad), float(global_x.max()) + x_pad)
    ylim = (float(global_y.min()) - y_pad, float(global_y.max()) + y_pad)
    for group in GROUPS:
        subset = metrics.loc[metrics["group"] == group].copy()
        fig, ax = plt.subplots(figsize=(7.6, 5.8), constrained_layout=True)
        _scatter_points(ax, subset)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_title(f"{GROUP_TITLES[group]}\nTAMSD amplitude and scaling exponent", fontsize=11, fontweight="bold")
        outputs.extend(_save(fig, figures / f"{group}_Kalpha_vs_alpha"))

    fig, axes = plt.subplots(1, 3, figsize=(18.0, 6.0), sharex=True, sharey=True, constrained_layout=True)
    for letter, ax, group in zip("abc", axes, GROUPS):
        subset = metrics.loc[metrics["group"] == group].copy()
        _scatter_points(ax, subset)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_title(GROUP_TITLES[group], fontsize=10, fontweight="bold")
        _panel(ax, letter)
    fig.suptitle(r"Three-group comparison of $K_\alpha$ and $\alpha$", fontsize=13, fontweight="bold")
    outputs.extend(_save(fig, figures / "three_groups_Kalpha_vs_alpha_multipanel"))
    return outputs


def _curve_legend(fig: plt.Figure, axes, ncol: int = 4) -> None:
    source = axes[0] if isinstance(axes, (list, tuple, np.ndarray)) else axes
    handles, labels = source.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=ncol,
        fontsize=6.1,
        frameon=True,
        edgecolor="#202428",
        fancybox=False,
    )


def _plot_group_curves(metrics: pd.DataFrame, raw: pd.DataFrame, figures: Path) -> list[Path]:
    outputs: list[Path] = []
    for group in GROUPS:
        subset_metrics = metrics.loc[metrics["group"] == group]
        fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.5), constrained_layout=True)
        for row in subset_metrics.itertuples(index=False):
            motion = raw.loc[(raw["curve_id"] == row.curve_id) & (raw["record_type"] == "displacement")]
            msd = raw.loc[(raw["curve_id"] == row.curve_id) & (raw["record_type"] == "tamsd")]
            label = row.display_label
            axes[0].plot(motion["time_ps"], motion["OPA_displacement_from_t0_A"], color=row.color_hex, lw=1.25, label=label)
            axes[1].plot(msd["lag_ps"], msd["TAMSD_A2"], color=row.color_hex, lw=1.35, label=label)
            fit = msd.loc[msd["in_alpha_fit_window"]]
            axes[1].plot(fit["lag_ps"], fit["TAMSD_powerlaw_fit_A2"], color=row.color_hex, lw=0.9, ls="--", alpha=0.9)
        axes[0].set(xlabel="Simulation time (ps)", ylabel="OPA displacement from t=0 (Å)")
        axes[1].set(xlabel="Lag time, τ (ps)", ylabel="TAMSD(τ) (Å²)")
        axes[1].set_xscale("log")
        axes[1].set_yscale("log")
        for letter, ax in zip("ab", axes):
            _boxed_axes(ax)
            _panel(ax, letter)
        _curve_legend(fig, axes, ncol=min(4, len(subset_metrics)))
        fig.suptitle(f"{GROUP_TITLES[group]} | displacement and TAMSD curves", fontsize=12, fontweight="bold")
        fig.subplots_adjust(bottom=0.18)
        outputs.extend(_save(fig, figures / f"{group}_displacement_and_TAMSD_curves"))
    return outputs


def _plot_all_curves(metrics: pd.DataFrame, raw: pd.DataFrame, figures: Path) -> list[Path]:
    outputs: list[Path] = []
    fig, ax = plt.subplots(figsize=(15.0, 9.0), constrained_layout=True)
    for row in metrics.itertuples(index=False):
        data = raw.loc[(raw["curve_id"] == row.curve_id) & (raw["record_type"] == "displacement")]
        ax.plot(
            data["time_ps"], data["OPA_displacement_from_t0_A"],
            color=row.color_hex, lw=1.15, ls=GROUP_LINESTYLES[row.group], label=row.curve_id,
        )
    ax.set(xlabel="Simulation time (ps)", ylabel="OPA displacement from t=0 (Å)")
    ax.set_title("All groups | complete OPA displacement trajectories", fontsize=13, fontweight="bold")
    _boxed_axes(ax)
    _curve_legend(fig, ax, ncol=4)
    fig.subplots_adjust(bottom=0.23)
    outputs.extend(_save(fig, figures / "three_groups_all_displacement_curves"))

    fig, ax = plt.subplots(figsize=(15.0, 9.0), constrained_layout=True)
    for row in metrics.itertuples(index=False):
        data = raw.loc[(raw["curve_id"] == row.curve_id) & (raw["record_type"] == "tamsd")]
        ax.plot(data["lag_ps"], data["TAMSD_A2"], color=row.color_hex, lw=1.25, label=row.curve_id)
    ax.set(xlabel="Lag time, τ (ps)", ylabel="TAMSD(τ) (Å²)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("All groups | time-averaged mean-squared displacement", fontsize=13, fontweight="bold")
    _boxed_axes(ax)
    _curve_legend(fig, ax, ncol=4)
    fig.subplots_adjust(bottom=0.23)
    outputs.extend(_save(fig, figures / "three_groups_all_TAMSD_curves"))
    return outputs


def _plot_bidirectional_bars(metrics: pd.DataFrame, figures: Path) -> list[Path]:
    """Plot mean force to the left and maximum displacement to the right."""
    data = metrics.sort_values(["group", "display_label"]).reset_index(drop=True)
    y = np.arange(len(data))
    force_max = float(data["mean_force_eV_A"].max()) * 1.30
    disp_max = float(data["max_disp_A"].max()) * 1.20
    fig = plt.figure(figsize=(16.0, 12.5), constrained_layout=False)
    grid = fig.add_gridspec(1, 2, width_ratios=(1, 1), wspace=0.0, left=0.13, right=0.97, top=0.91, bottom=0.12)
    left = fig.add_subplot(grid[0, 0])
    # Do not share the y-axis: left and right rankings are intentionally independent.
    right = fig.add_subplot(grid[0, 1])
    colors = data["color_hex"].tolist()
    left.barh(y, -data["mean_force_eV_A"], color=colors, edgecolor="#202428", linewidth=0.45, height=0.72)
    right.barh(y, data["max_disp_A"], color=colors, edgecolor="#202428", linewidth=0.45, height=0.72)
    # Fixed publication scales requested by the user: 0–0.8 and 0–8.
    left.set_xlim(-0.8, 0.0)
    right.set_xlim(0.0, 8.0)
    left.set_yticks(y, [f"{GROUP_SHORT[g]} | {label}" for g, label in zip(data["group"], data["display_label"])])
    left.tick_params(axis="y", labelsize=7.0)
    right.tick_params(axis="y", left=False, labelleft=False)
    left.set_xlabel("Mean force magnitude on OPA, |F| (eV Å⁻¹)")
    right.set_xlabel("Maximum OPA displacement (Å)")
    left.xaxis.set_label_position("top")
    right.xaxis.set_label_position("top")
    left.xaxis.tick_top()
    right.xaxis.tick_top()
    left.xaxis.set_major_formatter(FuncFormatter(lambda value, _position: f"{abs(value):.2f}"))
    # With xlim=(-max_force, 0), negative bar widths extend left from the divider.
    left.axvline(0.0, color="#202428", lw=1.2)
    right.axvline(0.0, color="#202428", lw=1.2)
    for ax in (left, right):
        _boxed_axes(ax)
        ax.grid(True, axis="x", color="#D7DCE0", linewidth=0.55, alpha=0.65)
        ax.grid(False, axis="y")
    # Group separators and labels improve readability without adding a second legend.
    group_starts = [0]
    for index in range(1, len(data)):
        if data.loc[index, "group"] != data.loc[index - 1, "group"]:
            group_starts.append(index)
    for start in group_starts[1:]:
        for ax in (left, right):
            ax.axhline(start - 0.5, color="#687078", lw=0.9)
    fig.suptitle("Three groups | mean force versus maximum displacement", fontsize=14, fontweight="bold", y=0.965)
    fig.text(0.5, 0.045, "Each row is one solvent or solvent composition; colors match the unified raw CSV.", ha="center", fontsize=9)
    return _save(fig, figures / "three_groups_bidirectional_mean_force_max_displacement")


def _plot_sorted_bar(
    metrics: pd.DataFrame,
    figures: Path,
    column: str,
    xlabel: str,
    stem: str,
    title: str,
    value_format: str,
) -> list[Path]:
    """Plot one metric as a readable all-condition horizontal ranking."""
    data = metrics.sort_values(column, ascending=True).reset_index(drop=True)
    y = np.arange(len(data))
    values = data[column].to_numpy(float)
    fig, ax = plt.subplots(figsize=(14.5, 11.5), constrained_layout=True)
    bars = ax.barh(
        y,
        values,
        color=data["color_hex"].tolist(),
        edgecolor="#202428",
        linewidth=0.55,
        height=0.70,
    )
    ax.set_yticks(
        y,
        [f"{GROUP_SHORT[g]} | {label}" for g, label in zip(data["group"], data["display_label"])],
    )
    ax.tick_params(axis="y", labelsize=7.2)
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlim(0.0, float(values.max()) * 1.16)
    ax.grid(True, axis="x", color="#D7DCE0", linewidth=0.55, alpha=0.65)
    ax.grid(False, axis="y")
    _boxed_axes(ax)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_width() + float(values.max()) * 0.012,
            bar.get_y() + bar.get_height() / 2,
            format(value, value_format),
            va="center",
            ha="left",
            fontsize=6.8,
        )
    ax.text(
        0.995,
        -0.055,
        "Sorted ascending; largest condition appears at the top.",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
    )
    return _save(fig, figures / stem)


def _plot_combined_sorted_bars(metrics: pd.DataFrame, figures: Path) -> list[Path]:
    """Combine the two descending rankings in one publication-style figure."""
    # The same acetone 5% / n-heptane 95% composition occurs in Group 1 and
    # Group 3. Keep the first (Group 1 concentration-series) record once.
    data = metrics.drop_duplicates(subset=["display_label"], keep="first").copy()
    fig, axes = plt.subplots(1, 2, figsize=(18.0, 12.5), sharey=False, constrained_layout=True)
    panels = (
        ("mean_force_eV_A", "Mean force magnitude on OPA, |F| (eV Å⁻¹)", "Mean force (descending)", ".3f"),
        ("max_disp_A", "Maximum OPA displacement (Å)", "Maximum displacement (descending)", ".2f"),
    )
    for ax, (column, xlabel, title, value_format) in zip(axes, panels):
        ordered = data.sort_values(column, ascending=False).reset_index(drop=True)
        y = np.arange(len(ordered))
        values = ordered[column].to_numpy(float)
        bars = ax.barh(
            y,
            values,
            color=ordered["color_hex"].tolist(),
            edgecolor="#202428",
            linewidth=0.55,
            height=0.70,
        )
        ax.invert_yaxis()
        ax.set_yticks(
            y,
            [f"{GROUP_SHORT[g]} | {label}" for g, label in zip(ordered["group"], ordered["display_label"])],
        )
        ax.tick_params(axis="y", labelsize=6.8)
        ax.set_xlabel(xlabel)
        ax.set_title(title, fontsize=11.5, fontweight="bold")
        ax.set_xlim(0.0, float(values.max()) * 1.22)
        ax.grid(True, axis="x", color="#D7DCE0", linewidth=0.55, alpha=0.65)
        ax.grid(False, axis="y")
        _boxed_axes(ax)
        for bar, value, label in zip(bars, values, ordered["display_label"]):
            ax.text(
                bar.get_width() + float(values.max()) * 0.012,
                bar.get_y() + bar.get_height() / 2,
                f"{format(value, value_format)}",
                va="center",
                ha="left",
                fontsize=6.3,
            )
    fig.suptitle(
        "Three groups | descending rankings of mean force and maximum displacement",
        fontsize=14,
        fontweight="bold",
    )
    return _save(fig, figures / "three_groups_sorted_force_max_displacement_combined")


def _plot_bidirectional_sorted_separate_labels(metrics: pd.DataFrame, figures: Path) -> list[Path]:
    """One diverging chart with independent left/right rankings and labels."""
    data = metrics.drop_duplicates(subset=["display_label"], keep="first").copy()
    left_data = data.sort_values("mean_force_eV_A", ascending=False).reset_index(drop=True)
    right_data = data.sort_values("max_disp_A", ascending=False).reset_index(drop=True)
    y = np.arange(len(data))
    force_max = float(left_data["mean_force_eV_A"].max()) * 1.30
    disp_max = float(right_data["max_disp_A"].max()) * 1.20
    fig = plt.figure(figsize=(18.0, 13.0), constrained_layout=False)
    grid = fig.add_gridspec(1, 2, width_ratios=(1, 1), wspace=0.0, left=0.16, right=0.96, top=0.91, bottom=0.09)
    left = fig.add_subplot(grid[0, 0])
    # Left and right lists are independently ranked; do not share tick labels.
    right = fig.add_subplot(grid[0, 1])
    left_values = left_data["mean_force_eV_A"].to_numpy(float)
    right_values = right_data["max_disp_A"].to_numpy(float)
    left_bars = left.barh(y, -left_values, color=left_data["color_hex"].tolist(), edgecolor="#202428", linewidth=0.55, height=0.70)
    right_bars = right.barh(y, right_values, color=right_data["color_hex"].tolist(), edgecolor="#202428", linewidth=0.55, height=0.70)
    left.invert_yaxis()
    right.invert_yaxis()
    right.set_ylim(left.get_ylim())
    left.set_xlim(-force_max, 0.0)
    right.set_xlim(0.0, disp_max)
    left.set_yticks(y, [f"{GROUP_SHORT[g]} | {label}" for g, label in zip(left_data["group"], left_data["display_label"])])
    right.set_yticks(y, [f"{GROUP_SHORT[g]} | {label}" for g, label in zip(right_data["group"], right_data["display_label"])])
    left.tick_params(axis="y", labelsize=6.6, pad=5)
    right.tick_params(axis="y", labelsize=6.6, pad=5, labelright=True, labelleft=False, right=True, left=False)
    right.yaxis.tick_right()
    left.set_xlabel("Mean force magnitude on OPA, |F| (eV Å⁻¹)")
    right.set_xlabel("Maximum OPA displacement (Å)")
    left.xaxis.set_label_position("top")
    right.xaxis.set_label_position("top")
    left.xaxis.tick_top()
    right.xaxis.tick_top()
    left.xaxis.set_major_formatter(FuncFormatter(lambda value, _position: f"{abs(value):.2f}"))
    left.axvline(0.0, color="#202428", lw=1.2)
    right.axvline(0.0, color="#202428", lw=1.2)
    for ax in (left, right):
        _boxed_axes(ax)
        ax.grid(True, axis="x", color="#D7DCE0", linewidth=0.55, alpha=0.65)
        ax.grid(False, axis="y")
    for bar, value in zip(left_bars, left_values):
        left.text(bar.get_width() - force_max * 0.012, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", ha="right", va="center", fontsize=6.3)
    for bar, value in zip(right_bars, right_values):
        right.text(bar.get_width() + disp_max * 0.012, bar.get_y() + bar.get_height() / 2, f"{value:.2f}", ha="left", va="center", fontsize=6.3)
    fig.suptitle("Three groups | independent descending rankings", fontsize=14, fontweight="bold", y=0.965)
    return _save(fig, figures / "three_groups_bidirectional_sorted_separate_labels")


def _plot_force_displacement_dual_axis(metrics: pd.DataFrame, figures: Path) -> list[Path]:
    """Plot force and displacement as two point series on one categorical y-axis."""
    data = metrics.drop_duplicates(subset=["display_label"], keep="first").copy()
    # Keep a stable, readable order while retaining group information in labels.
    data = data.sort_values(["group", "display_label"]).reset_index(drop=True)
    labels = [f"{GROUP_SHORT[g]} | {label}" for g, label in zip(data["group"], data["display_label"])]
    y = np.arange(len(data))
    fig, ax_force = plt.subplots(figsize=(15.5, 11.5), constrained_layout=False)
    fig.subplots_adjust(left=0.31, right=0.94, top=0.88, bottom=0.12)
    ax_disp = ax_force.twiny()
    force = data["mean_force_eV_A"].to_numpy(float)
    disp = data["max_disp_A"].to_numpy(float)
    colors = data["color_hex"].tolist()
    ax_force.scatter(force, y, c=colors, marker="o", s=74, edgecolor="#202428", linewidth=0.8,
                     label="Mean force", zorder=4)
    ax_disp.scatter(disp, y, c=colors, marker="D", s=66, edgecolor="#202428", linewidth=0.8,
                    label="Maximum displacement", zorder=5)
    ax_force.set_yticks(y, labels)
    ax_force.invert_yaxis()
    ax_force.set_xlabel("Mean force magnitude on OPA, |F| (eV Å⁻¹)", labelpad=10)
    ax_disp.set_xlabel("Maximum OPA displacement (Å)", labelpad=10)
    ax_force.xaxis.set_ticks_position("bottom")
    ax_force.xaxis.set_label_position("bottom")
    ax_disp.xaxis.set_ticks_position("top")
    ax_disp.xaxis.set_label_position("top")
    ax_force.set_xlim(0.0, 0.80)
    ax_disp.set_xlim(0.0, 8.0)
    ax_force.grid(True, axis="x", color="#D7DCE0", linewidth=0.55, alpha=0.65)
    ax_force.grid(False, axis="y")
    ax_force.tick_params(axis="y", labelsize=7.3)
    ax_force.tick_params(axis="x", labelsize=8)
    ax_disp.tick_params(axis="x", labelsize=8)
    _boxed_axes(ax_force)
    _boxed_axes(ax_disp)
    # A compact shape key explains the two metrics; colors are decoded in the
    # separate unified annotation figure.
    from matplotlib.lines import Line2D
    shape_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#5B6770", markeredgecolor="#202428", markersize=8, label="Mean force"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="#5B6770", markeredgecolor="#202428", markersize=7, label="Maximum displacement"),
    ]
    ax_force.legend(handles=shape_handles, loc="upper right", frameon=True, edgecolor="#202428", fontsize=8)
    fig.suptitle("Three groups | mean force and maximum displacement", fontsize=14, fontweight="bold", y=0.95)
    return _save(fig, figures / "three_groups_force_displacement_dual_axis_points")


def _plot_force_displacement_paired_points(metrics: pd.DataFrame, figures: Path) -> list[Path]:
    """Aligned dot plots for two metrics; avoids implying a correlation."""
    data = metrics.drop_duplicates(subset=["display_label"], keep="first").copy()
    # Group 1 is an acetone-concentration series: order it from low to high
    # concentration so the experimental trend is immediately readable.
    data["_sort_fraction"] = data["cosolvent_fraction"].where(data["group"].eq("group1"), np.nan)
    data = data.sort_values(["group", "_sort_fraction", "display_label"], na_position="last").reset_index(drop=True)
    y = np.arange(len(data))
    labels = [f"{GROUP_SHORT[g]} | {label}" for g, label in zip(data["group"], data["display_label"])]
    fig, (left, right) = plt.subplots(1, 2, figsize=(11.2, 11.5), sharey=True,
                                      gridspec_kw={"wspace": 0.06}, constrained_layout=False)
    fig.subplots_adjust(left=0.29, right=0.98, top=0.88, bottom=0.08)
    colors = data["color_hex"].tolist()
    left.scatter(data["mean_force_eV_A"], y, c=colors, marker="o", s=92,
                 edgecolor="#202428", linewidth=0.8, zorder=3)
    right.scatter(data["max_disp_A"], y, c=colors, marker="D", s=82,
                  edgecolor="#202428", linewidth=0.8, zorder=3)
    left.set_yticks(y)
    left.tick_params(axis="y", labelleft=False)
    left.invert_yaxis()
    right.tick_params(axis="y", left=False, labelleft=False)
    left.set_xlim(0.35, 0.80)
    right.set_xlim(0.0, 8.0)
    left.set_xlabel("Mean force magnitude on OPA, |F| (eV Å⁻¹)")
    right.set_xlabel("Maximum OPA displacement (Å)")
    left.set_title("Mean force", fontsize=11, fontweight="bold")
    right.set_title("Maximum displacement", fontsize=11, fontweight="bold")
    left.tick_params(axis="y", labelsize=7.2)
    for ax in (left, right):
        ax.grid(True, axis="x", color="#D7DCE0", linewidth=0.55, alpha=0.65)
        ax.grid(False, axis="y")
        _boxed_axes(ax)
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#5B6770", markeredgecolor="#202428", markersize=8, label="Mean force"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="#5B6770", markeredgecolor="#202428", markersize=7, label="Maximum displacement"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.925), ncol=2,
               frameon=True, edgecolor="#202428", fontsize=8)
    fig.suptitle("Three groups | aligned comparisons without a correlation fit",
                 fontsize=13, fontweight="bold", y=0.975)
    return _save(fig, figures / "three_groups_force_displacement_paired_points")


def _plot_force_vs_displacement_relationship(metrics: pd.DataFrame, figures: Path) -> list[Path]:
    """Show the direct relationship between mean force and maximum displacement."""
    data = metrics.drop_duplicates(subset=["display_label"], keep="first").copy()
    data = data.sort_values(["group", "display_label"]).reset_index(drop=True)
    group_colors = {"group1": "#24557A", "group2": "#D17C2F", "group3": "#3B7D6B"}
    group_markers = {"group1": "o", "group2": "s", "group3": "^"}
    fig, ax = plt.subplots(figsize=(12.5, 8.5), constrained_layout=True)
    for group in GROUPS:
        q = data.loc[data["group"] == group]
        x = q["mean_force_eV_A"].to_numpy(float)
        y = q["max_disp_A"].to_numpy(float)
        ax.scatter(x, y, s=70, color=group_colors[group], marker=group_markers[group],
                   edgecolor="#202428", linewidth=0.75, label=f"{GROUP_SHORT[group]} (n={len(q)})", zorder=3)
        if len(q) >= 3:
            slope, intercept = np.polyfit(x, y, 1)
            xx = np.linspace(x.min(), x.max(), 80)
            ax.plot(xx, slope * xx + intercept, color=group_colors[group], lw=1.2, alpha=0.75)
        # Label only the most separated points; the unified color/marker key
        # carries the complete composition names without overcrowding this
        # relationship panel.
        for row in q.itertuples(index=False):
            if row.max_disp_A >= 5.0 or row.max_disp_A <= 2.0:
                ax.annotate(row.display_label, (row.mean_force_eV_A, row.max_disp_A),
                            xytext=(4, 3), textcoords="offset points", fontsize=6.5,
                            color="#202428", ha="left", va="bottom",
                            arrowprops={"arrowstyle": "-", "color": "#7A8085", "lw": 0.45})
    x_all = data["mean_force_eV_A"].to_numpy(float)
    y_all = data["max_disp_A"].to_numpy(float)
    slope, intercept = np.polyfit(x_all, y_all, 1)
    xx = np.linspace(x_all.min(), x_all.max(), 120)
    ax.plot(xx, slope * xx + intercept, color="#202428", lw=1.5, ls="--", label="Overall linear trend", zorder=2)
    r = float(np.corrcoef(x_all, y_all)[0, 1])
    ax.text(0.03, 0.96, f"Overall Pearson r = {r:.2f}\n(n = {len(data)} unique conditions)",
            transform=ax.transAxes, va="top", ha="left", fontsize=8,
            bbox={"boxstyle": "square,pad=0.3", "fc": "white", "ec": "#202428", "alpha": 0.9})
    ax.set_xlabel("Mean force magnitude on OPA, |F| (eV Å⁻¹)")
    ax.set_ylabel("Maximum OPA displacement (Å)")
    ax.set_xlim(0.35, 0.80)
    ax.set_ylim(0.0, 8.0)
    ax.legend(loc="upper right", frameon=True, edgecolor="#202428", fontsize=8)
    _boxed_axes(ax)
    ax.grid(True, color="#D7DCE0", linewidth=0.55, alpha=0.65)
    fig.suptitle("Three groups | relationship between mean force and maximum displacement",
                 fontsize=13, fontweight="bold")
    return _save(fig, figures / "three_groups_force_vs_max_displacement_relationship")


def _plot_color_annotation(metrics: pd.DataFrame, figures: Path) -> list[Path]:
    """Export one reusable color/marker key for all figures."""
    data = metrics.drop_duplicates(subset=["display_label"], keep="first").copy()
    data["_sort_fraction"] = data["cosolvent_fraction"].where(data["group"].eq("group1"), np.nan)
    data = data.sort_values(["group", "_sort_fraction", "display_label"], na_position="last").reset_index(drop=True)
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4), sharex=True, constrained_layout=True)
    for ax, group in zip(axes, GROUPS):
        subset = data.loc[data["group"] == group].reset_index(drop=True)
        y = np.arange(len(subset))[::-1] * 0.84
        for ypos, row in zip(y, subset.itertuples(index=False)):
            ax.scatter(0.06, ypos, s=230, color=row.color_hex, marker=row.marker,
                       edgecolor="#202428", linewidth=0.7, zorder=3)
            ax.text(0.13, ypos, row.display_label, va="center", ha="left", fontsize=9.5)
            ax.text(0.98, ypos, row.color_hex, va="center", ha="right", fontsize=8.5, family="monospace")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(-0.75, max(float(y.max()) + 0.75, 1.0))
        ax.set_title(GROUP_SHORT[group], fontsize=10.5, fontweight="bold", pad=5)
        ax.axis("off")
    return _save(fig, figures / "three_groups_unified_color_annotation")


def _build_report(metrics: pd.DataFrame, output_root: Path) -> Path:
    reports = output_root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    table = metrics[[
        "group", "display_label", "color_hex", "marker", "alpha_prefactor_A2_ps_alpha",
        "alpha", "alpha_fit_r2", "alpha_fit_tau_min_ps", "alpha_fit_tau_max_ps",
    ]].copy()
    table.columns = ["Group", "System", "Color", "Marker", "Kα (Å² ps⁻α)", "α", "Fit R²", "τ min (ps)", "τ max (ps)"]
    cards = (
        ("three_groups_Kalpha_vs_alpha_multipanel.png", "三组Kα–α大图：相同组成条件在不同group中使用相同颜色和形状，其余条件使用不同编码。"),
        ("three_groups_all_displacement_curves.png", "全部26条OPA相对初始位置的位移曲线。"),
        ("three_groups_unified_color_annotation.png", "统一颜色和形状注释图，放置于全部轨迹图下方，作为所有图的公共图例。"),
        ("three_groups_force_displacement_paired_points.png", "平均力与最大位移并列点图：共享组分顺序但不显示文字标签。"),
        ("three_groups_all_TAMSD_curves.png", "全部26条TAMSD曲线；双对数坐标便于观察幂律斜率。"),
    )
    card_html = "".join(
        f'<figure><img src="../figures/{name}" alt="{html.escape(caption)}"><figcaption>{html.escape(caption)}</figcaption></figure>'
        for name, caption in cards
    )
    group_cards = "".join(
        f'<figure><img src="../figures/{group}_displacement_and_TAMSD_curves.png" alt="{group}"><figcaption>{html.escape(GROUP_TITLES[group])}：左为位移—时间，右为TAMSD—lag；虚线为幂律拟合。</figcaption></figure>'
        for group in GROUPS
    )
    report = reports / "index.html"
    report.write_text(
        f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>三组TAMSD与Kα综合图</title><style>
body{{font-family:Arial,'Microsoft YaHei',sans-serif;max-width:1500px;margin:auto;padding:28px;background:#edf1f4;color:#202428;line-height:1.65}}
main{{background:white;border:1.5px solid #202428;padding:32px 42px}}h1,h2{{color:#24557A}}.note{{border:1px solid #D17C2F;background:#fff8f0;padding:12px 16px}}
.grid{{display:grid;grid-template-columns:1fr;gap:20px}}figure{{border:1.5px solid #202428;padding:10px;margin:0;background:white}}img{{width:100%;height:auto}}figcaption{{font-size:13px;margin-top:7px}}
table{{width:100%;border-collapse:collapse;font-size:12px}}th,td{{border:1px solid #687078;padding:6px;text-align:right}}th:nth-child(2),td:nth-child(2){{text-align:left}}@media(max-width:900px){{main{{padding:18px}}}}
</style></head><body><main><h1>Group1–Group3：TAMSD、Kα与完整位移趋势</h1>
<div class="note">定义：TAMSD(τ)=Kα·τ<sup>α</sup>。散点图横轴为Kα，纵轴为α。每种组成条件在所有图中使用统一颜色和形状，颜色编码已写入raw CSV；重复组成在综合图中只保留一次。每个条件只有一条10 ps轨迹，因此结果属于描述性比较。当前目录仅保留600 dpi PNG图件；详见<a href="figure_contract.md">Figure contract</a>。</div>
<h2>合并图</h2><div class="grid">{card_html}</div><h2>各组曲线</h2><div class="grid">{group_cards}</div>
<h2>拟合参数与颜色</h2>{table.round(5).to_html(index=False, border=0, escape=False)}</main></body></html>""",
        encoding="utf-8",
    )
    return report


def build(project_root: Path) -> list[Path]:
    project_root = Path(project_root).resolve()
    result_root = project_root / "results" / "result2"
    output_root = result_root / "tamsd_kalpha_combined"
    figures = output_root / "figures"
    tables = output_root / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    _style()
    metrics, raw = _load_data(project_root, result_root)
    outputs: list[Path] = []
    outputs.extend(_plot_kalpha(metrics, figures))
    outputs.extend(_plot_group_curves(metrics, raw, figures))
    outputs.extend(_plot_all_curves(metrics, raw, figures))
    outputs.extend(_plot_force_displacement_dual_axis(metrics, figures))
    outputs.extend(_plot_force_displacement_paired_points(metrics, figures))
    outputs.extend(_plot_color_annotation(metrics, figures))
    raw_path = tables / "three_groups_TAMSD_Kalpha_raw_data.csv"
    raw.to_csv(raw_path, index=False, encoding="utf-8-sig")
    outputs.append(raw_path)
    summary_path = tables / "three_groups_TAMSD_Kalpha_fit_summary.csv"
    metrics[[
        "group", "system", "display_label", "curve_id", "color_hex", "marker",
        "alpha", "alpha_prefactor_A2_ps_alpha", "alpha_fit_r2",
        "alpha_fit_tau_min_ps", "alpha_fit_tau_max_ps", "alpha_fit_n_points",
    ]].rename(columns={"alpha_prefactor_A2_ps_alpha": "K_alpha_A2_ps_minus_alpha"}).to_csv(
        summary_path, index=False, encoding="utf-8-sig"
    )
    outputs.append(summary_path)
    dedup = metrics.drop_duplicates(subset=["display_label"], keep="first").copy()
    left_data = dedup.sort_values("mean_force_eV_A", ascending=False).reset_index(drop=True)
    right_data = dedup.sort_values("max_disp_A", ascending=False).reset_index(drop=True)
    n = len(dedup)
    bar_path = tables / "three_groups_force_displacement_dual_axis_points.csv"
    combined_bar = pd.DataFrame({
        "rank": np.arange(1, n + 1),
        "force_group": left_data["group"],
        "force_system": left_data["system"],
        "force_label": left_data["display_label"],
        "force_color_hex": left_data["color_hex"],
        "mean_force_eV_A": left_data["mean_force_eV_A"],
        "displacement_group": right_data["group"],
        "displacement_system": right_data["system"],
        "displacement_label": right_data["display_label"],
        "displacement_color_hex": right_data["color_hex"],
        "max_disp_A": right_data["max_disp_A"],
    })
    combined_bar.to_csv(bar_path, index=False, encoding="utf-8-sig")
    outputs.append(bar_path)
    outputs.append(_build_report(metrics, output_root))
    return outputs


if __name__ == "__main__":
    build(Path(__file__).resolve().parents[2])
