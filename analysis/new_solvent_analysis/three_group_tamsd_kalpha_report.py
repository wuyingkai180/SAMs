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
    ax.scatter(
        x,
        y,
        c=frame["color_hex"],
        marker=frame["marker"].iloc[0],
        s=68,
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
                fontsize=6.2,
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
    right = fig.add_subplot(grid[0, 1], sharey=left)
    colors = data["color_hex"].tolist()
    left.barh(y, -data["mean_force_eV_A"], color=colors, edgecolor="#202428", linewidth=0.45, height=0.72)
    right.barh(y, data["max_disp_A"], color=colors, edgecolor="#202428", linewidth=0.45, height=0.72)
    left.set_xlim(-force_max, 0.0)
    right.set_xlim(0.0, disp_max)
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
        ("three_groups_bidirectional_mean_force_max_displacement.png", "双向柱形图：左侧为平均力，右侧为最大位移；两侧使用独立物理单位。"),
        ("three_groups_all_displacement_curves.png", "全部26条OPA相对初始位置的位移曲线。"),
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
<div class="note">定义：TAMSD(τ)=Kα·τ<sup>α</sup>。散点图横轴为Kα，纵轴为α。所有条件均采用唯一颜色，且颜色编码已写入raw CSV。每个条件只有一条10 ps轨迹，因此结果属于描述性比较。</div>
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
    outputs.extend(_plot_bidirectional_bars(metrics, figures))
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
    bar_path = tables / "three_groups_bidirectional_mean_force_max_displacement.csv"
    metrics[[
        "group", "system", "display_label", "color_hex", "marker", "mean_force_eV_A",
        "max_disp_A", "alpha", "alpha_prefactor_A2_ps_alpha",
    ]].rename(columns={"alpha_prefactor_A2_ps_alpha": "K_alpha_A2_ps_minus_alpha"}).to_csv(
        bar_path, index=False, encoding="utf-8-sig"
    )
    outputs.append(bar_path)
    outputs.append(_build_report(metrics, output_root))
    return outputs


if __name__ == "__main__":
    build(Path(__file__).resolve().parents[2])
