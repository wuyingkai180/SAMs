"""Labelled, publication-oriented groupwise figures and HTML reports."""

from __future__ import annotations

import html
import json
import re
import shutil
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PALETTE = {
    "blue": "#24557A",
    "orange": "#D17C2F",
    "green": "#3B7D6B",
    "purple": "#75518D",
    "red": "#B34A4A",
    "grey": "#676B70",
}

GROUP_TITLES = {
    "group1": "Group 1 | Acetone concentration series in n-heptane",
    "group2": "Group 2 | Pure-solvent comparison",
    "group3": "Group 3 | Solvent-composition comparison",
}

NAME_MAP = {
    "thf": "THF",
    "dmc": "DMC",
    "cpme": "CPME",
    "n-heptane": "n-heptane",
    "p-xylene": "p-xylene",
    "propylene_carbonate": "propylene carbonate",
    "ethyl_acetate": "ethyl acetate",
}

HSP_REFERENCE_FILE = (
    Path(__file__).resolve().parents[1] / "reference_data" / "hansen_pure_solvents.csv"
)


def _component_name(value: str) -> str:
    return NAME_MAP.get(value.lower(), value.replace("_", " "))


def display_label(system: str) -> str:
    """Return a compact, explicit label while preserving solvent composition."""
    match = re.fullmatch(r"(.+?)_(\d+)_n-heptane_(\d+)", system)
    if match:
        component, fraction, base_fraction = match.groups()
        return f"{_component_name(component)} {fraction}% / n-heptane {base_fraction}%"
    if system.endswith("_n-heptane"):
        return f"{_component_name(system[:-10])} / n-heptane"
    if system == "thf_toluene":
        return "THF / toluene"
    return _component_name(system)


def _style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "font.size": 8.5,
            "axes.linewidth": 1.0,
            "axes.edgecolor": "#202428",
            "axes.spines.top": True,
            "axes.spines.right": True,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.width": 0.9,
            "ytick.major.width": 0.9,
            "legend.frameon": True,
            "legend.edgecolor": "#202428",
            "legend.fancybox": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def _boxed_axes(ax: plt.Axes) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
        spine.set_color("#202428")
    ax.grid(True, color="#D7DCE0", linewidth=0.55, alpha=0.65, zorder=0)
    ax.set_axisbelow(True)


def _panel(ax: plt.Axes, letter: str) -> None:
    ax.text(
        -0.11,
        1.05,
        letter,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="bottom",
    )


def _save(fig: plt.Figure, stem: Path) -> list[Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    outputs = [stem.with_suffix(suffix) for suffix in (".png", ".svg", ".pdf")]
    fig.savefig(outputs[0], dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(outputs[1], bbox_inches="tight", facecolor="white")
    fig.savefig(outputs[2], bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return outputs


def _merge_pure_solvent_hsp(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach literature HSP descriptors to Group 2 without deriving them from MD."""
    hsp = pd.read_csv(HSP_REFERENCE_FILE)
    hsp["delta_total_MPa05"] = np.sqrt(
        hsp["delta_d_MPa05"] ** 2
        + hsp["delta_p_MPa05"] ** 2
        + hsp["delta_h_MPa05"] ** 2
    )
    # Make refresh idempotent when a previously enriched Group 2 table is reused.
    frame = frame.drop(columns=[column for column in hsp.columns if column != "system" and column in frame], errors="ignore")
    frame = frame.drop(columns="delta_total_MPa05", errors="ignore")
    merged = frame.merge(hsp, on="system", how="left", validate="one_to_one")
    missing = merged.loc[merged["delta_d_MPa05"].isna(), "system"].tolist()
    if missing:
        raise ValueError(f"missing HSP reference data for Group 2: {missing}")
    return merged


def _hsp_parameter_figure(frame: pd.DataFrame, out_dir: Path) -> list[Path]:
    """Render the three literature HSP components for pure solvents."""
    data = frame.sort_values("delta_total_MPa05").reset_index(drop=True)
    labels = [display_label(value) for value in data["system"]]
    y = np.arange(len(data))
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 7.0), constrained_layout=True)
    offsets = (-0.23, 0.0, 0.23)
    columns = ("delta_d_MPa05", "delta_p_MPa05", "delta_h_MPa05")
    component_labels = (r"$\delta_D$ dispersion", r"$\delta_P$ polar", r"$\delta_H$ H-bonding")
    colors = (PALETTE["blue"], PALETTE["orange"], PALETTE["green"])
    for offset, column, label, color in zip(offsets, columns, component_labels, colors):
        axes[0].barh(y + offset, data[column], height=0.21, label=label,
                     color=color, edgecolor="#202428", linewidth=0.45)
    axes[0].set_yticks(y, labels)
    axes[0].tick_params(axis="y", labelsize=7)
    axes[0].set_xlabel(r"Hansen parameter (MPa$^{1/2}$)")
    axes[0].legend(loc="lower right", fontsize=7)
    _boxed_axes(axes[0])
    _panel(axes[0], "a")
    _labelled_scatter(
        axes[1], data, "delta_p_MPa05", "delta_h_MPa05",
        r"Polar component, $\delta_P$ (MPa$^{1/2}$)",
        r"Hydrogen-bonding component, $\delta_H$ (MPa$^{1/2}$)",
        PALETTE["purple"],
    )
    _panel(axes[1], "b")
    fig.suptitle(
        "Group 2 | Literature Hansen solubility parameters of pure solvents",
        fontsize=12, fontweight="bold"
    )
    return _save(fig, out_dir / "group2_hansen_parameters")


def _hsp_relationship_figure(frame: pd.DataFrame, out_dir: Path) -> list[Path]:
    """Relate external HSP descriptors to short-time MD observables."""
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.4), constrained_layout=True)
    panels = (
        ("delta_p_MPa05", "alpha", r"$\delta_P$ (MPa$^{1/2}$)", "TAMSD exponent, α", PALETTE["blue"]),
        ("delta_h_MPa05", "alpha", r"$\delta_H$ (MPa$^{1/2}$)", "TAMSD exponent, α", PALETTE["orange"]),
        ("delta_d_MPa05", "head_coord_number_4A", r"$\delta_D$ (MPa$^{1/2}$)", "OPA-head coordination number at 4 Å", PALETTE["green"]),
        ("delta_total_MPa05", "mean_force_eV_A", r"Total HSP, $\delta_T$ (MPa$^{1/2}$)", "Mean force on OPA (eV Å$^{-1}$)", PALETTE["purple"]),
    )
    for letter, ax, (x, y, xlabel, ylabel, color) in zip("abcd", axes.flat, panels):
        _labelled_scatter(ax, frame, x, y, xlabel, ylabel, color)
        rho = frame[x].rank().corr(frame[y].rank())
        ax.set_title(f"Spearman ρ = {rho:.2f}", fontsize=8.2, pad=5)
        _panel(ax, letter)
    fig.suptitle(
        "Group 2 | Exploratory HSP–OPA dynamics/solvation relationships",
        fontsize=12, fontweight="bold"
    )
    return _save(fig, out_dir / "group2_hansen_relationships")


def _labelled_scatter(
    ax: plt.Axes,
    frame: pd.DataFrame,
    x: str,
    y: str,
    xlabel: str,
    ylabel: str,
    color,
    *,
    point_size: float = 42,
    annotation_size: float = 6.6,
    axis_label_size: float | None = None,
    tick_label_size: float | None = None,
    label_offsets: dict[str, tuple[float, float]] | None = None,
    label_systems: set[str] | None = None,
) -> None:
    plot_frame = frame.copy()
    if isinstance(color, str):
        point_colors = color
    else:
        plot_frame["_point_color"] = list(color)
        point_colors = plot_frame.dropna(subset=[x, y])["_point_color"].tolist()
    valid = plot_frame.dropna(subset=[x, y]).reset_index(drop=True)
    ax.scatter(
        valid[x], valid[y], s=point_size, color=point_colors, edgecolor="#16191C",
        linewidth=0.8, zorder=3
    )
    x_span = max(float(valid[x].max() - valid[x].min()), 1e-6) if len(valid) else 1.0
    y_span = max(float(valid[y].max() - valid[y].min()), 1e-6) if len(valid) else 1.0
    candidates = (
        (0.035, 0.070), (0.035, -0.095), (-0.035, 0.105), (-0.035, -0.130),
        (0.120, 0.030), (-0.120, 0.030), (0.105, -0.100), (-0.105, 0.120),
    )
    placed: list[tuple[float, float]] = []
    for index, row in valid.iterrows():
        system = str(row["system"])
        if label_systems is not None and system not in label_systems:
            continue
        x_norm = (float(row[x]) - float(valid[x].min())) / x_span
        y_norm = (float(row[y]) - float(valid[y].min())) / y_span
        if label_offsets and system in label_offsets:
            dx, dy = label_offsets[system]
        elif placed:
            dx, dy = max(
                candidates,
                key=lambda offset: min(
                    (x_norm + offset[0] - old_x) ** 2
                    + (y_norm + offset[1] - old_y) ** 2
                    for old_x, old_y in placed
                ),
            )
        else:
            dx, dy = candidates[index % len(candidates)]
        placed.append((x_norm + dx, y_norm + dy))
        ax.annotate(
            display_label(system),
            (row[x], row[y]),
            xytext=(row[x] + dx * x_span, row[y] + dy * y_span),
            fontsize=annotation_size,
            ha="left" if dx > 0 else "right",
            va="bottom" if dy > 0 else "top",
            arrowprops={"arrowstyle": "-", "color": "#5D6267", "lw": 0.45},
            bbox={"boxstyle": "square,pad=0.12", "fc": "white", "ec": "none", "alpha": 0.82},
            zorder=4,
        )
    ax.set_xlabel(xlabel, fontsize=axis_label_size)
    ax.set_ylabel(ylabel, fontsize=axis_label_size)
    if tick_label_size is not None:
        ax.tick_params(axis="both", labelsize=tick_label_size)
    ax.margins(x=0.16, y=0.18)
    _boxed_axes(ax)


def _comparison_figure(frame: pd.DataFrame, group: str, out_dir: Path) -> list[Path]:
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.4), constrained_layout=True)
    _labelled_scatter(
        axes[0, 0], frame, "mean_force_eV_A", "alpha",
        "Mean force on OPA (eV Å$^{-1}$)", "TAMSD exponent, α", PALETTE["blue"]
    )
    _labelled_scatter(
        axes[0, 1], frame, "final_disp_A", "mean_force_eV_A",
        "Final OPA displacement (Å)", "Mean force on OPA (eV Å$^{-1}$)",
        PALETTE["orange"],
    )
    structural = frame.loc[frame["structural_data_status"] == "matched"]
    _labelled_scatter(
        axes[1, 0], structural, "head_coord_number_4A", "alpha",
        "OPA-head coordination number at 4 Å", "TAMSD exponent, α", PALETTE["green"]
    )
    _labelled_scatter(
        axes[1, 1], structural, "head_tail_coord_ratio", "mean_force_eV_A",
        "Head/tail coordination-number ratio", "Mean force on OPA (eV Å$^{-1}$)",
        PALETTE["purple"],
    )
    for letter, ax in zip("abcd", axes.flat):
        _panel(ax, letter)
    fig.suptitle(GROUP_TITLES[group], fontsize=13, fontweight="bold")
    return _save(fig, out_dir / f"{group}_labelled_comparison")


def _ranking_figure(frame: pd.DataFrame, group: str, out_dir: Path) -> list[Path]:
    data = frame.sort_values("alpha", ascending=True).reset_index(drop=True)
    labels = [display_label(value) for value in data["system"]]
    y = np.arange(len(data))
    fig_height = max(5.0, 0.44 * len(data) + 1.5)
    fig, axes = plt.subplots(1, 3, figsize=(12.0, fig_height), sharey=True, constrained_layout=True)
    columns = ("alpha", "mean_force_eV_A", "final_disp_A")
    xlabels = (
        "TAMSD exponent, α",
        "Mean force (eV Å$^{-1}$)",
        "Final displacement (Å)",
    )
    colors = (PALETTE["blue"], PALETTE["orange"], PALETTE["green"])
    for letter, ax, column, xlabel, color in zip("abc", axes, columns, xlabels, colors):
        ax.scatter(
            data[column], y, s=42, color=color, edgecolor="#16191C",
            linewidth=0.65, zorder=3
        )
        for xpos, ypos, label in zip(data[column], y, labels):
            ax.annotate(
                label, (xpos, ypos), xytext=(5, 0), textcoords="offset points",
                fontsize=6.5, va="center", ha="left"
            )
        ax.set_xlabel(xlabel)
        ax.margins(x=0.38)
        _boxed_axes(ax)
        _panel(ax, letter)
    axes[0].set_yticks(y, labels)
    axes[0].tick_params(axis="y", labelsize=7)
    fig.suptitle(f"{GROUP_TITLES[group]} | within-group ranking", fontsize=12, fontweight="bold")
    return _save(fig, out_dir / f"{group}_labelled_ranking")


def _rdf_figure(
    profiles: pd.DataFrame, group: str, out_dir: Path
) -> list[Path]:
    if profiles.empty:
        return []
    data = profiles.loc[(profiles["group"] == group) & (profiles["species"] == "all")]
    if data.empty:
        return []
    systems = list(dict.fromkeys(data["system"]))
    cmap = plt.get_cmap("tab20")
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.5), constrained_layout=True)
    for index, system in enumerate(systems):
        subset = data.loc[data["system"] == system].sort_values("r_A")
        color = cmap(index / max(1, len(systems) - 1))
        label = display_label(str(system))
        axes[0].plot(subset["r_A"], subset["g_head"], lw=1.15, color=color, label=label)
        axes[1].plot(subset["r_A"], subset["g_tail"], lw=1.15, color=color, label=label)
    for letter, ax, ylabel in zip("ab", axes, ("g(r), OPA head", "g(r), OPA terminal tail")):
        ax.set(xlabel="Distance, r (Å)", ylabel=ylabel, xlim=(0, 12))
        _boxed_axes(ax)
        _panel(ax, letter)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.08),
        ncol=min(4, max(1, len(labels))), fontsize=6.5
    )
    fig.suptitle(f"{GROUP_TITLES[group]} | total-solvent RDF", fontsize=12, fontweight="bold")
    return _save(fig, out_dir / f"{group}_all_systems_rdf")


def _motion_figure(
    frame: pd.DataFrame,
    group: str,
    out_dir: Path,
    project_root: Path,
) -> list[Path]:
    """Plot the complete time record and complementary mobility summaries."""
    series: list[tuple[str, pd.DataFrame]] = []
    for row in frame.itertuples():
        csv_value = getattr(row, "csv_path", None)
        if not csv_value:
            continue
        csv_path = Path(csv_value)
        if not csv_path.is_absolute():
            csv_path = project_root / csv_path
        if csv_path.exists():
            series.append((str(row.system), pd.read_csv(csv_path)))
    if not series:
        return []

    data = frame.copy()
    data["apparent_net_rate_A_ps"] = data["final_disp_A"] / data["time_span_ps"]
    ordered = data.sort_values("apparent_net_rate_A_ps").reset_index(drop=True)
    y = np.arange(len(ordered))
    cmap = plt.get_cmap("tab20")
    # Use well-separated hues first, then the paired hues.  The same solvent
    # keeps the same color in the trajectories and the TAMSD scatter panel.
    color_order = (0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 1, 5, 9, 13)
    systems = [str(value) for value in data["system"]]
    system_colors = {
        system: cmap(color_order[index % len(color_order)] / 19)
        for index, system in enumerate(systems)
    }
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.6), constrained_layout=True)
    for system, motion in series:
        color = system_colors[system]
        label = display_label(system)
        axes[0, 0].plot(
            motion["time_ps"], motion["OPA_disp_norm_A"], color=color, lw=1.05,
            label=label
        )
        axes[0, 1].plot(
            motion["time_ps"], motion["F_OPA_norm_eV_A"], color=color, lw=0.9,
            label=label, alpha=0.9
        )
    axes[0, 0].set(xlabel="Simulation time (ps)", ylabel="OPA displacement from t=0 (Å)")
    axes[0, 1].set(xlabel="Simulation time (ps)", ylabel="Instantaneous force on OPA (eV Å$^{-1}$)")
    axes[1, 0].scatter(
        ordered["apparent_net_rate_A_ps"], y, s=42, color=PALETTE["orange"],
        edgecolor="#16191C", linewidth=0.65, zorder=3
    )
    axes[1, 0].set_yticks(y, [display_label(value) for value in ordered["system"]])
    axes[1, 0].tick_params(axis="y", labelsize=7)
    axes[1, 0].set(xlabel="Apparent net movement rate, final displacement / total time (Å ps$^{-1}$)")
    _labelled_scatter(
        axes[1, 1], data, "alpha_prefactor_A2_ps_alpha", "alpha",
        "TAMSD prefactor (Å² ps$^{-α}$)", "TAMSD exponent, α",
        [system_colors[str(system)] for system in data["system"]],
        point_size=72,
        annotation_size=7.4,
        axis_label_size=10.5,
        tick_label_size=9.5,
        label_offsets={
            "acetone": (0.060, 0.030),
            "n-heptane": (-0.045, 0.075),
        },
        label_systems={"acetone", "n-heptane"},
    )
    for letter, ax in zip("abcd", axes.flat):
        _boxed_axes(ax)
        _panel(ax, letter)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.06),
        ncol=min(4, max(1, len(labels))), fontsize=6.4
    )
    fig.suptitle(
        f"{GROUP_TITLES[group]} | complete 0–{data['time_span_ps'].max():g} ps motion record",
        fontsize=12,
        fontweight="bold",
    )
    return _save(fig, out_dir / f"{group}_full_time_mobility")


def _group_findings(frame: pd.DataFrame, group: str) -> list[str]:
    high_alpha = frame.loc[frame["alpha"].idxmax()]
    low_alpha = frame.loc[frame["alpha"].idxmin()]
    low_force = frame.loc[frame["mean_force_eV_A"].idxmin()]
    high_force = frame.loc[frame["mean_force_eV_A"].idxmax()]
    findings = [
        f"扩散指数最高：{display_label(str(high_alpha.system))}（α={high_alpha.alpha:.3f}）；"
        f"最低：{display_label(str(low_alpha.system))}（α={low_alpha.alpha:.3f}）。",
        f"平均受力最低：{display_label(str(low_force.system))}（{low_force.mean_force_eV_A:.3f} eV Å⁻¹）；"
        f"最高：{display_label(str(high_force.system))}（{high_force.mean_force_eV_A:.3f} eV Å⁻¹）。",
    ]
    structural = frame.loc[frame["structural_data_status"] == "matched"].dropna(
        subset=["head_coord_number_4A"]
    )
    if not structural.empty:
        high_cn = structural.loc[structural["head_coord_number_4A"].idxmax()]
        findings.append(
            f"OPA头基4 Å配位数最高：{display_label(str(high_cn.system))}"
            f"（CN={high_cn.head_coord_number_4A:.3f}）。"
        )
    if group == "group1":
        ordered = frame.sort_values("cosolvent_fraction")
        corr = ordered[["cosolvent_fraction", "mean_force_eV_A"]].corr().iloc[0, 1]
        findings.append(f"丙酮比例与平均受力的Pearson相关系数为 r={corr:.3f}，但α随浓度并非单调变化。")
    if group == "group3" and (frame["structural_data_status"] != "matched").any():
        findings.append("THF/toluene结构轨迹与动力学CSV不匹配；该点仅参加动力学比较，不参加RDF/CN比较。")
    if group == "group2" and "delta_d_MPa05" in frame:
        max_p = frame.loc[frame["delta_p_MPa05"].idxmax()]
        max_h = frame.loc[frame["delta_h_MPa05"].idxmax()]
        findings.append(
            f"HSP中极性分量最高的是{display_label(str(max_p.system))}"
            f"（δP={max_p.delta_p_MPa05:.1f} MPa¹ᐟ²），氢键分量最高的是"
            f"{display_label(str(max_h.system))}（δH={max_h.delta_h_MPa05:.1f} MPa¹ᐟ²）。"
        )
    return findings


def _hsp_table_html(frame: pd.DataFrame) -> str:
    view = frame[
        ["system", "solvent_name", "cas_number", "delta_d_MPa05",
         "delta_p_MPa05", "delta_h_MPa05", "delta_total_MPa05",
         "source_key", "source_url"]
    ].copy()
    view["system"] = view["system"].map(display_label)
    view["source_key"] = [
        f'<a href="{html.escape(url)}">{html.escape(key)}</a>'
        for key, url in zip(view["source_key"], view["source_url"])
    ]
    view = view.drop(columns="source_url").rename(columns={
        "system": "Simulation label",
        "solvent_name": "Chemical identity",
        "cas_number": "CAS RN",
        "delta_d_MPa05": "δD (MPa¹ᐟ²)",
        "delta_p_MPa05": "δP (MPa¹ᐟ²)",
        "delta_h_MPa05": "δH (MPa¹ᐟ²)",
        "delta_total_MPa05": "δT (MPa¹ᐟ²)",
        "source_key": "Source",
    })
    view["δT (MPa¹ᐟ²)"] = view["δT (MPa¹ᐟ²)"].round(2)
    return view.to_html(index=False, border=0, classes="metrics", escape=False)


def _table_html(frame: pd.DataFrame) -> str:
    view = frame.copy()
    view["apparent_net_rate_A_ps"] = view["final_disp_A"] / view["time_span_ps"]
    view = view[
        ["system", "time_span_ps", "n_frames", "final_disp_A", "mean_disp_A",
         "apparent_net_rate_A_ps", "alpha", "alpha_fit_r2",
         "alpha_prefactor_A2_ps_alpha", "mean_force_eV_A",
         "head_coord_number_4A", "head_tail_coord_ratio", "structural_data_status"]
    ]
    view["system"] = view["system"].map(display_label)
    view = view.rename(
        columns={
            "system": "System / composition",
            "time_span_ps": "Total time (ps)",
            "n_frames": "Frames",
            "alpha": "α",
            "alpha_fit_r2": "α fit R²",
            "alpha_prefactor_A2_ps_alpha": "TAMSD prefactor",
            "mean_force_eV_A": "Mean force (eV Å⁻¹)",
            "final_disp_A": "Final displacement (Å)",
            "mean_disp_A": "Mean displacement (Å)",
            "apparent_net_rate_A_ps": "Net movement rate (Å ps⁻¹)",
            "head_coord_number_4A": "Head CN (4 Å)",
            "head_tail_coord_ratio": "Head/tail CN",
            "structural_data_status": "Structure status",
        }
    )
    numeric = view.select_dtypes(include=[np.number]).columns
    view[numeric] = view[numeric].round(3)
    return view.to_html(index=False, border=0, classes="metrics", na_rep="—")


def _report_html(
    frame: pd.DataFrame,
    group: str,
    include_motion: bool,
    hspipy_summary: dict[str, float] | None = None,
) -> str:
    findings = "".join(f"<li>{html.escape(item)}</li>" for item in _group_findings(frame, group))
    title = GROUP_TITLES[group]
    figures = [
        *(([(f"../figures/{group}_full_time_mobility.png", "Complete time record and mobility metrics")]) if include_motion else []),
        (f"../figures/{group}_labelled_comparison.png", "Labelled multi-metric comparison"),
        (f"../figures/{group}_labelled_ranking.png", "Within-group ranking; every point is labelled"),
        (f"../figures/{group}_all_systems_rdf.png", "RDF curves for every structurally valid system"),
        *(([
            ("../figures/group2_hansen_parameters.png", "Literature Hansen solubility parameters"),
            ("../figures/group2_hansen_relationships.png", "Exploratory HSP relationships with MD observables"),
            ("../figures/group2_hspipy_single_sphere_2d.png", "HSPiPy single-sphere fit to the MD mobility label"),
            ("../figures/group2_hspipy_double_sphere_2d.png", "HSPiPy double-sphere fit to the MD mobility label"),
            ("../figures/group2_hspipy_double_sphere_3d.png", "HSPiPy double-sphere 3D Hansen space"),
        ]) if group == "group2" else []),
        *(([
            (f"../figures/{group}_hspipy_single_sphere_2d.png", "HSPiPy single-sphere fit to mixed-solvent HSP"),
            (f"../figures/{group}_hspipy_double_sphere_2d.png", "HSPiPy double-sphere fit to mixed-solvent HSP"),
            (f"../figures/{group}_hspipy_double_sphere_3d.png", "HSPiPy double-sphere 3D Hansen space"),
        ]) if group in {"group1", "group3"} else []),
    ]
    cards = "".join(
        f'<figure><a href="{src}"><img src="{src}" alt="{html.escape(caption)}"></a>'
        f"<figcaption>{html.escape(caption)} — PNG 600 dpi；另提供同名SVG/PDF。</figcaption></figure>"
        for src, caption in figures
    )
    hsp_section = ""
    if group == "group2":
        fit_text = ""
        if hspipy_summary is not None:
            fit_text = (
                "<h3>HSPiPy球拟合检验</h3>"
                f"<p>将α≥{hspipy_summary['threshold']:.2f}预先定义为MD高移动性标签后，"
                f"单球训练准确率为{hspipy_summary['single_accuracy']:.3f}、DATAFIT为"
                f"{hspipy_summary['single_datafit']:.3f}；双球准确率仍为"
                f"{hspipy_summary['double_accuracy']:.3f}，并产生"
                f"{int(hspipy_summary['double_false_positive'])}个假阳性。"
                "因此高α条件没有形成可靠的单一Hansen区域。"
                "<strong>这里的球是MD移动性分类，不是OPA实验溶解度球。</strong> "
                '<a href="group2_hspipy_analysis.html">查看完整HSPiPy参数、RED和逐溶剂分类 →</a></p>'
            )
        hsp_section = f"""<h2>Hansen溶解度参数（外部物性）</h2>
<p>δD、δP和δH分别描述色散、极性及氢键相互作用；δT=√(δD²+δP²+δH²)。这些数值来自文献，不是由本次10 ps轨迹反演。相关图中的Spearman ρ仅用于14个纯溶剂的探索性趋势，不代表因果关系或统计显著性。由于尚无实验溶解度标签，本报告不据HSP单独给出“最好溶剂”排名；HSPiPy给出的RED仅相对于MD定义的移动性球。</p>
{fit_text}
{_hsp_table_html(frame)}"""
    elif group in {"group1", "group3"} and hspipy_summary is not None:
        hsp_section = f"""<h2>HSPiPy混合溶剂分析</h2>
<p>各HSP分量按名义体积分数线性混合，δk,mix=Σφiδk,i。将α≥{hspipy_summary['threshold']:.2f}定义为MD高移动性标签；本组共{int(hspipy_summary['n_systems'])}个条件，其中{int(hspipy_summary['n_high_mobility'])}个满足判据。单球准确率为{hspipy_summary['single_accuracy']:.3f}、DATAFIT为{hspipy_summary['single_datafit']:.3f}；双球准确率为{hspipy_summary['double_accuracy']:.3f}。由于样本很少且拟合中心出现非物理分量，这些球只检验移动性标签能否在Hansen空间聚类，不是OPA实验溶解度球。<a href="{group}_hspipy_analysis.html">查看完整参数、RED及阈值敏感性 →</a></p>"""
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)}</title>
<style>
body{{font-family:Arial,'Microsoft YaHei',sans-serif;max-width:1280px;margin:auto;padding:32px;background:#eef1f4;color:#202428;line-height:1.65}}
main{{background:white;border:1px solid #202428;padding:32px 42px}}h1{{border-bottom:3px solid #24557A;padding-bottom:10px}}
h2{{color:#24557A;margin-top:30px}}.notice{{border:1px solid #B4BDC5;background:#F7F9FA;padding:12px 16px}}
figure{{margin:22px 0;border:1.5px solid #202428;padding:10px;background:white}}img{{display:block;width:100%;height:auto}}
figcaption{{border-top:1px solid #B4BDC5;margin-top:8px;padding:8px 4px 2px;color:#4C535A}}
table.metrics{{width:100%;border-collapse:collapse;font-size:12px}}.metrics th,.metrics td{{border:1px solid #687078;padding:6px;text-align:right}}
.metrics th:first-child,.metrics td:first-child{{text-align:left}}a{{color:#24557A}}
</style></head><body><main><p><a href="index.html">← 返回三组目录</a></p><h1>{html.escape(title)}</h1>
<div class="notice">本报告仅在本组内部比较。α表示异常扩散指数，不是SAM有序参数；平均力、位移和配位数也不能单独等同于成膜优劣。</div>
<h2>组内直接结果</h2><ul>{findings}</ul><p>移动性同时报告完整0–10 ps位移轨迹、最终/平均位移、表观净移动速率、TAMSD指数α及前因子。表观净移动速率仅用于本组描述，不作为严格扩散系数。</p><h2>发表用图件</h2>{cards}
<h2>数据表</h2>{_table_html(frame)}
{hsp_section}
<h2>解释边界</h2><p>这些10 ps、单轨迹、无显式基底的数据描述OPA的预吸附溶剂化与短时动力学。图中完整标注体系名称，方便识别离群点；正式论文中仍需结合独立重复、基底吸附能、覆盖率、倾角或缺陷密度。</p>
</main></body></html>"""


def render_publication_group(
    metrics: pd.DataFrame,
    profiles: pd.DataFrame,
    group: str,
    output_root: Path,
    project_root: Path | None = None,
) -> list[Path]:
    """Render one group without comparing it to either of the other groups."""
    _style()
    frame = metrics.loc[metrics["group"] == group].copy()
    if frame.empty:
        raise ValueError(f"no data found for {group}")
    if group == "group2":
        frame = _merge_pure_solvent_hsp(frame)
    output_root = Path(output_root)
    figure_root = output_root / "figures"
    outputs = _comparison_figure(frame, group, figure_root)
    outputs.extend(_ranking_figure(frame, group, figure_root))
    outputs.extend(_rdf_figure(profiles, group, figure_root))
    hspipy_summary: dict[str, float] | None = None
    if group == "group2":
        outputs.extend(_hsp_parameter_figure(frame, figure_root))
        outputs.extend(_hsp_relationship_figure(frame, figure_root))
        from .hspipy_method import run_hspipy_pure_analysis

        hspipy_outputs, hspipy_summary = run_hspipy_pure_analysis(frame, output_root)
        outputs.extend(hspipy_outputs)
    elif group in {"group1", "group3"}:
        from .hspipy_method import run_hspipy_mixture_analysis

        hspipy_outputs, hspipy_summary = run_hspipy_mixture_analysis(
            frame, output_root, group
        )
        outputs.extend(hspipy_outputs)
    motion_outputs: list[Path] = []
    if project_root is not None:
        motion_outputs = _motion_figure(frame, group, figure_root, Path(project_root))
        outputs.extend(motion_outputs)
    report = output_root / "reports" / f"{group}.html"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        _report_html(frame, group, bool(motion_outputs), hspipy_summary),
        encoding="utf-8",
    )
    outputs.append(report)
    table = output_root / "tables" / f"{group}_metrics.csv"
    table.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(table, index=False, encoding="utf-8-sig")
    outputs.append(table)
    if group == "group2":
        hsp_table = output_root / "tables" / "group2_hansen_parameters.csv"
        hsp_columns = [
            "system", "solvent_name", "cas_number", "delta_d_MPa05",
            "delta_p_MPa05", "delta_h_MPa05", "delta_total_MPa05",
            "source_key", "source_url", "note",
        ]
        frame[hsp_columns].to_csv(hsp_table, index=False, encoding="utf-8-sig")
        outputs.append(hsp_table)
    return outputs


def _index_html(metrics: pd.DataFrame) -> str:
    cards = []
    descriptions = {
        "group1": "同一共溶剂（acetone）在n-heptane中的不同浓度",
        "group2": "同浓度设计下的不同纯溶剂组分",
        "group3": "不同混合组分及分层纯溶剂参照",
    }
    for group in ("group1", "group2", "group3"):
        count = int((metrics["group"] == group).sum())
        cards.append(
            f'<a class="card" href="{group}.html"><h2>{group.upper()}</h2>'
            f'<p>{html.escape(descriptions[group])}</p><strong>{count} systems →</strong></a>'
        )
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>SAM solvent analysis result2</title>
<style>body{{font-family:Arial,'Microsoft YaHei',sans-serif;max-width:1080px;margin:auto;padding:42px;background:#EEF1F4;color:#202428}}
main{{background:white;border:1.5px solid #202428;padding:38px}}h1{{border-bottom:3px solid #24557A;padding-bottom:12px}}
.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:18px;margin-top:28px}}.card{{display:block;border:1.5px solid #202428;padding:20px;color:#202428;text-decoration:none;background:#FAFBFC}}
.card:hover{{border-color:#24557A;box-shadow:0 4px 12px #0002}}.card h2{{color:#24557A}}.note{{border-left:4px solid #D17C2F;padding:10px 14px;background:#FFF8F0}}
@media(max-width:760px){{.grid{{grid-template-columns:1fr}}}}</style></head><body><main>
<h1>OPA/SAM溶剂效应：三组独立比较</h1><p class="note">三组只做组内比较，不建立跨组总排名。所有散点均标注具体组分；PNG为600 dpi，且同时提供SVG/PDF。</p>
<div class="grid">{''.join(cards)}</div><p><a href="publication_summary_zh.md">查看用于内容介绍与发表的中文Markdown总结报告 →</a></p></main></body></html>"""


def build_result2(
    source_root: Path, output_root: Path, refresh: bool = False
) -> list[Path]:
    """Build or refresh a self-contained result2 tree from validated tables."""
    source_root, output_root = Path(source_root), Path(output_root)
    if output_root.exists() and not refresh:
        raise FileExistsError(f"refusing to replace existing output: {output_root}")
    system_metrics = source_root / "tables" / "system_metrics.csv"
    if system_metrics.exists():
        metrics = pd.read_csv(system_metrics)
    else:
        group_tables = [source_root / "tables" / f"group{i}_metrics.csv" for i in (1, 2, 3)]
        missing = [path for path in group_tables if not path.exists()]
        if missing:
            raise FileNotFoundError(
                f"neither system_metrics.csv nor complete group tables found: {missing}"
            )
        metrics = pd.concat((pd.read_csv(path) for path in group_tables), ignore_index=True)
    profiles = pd.read_csv(source_root / "tables" / "rdf_profiles.csv")
    output_root.mkdir(parents=True, exist_ok=refresh)
    project_root = source_root.resolve().parents[1]
    outputs: list[Path] = []
    for group in ("group1", "group2", "group3"):
        outputs.extend(
            render_publication_group(
                metrics, profiles, group, output_root, project_root=project_root
            )
        )
    index = output_root / "reports" / "index.html"
    index.write_text(_index_html(metrics), encoding="utf-8")
    outputs.append(index)
    source_profiles = source_root / "tables" / "rdf_profiles.csv"
    output_profiles = output_root / "tables" / "rdf_profiles.csv"
    if source_profiles.resolve() != output_profiles.resolve():
        shutil.copy2(source_profiles, output_profiles)
    outputs.append(output_profiles)
    metadata = output_root / "build_summary.json"
    metadata.write_text(
        json.dumps(
            {
                "source": source_root.as_posix(),
                "groups": metrics.groupby("group").size().to_dict(),
                "png_dpi": 600,
                "formats": ["png", "svg", "pdf"],
                "comparison_scope": "within-group only",
                "all_scatter_points_labelled": True,
                "complete_motion_time_series": True,
                "group2_hansen_solubility_parameters": {
                    "scope": "14 pure solvents only",
                    "components": ["delta_D", "delta_P", "delta_H", "delta_T"],
                    "provenance": "literature descriptors; not inferred from MD",
                    "hansen_distance_Ra": "not calculated because validated OPA HSP/R0 are unavailable",
                },
                "group2_hspipy_fit": {
                    "package": "HSPiPy 1.1.8",
                    "scope": "14 pure solvents only",
                    "primary_label": "TAMSD alpha >= 0.80",
                    "models": ["single sphere", "double sphere"],
                    "interpretation": "MD mobility region; not experimental OPA solubility sphere",
                },
                "group1_group3_hspipy_fit": {
                    "package": "HSPiPy 1.1.8",
                    "scope": ["group1 mixed solvents", "group3 mixed solvents"],
                    "mixing_rule": "volume-fraction linear HSP",
                    "primary_label": "TAMSD alpha >= 0.80",
                    "models": ["single sphere", "double sphere"],
                    "interpretation": "small-sample MD mobility regions; not experimental OPA solubility spheres",
                },
                "mobility_metrics": [
                    "final displacement",
                    "mean displacement",
                    "apparent net movement rate",
                    "TAMSD exponent alpha",
                    "TAMSD prefactor",
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    outputs.append(metadata)
    return outputs
