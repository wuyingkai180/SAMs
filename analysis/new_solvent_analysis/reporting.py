"""Publication-style figures and synchronized Chinese Markdown/HTML reports."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Iterable

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {
    "blue": "#0F4D92",
    "blue_light": "#7884B4",
    "teal": "#42949E",
    "rose": "#B64342",
    "violet": "#9A4D8E",
    "grey": "#767676",
    "light": "#CFCECE",
}


def _configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "font.size": 8,
            "axes.linewidth": 0.8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def _save_figure(fig: plt.Figure, stem: Path) -> list[Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths = [stem.with_suffix(ext) for ext in (".svg", ".pdf", ".png")]
    fig.savefig(paths[0], bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    fig.savefig(paths[2], dpi=300, bbox_inches="tight")
    plt.close(fig)
    return paths


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.14, 1.06, label, transform=ax.transAxes, fontweight="bold", fontsize=9)


def render_group1_figures(
    frame: pd.DataFrame, out_dir: Path, block_frame: pd.DataFrame | None = None
) -> list[Path]:
    """Render concentration response without unsupported replicate error bars."""
    _configure_style()
    data = frame.sort_values("cosolvent_fraction")
    concentration = 100.0 * data["cosolvent_fraction"].to_numpy(float)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), constrained_layout=True)
    axes[0].plot(concentration, data["alpha"], "o-", color=COLORS["blue"], lw=1.6)
    axes[0].axhline(1.0, color=COLORS["grey"], ls="--", lw=0.8)
    axes[0].set(xlabel="Acetone concentration (%)", ylabel="TAMSD exponent, α")
    _panel_label(axes[0], "a")
    axes[1].plot(
        concentration, data["mean_force_eV_A"], "o-", color=COLORS["rose"], lw=1.6
    )
    axes[1].set(xlabel="Acetone concentration (%)", ylabel="Mean force on OPA (eV Å$^{-1}$)")
    _panel_label(axes[1], "b")
    fig.suptitle("Group 1 | Acetone / n-heptane concentration series", fontsize=9)
    paths = _save_figure(fig, Path(out_dir) / "group1_concentration_response")

    if block_frame is not None and not block_frame.empty:
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), constrained_layout=True)
        ordered = list(data["system"])
        positions = np.arange(len(ordered))
        for pos, system in zip(positions, ordered):
            subset = block_frame.loc[block_frame["system"] == system]
            axes[0].scatter(
                np.full(len(subset), pos), subset["alpha"], s=13,
                color=COLORS["blue_light"], alpha=0.8
            )
            axes[1].scatter(
                np.full(len(subset), pos), subset["mean_force_eV_A"], s=13,
                color=COLORS["rose"], alpha=0.8
            )
        labels = [f"{100*x:.0f}" for x in data["cosolvent_fraction"]]
        for ax, ylabel in zip(
            axes, ("Block TAMSD exponent, α", "Block mean force (eV Å$^{-1}$)")
        ):
            ax.set_xticks(positions, labels)
            ax.set(xlabel="Acetone concentration (%)", ylabel=ylabel)
        _panel_label(axes[0], "a")
        _panel_label(axes[1], "b")
        fig.suptitle("Within-trajectory window sensitivity (descriptive blocks)", fontsize=9)
        paths.extend(_save_figure(fig, Path(out_dir) / "group1_window_sensitivity"))
    return paths


def render_group2_figures(frame: pd.DataFrame, out_dir: Path) -> list[Path]:
    _configure_style()
    data = frame.sort_values("alpha").reset_index(drop=True)
    y = np.arange(len(data))
    classes = data.get("solvent_class", pd.Series("unclassified", index=data.index))
    class_colors = {
        "aprotic-polar": COLORS["blue"], "protic": COLORS["rose"],
        "nonpolar": COLORS["teal"], "unclassified": COLORS["grey"]
    }
    colors = [class_colors.get(value, COLORS["grey"]) for value in classes]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.2), sharey=True, constrained_layout=True)
    axes[0].scatter(data["alpha"], y, c=colors, s=28, edgecolor="white", linewidth=0.5)
    axes[0].axvline(1.0, color=COLORS["grey"], ls="--", lw=0.8)
    axes[0].set(xlabel="TAMSD exponent, α", yticks=y, yticklabels=data["system"])
    axes[1].scatter(data["mean_force_eV_A"], y, c=colors, s=28, edgecolor="white", linewidth=0.5)
    axes[1].set(xlabel="Mean force on OPA (eV Å$^{-1}$)")
    _panel_label(axes[0], "a")
    _panel_label(axes[1], "b")
    fig.suptitle("Group 2 | Pure-solvent descriptive ranking", fontsize=9)
    return _save_figure(fig, Path(out_dir) / "group2_pure_solvent_ranking")


def render_group3_figures(frame: pd.DataFrame, out_dir: Path) -> list[Path]:
    _configure_style()
    role_order = {
        "n-heptane-base-composition": 0,
        "alternate-base-composition": 1,
        "pure-reference": 2,
    }
    data = frame.assign(_order=frame["role"].map(role_order)).sort_values(["_order", "system"])
    y = np.arange(len(data))
    colors = [
        COLORS["blue"] if role == "n-heptane-base-composition"
        else COLORS["violet"] if role == "alternate-base-composition"
        else COLORS["grey"]
        for role in data["role"]
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 4.0), sharey=True, constrained_layout=True)
    columns = ["alpha", "head_coord_number_4A", "head_tail_coord_ratio"]
    labels = ["TAMSD exponent, α", "Head CN at 4 Å", "Head/tail CN ratio"]
    for index, (ax, column, label) in enumerate(zip(axes, columns, labels)):
        ax.scatter(data[column], y, c=colors, s=28, edgecolor="white", linewidth=0.5)
        ax.set_xlabel(label)
        _panel_label(ax, chr(ord("a") + index))
    axes[0].set(yticks=y, yticklabels=data["system"])
    fig.suptitle("Group 3 | Composition screen and separate reference systems", fontsize=9)
    return _save_figure(fig, Path(out_dir) / "group3_composition_and_references")


def render_rdf_figure(profiles: pd.DataFrame, out_dir: Path) -> list[Path]:
    _configure_style()
    selected = profiles.loc[profiles["species"] == "all"]
    systems = list(dict.fromkeys(selected["system"]))
    if len(systems) > 8:
        systems = systems[:8]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)
    cmap = plt.get_cmap("viridis")
    for index, system in enumerate(systems):
        subset = selected.loc[selected["system"] == system]
        color = cmap(index / max(1, len(systems) - 1))
        axes[0].plot(subset["r_A"], subset["g_head"], color=color, lw=1.0, label=system)
        axes[1].plot(subset["r_A"], subset["g_tail"], color=color, lw=1.0)
    axes[0].set(xlabel="r (Å)", ylabel="g(r), OPA head")
    axes[1].set(xlabel="r (Å)", ylabel="g(r), terminal tail")
    axes[0].legend(fontsize=5.5, ncol=1, loc="upper right")
    _panel_label(axes[0], "a")
    _panel_label(axes[1], "b")
    fig.suptitle("Representative total-solvent radial distributions", fontsize=9)
    return _save_figure(fig, Path(out_dir) / "combined_representative_rdf")


def render_combined_figure(frame: pd.DataFrame, out_dir: Path) -> list[Path]:
    _configure_style()
    group_colors = {"group1": COLORS["blue"], "group2": COLORS["teal"], "group3": COLORS["rose"]}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)
    for group, subset in frame.groupby("group"):
        axes[0].scatter(
            subset["mean_force_eV_A"], subset["alpha"], s=24,
            label=group, color=group_colors[group], alpha=0.85,
            edgecolor="white", linewidth=0.4
        )
        axes[1].scatter(
            subset["head_coord_number_4A"], subset["alpha"], s=24,
            color=group_colors[group], alpha=0.85, edgecolor="white", linewidth=0.4
        )
    axes[0].set(xlabel="Mean force on OPA (eV Å$^{-1}$)", ylabel="TAMSD exponent, α")
    axes[1].set(xlabel="Head CN at 4 Å", ylabel="TAMSD exponent, α")
    axes[0].legend()
    _panel_label(axes[0], "a")
    _panel_label(axes[1], "b")
    fig.suptitle("Structure–dynamics relationships across all systems", fontsize=9)
    return _save_figure(fig, Path(out_dir) / "combined_structure_dynamics")


def _format_findings(findings: Iterable[str]) -> str:
    rows = list(findings)
    return "\n".join(f"- {row}" for row in rows) if rows else "- 完整结果见各组表格与图件。"


def build_combined_markdown(context: dict) -> str:
    """Build the concise report from one context shared with HTML output."""
    limitations = list(context.get("limitations", []))
    if not any("单条轨迹" in item for item in limitations):
        limitations.append("每个条件只有单条轨迹，不能进行组间显著性检验。")
    if not any("预吸附" in item for item in limitations):
        limitations.append("模拟描述预吸附溶剂化，不等同于真实基底上的 SAM 生长。")
    literature = context.get("literature_findings", [])
    links = context.get("links", {})
    link_text = "\n".join(f"- [{label}]({target})" for label, target in links.items())
    return f"""# OPA/SAM 溶剂效应三组分析报告

## 分析范围

本次共分析 {context.get('system_count', '—')} 个体系。所有结果描述单个 OPA 分子在溶液中的短时预吸附动力学与局部溶剂化，未将其直接解释为已经形成的 SAM 质量。

## 主要结果

{_format_findings(context.get('headline_findings', []))}

## 文献方法对应

{_format_findings(literature)}

## 限制与复核要求

{_format_findings(limitations)}

## 输出文件

{link_text or '- 见同目录下的 CSV、图件及分组报告。'}
"""


def _markdown_to_html(markdown_text: str, title: str) -> str:
    return (
        "<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>"
        f"<title>{html.escape(title)}</title><style>body{{font-family:Arial,sans-serif;"
        "max-width:980px;margin:40px auto;line-height:1.65;color:#272727}}"
        "pre{white-space:pre-wrap;font-family:inherit} </style></head><body>"
        f"<pre>{html.escape(markdown_text)}</pre></body></html>"
    )


def _write_report(markdown_text: str, stem: Path) -> list[Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    markdown_path = stem.with_suffix(".md")
    html_path = stem.with_suffix(".html")
    markdown_path.write_text(markdown_text, encoding="utf-8")
    html_path.write_text(_markdown_to_html(markdown_text, stem.name), encoding="utf-8")
    return [markdown_path, html_path]


def render_group_reports(
    metrics: pd.DataFrame,
    profiles: pd.DataFrame,
    block_metrics: pd.DataFrame,
    output_root: Path,
) -> list[Path]:
    """Create the three report/figure families from the same result tables."""
    output_root = Path(output_root)
    outputs: list[Path] = []
    group1 = metrics.loc[metrics["group"] == "group1"]
    group2 = metrics.loc[metrics["group"] == "group2"]
    group3 = metrics.loc[metrics["group"] == "group3"]
    outputs.extend(render_group1_figures(group1, output_root / "figures", block_metrics))
    outputs.extend(render_group2_figures(group2, output_root / "figures"))
    outputs.extend(render_group3_figures(group3, output_root / "figures"))
    for group, data in (("group1", group1), ("group2", group2), ("group3", group3)):
        rows = data.sort_values("alpha", ascending=False)
        top = rows.iloc[0]
        markdown = f"""# {group} 分析

## 输入与方法

本组包含 {len(data)} 个条件；使用 OPA 质心 TAMSD、受力、4 Å 配位数和 RDF。每个条件仅有单条 10 ps 轨迹，因此排序属于描述性比较。

## 直接结果

- α 最高条件：{top['system']}（α={top['alpha']:.3f}）。
- 平均受力范围：{data['mean_force_eV_A'].min():.3f}–{data['mean_force_eV_A'].max():.3f} eV Å⁻¹。

## 解释边界

这些数据反映预吸附溶剂化，不直接等同于 SAM 覆盖率、倾角、缺陷密度或成膜质量。
"""
        outputs.extend(_write_report(markdown, output_root / "reports" / group))
    return outputs


def render_combined_report(context: dict, output_root: Path) -> list[Path]:
    return _write_report(build_combined_markdown(context), Path(output_root) / "reports" / "combined_report")

