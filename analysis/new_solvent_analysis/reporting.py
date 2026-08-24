"""Publication-style figures and synchronized Chinese Markdown/HTML reports."""

from __future__ import annotations

import html
import re
from pathlib import Path
from typing import Iterable

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


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
    present_classes = list(dict.fromkeys(classes))
    fig.legend(
        handles=[
            Line2D(
                [0], [0], marker="o", linestyle="none", markersize=5,
                color=class_colors.get(label, COLORS["grey"]), label=label
            )
            for label in present_classes
        ],
        loc="upper center", bbox_to_anchor=(0.5, 0.94), ncol=len(present_classes),
        fontsize=7,
    )
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


def _iter_rdf_series(
    profiles: pd.DataFrame, max_per_group: int = 2
) -> list[tuple[str, pd.DataFrame]]:
    selected = profiles.loc[profiles["species"] == "all"]
    series: list[tuple[str, pd.DataFrame]] = []
    for group, group_frame in selected.groupby("group", sort=False):
        systems = list(dict.fromkeys(group_frame["system"]))[:max_per_group]
        for system in systems:
            subset = group_frame.loc[group_frame["system"] == system].sort_values("r_A")
            series.append((f"{group}/{system}", subset))
    return series


def render_rdf_figure(profiles: pd.DataFrame, out_dir: Path) -> list[Path]:
    _configure_style()
    series = _iter_rdf_series(profiles)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)
    cmap = plt.get_cmap("viridis")
    for index, (label, subset) in enumerate(series):
        color = cmap(index / max(1, len(series) - 1))
        axes[0].plot(subset["r_A"], subset["g_head"], color=color, lw=1.0, label=label)
        axes[1].plot(subset["r_A"], subset["g_tail"], color=color, lw=1.0)
    axes[0].set(xlabel="r (Å)", ylabel="g(r), OPA head")
    axes[1].set(xlabel="r (Å)", ylabel="g(r), terminal tail")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, fontsize=5.5, ncol=3, loc="lower center",
        bbox_to_anchor=(0.5, -0.08)
    )
    _panel_label(axes[0], "a")
    _panel_label(axes[1], "b")
    fig.suptitle("Representative total-solvent radial distributions", fontsize=9)
    return _save_figure(fig, Path(out_dir) / "combined_representative_rdf")


def render_enrichment_figure(frame: pd.DataFrame, out_dir: Path) -> list[Path]:
    """Show molecule-fraction-normalized cosolvent enrichment in mixed systems."""
    _configure_style()
    rows = []
    for row in frame.loc[frame["cosolvent"].notna()].itertuples():
        suffix = str(row.cosolvent).replace("-", "_")
        head = getattr(row, f"head_enrichment_4A__{suffix}", float("nan"))
        tail = getattr(row, f"tail_enrichment_4A__{suffix}", float("nan"))
        bulk = getattr(row, f"bulk_molecule_fraction__{suffix}", float("nan"))
        if np.isfinite(head) or np.isfinite(tail):
            rows.append(
                {
                    "label": f"{row.group}: {row.cosolvent} ({100*bulk:.1f} mol%)",
                    "head": head,
                    "tail": tail,
                    "fraction": row.cosolvent_fraction,
                    "group": row.group,
                }
            )
    data = pd.DataFrame(rows).sort_values(["group", "fraction", "label"]).reset_index(drop=True)
    y = np.arange(len(data))
    fig, ax = plt.subplots(figsize=(7.2, max(3.2, 0.29 * len(data))), constrained_layout=True)
    ax.axvline(1.0, color=COLORS["grey"], lw=0.8, ls="--")
    ax.scatter(data["head"], y - 0.12, color=COLORS["blue"], s=24, label="OPA head")
    ax.scatter(data["tail"], y + 0.12, color=COLORS["rose"], s=24, label="terminal tail")
    ax.set(
        xlabel="Local enrichment at 4 Å (local fraction / bulk molecule fraction)",
        yticks=y,
        yticklabels=data["label"],
    )
    ax.legend(loc="lower right")
    _panel_label(ax, "a")
    fig.suptitle("Preferential cosolvent solvation in mixed systems", fontsize=9)
    return _save_figure(fig, Path(out_dir) / "combined_cosolvent_enrichment")


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


def _inline_html(text: str) -> str:
    escaped = html.escape(text)
    return re.sub(
        r"\[([^\]]+)\]\(([^)]+)\)",
        lambda match: (
            f'<a href="{html.escape(match.group(2), quote=True)}">{match.group(1)}</a>'
        ),
        escaped,
    )


def _markdown_body(markdown_text: str) -> str:
    parts: list[str] = []
    in_list = False
    for raw_line in markdown_text.splitlines():
        line = raw_line.strip()
        if not line:
            if in_list:
                parts.append("</ul>")
                in_list = False
            continue
        if line.startswith("- "):
            if not in_list:
                parts.append("<ul>")
                in_list = True
            parts.append(f"<li>{_inline_html(line[2:])}</li>")
            continue
        if in_list:
            parts.append("</ul>")
            in_list = False
        if line.startswith("## "):
            parts.append(f"<h2>{_inline_html(line[3:])}</h2>")
        elif line.startswith("# "):
            parts.append(f"<h1>{_inline_html(line[2:])}</h1>")
        else:
            parts.append(f"<p>{_inline_html(line)}</p>")
    if in_list:
        parts.append("</ul>")
    return "\n".join(parts)


def _markdown_to_html(
    markdown_text: str,
    title: str,
    figures: list[tuple[str, str]] | None = None,
) -> str:
    figure_html = ""
    if figures:
        cards = "".join(
            "<figure>"
            f'<a href="{html.escape(source, quote=True)}"><img src="{html.escape(source, quote=True)}" '
            f'alt="{html.escape(caption, quote=True)}" loading="lazy"></a>'
            f"<figcaption>{html.escape(caption)}</figcaption></figure>"
            for source, caption in figures
        )
        figure_html = f"<section class='figures'><h2>分析图件</h2><div class='figure-grid'>{cards}</div></section>"
    return (
        "<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>{html.escape(title)}</title><style>"
        ":root{color-scheme:light}body{font-family:Arial,'Microsoft YaHei',sans-serif;"
        "max-width:1180px;margin:0 auto;padding:36px 28px 64px;line-height:1.72;color:#272727;"
        "background:#f5f7fa}main{background:#fff;padding:34px 46px;border-radius:12px;"
        "box-shadow:0 3px 18px rgba(15,77,146,.08)}h1{font-size:28px;color:#163b66;"
        "border-bottom:3px solid #0f4d92;padding-bottom:12px}h2{margin-top:30px;font-size:20px;"
        "color:#0f4d92}p,li{font-size:15px}li{margin:.45em 0}a{color:#0f4d92}"
        ".figure-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:22px}"
        "figure{margin:0;padding:12px;border:1px solid #dbe3ec;border-radius:9px;background:#fff}"
        "img{display:block;width:100%;height:auto}figcaption{padding:10px 4px 2px;color:#555;"
        "font-size:13px}footer{margin-top:38px;padding-top:16px;border-top:1px solid #ddd;"
        "font-size:12px;color:#777}@media(max-width:640px){body{padding:12px}main{padding:22px 18px}"
        ".figure-grid{grid-template-columns:1fr}}</style></head><body><main>"
        f"{_markdown_body(markdown_text)}{figure_html}"
        "<footer>图件由同一分析流程生成；点击图片可打开原始 PNG。</footer>"
        "</main></body></html>"
    )


def _write_report(
    markdown_text: str,
    stem: Path,
    figures: list[tuple[str, str]] | None = None,
) -> list[Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    markdown_path = stem.with_suffix(".md")
    html_path = stem.with_suffix(".html")
    markdown_path.write_text(markdown_text, encoding="utf-8")
    html_path.write_text(
        _markdown_to_html(markdown_text, stem.name, figures=figures), encoding="utf-8"
    )
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
        if group == "group1":
            ordered = data.sort_values("cosolvent_fraction")
            alpha_min = ordered.loc[ordered["alpha"].idxmin()]
            force_r = ordered[["cosolvent_fraction", "mean_force_eV_A"]].corr().iloc[0, 1]
            interpretation = (
                f"α 在 20% 丙酮时达到 {top['alpha']:.3f}，在 30% 时降至 "
                f"{alpha_min['alpha']:.3f}，说明浓度效应非单调。平均受力随浓度总体增加"
                f"（Pearson r={force_r:.3f}），但单条轨迹不足以建立因果关系。"
            )
            figures = [
                ("../figures/group1_concentration_response.png", "丙酮浓度与 α、OPA 平均受力的关系"),
                ("../figures/group1_window_sensitivity.png", "四个时间窗口的轨迹内敏感性"),
            ]
        elif group == "group2":
            low_force = data.loc[data["mean_force_eV_A"].idxmin()]
            interpretation = (
                f"CPME 和 DMC 的 α 居前；{low_force['system']} 的平均受力最低"
                f"（{low_force['mean_force_eV_A']:.3f} eV Å⁻¹）。颜色表示人工溶剂类别，"
                "类别差异仅作描述，不能替代独立重复。"
            )
            figures = [
                ("../figures/group2_pure_solvent_ranking.png", "13 种纯溶剂的 α 与平均受力排序"),
                ("../figures/combined_structure_dynamics.png", "全部体系的结构–动力学关系"),
            ]
        else:
            core = data.loc[data["role"] == "n-heptane-base-composition"]
            head_top = core.loc[core["head_coord_number_4A"].idxmax()]
            interpretation = (
                f"四个 n-heptane 基混合物中，{head_top['system']} 的 OPA 头部 4 Å 配位数最高"
                f"（{head_top['head_coord_number_4A']:.3f}）。按实际分子数归一后，THF 和甲苯"
                "在头部富集，丙酮与异丙醇相对贫化。thf_toluene 因结构轨迹与 CSV 不一致，"
                "只保留动力学结果。"
            )
            figures = [
                ("../figures/group3_composition_and_references.png", "Group 3 核心混合物与分层参考体系"),
                ("../figures/combined_cosolvent_enrichment.png", "混合体系中共溶剂的 OPA 头部/尾部富集"),
            ]
        markdown = f"""# {group} 分析

## 输入与方法

本组包含 {len(data)} 个条件；使用 OPA 质心 TAMSD、受力、4 Å 配位数和 RDF。每个条件仅有单条 10 ps 轨迹，因此排序属于描述性比较。

## 直接结果

- α 最高条件：{top['system']}（α={top['alpha']:.3f}）。
- 平均受力范围：{data['mean_force_eV_A'].min():.3f}–{data['mean_force_eV_A'].max():.3f} eV Å⁻¹。

## 图表解读

{interpretation}

## 解释边界

这些数据反映预吸附溶剂化，不直接等同于 SAM 覆盖率、倾角、缺陷密度或成膜质量。
"""
        outputs.extend(_write_report(markdown, output_root / "reports" / group, figures))
    return outputs


def render_combined_report(context: dict, output_root: Path) -> list[Path]:
    figures = [
        ("../figures/group1_concentration_response.png", "Group 1：丙酮/n-heptane 浓度响应"),
        ("../figures/group2_pure_solvent_ranking.png", "Group 2：纯溶剂动力学与受力排序"),
        ("../figures/group3_composition_and_references.png", "Group 3：不同组分及参考体系比较"),
        ("../figures/combined_cosolvent_enrichment.png", "共溶剂在 OPA 头部与尾部的局部富集"),
        ("../figures/combined_structure_dynamics.png", "全部体系的结构–动力学关系"),
        ("../figures/combined_representative_rdf.png", "代表性头部与尾部径向分布函数"),
    ]
    return _write_report(
        build_combined_markdown(context),
        Path(output_root) / "reports" / "combined_report",
        figures,
    )
