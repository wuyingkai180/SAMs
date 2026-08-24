"""Build one report containing nine pairwise mobility plots for Groups 1–3."""

from __future__ import annotations

import html
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from publication_reporting import (
    GROUP_TITLES,
    PALETTE,
    _labelled_scatter,
    _save,
    _style,
    display_label,
)


PAIRINGS = (
    (
        "force_vs_tamsd",
        "mean_force_eV_A",
        "alpha",
        "Mean force on OPA (eV Å$^{-1}$)",
        "TAMSD exponent, α",
        PALETTE["blue"],
        "平均力—TAMSD指数",
    ),
    (
        "max_displacement_vs_tamsd",
        "max_disp_A",
        "alpha",
        "Maximum OPA displacement (Å)",
        "TAMSD exponent, α",
        PALETTE["green"],
        "最大位移—TAMSD指数",
    ),
    (
        "force_vs_max_displacement",
        "mean_force_eV_A",
        "max_disp_A",
        "Mean force on OPA (eV Å$^{-1}$)",
        "Maximum OPA displacement (Å)",
        PALETTE["orange"],
        "平均力—最大位移",
    ),
)

PLOT_TITLES = {
    "force_vs_tamsd": "Mean force vs TAMSD exponent",
    "max_displacement_vs_tamsd": "Maximum displacement vs TAMSD exponent",
    "force_vs_max_displacement": "Mean force vs maximum displacement",
}


def _plot_pair(frame: pd.DataFrame, group: str, pairing: tuple[str, ...], figures: Path) -> list[Path]:
    stem, x, y, xlabel, ylabel, color, _ = pairing
    fig, ax = plt.subplots(figsize=(7.2, 5.6), constrained_layout=True)
    _labelled_scatter(ax, frame, x, y, xlabel, ylabel, color)
    ax.set_title(
        f"{GROUP_TITLES[group]}\n{PLOT_TITLES[stem]}",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    return _save(fig, figures / f"{group}_{stem}")


def _group_summary(frame: pd.DataFrame) -> dict[str, object]:
    max_force = frame.loc[frame["mean_force_eV_A"].idxmax()]
    max_disp = frame.loc[frame["max_disp_A"].idxmax()]
    max_alpha = frame.loc[frame["alpha"].idxmax()]
    corr = frame[["mean_force_eV_A", "max_disp_A", "alpha"]].corr(method="spearman")
    return {
        "max_force_system": display_label(str(max_force.system)),
        "max_force": float(max_force.mean_force_eV_A),
        "max_disp_system": display_label(str(max_disp.system)),
        "max_disp": float(max_disp.max_disp_A),
        "max_alpha_system": display_label(str(max_alpha.system)),
        "max_alpha": float(max_alpha.alpha),
        "rho_force_alpha": float(corr.loc["mean_force_eV_A", "alpha"]),
        "rho_disp_alpha": float(corr.loc["max_disp_A", "alpha"]),
        "rho_force_disp": float(corr.loc["mean_force_eV_A", "max_disp_A"]),
    }


def build_report(root: Path) -> list[Path]:
    root = Path(root)
    tables = root / "tables"
    figures = root / "figures"
    reports = root / "reports"
    figures.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    _style()

    frames = []
    outputs: list[Path] = []
    summaries: dict[str, dict[str, object]] = {}
    figure_cards: dict[str, list[str]] = {}
    for group in ("group1", "group2", "group3"):
        frame = pd.read_csv(tables / f"{group}_metrics.csv")
        required = {"group", "system", "mean_force_eV_A", "max_disp_A", "alpha"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{group} missing required columns: {sorted(missing)}")
        selected = frame[["group", "system", "mean_force_eV_A", "max_disp_A", "alpha"]].copy()
        selected["display_label"] = selected["system"].map(display_label)
        frames.append(selected)
        summaries[group] = _group_summary(frame)
        cards = []
        for pairing in PAIRINGS:
            outputs.extend(_plot_pair(frame, group, pairing, figures))
            stem, *_, caption = pairing
            cards.append(
                f'<figure><img src="../figures/{group}_{stem}.png" '
                f'alt="{html.escape(group)} {html.escape(caption)}">'
                f'<figcaption>{html.escape(caption)}。每个点均标注具体溶剂或混合组成。</figcaption></figure>'
            )
        figure_cards[group] = cards

    combined = pd.concat(frames, ignore_index=True)
    table_path = tables / "three_group_force_maxdisplacement_tamsd.csv"
    combined.to_csv(table_path, index=False, encoding="utf-8-sig")
    outputs.append(table_path)

    sections = []
    for group in ("group1", "group2", "group3"):
        s = summaries[group]
        group_table = combined.loc[combined.group == group].copy()
        group_table = group_table[["display_label", "mean_force_eV_A", "max_disp_A", "alpha"]]
        group_table.columns = ["体系", "平均力 (eV Å⁻¹)", "最大位移 (Å)", "TAMSD指数 α"]
        sections.append(f"""
<section><h2>{html.escape(GROUP_TITLES[group])}</h2>
<p>平均力最高：{html.escape(str(s['max_force_system']))}（{s['max_force']:.3f} eV Å⁻¹）；
最大位移最高：{html.escape(str(s['max_disp_system']))}（{s['max_disp']:.3f} Å）；
TAMSD指数最高：{html.escape(str(s['max_alpha_system']))}（α={s['max_alpha']:.3f}）。</p>
<p>Spearman相关：平均力–α，ρ={s['rho_force_alpha']:.3f}；最大位移–α，ρ={s['rho_disp_alpha']:.3f}；平均力–最大位移，ρ={s['rho_force_disp']:.3f}。相关系数仅描述本组数据，不表示因果关系或统计显著性。</p>
<div class="grid">{''.join(figure_cards[group])}</div>
<details><summary>查看本组数值</summary>{group_table.round(4).to_html(index=False, border=0, classes='metrics')}</details>
</section>""")

    report_path = reports / "three_group_force_maxdisplacement_tamsd_report.html"
    report_path.write_text(f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>三组OPA移动性指标两两比较</title>
<style>body{{font-family:Arial,'Microsoft YaHei',sans-serif;max-width:1500px;margin:auto;padding:28px;background:#edf1f4;color:#202428;line-height:1.65}}
main{{background:#fff;border:1.5px solid #202428;padding:32px 42px}}h1,h2{{color:#24557A}}.note{{border:1px solid #D17C2F;background:#fff8f0;padding:12px 16px}}
.grid{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:18px}}figure{{border:1.5px solid #202428;padding:9px;margin:0;background:#fff}}img{{width:100%;height:auto}}figcaption{{font-size:13px;margin-top:6px}}
table.metrics{{width:100%;border-collapse:collapse;margin-top:12px}}.metrics th,.metrics td{{border:1px solid #687078;padding:6px;text-align:right}}.metrics th:first-child,.metrics td:first-child{{text-align:left}}
section{{border-top:2px solid #24557A;margin-top:32px;padding-top:14px}}summary{{cursor:pointer;color:#24557A;font-weight:bold}}@media(max-width:1000px){{.grid{{grid-template-columns:1fr}}}}</style></head>
<body><main><h1>Group1–Group3：平均力、最大位移与TAMSD指数的两两比较</h1>
<div class="note"><strong>指标定义：</strong>平均力为完整轨迹中OPA受力大小的时间平均；最大位移为OPA相对初始位置在完整轨迹中达到的最大距离，不是最终帧位移；TAMSD使用幂律拟合指数α，描述位移随滞后时间的标度行为。三组只在各自组内比较，不建立跨组统一排名。</div>
<p>每组分别绘制平均力–α、最大位移–α和平均力–最大位移三张散点图，共9张。数值全集同时保存在CSV中。每个条件只有一条10 ps轨迹，因此图中关系属于描述性结果。</p>
{''.join(sections)}
</main></body></html>""", encoding="utf-8")
    outputs.append(report_path)
    return outputs


if __name__ == "__main__":
    build_report(Path("results/result2"))
