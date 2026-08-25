"""Build a concise, data-grounded preview report for Group 2 pure solvents."""

from __future__ import annotations

import html
from pathlib import Path

import pandas as pd

from .publication_reporting import display_label


ROOT = Path(__file__).resolve().parents[2]
RESULT2 = ROOT / "results" / "result2"
TABLE = RESULT2 / "tables" / "group2_metrics.csv"
REPORTS = RESULT2 / "reports"


def _fmt(value: float, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def _to_markdown(frame: pd.DataFrame, include_index: bool = False) -> str:
    """Render a compact Markdown table without pandas' optional tabulate dependency."""
    table = frame.reset_index() if include_index else frame.copy()
    columns = [str(value) for value in table.columns]

    def cell(value: object) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    lines.extend(
        "| " + " | ".join(cell(value) for value in row) + " |"
        for row in table.itertuples(index=False, name=None)
    )
    return "\n".join(lines)


def _metric_table(data: pd.DataFrame) -> pd.DataFrame:
    table = data[
        [
            "system",
            "alpha",
            "alpha_prefactor_A2_ps_alpha",
            "mean_force_eV_A",
            "max_disp_A",
            "head_coord_number_4A",
            "delta_d_MPa05",
            "delta_p_MPa05",
            "delta_h_MPa05",
        ]
    ].copy()
    table["system"] = table["system"].map(display_label)
    table.columns = [
        "溶剂",
        "TAMSD α",
        "Kα (Å² ps⁻α)",
        "Mean force (eV Å⁻¹)",
        "最大位移 (Å)",
        "头基CN (4 Å)",
        "δD",
        "δP",
        "δH",
    ]
    numeric = table.columns[1:]
    table[numeric] = table[numeric].round(3)
    return table.sort_values("TAMSD α", ascending=False).reset_index(drop=True)


def _correlation_table(data: pd.DataFrame) -> pd.DataFrame:
    columns = {
        "alpha": "TAMSD α",
        "alpha_prefactor_A2_ps_alpha": "Kα",
        "mean_force_eV_A": "Mean force",
        "max_disp_A": "最大位移",
        "head_coord_number_4A": "头基CN",
        "delta_d_MPa05": "δD",
        "delta_p_MPa05": "δP",
        "delta_h_MPa05": "δH",
    }
    corr = data[list(columns)].corr(method="spearman").rename(
        index=columns, columns=columns
    )
    return corr.round(3)


def build() -> tuple[Path, Path]:
    data = pd.read_csv(TABLE)
    if len(data) != 14 or set(data["group"]) != {"group2"}:
        raise ValueError("expected the validated 14-solvent Group 2 table")

    metrics = _metric_table(data)
    corr = _correlation_table(data)
    acetone = data.loc[data["system"] == "acetone"].iloc[0]
    heptane = data.loc[data["system"] == "n-heptane"].iloc[0]

    summary = (
        f"本预览比较14种纯溶剂在10 ps轨迹中的OPA动力学、受力和局域溶剂化响应。"
        f"CPME具有最高TAMSD指数（α={_fmt(data['alpha'].max())}）；"
        f"n-heptane具有最高TAMSD前因子（Kα={_fmt(data['alpha_prefactor_A2_ps_alpha'].max())}）"
        f"和最大位移（{_fmt(data['max_disp_A'].max())} Å）。"
        f"mean force与头基4 Å配位数呈较强正向秩相关（ρ={_fmt(corr.loc['Mean force', '头基CN'])}），"
        f"并与HSP氢键分量δH呈较强正向秩相关（ρ={_fmt(corr.loc['Mean force', 'δH'])}）。"
        "这些关系属于单轨迹、小样本的描述性结果，不能直接解释为溶解度或因果机制。"
    )

    comparison_rows = [
        [
            "acetone",
            _fmt(acetone.alpha),
            _fmt(acetone.alpha_prefactor_A2_ps_alpha),
            _fmt(acetone.mean_force_eV_A),
            _fmt(acetone.max_disp_A),
            _fmt(acetone.head_coord_number_4A),
        ],
        [
            "n-heptane",
            _fmt(heptane.alpha),
            _fmt(heptane.alpha_prefactor_A2_ps_alpha),
            _fmt(heptane.mean_force_eV_A),
            _fmt(heptane.max_disp_A),
            _fmt(heptane.head_coord_number_4A),
        ],
    ]
    comparison = pd.DataFrame(
        comparison_rows,
        columns=["溶剂", "TAMSD α", "Kα", "Mean force", "最大位移", "头基CN"],
    )

    findings = [
        (
            "运动类型与幅度",
            f"α与Kα高度同向（Spearman ρ={_fmt(corr.loc['TAMSD α', 'Kα'])}），"
            f"α与最大位移也高度同向（ρ={_fmt(corr.loc['TAMSD α', '最大位移'])}）。"
            "这说明本组中运动更持续的体系通常也表现出更大的轨迹尺度，但三项指标并非相互独立的证据。",
        ),
        (
            "受力与局域溶剂化",
            f"mean force与头基CN的ρ={_fmt(corr.loc['Mean force', '头基CN'])}。"
            "较高受力通常伴随更多头基邻近原子，但mean force取合力模长，不能区分吸引、排斥和随机碰撞。",
        ),
        (
            "HSP机制线索",
            f"mean force与δH的ρ={_fmt(corr.loc['Mean force', 'δH'])}，"
            f"与δP的ρ={_fmt(corr.loc['Mean force', 'δP'])}。"
            "结果支持氢键/极性性质可能影响OPA受力环境，但缺少OPA的可靠HSP与作用半径R0，不能计算真实Ra/RED溶解度判据。",
        ),
        (
            "acetone与n-heptane",
            f"n-heptane的α、Kα和最大位移均高于acetone，而mean force较低"
            f"（{_fmt(heptane.mean_force_eV_A)} 对 {_fmt(acetone.mean_force_eV_A)} eV Å⁻¹）。"
            "在该10 ps轨迹中，这更接近“较自由、净移动更大”的动力学表现，不能单独推出其更利于SAM成膜。",
        ),
    ]

    md_findings = "\n\n".join(
        f"### {title}\n\n{text}" for title, text in findings
    )
    markdown = f"""# Group 2纯溶剂综合分析：预览报告

**日期：** 2026-08-25  
**范围：** 14种纯溶剂，OPA/SAM体系，单条0–10 ps轨迹  
**证据：** Group 2 MD数据、结构轨迹及项目内已记录的纯溶剂HSP表

## 执行摘要

{summary}

## 研究问题

现有MD响应量（mean force、TAMSD、位移、RDF和配位数）能否与Hansen溶解度参数结合，用于解释不同纯溶剂对OPA运动和局域溶剂化环境的影响？

## 数据与方法

- 每种溶剂使用101帧、0–10 ps轨迹。
- TAMSD拟合为 $\\overline{{\\delta^2(\\tau)}}=K_\\alpha\\tau^\\alpha$。
- mean force为 $\\langle|F_{{\\mathrm{{OPA}}}}|\\rangle$，只表示受力大小，不保留方向。
- 局域结构使用OPA头基4 Å配位数和RDF描述。
- HSP使用项目表中的 $\\delta_D$、$\\delta_P$、$\\delta_H$；相关性使用14种溶剂间的Spearman秩相关。
- 所有相关性均为探索性描述，不作为因果或统计确认。

## 主要结果

{md_findings}

## acetone与n-heptane直接比较

{_to_markdown(comparison)}

![完整动力学与TAMSD比较](../figures/group2_full_time_mobility.png)

## HSP与MD响应

![HSP参数](../figures/group2_hansen_parameters.png)

![HSP与动力学关系](../figures/group2_hansen_relationships.png)

## 14种溶剂数据表

{_to_markdown(metrics)}

## Spearman相关矩阵

{_to_markdown(corr, include_index=True)}

## 解释边界

- mean force不能直接转换为黏度、介电常数、表面张力、Kamlet–Taft参数、Hamaker常数或溶解度。
- 当前体系没有液–气界面，不能由现有轨迹计算表面张力。
- 10 ps、单轨迹不足以获得收敛黏度、介电常数和宏观扩散系数。
- 没有OPA的验证HSP和R0，不能将HSPiPy区域称为OPA实验溶解度球。
- CPME、n-heptane或其他溶剂在单项指标领先，不等于SAM生长综合性能最佳。

## 下一阶段建议

1. 为14种纯溶剂补充同一温度附近的黏度、介电常数、表面张力和Kamlet–Taft参数，并保留逐项来源。
2. 将外部参数作为解释变量，将mean force、TAMSD、最大位移和头基CN作为MD响应变量。
3. 只做Spearman相关、标准化热图和类别比较；在获得重复轨迹前不训练复杂预测模型。
4. 若研究目标是溶解度或吸附自由能，应新增solvation free-energy或umbrella-sampling/PMF计算。

## 数据来源

- [Group 2完整指标表](../tables/group2_metrics.csv)
- [纯溶剂HSP表](../tables/group2_hansen_parameters.csv)
- HSP来源字段记录为 Hansen2007，DOI: https://doi.org/10.1201/9781420006834

## AI使用说明

本报告由AI辅助整理项目数据、计算描述性相关并生成文字草稿。所有数值均来自项目表格；结论仍需研究者结合模拟设置、重复轨迹和实验结果审核。
"""

    findings_html = "".join(
        f"<section><h3>{html.escape(title)}</h3><p>{html.escape(text)}</p></section>"
        for title, text in findings
    )
    css = """
    :root{--ink:#172027;--muted:#56616a;--line:#cfd6dc;--blue:#24557a;--paper:#fff}
    *{box-sizing:border-box}body{margin:0;background:#eef1f3;color:var(--ink);font-family:Arial,"Microsoft YaHei",sans-serif;line-height:1.65}
    main{max-width:1180px;margin:24px auto;background:var(--paper);padding:34px 42px;border:1px solid var(--line)}
    h1{font-size:28px;margin:0 0 8px}h2{font-size:20px;margin-top:32px;border-bottom:2px solid var(--blue);padding-bottom:5px}h3{font-size:16px;margin-bottom:4px}
    .meta,.note{color:var(--muted)}.summary{background:#eef5f8;border-left:5px solid var(--blue);padding:14px 18px}
    table{border-collapse:collapse;width:100%;font-size:13px;margin:14px 0 24px}th,td{border:1px solid var(--line);padding:7px 8px;text-align:right}th{background:#edf2f5}th:first-child,td:first-child{text-align:left}
    figure{margin:20px 0}img{display:block;width:100%;height:auto;border:1px solid #202428;background:white}figcaption{font-size:12px;color:var(--muted);margin-top:5px}
    a{color:var(--blue)}code{font-size:12px}@media(max-width:760px){main{margin:0;padding:20px 16px;overflow-x:auto}}
    """
    html_report = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Group 2纯溶剂综合分析预览</title><style>{css}</style></head><body><main>
    <h1>Group 2纯溶剂综合分析：预览报告</h1>
    <p class="meta">日期：2026-08-25　|　14种纯溶剂　|　单条0–10 ps轨迹</p>
    <h2>执行摘要</h2><p class="summary">{html.escape(summary)}</p>
    <h2>数据与方法</h2><ul><li>101帧/体系；TAMSD拟合为 δ²(τ)=Kατ<sup>α</sup>。</li><li>mean force为OPA合力模长的时间平均，不保留方向。</li><li>结构响应使用4 Å头基配位数与RDF；HSP来自项目内记录表。</li><li>相关性为Spearman秩相关，只作探索性描述。</li></ul>
    <h2>主要结果</h2>{findings_html}
    <h2>acetone与n-heptane直接比较</h2>{comparison.to_html(index=False, border=0)}
    <figure><img src="../figures/group2_full_time_mobility.png" alt="Group 2完整动力学与TAMSD"><figcaption>图1　完整0–10 ps运动、受力、净移动率与TAMSD；右下角只标注acetone和n-heptane。</figcaption></figure>
    <h2>HSP与MD响应</h2>
    <figure><img src="../figures/group2_hansen_parameters.png" alt="HSP参数"><figcaption>图2　14种纯溶剂的Hansen参数。</figcaption></figure>
    <figure><img src="../figures/group2_hansen_relationships.png" alt="HSP与MD响应"><figcaption>图3　HSP与OPA动力学、受力及局域结构的探索性关系。</figcaption></figure>
    <h2>14种溶剂指标</h2>{metrics.to_html(index=False, border=0)}
    <h2>Spearman相关矩阵</h2>{corr.to_html(border=0)}
    <h2>解释边界</h2><ul><li>mean force不能直接转换为黏度、介电常数、表面张力、Kamlet–Taft参数、Hamaker常数或溶解度。</li><li>10 ps单轨迹不足以获得收敛的宏观输运性质。</li><li>缺少OPA验证HSP和R0，不能给出真实Ra/RED溶解度判据。</li><li>单项指标领先不等于SAM生长综合性能最佳。</li></ul>
    <h2>下一阶段建议</h2><ol><li>补充同温度黏度、介电常数、表面张力和Kamlet–Taft参数及逐项来源。</li><li>外部物性作为解释变量，MD指标作为响应变量。</li><li>优先做Spearman相关、标准化热图和溶剂类别比较。</li><li>若目标是溶解度或吸附自由能，新增自由能或PMF模拟。</li></ol>
    <h2>数据来源</h2><ul><li><a href="../tables/group2_metrics.csv">Group 2完整指标表</a></li><li><a href="../tables/group2_hansen_parameters.csv">纯溶剂HSP表</a></li><li>HSP来源字段：Hansen2007，DOI：<a href="https://doi.org/10.1201/9781420006834">10.1201/9781420006834</a></li></ul>
    <h2>AI使用说明</h2><p class="note">本报告由AI辅助整理项目数据、计算描述性相关并生成文字草稿。所有数值来自项目表格；结论仍需研究者结合模拟设置、重复轨迹和实验结果审核。</p>
    </main></body></html>"""

    REPORTS.mkdir(parents=True, exist_ok=True)
    md_path = REPORTS / "group2_integrated_solvent_report_preview.md"
    html_path = REPORTS / "group2_integrated_solvent_report_preview.html"
    md_path.write_text(markdown, encoding="utf-8")
    html_path.write_text(html_report, encoding="utf-8")
    return md_path, html_path


if __name__ == "__main__":
    for output in build():
        print(output)
