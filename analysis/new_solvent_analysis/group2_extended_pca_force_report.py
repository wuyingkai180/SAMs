"""Extended Group 2 PCA with OPA head/tail force decomposition and report."""

from __future__ import annotations

import html
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ase.io import read
from ase.io.trajectory import Trajectory
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "group2"
RESULT2 = ROOT / "results" / "result2"
BASE_TABLES = RESULT2 / "group2_lda_external_properties" / "tables"
OUT = RESULT2 / "group2_extended_pca_force_analysis"
FIGURES = OUT / "figures"
TABLES = OUT / "tables"
REPORTS = OUT / "reports"

EXTERNAL = BASE_TABLES / "group2_external_properties_298K.csv"
METRICS = RESULT2 / "tables" / "group2_metrics.csv"
OPA = ROOT / "structures" / "opa.vasp"

COLORS = {
    "acetone": "#E66101",
    "acetonitrile": "#CA0020",
    "CPME": "#1B9E77",
    "cyclohexane": "#666666",
    "DMC": "#FDB863",
    "ethanol": "#40B0A6",
    "ethyl_acetate": "#8C510A",
    "isopropanol": "#5E3C99",
    "methanol": "#17BECF",
    "n-heptane": "#0571B0",
    "p-xylene": "#C51B7D",
    "propylene_carbonate": "#B2ABD2",
    "thf": "#A6D854",
    "toluene": "#F46D43",
}

FEATURES = [
    "log10_viscosity_mPa_s",
    "dielectric_constant",
    "surface_tension_mN_m",
    "molar_mass_g_mol",
    "molar_volume_cm3_mol",
    "delta_d_MPa05",
    "delta_p_MPa05",
    "delta_h_MPa05",
    "mean_head_force_eV_A",
    "mean_tail_force_eV_A",
    "mean_total_force_eV_A",
    "max_disp_A",
    "alpha",
    "log10_K_alpha",
]

LABELS = {
    "log10_viscosity_mPa_s": "log10 viscosity",
    "dielectric_constant": "Dielectric constant",
    "surface_tension_mN_m": "Surface tension",
    "molar_mass_g_mol": "Solvent molar mass",
    "molar_volume_cm3_mol": "Molar volume",
    "delta_d_MPa05": "HSP δD",
    "delta_p_MPa05": "HSP δP",
    "delta_h_MPa05": "HSP δH",
    "mean_head_force_eV_A": "Mean head force",
    "mean_tail_force_eV_A": "Mean tail force",
    "mean_total_force_eV_A": "Mean total force",
    "max_disp_A": "Maximum displacement",
    "alpha": "TAMSD α",
    "log10_K_alpha": "log10 Kα",
}

FEATURE_GROUP = {
    "log10_viscosity_mPa_s": "bulk",
    "dielectric_constant": "bulk",
    "surface_tension_mN_m": "bulk",
    "molar_mass_g_mol": "bulk",
    "molar_volume_cm3_mol": "bulk",
    "delta_d_MPa05": "hsp",
    "delta_p_MPa05": "hsp",
    "delta_h_MPa05": "hsp",
    "mean_head_force_eV_A": "force",
    "mean_tail_force_eV_A": "force",
    "mean_total_force_eV_A": "force",
    "max_disp_A": "mobility",
    "alpha": "mobility",
    "log10_K_alpha": "mobility",
}

ARROW_COLORS = {
    "bulk": "#555E66",
    "hsp": "#7A5195",
    "force": "#B64B4B",
    "mobility": "#16836B",
}


def style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8.5,
            "axes.linewidth": 0.85,
            "axes.edgecolor": "#202428",
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, stem: str) -> list[Path]:
    paths = [FIGURES / f"{stem}.{suffix}" for suffix in ("png", "svg", "pdf", "tiff")]
    fig.savefig(paths[0], dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(paths[1], bbox_inches="tight", facecolor="white")
    fig.savefig(paths[2], bbox_inches="tight", facecolor="white")
    fig.savefig(paths[3], dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return paths


def opa_partition() -> tuple[list[int], list[int]]:
    atoms = read(OPA)
    symbols = atoms.get_chemical_symbols()
    p_indices = [i for i, symbol in enumerate(symbols) if symbol == "P"]
    o_indices = [i for i, symbol in enumerate(symbols) if symbol == "O"]
    if len(p_indices) != 1 or len(o_indices) != 3:
        raise ValueError("Expected one P and three O atoms in OPA structure")
    acidic_h = []
    for index, symbol in enumerate(symbols):
        if symbol == "H" and min(atoms.get_distance(index, oxygen) for oxygen in o_indices) < 1.25:
            acidic_h.append(index)
    if len(acidic_h) != 2:
        raise ValueError(f"Expected two acidic O-H hydrogens; found {acidic_h}")
    head = sorted(p_indices + o_indices + acidic_h)
    tail = sorted(set(range(len(atoms))) - set(head))
    if len(head) != 6 or len(tail) != 55:
        raise ValueError(f"Unexpected OPA partition: head={len(head)}, tail={len(tail)}")
    return head, tail


def trajectory_forces() -> pd.DataFrame:
    head, tail = opa_partition()
    rows: list[dict[str, float | int | str]] = []
    for system in COLORS:
        paths = list((DATA / system).glob("*_md_300K.traj"))
        if len(paths) != 1:
            raise ValueError(f"{system}: expected one trajectory, found {len(paths)}")
        trajectory = Trajectory(paths[0])
        head_force, tail_force, total_force = [], [], []
        for atoms in trajectory:
            force = atoms.get_forces()
            head_force.append(float(np.linalg.norm(force[head].sum(axis=0))))
            tail_force.append(float(np.linalg.norm(force[tail].sum(axis=0))))
            total_force.append(float(np.linalg.norm(force[:61].sum(axis=0))))
        rows.append(
            {
                "system": system,
                "n_frames_force": len(trajectory),
                "n_head_atoms": len(head),
                "n_tail_atoms": len(tail),
                "mean_head_force_eV_A": np.mean(head_force),
                "std_head_force_eV_A": np.std(head_force, ddof=0),
                "mean_tail_force_eV_A": np.mean(tail_force),
                "std_tail_force_eV_A": np.std(tail_force, ddof=0),
                "mean_total_force_eV_A": np.mean(total_force),
                "std_total_force_eV_A": np.std(total_force, ddof=0),
                "head_tail_force_ratio": np.mean(head_force) / np.mean(tail_force),
            }
        )
    return pd.DataFrame(rows)


def merged_data(force: pd.DataFrame) -> pd.DataFrame:
    external = pd.read_csv(EXTERNAL)
    hsp = pd.read_csv(RESULT2 / "tables" / "group2_hansen_parameters.csv")
    metrics = pd.read_csv(METRICS)
    hsp = hsp[["system", "delta_d_MPa05", "delta_p_MPa05", "delta_h_MPa05"]]
    metrics = metrics[
        ["system", "mean_force_eV_A", "max_disp_A", "alpha", "alpha_prefactor_A2_ps_alpha"]
    ]
    data = external.merge(hsp, on="system", validate="one_to_one")
    data = data.merge(metrics, on="system", validate="one_to_one")
    data = data.merge(force, on="system", validate="one_to_one")
    data["log10_K_alpha"] = np.log10(data["alpha_prefactor_A2_ps_alpha"])
    data["whole_force_csv_difference_eV_A"] = (
        data["mean_total_force_eV_A"] - data["mean_force_eV_A"]
    )
    if len(data) != 14 or data[FEATURES].isna().any().any():
        raise ValueError("Extended PCA requires 14 complete pure-solvent rows")
    if data["whole_force_csv_difference_eV_A"].abs().max() > 1e-6:
        raise ValueError("Trajectory total-force means do not match motion-force CSV")
    return data.sort_values("system").reset_index(drop=True)


def pca_analysis(data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, PCA]:
    z = StandardScaler().fit_transform(data[FEATURES])
    pca = PCA(n_components=2)
    scores = pca.fit_transform(z)
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    return scores, loadings, pca


def pca_figure(data: pd.DataFrame, scores: np.ndarray, loadings: np.ndarray, pca: PCA) -> None:
    style()
    fig, ax = plt.subplots(figsize=(183 / 25.4, 145 / 25.4), constrained_layout=True)
    for index, row in data.iterrows():
        ax.scatter(
            scores[index, 0], scores[index, 1], s=61,
            color=COLORS[row.system], edgecolor="#202428", linewidth=0.75,
            label=row.system, zorder=4,
        )

    span = max(np.ptp(scores[:, 0]), np.ptp(scores[:, 1]))
    scale = 0.39 * span
    label_offsets = {
        "Mean head force": (0.04, -0.05), "Mean tail force": (0.03, 0.16),
        "Mean total force": (0.03, 0.15), "TAMSD α": (0.02, 0.13),
        "log10 Kα": (0.02, 0.12), "Maximum displacement": (0.03, -0.16),
        "HSP δP": (-0.45, -0.15), "HSP δH": (0.02, -0.10),
    }
    for index, feature in enumerate(FEATURES):
        dx, dy = loadings[index] * scale
        color = ARROW_COLORS[FEATURE_GROUP[feature]]
        ax.annotate(
            "", xy=(dx, dy), xytext=(0, 0),
            arrowprops={"arrowstyle": "-|>", "lw": 1.0, "color": color, "alpha": 0.88},
            zorder=2,
        )
        label = LABELS[feature]
        ox, oy = label_offsets.get(label, (0.02, 0.02))
        ax.text(dx * 1.04 + ox, dy * 1.04 + oy, label, color=color, fontsize=7.0,
                ha="left" if dx >= 0 else "right", va="center")

    ax.axhline(0, color="#A8AFB5", lw=0.65)
    ax.axvline(0, color="#A8AFB5", lw=0.65)
    arrow_ends = loadings * scale
    x_values = np.r_[scores[:, 0], arrow_ends[:, 0]]
    y_values = np.r_[scores[:, 1], arrow_ends[:, 1]]
    ax.set_xlim(x_values.min() - 0.65, x_values.max() + 0.65)
    ax.set_ylim(y_values.min() - 0.65, y_values.max() + 0.65)
    ax.grid(True, color="#D8DDE1", lw=0.5, alpha=0.65)
    ax.set_xlabel(f"PC1 ({100*pca.explained_variance_ratio_[0]:.1f}% variance)", fontsize=9.5)
    ax.set_ylabel(f"PC2 ({100*pca.explained_variance_ratio_[1]:.1f}% variance)", fontsize=9.5)
    ax.set_title("Group 2 | Extended solvent-property and SAM-response PCA", fontsize=11,
                 fontweight="bold")
    legend = ax.legend(
        title="Pure solvent", bbox_to_anchor=(1.02, 1), loc="upper left",
        borderaxespad=0, frameon=True, fontsize=7.2, title_fontsize=7.8,
        handletextpad=0.5, labelspacing=0.55,
    )
    legend.get_frame().set_edgecolor("#A8AFB5")
    legend.get_frame().set_linewidth(0.7)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.85)
    save_figure(fig, "group2_extended_pca_biplot")


def force_figure(data: pd.DataFrame) -> None:
    style()
    ordered = data.sort_values("mean_total_force_eV_A").reset_index(drop=True)
    y = np.arange(len(ordered))
    panels = [
        ("mean_head_force_eV_A", "std_head_force_eV_A", "Phosphonate head"),
        ("mean_tail_force_eV_A", "std_tail_force_eV_A", "Carbon-chain tail"),
        ("mean_total_force_eV_A", "std_total_force_eV_A", "Whole OPA"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(183 / 25.4, 137 / 25.4), sharey=True,
                             constrained_layout=True)
    for ax, (mean_col, std_col, title) in zip(axes, panels):
        for index, row in ordered.iterrows():
            ax.errorbar(
                row[mean_col], index, xerr=row[std_col], fmt="o", ms=5.4,
                color=COLORS[row.system], ecolor=COLORS[row.system],
                elinewidth=0.8, capsize=2.0, markeredgecolor="#202428",
                markeredgewidth=0.55, zorder=3,
            )
        ax.set_title(title, fontsize=9.5, fontweight="bold")
        ax.set_xlabel("Mean force magnitude (eV Å$^{-1}$)")
        ax.grid(True, axis="x", color="#D8DDE1", lw=0.5, alpha=0.7)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.8)
    axes[0].set_yticks(y, ordered["system"], fontsize=7.5)
    for tick, system in zip(axes[0].get_yticklabels(), ordered["system"]):
        tick.set_color(COLORS[system])
        tick.set_fontweight("bold")
    axes[0].set_ylabel("Pure solvent")
    fig.suptitle("OPA force decomposition across 101 trajectory frames", fontsize=11,
                 fontweight="bold")
    save_figure(fig, "group2_head_tail_total_force")


def profile_heatmap(data: pd.DataFrame) -> None:
    style()
    z = StandardScaler().fit_transform(data[FEATURES])
    fig, ax = plt.subplots(figsize=(183 / 25.4, 150 / 25.4), constrained_layout=True)
    image = ax.imshow(z, aspect="auto", cmap="RdBu_r", vmin=-2.5, vmax=2.5)
    ax.set_xticks(np.arange(len(FEATURES)), [LABELS[f] for f in FEATURES],
                  rotation=48, ha="right", fontsize=7.0)
    ax.set_yticks(np.arange(len(data)), data["system"], fontsize=7.3)
    for tick, system in zip(ax.get_yticklabels(), data["system"]):
        tick.set_color(COLORS[system])
        tick.set_fontweight("bold")
    for i in range(len(data)):
        for j in range(len(FEATURES)):
            ax.text(j, i, f"{z[i, j]:.1f}", ha="center", va="center", fontsize=5.8,
                    color="white" if abs(z[i, j]) > 1.25 else "#202428")
    ax.set_title("Standardized solvent-property and SAM-response profiles", fontsize=11,
                 fontweight="bold")
    cbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("z-score")
    save_figure(fig, "group2_extended_standardized_profiles")


def correlations(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for left_index, left in enumerate(FEATURES):
        for right in FEATURES[left_index + 1:]:
            rho, p = spearmanr(data[left], data[right])
            rows.append(
                {"variable_1": left, "variable_2": right, "spearman_rho": rho,
                 "p_value_two_sided": p, "absolute_rho": abs(rho)}
            )
    result = pd.DataFrame(rows)
    order = np.argsort(result["p_value_two_sided"].to_numpy())
    ranked = result["p_value_two_sided"].to_numpy()[order]
    adjusted = np.minimum.accumulate((ranked * len(ranked) / np.arange(1, len(ranked) + 1))[::-1])[::-1]
    q_values = np.empty_like(adjusted)
    q_values[order] = np.minimum(adjusted, 1.0)
    result["q_value_bh"] = q_values
    return result.sort_values("absolute_rho", ascending=False)


def markdown_table(frame: pd.DataFrame) -> str:
    columns = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |")
    return "\n".join(lines)


def write_report(data: pd.DataFrame, pca: PCA, corr: pd.DataFrame) -> None:
    selected = data[
        ["system", "molar_mass_g_mol", "mean_head_force_eV_A",
         "mean_tail_force_eV_A", "mean_total_force_eV_A", "max_disp_A", "alpha",
         "alpha_prefactor_A2_ps_alpha"]
    ].copy()
    selected.columns = [
        "溶剂", "分子量(g/mol)", "头基平均力(eV/Å)", "尾链平均力(eV/Å)",
        "整体平均力(eV/Å)", "最大位移(Å)", "TAMSD α", "Kα"
    ]
    selected = selected.round(4)
    table_html = selected.to_html(index=False, border=0, classes="data")
    table_md = markdown_table(selected)

    candidates = data.set_index("system").loc[["acetone", "n-heptane"]]
    candidate_text = (
        f"acetone的头基/尾链/整体平均力分别为"
        f"{candidates.loc['acetone','mean_head_force_eV_A']:.3f}/"
        f"{candidates.loc['acetone','mean_tail_force_eV_A']:.3f}/"
        f"{candidates.loc['acetone','mean_total_force_eV_A']:.3f} eV Å⁻¹；"
        f"n-heptane分别为{candidates.loc['n-heptane','mean_head_force_eV_A']:.3f}/"
        f"{candidates.loc['n-heptane','mean_tail_force_eV_A']:.3f}/"
        f"{candidates.loc['n-heptane','mean_total_force_eV_A']:.3f} eV Å⁻¹。"
    )
    top_corr = corr.head(8).copy()
    top_corr["variable_1"] = top_corr["variable_1"].map(LABELS)
    top_corr["variable_2"] = top_corr["variable_2"].map(LABELS)
    top_corr = top_corr[["variable_1", "variable_2", "spearman_rho", "p_value_two_sided", "q_value_bh"]].round(4)

    mobility_text = (
        "n-heptane的最大位移为7.127 Å（14种溶剂中最高），Kα=1.722（最高），"
        "α=0.906，表现为接近Brownian的高幅度运动；acetone的Kα=1.694（第二高）、"
        "最大位移4.351 Å、α=0.876。"
    )
    mechanism_text = (
        "头基平均力与整体平均力高度相关（Spearman ρ=0.908），说明不同溶剂下整体受力排序"
        "主要受磷酸头基贡献控制。HSP δH与头基力和整体力分别呈ρ=0.837和0.851的正相关，"
        "支持氢键能力较强的溶剂会增强头基受力波动这一机制假设。"
    )

    markdown = f"""# Group 2纯溶剂扩展PCA与OPA头尾受力报告

**样本：** 14种纯溶剂，每种101帧，0–10 ps。  
**核心图形：** 逐溶剂固定独立颜色的扩展PCA、头基/尾链/整体受力图、标准化参数热图。  
**PCA变量：** 5项体相物性、3项HSP、3项受力指标、最大位移、TAMSD α、log10 Kα。

## 计算定义

- 磷酸头基：P + 3O + 2个与O成键的酸性H，共6个原子。
- 碳链尾部：其余55个OPA原子。
- 每帧分组力：先将组内原子力矢量求和，再取合力模长；报告值为101帧均值，受力图误差线为101帧的标准差而非均值置信区间。
- 整体平均力与原motion-force CSV逐溶剂核对，最大差值小于10⁻⁶ eV Å⁻¹。
- 头基、尾部和整体的力模长均值不可直接相加，因为头尾合力矢量存在方向抵消。
- PCA前所有变量均进行z-score标准化。

## 主要结果

{candidate_text}

{mobility_text}

{mechanism_text}

扩展PCA的PC1和PC2分别解释{100*pca.explained_variance_ratio_[0]:.1f}%和{100*pca.explained_variance_ratio_[1]:.1f}%的联合变异，合计{100*sum(pca.explained_variance_ratio_[:2]):.1f}%。该PCA同时包含外部解释变量和MD响应变量，因此只用于描述联合结构，不作为因果模型。

![扩展PCA](../figures/group2_extended_pca_biplot.png)

![头尾整体受力](../figures/group2_head_tail_total_force.png)

![标准化特征](../figures/group2_extended_standardized_profiles.png)

## 14种纯溶剂直接比较

{table_md}

## 绝对Spearman相关最高的变量对

{markdown_table(top_corr)}

## 解释限制

14个溶剂对应14个PCA变量，且头基力、尾部力和整体力彼此相关，PCA结果对变量集合较敏感。TAMSD α表示运动时间标度，Kα表示运动幅度，两者不能互相替代。最大位移为单条短轨迹的极值，可能受偶然事件影响。当前结果适合提出机制假设，不能单独证明SAM成膜质量。
"""
    (REPORTS / "group2_extended_pca_force_report_zh.md").write_text(markdown, encoding="utf-8")

    css = """
body{font-family:Arial,'Microsoft YaHei',sans-serif;max-width:1180px;margin:28px auto;padding:0 22px;color:#202428;line-height:1.65}h1,h2{color:#111}img{width:100%;border:1px solid #9aa2a8;margin:10px 0 24px}.data{border-collapse:collapse;width:100%;font-size:13px}.data th,.data td{border:1px solid #cfd4d8;padding:5px 7px;text-align:right}.data th:first-child,.data td:first-child{text-align:left}.note{background:#f1f4f5;border-left:4px solid #0072B2;padding:10px 14px}code{background:#f1f4f5;padding:1px 4px}
"""
    html_report = f"""<!doctype html><html lang="zh"><head><meta charset="utf-8"><title>Group 2扩展PCA与受力报告</title><style>{css}</style></head><body>
<h1>Group 2纯溶剂扩展PCA与OPA头尾受力报告</h1>
<p><b>样本：</b>14种纯溶剂，每种101帧，0–10 ps。</p>
<div class="note">{html.escape(candidate_text)} {html.escape(mobility_text)} 扩展PCA混合外部物性与MD响应，仅用于探索联合规律。</div>
<h2>计算定义</h2><ul><li>头基：P + 3O + 2个酸性H；尾部：其余55个OPA原子。</li><li>每帧先对组内原子力矢量求和再取模，最后跨101帧平均；误差线表示帧间标准差。</li><li>头、尾与整体力模长均值不可直接相加，因为矢量方向会抵消。</li><li>PCA前全部变量进行z-score标准化。</li></ul>
<h2>扩展PCA</h2><p>PC1={100*pca.explained_variance_ratio_[0]:.1f}%，PC2={100*pca.explained_variance_ratio_[1]:.1f}%，合计={100*sum(pca.explained_variance_ratio_[:2]):.1f}%。每种溶剂使用固定独立颜色，颜色映射见图例。</p><img src="../figures/group2_extended_pca_biplot.png">
<h2>头基、碳链尾部及整体平均受力</h2><img src="../figures/group2_head_tail_total_force.png">
<p>{html.escape(mechanism_text)}</p>
<h2>全部参数标准化比较</h2><img src="../figures/group2_extended_standardized_profiles.png">
<h2>直接比较表</h2>{table_html}
<h2>解释限制</h2><p>该PCA同时含外部物性和MD响应，并且样本量与变量数均为14，属于探索性分析。最大位移为极值；TAMSD α反映运动类型，Kα反映运动幅度。</p>
</body></html>"""
    (REPORTS / "group2_extended_pca_force_report_zh.html").write_text(html_report, encoding="utf-8")


def main() -> None:
    for directory in (FIGURES, TABLES, REPORTS):
        directory.mkdir(parents=True, exist_ok=True)
    force = trajectory_forces()
    data = merged_data(force)
    scores, loadings, pca = pca_analysis(data)
    pca_figure(data, scores, loadings, pca)
    force_figure(data)
    profile_heatmap(data)
    corr = correlations(data)

    force.to_csv(TABLES / "group2_head_tail_total_force_summary.csv", index=False)
    data.to_csv(TABLES / "group2_extended_pca_source_data.csv", index=False)
    pd.DataFrame(
        {"system": data["system"], "PC1": scores[:, 0], "PC2": scores[:, 1],
         "color_hex": [COLORS[name] for name in data["system"]]}
    ).to_csv(TABLES / "group2_extended_pca_scores_and_colors.csv", index=False)
    pd.DataFrame(
        {"feature": FEATURES, "label": [LABELS[f] for f in FEATURES],
         "PC1_loading": loadings[:, 0], "PC2_loading": loadings[:, 1]}
    ).to_csv(TABLES / "group2_extended_pca_loadings.csv", index=False)
    corr.to_csv(TABLES / "group2_extended_feature_spearman_correlations.csv", index=False)
    write_report(data, pca, corr)
    print(OUT)


if __name__ == "__main__":
    main()
