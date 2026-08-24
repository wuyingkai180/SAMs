"""HSPiPy analysis for the 14 pure-solvent Group 2 systems.

The fitted labels are explicitly mobility-defined (TAMSD alpha threshold), not
experimental solubility labels.  Therefore the fitted regions must not be
reported as an OPA solubility sphere.
"""

from __future__ import annotations

import html
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save(fig: plt.Figure, stem: Path) -> list[Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths = [stem.with_suffix(suffix) for suffix in (".png", ".svg", ".pdf")]
    fig.savefig(paths[0], dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(paths[1], bbox_inches="tight", facecolor="white")
    fig.savefig(paths[2], bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return paths


def _label_2d_figure(fig: plt.Figure, frame: pd.DataFrame, title: str) -> None:
    fig.set_size_inches(14.2, 5.8)
    fig.suptitle(title, fontsize=12, fontweight="bold")
    coords = (
        (frame["delta_p_MPa05"], frame["delta_h_MPa05"]),
        (frame["delta_h_MPa05"], 2.0 * frame["delta_d_MPa05"]),
        (frame["delta_p_MPa05"], 2.0 * frame["delta_d_MPa05"]),
    )
    axis_labels = (
        (r"$\delta_P$ (MPa$^{1/2}$)", r"$\delta_H$ (MPa$^{1/2}$)"),
        (r"$\delta_H$ (MPa$^{1/2}$)", r"$\delta_D$ (MPa$^{1/2}$)"),
        (r"$\delta_P$ (MPa$^{1/2}$)", r"$\delta_D$ (MPa$^{1/2}$)"),
    )
    for ax, (xs, ys), (xlabel, ylabel) in zip(fig.axes[:3], coords, axis_labels):
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, color="#D7DCE0", linewidth=0.55, alpha=0.65)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.0)
            spine.set_color("#202428")
        legend = ax.get_legend()
        if legend is not None:
            for text_item, replacement in zip(legend.get_texts(), ("α ≥ 0.80", "α < 0.80")):
                text_item.set_text(replacement)
        for index, (x, y) in enumerate(zip(xs, ys), start=1):
            ax.annotate(
                str(index), (x, y), xytext=(4, 4), textcoords="offset points",
                fontsize=6.0, fontweight="bold", ha="left", va="bottom",
                bbox={"boxstyle": "square,pad=0.08", "fc": "white", "ec": "none", "alpha": 0.78},
            )
    key_items = [f"{index} {str(name).replace('_', ' ')}" for index, name in enumerate(frame["system"], start=1)]
    key_lines = ["   |   ".join(key_items[:7]), "   |   ".join(key_items[7:])]
    fig.text(0.5, 0.018, "\n".join(key_lines), ha="center", va="bottom", fontsize=6.2)
    fig.tight_layout(rect=(0, 0.10, 1, 0.94))


def _label_3d_figure(fig: plt.Figure, frame: pd.DataFrame, title: str) -> None:
    fig.set_size_inches(9.2, 7.4)
    fig.suptitle(title, fontsize=12, fontweight="bold")
    ax = fig.axes[0]
    ax.set_xlabel(r"$\delta_H$ (MPa$^{1/2}$)")
    ax.set_ylabel(r"$\delta_D$ (MPa$^{1/2}$)")
    ax.set_zlabel(r"$\delta_P$ (MPa$^{1/2}$)")
    legend = ax.get_legend()
    if legend is not None:
        for text_item, replacement in zip(legend.get_texts(), ("α ≥ 0.80", "α < 0.80")):
            text_item.set_text(replacement)
    for index, row in enumerate(frame.itertuples(), start=1):
        ax.text(
            row.delta_h_MPa05, 2.0 * row.delta_d_MPa05, row.delta_p_MPa05,
            str(index), fontsize=6.0, fontweight="bold",
        )
    key_items = [f"{index} {str(name).replace('_', ' ')}" for index, name in enumerate(frame["system"], start=1)]
    fig.text(0.5, 0.012, "   |   ".join(key_items[:7]) + "\n" + "   |   ".join(key_items[7:]),
             ha="center", va="bottom", fontsize=6.0)
    fig.subplots_adjust(bottom=0.10)


def _report_html(
    primary: dict[str, float],
    fit_summary: pd.DataFrame,
    classifications: pd.DataFrame,
) -> str:
    single = fit_summary.loc[(fit_summary["model"] == "single") & (fit_summary["sphere_id"] == 1)].iloc[0]
    double = fit_summary.loc[fit_summary["model"] == "double"].iloc[0]
    table = classifications.round(3).to_html(index=False, border=0, classes="metrics")
    fits = fit_summary.round(4).to_html(index=False, border=0, classes="metrics")
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Group 2 HSPiPy analysis</title>
<style>body{{font-family:Arial,'Microsoft YaHei',sans-serif;max-width:1280px;margin:auto;padding:32px;background:#eef1f4;color:#202428;line-height:1.65}}
main{{background:white;border:1px solid #202428;padding:32px 42px}}h1,h2{{color:#24557A}}.notice{{border:1px solid #D17C2F;background:#FFF8F0;padding:12px 16px}}
figure{{border:1.5px solid #202428;padding:10px;margin:22px 0}}img{{width:100%;height:auto}}table.metrics{{width:100%;border-collapse:collapse;font-size:11px}}
.metrics th,.metrics td{{border:1px solid #687078;padding:5px;text-align:right}}.metrics th:first-child,.metrics td:first-child{{text-align:left}}</style></head>
<body><main><p><a href="group2.html">← 返回Group 2主报告</a></p><h1>HSPiPy：纯溶剂的MD定义高移动性区域</h1>
<div class="notice">Score=1仅表示本次10 ps轨迹中TAMSD指数α≥{primary['threshold']:.2f}；它不是实验溶解度标签。因此以下球体不是OPA的实验Hansen溶解度球。</div>
<h2>直接结果</h2><p>单球中心为({single.D:.2f}, {single.P:.2f}, {single.H:.2f}) MPa¹ᐟ²，R={single.R:.2f}，准确率={single.accuracy:.3f}，DATAFIT={single.datafit:.3f}。中心含负分量且半径异常大，说明单球模型不适用于该标签。</p>
<p>双球训练准确率={double.accuracy:.3f}，DATAFIT={double.datafit:.3f}，仍有{int(double.wrong_in)}个假阳性。14个点拟合8个球参数存在明显过拟合风险，双球只作描述。</p>
<figure><img src="../figures/group2_hspipy_single_sphere_2d.png"><figcaption>HSPiPy单球三种二维投影；所有纯溶剂均标注。</figcaption></figure>
<figure><img src="../figures/group2_hspipy_double_sphere_2d.png"><figcaption>HSPiPy双球三种二维投影。</figcaption></figure>
<figure><img src="../figures/group2_hspipy_double_sphere_3d.png"><figcaption>HSPiPy双球三维Hansen空间。</figcaption></figure>
<h2>拟合参数</h2>{fits}<h2>逐溶剂分类与RED</h2>{table}
<h2>结论边界</h2><p>该分析检验的是“高α条件能否在Hansen空间形成紧凑区域”。当前答案是否定的。若要拟合真正的OPA HSP，应提供每种纯溶剂中OPA的实验溶解/不溶或定量溶解度标签。</p>
</main></body></html>"""


def run_hspipy_pure_analysis(
    frame: pd.DataFrame,
    output_root: Path,
    alpha_threshold: float = 0.80,
) -> tuple[list[Path], dict[str, float]]:
    """Fit HSPiPy single/double spheres to an explicit MD mobility label."""
    from hspipy import HSP, HSPEstimator

    output_root = Path(output_root)
    required = {
        "system", "group", "alpha", "delta_d_MPa05", "delta_p_MPa05", "delta_h_MPa05"
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Group 2 HSPiPy analysis missing columns: {sorted(missing)}")
    data = frame.loc[frame["group"] == "group2"].copy().reset_index(drop=True)
    if len(data) != 14:
        raise ValueError(f"expected 14 pure solvents, found {len(data)}")
    data["Score"] = (data["alpha"] >= alpha_threshold).astype(int)
    input_table = pd.DataFrame({
        "Solvent": data["system"],
        "D": data["delta_d_MPa05"],
        "P": data["delta_p_MPa05"],
        "H": data["delta_h_MPa05"],
        "Score": data["Score"],
    })
    tables = output_root / "tables"
    figures = output_root / "figures"
    reports = output_root / "reports"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    input_path = tables / "group2_hspipy_input.csv"
    input_table.to_csv(input_path, index=False, encoding="utf-8-sig")

    X = input_table[["D", "P", "H"]].to_numpy()
    y = input_table["Score"].to_numpy()
    outputs: list[Path] = [input_path]
    model_records: list[dict[str, float | int | str]] = []
    predictions = data[["system", "alpha", "Score"]].rename(columns={"Score": "observed_high_mobility"})
    models: dict[str, HSP] = {}
    for model_name, n_spheres in (("single", 1), ("double", 2)):
        model = HSP(n_spheres=n_spheres).read(str(input_path))
        result = model.get(inside_limit=1, n_spheres=n_spheres)
        models[model_name] = model
        predictions[f"{model_name}_predicted_inside"] = model.predict(X)
        predictions[f"{model_name}_RED"] = model.transform(X).ravel()
        for sphere_id, sphere in enumerate(np.atleast_2d(model.hsp_), start=1):
            model_records.append({
                "model": model_name,
                "n_spheres": n_spheres,
                "sphere_id": sphere_id,
                "D": sphere[0], "P": sphere[1], "H": sphere[2], "R": sphere[3],
                "accuracy": result.accuracy, "datafit": result.datafit,
                "wrong_in": result.n_wrong_in, "wrong_out": result.n_wrong_out,
                "n_high_mobility": result.n_solvents_in,
                "n_low_mobility": result.n_solvents_out,
                "alpha_threshold": alpha_threshold,
                "label_definition": f"alpha >= {alpha_threshold:.2f}",
                "physically_plausible_center_and_radius": bool(
                    sphere[0] >= 0 and sphere[1] >= 0 and sphere[2] >= 0 and 0 < sphere[3] <= 20
                ),
                "interpretation": "MD mobility region; not experimental solubility sphere",
            })

    fit_summary = pd.DataFrame(model_records)
    fit_path = tables / "group2_hspipy_fit_summary.csv"
    class_path = tables / "group2_hspipy_classification.csv"
    fit_summary.to_csv(fit_path, index=False, encoding="utf-8-sig")
    predictions.to_csv(class_path, index=False, encoding="utf-8-sig")
    outputs.extend([fit_path, class_path])

    sensitivity_records = []
    for threshold in (0.70, 0.75, 0.80, 0.85, 0.90):
        labels = (data["alpha"] >= threshold).astype(int).to_numpy()
        estimator = HSPEstimator(method="classic", inside_limit=1, n_spheres=1).fit(X, labels)
        with np.errstate(divide="ignore", invalid="ignore"):
            estimator.score(X, labels)
        sphere = estimator.hsp_[0]
        sensitivity_records.append({
            "alpha_threshold": threshold, "n_high_mobility": int(labels.sum()),
            "D": sphere[0], "P": sphere[1], "H": sphere[2], "R": sphere[3],
            "accuracy": estimator.accuracy_, "datafit": estimator.datafit_,
            "wrong_in": estimator.n_wrong_in_, "wrong_out": estimator.n_wrong_out_,
            "degenerate_or_nonphysical": bool(
                sphere[0] < 0 or sphere[1] < 0 or sphere[2] < 0 or sphere[3] <= 1e-8 or sphere[3] > 20
            ),
        })
    sensitivity_path = tables / "group2_hspipy_threshold_sensitivity.csv"
    pd.DataFrame(sensitivity_records).to_csv(sensitivity_path, index=False, encoding="utf-8-sig")
    outputs.append(sensitivity_path)

    mpl.rcParams.update({"font.family": "sans-serif", "svg.fonttype": "none", "pdf.fonttype": 42})
    single_2d = models["single"].plot_2d(legend=True)
    _label_2d_figure(single_2d, data, "Group 2 | HSPiPy single-sphere fit to α ≥ 0.80")
    outputs.extend(_save(single_2d, figures / "group2_hspipy_single_sphere_2d"))
    double_2d = models["double"].plot_2d(legend=True)
    _label_2d_figure(double_2d, data, "Group 2 | HSPiPy double-sphere fit to α ≥ 0.80")
    outputs.extend(_save(double_2d, figures / "group2_hspipy_double_sphere_2d"))
    double_3d = models["double"].plot_3d(legend=True)
    _label_3d_figure(double_3d, data, "Group 2 | HSPiPy double-sphere fit to α ≥ 0.80")
    outputs.extend(_save(double_3d, figures / "group2_hspipy_double_sphere_3d"))

    primary = {
        "threshold": alpha_threshold,
        "n_high_mobility": int(data["Score"].sum()),
        "single_accuracy": float(models["single"].accuracy_),
        "single_datafit": float(models["single"].datafit_),
        "double_accuracy": float(models["double"].accuracy_),
        "double_datafit": float(models["double"].datafit_),
        "double_false_positive": int(models["double"].n_wrong_in_),
    }
    report_path = reports / "group2_hspipy_analysis.html"
    report_path.write_text(_report_html(primary, fit_summary, predictions), encoding="utf-8")
    outputs.append(report_path)
    return outputs, primary


def _mixture_report_html(
    group: str,
    primary: dict[str, float],
    fit_summary: pd.DataFrame,
    classifications: pd.DataFrame,
) -> str:
    single = fit_summary.loc[(fit_summary["model"] == "single") & (fit_summary["sphere_id"] == 1)].iloc[0]
    double = fit_summary.loc[fit_summary["model"] == "double"].iloc[0]
    table = classifications.round(3).to_html(index=False, border=0, classes="metrics")
    fits = fit_summary.round(4).to_html(index=False, border=0, classes="metrics")
    invalid_single = single.D < 0 or single.P < 0 or single.H < 0 or single.R > 20
    validity = (
        "中心含负分量或半径异常，不能解释为物理Hansen球。"
        if invalid_single else
        "参数位于通常范围内，但仍受极小样本限制。"
    )
    extra = (
        "Group 3的THF/toluene结构轨迹不匹配，但本分析只使用其动力学CSV中的α和名义组成，因此保留该点。"
        if group == "group3" else ""
    )
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{group} HSPiPy analysis</title>
<style>body{{font-family:Arial,'Microsoft YaHei',sans-serif;max-width:1280px;margin:auto;padding:32px;background:#eef1f4;color:#202428;line-height:1.65}}
main{{background:white;border:1px solid #202428;padding:32px 42px}}h1,h2{{color:#24557A}}.notice{{border:1px solid #D17C2F;background:#FFF8F0;padding:12px 16px}}
figure{{border:1.5px solid #202428;padding:10px;margin:22px 0}}img{{width:100%;height:auto}}table.metrics{{width:100%;border-collapse:collapse;font-size:11px}}
.metrics th,.metrics td{{border:1px solid #687078;padding:5px;text-align:right}}.metrics th:first-child,.metrics td:first-child{{text-align:left}}</style></head>
<body><main><p><a href="{group}.html">← 返回{group}主报告</a></p><h1>HSPiPy：{group}混合溶剂的MD定义高移动性区域</h1>
<div class="notice">混合HSP按名义体积分数线性计算；Score=1仅表示TAMSD指数α≥{primary['threshold']:.2f}。球体不是OPA实验溶解度球。</div>
<h2>直接结果</h2><p>共{int(primary['n_systems'])}个体系，其中{int(primary['n_high_mobility'])}个满足高移动性判据。单球中心为({single.D:.2f}, {single.P:.2f}, {single.H:.2f}) MPa¹ᐟ²，R={single.R:.2f}，准确率={single.accuracy:.3f}，DATAFIT={single.datafit:.3f}。{validity}</p>
<p>双球训练准确率={double.accuracy:.3f}，DATAFIT={double.datafit:.3f}，假阳性={int(double.wrong_in)}、假阴性={int(double.wrong_out)}。样本数远少于可靠球拟合通常所需的数据量，双球仅作描述。{extra}</p>
<figure><img src="../figures/{group}_hspipy_single_sphere_2d.png"><figcaption>HSPiPy单球二维投影；数字与图下体系表对应。</figcaption></figure>
<figure><img src="../figures/{group}_hspipy_double_sphere_2d.png"><figcaption>HSPiPy双球二维投影。</figcaption></figure>
<figure><img src="../figures/{group}_hspipy_double_sphere_3d.png"><figcaption>HSPiPy双球三维Hansen空间。</figcaption></figure>
<h2>拟合参数</h2>{fits}<h2>逐体系分类与RED</h2>{table}
<h2>解释边界</h2><p>该分析只检验高α条件能否在理想线性混合Hansen空间形成紧凑区域。体积分数线性混合忽略非理想混合、局部优先溶剂化和浓度依赖；RED只相对于MD移动性球。</p>
</main></body></html>"""


def run_hspipy_mixture_analysis(
    frame: pd.DataFrame,
    output_root: Path,
    group: str,
    alpha_threshold: float = 0.80,
) -> tuple[list[Path], dict[str, float]]:
    """Analyze Group 1 or 3 using volume-fraction mixed HSP descriptors."""
    from hspipy import HSP, HSPEstimator

    if group not in {"group1", "group3"}:
        raise ValueError(f"mixture HSPiPy analysis only supports group1/group3, got {group}")
    output_root = Path(output_root)
    data = frame.loc[frame["group"] == group].copy().reset_index(drop=True)
    required = {"system", "alpha", "cosolvent", "base_solvent", "cosolvent_fraction"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"{group} HSPiPy analysis missing columns: {sorted(missing)}")
    reference_path = Path(__file__).resolve().parents[1] / "reference_data" / "hansen_pure_solvents.csv"
    reference = pd.read_csv(reference_path).set_index("system")
    mixed_records = []
    for row in data.itertuples():
        phi = float(row.cosolvent_fraction)
        if not 0.0 <= phi <= 1.0:
            raise ValueError(f"{group}/{row.system}: invalid cosolvent fraction {phi}")
        cosolvent = reference.loc[str(row.cosolvent)]
        base = reference.loc[str(row.base_solvent)]
        mixed_records.append({
            "system": row.system,
            "cosolvent": row.cosolvent,
            "base_solvent": row.base_solvent,
            "cosolvent_volume_fraction": phi,
            "base_volume_fraction": 1.0 - phi,
            "D": phi * cosolvent.delta_d_MPa05 + (1.0 - phi) * base.delta_d_MPa05,
            "P": phi * cosolvent.delta_p_MPa05 + (1.0 - phi) * base.delta_p_MPa05,
            "H": phi * cosolvent.delta_h_MPa05 + (1.0 - phi) * base.delta_h_MPa05,
            "alpha": row.alpha,
            "Score": int(row.alpha >= alpha_threshold),
        })
    mixed = pd.DataFrame(mixed_records)
    # Restore the original report order so number keys remain stable.
    data = data[["system"]].merge(mixed, on="system", how="left", validate="one_to_one")
    tables = output_root / "tables"
    figures = output_root / "figures"
    reports = output_root / "reports"
    for directory in (tables, figures, reports):
        directory.mkdir(parents=True, exist_ok=True)
    mixture_path = tables / f"{group}_hspipy_mixture_hsp.csv"
    data.to_csv(mixture_path, index=False, encoding="utf-8-sig")
    input_table = data[["system", "D", "P", "H", "Score"]].rename(columns={"system": "Solvent"})
    input_path = tables / f"{group}_hspipy_input.csv"
    input_table.to_csv(input_path, index=False, encoding="utf-8-sig")
    outputs: list[Path] = [mixture_path, input_path]

    X = input_table[["D", "P", "H"]].to_numpy()
    y = input_table["Score"].to_numpy()
    models: dict[str, HSP] = {}
    records: list[dict[str, float | int | str]] = []
    classifications = data[["system", "cosolvent_volume_fraction", "D", "P", "H", "alpha", "Score"]].rename(
        columns={"Score": "observed_high_mobility"}
    )
    for model_name, n_spheres in (("single", 1), ("double", 2)):
        model = HSP(n_spheres=n_spheres).read(str(input_path))
        result = model.get(inside_limit=1, n_spheres=n_spheres)
        models[model_name] = model
        classifications[f"{model_name}_predicted_inside"] = model.predict(X)
        classifications[f"{model_name}_RED"] = model.transform(X).ravel()
        for sphere_id, sphere in enumerate(np.atleast_2d(model.hsp_), start=1):
            records.append({
                "group": group, "model": model_name, "n_spheres": n_spheres,
                "sphere_id": sphere_id, "D": sphere[0], "P": sphere[1],
                "H": sphere[2], "R": sphere[3], "accuracy": result.accuracy,
                "datafit": result.datafit, "wrong_in": result.n_wrong_in,
                "wrong_out": result.n_wrong_out, "n_high_mobility": result.n_solvents_in,
                "n_low_mobility": result.n_solvents_out,
                "alpha_threshold": alpha_threshold,
                "mixing_rule": "volume-fraction linear HSP",
                "physically_plausible_center_and_radius": bool(
                    sphere[0] >= 0 and sphere[1] >= 0 and sphere[2] >= 0 and 0 < sphere[3] <= 20
                ),
                "interpretation": "MD mobility region; not experimental solubility sphere",
            })
    fit_summary = pd.DataFrame(records)
    fit_path = tables / f"{group}_hspipy_fit_summary.csv"
    classification_path = tables / f"{group}_hspipy_classification.csv"
    fit_summary.to_csv(fit_path, index=False, encoding="utf-8-sig")
    classifications.to_csv(classification_path, index=False, encoding="utf-8-sig")
    outputs.extend([fit_path, classification_path])

    sensitivity_records = []
    for threshold in (0.70, 0.75, 0.80, 0.85, 0.90):
        labels = (data["alpha"] >= threshold).astype(int).to_numpy()
        estimator = HSPEstimator(method="classic", inside_limit=1, n_spheres=1).fit(X, labels)
        with np.errstate(divide="ignore", invalid="ignore"):
            estimator.score(X, labels)
        sphere = estimator.hsp_[0]
        sensitivity_records.append({
            "alpha_threshold": threshold, "n_high_mobility": int(labels.sum()),
            "D": sphere[0], "P": sphere[1], "H": sphere[2], "R": sphere[3],
            "accuracy": estimator.accuracy_, "datafit": estimator.datafit_,
            "wrong_in": estimator.n_wrong_in_, "wrong_out": estimator.n_wrong_out_,
            "degenerate_or_nonphysical": bool(
                sphere[0] < 0 or sphere[1] < 0 or sphere[2] < 0 or sphere[3] <= 1e-8 or sphere[3] > 20
            ),
        })
    sensitivity_path = tables / f"{group}_hspipy_threshold_sensitivity.csv"
    pd.DataFrame(sensitivity_records).to_csv(sensitivity_path, index=False, encoding="utf-8-sig")
    outputs.append(sensitivity_path)

    plot_frame = data.rename(columns={"D": "delta_d_MPa05", "P": "delta_p_MPa05", "H": "delta_h_MPa05"})
    single_2d = models["single"].plot_2d(legend=True)
    _label_2d_figure(single_2d, plot_frame, f"{group.upper()} | HSPiPy single-sphere fit to α ≥ 0.80")
    outputs.extend(_save(single_2d, figures / f"{group}_hspipy_single_sphere_2d"))
    double_2d = models["double"].plot_2d(legend=True)
    _label_2d_figure(double_2d, plot_frame, f"{group.upper()} | HSPiPy double-sphere fit to α ≥ 0.80")
    outputs.extend(_save(double_2d, figures / f"{group}_hspipy_double_sphere_2d"))
    double_3d = models["double"].plot_3d(legend=True)
    _label_3d_figure(double_3d, plot_frame, f"{group.upper()} | HSPiPy double-sphere fit to α ≥ 0.80")
    outputs.extend(_save(double_3d, figures / f"{group}_hspipy_double_sphere_3d"))

    primary = {
        "threshold": alpha_threshold, "n_systems": len(data),
        "n_high_mobility": int(data["Score"].sum()),
        "single_accuracy": float(models["single"].accuracy_),
        "single_datafit": float(models["single"].datafit_),
        "double_accuracy": float(models["double"].accuracy_),
        "double_datafit": float(models["double"].datafit_),
        "double_false_positive": int(models["double"].n_wrong_in_),
        "double_false_negative": int(models["double"].n_wrong_out_),
    }
    report_path = reports / f"{group}_hspipy_analysis.html"
    report_path.write_text(
        _mixture_report_html(group, primary, fit_summary, classifications), encoding="utf-8"
    )
    outputs.append(report_path)
    return outputs, primary
