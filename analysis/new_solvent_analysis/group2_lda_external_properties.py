"""External-property LDA/PCA and linear association analysis for Group 2."""

from __future__ import annotations

import html
import itertools
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from thermo import Chemical

from .publication_reporting import display_label


ROOT = Path(__file__).resolve().parents[2]
RESULT2 = ROOT / "results" / "result2"
OUT = RESULT2 / "group2_lda_external_properties"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
REPORTS = OUT / "reports"
MD_TABLE = RESULT2 / "tables" / "group2_metrics.csv"
HSP_TABLE = RESULT2 / "tables" / "group2_hansen_parameters.csv"

THERMO_URL = "https://thermo.readthedocs.io/thermo.chemical.html"
ZEON_URL = "https://www.zeon.co.jp/en/business/enterprise/special/solvent-cpme/"
HANSEN_URL = "https://doi.org/10.1201/9781420006834"

SYSTEM_TO_CHEMICAL = {
    "acetone": "acetone",
    "acetonitrile": "acetonitrile",
    "CPME": "cyclopentyl methyl ether",
    "cyclohexane": "cyclohexane",
    "DMC": "dimethyl carbonate",
    "ethanol": "ethanol",
    "ethyl_acetate": "ethyl acetate",
    "isopropanol": "isopropanol",
    "methanol": "methanol",
    "n-heptane": "n-heptane",
    "p-xylene": "p-xylene",
    "propylene_carbonate": "propylene carbonate",
    "thf": "tetrahydrofuran",
    "toluene": "toluene",
}

FEATURES = [
    "log10_viscosity_mPa_s",
    "dielectric_constant",
    "surface_tension_mN_m",
    "molar_volume_cm3_mol",
    "delta_d_MPa05",
    "delta_p_MPa05",
    "delta_h_MPa05",
]

FEATURE_LABELS = {
    "log10_viscosity_mPa_s": "log10 viscosity",
    "dielectric_constant": "Dielectric constant",
    "surface_tension_mN_m": "Surface tension",
    "molar_volume_cm3_mol": "Molar volume",
    "delta_d_MPa05": "HSP δD",
    "delta_p_MPa05": "HSP δP",
    "delta_h_MPa05": "HSP δH",
}

RESPONSES = [
    "alpha",
    "log10_tamsd_prefactor",
    "max_disp_A",
    "mean_force_eV_A",
    "head_coord_number_4A",
]

RESPONSE_LABELS = {
    "alpha": "TAMSD α",
    "log10_tamsd_prefactor": "log10 Kα",
    "max_disp_A": "Maximum displacement",
    "mean_force_eV_A": "Mean force",
    "head_coord_number_4A": "Head CN",
}

COLORS = {
    "acetone": "#D55E00",
    "n-heptane": "#0072B2",
    "other": "#9AA0A6",
    "positive": "#B24745",
    "negative": "#356C91",
}


def style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 9.2,
            "axes.linewidth": 1.0,
            "axes.edgecolor": "#202428",
            "axes.spines.top": True,
            "axes.spines.right": True,
            "xtick.major.width": 0.9,
            "ytick.major.width": 0.9,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def boxed(ax: plt.Axes) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
        spine.set_color("#202428")
    ax.grid(True, color="#D7DCE0", linewidth=0.55, alpha=0.65, zorder=0)
    ax.set_axisbelow(True)


def panel(ax: plt.Axes, letter: str) -> None:
    ax.text(
        -0.12,
        1.04,
        letter,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="bottom",
    )


def save_figure(fig: plt.Figure, stem: Path) -> list[Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    outputs = [stem.with_suffix(ext) for ext in (".png", ".svg", ".pdf")]
    fig.savefig(outputs[0], dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(outputs[1], bbox_inches="tight", facecolor="white")
    fig.savefig(outputs[2], bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return outputs


def build_external_table() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for system, chemical_name in SYSTEM_TO_CHEMICAL.items():
        chemical = Chemical(chemical_name, T=298.15, P=101325.0)
        row = {
            "system": system,
            "chemical_name": chemical_name,
            "cas_number": chemical.CAS,
            "temperature_K": 298.15,
            "viscosity_mPa_s": chemical.mu * 1000.0 if chemical.mu else np.nan,
            "dielectric_constant": chemical.permittivity,
            "surface_tension_mN_m": chemical.sigma * 1000.0 if chemical.sigma else np.nan,
            "density_g_cm3": chemical.rho / 1000.0 if chemical.rho else np.nan,
            "molar_mass_g_mol": chemical.MW,
            "viscosity_method": chemical.ViscosityLiquid.method,
            "dielectric_method": chemical.Permittivity.method,
            "surface_tension_method": chemical.SurfaceTension.method,
            "density_method": chemical.VolumeLiquid.method,
            "property_source": "thermo 0.6.1 property correlations/database",
            "property_source_url": THERMO_URL,
            "temperature_note": "all listed properties evaluated at 298.15 K",
        }
        if system == "CPME":
            row.update(
                {
                    "viscosity_mPa_s": 0.569,
                    "dielectric_constant": 4.76,
                    "surface_tension_mN_m": 25.17,
                    "density_g_cm3": 0.86,
                    "viscosity_method": "Zeon technical table (25 °C)",
                    "dielectric_method": "Zeon technical table (25 °C)",
                    "surface_tension_method": "Zeon technical table (20 °C)",
                    "density_method": "Zeon technical table (25 °C)",
                    "property_source": "Zeon CPME technical data",
                    "property_source_url": ZEON_URL,
                    "temperature_note": (
                        "viscosity, dielectric constant, density at 25 °C; "
                        "surface tension at 20 °C"
                    ),
                }
            )
        elif system == "propylene_carbonate":
            row.update(
                {
                    "viscosity_mPa_s": 2.530,
                    "dielectric_constant": 64.4,
                    "density_g_cm3": 1.198,
                    "viscosity_method": "IUPAC recommended value (25 °C)",
                    "dielectric_method": "IUPAC recommended value (25 °C)",
                    "density_method": "IUPAC recommended value (25 °C)",
                    "property_source": (
                        "IUPAC recommended propylene carbonate table for viscosity, "
                        "dielectric constant and density; thermo 0.6.1 for surface tension"
                    ),
                    "property_source_url": (
                        "https://media.iupac.org/publications/pac/1971/pdf/2701x0273.pdf"
                    ),
                    "temperature_note": (
                        "Viscosity, dielectric constant and density at 25 °C; surface "
                        "tension evaluated at 298.15 K with thermo 0.6.1"
                    ),
                }
            )
        row["molar_volume_cm3_mol"] = (
            float(row["molar_mass_g_mol"]) / float(row["density_g_cm3"])
        )
        rows.append(row)
    external = pd.DataFrame(rows)
    if external[["viscosity_mPa_s", "dielectric_constant", "surface_tension_mN_m"]].isna().any().any():
        raise ValueError("external property table contains missing primary features")
    external["log10_viscosity_mPa_s"] = np.log10(external["viscosity_mPa_s"])
    return external


def merge_data() -> pd.DataFrame:
    external = build_external_table()
    hsp = pd.read_csv(HSP_TABLE)
    hsp = hsp[
        [
            "system",
            "delta_d_MPa05",
            "delta_p_MPa05",
            "delta_h_MPa05",
            "source_key",
            "source_url",
        ]
    ].rename(columns={"source_key": "hsp_source", "source_url": "hsp_source_url"})
    md = pd.read_csv(MD_TABLE)
    md["log10_tamsd_prefactor"] = np.log10(md["alpha_prefactor_A2_ps_alpha"])
    keep = [
        "system",
        "alpha",
        "alpha_prefactor_A2_ps_alpha",
        "log10_tamsd_prefactor",
        "max_disp_A",
        "mean_force_eV_A",
        "head_coord_number_4A",
        "alpha_fit_r2",
    ]
    merged = external.merge(hsp, on="system", validate="one_to_one").merge(
        md[keep], on="system", validate="one_to_one"
    )
    merged["candidate_class"] = merged["system"].isin(["acetone", "n-heptane"]).astype(int)
    if len(merged) != 14:
        raise ValueError(f"expected 14 Group 2 solvents, found {len(merged)}")
    return merged


def bh_adjust(pvalues: np.ndarray) -> np.ndarray:
    pvalues = np.asarray(pvalues, dtype=float)
    order = np.argsort(pvalues)
    ranked = pvalues[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.clip(adjusted, 0.0, 1.0)
    return result


def correlation_table(data: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for feature in FEATURES:
        for response in RESPONSES:
            pearson_r, pearson_p = pearsonr(data[feature], data[response])
            spearman_r, spearman_p = spearmanr(data[feature], data[response])
            rows.append(
                {
                    "external_feature": feature,
                    "md_response": response,
                    "pearson_r": pearson_r,
                    "pearson_p": pearson_p,
                    "spearman_rho": spearman_r,
                    "spearman_p": spearman_p,
                }
            )
    result = pd.DataFrame(rows)
    result["pearson_q_bh"] = bh_adjust(result["pearson_p"].to_numpy())
    result["spearman_q_bh"] = bh_adjust(result["spearman_p"].to_numpy())
    return result


def model_analysis(data: pd.DataFrame) -> dict[str, object]:
    x = data[FEATURES].to_numpy(dtype=float)
    y = data["candidate_class"].to_numpy(dtype=int)
    scaler = StandardScaler()
    z = scaler.fit_transform(x)
    pca = PCA(n_components=2)
    scores = pca.fit_transform(z)
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

    lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    lda.fit(z, y)
    lda_score = lda.decision_function(z)
    coefficients = lda.coef_[0].copy()
    if np.mean(lda_score[y == 1]) < np.mean(lda_score[y == 0]):
        lda_score *= -1.0
        coefficients *= -1.0

    candidate_pair = tuple(int(value) for value in np.flatnonzero(y == 1))
    all_centroid_distances: list[float] = []
    for pair in itertools.combinations(range(len(data)), 2):
        labels = np.zeros(len(data), dtype=int)
        labels[list(pair)] = 1
        all_centroid_distances.append(
            float(np.linalg.norm(z[labels == 1].mean(axis=0) - z[labels == 0].mean(axis=0)))
        )
    candidate_centroid_distance = float(
        np.linalg.norm(z[y == 1].mean(axis=0) - z[y == 0].mean(axis=0))
    )
    centroid_percentile = 100.0 * np.mean(
        np.asarray(all_centroid_distances) <= candidate_centroid_distance
    )

    score_table = data[
        ["system", "candidate_class", *FEATURES, *RESPONSES]
    ].copy()
    score_table["PC1"] = scores[:, 0]
    score_table["PC2"] = scores[:, 1]
    score_table["lda_score"] = lda_score
    coefficient_table = pd.DataFrame(
        {
            "feature": FEATURES,
            "feature_label": [FEATURE_LABELS[value] for value in FEATURES],
            "standardized_lda_coefficient": coefficients,
            "absolute_coefficient": np.abs(coefficients),
        }
    ).sort_values("absolute_coefficient", ascending=False)
    loading_table = pd.DataFrame(
        {
            "feature": FEATURES,
            "feature_label": [FEATURE_LABELS[value] for value in FEATURES],
            "PC1_loading": loadings[:, 0],
            "PC2_loading": loadings[:, 1],
        }
    )
    robustness = {
        "candidate_pair_indices": list(candidate_pair),
        "standardized_centroid_distance": candidate_centroid_distance,
        "centroid_distance_percentile_among_all_91_pairs": centroid_percentile,
        "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "warning": (
            "Only two user-designated positive solvents; LDA is descriptive and "
            "must not be interpreted as a validated classifier. LOOCV was intentionally "
            "not reported because each positive-class test fold would leave only one "
            "positive sample for training."
        ),
    }
    return {
        "z": z,
        "scores": scores,
        "loadings": loadings,
        "lda_scores": lda_score,
        "coefficients": coefficients,
        "score_table": score_table,
        "coefficient_table": coefficient_table,
        "loading_table": loading_table,
        "robustness": robustness,
    }


def overview_figure(data: pd.DataFrame, model: dict[str, object]) -> list[Path]:
    style()
    scores = np.asarray(model["scores"])
    loadings = np.asarray(model["loadings"])
    lda_scores = np.asarray(model["lda_scores"])
    coefficients = np.asarray(model["coefficients"])
    z = np.asarray(model["z"])
    pca_ratio = model["robustness"]["pca_explained_variance_ratio"]
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 9.0), constrained_layout=True)

    ax = axes[0, 0]
    for index, row in data.reset_index(drop=True).iterrows():
        color = COLORS.get(row.system, COLORS["other"])
        size = 92 if row.candidate_class else 48
        ax.scatter(
            scores[index, 0],
            scores[index, 1],
            s=size,
            color=color,
            edgecolor="#202428",
            linewidth=0.8,
            zorder=3,
        )
        if row.candidate_class:
            ax.annotate(
                display_label(row.system),
                (scores[index, 0], scores[index, 1]),
                xytext=(8, 8),
                textcoords="offset points",
                fontsize=9.2,
                fontweight="bold",
            )
    score_span = max(np.ptp(scores[:, 0]), np.ptp(scores[:, 1]))
    arrow_scale = 0.40 * score_span
    for i, feature in enumerate(FEATURES):
        dx, dy = loadings[i] * arrow_scale
        ax.arrow(0, 0, dx, dy, width=0.006, head_width=0.08, color="#525B62", alpha=0.75)
        ax.text(dx * 1.08, dy * 1.08, FEATURE_LABELS[feature], fontsize=7.2, ha="center")
    ax.axhline(0, color="#A8AFB5", lw=0.7)
    ax.axvline(0, color="#A8AFB5", lw=0.7)
    ax.set_xlabel(f"PC1 ({100*pca_ratio[0]:.1f}% variance)", fontsize=10)
    ax.set_ylabel(f"PC2 ({100*pca_ratio[1]:.1f}% variance)", fontsize=10)
    ax.set_title("External-property PCA", fontsize=11, fontweight="bold")

    ax = axes[0, 1]
    ordered = np.argsort(lda_scores)
    y_positions = np.arange(len(data))
    for y_pos, index in enumerate(ordered):
        row = data.iloc[index]
        color = COLORS.get(row.system, COLORS["other"])
        size = 92 if row.candidate_class else 48
        ax.scatter(
            lda_scores[index], y_pos, s=size, color=color, edgecolor="#202428", linewidth=0.8, zorder=3
        )
        if row.candidate_class:
            ax.annotate(
                display_label(row.system),
                (lda_scores[index], y_pos),
                xytext=(7, 0),
                textcoords="offset points",
                va="center",
                fontsize=9.2,
                fontweight="bold",
            )
    ax.axvline(0, color="#202428", ls="--", lw=0.9)
    ax.set_yticks([])
    ax.set_xlabel("Shrinkage-LDA decision score", fontsize=10)
    ax.set_title("User-defined candidates vs comparison solvents", fontsize=11, fontweight="bold")

    ax = axes[1, 0]
    order = np.argsort(np.abs(coefficients))
    labels = [FEATURE_LABELS[FEATURES[i]] for i in order]
    values = coefficients[order]
    colors = [COLORS["positive"] if value > 0 else COLORS["negative"] for value in values]
    ax.barh(np.arange(len(values)), values, color=colors, edgecolor="#202428", linewidth=0.6)
    ax.axvline(0, color="#202428", lw=0.8)
    ax.set_yticks(np.arange(len(values)), labels)
    ax.set_xlabel("Standardized LDA coefficient", fontsize=10)
    ax.set_title("Direction associated with the candidate class", fontsize=11, fontweight="bold")

    ax = axes[1, 1]
    feature_x = np.arange(len(FEATURES))
    for system in ("acetone", "n-heptane"):
        index = int(data.index[data.system == system][0])
        ax.plot(
            feature_x,
            z[index],
            marker="o",
            ms=6.5,
            lw=1.8,
            color=COLORS[system],
            label=display_label(system),
        )
    ax.axhline(0, color="#202428", ls="--", lw=0.8)
    ax.set_xticks(feature_x, [FEATURE_LABELS[value] for value in FEATURES], rotation=30, ha="right")
    ax.set_ylabel("Standardized property value (z-score)", fontsize=10)
    ax.set_title("Candidate property profiles", fontsize=11, fontweight="bold")
    ax.legend(frameon=True, edgecolor="#202428", fontsize=8.5)

    for letter, ax in zip("abcd", axes.flat):
        boxed(ax)
        panel(ax, letter)
    fig.suptitle(
        "Group 2 | External solvent properties and descriptive LDA",
        fontsize=14,
        fontweight="bold",
    )
    return save_figure(fig, FIGURES / "group2_external_property_pca_lda")


def correlation_figure(correlations: pd.DataFrame) -> list[Path]:
    style()
    matrix = correlations.pivot(
        index="external_feature", columns="md_response", values="spearman_rho"
    ).loc[FEATURES, RESPONSES]
    fig, ax = plt.subplots(figsize=(9.2, 6.7), constrained_layout=True)
    image = ax.imshow(matrix.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(RESPONSES)), [RESPONSE_LABELS[value] for value in RESPONSES], rotation=25, ha="right")
    ax.set_yticks(np.arange(len(FEATURES)), [FEATURE_LABELS[value] for value in FEATURES])
    for i in range(len(FEATURES)):
        for j in range(len(RESPONSES)):
            value = matrix.iloc[i, j]
            color = "white" if abs(value) >= 0.58 else "#172027"
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", color=color, fontsize=9)
    cbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.03)
    cbar.set_label("Spearman ρ", fontsize=10)
    ax.set_title("External solvent properties vs MD responses (n = 14)", fontsize=13, fontweight="bold")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
        spine.set_color("#202428")
    return save_figure(fig, FIGURES / "group2_external_md_spearman_heatmap")


def linear_relationship_figure(data: pd.DataFrame) -> list[Path]:
    style()
    panels = [
        ("log10_viscosity_mPa_s", "alpha", "log10 viscosity (mPa s)", "TAMSD α"),
        ("log10_viscosity_mPa_s", "log10_tamsd_prefactor", "log10 viscosity (mPa s)", "log10 Kα"),
        ("surface_tension_mN_m", "log10_tamsd_prefactor", "Surface tension (mN m^-1)", "log10 Kα"),
        ("dielectric_constant", "mean_force_eV_A", "Dielectric constant", "Mean force (eV Å^-1)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.4), constrained_layout=True)
    for letter, ax, (xcol, ycol, xlabel, ylabel) in zip("abcd", axes.flat, panels):
        x = data[xcol].to_numpy(float)
        y = data[ycol].to_numpy(float)
        slope, intercept = np.polyfit(x, y, 1)
        xline = np.linspace(x.min(), x.max(), 120)
        ax.plot(xline, slope * xline + intercept, color="#565F66", lw=1.2, zorder=1)
        r, p = pearsonr(x, y)
        for _, row in data.iterrows():
            color = COLORS.get(row.system, COLORS["other"])
            size = 88 if row.candidate_class else 46
            ax.scatter(row[xcol], row[ycol], s=size, color=color, edgecolor="#202428", linewidth=0.75, zorder=3)
            if row.candidate_class:
                ax.annotate(
                    display_label(row.system),
                    (row[xcol], row[ycol]),
                    xytext=(6, 6),
                    textcoords="offset points",
                    fontsize=8.5,
                    fontweight="bold",
                )
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(f"Pearson r = {r:.2f}, p = {p:.3f}", fontsize=10.5, fontweight="bold")
        boxed(ax)
        panel(ax, letter)
    fig.suptitle("Pre-specified linear relationships", fontsize=14, fontweight="bold")
    return save_figure(fig, FIGURES / "group2_prespecified_linear_relationships")


def markdown_table(frame: pd.DataFrame, index: bool = False) -> str:
    table = frame.reset_index() if index else frame.copy()
    columns = [str(value) for value in table.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in table.itertuples(index=False, name=None):
        values = [str(value).replace("|", "\\|") for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def report(data: pd.DataFrame, correlations: pd.DataFrame, model: dict[str, object]) -> list[Path]:
    robustness = model["robustness"]
    coefficients = model["coefficient_table"].copy()
    coefficients["standardized_lda_coefficient"] = coefficients["standardized_lda_coefficient"].round(3)
    coefficients = coefficients[["feature_label", "standardized_lda_coefficient"]]
    strongest = correlations.reindex(correlations.spearman_rho.abs().sort_values(ascending=False).index).head(8).copy()
    strongest["external_feature"] = strongest["external_feature"].map(FEATURE_LABELS)
    strongest["md_response"] = strongest["md_response"].map(RESPONSE_LABELS)
    for column in ("pearson_r", "pearson_p", "pearson_q_bh", "spearman_rho", "spearman_p", "spearman_q_bh"):
        strongest[column] = strongest[column].round(3)

    candidate = data.loc[data.candidate_class == 1, ["system", *FEATURES, *RESPONSES]].copy()
    candidate["system"] = candidate["system"].map(display_label)
    candidate = candidate.rename(columns={**FEATURE_LABELS, **RESPONSE_LABELS})
    candidate = candidate.round(3)

    positive_coefficients = coefficients.loc[coefficients.standardized_lda_coefficient > 0]
    negative_coefficients = coefficients.loc[coefficients.standardized_lda_coefficient < 0]
    top_positive = positive_coefficients.iloc[0] if not positive_coefficients.empty else None
    top_negative = negative_coefficients.iloc[0] if not negative_coefficients.empty else None
    viscosity_alpha = correlations.query(
        "external_feature == 'log10_viscosity_mPa_s' and md_response == 'alpha'"
    ).iloc[0]
    viscosity_k = correlations.query(
        "external_feature == 'log10_viscosity_mPa_s' and md_response == 'log10_tamsd_prefactor'"
    ).iloc[0]
    surface_k = correlations.query(
        "external_feature == 'surface_tension_mN_m' and md_response == 'log10_tamsd_prefactor'"
    ).iloc[0]

    summary = (
        "acetone和n-heptane在化学极性上并不相似，因此不存在单一的“低极性”解释。"
        f"两者更明显的共同点是较低的黏度和较低的表面张力。"
        f"在14种溶剂中，log黏度与TAMSD α的Spearman ρ={viscosity_alpha.spearman_rho:.3f}，"
        f"与log Kα的ρ={viscosity_k.spearman_rho:.3f}；"
        f"表面张力与log Kα的Spearman ρ={surface_k.spearman_rho:.3f}，但Pearson "
        f"r={surface_k.pearson_r:.3f}，说明该关系对排序方法和个别点敏感。"
        "这些结果用于提出“低摩擦/低界面内聚有利于短时移动”的假设，而不是证明SAM生长质量。"
    )
    lda_note = (
        f"acetone+n-heptane相对于其余溶剂的标准化物性质心距离为"
        f"{robustness['standardized_centroid_distance']:.3f}，"
        f"处于{robustness['centroid_distance_percentile_among_all_91_pairs']:.1f}百分位。"
        "由于正类只有2个样本，留一验证会使训练集仅剩1个正类，因此不报告LOOCV准确率。"
    )
    coefficient_sentence = ""
    if top_positive is not None and top_negative is not None:
        coefficient_sentence = (
            f"LDA中最强正向系数为{top_positive.feature_label}"
            f"（{top_positive.standardized_lda_coefficient:.3f}），最强负向系数为"
            f"{top_negative.feature_label}（{top_negative.standardized_lda_coefficient:.3f}）。"
        )

    markdown = f"""# Group 2纯溶剂：外部物性LDA与线性关联分析

**日期：** 2026-08-26  
**样本：** 14种纯溶剂  
**候选类：** acetone、n-heptane（由研究者预先指定）  
**定位：** 探索性机制分析，不是经过外部标签验证的分类模型

## 执行摘要

{summary}

## 方法

- 外部性质：log10黏度、介电常数、表面张力、摩尔体积、HSP δD/δP/δH。
- 所有变量在LDA和PCA前进行z-score标准化。
- LDA使用自动收缩协方差，减轻14个样本、7个特征下的协方差不稳定。
- 连续关系同时报告Pearson和Spearman，并对35项比较进行Benjamini–Hochberg校正。
- 稳健性检查枚举全部91种“两种溶剂作为正类”的组合，比较标准化物性质心距离；因正类仅2个，不报告不可靠的LOOCV准确率。

## 主要判断

1. **共同点不是极性。** acetone具有明显极性，n-heptane近乎非极性；二者在δP、δH和介电常数上差异很大。
2. **共同点更接近低黏度和低表面张力。** 这两项可能降低短时间尺度上的摩擦与内聚阻力，但不能单独解释全部TAMSD差异。
3. **LDA只是描述性对照。** {coefficient_sentence} {lda_note}
4. **“好”必须由独立结果定义。** 当前标签来自研究者指定；若没有SAM覆盖率、缺陷率、吸附自由能或实验成膜质量，LDA不能证明二者为最佳溶剂。

![PCA与LDA概览](../figures/group2_external_property_pca_lda.png)

![外部性质与MD响应相关矩阵](../figures/group2_external_md_spearman_heatmap.png)

![预先指定的线性关系](../figures/group2_prespecified_linear_relationships.png)

## acetone与n-heptane数据

{markdown_table(candidate)}

## LDA标准化系数

正系数表示更偏向研究者指定的候选类，负系数表示更偏向其余12种溶剂；系数不等于因果效应。

{markdown_table(coefficients)}

## 绝对Spearman相关最高的8组

{markdown_table(strongest[['external_feature','md_response','pearson_r','pearson_q_bh','spearman_rho','spearman_q_bh']])}

## 结论边界

- 正类只有2种溶剂，不能用LDA分类准确率证明普遍规律。
- 外部物性来自数据库相关式/查表；CPME表面张力为20 °C，其余主要按298.15 K求值。
- HSP与介电常数、黏度等存在相关结构，LDA系数可能因共线性改变方向。
- TAMSD、最大位移和Kα来自同一条10 ps轨迹，不是三个独立实验终点。
- 本报告支持“低黏度与低表面张力可能有助于短时运动”的工作假设，但不等同于溶解度或SAM成膜质量。

## 数据与来源

- [合并分析表](../tables/group2_external_md_merged.csv)
- [外部物性表](../tables/group2_external_properties_298K.csv)
- [完整相关结果](../tables/group2_external_md_correlations.csv)
- [LDA得分](../tables/group2_lda_scores.csv)
- [LDA系数](../tables/group2_lda_coefficients.csv)
- [thermo属性对象文档]({THERMO_URL})
- [Zeon CPME技术数据]({ZEON_URL})
- [Hansen参数来源]({HANSEN_URL})

## AI使用说明

本报告由AI辅助完成物性表构建、统计计算、图形生成和文字整理。外部数值的具体方法与来源已保留在CSV中；用于正式发表前仍应由研究者逐项核对原始实验来源和温度条件。
"""

    css = """
    :root{--ink:#172027;--muted:#58636c;--line:#cfd6dc;--blue:#24557a;--light:#eef5f8;--warn:#fff6df}
    *{box-sizing:border-box}body{margin:0;background:#edf0f2;color:var(--ink);font-family:Arial,"Microsoft YaHei",sans-serif;line-height:1.68}
    main{max-width:1180px;margin:24px auto;padding:38px 46px;background:#fff;border:1px solid var(--line)}
    h1{margin:0 0 8px;font-size:29px}h2{margin-top:34px;padding-bottom:6px;font-size:21px;border-bottom:2px solid var(--blue)}
    .meta{color:var(--muted)}.summary,.warning{padding:14px 18px;background:var(--light);border-left:5px solid var(--blue)}.warning{background:var(--warn);border-left-color:#c78b22}
    figure{margin:24px 0}figure img{display:block;width:100%;height:auto;border:1px solid #202428}figcaption{color:var(--muted);font-size:13px;margin-top:6px}
    table{border-collapse:collapse;width:100%;font-size:12.5px;margin:14px 0 24px}th,td{border:1px solid var(--line);padding:7px 8px;text-align:right}th{background:#edf2f5}th:first-child,td:first-child{text-align:left}
    a{color:var(--blue)}@media(max-width:760px){main{margin:0;padding:22px 16px;overflow-x:auto}}
    """
    html_report = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Group 2外部物性LDA分析</title><style>{css}</style></head><body><main>
    <h1>Group 2纯溶剂：外部物性LDA与线性关联分析</h1>
    <p class="meta">2026-08-26｜14种纯溶剂｜acetone与n-heptane为研究者指定候选类</p>
    <h2>执行摘要</h2><p class="summary">{html.escape(summary)}</p>
    <h2>方法</h2><ul><li>外部特征：log10黏度、介电常数、表面张力、摩尔体积、HSP δD/δP/δH。</li><li>所有特征z-score标准化；LDA采用自动收缩协方差。</li><li>连续关系同时计算Pearson与Spearman，并进行BH多重比较校正。</li><li>枚举全部91种两溶剂正类组合，检查指定候选对是否异常可分。</li></ul>
    <h2>主要判断</h2><ol><li><strong>共同点不是极性：</strong>acetone与n-heptane的介电常数、δP和δH明显不同。</li><li><strong>共同点更接近低黏度和低表面张力：</strong>这可能降低短时运动阻力，但不是充分条件。</li><li><strong>LDA为描述性：</strong>{html.escape(coefficient_sentence)} {html.escape(lda_note)}</li><li><strong>候选标签并非实验真值：</strong>没有独立成膜终点时，不能据此宣布最佳溶剂。</li></ol>
    <figure><img src="../figures/group2_external_property_pca_lda.png" alt="PCA与LDA概览"><figcaption>图1　外部物性PCA、描述性LDA、LDA系数和两种候选溶剂的标准化性质曲线。</figcaption></figure>
    <figure><img src="../figures/group2_external_md_spearman_heatmap.png" alt="Spearman相关矩阵"><figcaption>图2　外部溶剂性质与MD响应之间的Spearman秩相关。</figcaption></figure>
    <figure><img src="../figures/group2_prespecified_linear_relationships.png" alt="线性关系"><figcaption>图3　预先指定的四组线性关系；仅标注acetone与n-heptane。</figcaption></figure>
    <h2>acetone与n-heptane数据</h2>{candidate.to_html(index=False,border=0)}
    <h2>LDA标准化系数</h2><p>正系数指向指定候选类，负系数指向其余溶剂；系数不代表因果效应。</p>{coefficients.to_html(index=False,border=0)}
    <h2>绝对Spearman相关最高的8组</h2>{strongest[['external_feature','md_response','pearson_r','pearson_q_bh','spearman_rho','spearman_q_bh']].to_html(index=False,border=0)}
    <h2>结论边界</h2><div class="warning"><ul><li>正类只有2种，LDA没有可靠的外部泛化能力。</li><li>CPME表面张力为20 °C；其余外部物性主要按298.15 K求值。</li><li>变量存在共线性，LDA系数方向可能随特征集合改变。</li><li>MD响应来自单条10 ps轨迹，不等于溶解度或成膜质量。</li></ul></div>
    <h2>数据与来源</h2><ul><li><a href="../tables/group2_external_md_merged.csv">合并分析表</a></li><li><a href="../tables/group2_external_properties_298K.csv">外部物性表</a></li><li><a href="../tables/group2_external_md_correlations.csv">完整相关结果</a></li><li><a href="../tables/group2_lda_scores.csv">LDA得分</a></li><li><a href="../tables/group2_lda_coefficients.csv">LDA系数</a></li><li><a href="{THERMO_URL}">thermo属性对象文档</a></li><li><a href="{ZEON_URL}">Zeon CPME技术数据</a></li><li><a href="{HANSEN_URL}">Hansen参数来源</a></li></ul>
    <h2>AI使用说明</h2><p class="meta">本报告由AI辅助完成物性表构建、统计计算、图形生成和文字整理。用于正式发表前仍应逐项核对原始实验来源和温度条件。</p>
    </main></body></html>"""

    REPORTS.mkdir(parents=True, exist_ok=True)
    md_path = REPORTS / "group2_lda_external_properties_report_zh.md"
    html_path = REPORTS / "group2_lda_external_properties_report_zh.html"
    md_path.write_text(markdown, encoding="utf-8")
    html_path.write_text(html_report, encoding="utf-8")
    return [md_path, html_path]


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    data = merge_data().reset_index(drop=True)
    correlations = correlation_table(data)
    model = model_analysis(data)

    external_columns = [
        "system",
        "chemical_name",
        "cas_number",
        "temperature_K",
        "viscosity_mPa_s",
        "log10_viscosity_mPa_s",
        "dielectric_constant",
        "surface_tension_mN_m",
        "density_g_cm3",
        "molar_mass_g_mol",
        "molar_volume_cm3_mol",
        "viscosity_method",
        "dielectric_method",
        "surface_tension_method",
        "density_method",
        "property_source",
        "property_source_url",
        "temperature_note",
    ]
    data[external_columns].to_csv(
        TABLES / "group2_external_properties_298K.csv", index=False, encoding="utf-8-sig"
    )
    data.to_csv(TABLES / "group2_external_md_merged.csv", index=False, encoding="utf-8-sig")
    correlations.to_csv(
        TABLES / "group2_external_md_correlations.csv", index=False, encoding="utf-8-sig"
    )
    model["score_table"].to_csv(TABLES / "group2_lda_scores.csv", index=False, encoding="utf-8-sig")
    model["coefficient_table"].to_csv(
        TABLES / "group2_lda_coefficients.csv", index=False, encoding="utf-8-sig"
    )
    model["loading_table"].to_csv(
        TABLES / "group2_pca_loadings.csv", index=False, encoding="utf-8-sig"
    )
    (TABLES / "group2_lda_robustness.json").write_text(
        json.dumps(model["robustness"], ensure_ascii=False, indent=2), encoding="utf-8"
    )

    outputs: list[Path] = []
    outputs.extend(overview_figure(data, model))
    outputs.extend(correlation_figure(correlations))
    outputs.extend(linear_relationship_figure(data))
    outputs.extend(report(data, correlations, model))
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
