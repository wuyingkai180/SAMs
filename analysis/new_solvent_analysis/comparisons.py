"""Group-specific comparisons with explicit small-sample safeguards."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


SOLVENT_CLASS = {
    "acetone": "aprotic-polar",
    "acetonitrile": "aprotic-polar",
    "CPME": "aprotic-polar",
    "DMC": "aprotic-polar",
    "thf": "aprotic-polar",
    "ethanol": "protic",
    "isopropanol": "protic",
    "methanol": "protic",
    "prol": "protic",
    "cyclohexane": "nonpolar",
    "n-heptane": "nonpolar",
    "p-xylene": "nonpolar",
    "toluene": "nonpolar",
    "ethyl_acetate": "aprotic-polar",
    "propylene_carbonate": "aprotic-polar",
}
COMPARISON_METRICS = (
    "alpha",
    "mean_force_eV_A",
    "final_disp_A",
    "head_coord_number_4A",
    "tail_coord_number_4A",
    "head_tail_coord_ratio",
)


@dataclass(frozen=True)
class CorrelationPair:
    pearson: float
    spearman: float
    n: int


@dataclass
class ComparisonResult:
    core_rows: pd.DataFrame
    alternate_base_rows: pd.DataFrame = field(default_factory=pd.DataFrame)
    pure_reference_rows: pd.DataFrame = field(default_factory=pd.DataFrame)
    correlations: pd.DataFrame = field(default_factory=pd.DataFrame)
    summary_tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        average = (start + 1 + end) / 2.0
        ranks[order[start:end]] = average
        start = end
    return ranks


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def pearson_spearman(x, y) -> CorrelationPair:
    x_values = np.asarray(x, dtype=float)
    y_values = np.asarray(y, dtype=float)
    if x_values.shape != y_values.shape:
        raise ValueError("x and y must have the same shape")
    finite = np.isfinite(x_values) & np.isfinite(y_values)
    x_values, y_values = x_values[finite], y_values[finite]
    return CorrelationPair(
        pearson=_pearson(x_values, y_values),
        spearman=_pearson(_average_ranks(x_values), _average_ranks(y_values)),
        n=len(x_values),
    )


def leave_one_out_correlations(x, y) -> list[CorrelationPair]:
    x_values = np.asarray(x, dtype=float)
    y_values = np.asarray(y, dtype=float)
    finite = np.isfinite(x_values) & np.isfinite(y_values)
    x_values, y_values = x_values[finite], y_values[finite]
    if len(x_values) < 4:
        return []
    return [
        pearson_spearman(np.delete(x_values, index), np.delete(y_values, index))
        for index in range(len(x_values))
    ]


def collapse_duplicate_trajectories(frame: pd.DataFrame) -> pd.DataFrame:
    if "sha256" not in frame:
        raise ValueError("sha256 column is required to identify copied trajectories")
    result = frame.copy()
    identifiers: dict[str, str] = {}
    values = []
    for sha256 in result["sha256"].astype(str):
        identifiers.setdefault(sha256, f"trajectory-{len(identifiers) + 1:02d}")
        values.append(identifiers[sha256])
    result["independent_trajectory"] = values
    result["is_duplicate_copy"] = result.duplicated("sha256", keep="first")
    return result


def _correlation_table(frame: pd.DataFrame, x_column: str) -> pd.DataFrame:
    rows = []
    for metric in COMPARISON_METRICS:
        if metric not in frame or x_column not in frame:
            continue
        pair = pearson_spearman(frame[x_column], frame[metric])
        leave_one_out = leave_one_out_correlations(frame[x_column], frame[metric])
        pearson_values = [item.pearson for item in leave_one_out if np.isfinite(item.pearson)]
        spearman_values = [item.spearman for item in leave_one_out if np.isfinite(item.spearman)]
        rows.append(
            {
                "x": x_column,
                "metric": metric,
                "n": pair.n,
                "pearson": pair.pearson,
                "spearman": pair.spearman,
                "loo_pearson_min": min(pearson_values) if pearson_values else float("nan"),
                "loo_pearson_max": max(pearson_values) if pearson_values else float("nan"),
                "loo_spearman_min": min(spearman_values) if spearman_values else float("nan"),
                "loo_spearman_max": max(spearman_values) if spearman_values else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def _direction_changes(values: pd.Series) -> int:
    differences = np.diff(values.to_numpy(float))
    signs = np.sign(differences[np.abs(differences) > 1e-12])
    return int(np.count_nonzero(signs[1:] != signs[:-1])) if len(signs) > 1 else 0


def build_group1_comparison(frame: pd.DataFrame) -> ComparisonResult:
    core = frame.loc[frame["group"] == "group1"].copy()
    core = core.sort_values("cosolvent_fraction", kind="stable").reset_index(drop=True)
    notes = []
    for metric in COMPARISON_METRICS:
        if metric in core:
            changes = _direction_changes(core[metric].dropna())
            if changes:
                notes.append(f"{metric}: {changes} direction change(s); non-monotonic")
    return ComparisonResult(
        core_rows=core,
        correlations=_correlation_table(core, "cosolvent_fraction"),
        notes=notes,
    )


def build_group2_comparison(frame: pd.DataFrame) -> ComparisonResult:
    core = frame.loc[frame["group"] == "group2"].copy()
    core["solvent_class"] = core["system"].map(SOLVENT_CLASS).fillna("unclassified")
    numeric_metrics = [metric for metric in COMPARISON_METRICS if metric in core]
    class_summary = (
        core.groupby("solvent_class", as_index=False)[numeric_metrics].mean(numeric_only=True)
        if numeric_metrics
        else pd.DataFrame()
    )
    return ComparisonResult(
        core_rows=core.sort_values("system", kind="stable").reset_index(drop=True),
        summary_tables={"class_means": class_summary},
        notes=["Chemical classes are manual interpretation labels, not simulation outputs."],
    )


def build_group3_comparison(frame: pd.DataFrame) -> ComparisonResult:
    group = frame.loc[frame.get("group", "group3") == "group3"].copy() if "group" in frame else frame.copy()
    core = group.loc[group["role"] == "n-heptane-base-composition"].copy()
    alternate = group.loc[group["role"] == "alternate-base-composition"].copy()
    references = group.loc[group["role"] == "pure-reference"].copy()
    return ComparisonResult(
        core_rows=core.sort_values("system", kind="stable").reset_index(drop=True),
        alternate_base_rows=alternate.reset_index(drop=True),
        pure_reference_rows=references.sort_values("system", kind="stable").reset_index(drop=True),
        correlations=pd.DataFrame(),
        notes=[
            "Only the four n-heptane-base mixtures enter the core composition ranking.",
            "Alternate-base and pure-reference systems are descriptive context only.",
        ],
    )
