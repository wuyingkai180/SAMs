"""Extract displacement-only data from the three Excel workbooks and plot it."""

from __future__ import annotations

from pathlib import Path
import re

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from publication_reporting import _boxed_axes, _save, _style, display_label


GROUPS = ("group1", "group2", "group3")
GROUP_TITLES = {
    "group1": "Group 1 | Acetone concentration series in n-heptane",
    "group2": "Group 2 | Pure-solvent comparison",
    "group3": "Group 3 | Solvent-composition comparison",
}


def _system_from_sheet(sheet: str) -> str:
    return sheet.strip()


def _load_color_map(root: Path) -> dict[str, str]:
    summary = root / "results" / "result2" / "tamsd_kalpha_combined" / "tables" / "three_groups_TAMSD_Kalpha_fit_summary.csv"
    if not summary.exists():
        return {}
    frame = pd.read_csv(summary)
    return dict(frame.drop_duplicates("display_label").set_index("display_label")["color_hex"])


def extract(root: Path) -> pd.DataFrame:
    records: list[pd.DataFrame] = []
    for group in GROUPS:
        workbook = root / "data" / f"{group}.xlsx"
        excel = pd.ExcelFile(workbook)
        for sheet in excel.sheet_names:
            frame = pd.read_excel(workbook, sheet_name=sheet)
            required = {"time_ps", "OPA_disp_norm_A"}
            missing = required - set(frame.columns)
            if missing:
                raise ValueError(f"{workbook.name}/{sheet} missing columns: {sorted(missing)}")
            system = _system_from_sheet(sheet)
            label = display_label(system)
            part = frame[["time_ps", "OPA_disp_norm_A"]].copy()
            part.insert(0, "group", group)
            part.insert(1, "system", system)
            part.insert(2, "display_label", label)
            part["source_workbook"] = workbook.name
            part["source_sheet"] = sheet
            records.append(part)
    return pd.concat(records, ignore_index=True)


def plot(root: Path, data: pd.DataFrame) -> Path:
    figures = root / "results" / "result2" / "tamsd_kalpha_combined" / "figures"
    colors = _load_color_map(root)
    fallback = mpl.colormaps["tab20"]
    labels = list(dict.fromkeys(data["display_label"].tolist()))
    fallback_colors = {label: fallback(i / max(len(labels) - 1, 1)) for i, label in enumerate(labels)}
    fig, ax = plt.subplots(figsize=(15.0, 9.0), constrained_layout=True)
    for index, row in data.drop_duplicates(["group", "system"])[["group", "system", "display_label"]].iterrows():
        trace = data[(data["group"] == row.group) & (data["system"] == row.system)]
        color = colors.get(row.display_label, fallback_colors[row.display_label])
        ax.plot(trace["time_ps"], trace["OPA_disp_norm_A"], color=color, lw=1.15,
                label=f"{row.group} | {row.display_label}")
    ax.set_xlabel("Simulation time (ps)")
    ax.set_ylabel("OPA displacement from t=0 (Å)")
    ax.set_title("All groups | complete OPA displacement trajectories", fontsize=13, fontweight="bold")
    _boxed_axes(ax)
    handles, labels_out = ax.get_legend_handles_labels()
    fig.legend(handles, labels_out, loc="lower center", bbox_to_anchor=(0.5, -0.02),
               ncol=4, fontsize=6.1, frameon=True, edgecolor="#202428", fancybox=False)
    fig.subplots_adjust(bottom=0.23)
    return _save(fig, figures / "three_groups_all_displacement_curves_from_excel")[0]


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    out = root / "results" / "result2" / "tamsd_kalpha_combined" / "tables"
    out.mkdir(parents=True, exist_ok=True)
    _style()
    data = extract(root)
    csv_path = out / "three_groups_displacement_curves_raw_from_excel.csv"
    data.to_csv(csv_path, index=False, encoding="utf-8-sig")
    figure_path = plot(root, data)
    print(csv_path)
    print(figure_path)
    print(f"rows={len(data)}, trajectories={data[['group','system']].drop_duplicates().shape[0]}")


if __name__ == "__main__":
    main()
