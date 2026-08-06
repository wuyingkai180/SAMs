"""
整合 0acetone_nheptane_mix 目录下所有体系的 MD 数据为一个 CSV。

每个体系目录下有一个 *_opa_motion_force.csv（含 OPA 质心位移与受力随时间的轨迹），
这里将全部体系的轨迹合并成一个"长格式"整合文件，并额外输出每个体系的汇总统计。

输出：
  - all_systems_opa_motion_force.csv   （合并后的长格式轨迹数据）
  - all_systems_summary.csv            （每个体系的汇总统计）
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS_DIR = Path(r"C:\Users\yingkaiwu\Desktop\results\0acetone_nheptane_mix")


def list_system_csvs(root: Path) -> list[tuple[Path, str]]:
    """返回 (csv_path, system_name) 列表。"""
    found = []
    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        for csv_file in sub.glob("*_opa_motion_force.csv"):
            found.append((csv_file, sub.name))
    return found


def merge_all(root: Path) -> pd.DataFrame:
    items = list_system_csvs(root)
    if not items:
        raise FileNotFoundError(f"No *_opa_motion_force.csv found under {root}")

    frames = []
    for csv_path, system in items:
        df = pd.read_csv(csv_path)

        # 从目录名解析丙酮体积分数，便于排序与标记。
        vol_frac = None
        m = re.search(r"acetone_(\d+)_n-heptane_(\d+)", system)
        if m:
            vol_frac = int(m.group(1))

        df.insert(0, "system", system)
        df.insert(1, "acetone_vol_frac", vol_frac)

        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def summarize(merged: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for system, grp in merged.groupby("system", sort=False):
        vol_frac = grp["acetone_vol_frac"].iloc[0]
        rows.append(
            {
                "system": system,
                "acetone_vol_frac": vol_frac,
                "n_frames": int(len(grp)),
                "time_total_ps": float(grp["time_ps"].iloc[-1]),
                "final_opa_disp_norm_A": float(grp["OPA_disp_norm_A"].iloc[-1]),
                "mean_opa_disp_norm_A": float(grp["OPA_disp_norm_A"].mean()),
                "mean_F_OPA_norm_eV_A": float(grp["F_OPA_norm_eV_A"].mean()),
                "max_F_OPA_norm_eV_A": float(grp["F_OPA_norm_eV_A"].max()),
                "rms_F_OPA_norm_eV_A": float(np.sqrt((grp["F_OPA_norm_eV_A"] ** 2).mean())),
            }
        )
    return pd.DataFrame(rows).sort_values("acetone_vol_frac")


def main() -> None:
    merged = merge_all(RESULTS_DIR)
    summary = summarize(merged)

    merged_out = Path(RESULTS_DIR) / "all_systems_opa_motion_force.csv"
    summary_out = Path(RESULTS_DIR) / "all_systems_summary.csv"

    merged.to_csv(merged_out, index=False)
    summary.to_csv(summary_out, index=False)

    print(f"整合轨迹文件已保存: {merged_out}")
    print(f"  总行数 (帧): {len(merged)}")

    print(f"\n汇总文件已保存: {summary_out}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
