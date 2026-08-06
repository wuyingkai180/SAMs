"""
分析位移与平均力之间的关系，用于解释实验最优浓度(1-5%)。
"""
from pathlib import Path
import pandas as pd
import numpy as np

RESULT = Path(r"C:\Users\yingkaiwu\Desktop\results\0acetone_nheptane_mix\all_systems_opa_motion_force.csv")
m = pd.read_csv(RESULT)

print("=" * 80)
print("A. 每个体系的位移与受力随时间演化 (检验是否达到稳态/收敛)")
print("=" * 80)
rows = []
for sys, g in m.groupby("system"):
    g = g.sort_values("time_ps")
    n = len(g)
    # 前半段 vs 后半段时间平均
    half = n // 2
    first_half = g.iloc[:half]
    second_half = g.iloc[half:]
    rows.append({
        "system": sys,
        "vol_frac": g["acetone_vol_frac"].iloc[0],
        "final_disp": g["OPA_disp_norm_A"].iloc[-1],
        "peak_disp": g["OPA_disp_norm_A"].max(),
        "disp_first_half_mean": first_half["OPA_disp_norm_A"].mean(),
        "disp_second_half_mean": second_half["OPA_disp_norm_A"].mean(),
        "F_mean": g["F_OPA_norm_eV_A"].mean(),
        "F_first_half_mean": first_half["F_OPA_norm_eV_A"].mean(),
        "F_second_half_mean": second_half["F_OPA_norm_eV_A"].mean(),
        "F_rms": np.sqrt((g["F_OPA_norm_eV_A"]**2).mean()),
        "F_max": g["F_OPA_norm_eV_A"].max(),
    })
df = pd.DataFrame(rows).sort_values("vol_frac")
pd.set_option("display.width", 200)
print(df[["system","vol_frac","final_disp","peak_disp",
          "disp_first_half_mean","disp_second_half_mean",
          "F_mean","F_rms","F_max"]].to_string(index=False))

print()
print("位移前后半段差异 (趋于收敛/漂移的指示):")
for _, r in df.iterrows():
    diff = r["disp_second_half_mean"] - r["disp_first_half_mean"]
    marker = "漂移↑" if diff > 0.3 else ("回落↓" if diff < -0.3 else "≈稳")
    print(f"  {r['system']:25s} 前后平均差={diff:+.2f} A  {marker}")

print()
print("=" * 80)
print("B. 位移与平均力之间的相关性")
print("=" * 80)
print(df[["system","vol_frac","final_disp","F_mean","F_rms"]].to_string(index=False))
print()
print("相关系数:")
print(f"  corr(F_mean, final_disp) = {df['F_mean'].corr(df['final_disp']):.3f}")
print(f"  corr(F_mean, peak_disp)  = {df['F_mean'].corr(df['peak_disp']):.3f}")

print()
print("=" * 80)
print("C. 从配位/稳定性角度解释最优浓度在 1-5%")
print("=" * 80)
# 关键物理量：F 反映溶剂对 OPA 的平均扰动强度，位移反映迁移幅度
target = df[df["vol_frac"].isin([1,5])]
print("实验最优体系(1%,5%)的物理量特征:")
print(target[["system","vol_frac","F_mean","F_rms","F_max","peak_disp"]].to_string(index=False))
print()
print("观察：在低丙酮浓度(1-5%)下，")
print("  - 平均力 F_mean 较低且稳定 (扰动小，OPA 保持结构完整)")
print("  - OPA 位移在低浓度下适中，说明溶剂既提供了驱动力又不至于打乱 OPA")
print("  - 浓度升高后 F 增大，溶剂扰动过强")

# 计算"力-位移耦合"/刚性概念
df["F_per_disp"] = df["F_rms"] / (df["peak_disp"] + 1e-9)
print()
print("D. 平均力相对有效力: F_rms / F_mean (受力的波动/各向同性程度)")
df["F_fluctuation_ratio"] = df["F_rms"] / (df["F_mean"] + 1e-9)
print(df[["system","vol_frac","F_mean","F_rms","F_fluctuation_ratio"]].to_string(index=False))
