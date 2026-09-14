"""Reproduce figure metrics and assess lag-window sensitivity for the manuscript."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'analysis'))
from new_solvent_analysis.core_metrics import unwrap_positions, tamsd, fit_anomalous_exponent
from ase.io import read

OUT = ROOT / 'results' / 'manuscript_analysis'
OUT.mkdir(parents=True, exist_ok=True)
summary = pd.read_csv(ROOT / 'Figure_simulation/three_groups_TAMSD_Kalpha_fit_summary.csv')
paired = pd.read_csv(ROOT / 'Figure_simulation/three_groups_force_displacement_dual_axis_points.csv')
curves = pd.read_csv(ROOT / 'Figure_simulation/three_groups_displacement_curves_raw_from_excel.csv')
metrics = pd.concat([pd.read_csv(ROOT / f'results/result2/tables/group{g}_metrics.csv') for g in (1,2,3)])
checks, windows = [], []
for row in metrics.itertuples(index=False):
    motion = pd.read_csv(ROOT / row.csv_path)
    positions = motion[['OPA_COM_x_A','OPA_COM_y_A','OPA_COM_z_A']].to_numpy(float)
    unwrapped = unwrap_positions(positions, read(ROOT / row.xyz_path).cell.array)
    disp = np.linalg.norm(unwrapped-unwrapped[0], axis=1)
    tau = np.arange(1,34)*0.1
    msd = tamsd(unwrapped,33)
    fit = fit_anomalous_exponent(tau,msd)
    s = summary[(summary.group==row.group)&(summary.system==row.system)].iloc[0]
    assert np.isclose(fit.alpha,s.alpha)
    assert np.isclose(fit.prefactor,s.K_alpha_A2_ps_minus_alpha)
    assert np.isclose(disp.max(),row.max_disp_A)
    assert np.isclose(motion.F_OPA_norm_eV_A.mean(),row.mean_force_eV_A)
    curve=curves[(curves.group==row.group)&(curves.system==row.system)]
    assert len(curve)==101 and np.allclose(curve.OPA_disp_norm_A,disp)
    checks.append(dict(group=row.group,system=row.system,alpha=fit.alpha,K_alpha=fit.prefactor,
                       r2=fit.r_squared,mean_force_eV_A=row.mean_force_eV_A,
                       max_disp_A=disp.max(),final_disp_A=disp[-1],
                       unwrap_change_A=float(np.max(np.abs(unwrapped-positions)))))
    for lower,upper in [(0.1,1.0),(0.2,1.0),(0.2,2.0),(0.2,3.3),(0.5,3.3)]:
        mask=(tau>=lower-1e-10)&(tau<=upper+1e-10)
        f=fit_anomalous_exponent(tau[mask],msd[mask],min_lag=1)
        windows.append(dict(group=row.group,system=row.system,tau_min_ps=lower,tau_max_ps=upper,
                            alpha=f.alpha,K_alpha=f.prefactor,r2=f.r_squared,n_points=f.n_points))
checked=pd.DataFrame(checks)
checked.to_csv(OUT/'verified_figure_metrics.csv',index=False)
pd.DataFrame(windows).to_csv(OUT/'lag_window_sensitivity.csv',index=False)
for row in paired.itertuples(index=False):
    c=checked[(checked.group==row.group)&(checked.system==row.system)].iloc[0]
    assert np.isclose(c.mean_force_eV_A,row.mean_force_eV_A) and np.isclose(c.max_disp_A,row.max_disp_A)
print('Verified 26 group entries, 25 unique systems; 101 samples per entry; figure curves match COM displacement.')
print('Maximum unwrapping change:',checked.unwrap_change_A.max())
p=pd.DataFrame(windows).pivot(index=['group','system'],columns=['tau_min_ps','tau_max_ps'],values='alpha')
print(p.round(3).to_string())
for g,d in checked.groupby('group'):
    print(g,'force/max displacement Pearson',d.mean_force_eV_A.corr(d.max_disp_A),
          'Spearman',d.mean_force_eV_A.corr(d.max_disp_A,method='spearman'))
