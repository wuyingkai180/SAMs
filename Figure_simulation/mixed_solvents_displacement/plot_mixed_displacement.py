"""Mixed-solvent displacement traces and a separate endpoint-sorted legend."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT=Path(__file__).resolve().parent
DATA=ROOT.parent
sys.path.insert(0, str(DATA))
from plot import style, boxed

raw=pd.read_csv(DATA/'three_groups_displacement_curves_raw_from_excel.csv')
summary=pd.read_csv(DATA/'three_groups_TAMSD_Kalpha_fit_summary.csv')
mixed=raw[raw.group.isin(['group1','group3'])].copy()
# The shared acetone 5% mixture is the same trajectory, not an independent repeat.
for system, rows in mixed.groupby('system'):
    parts=[p.sort_values('time_ps') for _,p in rows.groupby('group')]
    for p in parts[1:]:
        assert np.allclose(parts[0][['time_ps','OPA_disp_norm_A']],p[['time_ps','OPA_disp_norm_A']])
mixed=mixed.drop_duplicates(['system','time_ps'])
colors=summary[summary.group.isin(['group1','group3'])].drop_duplicates('system').set_index('system').color_hex
assert mixed.system.nunique()==11
style()
fig,ax=plt.subplots(figsize=(15,7.5))
for system,trace in mixed.groupby('system',sort=False):
    trace=trace.sort_values('time_ps')
    assert len(trace)==101 and np.allclose(trace.time_ps,np.linspace(0,10,101))
    ax.plot(trace.time_ps,trace.OPA_disp_norm_A,color=colors[system],lw=1.6)
ax.set_xlabel('Simulation time (ps)',fontsize=20,labelpad=10)
ax.set_ylabel('OPA displacement from t=0 (Å)',fontsize=20,labelpad=10)
ax.tick_params(axis='both',labelsize=18)
ax.set_title('Mixed solvents | complete OPA displacement trajectories',fontsize=21,fontweight='bold',pad=14)
ax.set(xlim=(-.5,10.5),ylim=(-.3,5.7),xticks=np.arange(0,11,2),yticks=np.arange(0,6))
boxed(ax)
fig.subplots_adjust(left=.095,right=.99,bottom=.15,top=.90)
for ext in ('png','svg'):
    fig.savefig(ROOT/f'mixed_solvents_displacement_curves_from_excel.{ext}',dpi=400,bbox_inches='tight',facecolor='white')
plt.close(fig)
final=(mixed.sort_values('time_ps').groupby('system',sort=False).tail(1)
       .sort_values('OPA_disp_norm_A',ascending=False).copy())
assert np.allclose(final.time_ps,10)
final['color_hex']=final.system.map(colors)
handles=[Line2D([],[],color=r.color_hex,lw=2,label=r.display_label) for r in final.itertuples()]
fig=plt.figure(figsize=(9,6.5))
fig.legend(handles=handles,loc='center',ncol=1,fontsize=20,frameon=False,handlelength=2.5,labelspacing=.65)
for ext in ('png','svg'):
    fig.savefig(ROOT/f'mixed_solvents_legend_11rows.{ext}',dpi=400,bbox_inches='tight',pad_inches=.08,transparent=True)
plt.close(fig)
mixed.to_csv(ROOT/'mixed_solvents_displacement_curves_from_excel.csv',index=False)
final[['system','display_label','time_ps','OPA_disp_norm_A','color_hex']].to_csv(ROOT/'mixed_solvents_legend_final_displacement_order.csv',index=False)
print(final[['display_label','OPA_disp_norm_A']].to_string(index=False))
