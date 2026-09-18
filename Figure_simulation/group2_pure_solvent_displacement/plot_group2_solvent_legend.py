"""Standalone legend ordered by decreasing final displacement at 10 ps."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT=Path(__file__).resolve().parent
DATA=ROOT.parent
motion=pd.read_csv(DATA/'three_groups_displacement_curves_raw_from_excel.csv')
summary=pd.read_csv(DATA/'three_groups_TAMSD_Kalpha_fit_summary.csv')
colors=summary[summary.group.eq('group2')].set_index('system').color_hex
final=(motion[motion.group.eq('group2')].sort_values('time_ps')
       .groupby('system',sort=False).tail(1)
       .sort_values('OPA_disp_norm_A',ascending=False).copy())
assert len(final)==14 and np.allclose(final.time_ps,10)
final['color_hex']=final.system.map(colors)
assert final.color_hex.notna().all()
plt.rcParams.update({'font.family':'Arial','svg.fonttype':'none'})
handles=[Line2D([],[],color=r.color_hex,lw=2,label=r.display_label)
         for r in final.itertuples()]
fig=plt.figure(figsize=(6,8))
fig.legend(handles=handles,loc='center',ncol=1,fontsize=20,frameon=False,
           handlelength=2.5,labelspacing=.65)
for ext in ('png','svg'):
    fig.savefig(ROOT/f'group2_solvent_legend_14rows.{ext}',dpi=400,
                bbox_inches='tight',pad_inches=.08,transparent=True)
plt.close(fig)
final[['system','display_label','time_ps','OPA_disp_norm_A','color_hex']].to_csv(
    ROOT/'group2_solvent_legend_final_displacement_order.csv',index=False)
print(final[['display_label','OPA_disp_norm_A']].to_string(index=False))
