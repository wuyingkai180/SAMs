"""Place three XZ projections in one axes using translation only."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

ARROW_LEN=.5
def add_arrow(ax,tail,head,color):
    ax.add_patch(FancyArrowPatch(tail,head,arrowstyle='-|>',mutation_scale=26,
                                  linewidth=1.3,color=color,zorder=4,
                                  shrinkA=0,shrinkB=0))

ROOT=Path(__file__).resolve().parent
DATA=ROOT.parent
raw=pd.read_csv(DATA/'three_groups_TAMSD_Kalpha_raw_data.csv')
summary=pd.read_csv(DATA/'three_groups_TAMSD_Kalpha_fit_summary.csv')
selected=[('group1','acetone_5_n-heptane_95'),('group2','n-heptane'),('group2','acetone')]
plt.rcParams.update({'font.family':'Arial','font.size':13,'axes.labelsize':16,'svg.fonttype':'none'})
fig,ax=plt.subplots(figsize=(8,8))
boxes=[];records=[];allpoints=[];labels=[]
for i,(group,system) in enumerate(selected):
    m=raw[(raw.group==group)&(raw.system==system)&(raw.record_type=='displacement')].sort_values('time_ps')
    r=summary[(summary.group==group)&(summary.system==system)].iloc[0]
    xyz=m[['OPA_COM_x_A','OPA_COM_y_A','OPA_COM_z_A']].to_numpy()
    pos=(xyz-xyz[0])[:,[0,2]]
    lo=pos.min(axis=0);hi=pos.max(axis=0)
    if i==0: offset=np.array([-1.5-hi[0],1.5-lo[1]])
    elif i==1: offset=np.array([1.5-lo[0],1.5-lo[1]])
    else: offset=np.array([-(lo[0]+hi[0])/2,.5-hi[1]])
    shifted=pos+offset
    assert np.allclose(np.diff(shifted,axis=0),np.diff(pos,axis=0))
    assert np.allclose(shifted-shifted[0],pos)
    lower=shifted.min(axis=0);upper=shifted.max(axis=0)
    boxes.append((lower,upper));allpoints.append(shifted)
    ax.plot(*shifted.T,color=r.color_hex,lw=2)
    ax.scatter(*shifted[0],c='black',s=44,zorder=4)
    end_dir=shifted[-1]-shifted[-2]
    end_dir=end_dir/np.linalg.norm(end_dir)
    add_arrow(ax,shifted[-1]-end_dir*ARROW_LEN,shifted[-1],'black')
    label=r.display_label.replace(' / ',' /\n')
    labelpos=((lower[0]+upper[0])/2,upper[1]+.55)
    ax.text(*labelpos,label,ha='center',va='bottom',fontsize=14,fontweight='bold')
    labels.append(labelpos)
    records.append({'system':system,'display_offset_x_A':offset[0],'display_offset_z_A':offset[1]})
for i,(lo,hi) in enumerate(boxes):
    for otherlo,otherhi in boxes[i+1:]:
        assert np.any(hi<otherlo) or np.any(otherhi<lo), 'Trajectories overlap'
ax.set(xlim=(-8,6),ylim=(-4,9),
       aspect='equal',xlabel='Δx + display offset (Å)',ylabel='Δz + display offset (Å)')
ax.grid(color='#E3E7EB',lw=.6);ax.set_axisbelow(True)
handles=[Line2D([],[],marker='o',color='black',ls='',label='Start (0 ps)',markersize=8),
         Line2D([],[],marker='>',color='black',ls='',label='End (10 ps)',markersize=10)]
fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.045),ncol=2,frameon=False,fontsize=12)
fig.text(.5,.018,'XZ projections; translated for display only. Identical scale; no rotation or rescaling.',ha='center',fontsize=10)
fig.subplots_adjust(left=.14,right=.98,top=.97,bottom=.17)
for ext in ('png','svg'):
    fig.savefig(ROOT/f'three_selected_solvents_xz_triangle.{ext}',dpi=400,bbox_inches='tight',facecolor='white')
pd.DataFrame(records).to_csv(ROOT/'three_selected_solvents_xz_triangle_offsets.csv',index=False)
plt.close(fig)
print('Triangle layout saved; shapes preserved and trajectory bounding boxes do not overlap.')
