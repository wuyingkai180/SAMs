"""Three independent COM trajectories in one space, separated by display offsets."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT=Path(__file__).resolve().parent
DATA=ROOT.parent
raw=pd.read_csv(DATA/'three_groups_TAMSD_Kalpha_raw_data.csv')
summary=pd.read_csv(DATA/'three_groups_TAMSD_Kalpha_fit_summary.csv')
selected=[('group1','acetone_5_n-heptane_95'),('group2','acetone'),('group2','n-heptane')]
plt.rcParams.update({'font.family':'Arial','font.size':14,'axes.labelsize':17,'svg.fonttype':'none'})
fig=plt.figure(figsize=(14,8))
ax=fig.add_subplot(111,projection='3d')
cursor=0.
points=[]
handles=[]
records=[]
for group,system in selected:
    m=raw[(raw.group==group)&(raw.system==system)&(raw.record_type=='displacement')].sort_values('time_ps')
    r=summary[(summary.group==group)&(summary.system==system)].iloc[0]
    xyz=m[['OPA_COM_x_A','OPA_COM_y_A','OPA_COM_z_A']].to_numpy()
    rel=xyz-xyz[0]
    offset=cursor-rel[:,0].min()
    pos=rel+np.array([offset,0,0])
    cursor=pos[:,0].max()+4
    assert np.allclose(np.diff(pos,axis=0),np.diff(rel,axis=0))
    assert np.allclose(np.linalg.norm(pos-pos[0],axis=1),m.OPA_displacement_from_t0_A)
    ax.plot(*pos.T,color=r.color_hex,lw=2)
    ax.scatter(*pos[0],c='black',s=48,depthshade=False)
    ax.scatter(*pos[-1],c=r.color_hex,s=64,marker='s',edgecolors='black',linewidth=.7,depthshade=False)
    handles.append(Line2D([],[],color=r.color_hex,lw=2.5,label=r.display_label))
    points.append(pos)
    records.append({'group':group,'system':system,'display_offset_x_A':offset,'samples':len(pos)})
allpos=np.concatenate(points)
lo=np.floor(allpos.min(axis=0))-1
hi=np.ceil(allpos.max(axis=0))+1
ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),zlim=(lo[2],hi[2]),
       xlabel='Δx + display offset (Å)',ylabel='Δy (Å)',zlabel='Δz (Å)')
ax.set_box_aspect(hi-lo)
ax.set_xticks(np.arange(0,hi[0]+.01,5))
ax.set_yticks([lo[1],0,hi[1]])
ax.view_init(elev=22,azim=-78)
ax.set_proj_type('ortho')
ax.tick_params(labelsize=13,pad=2)
for axis in (ax.xaxis,ax.yaxis,ax.zaxis):
    axis.labelpad=12
    axis.pane.set_facecolor((.97,.98,1,.25))
    axis._axinfo['grid'].update(color='#DDE2E8',linewidth=.6)
fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.93),ncol=3,frameon=False,fontsize=14)
fig.suptitle('OPA centre-of-mass trajectories in three solvent environments',fontsize=20,y=.99)
fig.text(.5,.045,'Black circle: start (0 ps)  •  Coloured square: end (10 ps)',ha='center',fontsize=13)
fig.text(.5,.015,'Independent trajectories translated along x for display only; equal Å scales; no rotation or rescaling.',ha='center',fontsize=12)
fig.subplots_adjust(left=.03,right=.93,bottom=.10,top=.85)
for ext in ('png','svg'):
    fig.savefig(ROOT/f'three_selected_solvents_xyz_combined.{ext}',dpi=400,bbox_inches='tight',facecolor='white')
pd.DataFrame(records).to_csv(ROOT/'three_selected_solvents_xyz_display_offsets.csv',index=False)
plt.close(fig)
print('Saved one shared 3D axes with three complete, translated trajectories; geometry verified.')
