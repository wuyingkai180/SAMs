"""Three original COM trajectories in a horizontal row; PNG and SVG only."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
DATA=ROOT.parent
raw=pd.read_csv(DATA/'three_groups_TAMSD_Kalpha_raw_data.csv')
summary=pd.read_csv(DATA/'three_groups_TAMSD_Kalpha_fit_summary.csv')
selected=[('group1','acetone_5_n-heptane_95',7.5),
          ('group2','acetone',7.5),('group2','n-heptane',7.5)]
trajectories=[]
plt.rcParams.update({'font.family':'Arial','font.size':13,'axes.labelsize':16,'svg.fonttype':'none'})
fig=plt.figure(figsize=(15,5.7))
for i,(group,system,extent) in enumerate(selected):
    r=summary[(summary.group==group)&(summary.system==system)].iloc[0]
    m=raw[(raw.group==group)&(raw.system==system)&(raw.record_type=='displacement')].sort_values('time_ps')
    xyz=m[['OPA_COM_x_A','OPA_COM_y_A','OPA_COM_z_A']].to_numpy()
    pos=xyz-xyz[0]
    assert np.all(np.abs(pos)<extent)
    assert np.allclose(np.linalg.norm(pos,axis=1),m.OPA_displacement_from_t0_A)
    trajectories.append((r,pos))
    ax=fig.add_subplot(1,3,i+1,projection='3d')
    ax.plot(*pos.T,color=r.color_hex,lw=1.8)
    ax.scatter(*pos[0],c='black',s=42,depthshade=False)
    ax.scatter(*pos[-1],c=r.color_hex,s=55,marker='s',edgecolors='black',linewidth=.6,depthshade=False)
    ax.set(xlim=(-extent,extent),ylim=(-extent,extent),zlim=(-extent,extent),
           xticks=[-extent,0,extent],yticks=[-extent,0,extent],zticks=[-extent,0,extent],
           xlabel='Δx (Å)',ylabel='Δy (Å)',zlabel='Δz (Å)')
    ax.set_box_aspect((1,1,1))
    # Omit the near-corner y label where it would collide with the x endpoint label.
    ax.set_yticklabels(['', '0', f'{extent:g}'])
    ax.view_init(elev=23,azim=-57)
    ax.set_proj_type('ortho')
    ax.tick_params(labelsize=12,pad=1)
    ax.set_title(r.display_label.replace(' / ',' /\n'),fontsize=16,fontweight='bold',pad=14)
    for axis in (ax.xaxis,ax.yaxis,ax.zaxis):
        axis.labelpad=7
        axis.pane.set_facecolor((.97,.98,1,.25))
        axis._axinfo['grid'].update(color='#DDE2E8',linewidth=.5)
fig.subplots_adjust(left=.015,right=.96,bottom=.20,top=.86,wspace=.12)
fig.text(.5,.075,'Black circle: start (0 ps)  •  Coloured square: end (10 ps)',ha='center',fontsize=13)
fig.text(.5,.025,'All spatial axes: −7.5 to +7.5 Å; equal XYZ scales across all panels.',ha='center',fontsize=11)
for ext in ('png','svg'):
    fig.savefig(ROOT/f'three_selected_solvents_xyz_side_by_side.{ext}',dpi=400,bbox_inches='tight',facecolor='white')
plt.close(fig)
for plane,first,second in [('xy',0,1),('yz',1,2),('xz',0,2)]:
    fig,axes=plt.subplots(1,3,figsize=(15,5.7))
    for ax,(r,pos) in zip(axes,trajectories):
        ax.plot(pos[:,first],pos[:,second],color=r.color_hex,lw=1.8)
        ax.scatter(0,0,c='black',s=42,zorder=4)
        ax.scatter(pos[-1,first],pos[-1,second],c=r.color_hex,s=55,
                   marker='s',edgecolors='black',linewidth=.6,zorder=4)
        ax.set(xlim=(-7.5,7.5),ylim=(-7.5,7.5),aspect='equal',
               xticks=[-7.5,-5,-2.5,0,2.5,5,7.5],
               yticks=[-7.5,-5,-2.5,0,2.5,5,7.5],
               xlabel=f'Δ{"xyz"[first]} (Å)',ylabel=f'Δ{"xyz"[second]} (Å)')
        ax.tick_params(labelsize=12)
        ax.grid(color='#DDE2E8',lw=.5)
        ax.set_axisbelow(True)
        ax.set_title(r.display_label.replace(' / ',' /\n'),fontsize=16,fontweight='bold',pad=14)
    fig.subplots_adjust(left=.055,right=.985,bottom=.20,top=.83,wspace=.30)
    fig.text(.5,.075,'Black circle: start (0 ps)  •  Coloured square: end (10 ps)',ha='center',fontsize=13)
    fig.text(.5,.025,f'{plane.upper()} projection  •  All axes: −7.5 to +7.5 Å; equal scales across all panels.',ha='center',fontsize=11)
    for ext in ('png','svg'):
        fig.savefig(ROOT/f'three_selected_solvents_{plane}_side_by_side.{ext}',dpi=400,bbox_inches='tight',facecolor='white')
    plt.close(fig)
print('Saved XYZ, XY, YZ and XZ figures with identical limits; complete trajectories verified.')
