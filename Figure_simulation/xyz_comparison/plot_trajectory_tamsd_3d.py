"""Generate a separate XYZ alternative and matched XY/XYZ comparison; preserve XY figures."""
from pathlib import Path
import math
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE=Path(__file__).resolve().parent
DATA=HERE.parent
parser=argparse.ArgumentParser()
parser.add_argument('--output-dir',type=Path,default=HERE)
parser.add_argument('--group2-only',action='store_true')
args=parser.parse_args()
OUT=args.output_dir
OUT.mkdir(parents=True,exist_ok=True)
summary=pd.read_csv(DATA/'three_groups_TAMSD_Kalpha_fit_summary.csv')
raw=pd.read_csv(DATA/'three_groups_TAMSD_Kalpha_raw_data.csv')
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
 'font.size':11,'axes.titlesize':12,'axes.labelsize':12,'xtick.labelsize':11,'ytick.labelsize':11,'svg.fonttype':'none','pdf.fonttype':42})
data={}
for r in summary.itertuples():
 q=raw[(raw.group==r.group)&(raw.system==r.system)]
 m=q[q.record_type=='displacement'].sort_values('time_ps')
 xyz=m[['OPA_COM_x_A','OPA_COM_y_A','OPA_COM_z_A']].to_numpy(float)
 rel=xyz-xyz[0]
 assert np.allclose(np.linalg.norm(rel,axis=1),m.OPA_displacement_from_t0_A)
 data[(r.group,r.system)]=(rel,q[q.record_type=='tamsd'].sort_values('lag_ps'))
extent=4
limits=(-extent,extent)
ticks=np.linspace(-extent,extent,5)
ymax=math.ceil(raw.TAMSD_A2.max())
titles=['Acetone concentration series','Pure solvents','Mixed solvents']
clipping=[]
for (group,system),(pos,_) in data.items():
 outside=np.any(np.abs(pos)>extent,axis=1)
 clipping.append({'group':group,'system':system,'points_outside_view':int(outside.sum()),
                  'total_points':len(pos),'end_outside_view':bool(outside[-1]),
                  'maximum_absolute_coordinate_A':float(np.abs(pos).max())})
if not args.group2_only:
 pd.DataFrame(clipping).to_csv(OUT/'range4_clipping_report.csv',index=False)

def frame3d(ax,view_extent=4):
 limits=(-view_extent,view_extent)
 ticks=np.array([-view_extent,0,view_extent])
 ax.set(xlim=limits,ylim=limits,zlim=limits,xticks=ticks,yticks=ticks,zticks=ticks)
 ax.set_box_aspect((1,1,1))
 ax.view_init(elev=23,azim=-57)
 ax.set_proj_type('ortho')
 ax.set_xlabel('Δx (Å)',labelpad=5);ax.set_ylabel('Δy (Å)',labelpad=5);ax.set_zlabel('Δz (Å)',labelpad=5)
 ax.tick_params(labelsize=10,pad=1)
 for axis in (ax.xaxis,ax.yaxis,ax.zaxis):
  axis.pane.set_facecolor((.97,.98,1,.25))
  axis._axinfo['grid'].update(color='#DDE2E8',linewidth=.5)

def trace(ax,r,three=True,expand_group2=False):
 pos,_=data[(r.group,r.system)]
 if three:
  view_extent=max(4,math.ceil((np.abs(pos).max()+.1)*2)/2) if expand_group2 and r.group=='group2' else extent
  if r.group=='group3' and r.system=='acetone_5_n-heptane_95':
   view_extent=4.5
   assert np.all(np.abs(pos)<view_extent)
  frame3d(ax,view_extent)
  ax.plot(*pos.T,color=r.color_hex,lw=1.2,axlim_clip=True)
  ax.scatter(*pos[0],s=26,c='black',marker='o',depthshade=False)
  ax.scatter(*pos[-1],s=35,c=r.color_hex,marker='s',edgecolors='black',linewidths=.45,depthshade=False,axlim_clip=True)
  outside=np.any(np.abs(pos)>view_extent,axis=1)
  if outside.any():
   note=f'View clipped: {outside.sum()}/{len(pos)} points outside'
   if outside[-1]: note+='\nEnd outside view'
   ax.text2D(.02,.94,note,transform=ax.transAxes,fontsize=9,color='#8B3A26')
 else:
  ax.plot(pos[:,0],pos[:,1],color=r.color_hex,lw=1.2)
  ax.scatter(0,0,s=26,c='black',zorder=4)
  ax.scatter(*pos[-1,:2],s=35,c=r.color_hex,marker='s',edgecolors='black',linewidths=.45,zorder=4)
  ax.set(xlim=limits,ylim=limits,xticks=ticks,yticks=ticks,aspect='equal',xlabel='Δx (Å)',ylabel='Δy (Å)')
  ax.spines[['top','right']].set_visible(False);ax.grid(color='#E5E9EE',lw=.5)

def msd(ax,r):
 _,m=data[(r.group,r.system)]
 use=m.in_alpha_fit_window.astype(str).str.lower()=='true'
 ax.plot(m.lag_ps,m.TAMSD_A2,color=r.color_hex,lw=1.4)
 ax.plot(m.loc[use,'lag_ps'],m.loc[use,'TAMSD_powerlaw_fit_A2'],color=r.color_hex,ls='--',lw=1)
 ax.set(xlim=(0,3.4),ylim=(0,ymax),xlabel='Lag time, τ (ps)',ylabel='3D TAMSD (Å²)')
 ax.spines[['top','right']].set_visible(False);ax.grid(color='#E5E9EE',lw=.5)
 ax.text(.04,.96,f'α = {r.alpha:.3f}   Kα = {r.K_alpha_A2_ps_minus_alpha:.3f}',
   transform=ax.transAxes,va='top',fontsize=10)

def finish(fig,name):
 for ext in ('png','svg'):
  fig.savefig(OUT/f'{name}.{ext}',dpi=300,bbox_inches='tight',facecolor='white')
 plt.close(fig)

legend=[Line2D([],[],color='black',marker='o',ls='',label='Start (0 ps)'),
        Line2D([],[],color='#666666',marker='s',markeredgecolor='black',ls='',label='End (10 ps)')]
for i,g in enumerate(['group1','group2','group3']):
 if args.group2_only and g!='group2': continue
 q=summary[summary.group==g].copy()
 if g=='group1':
  q['order']=q.system.str.extract(r'acetone_(\d+)_')[0].astype(int);q=q.sort_values('order')
 rr=list(q.itertuples());nr=math.ceil(len(rr)/2)
 fig=plt.figure(figsize=(16,3.1*nr+1),layout='constrained')
 outer=fig.add_gridspec(nr,2,wspace=.07,hspace=.10)
 for j,r in enumerate(rr):
  sub=outer[j//2,j%2].subgridspec(1,2,width_ratios=[1.25,1.45],wspace=.16)
  ax=fig.add_subplot(sub[0],projection='3d');ma=fig.add_subplot(sub[1])
  trace(ax,r,expand_group2=True);msd(ma,r)
  ax.set_title(r.display_label.replace(' / ',' /\n') if len(r.display_label)>28 else r.display_label,
               loc='left',pad=8,fontweight='bold')
  ma.set_title(r.display_label.replace(' / ',' /\n') if len(r.display_label)>28 else r.display_label,
               loc='left',fontweight='bold')
 fig.suptitle(f'Group {i+1} | {titles[i]} — 3D trajectory alternative',fontsize=16,fontweight='bold')
 footer=(('XYZ limits adjusted per system to show the full trajectory; equal scales within each panel and fixed view'
  '\nBlack circle: start; coloured square: end. Compare axis ticks across panels.  •  ')
  if g in ('group2','group3') else f'All XYZ axes: −{extent} to +{extent} Å; equal scales and fixed view  •  Black circle: start; coloured square: end\nZoomed view; out-of-range points hidden. TAMSD uses the full trajectory.  •  ')
 fig.supxlabel(footer+r'$K_\alpha$ in Å² ps$^{-\alpha}$',fontsize=9)
 finish(fig,f'{g}_xyz_tamsd_gallery')

if args.group2_only:
 report=[]
 for (group,system),(pos,_) in data.items():
  if group!='group2': continue
  e=max(4,math.ceil((np.abs(pos).max()+.1)*2)/2)
  assert not np.any(np.abs(pos)>e)
  report.append({'system':system,'axis_min_A':-e,'axis_max_A':e,'points_outside_view':0})
 pd.DataFrame(report).to_csv(OUT/'group2_axis_ranges.csv',index=False)
 print('Updated Group 2 in PNG, PDF and SVG; all trajectory points are inside the axes.')
 raise SystemExit(0)

# Same coordinates, scale and MSD for a direct comparison of representation only.
selected=[('group2','n-heptane'),('group1','acetone_5_n-heptane_95'),('group2','isopropanol')]
fig=plt.figure(figsize=(14,11),layout='constrained');gs=fig.add_gridspec(3,3,width_ratios=[1,1.15,1.3])
for i,key in enumerate(selected):
 r=next(r for r in summary.itertuples() if (r.group,r.system)==key)
 xy=fig.add_subplot(gs[i,0]);xyz=fig.add_subplot(gs[i,1],projection='3d');ma=fig.add_subplot(gs[i,2])
 trace(xy,r,False);trace(xyz,r);msd(ma,r)
 xy.set_title(f'{chr(97+i*3)}  XY projection',loc='left',fontweight='bold')
 xyz.set_title(f'{chr(98+i*3)}  XYZ trajectory',loc='left',fontweight='bold')
 ma.set_title(f'{chr(99+i*3)}  {r.display_label}',loc='left',fontweight='bold')
fig.suptitle('XY projection versus XYZ trajectory — identical data and scales',fontsize=16,fontweight='bold')
fig.supxlabel(f'All spatial axes: −{extent} to +{extent} Å  •  Black circle: start (0 ps); coloured square: end (10 ps)\n'+
 'Zoomed views; out-of-range points hidden. TAMSD uses the full XYZ trajectory.',fontsize=10)
finish(fig,'xy_vs_xyz_comparison')
print(f'Created 4 figures in 3 formats. Shared spatial limits: {limits} Å. Output: {OUT}')
