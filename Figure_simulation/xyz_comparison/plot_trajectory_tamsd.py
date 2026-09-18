"""Plot measured COM trajectories and 3D TAMSD using only the adjacent CSVs.

Run: .venv/Scripts/python.exe Figure_simulation/plot_trajectory_tamsd.py
"""
from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
DATA = HERE.parent
OUT = HERE / 'trajectory_tamsd_figures'
OUT.mkdir(exist_ok=True)
summary = pd.read_csv(DATA / 'three_groups_TAMSD_Kalpha_fit_summary.csv')
raw = pd.read_csv(DATA / 'three_groups_TAMSD_Kalpha_raw_data.csv')
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
                     'font.size':9,'axes.titlesize':11,'axes.labelsize':9,
                     'svg.fonttype':'none','pdf.fonttype':42,'axes.linewidth':0.8})
titles = ['Acetone concentration series','Pure solvents','Mixed solvents']
datasets = {}
checks = []
for r in summary.itertuples(index=False):
    q=raw[(raw.group==r.group)&(raw.system==r.system)]
    motion=q[q.record_type=='displacement'].sort_values('time_ps')
    msd=q[q.record_type=='tamsd'].sort_values('lag_ps')
    xyz=motion[['OPA_COM_x_A','OPA_COM_y_A','OPA_COM_z_A']].to_numpy(float)
    rel=xyz-xyz[0]
    calc=np.array([np.mean(np.sum((xyz[m:]-xyz[:-m])**2,axis=1)) for m in range(1,len(msd)+1)])
    assert np.allclose(calc,msd.TAMSD_A2)
    use=msd.in_alpha_fit_window.astype(str).str.lower()=='true'
    a,b=np.polyfit(np.log(msd.loc[use,'lag_ps']),np.log(msd.loc[use,'TAMSD_A2']),1)
    assert np.isclose(a,r.alpha) and np.isclose(np.exp(b),r.K_alpha_A2_ps_minus_alpha)
    datasets[(r.group,r.system)]=(rel,msd)
    checks.append({'group':r.group,'system':r.system,'alpha_verified':a,
                   'K_alpha_verified':np.exp(b),'max_TAMSD_error_A2':np.max(np.abs(calc-msd.TAMSD_A2))})
pd.DataFrame(checks).to_csv(OUT/'data_verification.csv',index=False)

def rows(group):
    q=summary[summary.group==group].copy()
    if group=='group1':
        q['order']=q.system.str.extract(r'acetone_(\d+)_')[0].astype(int)
        q=q.sort_values('order')
    return list(q.itertuples(index=False))

def clean(ax):
    ax.spines[['top','right']].set_visible(False)
    ax.grid(True,color='#E8EBEF',lw=.55)
    ax.set_axisbelow(True)

def save(fig,name):
    for ext in ['png','svg']:
        fig.savefig(OUT/f'{name}.{ext}',dpi=320,bbox_inches='tight',facecolor='white')
    plt.close(fig)

allrel=np.concatenate([v[0] for v in datasets.values()])
xylim=(math.floor(allrel[:,:2].min())-0.3,math.ceil(allrel[:,:2].max())+0.3)
ymax=math.ceil(raw.TAMSD_A2.max())

def trajectory(ax,r):
    pos,_=datasets[(r.group,r.system)]
    ax.plot(pos[:,0],pos[:,1],color=r.color_hex,lw=1.05,alpha=.9)
    ax.scatter(pos[-1,0],pos[-1,1],s=18,marker='s',color=r.color_hex,zorder=4)

def tamsd(ax,r,log=False):
    _,m=datasets[(r.group,r.system)]
    ax.plot(m.lag_ps,m.TAMSD_A2,color=r.color_hex,lw=1.35)
    if log:
        use=m.in_alpha_fit_window.astype(str).str.lower()=='true'
        ax.plot(m.loc[use,'lag_ps'],m.loc[use,'TAMSD_powerlaw_fit_A2'],
                color=r.color_hex,lw=.95,ls='--',alpha=.8)

fig=plt.figure(figsize=(17,12),layout='constrained')
gs=fig.add_gridspec(3,4,width_ratios=[1,1.13,1.13,.98])
for i,group in enumerate(['group1','group2','group3']):
    rr=rows(group)
    ax=fig.add_subplot(gs[i,0]);lin=fig.add_subplot(gs[i,1]);log=fig.add_subplot(gs[i,2]);leg=fig.add_subplot(gs[i,3])
    for r in rr:
        trajectory(ax,r);tamsd(lin,r);tamsd(log,r,True)
    ax.scatter(0,0,c='black',marker='o',s=18,zorder=5)
    ax.set(xlim=xylim,ylim=xylim,aspect='equal',xlabel=r'$\Delta x$ (Å)',ylabel=r'$\Delta y$ (Å)')
    ax.set_title(f'{chr(97+3*i)}  Group {i+1}: XY trajectories',loc='left',fontweight='bold')
    lin.set(xlim=(0,3.4),ylim=(0,ymax),xlabel='Lag time, τ (ps)',ylabel='3D TAMSD (Å²)')
    lin.set_title(f'{chr(98+3*i)}  TAMSD: linear axes',loc='left',fontweight='bold')
    log.set(xscale='log',yscale='log',xlim=(.09,3.5),ylim=(.03,ymax),xlabel='Lag time, τ (ps)',ylabel='3D TAMSD (Å²)')
    log.axvspan(.09,.2,color='#D9DEE5',alpha=.35,zorder=0)
    log.set_title(f'{chr(99+3*i)}  TAMSD: log–log fit',loc='left',fontweight='bold')
    for a in [ax,lin,log]:clean(a)
    leg.axis('off')
    handles=[Line2D([0],[0],color=r.color_hex,lw=2,label=r.display_label) for r in rr]
    leg.legend(handles=handles,loc='center left',frameon=False,fontsize=8,title=titles[i],title_fontsize=10,handlelength=1.5,labelspacing=.75)
fig.suptitle('OPA trajectories and time-averaged mean-squared displacement',fontsize=17,fontweight='bold')
fig.supxlabel('One 10-ps trajectory per condition  •  XY projections: shared origin, square = final position  •  Solid: raw TAMSD; dashed: fit over 0.2–3.3 ps',fontsize=10)
save(fig,'three_groups_trajectory_tamsd')

# Literature-like small multiples: one trajectory and one MSD panel for each condition.
for group in ['group1','group2','group3']:
    rr=rows(group);nrow=math.ceil(len(rr)/2)
    fig=plt.figure(figsize=(14,2.55*nrow+.8),layout='constrained')
    outer=fig.add_gridspec(nrow,2,wspace=.10,hspace=.13)
    for k,r in enumerate(rr):
        inner=outer[k//2,k%2].subgridspec(1,2,width_ratios=[1,1.65],wspace=.14)
        ax=fig.add_subplot(inner[0]);ms=fig.add_subplot(inner[1])
        trajectory(ax,r);ax.scatter(0,0,color='black',s=14,zorder=5)
        ax.set(xlim=xylim,ylim=xylim,aspect='equal',xlabel='Δx (Å)',ylabel='Δy (Å)')
        ax.set_title(f'{k+1:02d}  XY projection',loc='left',fontsize=9)
        tamsd(ms,r,True)
        ms.set(xlim=(0,3.4),ylim=(0,ymax),xlabel='Lag time, τ (ps)',ylabel='3D TAMSD (Å²)')
        label=r.display_label.replace(' / ',' /\n') if len(r.display_label)>28 else r.display_label
        ms.set_title(label,fontsize=10,fontweight='bold',loc='left')
        ms.text(.04,.95,f'α = {r.alpha:.3f}   Kα = {r.K_alpha_A2_ps_minus_alpha:.3f}\nR² = {r.alpha_fit_r2:.3f}',
                transform=ms.transAxes,va='top',fontsize=8,bbox={'facecolor':'white','edgecolor':'none','alpha':.8})
        for a in [ax,ms]:clean(a)
    fig.suptitle(f'Group {group[-1]} | {titles[int(group[-1])-1]}',fontsize=16,fontweight='bold')
    fig.supxlabel(r'Shared axis limits across conditions  •  Solid: raw 3D TAMSD; dashed: power-law fit (0.2–3.3 ps)  •  $K_\alpha$ in Å² ps$^{-\alpha}$',fontsize=9)
    save(fig,f'{group}_trajectory_tamsd_gallery')
print('Created four figures in PNG, PDF and SVG:',OUT)
