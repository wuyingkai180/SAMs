"""Rebuild existing root figures with larger, readable typography from source data."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import matplotlib.pyplot as plt
import plot as _plot
from plot import style, boxed, save, load_tables, paired_points, displacement, GROUPS, GROUP_TITLES

HERE = Path(__file__).resolve().parent
_plot.OUT = HERE


def scatter_figures(summary, groups=GROUPS, quadrants=False):
    positions = {
        'acetone_1_n-heptane_99': (1.78, .78),
        'acetone_5_n-heptane_95': (1.98, .95),
        'acetone_10_n-heptane_90': (1.72, 1.06),
        'acetone_20_n-heptane_80': (1.25, 1.075),
        'acetone_30_n-heptane_70': (.96, .40),
        'acetone_40_n-heptane_60': (.89, .83),
        'acetone_50_n-heptane_50': (1.43, .68),
    }
    offsets = {'acetone': (12, 28), 'n-heptane': (22,-6),
               'acetonitrile': (0,-15), 'toluene': (-10,16),
               'ethanol': (10,-10), 'methanol': (-10,13),
               'thf': (10,8), 'ethyl_acetate': (10,-14),
               'cyclohexane': (-10,6), 'CPME': (-8,10),
               'DMC': (-8,10), 'isopropanol': (8,4),
               'p-xylene': (0,-18), 'propylene_carbonate': (0,14)}
    for group in groups:
        q=summary[summary.group==group]
        fig,ax=plt.subplots(figsize=(7.6,5.8),layout='constrained')
        # Four horizontal alpha ranges; only the middle two are shaded.
        ax.axhspan(.6,.8,facecolor='#DAEAF7',edgecolor='none',zorder=0)
        ax.axhspan(.8,1.,facecolor='#F8E3CE',edgecolor='none',zorder=0)
        for boundary in (.6,.8):
            ax.axhline(boundary,color='#656A6F',lw=.9,ls='--',zorder=2)
        for r in q.itertuples():
            x=r.alpha_prefactor_A2_ps_alpha
            ax.scatter(x,r.alpha,c=r.color_hex,marker=r.marker,s=102,
                       edgecolor='#171A1D',linewidth=.8,zorder=3)
            label=r.display_label.replace(' / ',' /\n')
            kw=dict(fontsize=10.5,arrowprops=dict(arrowstyle='-',color='#7A8085',lw=.6),
                    va='center',annotation_clip=False)
            if group=='group1':
                ax.annotate(label,(x,r.alpha),xytext=positions[r.system],ha='center',**kw)
            elif group=='group2':
                dx,dy=offsets[r.system]
                ax.annotate(label,(x,r.alpha),xytext=(dx,dy),textcoords='offset points',
                            ha='right' if dx<0 else 'left',**kw)
            else:
                ax.annotate(label,(x,r.alpha),xytext=(-12,22),textcoords='offset points',ha='right',**kw)
        ax.axhline(1,color='#656A6F',lw=.9,ls='--')
        ax.set(xlim=(.37,2.16),ylim=(.35,1.16),
               xlabel=r'TAMSD prefactor, $K_\alpha$ ($\mathrm{\AA^2\,ps^{-\alpha}}$)',
               ylabel=r'TAMSD exponent, $\alpha$')
        ax.set_title(GROUP_TITLES[group]+'\nTAMSD amplitude and scaling exponent',fontsize=11,fontweight='bold')
        boxed(ax)
        suffix='_quadrants' if quadrants else ''
        for ext in ('png','svg'):
            fig.savefig(HERE/f'{group}_Kalpha_vs_alpha{suffix}.{ext}',dpi=600,
                        bbox_inches='tight',facecolor='white')
        plt.close(fig)


if __name__=='__main__':
    style()
    summary,paired,motion=load_tables()
    scatter_figures(summary)
    paired_points(paired)
    displacement(motion,summary)
    (HERE/'three_groups_all_displacement_curves.png').replace(HERE/'three_groups_all_displacement_curves_from_excel.png')
