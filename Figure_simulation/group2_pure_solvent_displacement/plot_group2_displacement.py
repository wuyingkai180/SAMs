"""Plot Group 2 displacement traces with the current manuscript colour map."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DATA = HERE.parent
sys.path.insert(0, str(DATA))
from plot import style, boxed


def main():
    motion = pd.read_csv(DATA / 'three_groups_displacement_curves_raw_from_excel.csv')
    summary = pd.read_csv(DATA / 'three_groups_TAMSD_Kalpha_fit_summary.csv')
    motion = motion.loc[motion.group.eq('group2')].copy()
    colors = summary.loc[summary.group.eq('group2')].set_index('system')
    assert motion.system.nunique() == 14
    assert set(motion.system) == set(colors.index)
    style()
    fig, ax = plt.subplots(figsize=(15, 12))
    for system, trace in motion.groupby('system', sort=False):
        trace = trace.sort_values('time_ps')
        assert len(trace) == 101
        assert np.allclose(trace.time_ps, np.linspace(0, 10, 101))
        assert np.isfinite(trace.OPA_disp_norm_A).all()
        row = colors.loc[system]
        ax.plot(trace.time_ps, trace.OPA_disp_norm_A,
                color=row.color_hex, lw=1.6, label=row.display_label)
    ax.set_xlabel('Simulation time (ps)', fontsize=20, labelpad=10)
    ax.set_ylabel('OPA displacement from t=0 (Å)', fontsize=20, labelpad=10)
    ax.tick_params(axis='both', labelsize=18)
    ax.set_title('Group 2 | complete OPA displacement trajectories',
                 fontsize=21, fontweight='bold', pad=14)
    # Match the reference panel's padding and integer y ticks.
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-0.35635, 7.48335)
    ax.set_xticks(np.arange(0, 11, 2))
    ax.set_yticks(np.arange(0, 8))
    boxed(ax)
    fig.legend(*ax.get_legend_handles_labels(), loc='lower center',
               bbox_to_anchor=(0.5, 0.015), ncol=2, fontsize=20,
               frameon=False, columnspacing=2.5, handlelength=2.5)
    fig.subplots_adjust(left=0.095, right=0.99, top=0.94, bottom=0.39)
    stem = HERE / 'group2_displacement_curves_from_excel'
    for ext in ('png', 'svg'):
        fig.savefig(stem.with_suffix('.' + ext), dpi=600,
                    bbox_inches='tight', facecolor='white')
    plt.close(fig)
    motion['color_hex'] = motion.system.map(colors.color_hex)
    motion.to_csv(stem.with_suffix('.csv'), index=False)
    print('Verified and plotted 14 Group 2 traces, 101 original samples each; no smoothing or fitting.')


if __name__ == '__main__':
    main()
