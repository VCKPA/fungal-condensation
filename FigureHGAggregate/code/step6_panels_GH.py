#!/usr/bin/env python3
"""
step6_panels_EH.py
==================
Panel E: R(d) exemplar scatter with shaded Near / Mid / Far zones.
Panel F: Zone-based size gradient (R_far - R_near) / R_mid vs (1 - a_w).

INPUT:
    ../raw_data/aggregate_edt/*_edt_droplets.csv
    ../output/hydrogel_metrics.csv

OUTPUT:
    ../output/panels_EH.pdf / .svg / .png
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Paths ────────────────────────────────────────────────────────────────────
THIS_DIR    = Path(__file__).parent
PROJECT_DIR = THIS_DIR.parent
OUTPUT_DIR  = PROJECT_DIR / 'output'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HG_METRICS = OUTPUT_DIR / 'hydrogel_metrics.csv'
AGG_DIR    = PROJECT_DIR / 'raw_data' / 'aggregate_edt'

# ── Colors ────────────────────────────────────────────────────────────────────
COLORS = {'Agar': '#3A9E6F', '1:1': '#5B8FC9', '2:1': '#C0392B'}

# ── Exemplar trials ──────────────────────────────────────────────────────────
EXEMPLAR_AGAR = 'agar.4'
EXEMPLAR_NACL = '2to1.2'

# ── Zone definitions ─────────────────────────────────────────────────────────
T_WINDOW     = (14.5, 15.5)
BIN_WIDTH_UM = 50
MIN_DROPS    = 10
MAX_DIST_MM  = 2.5

MID_LO_UM  = 750
MID_HI_UM  = 1250
FAR_LO_UM  = 1500

# ── Style (IDENTICAL to step4_panels_BC.py) ──────────────────────────────────
MM = 1 / 25.4
TS = 7.0;  LS = 8.5;  PL = 12.0
LW = 0.6

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': TS,
    'axes.linewidth': LW,
    'xtick.major.width': LW, 'ytick.major.width': LW,
    'xtick.major.size': 3.5, 'ytick.major.size': 3.5,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'svg.fonttype': 'none',
})


def style_ax(ax):
    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=TS)


def load_droplets(trial_id):
    path = AGG_DIR / f'{trial_id}_edt_droplets.csv'
    df = pd.read_csv(path)
    tw = df[(df['time_min'] >= T_WINDOW[0]) &
            (df['time_min'] <= T_WINDOW[1]) &
            (df['radius_um'] > 0)].copy()
    return tw


def binned_profile(tw):
    bins = np.arange(0, tw['distance_um'].max() + BIN_WIDTH_UM, BIN_WIDTH_UM)
    tw = tw.copy()
    tw['bin'] = pd.cut(tw['distance_um'], bins=bins,
                       labels=bins[:-1] + BIN_WIDTH_UM / 2).astype(float)
    grp = tw.groupby('bin')['radius_um']
    prof = grp.agg(r='mean', n='count').reset_index()
    prof = prof[prof['n'] >= MIN_DROPS].sort_values('bin')
    return prof['bin'].values, prof['r'].values


def compute_zone_metric(trial_id, delta_um):
    tw = load_droplets(trial_id)
    if len(tw) < 30:
        return None

    d_vals, r_vals = binned_profile(tw)
    if d_vals is None or len(d_vals) < 5:
        return None

    # Near: first 5 valid bins
    r_near = r_vals[:5].mean() if len(r_vals) >= 5 else r_vals.mean()

    # Mid: 750-1250 um
    mid_mask = (d_vals >= MID_LO_UM) & (d_vals <= MID_HI_UM)
    if mid_mask.sum() < 2:
        return None
    r_mid = r_vals[mid_mask].mean()

    # Far: >= 1500 um
    far_mask = d_vals >= FAR_LO_UM
    if far_mask.sum() < 2:
        return None
    r_far = r_vals[far_mask].mean()

    metric = (r_far - r_near) / r_mid
    return {'r_near': r_near, 'r_mid': r_mid, 'r_far': r_far,
            'metric': metric}


# ═══════════════════════════════════════════════════════════════════════════════
#  Panel G: R(d) with shaded zones
# ═══════════════════════════════════════════════════════════════════════════════

def plot_panel_G(ax, hg):
    agar_row = hg[hg['trial_id'] == EXEMPLAR_AGAR].iloc[0]
    nacl_row = hg[hg['trial_id'] == EXEMPLAR_NACL].iloc[0]

    for tid, color, label in [(EXEMPLAR_NACL, COLORS['2:1'], 'NaCl'),
                               (EXEMPLAR_AGAR, COLORS['Agar'], 'Agar')]:
        tw = load_droplets(tid)
        dist_mm = tw['distance_um'].values / 1000.0
        radius = tw['radius_um'].values

        # Individual droplets
        ax.scatter(dist_mm, radius, s=1.5, c=color, alpha=0.15,
                   edgecolors='none', rasterized=True, zorder=1)

        # Binned mean
        d_vals, r_vals = binned_profile(tw)
        xm = d_vals / 1000.0
        first = True
        xm_plot = []
        for i, x in enumerate(xm):
            if first:
                mask_bin = (tw['distance_um'] >= d_vals[i] - BIN_WIDTH_UM/2) & \
                           (tw['distance_um'] < d_vals[i] + BIN_WIDTH_UM/2)
                x_snap = tw.loc[mask_bin, 'distance_um'].min() / 1000.0 \
                    if mask_bin.any() else x
                xm_plot.append(x_snap)
                first = False
            else:
                xm_plot.append(x)
        ax.plot(xm_plot, r_vals, 'o-', color=color, ms=3.5, lw=1.3,
                markeredgecolor='white', markeredgewidth=0.4,
                label=label, zorder=3)

    # Shaded zones
    ax.axvspan(MID_LO_UM / 1000, MID_HI_UM / 1000,
               color='#FF9800', alpha=0.10, zorder=0)
    ax.axvspan(FAR_LO_UM / 1000, MAX_DIST_MM,
               color='#2196F3', alpha=0.08, zorder=0)

    # Zone labels at top
    ax.text((MID_LO_UM + MID_HI_UM) / 2 / 1000, 68, 'Mid',
            ha='center', va='bottom', fontsize=TS, color='#E65100',
            fontstyle='italic')
    ax.text((FAR_LO_UM / 1000 + MAX_DIST_MM) / 2, 68, 'Far',
            ha='center', va='bottom', fontsize=TS, color='#1565C0',
            fontstyle='italic')

    # Near zone annotations per exemplar
    for tid, color, label in [(EXEMPLAR_NACL, COLORS['2:1'], 'NaCl'),
                               (EXEMPLAR_AGAR, COLORS['Agar'], 'Agar')]:
        tw = load_droplets(tid)
        d_vals, r_vals = binned_profile(tw)
        near_end = d_vals[4] / 1000 if len(d_vals) >= 5 else d_vals[-1] / 1000
        near_start = d_vals[0] / 1000

        # Near bracket
        y_bracket = 5 if 'NaCl' in label else 10
        ax.annotate('', xy=(near_start, y_bracket), xytext=(near_end, y_bracket),
                    arrowprops=dict(arrowstyle='<->', color=color, lw=1.0))
        ax.text((near_start + near_end) / 2, y_bracket + 2, 'Near',
                ha='center', va='bottom', fontsize=TS - 1, color=color,
                fontstyle='italic')

    # Horizontal reference lines for NaCl exemplar
    zm = compute_zone_metric(EXEMPLAR_NACL, nacl_row['delta_um'])
    if zm:
        for val, lbl, y_off in [(zm['r_near'], r'$R_{\mathrm{near}}$', -3),
                                 (zm['r_mid'],  r'$R_{\mathrm{mid}}$', 2),
                                 (zm['r_far'],  r'$R_{\mathrm{far}}$', 2)]:
            ax.axhline(val, color=COLORS['2:1'], ls=':', lw=0.5, alpha=0.5)
            ax.text(MAX_DIST_MM - 0.02, val + y_off, lbl,
                    ha='right', va='center', fontsize=TS - 1.5,
                    color=COLORS['2:1'], alpha=0.8)

    ax.set_xlabel('Distance from source (mm)', fontsize=LS, labelpad=3)
    ax.set_ylabel(r'Radius $R$ ($\mu$m)', fontsize=LS, labelpad=3)
    ax.set_xlim(0, MAX_DIST_MM)
    ax.set_ylim(0, 75)
    ax.legend(fontsize=TS - 0.5, loc='upper left', framealpha=0.9,
              handletextpad=0.3, borderpad=0.3)
    style_ax(ax)
    ax.text(-0.18, 1.05, 'E', transform=ax.transAxes,
            fontsize=PL, fontweight='bold', va='top')


# ═══════════════════════════════════════════════════════════════════════════════
#  Panel H: Zone metric vs water activity
# ═══════════════════════════════════════════════════════════════════════════════

def plot_panel_H(ax, hg):
    x_all, y_all = [], []

    for _, row in hg.iterrows():
        zm = compute_zone_metric(row['trial_id'], row['delta_um'])
        if zm is None:
            continue
        group = row['hydrogel_type']
        group_label = {'agar': 'Agar', '1:1': '1:1', '2:1': '2:1'}[group]
        x_all.append(row['one_minus_aw'])
        y_all.append(zm['metric'])

        ax.scatter(row['one_minus_aw'], zm['metric'],
                   c=COLORS[group_label], s=35, alpha=0.65,
                   edgecolors='white', linewidths=0.4, zorder=3,
                   label=group_label)

    # De-duplicate legend entries
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(),
              fontsize=TS - 0.5, loc='upper left', framealpha=0.9,
              handletextpad=0.3, borderpad=0.3)

    # Regression
    x_arr = np.array(x_all)
    y_arr = np.array(y_all)
    valid = ~np.isnan(y_arr)
    sl, ic, r, p, se = stats.linregress(x_arr[valid], y_arr[valid])
    r2 = r**2
    xfit = np.linspace(-0.02, 0.30, 100)
    ax.plot(xfit, ic + sl * xfit, 'k--', lw=1.2, alpha=0.7, zorder=2)
    ax.text(0.95, 0.08, f'$R^2$ = {r2:.3f}',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=TS)

    ax.set_xlabel(r'$(1 - a_w)$', fontsize=LS, labelpad=3)
    ax.set_ylabel(r'$(R_{\mathrm{far}} - R_{\mathrm{near}})\,/\,R_{\mathrm{mid}}$',
                  fontsize=LS, labelpad=3)
    ax.set_xlim(-0.03, 0.30)
    ax.set_xticks([0.0, 0.1, 0.2, 0.3])
    style_ax(ax)
    ax.text(-0.18, 1.05, 'H', transform=ax.transAxes,
            fontsize=PL, fontweight='bold', va='top')

    print(f"  Panel H: R\u00b2={r2:.3f}, slope={sl:.3f}, p={p:.2e}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    hg = pd.read_csv(HG_METRICS)
    print(f"Loaded {len(hg)} hydrogel trials")

    fig, (ax_g, ax_h) = plt.subplots(1, 2, figsize=(170 * MM, 75 * MM))
    fig.subplots_adjust(left=0.12, right=0.96, top=0.90, bottom=0.17,
                        wspace=0.38)

    plot_panel_G(ax_g, hg)
    plot_panel_H(ax_h, hg)

    for ext in ('.png', '.pdf', '.svg'):
        dpi = 300 if ext == '.png' else None
        fig.savefig(OUTPUT_DIR / f'panels_EH{ext}',
                    bbox_inches='tight', facecolor='white',
                    dpi=dpi if dpi else 'figure')
    plt.close(fig)
    print(f"\nSaved to {OUTPUT_DIR}/panels_EH.*")


if __name__ == '__main__':
    main()
