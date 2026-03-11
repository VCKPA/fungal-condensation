#!/usr/bin/env python3
"""
step4_universal_table.py
========================
Merge all metrics into a single universal table (per-trial + group summary).

Combines:
  - hydrogel_metrics.csv  (15 HG trials: tanh fit metrics)
  - fungi_metrics.csv     (15 fungi trials: tanh fit metrics)
  - all_metrics.csv       (30 trials: broken-stick S, r_break)
  - rsr_finalized_metrics.csv (6 RSR leaf trials)
  - Tracking data -> dtau50/dr per trial (30 tracked trials)
  - EDT droplet data -> zone metric per trial (30 lab trials)

OUTPUT:
    output/universal_metrics.csv          (36 rows: 30 lab + 6 RSR)
    output/universal_metrics_summary.csv  (8 groups: mean +/- SEM)

USAGE:
    python step4_universal_table.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from lifelines import KaplanMeierFitter

# ── Paths ────────────────────────────────────────────────────────────────────
THIS_DIR   = Path(__file__).parent
OUTPUT_DIR = THIS_DIR.parent / 'output'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HG_METRICS  = Path('/Users/yany/Downloads/OSF/FigureHGAggregate/output/hydrogel_metrics.csv')
F_METRICS   = Path('/Users/yany/Downloads/OSF/FigureFungi/output/fungi_metrics.csv')
ALL_METRICS = OUTPUT_DIR / 'all_metrics.csv'
RSR_METRICS = Path('/Users/yany/Downloads/Fungal Pub/RSRFIGURE/rsr_finalized_metrics.csv')

TRACK_DIR   = Path('/Users/yany/Downloads/OSF/FigureHGAggregate/code/test_tracking/output')
HG_AGG_DIR  = Path('/Users/yany/Downloads/OSF/FigureHGAggregate/raw_data/aggregate_edt')
F_AGG_DIR   = Path('/Users/yany/Downloads/OSF/FigureFungi/raw_data/aggregate_edt')

# ── Trial configuration ─────────────────────────────────────────────────────
ALL_TRIALS_TRACKED = {
    'agar.2': 'Agar', 'agar.3': 'Agar', 'agar.4': 'Agar',
    'agar.5': 'Agar', 'agar.6': 'Agar',
    '1to1.1': '1:1', '1to1.2': '1:1', '1to1.3': '1:1',
    '1to1.4': '1:1', '1to1.5': '1:1',
    '2to1.1': '2:1', '2to1.2': '2:1', '2to1.3': '2:1',
    '2to1.4': '2:1', '2to1.6': '2:1',
    'Green.1': 'Green', 'Green.2': 'Green', 'Green.3': 'Green',
    'Green.3_new': 'Green', 'Green.4': 'Green', 'Green.5': 'Green',
    'white.1': 'White', 'white.3': 'White', 'white.4': 'White',
    'white.5': 'White', 'white.6': 'White',
    'Black.2': 'Black', 'black.3': 'Black', 'black.4': 'Black',
    'black.new': 'Black', 'black.new2': 'Black',
}

# ── Parameters (matching step5_universal_panels.py) ──────────────────────────
MIN_FRAMES   = 3
DIST_BIN     = 200       # um
T_SEED       = 900       # s
H_BIN_UM     = 50
H_MIN_DROP   = 10
MID_LO_UM    = 750
MID_HI_UM    = 1250
FAR_LO_UM    = 1500
H_T_WINDOW   = (14.5, 15.5)


# ═══════════════════════════════════════════════════════════════════════════════
#  Survival analysis helpers (from step5_universal_panels.py)
# ═══════════════════════════════════════════════════════════════════════════════

def load_trial(trial_id):
    path = TRACK_DIR / f'{trial_id}_track_histories.csv'
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df = df[df['n_frames'] >= MIN_FRAMES].copy()
    df['tau_fwd_min'] = (df['t_death_s'] - T_SEED) / 60.0
    df = df[df['tau_fwd_min'] > 0].copy()
    return df


def tau50_profile(trial_id, size_range=None, min_per_bin=15):
    df = load_trial(trial_id)
    if df is None:
        return None, None
    if size_range is not None:
        df = df.dropna(subset=['R_eq_seed']).copy()
        df = df[(df['R_eq_seed'] >= size_range[0]) &
                (df['R_eq_seed'] <= size_range[1])].copy()
        if len(df) < 30:
            return None, None
    d_max = df['distance_um'].max()
    bins = np.arange(0, d_max + DIST_BIN, DIST_BIN)
    df['db'] = pd.cut(df['distance_um'], bins=bins, labels=False)
    kmf = KaplanMeierFitter()
    d_vals, tau_vals = [], []
    for b in sorted(df['db'].dropna().unique()):
        sub = df[df['db'] == b]
        if len(sub) < min_per_bin:
            continue
        center = bins[int(b)] + DIST_BIN / 2
        kmf.fit(sub['tau_fwd_min'], event_observed=~sub['censored'])
        t50 = kmf.median_survival_time_
        if np.isfinite(t50):
            d_vals.append(center / 1000.0)
            tau_vals.append(t50)
    if not d_vals:
        return None, None
    return np.array(d_vals), np.array(tau_vals)


def get_iqr_band(trial_id):
    df = load_trial(trial_id)
    if df is None:
        return None
    r = df['R_eq_seed'].dropna()
    r = r[r > 0]
    if len(r) < 30:
        return None
    return (r.quantile(0.25), r.quantile(0.75))


def compute_dtau50_dr(trial_id):
    """Return (dtau50_dr_overall, dtau50_dr_sizematched) for one trial."""
    overall, sm = np.nan, np.nan

    d, t = tau50_profile(trial_id, size_range=None, min_per_bin=10)
    if d is not None and len(d) >= 3:
        overall = stats.linregress(d, t).slope

    band = get_iqr_band(trial_id)
    if band is not None:
        d2, t2 = tau50_profile(trial_id, size_range=band, min_per_bin=10)
        if d2 is not None and len(d2) >= 3:
            sm = stats.linregress(d2, t2).slope

    return overall, sm


# ═══════════════════════════════════════════════════════════════════════════════
#  Zone metric helper (from step5_universal_panels.py)
# ═══════════════════════════════════════════════════════════════════════════════

def _load_edt_droplets(trial_id):
    for d in (HG_AGG_DIR, F_AGG_DIR):
        p = d / f'{trial_id}_edt_droplets.csv'
        if p.exists():
            df = pd.read_csv(p)
            tw = df[(df['time_min'] >= H_T_WINDOW[0]) &
                    (df['time_min'] <= H_T_WINDOW[1]) &
                    (df['radius_um'] > 0)].copy()
            return tw
    return None


def _size_gradient(trial_id):
    tw = _load_edt_droplets(trial_id)
    if tw is None or len(tw) < 30:
        return np.nan, 0

    bins = np.arange(0, tw['distance_um'].max() + H_BIN_UM, H_BIN_UM)
    tw = tw.copy()
    tw['bin'] = pd.cut(tw['distance_um'], bins=bins,
                       labels=bins[:-1] + H_BIN_UM / 2).astype(float)
    grp = tw.groupby('bin')['radius_um']
    prof = grp.agg(r='mean', n='count').reset_index()
    prof = prof[prof['n'] >= H_MIN_DROP].sort_values('bin')
    if len(prof) < 5:
        return np.nan, len(tw)

    d_vals = prof['bin'].values
    r_vals = prof['r'].values

    r_near = r_vals[:5].mean()

    mid_mask = (d_vals >= MID_LO_UM) & (d_vals <= MID_HI_UM)
    if mid_mask.sum() < 2:
        return np.nan, len(tw)
    r_mid = r_vals[mid_mask].mean()

    far_mask = d_vals >= FAR_LO_UM
    if far_mask.sum() < 2:
        return np.nan, len(tw)
    r_far = r_vals[far_mask].mean()

    return (r_far - r_near) / r_mid, len(tw)


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print('=== Building universal metrics table ===\n')

    # ── 1. Load tanh metrics ─────────────────────────────────────────────────
    hg = pd.read_csv(HG_METRICS)
    hg = hg.rename(columns={'hydrogel_type': 'group'})
    hg['group'] = hg['group'].replace({'agar': 'Agar'})
    hg['system'] = 'Hydrogel'

    fm = pd.read_csv(F_METRICS)
    fm = fm.rename(columns={'species': 'group'})
    fm['system'] = 'Fungi'
    fm['a_w'] = np.nan

    tanh_cols = ['trial_id', 'group', 'system', 'a_w',
                 'delta_um', 'max_slope', 'alpha', 'y_near', 'y_far',
                 'r0_um', 'transition_width_um']

    # Ensure a_w column exists in both
    if 'a_w' not in hg.columns:
        hg['a_w'] = hg['one_minus_aw'].apply(lambda x: round(1 - x, 2))

    tanh_df = pd.concat([hg[tanh_cols], fm[tanh_cols]], ignore_index=True)

    # ── 2. Load broken-stick metrics ─────────────────────────────────────────
    bs = pd.read_csv(ALL_METRICS)
    bs_cols = ['trial_id', 'S', 'r_break_um']
    bs_sub = bs[bs_cols].copy()

    # ── 3. Merge tanh + broken-stick ─────────────────────────────────────────
    lab = tanh_df.merge(bs_sub, on='trial_id', how='left')

    # ── 4. Compute zone metric and n_droplets for all 30 lab trials ──────────
    print('Computing zone metrics (size gradient)...')
    zone_vals = {}
    n_drops_vals = {}
    all_lab_ids = lab['trial_id'].tolist()
    for tid in all_lab_ids:
        zg, nd = _size_gradient(tid)
        zone_vals[tid] = zg
        n_drops_vals[tid] = nd
        if not np.isnan(zg):
            print(f'  {tid:14s}: zone_metric={zg:+.3f}  n={nd}')
        else:
            print(f'  {tid:14s}: zone_metric=NaN  n={nd}')

    lab['zone_metric'] = lab['trial_id'].map(zone_vals)
    lab['n_droplets'] = lab['trial_id'].map(n_drops_vals)

    # ── 5. Compute dtau50/dr for tracked trials ──────────────────────────────
    print('\nComputing survival gradients (dtau50/dr)...')

    # Build delta map for Green.2 (not in tanh metrics)
    delta_map = dict(zip(lab['trial_id'], lab['delta_um']))
    delta_map.setdefault('Green.2', 231.0)

    dtau_overall = {}
    dtau_sm = {}
    for tid in ALL_TRIALS_TRACKED:
        ov, sm = compute_dtau50_dr(tid)
        dtau_overall[tid] = ov
        dtau_sm[tid] = sm
        ov_s = f'{ov:.3f}' if np.isfinite(ov) else 'NaN'
        sm_s = f'{sm:.3f}' if np.isfinite(sm) else 'NaN'
        print(f'  {tid:14s}: dtau50/dr={ov_s}  size-matched={sm_s}')

    lab['dtau50_dr'] = lab['trial_id'].map(dtau_overall)
    lab['dtau50_dr_sizematched'] = lab['trial_id'].map(dtau_sm)

    # RSR-only columns are NaN for lab trials
    lab['dR_dr_um_per_mm'] = np.nan
    lab['pearson_r'] = np.nan
    lab['pearson_p'] = np.nan

    # ── 6. Load RSR metrics ──────────────────────────────────────────────────
    print('\nLoading RSR leaf metrics...')
    rsr = pd.read_csv(RSR_METRICS)
    rsr_rows = []
    for _, row in rsr.iterrows():
        rsr_rows.append({
            'trial_id': row['sample'],
            'group': row['condition'],
            'system': 'Leaf',
            'a_w': np.nan,
            'n_droplets': row['n_droplets'],
            'delta_um': np.nan,
            'max_slope': np.nan,
            'alpha': np.nan,
            'y_near': np.nan,
            'y_far': np.nan,
            'r0_um': np.nan,
            'transition_width_um': np.nan,
            'S': np.nan,
            'r_break_um': np.nan,
            'zone_metric': np.nan,
            'dR_dr_um_per_mm': row['dR_dr_um_per_mm'],
            'pearson_r': row['pearson_r'],
            'pearson_p': row['pearson_p'],
            'dtau50_dr': np.nan,
            'dtau50_dr_sizematched': np.nan,
        })
    rsr_df = pd.DataFrame(rsr_rows)

    # ── 7. Combine all ───────────────────────────────────────────────────────
    final_cols = ['trial_id', 'group', 'system', 'a_w', 'n_droplets',
                  'delta_um', 'max_slope', 'alpha', 'y_near', 'y_far',
                  'r0_um', 'transition_width_um', 'S', 'r_break_um',
                  'zone_metric', 'dR_dr_um_per_mm', 'pearson_r', 'pearson_p',
                  'dtau50_dr', 'dtau50_dr_sizematched']

    universal = pd.concat([lab[final_cols], rsr_df[final_cols]], ignore_index=True)

    out_path = OUTPUT_DIR / 'universal_metrics.csv'
    universal.to_csv(out_path, index=False)
    print(f'\n-> Saved {out_path}  ({len(universal)} rows)')

    # ── 8. Group summary ─────────────────────────────────────────────────────
    print('\n=== Group Summary (mean +/- SEM) ===')
    group_order = ['Agar', '1:1', '2:1', 'Green', 'White', 'Black',
                   'Healthy', 'Diseased']

    summary_rows = []
    numeric_cols = ['delta_um', 'max_slope', 'alpha', 'y_near', 'y_far',
                    'r0_um', 'transition_width_um', 'S', 'r_break_um',
                    'zone_metric', 'dR_dr_um_per_mm', 'pearson_r',
                    'dtau50_dr', 'dtau50_dr_sizematched']

    for grp in group_order:
        g = universal[universal['group'] == grp]
        if len(g) == 0:
            continue
        row = {'group': grp, 'n': len(g),
               'system': g['system'].iloc[0]}
        for col in numeric_cols:
            vals = g[col].dropna()
            if len(vals) > 0:
                row[f'{col}_mean'] = vals.mean()
                row[f'{col}_sem'] = vals.sem() if len(vals) > 1 else np.nan
            else:
                row[f'{col}_mean'] = np.nan
                row[f'{col}_sem'] = np.nan
        summary_rows.append(row)

        d = row.get('delta_um_mean', np.nan)
        s = row.get('S_mean', np.nan)
        d_s = f'{d:.0f}' if np.isfinite(d) else 'N/A'
        s_s = f'{s:.4f}' if np.isfinite(s) else 'N/A'
        print(f'  {grp:<10}: n={len(g)}  delta={d_s}  S={s_s}')

    summary_df = pd.DataFrame(summary_rows)
    sum_path = OUTPUT_DIR / 'universal_metrics_summary.csv'
    summary_df.to_csv(sum_path, index=False)
    print(f'\n-> Saved {sum_path}')

    # ── 9. Verification ──────────────────────────────────────────────────────
    print('\n=== Verification ===')
    print(f'Total rows: {len(universal)} (expected 36)')
    lab_rows = universal[universal['system'] != 'Leaf']
    rsr_rows_check = universal[universal['system'] == 'Leaf']
    print(f'Lab rows: {len(lab_rows)} (expected 30)')
    print(f'RSR rows: {len(rsr_rows_check)} (expected 6)')

    has_delta = lab_rows['delta_um'].notna().sum()
    has_slope = lab_rows['max_slope'].notna().sum()
    has_S = lab_rows['S'].notna().sum()
    print(f'Lab with delta: {has_delta}/30')
    print(f'Lab with max_slope: {has_slope}/30')
    print(f'Lab with S: {has_S}/30')

    has_dtau = universal['dtau50_dr'].notna().sum()
    print(f'Trials with dtau50/dr: {has_dtau} (expected 30)')

    has_rsr = rsr_rows_check['dR_dr_um_per_mm'].notna().sum()
    print(f'RSR with dR/dr: {has_rsr}/6')


if __name__ == '__main__':
    main()
