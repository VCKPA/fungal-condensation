#!/usr/bin/env python3
"""
step1_compute_all_metrics.py
============================
Compute δ, S, and r_break for ALL 30 trials (hydrogels + fungi).

This is the FIRST step for FigureTable.  It uses the same methodology
as FigureHGAggregate/step2 and FigureFungi/step2:
  - δ via raycast along boundary polygon (pre-computed, embedded here)
  - Broken-stick on absolute distance → r_break
  - S = R_far / R_near at absolute-distance r_break

INPUT:
    raw_data/aggregate_edt/<trial>_edt_binned_statistics.csv

OUTPUT:
    output/all_metrics.csv        — combined table (30 rows)

USAGE:
    python step1_compute_all_metrics.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ─── Paths ─────────────────────────────────────────────────────────────────────
THIS_DIR   = Path(__file__).parent
AGG_DIR    = THIS_DIR.parent / 'raw_data' / 'aggregate_edt'
OUTPUT_DIR = THIS_DIR.parent / 'output'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Trial configuration ───────────────────────────────────────────────────────
HYDROGEL_TRIALS = {
    'agar.2': 'agar',  'agar.3': 'agar',
    'agar.4': 'agar',  'agar.5': 'agar',  'agar.6': 'agar',
    '1to1.1': '1:1',   '1to1.2': '1:1',
    '1to1.3': '1:1',   '1to1.4': '1:1',   '1to1.5': '1:1',
    '2to1.1': '2:1',   '2to1.2': '2:1',
    '2to1.3': '2:1',   '2to1.4': '2:1',   '2to1.6': '2:1',
}

FUNGI_TRIALS = {
    'Green.1': 'Green', 'Green.3': 'Green', 'Green.3_new': 'Green',
    'Green.4': 'Green', 'Green.5': 'Green',
    'white.1': 'White', 'white.3': 'White', 'white.4': 'White',
    'white.5': 'White', 'white.6': 'White',
    'Black.2': 'Black', 'black.3': 'Black', 'black.4': 'Black',
    'black.new': 'Black', 'black.new2': 'Black',
}

AW = {'agar': 1.00, '1:1': 0.87, '2:1': 0.75}

# ─── Raycast delta values (µm) ──────────────────────────────────────────────
# Pre-computed from per-droplet CSVs via step1_batch_process.py.
DELTA_RAYCAST = {
    'agar.2':  77.8,  'agar.3':  92.4,  'agar.4': 163.5,
    'agar.5': 100.1,  'agar.6': 101.6,
    '1to1.1': 420.5,  '1to1.2': 286.6,  '1to1.3': 420.1,
    '1to1.4': 363.7,  '1to1.5': 462.7,
    '2to1.1': 681.1,  '2to1.2': 803.2,  '2to1.3': 933.4,
    '2to1.4': 872.9,  '2to1.6': 1005.4,
    'Green.1':   279.5,  'Green.3':   316.0,  'Green.3_new': 297.8,
    'Green.4':   285.6,  'Green.5':   311.7,
    'white.1':   198.5,  'white.3':   126.2,  'white.4':     120.0,
    'white.5':   131.6,  'white.6':   123.0,
    'Black.2':   125.4,  'black.3':    98.0,  'black.4':     111.7,
    'black.new': 135.7,  'black.new2':  78.3,
}

# ─── Analysis parameters ─────────────────────────────────────────────────────
T_WINDOW      = (14.5, 15.5)
BIN_WIDTH_UM  = 200
MIN_DROPS_BIN = 5


# ═══════════════════════════════════════════════════════════════════════════════
#  Core computation
# ═══════════════════════════════════════════════════════════════════════════════

def compute_trial_metrics(trial_id):
    """Compute S, r_break for one trial from binned statistics (absolute distance)."""
    path = AGG_DIR / f'{trial_id}_edt_binned_statistics.csv'
    if not path.exists():
        return None

    df = pd.read_csv(path)

    tw = df[(df['time_min'] >= T_WINDOW[0]) &
            (df['time_min'] <= T_WINDOW[1])].copy()
    if len(tw) < 3:
        return None

    profile = []
    for d_bin, grp in tw.groupby('distance_bin_um'):
        n_total = grp['n_droplets'].sum()
        if n_total >= MIN_DROPS_BIN:
            w = grp['n_droplets'].values
            r = grp['mean_radius_um'].values
            wmean = np.average(r, weights=w)
            profile.append({'distance_bin_um': d_bin,
                            'mean_radius_um': wmean,
                            'n_droplets': n_total})

    if len(profile) < 5:
        return None

    prof = pd.DataFrame(profile).sort_values('distance_bin_um')
    x = prof['distance_bin_um'].values
    y = prof['mean_radius_um'].values

    delta = DELTA_RAYCAST.get(trial_id, np.nan)

    best_sse, best_rb = np.inf, None
    for rb in x[1:-1]:
        x_bs = np.minimum(x, rb)
        A = np.column_stack([np.ones_like(x_bs), x_bs])
        params, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        if params[1] <= 0:
            continue
        pred = A @ params
        sse = np.sum((y - pred) ** 2)
        if sse < best_sse:
            best_sse = sse
            best_rb = rb

    if best_rb is None:
        return {'delta_um': delta}

    near = prof[prof['distance_bin_um'] <= best_rb]
    far  = prof[prof['distance_bin_um'] > best_rb]

    if len(near) >= 1 and len(far) >= 1:
        R_near = np.average(near['mean_radius_um'].values,
                            weights=near['n_droplets'].values)
        R_far  = np.average(far['mean_radius_um'].values,
                            weights=far['n_droplets'].values)
        S = R_far / R_near if R_near > 0 else np.nan
    else:
        S = np.nan

    return {'delta_um': delta, 'S': S, 'r_break_um': best_rb}


def main():
    print("Computing all metrics (broken-stick on absolute distance)...\n")

    all_rows = []

    print('--- Hydrogels ---')
    for tid, htype in HYDROGEL_TRIALS.items():
        result = compute_trial_metrics(tid)
        if result is None:
            print(f'  [SKIP] {tid}: not enough data')
            continue
        aw = AW[htype]
        row = {'trial_id': tid, 'group': htype, 'a_w': aw, **result}
        all_rows.append(row)
        S_str = f'{result.get("S", np.nan):.4f}' if not np.isnan(result.get('S', np.nan)) else '  N/A'
        rb_str = f'{result["r_break_um"]:.0f}' if result.get('r_break_um') else '  N/A'
        print(f'  {tid:<10} ({htype:<5})  delta={result["delta_um"]:6.1f}  S={S_str}  r_break={rb_str}')

    print('\n--- Fungi ---')
    for tid, species in FUNGI_TRIALS.items():
        result = compute_trial_metrics(tid)
        if result is None:
            print(f'  [SKIP] {tid}: not enough data')
            continue
        row = {'trial_id': tid, 'group': species, 'a_w': np.nan, **result}
        all_rows.append(row)
        S_str = f'{result.get("S", np.nan):.4f}' if not np.isnan(result.get('S', np.nan)) else '  N/A'
        rb_str = f'{result["r_break_um"]:.0f}' if result.get('r_break_um') else '  N/A'
        print(f'  {tid:<14} ({species:<6})  delta={result["delta_um"]:6.1f}  S={S_str}  r_break={rb_str}')

    all_df = pd.DataFrame(all_rows)
    out = OUTPUT_DIR / 'all_metrics.csv'
    all_df.to_csv(out, index=False)
    print(f'\n-> Saved {out}')

    print('\n── Group means ──')
    for grp in ['agar', '1:1', '2:1', 'Green', 'White', 'Black']:
        g = all_df[all_df['group'] == grp]
        d = g['delta_um'].dropna()
        S = g['S'].dropna()
        rb = g['r_break_um'].dropna()
        if len(d) > 0:
            print(f'  {grp:<7}: delta={d.mean():.0f}+/-{d.sem():.0f}  '
                  f'S={S.mean():.4f}+/-{S.sem():.4f}  '
                  f'r_break={rb.mean():.0f}+/-{rb.sem():.0f}  (n={len(d)})')


if __name__ == '__main__':
    main()
