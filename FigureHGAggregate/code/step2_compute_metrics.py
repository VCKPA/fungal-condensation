#!/usr/bin/env python3
"""
step2_compute_metrics.py
========================
Compute condensation metrics for each hydrogel trial:

  1. delta (dry-zone width)     — from raycast (computed by step1_batch_process.py)
  2. max_slope (dR/dd|max)      — peak gradient of R(d) at inflection point

The hyperbolic-tangent model
    R(d) = (y_near + y_far)/2 + (y_far - y_near)/2 · tanh(α·(d − r0))
is fitted to 100 µm-binned **mean** radial profiles at t = 14.5–15.5 min,
where d = distance_um (absolute distance from the source boundary).

The primary steepness metric is max_slope = α·(y_far − y_near)/2, the
derivative dR/dd evaluated at d = r0 (the inflection point).  This is the
dimensional gradient (µm radius / µm distance) and correctly accounts for
both steepness and transition amplitude.  Secondary observables r0 and
transition_width = 2/α are also reported.

Delta is independently measured via raycast along the source boundary
polygon (see step1_batch_process.py).  Pre-computed raycast values are
embedded below; to update them, re-run step1 with per-droplet CSVs.

INPUT:
    raw_data/aggregate_edt/<trial>_edt_droplets.csv

OUTPUT:
    output/hydrogel_metrics.csv

USAGE:
    python3 step2_compute_metrics.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import curve_fit

# ─── Paths ─────────────────────────────────────────────────────────────────────
THIS_DIR   = Path(__file__).parent
AGG_DIR    = THIS_DIR.parent / 'raw_data' / 'aggregate_edt'
OUTPUT_DIR = THIS_DIR.parent / 'output'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Trial configuration ───────────────────────────────────────────────────────
TRIALS = {
    'agar.2': 'agar',  'agar.3': 'agar',
    'agar.4': 'agar',  'agar.5': 'agar',  'agar.6': 'agar',
    '1to1.1': '1:1',   '1to1.2': '1:1',
    '1to1.3': '1:1',   '1to1.4': '1:1',   '1to1.5': '1:1',
    '2to1.1': '2:1',   '2to1.2': '2:1',
    '2to1.3': '2:1',   '2to1.4': '2:1',   '2to1.6': '2:1',
}

AW = {'agar': 1.00, '1:1': 0.87, '2:1': 0.75}

# ─── Raycast delta values (µm) ──────────────────────────────────────────────
# Pre-computed from per-droplet CSVs via step1_batch_process.py.
# Method: sample 100 points along source boundary, find nearest droplet
# to each, take mean of per-ray minimum distances.
DELTA_RAYCAST = {
    'agar.2':  77.8,  'agar.3':  92.4,  'agar.4': 163.5,
    'agar.5': 100.1,  'agar.6': 101.6,
    '1to1.1': 420.5,  '1to1.2': 286.6,  '1to1.3': 420.1,
    '1to1.4': 363.7,  '1to1.5': 462.7,
    '2to1.1': 681.1,  '2to1.2': 803.2,  '2to1.3': 933.4,
    '2to1.4': 872.9,  '2to1.6': 1005.4,
}

# ─── Analysis parameters ──────────────────────────────────────────────────────
T_WINDOW      = (14.5, 15.5)
BIN_WIDTH_UM  = 100
MIN_DROPS_BIN = 5


# ═══════════════════════════════════════════════════════════════════════════════
#  Core computation
# ═══════════════════════════════════════════════════════════════════════════════

def compute_trial_metrics(trial_id):
    """Compute delta and max_slope for one trial via tanh fit to per-droplet data."""
    path = AGG_DIR / f'{trial_id}_edt_droplets.csv'
    if not path.exists():
        return None

    df = pd.read_csv(path)

    # ── Analysis window ──
    tw = df[(df['time_min'] >= T_WINDOW[0]) &
            (df['time_min'] <= T_WINDOW[1])].copy()
    if len(tw) < 50:
        return None

    # ── Delta from raycast lookup ──
    delta = DELTA_RAYCAST.get(trial_id, np.nan)

    # ── Bin per-droplet data at BIN_WIDTH_UM resolution ──
    bins = np.arange(0, tw['distance_um'].max() + BIN_WIDTH_UM, BIN_WIDTH_UM)
    tw['distance_bin_um'] = pd.cut(
        tw['distance_um'], bins=bins,
        labels=bins[:-1] + BIN_WIDTH_UM / 2).astype(float)
    grouped = tw.groupby('distance_bin_um')['radius_um']
    prof = grouped.agg(mean_radius_um='mean', n_droplets='count').reset_index()
    prof = prof[prof['n_droplets'] >= MIN_DROPS_BIN].sort_values('distance_bin_um')

    if len(prof) < 5:
        return None

    x = prof['distance_bin_um'].values
    y = prof['mean_radius_um'].values

    # ── Tanh fit: R(d) = (y_near+y_far)/2 + (y_far-y_near)/2 * tanh(alpha*(d-r0)) ──
    def tanh_model(d, y_near, y_far, alpha, r0):
        return (y_near + y_far) / 2 + (y_far - y_near) / 2 * np.tanh(alpha * (d - r0))

    y_floor = max(1.0, y.min() * 0.3)
    p0 = [y.min(), y.max(), 0.002, np.median(x)]
    bounds = ([y_floor, y_floor, 1e-6, x.min()],
              [np.inf, np.inf, 1.0, x.max()])

    try:
        popt, pcov = curve_fit(tanh_model, x, y, p0=p0, bounds=bounds, maxfev=10000)
        y_near, y_far, alpha, r0 = popt
    except RuntimeError:
        return {'delta_um': delta}

    max_slope = alpha * (y_far - y_near) / 2.0
    transition_width = 2.0 / alpha if alpha > 0 else np.nan

    # r0 undefined when transition spans observation window
    obs_range = x.max() - x.min()
    r0_out = np.nan if transition_width > 0.75 * obs_range else r0

    return {
        'delta_um':            delta,
        'max_slope':           max_slope,
        'r0_um':               r0_out,
        'alpha':               alpha,
        'y_near':              y_near,
        'y_far':               y_far,
        'transition_width_um': transition_width,
    }


def main():
    print("Computing hydrogel metrics (tanh fit on absolute distance)")
    print(f'  t = {T_WINDOW[0]}–{T_WINDOW[1]} min, bin = {BIN_WIDTH_UM} µm\n')

    rows = []

    for tid, htype in TRIALS.items():
        result = compute_trial_metrics(tid)
        if result is None:
            print(f'  [SKIP] {tid}: not enough data')
            continue

        aw = AW[htype]
        row = {
            'trial_id': tid, 'hydrogel_type': htype,
            'a_w': aw, 'one_minus_aw': round(1 - aw, 2),
            **result,
        }
        rows.append(row)

        ms_str = f'{result["max_slope"]:.5f}' if not np.isnan(result.get('max_slope', np.nan)) else '  N/A'

        print(f'  {tid:<10} ({htype:<5})  '
              f'delta={result["delta_um"]:6.1f}  '
              f'max_slope={ms_str}')

    df = pd.DataFrame(rows)
    out = OUTPUT_DIR / 'hydrogel_metrics.csv'
    df.to_csv(out, index=False)
    print(f'\n-> saved {out}')

    # ── Group summary ──
    print('\n── Group means ──')
    for htype in ['agar', '1:1', '2:1']:
        g = df[df['hydrogel_type'] == htype]
        n = len(g)
        d  = g['delta_um']
        ms = g['max_slope'].dropna()
        print(f'  {htype:<5}: n={n}  '
              f'delta={d.mean():.0f}+/-{d.sem():.0f}  '
              f'max_slope={ms.mean():.5f}+/-{ms.sem():.5f}')


if __name__ == '__main__':
    main()
