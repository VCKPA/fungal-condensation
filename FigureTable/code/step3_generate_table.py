#!/usr/bin/env python3
"""
step3_generate_table.py
=======================
Merge all metrics and generate the paper table.

Columns:
    Group | n | δ (µm) | S | r_break (µm)

INPUT:
    output/all_metrics.csv        (from step1_compute_all_metrics.py)

OUTPUT:
    output/Table_metrics.csv      — machine-readable table
    output/Table_metrics.tex      — LaTeX table (booktabs)
    output/Table_metrics.png      — rendered table figure

USAGE:
    python step3_generate_table.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# ─── Paths ─────────────────────────────────────────────────────────────────────
THIS_DIR   = Path(__file__).parent
OUTPUT_DIR = THIS_DIR.parent / 'output'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Group display order and labels
GROUP_ORDER = ['agar', '1:1', '2:1', 'Green', 'White', 'Black']
GROUP_LABELS = {
    'agar':  'Agar ($a_w$=1.00)',
    '1:1':   '1:1 NaCl ($a_w$=0.87)',
    '2:1':   '2:1 NaCl ($a_w$=0.75)',
    'Green': 'Green fungi',
    'White': 'White fungi',
    'Black': 'Black fungi',
}
GROUP_LABELS_PLAIN = {
    'agar':  'Agar (aw=1.00)',
    '1:1':   '1:1 NaCl (aw=0.87)',
    '2:1':   '2:1 NaCl (aw=0.75)',
    'Green': 'Green fungi',
    'White': 'White fungi',
    'Black': 'Black fungi',
}


def fmt(mean, sem, decimals=2):
    """Format mean +/- SEM for table cell."""
    if np.isnan(mean):
        return '—'
    return f'{mean:.{decimals}f} \u00b1 {sem:.{decimals}f}'


def fmt_delta(mean, sem):
    """Format δ in µm, 0 decimal places."""
    if np.isnan(mean):
        return '—'
    return f'{mean:.0f} \u00b1 {sem:.0f}'


def main():
    # ── Load ───────────────────────────────────────────────────────────────────
    metrics_path = OUTPUT_DIR / 'all_metrics.csv'

    if not metrics_path.exists():
        print(f'[ERROR] {metrics_path} not found. Run step1_compute_all_metrics.py first.')
        return

    metrics = pd.read_csv(metrics_path)

    # ── Build summary table ────────────────────────────────────────────────────
    table_rows = []
    for grp in GROUP_ORDER:
        g = metrics[metrics['group'] == grp]

        n      = len(g)
        delta  = g['delta_um'].dropna()
        S      = g['S'].dropna()
        rb     = g['r_break_um'].dropna()

        row = {
            'Group':          GROUP_LABELS_PLAIN[grp],
            'n':              n,
            'delta_mean':     delta.mean()  if len(delta) > 0 else np.nan,
            'delta_sem':      delta.sem()   if len(delta) > 0 else np.nan,
            'S_mean':         S.mean()      if len(S) > 0 else np.nan,
            'S_sem':          S.sem()       if len(S) > 0 else np.nan,
            'rbreak_mean':    rb.mean()     if len(rb) > 0 else np.nan,
            'rbreak_sem':     rb.sem()      if len(rb) > 0 else np.nan,
        }
        table_rows.append(row)

    table_df = pd.DataFrame(table_rows)
    table_df.to_csv(OUTPUT_DIR / 'Table_metrics.csv', index=False)
    print(f'-> Saved Table_metrics.csv')

    # ── LaTeX table ────────────────────────────────────────────────────────────
    latex = [
        r'\begin{table}[h]',
        r'\centering',
        r'\caption{Summary of condensation metrics for hydrogel and fungal trials.}',
        r'\label{tab:metrics}',
        r'\begin{tabular}{lc r@{$\pm$}l r@{$\pm$}l r@{$\pm$}l}',
        r'\toprule',
        r'Group & $n$ & \multicolumn{2}{c}{$\delta$ (\textmu m)} '
        r'& \multicolumn{2}{c}{$S$} '
        r'& \multicolumn{2}{c}{$r_{\mathrm{break}}$ (\textmu m)} \\',
        r'\midrule',
    ]

    for grp, row in zip(GROUP_ORDER, table_rows):
        lbl = GROUP_LABELS[grp].replace('_', r'\_')
        n = row['n']

        delta_str = f"{row['delta_mean']:.0f} & {row['delta_sem']:.0f}" if not np.isnan(row['delta_mean']) else r'\multicolumn{2}{c}{---}'
        S_str     = f"{row['S_mean']:.3f} & {row['S_sem']:.3f}"         if not np.isnan(row['S_mean'])     else r'\multicolumn{2}{c}{---}'
        rb_str    = f"{row['rbreak_mean']:.0f} & {row['rbreak_sem']:.0f}" if not np.isnan(row['rbreak_mean']) else r'\multicolumn{2}{c}{---}'

        latex.append(f'{lbl} & {n} & {delta_str} & {S_str} & {rb_str} \\\\')

    latex += [
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{table}',
    ]

    latex_path = OUTPUT_DIR / 'Table_metrics.tex'
    with open(latex_path, 'w') as f:
        f.write('\n'.join(latex))
    print(f'-> Saved Table_metrics.tex')

    # ── Matplotlib table figure ────────────────────────────────────────────────
    col_labels = ['Group', 'n', '\u03b4 (\u00b5m)', 'S', 'r_break (\u00b5m)']
    table_data = []
    for grp, row in zip(GROUP_ORDER, table_rows):
        table_data.append([
            GROUP_LABELS_PLAIN[grp],
            str(row['n']),
            fmt_delta(row['delta_mean'], row['delta_sem']),
            fmt(row['S_mean'], row['S_sem'], 3),
            fmt_delta(row['rbreak_mean'], row['rbreak_sem']),
        ])

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis('off')
    t = ax.table(cellText=table_data, colLabels=col_labels,
                 cellLoc='center', loc='center')
    t.auto_set_font_size(False)
    t.set_fontsize(9)
    t.scale(1, 1.6)

    # Header styling
    for j in range(len(col_labels)):
        t[0, j].set_facecolor('#2c3e50')
        t[0, j].set_text_props(color='white', fontweight='bold')

    # Row shading — separator between hydrogels and fungi
    for i in range(1, len(table_data) + 1):
        grp = GROUP_ORDER[i - 1]
        if grp in ['Green', 'White', 'Black']:
            for j in range(len(col_labels)):
                t[i, j].set_facecolor('#f0f8f0')

    fig.tight_layout()
    out_png = OUTPUT_DIR / 'Table_metrics.png'
    fig.savefig(out_png, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'-> Saved Table_metrics.png')

    # Print text table
    print('\n── Table ──')
    print(f"{'Group':<26} {'n':>3}  {'\u03b4 (\u00b5m)':>12}  {'S':>14}  {'r_break':>14}")
    print('-' * 75)
    for grp, row in zip(GROUP_ORDER, table_rows):
        print(f"{GROUP_LABELS_PLAIN[grp]:<26} {row['n']:>3}  "
              f"{fmt_delta(row['delta_mean'], row['delta_sem']):>12}  "
              f"{fmt(row['S_mean'], row['S_sem'], 3):>14}  "
              f"{fmt_delta(row['rbreak_mean'], row['rbreak_sem']):>14}")


if __name__ == '__main__':
    main()
