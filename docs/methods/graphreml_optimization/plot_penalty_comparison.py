"""Render validated primary and prespecified secondary penalty comparisons."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parent / 'penalty_comparison'
SOURCE = ROOT / 'all_annotation_penalty_changes.tsv'
rows = list(csv.DictReader(SOURCE.open(), delimiter='\t'))
PRIMARY = ['Coding', 'Conserved', 'DHS', 'Enhancer', 'Promoter', 'Repressed']
SECONDARY = [
    ('Human_Promoter_Villar_ExAC.flanking.500', 'ExAC promoter\nflanks'),
    ('SuperEnhancer_Hnisz.flanking.500', 'Superenhancer\nflanks'),
    ('Human_Enhancer_Villar.flanking.500', 'Human enhancer\nflanks'),
]
TRAITS = ['weight', 'hemoglobin']
selected = []
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                     'axes.labelsize': 9, 'xtick.labelsize': 8,
                     'ytick.labelsize': 9, 'pdf.fonttype': 42, 'ps.fonttype': 42})
fig, axes = plt.subplots(2, 2, figsize=(8.8, 5.6), sharey='row',
                         gridspec_kw={'height_ratios': [1.15, 1]})
for j, trait in enumerate(TRAITS):
    trait_rows = [r for r in rows if r['trait'] == trait]
    if len(trait_rows) != 97:
        raise ValueError('Expected all 97 annotation rows per trait')
    for r in trait_rows:
        if r['status_off'] != 'stationary' or r['status_on'] != 'stationary':
            raise ValueError('Plot requires stationary paired fits')
    main = [next(r for r in trait_rows if r['published_label'] == label) for label in PRIMARY]
    sec = [next(r for r in trait_rows if r['annotation'] == column) for column, _ in SECONDARY]
    if any(r['annotation_interpretation'] != 'verified_binary' for r in sec):
        raise ValueError('Secondary annotation domain is unverified')
    for r in main + sec:
        selected.append(dict(r, display_group='primary' if r in main else 'secondary'))
    ax = axes[0, j]
    delta = np.array([float(r['relative_change_percent']) for r in main])
    ax.axvline(0, color='#BBBBBB', linewidth=.8, zorder=0)
    ax.hlines(np.arange(6), 0, delta, color='#24789A', linewidth=1.5)
    ax.scatter(delta, np.arange(6), s=26, color='#24789A', zorder=3)
    ax.set(yticks=np.arange(6), yticklabels=PRIMARY, ylim=(5.5, -.5),
           xlim=(-3.3, 3.3), xticks=[-3, 0, 3], xlabel='Enrichment change (%)')
    ax.set_title(trait.capitalize(), pad=11, fontsize=11)
    ax = axes[1, j]
    off = np.array([float(r['enrichment_off']) for r in sec])
    on = np.array([float(r['enrichment_on']) for r in sec])
    if np.any(off <= 0) or np.any(on <= 0):
        raise ValueError('Log-scale display requires positive enrichments')
    for y, a, b in zip(range(3), off, on):
        ax.plot([a, b], [y, y], color='#AAAAAA', linewidth=1.4, zorder=1)
    ax.scatter(off, np.arange(3), s=30, color='#777777', marker='o', zorder=3)
    ax.scatter(on, np.arange(3), s=32, color='#24789A', marker='D', zorder=3)
    ax.axvline(1, color='#BBBBBB', linewidth=.8, linestyle='--', zorder=0)
    ax.set_xscale('log')
    ax.set(yticks=np.arange(3), yticklabels=[label for _, label in SECONDARY],
           ylim=(2.6, -.6), xlim=(.008, 10), xticks=[.01, .1, 1, 10],
           xticklabels=['0.01', '0.1', '1', '10'], xlabel='Enrichment')
for label, ax in zip('ABCD', axes.flat):
    ax.text(-.075, 1.025, label, transform=ax.transAxes, weight='bold', fontsize=11)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(axis='y', length=0)
fig.legend(handles=[Line2D([], [], linestyle='', marker='o', color='#777777', label='Penalty off'),
                    Line2D([], [], linestyle='', marker='D', color='#24789A', label='Penalty on')],
           loc='lower center', bbox_to_anchor=(.56, -.005), ncol=2, frameon=False)
fig.subplots_adjust(left=.21, right=.98, bottom=.14, top=.93, hspace=.57, wspace=.2)
for ext in ['pdf', 'svg', 'png']:
    fig.savefig(ROOT / f'penalty_enrichment_comparison.{ext}', dpi=180, facecolor='white')
plt.close(fig)
source = ROOT / 'penalty_figure_source.tsv'
with source.open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(selected[0]), delimiter='\t')
    writer.writeheader(); writer.writerows(selected)
def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()
(ROOT / 'penalty_figure_manifest.json').write_text(json.dumps(dict(
    source=str(SOURCE), source_sha256=digest(SOURCE), script_sha256=digest(__file__),
    figure_source_sha256=digest(source),
    secondary_selection='Three binary flank annotations selected for large Weight changes before the Hemoglobin penalty endpoint; all three shown for both traits.',
    secondary_columns=[x[0] for x in SECONDARY],
    interpretation='Point estimates; both fits stationary. Penalty uncertainty calibration not evaluated. Secondary axis is logarithmic.',
), indent=2)+'\n')
print('Rendered penalty comparison and saved 18 source rows.')
