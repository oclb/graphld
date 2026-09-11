"""Render the validated historical versus AI Weight comparison and its source table."""
from pathlib import Path
import csv
import hashlib
import json
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

default_root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source', type=Path, default=default_root / 'matched_start_enrichment_estimates.tsv')
parser.add_argument('--output', type=Path, default=default_root)
parser.add_argument('--variant', choices=['initial', 'safeguarded'], default='initial')
args = parser.parse_args()
ROOT = args.output.resolve()
SOURCE = args.source.resolve()
ANNOTATIONS = ['Coding', 'Conserved', 'DHS', 'Enhancer', 'Promoter', 'Repressed']
ROLES = (['historical', 'new_ai'] if args.variant == 'initial' else
         ['historical', 'new_safeguarded_ai'])
LABELS = ['Old method', 'AI']
COLORS = ['#777777', '#24789A']
rows = list(csv.DictReader(SOURCE.open(), delimiter='\t'))
data = {}
for role in ROLES:
    selected = [r for r in rows if r['trait'] == 'weight' and r['role'] == role]
    if len(selected) != 6 or {r['annotation'] for r in selected} != set(ANNOTATIONS):
        raise ValueError(f'Incomplete or duplicate six-annotation result for {role}')
    if any(r['matched_start'] != 'True' or float(r['penalty_weight']) != 0 for r in selected):
        raise ValueError('Expected matched neutral starts and penalty off')
    for field in ['run', 'process_seconds', 'native_status', 'precise_score_criterion']:
        if len({r[field] for r in selected}) != 1:
            raise ValueError(f'Inconsistent repeated fit field {field}')
    data[role] = {r['annotation']: r for r in selected}

def first(role):
    return data[role][ANNOTATIONS[0]]

source_rows = []
for role, label in zip(ROLES, LABELS):
    for name in ANNOTATIONS:
        r = data[role][name]
        old = float(data['historical'][name]['enrichment'])
        enrichment = float(r['enrichment'])
        delta = 100 * (enrichment / old - 1)
        if not np.isfinite(enrichment) or not np.isfinite(delta):
            raise ValueError('Nonfinite enrichment')
        source_rows.append(dict(method=label, role=role, annotation=name,
                                enrichment=enrichment, change_percent=delta,
                                fit_minutes=float(r['process_seconds']) / 60,
                                iterations=r['iterations'], status=r['native_status'],
                                score_criterion=r['precise_score_criterion'],
                                independent_audit_minutes=float(r['independent_audit_seconds'] or 0) / 60))
ROOT.mkdir(parents=True, exist_ok=True)
out = ROOT / 'matched_optimizer_figure_source.tsv'
with out.open('w') as f:
    writer = csv.DictWriter(f, fieldnames=list(source_rows[0]), delimiter='\t')
    writer.writeheader()
    writer.writerows(source_rows)

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                     'axes.labelsize': 9, 'xtick.labelsize': 8,
                     'ytick.labelsize': 9, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), gridspec_kw={'width_ratios': [1, 1.12]})
fig.subplots_adjust(left=.16, right=.985, bottom=.22, top=.91, wspace=.66)
ax = axes[0]
minutes = [float(first(role)['process_seconds']) / 60 for role in ROLES]
ax.barh(np.arange(len(ROLES)), minutes, height=.52, color=COLORS)
ax.set_yticks(np.arange(len(ROLES)), LABELS)
ax.invert_yaxis()
ax.set_xlim(0, max(minutes) * 1.34)
ax.set_xlabel('Fit runtime (minutes)')
for i, (role, value) in enumerate(zip(ROLES, minutes)):
    converged = first(role)['native_converged'] == 'True'
    ax.text(value + 2, i, f'{value:.1f}' + ('' if converged else '*'), va='center', fontsize=8)
ax.text(0, -.24, '* Did not converge', transform=ax.transAxes, fontsize=8)
ax = axes[1]
y = np.arange(6)
ax.axvline(0, color=COLORS[0], linewidth=.8, zorder=0)
for role, label, color, offset, marker in zip(ROLES[1:], LABELS[1:], COLORS[1:], [0.], ['o']):
    changes = [next(r['change_percent'] for r in source_rows if r['role'] == role and r['annotation'] == a) for a in ANNOTATIONS]
    ax.scatter(changes, y + offset, color=color, marker=marker, s=25, label=label, zorder=3)
ax.set_yticks(y, ANNOTATIONS)
ax.invert_yaxis()
ax.set_xlabel('Enrichment change (%)')
ax.margins(x=.15, y=.13)
ax.legend(frameon=False, loc='lower left', bbox_to_anchor=(-.08, 1.01), ncol=2,
          fontsize=8, handletextpad=.25, columnspacing=.7, borderaxespad=0)
for panel, ax in zip('ab', axes):
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(length=3)
    ax.text(-.12, 1.08, panel, transform=ax.transAxes, fontweight='bold', fontsize=11)
for ext in ['pdf', 'svg', 'png']:
    fig.savefig(ROOT / f'matched_optimizer_comparison.{ext}', dpi=200)
plt.close(fig)

table = ['\\begin{tabular}{lrr}', '\\toprule',
         'Annotation & Old method & AI \\\\', '\\midrule']
for a in ANNOTATIONS:
    values = [float(data[role][a]['enrichment']) for role in ROLES]
    table.append(a + ' & ' + ' & '.join(f'{x:.3f}' for x in values) + r' \\')
table += [r'\bottomrule', r'\end{tabular}']
(ROOT / 'matched_optimizer_enrichment_table.tex').write_text('\n'.join(table) + '\n')
manifest = dict(input=str(SOURCE), input_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                plot_source_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),
                trait='weight', comparison='two matched-start unpenalized fits',
                variant=args.variant,
                endpoint_statuses={role: first(role)['native_status'] for role in ROLES},
                uncertainty='Point estimates only; intervals and calibration are not compared.',
                timing='Whole subprocess; independent historical audit excluded; single sequential runs.',
                outputs=['matched_optimizer_comparison.' + ext for ext in ['pdf', 'svg', 'png']])
(ROOT / 'matched_optimizer_figure_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print('Rendered historical versus AI comparison and wrote source data, table, and provenance.')
