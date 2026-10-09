"""Summarize the four controlled runs against a constant-label baseline."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path('/e/scratch/reformo/schuhmann1_moss/whisper_score_regression')
RUNS = Path('/e/project1/reformo/schuhmann1/whisper_score_regression/runs')
CONDITIONS = [('tiny', 'S10'), ('base', 'S10'), ('tiny', 'S9'), ('base', 'S9')]


def baseline(meta):
    with np.load(ROOT / 'val.npz') as val:
        y = val['y'].astype(np.float64)
    mean = np.asarray(meta['mean'])
    std = np.asarray(meta['std'])
    z = (y - mean) / std
    abs_z = np.abs(z)
    huber = np.where(abs_z < 1, 0.5 * z * z, abs_z - 0.5)
    result = {'huber_z_macro': float(np.nanmean(huber))}
    for name, (a, b) in meta['target_groups'].items():
        result[name] = {'huber_z': float(np.nanmean(huber[:, a:b])),
                        'mae': float(np.nanmean(np.abs(y[:, a:b] - mean[a:b])))}
    return result


def main():
    meta = json.loads((ROOT / 'manifest.json').read_text())
    runs = {}
    for model, stage in CONDITIONS:
        path = RUNS / f'{model}_{stage.lower()}' / 'metrics.jsonl'
        if path.is_file():
            epochs = [json.loads(line) for line in path.read_text().splitlines() if line]
            runs[f'{model}_{stage}'] = min(epochs, key=lambda x: x['huber_z_macro'])
    base = baseline(meta)
    out = {'baseline': base, 'best_runs': runs, 'complete': len(runs) == 4}
    RUNS.parent.mkdir(parents=True, exist_ok=True)
    (RUNS.parent / 'comparison.json').write_text(json.dumps(out, indent=2) + '\n')
    lines = [
        '# Whisper score regression: Tiny/Base × S10/S9', '',
        f"Training clips: S10 {meta['rows']['S10']:,}; S9 {meta['rows']['S9']:,}. "
        f"Common validation: {meta['rows']['val']:,} held-out clips.", '',
        'Encoder parameters: Tiny 8,208,384; Base 20,590,592 (2.51×). '
        'S9 has 2.41× as many training clips as S10.', '',
        'All metrics use the same 100 regression targets. Huber is averaged after '
        'standardizing each target with S10 training statistics; lower is better.', '',
        '| Encoder | Stage | Best epoch | Train Huber | Val Huber | Emotion Huber | VoiceNet Huber | Other Huber |',
        '|---|---|---:|---:|---:|---:|---:|---:|',
        f"| Constant training mean | — | — | — | {base['huber_z_macro']:.4f} | "
        f"{base['emotion']['huber_z']:.4f} | {base['voicenet']['huber_z']:.4f} | {base['other']['huber_z']:.4f} |",
    ]
    for model, stage in CONDITIONS:
        r = runs.get(f'{model}_{stage}')
        if r is None:
            lines.append(f'| {model} | {stage} | pending | — | — | — | — | — |')
        else:
            lines.append(f"| {model} | {stage} | {r['epoch']} | {r['train_huber_z']:.4f} | "
                         f"{r['huber_z_macro']:.4f} | {r['emotion']['huber_z']:.4f} | "
                         f"{r['voicenet']['huber_z']:.4f} | {r['other']['huber_z']:.4f} |")
    if len(runs) == 4:
        lines += ['', '## Controlled differences', '']
        for stage in ['S10', 'S9']:
            t, b = runs[f'tiny_{stage}']['huber_z_macro'], runs[f'base_{stage}']['huber_z_macro']
            lines.append(f'- {stage}, Base versus Tiny: {b - t:+.4f} normalized Huber (negative favors Base).')
        for model in ['tiny', 'base']:
            s10, s9 = runs[f'{model}_S10']['huber_z_macro'], runs[f'{model}_S9']['huber_z_macro']
            lines.append(f'- {model}, S9 versus S10: {s9 - s10:+.4f} normalized Huber (negative favors S9).')
        lines += ['', '## Correlation and target patterns', '']
        for model, stage in CONDITIONS:
            r = runs[f'{model}_{stage}']
            lines.append(f"- {model}/{stage}: mean Pearson emotion {r['emotion']['pearson']:.3f}, "
                         f"VoiceNet {r['voicenet']['pearson']:.3f}, other {r['other']['pearson']:.3f}.")
        for title, left, right in [
            ('Base minus Tiny on S10', 'tiny_S10', 'base_S10'),
            ('Base minus Tiny on S9', 'tiny_S9', 'base_S9'),
            ('S9 minus S10 for Tiny', 'tiny_S10', 'tiny_S9'),
            ('S9 minus S10 for Base', 'base_S10', 'base_S9'),
        ]:
            difference = np.asarray(runs[right]['huber_z']) - np.asarray(runs[left]['huber_z'])
            improved = np.argsort(difference)[:5]
            worsened = np.argsort(difference)[-5:][::-1]
            fmt = lambda ids: ', '.join(f"`{meta['targets'][i]}` {difference[i]:+.3f}" for i in ids)
            lines += ['', f'**{title}.** Five largest gains: {fmt(improved)}. '
                      f'Five largest regressions: {fmt(worsened)}.']
    lines += ['', 'The reference labels are teacher scores from the Ladder, not independent human ratings.',
              'Audio longer than 30 seconds is truncated to the first 30 seconds by the Whisper input limit.',
              'The validation is UID-disjoint; it is not guaranteed speaker-disjoint.',
              'At equal epoch count, S9 also receives more optimizer updates; data size and compute exposure '
              'are therefore coupled in the S9-versus-S10 contrast.', '']
    (RUNS.parent / 'comparison.md').write_text('\n'.join(lines))
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
