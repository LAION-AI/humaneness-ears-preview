#!/usr/bin/env python3
"""Deterministic S1→S10 exposure plan for the frozen P3 train split."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from p3_mix_schedule import LADDER_STAGE_ROWS
from prepare_p3_frozen import ROOT


def coprime_multiplier(size: int) -> int:
    candidate = 1_000_003
    while math.gcd(candidate, size) != 1:
        candidate += 2
    return candidate


def plan(train_rows: int) -> dict:
    if train_rows < 100_000:
        raise ValueError('frozen P3 train split is too small')
    # With L Ladder clips, L/9 P3 clips make P3 one tenth of the combined
    # stage. Round to a whole clip; the resulting share is closest to 10%.
    target_fraction = .10
    stages = []
    exposure_cursor = 0
    for stage, ladder in LADDER_STAGE_ROWS.items():
        p3 = (ladder + 4) // 9
        total = ladder + p3
        if abs(p3 / total - target_fraction) > 1 / total:
            raise RuntimeError(f'{stage}: P3 share deviates from 10%')
        stages.append({'stage': stage, 'ladder': ladder, 'p3': p3,
                       'total': total, 'p3_share': p3 / total,
                       'p3_exposure_start': exposure_cursor,
                       'p3_exposure_end': exposure_cursor + p3})
        exposure_cursor += p3
    ladder_total = sum(stage['ladder'] for stage in stages)
    global_fraction = exposure_cursor / (exposure_cursor + ladder_total)
    if abs(global_fraction - target_fraction) > 1 / (exposure_cursor + ladder_total):
        raise RuntimeError('global P3 share deviates from 10%')
    return {'schema_version': 'p3-frozen-curriculum-v2',
            'p3_train_unique': train_rows,
            'target_p3_fraction_per_stage': target_fraction,
            'ladder_total_exposures': ladder_total,
            'p3_total_exposures': exposure_cursor,
            'mean_exposures_per_p3_key': exposure_cursor / train_rows,
            'min_exposures_per_p3_key': exposure_cursor // train_rows,
            'max_exposures_per_p3_key': math.ceil(exposure_cursor / train_rows),
            'global_p3_fraction': global_fraction,
            'permutation_a': coprime_multiplier(train_rows),
            'permutation_b': 640913 % train_rows,
            'stages': stages,
            'index_semantics': 'P3 index is a row of train_indices.npy, never a production global sample ID',
            'mix_rule': 'sample-level reference: floor((position+1)*p3/total)>floor(position*p3/total)',
            'training_batch_rule': 'whole-domain DDP global batches interleaved at approximately 10% P3; each stage consumes exactly its scheduled Ladder and P3 sample counts; only last batches are zero-loss padded'}


def p3_train_position(report: dict, exposure: int) -> int:
    size = report['p3_train_unique']
    return (report['permutation_a'] * (exposure % size) +
            report['permutation_b'] + exposure // size) % size


def mixture_at(report: dict, stage: dict, position: int) -> tuple[str, int]:
    if not 0 <= position < stage['total']:
        raise IndexError(position)
    p3_before = position * stage['p3'] // stage['total']
    p3_after = (position + 1) * stage['p3'] // stage['total']
    if p3_after > p3_before:
        exposure = stage['p3_exposure_start'] + p3_before
        return 'p3', p3_train_position(report, exposure)
    return 'ladder', position - p3_before


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=ROOT)
    args = ap.parse_args()
    index = json.loads((args.root / 'training_index_report.json').read_text())
    result = plan(index['split_counts']['train']['samples'])
    out = args.root / 'curriculum_mix_schedule.json'
    tmp = out.with_suffix(out.suffix + '.incomplete')
    tmp.write_text(json.dumps(result, indent=2) + '\n')
    tmp.replace(out)
    print({'p3_train_unique': result['p3_train_unique'],
           'p3_total_exposures': result['p3_total_exposures'],
           'mean_exposures': result['mean_exposures_per_p3_key'],
           'global_p3_fraction': result['global_p3_fraction'],
           'output': str(out)}, flush=True)


if __name__ == '__main__':
    main()
