#!/usr/bin/env python3
"""Historical 10M P3 exposure plan; use p3_frozen_mix_schedule.py for 1.12M.

Deterministic P3 exposure plan for one uninterrupted S1→S10 epoch per stage.

This plans sample IDs and positions; it does not start the large training run.
Ladder and P3 loaders can consume the returned domain/index pair without
materializing an 83-million-row shuffle or mixing train with P3 holdout keys.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

LADDER_STAGE_ROWS = {
    'S1': 34_112_433, 'S2': 17_135_901, 'S3': 6_666_998,
    'S4': 3_425_099, 'S5': 1_764_990, 'S6': 912_789,
    'S7': 382_768, 'S8': 197_662, 'S9': 100_858, 'S10': 41_771,
}
P3_TRAIN_ROWS = 9_800_000  # 980 of the planned 1,000 ten-thousand-row shards.
P3_S1_EXPOSURES = 11_000_000  # One pass of all train IDs plus 1.2M repeats.
PERMUTATION_A = 1_000_003
PERMUTATION_B = 640_913
assert math.gcd(PERMUTATION_A, P3_TRAIN_ROWS) == 1


def plan() -> list[dict]:
    rows = []
    cursor = 0
    for stage, ladder in LADDER_STAGE_ROWS.items():
        p3 = P3_S1_EXPOSURES if stage == 'S1' else round(ladder / 4)
        total = ladder + p3
        assert 3 * p3 <= ladder, f'{stage}: P3 exceeds 25% of stage'
        rows.append({'stage': stage, 'ladder': ladder, 'p3': p3,
                     'total': total, 'p3_share': p3 / total,
                     'p3_exposure_start': cursor,
                     'p3_exposure_end': cursor + p3})
        cursor += p3
    ladder_total = sum(row['ladder'] for row in rows)
    p3_total = sum(row['p3'] for row in rows)
    assert 3 * p3_total <= ladder_total, 'P3 exceeds 25% of whole run'
    assert p3_total < 2 * P3_TRAIN_ROWS, 'a P3 train key would need a third pass'
    return rows


def p3_train_id(exposure: int) -> int:
    """A shuffled full first pass, then shuffled second pass over train keys."""
    return (PERMUTATION_A * (exposure % P3_TRAIN_ROWS) + PERMUTATION_B) % P3_TRAIN_ROWS


def mixture_at(row: dict, position: int) -> tuple[str, int]:
    """Return (domain, index), with P3 IDs spread through the whole stage."""
    if not 0 <= position < row['total']:
        raise IndexError(position)
    p3_before = position * row['p3'] // row['total']
    p3_after = (position + 1) * row['p3'] // row['total']
    if p3_after > p3_before:
        return 'p3', p3_train_id(row['p3_exposure_start'] + p3_before)
    return 'ladder', position - p3_before


def rank_batch(row: dict, update: int, rank: int, world: int,
               batch_per_rank: int) -> list[tuple[str, int] | None]:
    """Equal-size DDP batches; None is a zero-loss padding slot on the last step."""
    if not (0 <= rank < world and batch_per_rank > 0 and update >= 0):
        raise ValueError('invalid distributed batch coordinates')
    width = world * batch_per_rank
    if update >= math.ceil(row['total'] / width):
        raise IndexError(update)
    start = update * width + rank * batch_per_rank
    return [mixture_at(row, start + j) if start + j < row['total'] else None
            for j in range(batch_per_rank)]


def report() -> dict:
    rows = plan()
    ladder = sum(row['ladder'] for row in rows)
    p3 = sum(row['p3'] for row in rows)
    return {'schema': 'p3-curriculum-mix-v1', 'ladder_exposures': ladder,
            'p3_train_unique': P3_TRAIN_ROWS, 'p3_validation_unique': 100_000,
            'p3_test_unique': 100_000, 'p3_exposures': p3,
            'p3_global_fraction': p3 / (ladder + p3),
            'p3_mean_exposures_per_train_key': p3 / P3_TRAIN_ROWS,
            'p3_max_exposures_per_train_key': 2,
            'global_batch_reference': 256,
            'optimizer_updates_at_global_batch_256':
                sum(math.ceil(row['total'] / 256) for row in rows),
            'permutation_a': PERMUTATION_A, 'permutation_b': PERMUTATION_B,
            'mix_rule': 'floor((position+1)*p3/total)>floor(position*p3/total)',
            'stages': rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path)
    args = ap.parse_args()
    result = report()
    content = json.dumps(result, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content)
    print(content)


if __name__ == '__main__':
    main()
