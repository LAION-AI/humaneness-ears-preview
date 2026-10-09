#!/usr/bin/env python3
"""Evaluate every trained probe and matched score adapters, once per artifact."""
import argparse
import json
import os
import numpy as np
from study_paths import ROOT, BENCH, read
from predict_benchmarks import run as predict
from matched_acted import run as acted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', choices=['legacy', 'gemini'], required=True)
    ap.add_argument('--models', nargs='+')
    ap.add_argument('--include-whisper-baselines', action='store_true')
    args = ap.parse_args()
    rank, world = int(os.environ.get('LOCAL_RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    tasks = [(m['id'], v) for m in read(ROOT / 'study.json')['models'] if not args.models or m['id'] in args.models for v in ('linear', 'mlp')]
    for model, variant in tasks[rank::world]:
        out = ROOT / 'probes' / args.phase / model / variant
        if not (out / 'public_metrics.json').exists():
            predict(model, variant, args.phase)
        for kind in ('crema', 'ravdess'):
            scores = np.load(out / (kind + '_predictions.npz'))['scores']
            rows = [json.loads(line) for line in (out / (kind + '_predictions.jsonl')).open()]
            acted(kind, scores, rows, out)
    # Apply the same new MLPs to the original best Whisper checkpoints.
    for index, model in enumerate(('base', 'small')):
        if not args.include_whisper_baselines or index % world != rank:
            continue
        stage = 'S4' if model == 'base' else 'S3'
        out = ROOT / 'whisper' / 'legacy' / model
        out.mkdir(parents=True, exist_ok=True)
        for kind in ('crema', 'ravdess'):
            stem = BENCH / 'predictions' / (kind + '_' + model + '_' + stage)
            acted(kind, np.load(stem.with_suffix('.npz'))['scores'],
                  [json.loads(line) for line in stem.with_suffix('.jsonl').open()], out)


if __name__ == '__main__':
    main()
