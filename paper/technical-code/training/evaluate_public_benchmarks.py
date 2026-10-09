#!/usr/bin/env python3
"""Reproducible inference of best layered checkpoints on public emotion benchmarks.

Writes one NPZ with raw 192 target scores per benchmark and an aligned JSONL
manifest. Metric calculation and HTML rendering are intentionally separate.
"""
from __future__ import annotations

import argparse
import ast
import csv
import io
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import torch
from safetensors.torch import load_file
from scipy.signal import resample_poly
from transformers import WhisperFeatureExtractor

from layered_multitask_model import LayeredMultiTaskWhisper


ROOT = Path('/e/scratch/reformo/schuhmann1_moss/whisper_score_regression/bench_eval_20261004')
RELEASE = Path('/e/scratch/reformo/schuhmann1_moss/whisper_score_regression/hf_layered_release/model')
STAGES = {'base': 'S4', 'small': 'S3'}


@dataclass
class Sample:
    key: str
    audio: bytes | Path
    metadata: dict


def parquet_samples(kind: str) -> list[Sample]:
    folder = {'emonet': 'emonet', 'crema': 'crema', 'ravdess': 'ravdess'}[kind]
    samples: list[Sample] = []
    for path in sorted((ROOT / folder / 'data').glob('*.parquet')):
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=256):
            for row in batch.to_pylist():
                col = 'audioId' if kind == 'emonet' else 'audio'
                audio = row[col]
                if kind == 'emonet':
                    ratings = ast.literal_eval(row['label'])
                    targets = []
                    for annotation in ratings:
                        for response in annotation.values():
                            for name, intensity in response.items():
                                targets.append([name, int(intensity)])
                    meta = {'ratings': targets, 'audio_name': audio['path']}
                elif kind == 'crema':
                    meta = {'emotion': row['emotion'], 'audio_name': audio['path']}
                else:
                    meta = {'emotion': row['emotion'], 'audio_name': audio['path'],
                            'actor': row['actor'], 'vocal_channel': row['vocal_channel']}
                samples.append(Sample(f'{kind}:{len(samples)}', audio['bytes'], meta))
    return samples


def emolia_samples(kind: str) -> list[Sample]:
    repo = ROOT / 'emolia' / 'repo'
    labels = repo / 'analysis_outputs' / kind / 'benchmark_labels.csv'
    rows = list(csv.DictReader(labels.open()))
    samples: list[Sample] = []
    if kind == 'emolia-emo':
        index: dict[str, Path] = {}
        for path in sorted((repo / 'dataset' / kind / 'data').rglob('*.mp3')):
            index.setdefault(path.stem, path)
        grouped: dict[str, list[dict]] = {}
        for row in rows:
            grouped.setdefault(Path(row['file_name']).stem, []).append(row)
        for stem, group in grouped.items():
            path = index.get(stem)
            if path is None:
                raise FileNotFoundError(stem)
            samples.append(Sample(f'{kind}:{stem}', path,
                                  {'audio_name': path.name, 'rows': group}))
    else:
        for row in rows:
            if row['dimension'] == 'LANG':
                continue  # No LANG head; all 57 ordinal dimensions are retained.
            path = (repo / 'dataset' / kind / 'data' / row['dimension'] /
                    row['level'] / row['polarity'] / row['file_name'])
            if not path.is_file():
                raise FileNotFoundError(path)
            samples.append(Sample(f'{kind}:{len(samples)}', path,
                                  {'audio_name': str(path.relative_to(repo)), 'rows': [row]}))
    return samples


def decode(sample: Sample) -> tuple[np.ndarray, float, bool]:
    source = io.BytesIO(sample.audio) if isinstance(sample.audio, bytes) else sample.audio
    wave, rate = sf.read(source, dtype='float32', always_2d=True)
    wave = wave.mean(axis=1)
    duration = len(wave) / rate
    if rate != 16000:
        divisor = math.gcd(rate, 16000)
        wave = resample_poly(wave, 16000 // divisor, rate // divisor).astype(np.float32)
    truncated = len(wave) > 480000
    wave = wave[:480000]
    if len(wave) < 1600:
        wave = np.pad(wave, (0, 1600 - len(wave)))
    return np.nan_to_num(wave), duration, truncated


def run(args: argparse.Namespace) -> None:
    torch.set_num_threads(2)
    device = torch.device(args.device)
    stage = STAGES[args.model]
    suffix = f'_smoke{args.limit}' if args.limit else ''
    out = ROOT / 'predictions' / f'{args.dataset}_{args.model}_{stage}{suffix}'
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.with_suffix('.npz').exists() and not args.overwrite:
        print('EXISTS', out, flush=True)
        return
    print('LOADING', args.dataset, flush=True)
    samples = (parquet_samples(args.dataset) if args.dataset in ('emonet', 'crema', 'ravdess')
               else emolia_samples(args.dataset))
    if args.limit:
        samples = samples[:args.limit]
    print('DECODING', len(samples), flush=True)
    with ThreadPoolExecutor(max_workers=args.decode_workers) as pool:
        decoded = list(pool.map(decode, samples))
    print('DECODED', len(decoded), 'seconds', round(sum(d for _, d, _ in decoded)), flush=True)
    model_dir = RELEASE / args.model
    classes = json.loads((RELEASE / 'classes.json').read_text())['names']
    train_norm = json.loads((RELEASE / 'training_normalization.json').read_text())
    mean = np.asarray(train_norm['score_mean'], dtype=np.float32)
    std = np.asarray(train_norm['score_std'], dtype=np.float32)
    model = LayeredMultiTaskWhisper(model_dir, len(classes), initialize_pretrained=False)
    model.load_state_dict(load_file(str(model_dir / stage / 'model.safetensors'), device='cpu'), strict=True)
    model = model.to(device).eval()
    extractor = WhisperFeatureExtractor.from_pretrained(model_dir, local_files_only=True)
    order = sorted(range(len(samples)), key=lambda i: len(decoded[i][0]))
    result = np.empty((len(samples), 192), dtype=np.float32)
    print('INFER', args.model, stage, args.dataset, flush=True)
    with torch.inference_mode():
        for offset in range(0, len(order), args.batch_size):
            ids = order[offset:offset + args.batch_size]
            wave = [decoded[i][0] for i in ids]
            features = extractor(wave, sampling_rate=16000, padding='longest',
                                 truncation=True, max_length=480000,
                                 return_attention_mask=True, return_tensors='np')
            mel = features['input_features'].astype(np.float32)
            mask = features['attention_mask'].astype(np.int64)
            if mel.shape[-1] % 2:
                mel = np.pad(mel, ((0, 0), (0, 0), (0, 1)))
                mask = np.pad(mask, ((0, 0), (0, 1)))
            if device.type == 'cuda':
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    output = model(torch.from_numpy(mel).to(device),
                                   torch.from_numpy(mask).to(device))
            else:
                output = model(torch.from_numpy(mel).to(device),
                               torch.from_numpy(mask).to(device))
            scores = output['scores'].float().cpu().numpy() * std + mean
            result[ids] = scores
            if offset // args.batch_size % 50 == 0 or offset + len(ids) == len(order):
                print(args.model, args.dataset, min(offset + len(ids), len(order)), '/', len(order), flush=True)
    rows = [{**sample.metadata, 'key': sample.key, 'duration_s': float(decoded[i][1]),
             'truncated_30s': bool(decoded[i][2])} for i, sample in enumerate(samples)]
    np.savez_compressed(out.with_suffix('.npz'), scores=result)
    tmp = out.with_suffix('.jsonl.incomplete')
    tmp.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
    os.replace(tmp, out.with_suffix('.jsonl'))
    print('COMPLETE', out, len(samples), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=('base', 'small'), required=True)
    parser.add_argument('--dataset', choices=('emolia-emo', 'emolia-dim', 'emonet', 'crema', 'ravdess'), required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--decode-workers', type=int, default=16)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--overwrite', action='store_true')
    run(parser.parse_args())
