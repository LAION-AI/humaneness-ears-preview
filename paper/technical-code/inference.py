#!/usr/bin/env python3
"""Inference for LAION Humaneness Ears Base and Medium audio encoders."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from safetensors.torch import load_file
from scipy.signal import resample_poly
from transformers import WhisperFeatureExtractor

from model import LayeredMultiTaskWhisper


def read_audio(path):
    wave, rate = sf.read(path, dtype='float32', always_2d=True)
    wave = wave.mean(axis=1)
    duration = len(wave) / rate
    if not .1 <= duration <= 30.:
        raise ValueError(f'Audio must last 0.1–30 seconds; received {duration:.3f}s')
    if not np.isfinite(wave).all():
        raise ValueError('Audio contains nonfinite values')
    if rate != 16000:
        divisor = math.gcd(rate, 16000)
        wave = resample_poly(wave, 16000 // divisor, rate // divisor).astype(np.float32)
    return wave[:480000], duration


class Predictor:
    """Load once and reuse. Local safe weights; no teachers or ASR decoder."""
    def __init__(self, repo_dir, model_size='medium', device='cpu', threads=4):
        if model_size not in ('base', 'medium', 'small') or threads < 1:
            raise ValueError('Use base/medium and at least one CPU thread')
        # Accept the historical architecture label for existing callers.
        model_size = 'medium' if model_size == 'small' else model_size
        torch.set_num_threads(threads)
        self.repo = Path(repo_dir)
        self.model_size, self.device = model_size, device
        self.norm = json.loads((self.repo / 'training_normalization.json').read_text())
        self.display = json.loads((self.repo / 'display_normalization.json').read_text())
        self.class_names = json.loads((self.repo / 'classes.json').read_text())['names']
        self.names = self.norm['score_names']
        if self.names != [r['name'] for r in self.display['targets']]:
            raise ValueError('Display and training target orders differ')
        directory = self.repo / model_size
        self.model = LayeredMultiTaskWhisper(directory, len(self.class_names), initialize_pretrained=False)
        self.model.load_state_dict(load_file(directory / 'model.safetensors', device='cpu'), strict=True)
        self.model.to(device).eval()
        self.extractor = WhisperFeatureExtractor.from_pretrained(directory, local_files_only=True)

    def predict(self, audio, *, frame_threshold=.5, event_threshold=.5, max_events=32, include_frames=False):
        if not 0 < frame_threshold < 1 or not 0 < event_threshold < 1 or max_events < 1:
            raise ValueError('Thresholds must be between zero and one, max_events positive')
        wave, duration = read_audio(audio)
        features = self.extractor([wave], sampling_rate=16000, padding='longest', truncation=True,
                                  max_length=480000, return_attention_mask=True, return_tensors='np')
        mel = features['input_features'].astype(np.float32)
        mask = features['attention_mask'].astype(np.int64)
        if mel.shape[-1] % 2:
            mel = np.pad(mel, ((0, 0), (0, 0), (0, 1)))
            mask = np.pad(mask, ((0, 0), (0, 1)))
        from contextlib import nullcontext
        autocast = torch.autocast('cuda', dtype=torch.bfloat16) if self.device.startswith('cuda') else nullcontext()
        with torch.inference_mode(), autocast:
            out = self.model(torch.from_numpy(mel).to(self.device), torch.from_numpy(mask).to(self.device),
                             predict_events=True, event_threshold=event_threshold, max_pred_events=max_events)
        standardized = out['scores'][0].float().cpu().numpy()
        raw = standardized * np.asarray(self.norm['score_std']) + np.asarray(self.norm['score_mean'])
        z = {name: ((float(value) - row['median']) / row['std']
                    if row['median'] is not None and row['std'] not in (None, 0) else None)
             for name, value, row in zip(self.names, raw, self.display['targets'])}
        count = min(int((mask.sum() + 1) // 2), out['frame'].shape[1])
        frame = out['frame'][0, :count].float().sigmoid().cpu().numpy()
        edges = np.diff(np.pad((frame >= frame_threshold).astype(np.int8), (1, 1)))
        regions = [{'start_s': round(float(a) * .02, 6), 'end_s': round(min(float(b) * .02, duration), 6)}
                   for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))
                   if float(a) * .02 < duration]
        probabilities = out['event_class'][0].float().softmax(-1).cpu().numpy()
        starts = out['predicted_event_starts'][0].cpu().numpy()
        ends = out['predicted_event_ends'][0].cpu().numpy()
        valid = out['predicted_event_valid'][0].cpu().numpy()
        onset = out['onset'][0].float().sigmoid().cpu().numpy()
        proposals = []
        for j in np.flatnonzero(valid):
            start, end = float(starts[j]) * .02, min(float(ends[j]) * .02, duration)
            if start >= end:
                continue
            top = np.argsort(probabilities[j])[-3:][::-1]
            proposals.append({'start_s': round(start, 6), 'end_s': round(end, 6),
                              'onset_probability': float(onset[int(starts[j])]),
                              'top3_classes': [{'label': self.class_names[int(k)],
                                                'softmax_probability': float(probabilities[j, k])} for k in top]})
        cps_z = float(out['cps'][0].float().cpu())
        result = {
            'model': self.model_size, 'checkpoint': 'best Gemini Flash 3.8 fine-tune; epoch 2',
            'duration_s': duration,
            'scores_raw': dict(zip(self.names, map(float, raw))),
            'scores_training_z': dict(zip(self.names, map(float, standardized))),
            'scores_z': z,
            'characters_per_second': cps_z * self.norm['cps_std'] + self.norm['cps_mean'],
            'characters_per_second_training_z': cps_z,
            'estimated_burst_count': float(np.expm1(np.clip(raw[130], 0, 8))),
            'orange_timbre_128d': out['timbre'][0].float().cpu().tolist(),
            'orange_identity_250d': out['identity'][0].float().cpu().tolist(),
            'binary_burst_regions': regions, 'burst_event_proposals': proposals,
            'decoding': {'frame_threshold': frame_threshold, 'event_threshold': event_threshold, 'max_events': max_events},
            'notes': 'Encoder only: no transcript or captions. Raw regression scores are not clipped to ordinal scales. '
                     'Display z-scores use the original train-only reference median/std; they are not probabilities. '
                     'Class probabilities are uncalibrated. Blend is only meaningful when vocal bursts are present.',
        }
        if include_frames:
            result['burst_probability_per_20ms_frame'] = frame.tolist()
        return result


def predict(audio, repo_dir, *, model_size='medium', device='cpu', threads=4, **options):
    """Convenience call; reuse Predictor for many clips to avoid reloading weights."""
    return Predictor(repo_dir, model_size, device, threads).predict(audio, **options)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('audio', type=Path)
    parser.add_argument('--repo-dir', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--model', choices=('base', 'medium', 'small'), default='medium',
                        help='Release variant (small is a legacy alias for medium)')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--frame-threshold', type=float, default=.5)
    parser.add_argument('--event-threshold', type=float, default=.5)
    parser.add_argument('--max-events', type=int, default=32)
    parser.add_argument('--include-frame-probabilities', action='store_true')
    args = parser.parse_args()
    result = predict(args.audio, args.repo_dir, model_size=args.model, device=args.device, threads=args.threads,
                     frame_threshold=args.frame_threshold, event_threshold=args.event_threshold,
                     max_events=args.max_events, include_frames=args.include_frame_probabilities)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
