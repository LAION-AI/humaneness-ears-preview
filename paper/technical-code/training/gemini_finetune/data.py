"""TAR-backed dataset for the existing encoder-only Whisper multitask architecture."""
from __future__ import annotations

import io
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from common import CODE, TarReader, jsonl, read_json
sys.path.insert(0, str(CODE))
from full_ladder_dataset import MixedCollator


class GeminiDataset(Dataset):
    def __init__(self, root, split, require_complete=True):
        self.root = Path(root)
        if require_complete and not (self.root / 'TRAINING_READY.json').exists():
            raise RuntimeError('Training gate is not satisfied; inspect coverage.json')
        self.norm = read_json(self.root / 'checkpoint_normalization.json')
        self.rows = [row for row in jsonl(self.root / 'targets.jsonl') if row['split'] == split and row['ready_for_whisper_training']]
        self.reader = None
        self.mean = np.asarray(self.norm['score_mean'], np.float32)
        self.std = np.asarray(self.norm['score_std'], np.float32)

    def __len__(self):
        return len(self.rows)

    def __getstate__(self):
        return {**self.__dict__, 'reader': None}

    def __getitem__(self, index):
        if self.reader is None:
            self.reader = TarReader()
        row = self.rows[index]
        wav = self.reader.audio(row)
        n_frames = min(1500, math.ceil(len(wav) / 320))
        raw = np.asarray([row['raw_scores'].get(n, np.nan) for n in self.norm['score_names']], np.float32)
        valid = np.isfinite(raw)
        teachers = row.get('teacher_metadata', {})
        if not teachers.get('dnsmos', {}).get('valid_for_speech', False):
            valid[123:130] = False
        if not teachers.get('empathic', {}).get('speech_domain_valid', False):
            valid[100:119] = False
        if not teachers.get('voiceclap', {}).get('speech_domain_valid', False):
            for i in range(131, 192):
                # Valid Gemini emotion annotations remain usable even where
                # the older speech-only attribute teacher is outside its domain.
                if row['score_provenance'].get(self.norm['score_names'][i]) != 'gemini-3.8-flash':
                    valid[i] = False
        scores = np.where(valid, (raw - self.mean) / self.std, 0).astype(np.float32)
        frame = np.zeros(n_frames, np.float32)
        spans, classes = [], []
        for event in row['vocal_bursts']:
            start = max(0, min(n_frames - 1, int(event['start'] / .02)))
            end = min(n_frames, max(start + 1, math.ceil(event['end'] / .02)))
            frame[start:end] = 1
            spans.append((start, end))
            classes.append(event['checkpoint_class_id'])
        result = {'key': row['sha256_mp3'], 'uid': row['sha256_mp3'], 'domain': 'gemini',
            'audio': wav, 'scores': scores, 'score_mask': valid,
            'frame': frame, 'frame_mask': np.full(n_frames, row['burst_frames_valid'], np.float32),
            'events_frames': spans, 'event_class_ids': classes,
            'cps': row['characters_per_second'], 'enabled': True}
        speaker_valid = bool(teachers.get('orange', {}).get('valid_single_speaker_target'))
        for name, dim in [('timbre', 128), ('identity', 250)]:
            reference = row['embeddings'].get(name)
            result[name] = np.zeros(dim, np.float32)
            if reference:
                value = np.load(io.BytesIO(self.reader.read(reference['tar'], reference['member'])), allow_pickle=False)
                if value.shape != (dim,) or not np.isfinite(value).all():
                    raise ValueError(f'Invalid {name} target for {row["sha256_mp3"]}')
                result[name] = value.astype(np.float32)
            result[name + '_valid'] = bool(reference and speaker_valid)
        return result


class GeminiCollator:
    def __init__(self, model_path, normalization):
        self.base = MixedCollator(Path(model_path))
        self.cps_mean = normalization['cps_mean']
        self.cps_std = normalization['cps_std']

    def __call__(self, items):
        batch = self.base(items)
        n_frames = batch['frame'].shape[1]
        width = max(1, max(len(row['events_frames']) for row in items))
        starts = np.zeros((len(items), width), np.int64)
        ends = np.zeros_like(starts)
        classes = np.zeros_like(starts)
        event_valid = np.zeros_like(starts, np.bool_)
        cps = np.zeros(len(items), np.float32)
        cps_valid = np.zeros(len(items), np.bool_)
        for i, row in enumerate(items):
            for j, ((start, end), class_id) in enumerate(zip(row['events_frames'], row['event_class_ids'])):
                starts[i, j], ends[i, j] = min(start, n_frames - 1), min(end, n_frames)
                classes[i, j] = class_id
                event_valid[i, j] = ends[i, j] > starts[i, j]
            if row['cps'] is not None and math.isfinite(row['cps']):
                cps[i], cps_valid[i] = row['cps'], True
        batch.update(event_starts=torch.from_numpy(starts), event_ends=torch.from_numpy(ends),
            event_classes=torch.from_numpy(classes), event_valid=torch.from_numpy(event_valid),
            event_span_valid=torch.from_numpy(event_valid.copy()), cps_raw=torch.from_numpy(cps),
            cps=torch.from_numpy((cps - self.cps_mean) / self.cps_std), cps_valid=torch.from_numpy(cps_valid))
        return batch
