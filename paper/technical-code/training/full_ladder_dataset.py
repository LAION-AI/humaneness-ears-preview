#!/usr/bin/env python3
"""Mixed random-access Ladder/P3 loader for the continuous score curriculum."""
from __future__ import annotations

import bisect
import io
import json
import math
import os
from collections import OrderedDict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from torch.utils.data import Dataset, Sampler
from transformers import WhisperFeatureExtractor

from p3_frozen_loader import P3FrozenDataset
from p3_frozen_mix_schedule import p3_train_position
from prepare_full_ladder import DTYPE, LEVELS, OUT


class LadderAudioDataset(Dataset):
    def __init__(self, mean: np.ndarray, std: np.ndarray, root: Path = OUT):
        spec = json.loads((root / 'manifest.json').read_text())
        if spec['status'] != 'complete' or spec['levels'] != list(LEVELS):
            raise RuntimeError('full Ladder index is not complete')
        self.shards = spec['shards']
        self.counts = np.asarray([x['rows'] for x in self.shards], dtype=np.int64)
        self.ends = np.cumsum(self.counts)
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        if self.mean.shape != (192,) or self.std.shape != (192,):
            raise ValueError('expected 192 shared score normalization entries')
        self._rows: OrderedDict[int, np.ndarray] = OrderedDict()
        self._fds: OrderedDict[int, int] = OrderedDict()

    def __len__(self) -> int:
        return int(self.ends[-1])

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_rows'] = OrderedDict()
        state['_fds'] = OrderedDict()
        return state

    def close(self):
        for fd in self._fds.values():
            os.close(fd)
        self._fds.clear()
        self._rows.clear()

    def __del__(self):
        if hasattr(self, '_fds'):
            self.close()

    def _row(self, shard: int, local: int):
        if shard not in self._rows:
            if len(self._rows) >= 32:
                self._rows.popitem(last=False)
            rows = np.load(self.shards[shard]['meta'], mmap_mode='r')
            if rows.dtype != DTYPE or len(rows) != self.shards[shard]['rows']:
                raise RuntimeError(f'stale metadata shard {shard}')
            self._rows[shard] = rows
        else:
            self._rows.move_to_end(shard)
        return self._rows[shard][local]

    def _audio(self, shard: int, row) -> np.ndarray:
        if shard not in self._fds:
            if len(self._fds) >= 32:
                _, fd = self._fds.popitem(last=False)
                os.close(fd)
            self._fds[shard] = os.open(self.shards[shard]['audio'], os.O_RDONLY)
        else:
            self._fds.move_to_end(shard)
        raw = os.pread(self._fds[shard], int(row['audio_size']), int(row['audio_offset']))
        if len(raw) != int(row['audio_size']):
            raise RuntimeError(f'short audio read in shard {shard}')
        audio, sr = sf.read(io.BytesIO(raw), dtype='float32', always_2d=False)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        if sr != 16000:
            common = math.gcd(sr, 16000)
            audio = resample_poly(audio, 16000 // common, sr // common).astype(np.float32)
        audio = np.nan_to_num(audio[:480000], copy=False)
        if len(audio) < 1600:
            audio = np.pad(audio, (0, 1600 - len(audio)))
        return audio

    def get_shard_row(self, shard: int, local: int) -> dict:
        row = self._row(shard, local)
        audio = self._audio(shard, row)
        raw_scores = row['scores'].astype(np.float32)
        valid = np.isfinite(raw_scores)
        scores = np.zeros(192, dtype=np.float32)
        score_mask = np.zeros(192, dtype=np.bool_)
        scores[:100] = np.where(valid, (raw_scores - self.mean[:100]) / self.std[:100], 0)
        score_mask[:100] = valid
        if row['burst_count_valid']:
            scores[130] = (float(row['burst_count_log1p']) - self.mean[130]) / self.std[130]
            score_mask[130] = True
        n_frames = min(1500, (len(audio) + 319) // 320)
        frame = np.zeros(n_frames, dtype=np.float32)
        frame_mask = np.full(n_frames, float(row['burst_frame_valid']), dtype=np.float32)
        if row['burst_frame_valid']:
            for i in range(int(row['event_count'])):
                start = int(float(row['starts'][i]) * 50)
                end = math.ceil(float(row['ends'][i]) * 50)
                frame[max(0, start):min(n_frames, end)] = 1
        uid = bytes(row['uid']).rstrip(b'\0').decode('utf-8')
        return {'key': uid, 'uid': uid, 'domain': 'ladder', 'audio': audio,
                'scores': scores, 'score_mask': score_mask,
                'frame': frame, 'frame_mask': frame_mask,
                'timbre': np.zeros(128, np.float32),
                'identity': np.zeros(250, np.float32),
                'timbre_valid': False, 'identity_valid': False}

    def __getitem__(self, index: int) -> dict:
        shard = bisect.bisect_right(self.ends, int(index))
        start = 0 if shard == 0 else int(self.ends[shard - 1])
        return self.get_shard_row(shard, int(index) - start)


class MixedDataset(Dataset):
    def __init__(self, ladder: LadderAudioDataset, p3: P3FrozenDataset):
        self.ladder, self.p3 = ladder, p3

    def __len__(self):
        return len(self.ladder) + len(self.p3)

    def __getitem__(self, coordinate: tuple[str, int, bool]) -> dict:
        domain, index, enabled = coordinate
        if domain == 'ladder':
            result = self.ladder[int(index)]
        elif domain == 'p3':
            result = self.p3[int(index)]
            result['uid'] = result['key']
            result['domain'] = 'p3'
            result['timbre_valid'] = result['speaker_valid']
            result['identity_valid'] = result['speaker_valid']
        else:
            raise ValueError(domain)
        result['enabled'] = bool(enabled)
        if not enabled:
            result['score_mask'][:] = False
            result['frame_mask'][:] = 0
            result['timbre_valid'] = False
            result['identity_valid'] = False
        return result


class StageBatches(Sampler[list[tuple[str, int, bool]]]):
    """Exact stage counts, whole-domain batches, deterministic DDP sharding."""
    def __init__(self, ladder: LadderAudioDataset, schedule: dict, stage_index: int,
                 *, batch_per_rank: int, world: int, rank: int, start_batch: int = 0):
        self.ladder = ladder
        self.schedule = schedule
        self.stage = schedule['stages'][stage_index]
        self.stage_index = stage_index
        self.batch_per_rank = batch_per_rank
        self.world = world
        self.rank = rank
        self.width = batch_per_rank * world
        self.ladder_batches = math.ceil(self.stage['ladder'] / self.width)
        self.p3_batches = math.ceil(self.stage['p3'] / self.width)
        self.total_batches = self.ladder_batches + self.p3_batches
        self.start_batch = start_batch
        if not 0 <= rank < world or not 0 <= start_batch <= self.total_batches:
            raise ValueError('invalid DDP batch coordinate')
        allowed = set(LEVELS[:len(LEVELS) - stage_index])
        rng = np.random.default_rng(640913 + stage_index)
        self.shards = np.asarray([i for i, spec in enumerate(ladder.shards)
                                  if spec['level'] in allowed], dtype=np.int32)
        rng.shuffle(self.shards)
        self.ends = np.cumsum(ladder.counts[self.shards])
        if int(self.ends[-1]) != self.stage['ladder']:
            raise RuntimeError(f'{self.stage["stage"]}: Ladder row count mismatch')

    def __len__(self) -> int:
        return self.total_batches - self.start_batch

    def _ladder_index(self, position: int) -> int:
        shard_position = bisect.bisect_right(self.ends, position)
        shard = int(self.shards[shard_position])
        start = 0 if shard_position == 0 else int(self.ends[shard_position - 1])
        local = position - start
        return (0 if shard == 0 else int(self.ladder.ends[shard - 1])) + local

    def __iter__(self):
        for batch in range(self.start_batch, self.total_batches):
            p3_before = batch * self.p3_batches // self.total_batches
            p3_after = (batch + 1) * self.p3_batches // self.total_batches
            if p3_after > p3_before:
                domain = 'p3'
                first = p3_before * self.width + self.rank * self.batch_per_rank
                limit = self.stage['p3']
            else:
                domain = 'ladder'
                first = (batch - p3_before) * self.width + self.rank * self.batch_per_rank
                limit = self.stage['ladder']
            result = []
            for i in range(first, first + self.batch_per_rank):
                enabled = i < limit
                position = i if enabled else 0
                index = (p3_train_position(self.schedule,
                           self.stage['p3_exposure_start'] + position)
                         if domain == 'p3' else self._ladder_index(position))
                result.append((domain, index, enabled))
            yield result


class MixedCollator:
    def __init__(self, model_path: Path):
        self.model_path = str(model_path)
        self.extractor = None

    def __getstate__(self):
        return {'model_path': self.model_path, 'extractor': None}

    def __call__(self, batch: list[dict]) -> dict:
        if self.extractor is None:
            self.extractor = WhisperFeatureExtractor.from_pretrained(
                self.model_path, local_files_only=True)
        feat = self.extractor([x['audio'] for x in batch], sampling_rate=16000,
                              padding='longest', truncation=True, max_length=480000,
                              return_attention_mask=True, return_tensors='np')
        mel = feat['input_features'].astype(np.float32)
        mel_mask = feat['attention_mask'].astype(np.int64)
        if mel.shape[-1] % 2:
            mel = np.pad(mel, ((0, 0), (0, 0), (0, 1)))
            mel_mask = np.pad(mel_mask, ((0, 0), (0, 1)))
        n_frames = mel.shape[-1] // 2
        frame = np.zeros((len(batch), n_frames), dtype=np.float32)
        frame_mask = np.zeros_like(frame)
        for i, item in enumerate(batch):
            n = min(n_frames, len(item['frame']))
            frame[i, :n] = item['frame'][:n]
            frame_mask[i, :n] = item['frame_mask'][:n]
        return {'uid': [x['uid'] for x in batch],
                'domain': [x['domain'] for x in batch],
                'enabled': torch.tensor([x.get('enabled', True) for x in batch]),
                'mel': torch.from_numpy(mel), 'mel_mask': torch.from_numpy(mel_mask),
                'scores': torch.from_numpy(np.stack([x['scores'] for x in batch])),
                'score_mask': torch.from_numpy(np.stack([x['score_mask'] for x in batch])),
                'frame': torch.from_numpy(frame),
                'frame_mask': torch.from_numpy(frame_mask),
                'timbre': torch.from_numpy(np.stack([x['timbre'] for x in batch])),
                'identity': torch.from_numpy(np.stack([x['identity'] for x in batch])),
                'timbre_valid': torch.tensor([x['timbre_valid'] for x in batch]),
                'identity_valid': torch.tensor([x['identity_valid'] for x in batch])}
