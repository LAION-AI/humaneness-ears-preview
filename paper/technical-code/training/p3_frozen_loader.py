#!/usr/bin/env python3
"""Random-access WebDataset reader for the frozen P3 training snapshot."""
from __future__ import annotations

import io
import json
import os
from collections import OrderedDict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset
from transformers import WhisperFeatureExtractor

from prepare_p3_frozen import ROOT, score_values


class P3FrozenDataset(Dataset):
    def __init__(self, split: str, *, root: Path = ROOT,
                 include_teacher_embeddings: bool = False,
                 score_mean: np.ndarray | None = None,
                 score_std: np.ndarray | None = None,
                 event_class_path: Path | None = None):
        if split not in ('train', 'validation', 'test'):
            raise ValueError(split)
        self.root = Path(root)
        self.split = split
        self.rows = np.load(self.root / 'sample_index.npy', mmap_mode='r')
        self.indices = np.load(self.root / f'{split}_indices.npy', mmap_mode='r')
        frozen = json.loads((self.root / 'frozen_manifest.json').read_text())
        self.shards = frozen['shards']
        self.norm = json.loads((self.root / 'score_normalization.json').read_text())
        self.names = self.norm['score_names']
        self.mean = np.asarray(self.norm['mean'], dtype=np.float32)
        self.std = np.asarray(self.norm['std'], dtype=np.float32)
        if len(self.names) != 192:
            raise RuntimeError('wrong frozen target width')
        if score_mean is not None or score_std is not None:
            if score_mean is None or score_std is None:
                raise ValueError('score_mean and score_std must be supplied together')
            self.mean = np.asarray(score_mean, dtype=np.float32)
            self.std = np.asarray(score_std, dtype=np.float32)
            if self.mean.shape != (192,) or self.std.shape != (192,) or np.any(self.std <= 0):
                raise ValueError('invalid score normalization override')
        self.include_teacher_embeddings = include_teacher_embeddings
        self.event_class_path = Path(event_class_path) if event_class_path else None
        self.event_class_ids = (np.load(self.event_class_path,mmap_mode='r')
                                if self.event_class_path else None)
        if self.event_class_ids is not None and len(self.event_class_ids)!=len(self.rows):
            raise RuntimeError('P3 event class sidecar/index row mismatch')
        self._fds: OrderedDict[str, int] = OrderedDict()

    def __len__(self) -> int:
        return len(self.indices)

    def close(self) -> None:
        fds = getattr(self, '_fds', None)
        if fds is None:
            return
        for fd in fds.values():
            os.close(fd)
        fds.clear()

    def __del__(self):
        self.close()

    def __getstate__(self):
        state = self.__dict__.copy()
        state['rows'] = None
        state['indices'] = None
        state['event_class_ids'] = None
        state['_fds'] = OrderedDict()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.rows = np.load(self.root / 'sample_index.npy', mmap_mode='r')
        self.indices = np.load(self.root / f'{self.split}_indices.npy', mmap_mode='r')
        if self.event_class_path:
            self.event_class_ids=np.load(self.event_class_path,mmap_mode='r')

    def _read(self, path: str, offset: int, size: int) -> bytes:
        if path not in self._fds:
            if len(self._fds) >= 64:
                _, fd = self._fds.popitem(last=False)
                os.close(fd)
            self._fds[path] = os.open(path, os.O_RDONLY)
        else:
            self._fds.move_to_end(path)
        raw = os.pread(self._fds[path], size, offset)
        if len(raw) != size:
            raise RuntimeError(f'short read {path}:{offset}+{size}')
        return raw

    def _payload(self, shard: dict, row: np.void, kind: str) -> bytes:
        if kind == 'audio':
            path = shard['audio']['path']
        elif kind == 'targets':
            path = shard['sidecars']['targets']['path']
        else:
            source_kind = {'timbre': 'orange', 'identity': 'orange',
                           'voiceclap': 'voiceclap', 'bud_e': 'empathic'}[kind]
            path = shard['sidecars'][source_kind]['path']
        return self._read(path, int(row[f'{kind}_offset']), int(row[f'{kind}_size']))

    def __getitem__(self, position: int) -> dict:
        index = int(self.indices[position])
        row = self.rows[index]
        shard = self.shards[index // 10000]
        key = f"p3_{int(row['sample_id']):08d}"
        wave, sr = sf.read(io.BytesIO(self._payload(shard, row, 'audio')),
                           dtype='float32')
        if sr != 16000 or wave.ndim != 1 or not 1600 <= len(wave) <= 480000:
            raise RuntimeError(f'{key}: invalid 16-kHz mono audio')
        target = json.loads(self._payload(shard, row, 'targets'))
        if target['sample_id'] != key:
            raise RuntimeError(f'{key}: target/audio key mismatch')
        score = score_values(target, self.names).astype(np.float32)
        valid = np.isfinite(score)
        normalized = np.where(valid, (score - self.mean) / self.std, 0).astype(np.float32)
        frame = np.asarray(target['burst_frame_labels'], dtype=np.float32)
        frame_mask = np.asarray(target['burst_frame_valid_mask'], dtype=np.float32)
        if len(frame) != len(frame_mask) or not frame_mask.all():
            raise RuntimeError(f'{key}: invalid frame target')
        result = {'key': key, 'audio': wave, 'scores': normalized,
                  'score_mask': valid, 'frame': frame, 'frame_mask': frame_mask,
                  'auxiliary_targets': target['auxiliary_targets'],
                  'events_samples': target['burst_frame_target']['positive_intervals_samples'],
                  'event_raw_class_ids': (self.event_class_ids[index].copy()
                                          if self.event_class_ids is not None else None),
                  'negative_subtype': target['negative_subtype'],
                  'timbre': np.load(io.BytesIO(self._payload(shard, row, 'timbre')),
                                    allow_pickle=False).astype(np.float32),
                  'identity': np.load(io.BytesIO(self._payload(shard, row, 'identity')),
                                      allow_pickle=False).astype(np.float32),
                  'speaker_valid': bool(target['orange_timbre']['valid_single_speaker']
                                        and target['orange_identity']['valid_single_speaker'])}
        if result['timbre'].shape != (128,) or result['identity'].shape != (250,):
            raise RuntimeError(f'{key}: wrong Orange embedding shapes')
        if self.include_teacher_embeddings:
            result['voiceclap_embedding'] = np.load(
                io.BytesIO(self._payload(shard, row, 'voiceclap')),
                allow_pickle=False).astype(np.float32)
            result['bud_e_embedding'] = np.load(
                io.BytesIO(self._payload(shard, row, 'bud_e')),
                allow_pickle=False).astype(np.float32)
            if (result['voiceclap_embedding'].shape != (768,) or
                    result['bud_e_embedding'].shape != (3072,)):
                raise RuntimeError(f'{key}: wrong teacher embedding shapes')
        return result


class ShardBucketBatches:
    """Length-bucketed DDP batches with local TAR reads and no repeated keys."""

    def __init__(self, dataset: P3FrozenDataset, batch_per_rank: int,
                 world: int = 1, rank: int = 0, seed: int = 640913):
        if batch_per_rank < 1 or world < 1 or not 0 <= rank < world:
            raise ValueError('invalid distributed batch coordinates')
        self.dataset = dataset
        self.batch_per_rank, self.world, self.rank = batch_per_rank, world, rank
        self.seed = seed
        self.epoch = 0
        self.width = batch_per_rank * world
        self.shard = dataset.rows['shard'][dataset.indices]
        self.duration = dataset.rows['duration_s'][dataset.indices]

    def __len__(self) -> int:
        return len(self.dataset) // self.width

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        batches = []
        tails = []
        for shard in rng.permutation(np.unique(self.shard)):
            positions = np.flatnonzero(self.shard == shard)
            rng.shuffle(positions)
            for start in range(0, len(positions), self.width * 32):
                pool = positions[start:start + self.width * 32]
                pool = pool[np.argsort(self.duration[pool], kind='stable')]
                full = len(pool) // self.width * self.width
                for i in range(0, full, self.width):
                    batches.append(pool[i:i + self.width])
                tails.extend(pool[full:].tolist())
        rng.shuffle(tails)
        for start in range(0, len(tails) - self.width + 1, self.width):
            batches.append(np.asarray(tails[start:start + self.width]))
        rng.shuffle(batches)
        for group in batches:
            lo = self.rank * self.batch_per_rank
            yield group[lo:lo + self.batch_per_rank].tolist()


class P3WhisperCollator:
    def __init__(self, model_path: Path):
        self.model_path = str(model_path)
        self.extractor = None

    def __getstate__(self):
        return {'model_path': self.model_path, 'extractor': None}

    def __call__(self, batch: list[dict]) -> dict:
        if self.extractor is None:
            self.extractor = WhisperFeatureExtractor.from_pretrained(
                self.model_path, local_files_only=True)
        feature = self.extractor(
            [item['audio'] for item in batch], sampling_rate=16000,
            padding='longest', truncation=True, max_length=480000,
            return_attention_mask=True, return_tensors='np')
        mel = feature['input_features'].astype(np.float32)
        mel_mask = feature['attention_mask'].astype(np.int64)
        if mel.shape[-1] % 2:
            mel = np.pad(mel, ((0, 0), (0, 0), (0, 1)))
            mel_mask = np.pad(mel_mask, ((0, 0), (0, 1)))
        n_frames = mel.shape[-1] // 2
        frame = np.zeros((len(batch), n_frames), np.float32)
        frame_mask = np.zeros_like(frame)
        for i, item in enumerate(batch):
            n = min(n_frames, len(item['frame']))
            frame[i, :n] = item['frame'][:n]
            frame_mask[i, :n] = item['frame_mask'][:n]
        result = {
            'key': [item['key'] for item in batch],
            'mel': torch.from_numpy(mel),
            'mel_mask': torch.from_numpy(mel_mask),
            'scores': torch.from_numpy(np.stack([item['scores'] for item in batch])),
            'score_mask': torch.from_numpy(np.stack([item['score_mask'] for item in batch])),
            'frame': torch.from_numpy(frame),
            'frame_mask': torch.from_numpy(frame_mask),
            'timbre': torch.from_numpy(np.stack([item['timbre'] for item in batch])),
            'identity': torch.from_numpy(np.stack([item['identity'] for item in batch])),
            'speaker_valid': torch.tensor([item['speaker_valid'] for item in batch]),
            'auxiliary_targets': [item['auxiliary_targets'] for item in batch],
            'negative_subtype': [item['negative_subtype'] for item in batch],
        }
        if 'voiceclap_embedding' in batch[0]:
            for name in ('voiceclap_embedding', 'bud_e_embedding'):
                result[name] = torch.from_numpy(np.stack([item[name] for item in batch]))
        return result
