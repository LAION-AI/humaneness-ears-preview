"""Frozen selections and targets from existing Ladder/P3/Gemini readers."""
import hashlib
import math
import sys
import numpy as np
from study_paths import CODE, ROOT, GEMINI, RELEASE, SEED, read

sys.path.insert(0, str(CODE))
from train_full_curriculum import normalization, TIMBRE_INDEX, IDENTITY_INDEX
from layered_curriculum_data import LadderAuxDataset, taxonomy
from p3_frozen_loader import P3FrozenDataset
from timbre_targets import TimbreIndex


class SourceData:
    def __init__(self, domain):
        self.domain = domain
        self.mean, self.std, self.names = normalization()
        self.norm = read(RELEASE / 'training_normalization.json')
        self.classes = read(RELEASE / 'classes.json')
        self.map = self.classes if 'ladder_raw_to_canonical' in self.classes else taxonomy()
        if domain == 'ladder':
            self.data = LadderAuxDataset(self.mean, self.std)
            self.indices = np.asarray([i for s, shard in enumerate(self.data.shards)
                                       if shard['level'] in ('t0', 't1', 't2')
                                       for i in range(0 if s == 0 else int(self.data.ends[s-1]), int(self.data.ends[s]))], np.int64)
            self.speakers = {'timbre': TimbreIndex(TIMBRE_INDEX, 128), 'identity': TimbreIndex(IDENTITY_INDEX, 250)}
        elif domain.startswith('p3_'):
            split = domain[3:]
            self.data = P3FrozenDataset(split, score_mean=self.mean, score_std=self.std,
                                       event_class_path=__import__('pathlib').Path('/e/scratch/reformo/schuhmann1_moss/vocalburst_p3/frozen_1120k/event_class_ids.npy'))
            # Enough training clips for all three 10% stage mixes. Validation
            # and test use the same fixed 2k-clips-per-split pilot across models.
            size = min(len(self.data), 22000 if split == 'train' else 2000)
            self.indices = np.random.default_rng(SEED).choice(len(self.data), size, replace=False)
        elif domain == 'gemini':
            sys.path.insert(0, str(CODE / 'gemini_finetune'))
            from common import TarReader
            self.reader = TarReader()
            selection = GEMINI.parent / 'gemini_multisource_100h_20261004/combined_unique_manifest.jsonl'
            self.data = [__import__('json').loads(line) for line in selection.open()]
            self.indices = np.arange(len(self.data))
        else:
            from evaluate_public_benchmarks import parquet_samples, emolia_samples
            self.data = parquet_samples(domain) if domain in ('emonet', 'crema', 'ravdess') else emolia_samples(domain)
            self.indices = np.arange(len(self.data))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        index = int(self.indices[position])
        if self.domain == 'gemini':
            row = self.data[index]
            return row['sha256_mp3'], self.reader.audio(row, verify=True), {'source': 'gemini', 'key': row['sha256_mp3']}, {}
        if self.domain not in ('ladder', 'p3_train', 'p3_validation', 'p3_test'):
            from evaluate_public_benchmarks import decode
            sample = self.data[index]
            wave, duration, truncated = decode(sample)
            return sample.key, wave, {**sample.metadata, 'source': self.domain, 'key': sample.key,
                                     'duration_s': duration, 'truncated_30s': truncated}, {}
        item = self.data[index]
        if self.domain == 'ladder':
            shard = int(np.searchsorted(self.data.ends, index, side='right'))
            level = self.data.shards[shard]['level']
            stages = ['S8'] + (['S9'] if level in ('t0', 't1') else []) + (['S10'] if level == 't0' else [])
            value = int(hashlib.sha256(item['uid'].encode()).hexdigest()[:8], 16) % 100
            split = 'train' if value < 90 else 'validation' if value < 95 else 'test'
            lookup = self.map['ladder_raw_to_canonical']
            spans = item['events_frames']
            for name, speaker in self.speakers.items():
                vector, valid = speaker.lookup([item['uid']])
                item[name], item[name + '_valid'] = vector[0].astype(np.float32), bool(valid[0])
        else:
            stages = ['S8', 'S9', 'S10']
            split = self.domain[3:]
            lookup = self.map['p3_raw_to_canonical']
            spans = [(start // 320, math.ceil(end / 320)) for start, end in item['events_samples']]
            for name in ('timbre', 'identity'):
                item[name + '_valid'] = item['speaker_valid']
        classes = [lookup[int(c)] for c in item['event_raw_class_ids'][:len(spans)]]
        target = {k: np.asarray(item[k]) for k in ('scores', 'score_mask', 'frame', 'frame_mask', 'timbre', 'identity', 'timbre_valid', 'identity_valid')}
        cps = item.get('cps_raw')
        cps_valid = cps is not None and np.isfinite(cps)
        target.update(cps=np.float32((cps - self.norm['cps_mean']) / self.norm['cps_std']) if cps_valid else np.float32(0),
                      cps_valid=np.bool_(cps_valid), event_starts=np.asarray([x[0] for x in spans], np.int64),
                      event_ends=np.asarray([x[1] for x in spans], np.int64), event_classes=np.asarray(classes, np.int64))
        return self.domain + ':' + item['key'], item['audio'], {'source': self.domain, 'key': item['key'], 'split': split, 'stages': stages}, target
