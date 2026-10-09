#!/usr/bin/env python3
"""Index frozen P3 WebDataset TARs and remove exact-source holdout overlap."""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import tarfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from p3_target_adapter import target_names


ROOT = Path('/e/scratch/reformo/schuhmann1_moss/vocalburst_p3/frozen_1120k')
DTYPE = np.dtype([
    ('sample_id', '<u4'), ('shard', '<u2'), ('split', 'u1'),
    ('positive', 'u1'), ('duration_s', '<f4'),
    *[(f'{kind}_{field}', '<u8' if field == 'offset' else '<u4')
      for kind in ('audio', 'targets', 'timbre', 'identity', 'voiceclap', 'bud_e')
      for field in ('offset', 'size')],
])
SPLITS = {'train': 1, 'validation': 2, 'test': 3, 'excluded': 4}
EMBEDDINGS = {
    'timbre': ('orange', '.orange_timbre.npy'),
    'identity': ('orange', '.orange_identity.npy'),
    'voiceclap': ('voiceclap', '.voiceclap.npy'),
    'bud_e': ('empathic', '.bud_e_pooled.npy'),
}
DNSSCORES = ('SIG_raw', 'BAK_raw', 'OVRL_raw', 'SIG', 'BAK', 'OVRL', 'P808_MOS')


def members(path: Path):
    with tarfile.open(path, 'r:') as archive:
        for member in archive:
            if member.isfile():
                yield archive, member


def sample_number(key: str, shard: int, shard_position: int) -> int:
    if not key.startswith('p3_') or not key[3:].isdigit():
        raise RuntimeError(f'bad P3 key: {key}')
    number = int(key[3:])
    if number // 10000 != shard:
        raise RuntimeError(f'{key} does not belong to shard {shard}')
    return shard_position * 10000 + number % 10000


def groups(meta: dict) -> frozenset[str]:
    sources = [item['source'] for item in meta['components'] + meta['events']]
    result = []
    for source in sources:
        if source.get('sha256'):
            source_id = 'sha256:' + source['sha256']
        elif source.get('hf_dataset') and source.get('key'):
            source_id = source['hf_dataset'] + ':' + source['key']
        else:
            raise RuntimeError(f"{meta['sample_id']}: source lacks stable identity")
        result.append(source_id)
    return frozenset(result)


def score_schema(example: dict) -> list[str]:
    names = target_names()
    extra = sorted(example['extra_eiv_scores'])
    attrs = sorted(example['voiceclap_attribute_scores'])
    if len(names) != 100 or len(extra) != 19 or len(attrs) != 61:
        raise RuntimeError('wrong P3 score schema')
    return (names + [f'eiv_extra:{x}' for x in extra] +
            [f'audiobox:{x}' for x in ('CE', 'CU', 'PC', 'PQ')] +
            [f'dnsmos:{x}' for x in DNSSCORES] +
            ['burst_count_log1p'] + [f'voiceclap_attribute:{x}' for x in attrs])


def score_values(row: dict, names: list[str]) -> np.ndarray:
    values = []
    for name in names:
        if name.startswith('eiv_extra:'):
            value = row['extra_eiv_scores'][name.split(':', 1)[1]]
        elif name.startswith('audiobox:'):
            value = row['audiobox_scores'][name.split(':', 1)[1]]
        elif name.startswith('dnsmos:'):
            value = row['dnsmos_scores'][name.split(':', 1)[1]]
        elif name.startswith('voiceclap_attribute:'):
            value = (row['voiceclap_attribute_scores'][name.split(':', 1)[1]]
                     if row['voiceclap_attributes_valid_for_speech'] else None)
        elif name == 'burst_count_log1p':
            value = row[name]
        else:
            value = row['ladder_scores'][name]
        values.append(math.nan if value is None else float(value))
    return np.asarray(values, dtype=np.float64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=ROOT)
    ap.add_argument('--validation-shard', type=int, default=188)
    ap.add_argument('--test-shard', type=int, default=200)
    args = ap.parse_args()
    root = args.root
    frozen = json.loads((root / 'frozen_manifest.json').read_text())
    shards = frozen['shards']
    if len(shards) != 112 or args.validation_shard == args.test_shard:
        raise RuntimeError('wrong frozen snapshot or identical holdout shards')
    if {args.validation_shard, args.test_shard} - {x['number'] for x in shards}:
        raise RuntimeError('holdout shard is not frozen')
    n = len(shards) * 10000
    index_path = root / 'sample_index.npy'
    rows = np.lib.format.open_memmap(index_path.with_suffix('.npy.incomplete'),
                                     mode='w+', dtype=DTYPE, shape=(n,))
    source_groups: list[frozenset[str] | None] = [None] * n
    shard_ids = np.empty(n, dtype=np.uint16)
    for shard_position, shard in enumerate(shards):
        number = shard['number']
        shard_ids[shard_position * 10000:(shard_position + 1) * 10000] = number
        audio_path = Path(shard['audio']['path'])
        counts = Counter()
        for archive, member in members(audio_path):
            if member.name.endswith('.opus'):
                pos = sample_number(member.name[:-5], number, shard_position)
                rows[pos]['audio_offset'] = member.offset_data
                rows[pos]['audio_size'] = member.size
                counts['audio'] += 1
            elif member.name.endswith('.json'):
                pos = sample_number(member.name[:-5], number, shard_position)
                meta = json.load(archive.extractfile(member))
                rows[pos]['sample_id'] = int(meta['sample_id'][3:])
                rows[pos]['shard'] = number
                rows[pos]['positive'] = int(bool(meta['events']))
                rows[pos]['duration_s'] = float(meta['duration_s'])
                source_groups[pos] = groups(meta)
                counts['meta'] += 1
        if counts != {'audio': 10000, 'meta': 10000}:
            raise RuntimeError(f'{number}: incomplete audio TAR {counts}')
        print('audio indexed', number, flush=True)
    if any(x is None for x in source_groups):
        raise RuntimeError('some source groups are missing')
    val_groups, test_groups = set(), set()
    for i, sample_groups in enumerate(source_groups):
        if shard_ids[i] == args.validation_shard:
            val_groups.update(sample_groups)
        elif shard_ids[i] == args.test_shard:
            test_groups.update(sample_groups)
    holdout_groups = val_groups | test_groups
    exclusions = Counter()
    for i, sample_groups in enumerate(source_groups):
        if shard_ids[i] == args.validation_shard:
            split = 'validation'
        elif shard_ids[i] == args.test_shard:
            split = 'test' if sample_groups.isdisjoint(val_groups) else 'excluded'
            if split == 'excluded':
                exclusions['test_overlap_validation'] += 1
        else:
            split = 'train' if sample_groups.isdisjoint(holdout_groups) else 'excluded'
            if split == 'excluded':
                exclusions['train_overlap_holdouts'] += 1
        rows[i]['split'] = SPLITS[split]
    del source_groups
    print('source-disjoint split complete', dict(exclusions), flush=True)
    score_names = None
    stats = {name: {'count': None, 'sum': None, 'sum2': None}
             for name in ('train', 'validation', 'test')}
    frame_counts = Counter()
    for shard_position, shard in enumerate(shards):
        number = shard['number']
        counts = Counter()
        for archive, member in members(Path(shard['sidecars']['targets']['path'])):
            if not member.name.endswith('.targets.json'):
                raise RuntimeError(f'{number}: unexpected target member {member.name}')
            key = member.name[:-len('.targets.json')]
            pos = sample_number(key, number, shard_position)
            rows[pos]['targets_offset'] = member.offset_data
            rows[pos]['targets_size'] = member.size
            target = json.load(archive.extractfile(member))
            if target['sample_id'] != key:
                raise RuntimeError(f'{key}: target mismatch')
            if score_names is None:
                score_names = score_schema(target)
                for data in stats.values():
                    data['count'] = np.zeros(len(score_names), np.int64)
                    data['sum'] = np.zeros(len(score_names), np.float64)
                    data['sum2'] = np.zeros(len(score_names), np.float64)
            split = next(name for name, code in SPLITS.items()
                         if code == rows[pos]['split'])
            if split in stats:
                values = score_values(target, score_names)
                valid = np.isfinite(values)
                clean = np.where(valid, values, 0)
                data = stats[split]
                data['count'] += valid
                data['sum'] += clean
                data['sum2'] += clean * clean
                frame_counts[split] += len(target['burst_frame_labels'])
            counts['targets'] += 1
        if counts['targets'] != 10000:
            raise RuntimeError(f'{number}: incomplete targets {counts}')
        for kind, (source_kind, suffix) in EMBEDDINGS.items():
            counts[kind] = 0
            for _archive, member in members(Path(shard['sidecars'][source_kind]['path'])):
                if not member.name.endswith(suffix):
                    continue
                pos = sample_number(member.name[:-len(suffix)], number, shard_position)
                rows[pos][f'{kind}_offset'] = member.offset_data
                rows[pos][f'{kind}_size'] = member.size
                counts[kind] += 1
            if counts[kind] != 10000:
                raise RuntimeError(f'{number}: {kind} embeddings incomplete')
        print('targets and embeddings indexed', number, flush=True)
    for kind in ('audio', 'targets', 'timbre', 'identity', 'voiceclap', 'bud_e'):
        if np.any(rows[f'{kind}_size'] == 0):
            raise RuntimeError(f'missing {kind} offsets')
    split_counts = {}
    for name, code in SPLITS.items():
        indexes = np.flatnonzero(rows['split'] == code).astype(np.uint32)
        np.save(root / f'{name}_indices.npy', indexes)
        split_counts[name] = {'samples': int(len(indexes)),
                              'positive': int(rows['positive'][indexes].sum()),
                              'negative': int(len(indexes) - rows['positive'][indexes].sum())}
    if any(split_counts[x]['samples'] == 0 for x in ('train', 'validation', 'test')):
        raise RuntimeError(f'empty split: {split_counts}')
    train = stats['train']
    count = np.maximum(train['count'], 1)
    mean = train['sum'] / count
    variance = (train['sum2'] / count - mean * mean).clip(min=0)
    std = np.where(train['count'] > 0, np.maximum(np.sqrt(variance), .25), 1)
    mean = np.where(train['count'] > 0, mean, 0)
    norm = {'score_names': score_names, 'mean': mean.tolist(), 'std': std.tolist(),
            'train_valid_counts': train['count'].tolist(),
            'validation_valid_counts': stats['validation']['count'].tolist(),
            'test_valid_counts': stats['test']['count'].tolist(),
            'method': 'train-only mean, population std with minimum 0.25; zero-count targets masked'}
    (root / 'score_normalization.json').write_text(json.dumps(norm, indent=2) + '\n')
    rows.flush()
    del rows
    index_path.with_suffix('.npy.incomplete').replace(index_path)
    report = {'schema_version': 'p3-frozen-training-index-v1',
              'created_utc': datetime.now(timezone.utc).isoformat(),
              'frozen_manifest': str(root / 'frozen_manifest.json'),
              'index': str(index_path), 'dtype_descr': DTYPE.descr,
              'total_samples': n, 'split_counts': split_counts,
              'holdout_shards': {'validation': args.validation_shard,
                                 'test': args.test_shard},
              'holdout_exact_source_id_count': {'validation': len(val_groups),
                                                'test': len(test_groups)},
              'exclusions': dict(exclusions), 'frame_counts': dict(frame_counts),
              'normalization': str(root / 'score_normalization.json'),
              'split_method': 'two balanced completed shards held out; train clips reusing any exact holdout source audio excluded; test clips reusing any validation source audio excluded',
              'source_identity_scope': 'dataset-qualified source key or burst SHA-256 across speech, bursts, AudioSet effects and music',
              'speaker_disjoint': False,
              'annotation_note': 'teacher values are model predictions; masks retain domain validity'}
    tmp = root / 'training_index_report.json.incomplete'
    tmp.write_text(json.dumps(report, indent=2) + '\n')
    tmp.replace(root / 'training_index_report.json')
    print(report['split_counts'], flush=True)
    subprocess.run(['python', str(Path(__file__).with_name('p3_frozen_mix_schedule.py')),
                    '--root', str(root)], check=True)
    if root == ROOT:
        subprocess.run(['python', str(Path(__file__).with_name('verify_p3_frozen.py'))],
                       check=True)
        subprocess.run(['python', str(Path(__file__).with_name('write_p3_frozen_report.py'))],
                       check=True)


if __name__ == '__main__':
    main()
