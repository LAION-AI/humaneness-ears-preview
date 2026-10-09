#!/usr/bin/env python3
"""Build resumable random-access metadata for every internal Ladder audio shard.

The complete S1 stock is in out/scaling_stages/pack/out, whereas the published
copy contains only the small tiers. Existing TAR member offsets are reused;
audio TARs are never copied or rescanned.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from prepare import target_names
from p3_mix_schedule import LADDER_STAGE_ROWS


INDEX = Path('/e/scratch/reformo/schuhmann1_moss/out/scaling_stages/pack/out/index')
AUDIO = Path('/e/scratch/reformo/schuhmann1_moss/out/scaling_stages/pack/out/data')
OUT = Path('/e/scratch/reformo/schuhmann1_moss/whisper_score_regression/full_ladder')
MAP_ROOT = Path('/e/scratch/reformo/schuhmann1_moss/out/m2_600m_ladder/tar_member_map')
MAP_CODE = Path('/e/scratch/reformo/schuhmann1_moss/out/m2_600m_ladder/code_v4_highlr_20260920')
LEVELS = ('t0', 't1', 't2', 't3', 't4', 't5', 't6', 't7', 'x50', 'x100')
MAX_EVENTS = 32
DTYPE = np.dtype([
    ('audio_offset', '<u8'), ('audio_size', '<u4'), ('dur_s', '<f4'),
    ('uid', 'S128'), ('scores', '<f2', (100,)),
    ('burst_count_log1p', '<f4'), ('burst_count_valid', 'u1'),
    ('burst_frame_valid', 'u1'), ('event_count', 'u1'),
    ('starts', '<f4', (MAX_EVENTS,)), ('ends', '<f4', (MAX_EVENTS,)),
])
TARGETS = target_names()
_MEMBER_MAP = None


def level(path: Path) -> str:
    value = path.stem.rsplit('_', 1)[-1].split('-', 1)[0]
    if value not in LEVELS:
        raise RuntimeError(f'unexpected Ladder level: {path}')
    return value


def member_map():
    global _MEMBER_MAP
    if _MEMBER_MAP is None:
        sys.path.insert(0, str(MAP_CODE))
        from tar_member_map import TarMemberMap
        _MEMBER_MAP = TarMemberMap(MAP_ROOT / 'manifest.json', capacity=2)
    return _MEMBER_MAP


def process(path_text: str) -> dict:
    path = Path(path_text)
    relative = path.relative_to(INDEX)
    audio = AUDIO / relative.with_suffix('.tar')
    out = OUT / 'shards' / relative.with_suffix('.npy')
    if out.exists():
        rows = np.load(out, mmap_mode='r')
        if rows.dtype != DTYPE or len(rows) != pq.ParquetFile(path).metadata.num_rows:
            raise RuntimeError(f'stale full Ladder metadata: {out}')
        return {'level': level(path), 'parquet': str(path), 'audio': str(audio),
                'meta': str(out), 'rows': len(rows)}
    columns = ['uid', 'dur_s', 'audio_bytes', 'n_bursts',
               'burst_starts', 'burst_ends'] + TARGETS
    table = pq.read_table(path, columns=columns)
    n = len(table)
    rows = np.empty(n, dtype=DTYPE)
    rows['starts'].fill(-1)
    rows['ends'].fill(-1)
    rows['event_count'].fill(0)
    rows['burst_count_valid'].fill(0)
    rows['burst_frame_valid'].fill(0)
    rows['burst_count_log1p'].fill(0)
    rows['dur_s'] = np.asarray(table['dur_s'].to_numpy(zero_copy_only=False), np.float32)
    scores = np.stack([np.asarray(table[name].to_numpy(zero_copy_only=False), np.float32)
                       for name in TARGETS], axis=1)
    rows['scores'] = scores.astype(np.float16)
    uids = table['uid'].to_pylist()
    audio_sizes = table['audio_bytes'].to_pylist()
    burst_counts = table['n_bursts'].to_pylist()
    starts = table['burst_starts'].to_pylist()
    ends = table['burst_ends'].to_pylist()
    mapper = member_map()
    for i, uid in enumerate(uids):
        if not uid or len(uid.encode()) > 128:
            raise RuntimeError(f'invalid UID at {path}:{i}')
        rows['uid'][i] = uid.encode()
        size = audio_sizes[i]
        if size is None or not 0 < int(size) < 2**32:
            raise RuntimeError(f'{path}:{i}: missing MP3 byte count')
        moss_offset, _ = mapper.locate(str(audio), uid + '.moss.npy')
        # Packer order is MP3, MOSS, JSON. The MOSS data offset follows the
        # padded MP3 payload and its own 512-byte TAR header.
        offset = moss_offset - 512 - ((int(size) + 511) // 512) * 512
        if offset < 512:
            raise RuntimeError(f'{audio}:{uid}: impossible MP3 offset')
        rows['audio_offset'][i] = offset
        rows['audio_size'][i] = int(size)
        if burst_counts[i] is not None:
            rows['burst_count_valid'][i] = 1
            if starts[i] is not None and ends[i] is not None:
                if len(starts[i]) != len(ends[i]) or len(starts[i]) > MAX_EVENTS:
                    raise RuntimeError(f'invalid burst intervals at {path}:{i}')
                events = [(max(0., float(a)), min(30., float(b)))
                          for a, b in zip(starts[i], ends[i])
                          if a is not None and b is not None and float(a) < 30
                          and float(b) > 0 and float(a) < float(b)]
                rows['burst_count_log1p'][i] = np.log1p(len(events))
                rows['burst_frame_valid'][i] = 1
                rows['event_count'][i] = len(events)
                for j, (start, end) in enumerate(events):
                    rows['starts'][i, j] = start
                    rows['ends'][i, j] = end
            else:
                rows['burst_count_log1p'][i] = np.log1p(int(burst_counts[i]))
    # Verify the inferred offset against real TAR headers at both ends of
    # every shard. This catches a changed member order or byte-count schema.
    fd = os.open(audio, os.O_RDONLY)
    try:
        for i in sorted({0, n - 1}):
            header = os.pread(fd, 512, int(rows['audio_offset'][i]) - 512)
            name = header[:100].split(b'\0', 1)[0]
            size = int(header[124:136].rstrip(b'\0 ').strip() or b'0', 8)
            if name != (uids[i] + '.mp3').encode() or size != int(rows['audio_size'][i]):
                raise RuntimeError(f'{audio}:{uids[i]}: inferred MP3 offset failed TAR header check')
    finally:
        os.close(fd)
    # Keep consecutive training batches near the same duration. The audio
    # offsets and UIDs travel with their score and burst targets.
    rows = rows[np.argsort(rows['dur_s'], kind='stable')]
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix('.npy.incomplete')
    with tmp.open('wb') as stream:
        np.save(stream, rows, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(out)
    return {'level': level(path), 'parquet': str(path), 'audio': str(audio),
            'meta': str(out), 'rows': n}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=32)
    parser.add_argument('--limit', type=int, default=0,
                        help='Only for a small smoke test; no complete manifest')
    args = parser.parse_args()
    paths = sorted(str(p) for p in INDEX.rglob('*.parquet'))
    if args.limit:
        paths = paths[:args.limit]
    print('shards', len(paths), 'workers', args.workers, flush=True)
    shards = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process, path): path for path in paths}
        for completed, future in enumerate(as_completed(futures), 1):
            result = future.result()
            shards.append(result)
            if completed % 100 == 0 or completed == len(paths):
                print('indexed', completed, '/', len(paths), flush=True)
    shards.sort(key=lambda x: (LEVELS.index(x['level']), x['parquet']))
    counts = Counter()
    for shard in shards:
        counts[shard['level']] += shard['rows']
    if not args.limit:
        for i, stage in enumerate(('S10', 'S9', 'S8', 'S7', 'S6', 'S5',
                                   'S4', 'S3', 'S2', 'S1'), 1):
            expected = LADDER_STAGE_ROWS[stage]
            actual = sum(counts[name] for name in LEVELS[:i])
            if actual != expected:
                raise RuntimeError(f'{stage}: indexed {actual}, expected {expected}')
    report = {'schema_version': 'full-ladder-random-access-v1',
              'status': 'smoke' if args.limit else 'complete',
              'target_names': TARGETS, 'dtype_descr': DTYPE.descr,
              'source_index': str(INDEX), 'source_audio': str(AUDIO),
              'offset_method': 'derive MP3 offset from packed MOSS member map and Parquet audio_bytes; verify first and last TAR headers per shard',
              'levels': list(LEVELS), 'level_counts': dict(counts),
              'shards': shards}
    filename = 'smoke_manifest.json' if args.limit else 'manifest.json'
    out = OUT / filename
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + '.incomplete')
    tmp.write_text(json.dumps(report, indent=2) + '\n')
    tmp.replace(out)
    print('COMPLETE', filename, dict(counts), flush=True)


if __name__ == '__main__':
    main()
