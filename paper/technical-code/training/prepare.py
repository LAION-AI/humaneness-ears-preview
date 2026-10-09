"""Index the released Ladder TARs without copying audio; make a fixed validation set."""
from __future__ import annotations

import json
import re
import tarfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


ROOT = Path('/e/scratch/reformo/schuhmann1_moss/publish/scaling_ladder')
OUT = Path('/e/scratch/reformo/schuhmann1_moss/whisper_score_regression')
SEED = 20260928


def target_names() -> list[str]:
    text = (ROOT / 'COLUMNS.md').read_text()
    columns = re.findall(r'^\| `([^`]+)`', text, re.M)
    emo = [c for c in columns if c.startswith('emo_')]
    voice = [c for c in columns if c.startswith('vn_') and c.endswith('_reg')]
    assert len(emo) == 40 and len(voice) == 57, (len(emo), len(voice))
    return emo + voice + ['genuineness_0_6', 'blend_0_10', 'R_quality']


def build() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    targets = target_names()
    wanted = ['uid', 'dur_s', 'lang', 'origin', 'source'] + targets
    paths: list[str] = []
    sources: list[str] = []
    accum: dict[str, dict[str, list[np.ndarray]]] = {
        s: {k: [] for k in ['path', 'offset', 'size', 'dur', 'uid', 'cell', 'source', 'y']}
        for s in ['t0', 't1', 'heldout']
    }
    for stage, pattern in [('t0', '*_t0-*.parquet'), ('t1', '*_t1-*.parquet'), ('heldout', '*_h-*.parquet')]:
        for index_path in sorted((ROOT / 'index').rglob(pattern)):
            tar_path = ROOT / 'data' / index_path.relative_to(ROOT / 'index').with_suffix('.tar')
            assert tar_path.is_file(), tar_path
            table = pq.read_table(index_path, columns=wanted)
            uids = table['uid'].to_pylist()
            n = len(uids)
            offsets = np.empty(n, dtype=np.int64)
            sizes = np.empty(n, dtype=np.int32)
            with tarfile.open(tar_path, 'r:') as archive:
                j = 0
                for member in archive:
                    if not member.name.endswith('.mp3'):
                        continue
                    assert j < n and member.name.rsplit('/', 1)[-1][:-4] == uids[j], (tar_path, j)
                    offsets[j], sizes[j] = member.offset_data, member.size
                    j += 1
                assert j == n, (tar_path, j, n)
            path_id = len(paths)
            paths.append(str(tar_path))
            lang = table['lang'].to_pylist()
            origin = table['origin'].to_pylist()
            cells = np.array([{'de_real': 0, 'de_synth': 1, 'en_real': 2, 'en_synth': 3}[f'{a}_{b}']
                              for a, b in zip(lang, origin)], dtype=np.uint8)
            source_names = table['source'].to_pylist()
            for name in source_names:
                if name not in sources:
                    sources.append(name)
            source_id = np.array([sources.index(name) for name in source_names], dtype=np.uint8)
            label_cols = [np.asarray(table[name].to_numpy(zero_copy_only=False), dtype=np.float32)
                          for name in targets]
            labels = np.stack(label_cols, axis=1)
            chunk = {
                'path': np.full(n, path_id, dtype=np.uint16),
                'offset': offsets, 'size': sizes,
                'dur': np.asarray(table['dur_s'].to_numpy(), dtype=np.float32),
                'uid': np.asarray(uids, dtype='S128'),
                'cell': cells, 'source': source_id, 'y': labels,
            }
            for name, value in chunk.items():
                accum[stage][name].append(value)
            print(stage, tar_path.name, n, flush=True)

    data = {stage: {name: np.concatenate(parts) for name, parts in columns.items()}
            for stage, columns in accum.items()}
    # Shared normalization must not reveal S9-only labels to the S10 run.
    train = data['t0']['y']
    with np.errstate(invalid='ignore'):
        mean = np.nanmean(train, axis=0).astype(np.float32)
        std = np.nanstd(train, axis=0).astype(np.float32)
    assert np.all(np.isfinite(mean)) and np.all(np.isfinite(std))
    std = np.maximum(std, 0.25)
    # Separate held-out UIDs are a release invariant; fail if it changed.
    train_uids = set(np.concatenate([data['t0']['uid'], data['t1']['uid']]).tolist())
    val_uids = set(data['heldout']['uid'].tolist())
    assert len(train_uids) == len(data['t0']['uid']) + len(data['t1']['uid'])
    assert not (train_uids & val_uids)

    selected = balanced_val(data['heldout'], 2048)
    for stage, arrays in data.items():
        if stage == 'heldout':
            arrays = {name: value[selected] for name, value in arrays.items()}
            stage = 'val'
        np.savez(OUT / f'{stage}.npz', **arrays)
    meta = {
        'root': str(ROOT), 'seed': SEED, 'targets': targets, 'paths': paths,
        'sources': sources, 'mean': mean.tolist(), 'std': std.tolist(),
        'rows': {'S10': len(data['t0']['uid']),
                 'S9': len(data['t0']['uid']) + len(data['t1']['uid']),
                 'heldout_full': len(data['heldout']['uid']), 'val': len(selected)},
        'val_policy': '512 per DE/EN x real/synth cell; half uniform and half score-tail weighted, fixed seed',
        'target_groups': {'emotion': [0, 40], 'voicenet': [40, 97], 'other': [97, 100]},
    }
    (OUT / 'manifest.json').write_text(json.dumps(meta, indent=2) + '\n')
    print('DONE', meta['rows'], flush=True)


def balanced_val(heldout: dict[str, np.ndarray], n: int) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    y = heldout['y']
    low = np.nanquantile(y, 0.1, axis=0)
    high = np.nanquantile(y, 0.9, axis=0)
    extreme = np.isfinite(y) & ((y <= low) | (y >= high))
    # Rarer simultaneous tails get more weight. The same indices serve all four runs.
    score = np.sqrt(1.0 + extreme[:, :99].sum(axis=1))
    source_counts = np.bincount(heldout['source'].astype(int))
    score *= 1 / np.sqrt(source_counts[heldout['source'].astype(int)] / source_counts.max())
    out: list[np.ndarray] = []
    assert n % 4 == 0
    for cell in range(4):
        candidates = np.flatnonzero(heldout['cell'] == cell)
        assert len(candidates) >= n // 4
        ordinary = rng.choice(candidates, n // 8, replace=False)
        remaining = np.setdiff1d(candidates, ordinary, assume_unique=True)
        p = score[remaining] / score[remaining].sum()
        tails = rng.choice(remaining, n // 8, replace=False, p=p)
        out.extend([ordinary, tails])
    chosen = np.sort(np.concatenate(out))
    assert len(chosen) == n and len(np.unique(chosen)) == n
    return chosen


if __name__ == '__main__':
    build()
