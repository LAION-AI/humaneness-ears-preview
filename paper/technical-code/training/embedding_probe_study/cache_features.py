#!/usr/bin/env python3
"""Four independent inference ranks; immutable resumable WebDataset caches."""
import argparse
import io
import json
import os
import tarfile
import time
import hashlib
from pathlib import Path
import numpy as np
import torch
from study_paths import ROOT, read, write
from backbones import FrozenEncoder
from study_data import SourceData


def member(tf, name, payload):
    offset = tf.offset + 512
    info = tarfile.TarInfo(name)
    info.size, info.mtime = len(payload), 0
    tf.addfile(info, io.BytesIO(payload))
    return offset, len(payload)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--domains', nargs='+')
    args = ap.parse_args()
    cfg = read(ROOT / 'study.json')
    spec = next(m for m in cfg['models'] if m['id'] == args.model)
    rank, world = int(os.environ.get('LOCAL_RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    torch.set_num_threads(2)
    torch.cuda.set_device(rank)
    encoder = FrozenEncoder(spec, torch.device('cuda', rank))
    out = ROOT / ('smoke' if args.smoke else 'features') / args.model
    out.mkdir(parents=True, exist_ok=True)
    domains = args.domains or ['ladder', 'p3_train', 'p3_validation', 'p3_test', 'gemini', *cfg['benchmarks']]
    for domain in domains:
        data = SourceData(domain)
        positions = list(range(rank, len(data), world))
        if args.smoke:
            positions = positions[:4]
        completed = 0
        started = time.monotonic()
        for chunk_number, offset in enumerate(range(0, len(positions), 512)):
            name = f'{domain}-rank{rank}-{chunk_number:05d}'
            destination = out / (name + '.tar')
            manifest = out / (name + '.jsonl')
            commit = out / (name + '.commit.json')
            chunk = positions[offset:offset + 512]
            if commit.exists():
                completed += len(chunk)
                continue
            rows = []
            temp = destination.with_suffix('.tar.incomplete')
            with tarfile.open(temp, 'w', format=tarfile.USTAR_FORMAT) as tf:
                for step in range(0, len(chunk), spec['batch']):
                    items = [data[p] for p in chunk[step:step + spec['batch']]]
                    waves = [row[1] for row in items]
                    encoded = encoder.encode(waves)
                    for local_index, ((key, wave, meta, targets), features) in enumerate(zip(items, encoded)):
                        signature = hashlib.sha256(key.encode()).hexdigest()
                        buffer = io.BytesIO()
                        np.savez(buffer, **features, **targets)
                        payload = buffer.getvalue()
                        position, size = member(tf, signature + '.npz', payload)
                        rows.append({**meta, 'feature_key': key, 'source_position': int(chunk[step + local_index]),
                                     'feature_tar': str(destination), 'feature_offset': position, 'feature_size': size,
                                     'feature_sha256': hashlib.sha256(payload).hexdigest(),
                                     'decoded_waveform_sha256': hashlib.sha256(wave.astype('<f4').tobytes()).hexdigest(),
                                     'native_dim': spec['native_dim'], 'duration_s': float(features['duration_s']),
                                     'frame_count': len(features['frame_features'])})
            temp.replace(destination)
            temporary = manifest.with_suffix('.jsonl.incomplete')
            temporary.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
            temporary.replace(manifest)
            write(commit, {'clips': len(rows), 'tar': str(destination), 'index': str(manifest),
                           'frozen_backbone': spec, 'feature_schema': 'pooled-native+temporal-fixed64-v1'})
            completed += len(rows)
            write(out / f'{domain}-rank{rank}-progress.json', {'completed': completed, 'total': len(positions),
                  'clips_per_second': completed / max(.001, time.monotonic() - started), 'model': args.model, 'domain': domain})
            print(args.model, domain, rank, completed, '/', len(positions), flush=True)
        write(out / f'{domain}-rank{rank}-COMPLETE.json', {'clips': completed, 'total': len(positions), 'world': world})
    if args.smoke:
        # Exercise real cached Ladder/P3 targets, the loss, and a head backward
        # pass before committing a full encoder-cache allocation.
        import sys
        from study_paths import CODE, RELEASE
        from cache_dataset import collate
        from probe_model import Probe
        sys.path.insert(0, str(CODE))
        from train_layered_curriculum import loss_terms
        samples = []
        for domain in ('ladder', 'p3_train'):
            source = SourceData(domain)
            _, wave, _, targets = source[rank]
            features = encoder.encode([wave])[0]
            ticks = (np.arange(len(targets['frame'])) + .5) / 50
            native = features['frame_features']
            samples.append({**targets, 'embedding': np.pad(features['embedding'].astype(np.float32)[:256],
                           (0, max(0, 256-len(features['embedding'])))),
                           'frame_features': np.stack([np.interp(ticks, features['frame_times_s'], native[:, i])
                                                       for i in range(64)], axis=1).astype(np.float32),
                           'feature_key': domain + '-contract'})
        batch = {k: v.to(encoder.device) if torch.is_tensor(v) else v for k,v in collate(samples).items()}
        classes = read(RELEASE / 'classes.json')['names']
        probe = Probe(len(classes)).to(encoder.device)
        result = probe(batch['embedding'], batch['frame_features'], batch['event_starts'], batch['event_ends'])
        loss = sum(loss_terms(result, batch, torch.ones(len(classes), device=encoder.device)).values())
        loss.backward()
        if not torch.isfinite(loss) or not any(p.grad is not None and torch.isfinite(p.grad).all() for p in probe.parameters()):
            raise RuntimeError('Probe backward contract failed')
        assert not any(p.requires_grad or p.grad is not None for p in encoder.model.parameters())
        write(out / f'CONTRACT-rank{rank}.json', {'loss': float(loss.detach()), 'finite_gradient': True,
                                               'frozen_backbone': True, 'classes': len(classes)})


if __name__ == '__main__':
    main()
