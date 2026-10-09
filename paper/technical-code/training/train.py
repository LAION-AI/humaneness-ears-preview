"""End-to-end Whisper encoder regression for Ladder S10/S9.

Use under the local fastgen environment. No Whisper decoder is instantiated.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from functools import partial
import io
import json
import math
import os
from pathlib import Path
from typing import Iterator

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from safetensors import safe_open
import torch
import transformers
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, Sampler
from transformers import WhisperConfig, WhisperFeatureExtractor
from transformers.models.whisper.modeling_whisper import WhisperEncoder
from timbre_targets import (OrangeTimbreTeacher, OrangeIdentityTeacher, TimbreChunkWriter,
                            TimbreIndex, TimbreTargetCache)


ROOT = Path('/e/scratch/reformo/schuhmann1_moss/whisper_score_regression')
RUNS = Path('/e/project1/reformo/schuhmann1/whisper_score_regression/runs')
MODEL_CACHE = Path('/e/scratch/reformo/schuhmann1_moss/hf_runtime_whisper_regression/hub')
_EXTRACTOR = None


def snapshot(name: str) -> Path:
    matches = list((MODEL_CACHE / f'models--openai--whisper-{name}' / 'snapshots').glob('*/config.json'))
    assert len(matches) == 1, matches
    return matches[0].parent


class AudioRows(Dataset):
    def __init__(self, arrays: list[dict[str, np.ndarray]], paths: list[str], mean: np.ndarray, std: np.ndarray):
        self.columns = {key: np.concatenate([part[key] for part in arrays]) for key in arrays[0]}
        self.paths = paths
        self.mean, self.std = mean, std
        self.open_files: dict[int, io.BufferedReader] = {}
        self.durations = self.columns['dur']

    def __len__(self) -> int:
        return len(self.durations)

    def __getitem__(self, i: int):
        c = self.columns
        path_id = int(c['path'][i])
        if path_id not in self.open_files:
            self.open_files[path_id] = open(self.paths[path_id], 'rb', buffering=0)
        file = self.open_files[path_id]
        file.seek(int(c['offset'][i]))
        raw = file.read(int(c['size'][i]))
        assert len(raw) == int(c['size'][i]), (i, path_id)
        audio, sr = sf.read(io.BytesIO(raw), dtype='float32', always_2d=False)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        if sr != 16000:
            common = math.gcd(sr, 16000)
            audio = resample_poly(audio, 16000 // common, sr // common).astype(np.float32)
        audio = np.nan_to_num(audio[:480000], copy=False)
        if len(audio) < 1600:
            audio = np.pad(audio, (0, 1600 - len(audio)))
        y = c['y'][i]
        mask = np.isfinite(y)
        target = np.where(mask, (y - self.mean) / self.std, 0).astype(np.float32)
        return audio, target, mask, i


def collate(batch, model_path: str, include_audio: bool = False):
    global _EXTRACTOR
    if _EXTRACTOR is None:
        _EXTRACTOR = WhisperFeatureExtractor.from_pretrained(model_path, local_files_only=True)
    audio, targets, masks, indices = zip(*batch)
    feat = _EXTRACTOR(list(audio), sampling_rate=16000, padding='longest',
                      truncation=True, max_length=480000, return_attention_mask=True,
                      return_tensors='np')
    mel = feat['input_features'].astype(np.float32)
    attention = feat['attention_mask'].astype(np.int64)
    # The second Whisper convolution has stride two.
    if mel.shape[-1] % 2:
        mel = np.pad(mel, ((0, 0), (0, 0), (0, 1)))
        attention = np.pad(attention, ((0, 0), (0, 1)))
    result = (torch.from_numpy(mel), torch.from_numpy(attention),
              torch.from_numpy(np.stack(targets)), torch.from_numpy(np.stack(masks)), indices)
    return result + (audio,) if include_audio else result


def init_worker(_worker_id: int):
    # OpenMP/ACL was initialized before forking in the parent after CUDA setup.
    # Spawned workers and one CPU thread avoid inherited OpenMP barriers.
    torch.set_num_threads(1)


class BucketBatches(Sampler[list[int]]):
    def __init__(self, duration: np.ndarray, batch: int, world: int, rank: int, seed: int):
        self.duration, self.batch, self.world, self.rank, self.seed = duration, batch, world, rank, seed
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.duration) // (self.batch * self.world)

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        order = rng.permutation(len(self.duration))
        width = self.batch * self.world
        batches = []
        for start in range(0, len(order), width * 32):
            pool = order[start:start + width * 32]
            pool = pool[np.argsort(self.duration[pool], kind='stable')]
            for j in range(0, len(pool) - width + 1, width):
                batches.append(pool[j:j + width])
        rng.shuffle(batches)
        for group in batches:
            yield group[self.rank * self.batch:(self.rank + 1) * self.batch].tolist()


class EncoderRegressor(nn.Module):
    def __init__(self, model_path: Path, outputs: int, timbre: bool = False,
                 identity: bool = False):
        super().__init__()
        config = WhisperConfig.from_pretrained(model_path, local_files_only=True)
        self.encoder = WhisperEncoder(config)
        with safe_open(model_path / 'model.safetensors', framework='pt', device='cpu') as file:
            prefix = 'model.encoder.'
            state = {key[len(prefix):]: file.get_tensor(key) for key in file.keys() if key.startswith(prefix)}
        missing, unexpected = self.encoder.load_state_dict(state, strict=True)
        assert not missing and not unexpected, (missing, unexpected)
        self.encoder.embed_positions.weight.requires_grad_(True)
        dim = config.d_model
        self.head = nn.Sequential(nn.LayerNorm(dim * 4), nn.Linear(dim * 4, 512), nn.GELU(),
                                  nn.Dropout(0.1), nn.Linear(512, 256), nn.GELU(),
                                  nn.Dropout(0.1), nn.Linear(256, outputs))
        self.timbre_head = (nn.Sequential(nn.LayerNorm(dim * 4), nn.Linear(dim * 4, 512),
                                          nn.GELU(), nn.Dropout(0.1), nn.Linear(512, 128))
                            if timbre else None)
        self.identity_head = (nn.Sequential(nn.LayerNorm(dim * 4), nn.Linear(dim * 4, 512),
                                            nn.GELU(), nn.Dropout(0.1), nn.Linear(512, 250))
                              if identity else None)

    def forward(self, mel: torch.Tensor, frame_mask: torch.Tensor) -> torch.Tensor:
        # Identical pretrained encoder blocks, with positional embeddings sliced to the
        # longest item in this duration bucket. This avoids 30 s padding for every clip.
        enc = self.encoder
        h = F.gelu(enc.conv1(mel))
        h = F.gelu(enc.conv2(h)).transpose(1, 2)
        length = h.shape[1]
        assert length <= enc.max_source_positions
        h = h + enc.embed_positions(torch.arange(length, device=h.device))
        valid = torch.arange(length, device=h.device)[None, :] < ((frame_mask.sum(1) + 1) // 2)[:, None]
        # Whisper's stock forward ignores masks; here padded frames cannot act as keys.
        attention = torch.zeros((len(h), 1, 1, length), device=h.device, dtype=h.dtype)
        attention.masked_fill_(~valid[:, None, None, :], torch.finfo(h.dtype).min)
        for layer in enc.layers:
            h = layer(h, attention)
        h = enc.layer_norm(h).float()
        mask = valid[:, :, None]
        count = mask.sum(1).clamp_min(1)
        mean = (h * mask).sum(1) / count
        variance = ((h - mean[:, None, :]).square() * mask).sum(1) / count
        minimum = h.masked_fill(~mask, torch.inf).amin(1)
        maximum = h.masked_fill(~mask, -torch.inf).amax(1)
        pooled = torch.cat([mean, minimum, maximum, variance.clamp_min(1e-8).sqrt()], dim=-1)
        scores = self.head(pooled)
        if self.timbre_head is None and self.identity_head is None:
            return scores
        vectors = {}
        if self.timbre_head is not None:
            vectors['timbre'] = F.normalize(self.timbre_head(pooled).float(), dim=-1)
        if self.identity_head is not None:
            vectors['identity'] = F.normalize(self.identity_head(pooled).float(), dim=-1)
        return scores, vectors


def regression_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    element = F.huber_loss(pred.float(), target.float(), reduction='none', delta=1.0)
    return (element * mask).sum() / mask.sum().clamp_min(1)


def embedding_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # Unit vectors: cosine is the primary metric. A small Huber term anchors
    # individual coordinates without letting dimensions swamp the score loss.
    target = F.normalize(target.float(), dim=-1)
    return (1 - (pred.float() * target).sum(-1)).mean() + 0.1 * F.huber_loss(
        pred.float(), target, delta=0.1)


def embedding_targets(caches: dict[str, TimbreTargetCache], columns: dict,
                      indices, audio, device):
    uids = [columns['uid'][i].decode('utf-8') for i in indices]
    return {name: torch.from_numpy(cache.get_or_compute(uids, audio).copy()).to(
        device, dtype=torch.float32) for name, cache in caches.items()}


@torch.no_grad()
def evaluate(model, loader, mean: torch.Tensor, std: torch.Tensor, device, world: int,
             max_batches: int = 0, embedding_caches: dict[str, TimbreTargetCache] | None = None,
             columns: dict | None = None):
    model.eval()
    n = len(mean)
    sums = torch.zeros((9, n), dtype=torch.float64, device=device)
    embedding_caches = embedding_caches or {}
    embedding_sums = {name: torch.zeros(3, dtype=torch.float64, device=device)
                      for name in embedding_caches}
    for step, batch in enumerate(loader):
        if max_batches and step >= max_batches:
            break
        mel, frame_mask, target, mask, indices = batch[:5]
        mel, frame_mask = mel.to(device), frame_mask.to(device)
        target, mask = target.to(device), mask.to(device)
        if embedding_caches:
            assert columns is not None
            target_vectors = embedding_targets(embedding_caches, columns, indices,
                                               batch[5], device)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            output = model(mel, frame_mask)
        if embedding_caches:
            pred, pred_vectors = output
            for name, target_vector in target_vectors.items():
                pred_vector = pred_vectors[name]
                stats = embedding_sums[name]
                stats[0] += len(target_vector)
                stats[1] += (pred_vector * F.normalize(target_vector, dim=-1)).sum().double()
                stats[2] += F.huber_loss(pred_vector, target_vector, delta=0.1,
                                         reduction='sum').double() / target_vector.shape[1]
        else:
            pred = output
        pred, target, mask = pred.double(), target.double(), mask.double()
        y = target * std + mean
        p = pred * std + mean
        sums[0] += mask.sum(0)
        sums[1] += (F.huber_loss(pred, target, reduction='none', delta=1.0) * mask).sum(0)
        sums[2] += ((p - y).abs() * mask).sum(0)
        sums[3] += ((p - y).square() * mask).sum(0)
        sums[4] += (y * mask).sum(0)
        sums[5] += (y.square() * mask).sum(0)
        sums[6] += (p * mask).sum(0)
        sums[7] += (p.square() * mask).sum(0)
        sums[8] += (p * y * mask).sum(0)
    if world > 1:
        torch.distributed.all_reduce(sums)
        for stats in embedding_sums.values():
            torch.distributed.all_reduce(stats)
    count = sums[0].clamp_min(1)
    mae = sums[2] / count
    huber = sums[1] / count
    y_var = (sums[5] - sums[4].square() / count).clamp_min(1e-10)
    p_var = (sums[7] - sums[6].square() / count).clamp_min(1e-10)
    corr = (sums[8] - sums[4] * sums[6] / count) / (y_var * p_var).sqrt()
    r2 = 1 - sums[3] / y_var
    result = {'count': sums[0].int().tolist(), 'mae': mae.tolist(),
              'huber_z': huber.tolist(), 'pearson': corr.tolist(), 'r2': r2.tolist()}
    for name, a, b in [('emotion', 0, 40), ('voicenet', 40, 97), ('other', 97, 100)]:
        result[name] = {'mae': float(mae[a:b].mean()), 'huber_z': float(huber[a:b].mean()),
                        'pearson': float(corr[a:b].mean()), 'r2': float(r2[a:b].mean())}
    result['huber_z_macro'] = float(huber.mean())
    for name, stats in embedding_sums.items():
        result[f'{name}_count'] = int(stats[0])
        result[f'{name}_cosine'] = float(stats[1] / stats[0].clamp_min(1))
        result[f'{name}_huber'] = float(stats[2] / stats[0].clamp_min(1))
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', choices=['tiny', 'base'], required=True)
    p.add_argument('--stage', choices=['S10', 'S9'], required=True)
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--batch-per-gpu', type=int, default=16)
    p.add_argument('--accum', type=int, default=2)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--encoder-lr', type=float, default=2e-5)
    p.add_argument('--head-lr', type=float, default=3e-4)
    p.add_argument('--timbre', action='store_true', help='Train the 128-D Orange TBR head')
    p.add_argument('--timbre-weight', type=float, default=0.2)
    p.add_argument('--timbre-index', type=Path, help='Sorted index of precomputed Emolia embeddings')
    p.add_argument('--timbre-generated-index', type=Path, help='Index of earlier generated chunks')
    p.add_argument('--timbre-cache-dir', type=Path, default=ROOT / 'timbre' / 'chunks')
    p.add_argument('--identity', action='store_true', help='Train the 250-D Orange ID head')
    p.add_argument('--identity-weight', type=float, default=0.2)
    p.add_argument('--identity-index', type=Path, help='Sorted index of precomputed KartoffelPhon embeddings')
    p.add_argument('--identity-generated-index', type=Path, help='Index of earlier generated ID chunks')
    p.add_argument('--identity-cache-dir', type=Path, default=ROOT / 'identity' / 'chunks')
    p.add_argument('--max-train-batches', type=int, default=0)
    p.add_argument('--max-val-batches', type=int, default=0)
    p.add_argument('--out', type=Path)
    args = p.parse_args()
    assert torch.cuda.is_available(), 'Run on a Booster GPU node'
    world = int(os.environ.get('WORLD_SIZE', os.environ.get('SLURM_NTASKS', '1')))
    rank = int(os.environ.get('RANK', os.environ.get('SLURM_PROCID', '0')))
    if world > 1:
        os.environ.setdefault('RANK', str(rank))
        os.environ.setdefault('WORLD_SIZE', str(world))
        torch.distributed.init_process_group('nccl')
    torch.cuda.set_device(0)
    torch.set_num_threads(2)
    torch.manual_seed(20260928)
    device = torch.device('cuda:0')
    meta = json.loads((ROOT / 'manifest.json').read_text())
    mean = torch.tensor(meta['mean'], dtype=torch.float64, device=device)
    std = torch.tensor(meta['std'], dtype=torch.float64, device=device)
    def load(name):
        with np.load(ROOT / f'{name}.npz') as file:
            return {key: file[key] for key in file.files}
    train_parts = [load('t0')] + ([load('t1')] if args.stage == 'S9' else [])
    train_data = AudioRows(train_parts, meta['paths'], np.asarray(meta['mean']), np.asarray(meta['std']))
    val_data = AudioRows([load('val')], meta['paths'], np.asarray(meta['mean']), np.asarray(meta['std']))
    path = snapshot(args.model)
    sampler = BucketBatches(train_data.durations, args.batch_per_gpu, world, rank, meta['seed'])
    collate_fn = partial(collate, model_path=str(path), include_audio=args.timbre or args.identity)
    loader_options = {'num_workers': args.workers, 'pin_memory': True,
                      'persistent_workers': args.workers > 0, 'collate_fn': collate_fn}
    if args.workers:
        loader_options.update({'multiprocessing_context': 'spawn', 'worker_init_fn': init_worker})
    train_loader = DataLoader(train_data, batch_sampler=sampler, **loader_options)
    val_indices = np.arange(rank, len(val_data), world)
    val_indices = val_indices[np.argsort(val_data.durations[val_indices])]
    val_batches = [val_indices[i:i + args.batch_per_gpu].tolist()
                   for i in range(0, len(val_indices), args.batch_per_gpu)]
    val_loader = DataLoader(val_data, batch_sampler=val_batches, **loader_options)
    core = EncoderRegressor(path, len(meta['targets']), timbre=args.timbre,
                            identity=args.identity).to(device)
    model = DDP(core, device_ids=[0], broadcast_buffers=False) if world > 1 else core
    parameter_groups = [
        {'params': core.encoder.parameters(), 'lr': args.encoder_lr},
        {'params': core.head.parameters(), 'lr': args.head_lr}]
    if args.timbre:
        parameter_groups.append({'params': core.timbre_head.parameters(), 'lr': args.head_lr})
    if args.identity:
        parameter_groups.append({'params': core.identity_head.parameters(), 'lr': args.head_lr})
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=0.01)
    embedding_caches = {}
    if args.timbre:
        assert args.timbre_index is not None and (args.timbre_index / 'manifest.json').exists(), \
            'Build the Emolia timbre index first: python timbre_targets.py index-emolia --output PATH'
        teacher = OrangeTimbreTeacher().to(device).eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
        precomputed = TimbreIndex(args.timbre_index, expected_dim=128)
        generated = TimbreIndex(args.timbre_generated_index, expected_dim=128) if args.timbre_generated_index else None
        writer = TimbreChunkWriter(args.timbre_cache_dir, rank, flush_rows=4096)
        embedding_caches['timbre'] = TimbreTargetCache(precomputed, generated, teacher, writer, device)
    if args.identity:
        assert args.identity_index is not None and (args.identity_index / 'manifest.json').exists(), \
            'Build the Orange ID index first: python timbre_targets.py index-identity --output PATH'
        teacher = OrangeIdentityTeacher().to(device).eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
        precomputed = TimbreIndex(args.identity_index, expected_dim=250)
        generated = TimbreIndex(args.identity_generated_index, expected_dim=250) if args.identity_generated_index else None
        writer = TimbreChunkWriter(args.identity_cache_dir, rank, flush_rows=4096,
                                   dim=250, model='Orange/Speaker-wavLM-id',
                                   revision='abbb3c7b8d220ceebc33d9bd3bb5aedb580342c')
        embedding_caches['identity'] = TimbreTargetCache(precomputed, generated, teacher, writer, device)
    batches = min(len(sampler), args.max_train_batches or len(sampler))
    total_steps = args.epochs * math.ceil(batches / args.accum)
    warmup = max(1, int(total_steps * 0.05))
    def schedule(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    suffix = '_timbre_identity' if args.timbre and args.identity else \
             '_timbre' if args.timbre else '_identity' if args.identity else ''
    out = args.out or RUNS / f'{args.model}_{args.stage.lower()}{suffix}'
    out.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        (out / 'config.json').write_text(json.dumps(vars(args) | {
            'out': str(out), 'world': world, 'train_rows': len(train_data),
            'val_rows': len(val_data), 'model_snapshot': str(path),
            'torch_version': torch.__version__, 'transformers_version': transformers.__version__,
        }, indent=2, default=str) + '\n')
        print('CONFIG', args.model, args.stage, 'train', len(train_data), 'val', len(val_data),
              'world', world, 'batches', batches, flush=True)
    best = math.inf
    updates = 0
    for epoch in range(args.epochs):
        sampler.epoch = epoch
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running = 0.0
        for step, batch in enumerate(train_loader):
            if step >= batches:
                break
            mel, frame_mask, target, mask, indices = batch[:5]
            mel, frame_mask = mel.to(device, non_blocking=True), frame_mask.to(device, non_blocking=True)
            target, mask = target.to(device, non_blocking=True), mask.to(device, non_blocking=True)
            if embedding_caches:
                target_vectors = embedding_targets(embedding_caches, train_data.columns,
                                                   indices, batch[5], device)
            sync = ((step + 1) % args.accum == 0) or (step + 1 == batches)
            context = nullcontext() if sync or world == 1 else model.no_sync()
            with context:
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    output = model(mel, frame_mask)
                    if embedding_caches:
                        pred, pred_vectors = output
                        loss = regression_loss(pred, target, mask)
                        for name, target_vector in target_vectors.items():
                            weight = args.timbre_weight if name == 'timbre' else args.identity_weight
                            loss = loss + weight * embedding_loss(pred_vectors[name], target_vector)
                    else:
                        loss = regression_loss(output, target, mask)
                group_start = (step // args.accum) * args.accum
                group_size = min(args.accum, batches - group_start)
                (loss / group_size).backward()
            running += float(loss.detach())
            if sync:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                updates += 1
                if rank == 0 and (updates % 50 == 0 or step + 1 == batches):
                    print('STEP', epoch + 1, updates, step + 1, 'loss', running / (step + 1), flush=True)
        metrics = evaluate(core, val_loader, mean, std, device, world, args.max_val_batches,
                           embedding_caches, val_data.columns)
        for name, cache in embedding_caches.items():
            cache.writer.flush()
            count = torch.tensor(cache.computed, dtype=torch.long, device=device)
            if world > 1:
                torch.distributed.all_reduce(count)
            metrics[f'{name}_new_saved'] = int(count)
        if rank == 0:
            selection = metrics['huber_z_macro']
            for name in embedding_caches:
                weight = args.timbre_weight if name == 'timbre' else args.identity_weight
                selection += weight * (1 - metrics[f'{name}_cosine'])
            metrics.update({'epoch': epoch + 1, 'train_loss': running / batches,
                            'selection_loss': selection,
                            'model': args.model, 'stage': args.stage, 'updates': updates})
            if not embedding_caches:
                metrics['train_huber_z'] = running / batches
            with (out / 'metrics.jsonl').open('a') as file:
                file.write(json.dumps(metrics) + '\n')
            print('EPOCH', epoch + 1, 'train', metrics['train_loss'],
                  'val', metrics['huber_z_macro'], 'emo', metrics['emotion'],
                  'vn', metrics['voicenet'], flush=True)
            if selection < best:
                best = selection
                torch.save({'model': core.state_dict(), 'epoch': epoch + 1,
                            'targets': meta['targets'], 'mean': meta['mean'], 'std': meta['std']},
                           out / 'best.pt')
        if world > 1:
            torch.distributed.barrier()
    if world > 1:
        torch.distributed.destroy_process_group()
    if rank == 0 and not args.max_train_batches and not embedding_caches:
        from report import main as write_comparison
        write_comparison()


if __name__ == '__main__':
    main()
