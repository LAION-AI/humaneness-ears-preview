#!/usr/bin/env python3
"""Continuous Whisper Tiny/Base S1→S10 multitask training with 10% frozen P3.

One optimizer and one cosine schedule span all ten stages. Ladder has the 100
released scores and any valid burst intervals; the frozen P3 sidecars supply
the additional teacher targets. Missing targets remain masked, not imputed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, Subset

from full_ladder_dataset import (LadderAudioDataset, MixedCollator, MixedDataset,
                                 StageBatches)
from multitask_p3_smoke import GROUPS, MultiTaskWhisper
from p3_frozen_loader import P3FrozenDataset
from prepare_full_ladder import OUT
from timbre_targets import TimbreIndex
from train import AudioRows, ROOT as OLD_ROOT, RUNS, snapshot


P3_ROOT = Path('/e/scratch/reformo/schuhmann1_moss/vocalburst_p3/frozen_1120k')
TIMBRE_INDEX = OLD_ROOT / 'timbre' / 'emolia_index'
IDENTITY_INDEX = OLD_ROOT / 'identity' / 'kartoffelphon_index'
_STOP = False


def request_stop(_signal, _frame):
    global _STOP
    _STOP = True


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalization() -> tuple[np.ndarray, np.ndarray, list[str]]:
    ladder = json.loads((OLD_ROOT / 'manifest.json').read_text())
    p3 = json.loads((P3_ROOT / 'score_normalization.json').read_text())
    if p3['score_names'][:100] != ladder['targets']:
        raise RuntimeError('Ladder and P3 target order differ')
    mean = np.asarray(p3['mean'], np.float32)
    std = np.asarray(p3['std'], np.float32)
    mean[:100] = ladder['mean']
    std[:100] = ladder['std']
    return mean, std, p3['score_names']


class LadderVal(Dataset):
    def __init__(self, mean: np.ndarray, std: np.ndarray):
        manifest = json.loads((OLD_ROOT / 'manifest.json').read_text())
        with np.load(OLD_ROOT / 'val.npz') as source:
            rows = {key: source[key] for key in source.files}
        self.rows = AudioRows([rows], manifest['paths'], mean[:100], std[:100])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        audio, score100, mask100, _ = self.rows[i]
        score = np.zeros(192, np.float32)
        mask = np.zeros(192, np.bool_)
        score[:100], mask[:100] = score100, mask100
        uid = self.rows.columns['uid'][i].decode('utf-8')
        n_frames = min(1500, (len(audio) + 319) // 320)
        return {'key': uid, 'uid': uid, 'domain': 'ladder', 'audio': audio,
                'scores': score, 'score_mask': mask,
                'frame': np.zeros(n_frames, np.float32),
                'frame_mask': np.zeros(n_frames, np.float32),
                'timbre': np.zeros(128, np.float32),
                'identity': np.zeros(250, np.float32),
                'timbre_valid': False, 'identity_valid': False}


class P3Val(Dataset):
    def __init__(self, p3: P3FrozenDataset):
        self.p3 = p3

    def __len__(self):
        return len(self.p3)

    def __getitem__(self, i: int) -> dict:
        result = self.p3[i]
        result['uid'] = result['key']
        result['domain'] = 'p3'
        result['timbre_valid'] = result['speaker_valid']
        result['identity_valid'] = result['speaker_valid']
        return result


def collate_options(model_path: Path, workers: int) -> dict:
    result = {'num_workers': workers, 'pin_memory': True,
              'persistent_workers': workers > 0,
              'collate_fn': MixedCollator(model_path)}
    if workers:
        result['multiprocessing_context'] = 'spawn'
    return result


def fill_precomputed(batch: dict, indices: dict[str, TimbreIndex]) -> None:
    positions = [i for i, (domain, enabled) in enumerate(zip(batch['domain'],
                                                            batch['enabled']))
                 if domain == 'ladder' and bool(enabled)]
    if not positions:
        return
    uids = [batch['uid'][i] for i in positions]
    for name in ('timbre', 'identity'):
        values, valid = indices[name].lookup(uids)
        for source, use in enumerate(valid):
            if use:
                dest = positions[source]
                batch[name][dest] = torch.from_numpy(values[source].astype(np.float32))
                batch[f'{name}_valid'][dest] = True


def gpu_batch(batch: dict, device: torch.device) -> tuple[torch.Tensor, ...]:
    fields = ('mel', 'mel_mask', 'scores', 'score_mask', 'frame', 'frame_mask',
              'timbre', 'identity', 'timbre_valid', 'identity_valid')
    return tuple(batch[name].to(device, non_blocking=True) for name in fields)


def loss_terms(output: dict, batch: tuple[torch.Tensor, ...]) -> dict[str, torch.Tensor]:
    _, _, targets, score_mask, frame, frame_mask, timbre, identity, timbre_valid, identity_valid = batch
    pred = output['scores'].float()
    result = {}
    for name, (start, stop, weight) in GROUPS.items():
        valid = score_mask[:, start:stop].float()
        huber = F.huber_loss(pred[:, start:stop], targets[:, start:stop].float(),
                             delta=1., reduction='none')
        result[name] = weight * (huber * valid).sum() / valid.sum().clamp_min(1)
    valid = frame_mask.float()
    logits = output['frame'].float()
    bce = F.binary_cross_entropy_with_logits(logits, frame.float(),
                                              pos_weight=torch.tensor(5., device=frame.device),
                                              reduction='none')
    result['frame_bce'] = .5 * (bce * valid).sum() / valid.sum().clamp_min(1)
    prob = logits.sigmoid() * valid
    intersection = (prob * frame).sum()
    result['frame_dice'] = .25 * (1 - (2 * intersection + 1) /
                                     (prob.sum() + (frame * valid).sum() + 1))
    for name, target, mask in (('timbre', timbre, timbre_valid),
                               ('identity', identity, identity_valid)):
        if mask.any():
            predicted = output[name][mask].float()
            reference = F.normalize(target[mask].float(), dim=-1)
            result[name] = .2 * ((1 - (predicted * reference).sum(-1)).mean() +
                                 .1 * F.huber_loss(predicted, reference, delta=.1))
        else:
            result[name] = output[name].sum() * 0
    return result


def lr_factor(update: int, total: int, warmup: int, floor: float) -> float:
    if update < warmup:
        return (update + 1) / max(1, warmup)
    progress = min(1., (update - warmup) / max(1, total - warmup))
    return floor + (1 - floor) * .5 * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def validate(model, sources: dict[str, DataLoader], device: torch.device,
             world: int, indices: dict[str, TimbreIndex]) -> dict:
    model.eval()
    report = {}
    for source, loader in sources.items():
        names = list(GROUPS) + ['frame_bce', 'frame_dice', 'timbre', 'identity']
        sums = torch.zeros((len(names), 2), dtype=torch.float64, device=device)
        counts = torch.zeros((4,), dtype=torch.float64, device=device)
        for raw in loader:
            fill_precomputed(raw, indices)
            batch = gpu_batch(raw, device)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                output = model(batch[0], batch[1])
                terms = loss_terms(output, batch)
            for i, name in enumerate(names):
                has_targets = (batch[3][:, GROUPS[name][0]:GROUPS[name][1]].any()
                               if name in GROUPS else
                               batch[5].any() if name.startswith('frame_') else
                               batch[8].any() if name == 'timbre' else batch[9].any())
                if has_targets:
                    sums[i, 0] += terms[name].double() * len(raw['uid'])
                    sums[i, 1] += len(raw['uid'])
            if source == 'p3':
                use = batch[5].bool()
                predicted = (output['frame'].float().sigmoid() > .5) & use
                truth = batch[4].bool() & use
                counts[0] += (predicted & truth).sum()
                counts[1] += (predicted & ~truth).sum()
                counts[2] += (~predicted & truth).sum()
                counts[3] += use.sum()
        if world > 1:
            torch.distributed.all_reduce(sums)
            torch.distributed.all_reduce(counts)
        result = {name: float(sums[i, 0] / sums[i, 1]) if sums[i, 1] else None
                  for i, name in enumerate(names)}
        result['weighted_loss_sum'] = sum(x for x in result.values() if x is not None)
        if source == 'p3':
            tp, fp, fn = counts[:3].tolist()
            result['frame_precision_at_0.5'] = tp / max(1., tp + fp)
            result['frame_recall_at_0.5'] = tp / max(1., tp + fn)
            result['valid_frame_count'] = int(counts[3])
        report[source] = result
    return report


def save(out: Path, model: MultiTaskWhisper, optimizer, *, stage_index: int,
         batch_index: int, update: int, hashes: dict, best: float,
         args: argparse.Namespace):
    state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
             'stage_index': stage_index, 'batch_index': batch_index,
             'update': update, 'hashes': hashes, 'best': best,
             'config': vars(args)}
    tmp = out / 'latest.pt.incomplete'
    torch.save(state, tmp)
    os.replace(tmp, out / 'latest.pt')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=('tiny', 'base'), required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--batch-per-gpu', type=int, default=8)
    parser.add_argument('--accum', type=int, default=2)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--encoder-lr', type=float, default=2e-5)
    parser.add_argument('--head-lr', type=float, default=3e-4)
    parser.add_argument('--min-lr-ratio', type=float, default=.10)
    parser.add_argument('--warmup-ratio', type=float, default=.02)
    parser.add_argument('--checkpoint-every', type=int, default=5000)
    parser.add_argument('--max-updates', type=int, default=0,
                        help='Smoke mode: stop after this many optimizer updates')
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('training requires Booster GPUs')
    world = int(os.environ.get('WORLD_SIZE', os.environ.get('SLURM_NTASKS', '1')))
    rank = int(os.environ.get('RANK', os.environ.get('SLURM_PROCID', '0')))
    if world > 1:
        os.environ.setdefault('RANK', str(rank))
        os.environ.setdefault('WORLD_SIZE', str(world))
        torch.distributed.init_process_group('nccl')
    torch.cuda.set_device(0)
    torch.set_num_threads(2)
    torch.manual_seed(640913)
    signal.signal(signal.SIGUSR1, request_stop)
    device = torch.device('cuda:0')
    schedule_path = P3_ROOT / 'curriculum_mix_schedule.json'
    index_path = OUT / 'manifest.json'
    p3_index_path = P3_ROOT / 'training_index_report.json'
    hashes = {'schedule': digest(schedule_path), 'ladder_index': digest(index_path),
              'p3_index': digest(p3_index_path)}
    schedule = json.loads(schedule_path.read_text())
    if (schedule['schema_version'] != 'p3-frozen-curriculum-v2' or
            schedule['target_p3_fraction_per_stage'] != .10):
        raise RuntimeError('wrong P3 curriculum schedule')
    mean, std, names = normalization()
    ladder = LadderAudioDataset(mean, std)
    p3_train = P3FrozenDataset('train', score_mean=mean, score_std=std)
    if len(p3_train) != schedule['p3_train_unique']:
        raise RuntimeError('P3 train split changed after schedule creation')
    mixed = MixedDataset(ladder, p3_train)
    path = snapshot(args.model)
    options = collate_options(path, args.workers)
    val_ladder = LadderVal(mean, std)
    val_p3 = P3Val(P3FrozenDataset('validation', score_mean=mean, score_std=std))
    validation = {
        'ladder': DataLoader(Subset(val_ladder, range(rank, len(val_ladder), world)),
                             batch_size=args.batch_per_gpu, **options),
        'p3': DataLoader(Subset(val_p3, range(rank, len(val_p3), world)),
                         batch_size=args.batch_per_gpu, **options)}
    indices = {'timbre': TimbreIndex(TIMBRE_INDEX, expected_dim=128),
               'identity': TimbreIndex(IDENTITY_INDEX, expected_dim=250)}
    core = MultiTaskWhisper(path).to(device)
    model = DDP(core, device_ids=[0], broadcast_buffers=False) if world > 1 else core
    heads = [parameter for name, parameter in core.named_parameters()
             if not name.startswith('encoder.')]
    optimizer = torch.optim.AdamW([
        {'params': core.encoder.parameters(), 'lr': args.encoder_lr,
         'base_lr': args.encoder_lr},
        {'params': heads, 'lr': args.head_lr, 'base_lr': args.head_lr}],
        weight_decay=.01)
    width = world * args.batch_per_gpu
    total_updates = sum(math.ceil((math.ceil(stage['ladder'] / width) +
                                   math.ceil(stage['p3'] / width)) / args.accum)
                        for stage in schedule['stages'])
    warmup = round(total_updates * args.warmup_ratio)
    args.out.mkdir(parents=True, exist_ok=True)
    start_stage = start_batch = update = 0
    best = math.inf
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        if checkpoint['hashes'] != hashes or checkpoint['config']['model'] != args.model:
            raise RuntimeError('resume checkpoint/data/model mismatch')
        core.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_stage = int(checkpoint['stage_index'])
        start_batch = int(checkpoint['batch_index'])
        update = int(checkpoint['update'])
        best = float(checkpoint['best'])
    if rank == 0:
        config = dict(vars(args), world=world, total_updates=total_updates,
                      warmup_updates=warmup, score_names=names, hashes=hashes,
                      ladder_speaker_targets='precomputed Emolia TBR and KartoffelPhon ID only; missing masked',
                      ladder_quality_targets='missing and masked; P3 quality sidecars active')
        (args.out / 'config.json').write_text(json.dumps(config, indent=2, default=str) + '\n')
        print('CONFIG', args.model, 'world', world, 'updates', total_updates,
              'resume', start_stage, start_batch, update, flush=True)
    for stage_index in range(start_stage, len(schedule['stages'])):
        stage = schedule['stages'][stage_index]
        sampler = StageBatches(ladder, schedule, stage_index,
                               batch_per_rank=args.batch_per_gpu,
                               world=world, rank=rank,
                               start_batch=start_batch if stage_index == start_stage else 0)
        loader = DataLoader(mixed, batch_sampler=sampler, **options)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for local_step, raw in enumerate(loader):
            batch_index = sampler.start_batch + local_step
            fill_precomputed(raw, indices)
            batch = gpu_batch(raw, device)
            last = batch_index + 1 == sampler.total_batches
            sync = (batch_index + 1) % args.accum == 0 or last
            group_start = (batch_index // args.accum) * args.accum
            group_size = min(args.accum, sampler.total_batches - group_start)
            context = nullcontext() if sync or world == 1 else model.no_sync()
            with context:
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    output = model(batch[0], batch[1])
                    terms = loss_terms(output, batch)
                    loss = sum(terms.values()) / group_size
                loss.backward()
            if sync:
                factor = lr_factor(update, total_updates, warmup, args.min_lr_ratio)
                for group in optimizer.param_groups:
                    group['lr'] = group['base_lr'] * factor
                nn.utils.clip_grad_norm_(core.parameters(), 1.)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                update += 1
                if rank == 0 and (update % 100 == 0 or update == 1):
                    print('STEP', stage['stage'], update, 'batch', batch_index + 1,
                          'loss', float(loss.detach() * group_size),
                          'lr_encoder', optimizer.param_groups[0]['lr'], flush=True)
                requested = torch.tensor(int(_STOP or bool(args.max_updates and
                                                      update >= args.max_updates)),
                                         dtype=torch.int32, device=device)
                if world > 1:
                    torch.distributed.all_reduce(requested, op=torch.distributed.ReduceOp.MAX)
                stop_now = bool(requested.item())
                if (update % args.checkpoint_every == 0 or stop_now) and rank == 0:
                    save(args.out, core, optimizer, stage_index=stage_index,
                         batch_index=batch_index + 1, update=update,
                         hashes=hashes, best=best, args=args)
                if stop_now:
                    if world > 1:
                        torch.distributed.barrier()
                    print('STOP', rank, stage['stage'], update, flush=True)
                    return
        if rank == 0:
            save(args.out, core, optimizer, stage_index=stage_index + 1,
                 batch_index=0, update=update, hashes=hashes, best=best, args=args)
        if world > 1:
            torch.distributed.barrier()
        report = validate(core, validation, device, world, indices)
        selection = (report['ladder']['weighted_loss_sum'] +
                     report['p3']['weighted_loss_sum'])
        best = min(best, selection)
        if rank == 0:
            row = {'stage': stage['stage'], 'update': update,
                   'p3_share': stage['p3_share'], 'validation': report,
                   'selection_loss': selection, 'best': best}
            with (args.out / 'metrics.jsonl').open('a') as stream:
                stream.write(json.dumps(row) + '\n')
            save(args.out, core, optimizer, stage_index=stage_index + 1,
                 batch_index=0, update=update, hashes=hashes, best=best, args=args)
            print('STAGE', json.dumps(row), flush=True)
        if world > 1:
            torch.distributed.barrier()
        start_batch = 0
    ladder.close()
    p3_train.close()
    val_p3.p3.close()
    if rank == 0:
        marker = args.out / 'COMPLETE.json.incomplete'
        marker.write_text(json.dumps({'status': 'complete', 'model': args.model,
                                      'updates': update, 'hashes': hashes}, indent=2) + '\n')
        marker.replace(args.out / 'COMPLETE.json')
    if world > 1:
        torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
