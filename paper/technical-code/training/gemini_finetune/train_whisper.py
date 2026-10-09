#!/usr/bin/env python3
"""Fine-tune the best Base/Small checkpoint on completed Flash-priority targets."""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from common import CODE, ROOT, jsonl, read_json, write_json
from data import GeminiCollator, GeminiDataset
sys.path.insert(0, str(CODE))
from layered_multitask_model import LayeredMultiTaskWhisper
from train_layered_curriculum import gpu_batch, loss_terms


@torch.no_grad()
def evaluate(core, loader, device, weights):
    core.eval()
    total = torch.zeros(2, device=device)
    for item in loader:
        batch = gpu_batch(item, device)
        with torch.autocast('cuda', torch.bfloat16):
            output = core(batch['mel'], batch['mel_mask'], batch['event_starts'], batch['event_ends'])
            loss = sum(loss_terms(output, batch, weights).values())
        size = len(batch['mel'])
        total += torch.tensor([float(loss) * size, size], device=device)
    return float(total[0] / total[1].clamp_min(1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    cfg = read_json(args.config)
    seed = cfg['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    world = int(os.environ.get('WORLD_SIZE', 1))
    rank = int(os.environ.get('RANK', 0))
    local = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local)
    if world > 1:
        torch.distributed.init_process_group('nccl')
    device = torch.device('cuda', local)
    root = Path(cfg['data_root'])
    train = GeminiDataset(root, 'train')
    validation = GeminiDataset(root, 'validation')
    if not train or not validation:
        raise RuntimeError('Nonempty train and validation partitions are required')
    taxonomy = read_json(root / 'gemini_burst_mapping.json')
    n_classes = len(taxonomy['checkpoint_classes'])
    # Fine-tuning optimizer/schedule start fresh; curriculum model weights are
    # restored strictly, without resetting heads or changing output semantics.
    checkpoint = torch.load(cfg['checkpoint'], map_location='cpu', weights_only=False)
    core = LayeredMultiTaskWhisper(cfg['model_config'], n_classes, initialize_pretrained=False)
    core.load_state_dict(checkpoint['model'], strict=True)
    del checkpoint
    core.to(device)
    # HF initializes the sinusoidal position table with requires_grad=False.
    # The requested full fine-tune explicitly unfreezes the entire encoder.
    core.requires_grad_(True)
    if not all(p.requires_grad for p in core.encoder.parameters()):
        raise RuntimeError('Full fine-tuning requires every encoder parameter to be trainable')
    params = [
        {'params': list(core.encoder.parameters()), 'lr': cfg['encoder_lr'], 'peak_lr': cfg['encoder_lr']},
        {'params': [p for name, p in core.named_parameters() if not name.startswith('encoder.')], 'lr': cfg['head_lr'], 'peak_lr': cfg['head_lr']},
    ]
    optimizer = torch.optim.AdamW(params, weight_decay=cfg['weight_decay'])
    model = DistributedDataParallel(core, device_ids=[local], find_unused_parameters=False) if world > 1 else core
    sampler = DistributedSampler(train, num_replicas=world, rank=rank, shuffle=True, seed=seed) if world > 1 else None
    collator = GeminiCollator(cfg['model_config'], train.norm)
    options = {'num_workers': cfg['workers'], 'pin_memory': True, 'collate_fn': collator}
    if cfg['workers']:
        options.update(persistent_workers=True, multiprocessing_context='spawn')
    loader = DataLoader(train, batch_size=cfg['batch_per_gpu'], sampler=sampler, shuffle=sampler is None, **options)
    # Rank zero evaluates the complete validation partition. Other ranks wait;
    # this avoids DistributedSampler padding/duplicate validation rows.
    val_loader = DataLoader(validation, batch_size=cfg['batch_per_gpu'], shuffle=False, **options) if rank == 0 else None
    event_counts = np.zeros(n_classes, np.float64)
    for row in train.rows:
        for event in row['vocal_bursts']:
            event_counts[event['checkpoint_class_id']] += 1
    active = event_counts > 0
    weight = np.ones(n_classes, np.float32)
    if active.any():
        weight[active] = np.sqrt(np.median(event_counts[active]) / event_counts[active]).clip(.25, 4)
    weights = torch.as_tensor(weight, device=device)
    updates = math.ceil(len(loader) / cfg['accumulation']) * cfg['epochs']
    warmup = max(1, round(updates * cfg['warmup_ratio']))
    def factor(update):
        if update < warmup:
            return max(1e-4, (update + 1) / warmup)
        progress = min(1., (update - warmup) / max(1, updates - warmup))
        return cfg['minimum_lr_ratio'] + (1 - cfg['minimum_lr_ratio']) * .5 * (1 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
    start_epoch, update, best = 0, 0, float('inf')
    if args.resume:
        state = torch.load(args.resume, map_location='cpu', weights_only=False)
        core.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        start_epoch, update, best = state['epoch_completed'], state['update'], state['best_validation_loss']
        if state['world_size'] != world:
            raise RuntimeError('Resume with the saved DDP world size')
        rng = state['rank_rng'][rank]
        torch.set_rng_state(rng['torch'])
        torch.cuda.set_rng_state(rng['cuda'])
        np.random.set_state(rng['numpy'])
        random.setstate(rng['python'])
    out = Path(cfg['output'])
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
        write_json(out / 'config.json', {**cfg, 'world': world, 'updates': updates,
            'train_clips': len(train), 'validation_clips': len(validation), 'event_class_counts': event_counts.tolist(),
            'fine_tune_mode': 'Full encoder and every multitask head; no LoRA or frozen encoder',
            'trainable_encoder_parameters': sum(p.numel() for p in core.encoder.parameters() if p.requires_grad),
            'trainable_total_parameters': sum(p.numel() for p in core.parameters() if p.requires_grad)})
    for epoch in range(start_epoch, cfg['epochs']):
        if sampler:
            sampler.set_epoch(epoch)
        core.train()
        optimizer.zero_grad(set_to_none=True)
        for step, item in enumerate(loader):
            batch = gpu_batch(item, device)
            accumulation_group = min(cfg['accumulation'], len(loader) - (step // cfg['accumulation']) * cfg['accumulation'])
            with torch.autocast('cuda', torch.bfloat16):
                output = model(batch['mel'], batch['mel_mask'], batch['event_starts'], batch['event_ends'])
                loss = sum(loss_terms(output, batch, weights).values()) / accumulation_group
            loss.backward()
            if (step + 1) % cfg['accumulation'] == 0 or step + 1 == len(loader):
                torch.nn.utils.clip_grad_norm_(core.parameters(), 1.)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                update += 1
        if world > 1:
            torch.distributed.barrier()
        val = evaluate(core, val_loader, device, weights) if rank == 0 else 0.
        val_tensor = torch.tensor(val, device=device)
        if world > 1:
            torch.distributed.broadcast(val_tensor, 0)
        val = float(val_tensor)
        improved = val < best
        best = min(val, best)
        rng = {'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state(), 'numpy': np.random.get_state(), 'python': random.getstate()}
        all_rng = [None] * world
        if world > 1:
            torch.distributed.all_gather_object(all_rng, rng)
        else:
            all_rng[0] = rng
        if rank == 0:
            state = {'model': core.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                'epoch_completed': epoch + 1, 'update': update, 'best_validation_loss': best,
                'world_size': world, 'rank_rng': all_rng, 'config': cfg,
                'normalization': train.norm, 'classes': taxonomy}
            temporary = out / 'latest.pt.incomplete'
            torch.save(state, temporary)
            os.replace(temporary, out / 'latest.pt')
            if improved:
                import shutil
                shutil.copy2(out / 'latest.pt', out / 'best.pt')
            with (out / 'metrics.jsonl').open('a') as stream:
                stream.write(json.dumps({'epoch': epoch + 1, 'update': update, 'validation_loss': val, 'best': improved}) + '\n')
            print('epoch', epoch + 1, 'validation_loss', val, flush=True)
    if rank == 0:
        write_json(out / 'COMPLETE.json', {'epochs': cfg['epochs'], 'best_validation_loss': best, 'updates': update})
    if world > 1:
        torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
