#!/usr/bin/env python3
"""One-GPU Tiny/Base P3 smoke test for clip, frame and Orange objectives.

The 100-example pilot is too small for an accuracy claim. This checks actual
pretrained encoder loading, masked sidecar joins, gradients and holdout metrics.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import tarfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from safetensors import safe_open
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import WhisperConfig, WhisperFeatureExtractor
from transformers.models.whisper.modeling_whisper import WhisperEncoder

from p3_target_adapter import SIDECARS, adapt, json_members, rasterize_burst_frames, target_names
from train import snapshot

ROOT = Path('/e/scratch/reformo/schuhmann1_moss/vocalburst_p3/pilot')
SIDECAR_PATHS = {name: ROOT / f'p3-pilot-{name}-sidecars-00000.tar' for name in SIDECARS}
GROUPS = {
    'emotion': (0, 40, 1.0), 'voicenet': (40, 97, 1.0),
    'other': (97, 100, .3), 'eiv_extra': (100, 119, .5),
    'audiobox': (119, 123, .4), 'dnsmos': (123, 130, .4),
    'burst_count': (130, 131, .3),
    'voiceclap_attributes': (131, 192, .15),
}


def tar_payloads(path: Path, suffix: str) -> dict[str, bytes]:
    result = {}
    with tarfile.open(path, 'r:') as tf:
        for member in tf:
            if member.isfile() and member.name.endswith(suffix):
                result[member.name[:-len(suffix)]] = tf.extractfile(member).read()
    return result


def make_rows(audio_tar: Path, sidecar_paths: dict[str, Path],
              locator_path: Path) -> list[dict]:
    audio_meta = json_members(audio_tar, '.json')
    audio = tar_payloads(audio_tar, '.opus')
    side = {name: json_members(path, suffix) for name, (path, suffix) in
            ((name, (sidecar_paths[name], ext)) for name, ext in SIDECARS.items())}
    timbre = tar_payloads(sidecar_paths['orange'], '.orange_timbre.npy')
    identity = tar_payloads(sidecar_paths['orange'], '.orange_identity.npy')
    locator = {r['sample_id']: r for r in json.loads(locator_path.read_text())['rows']}
    keys = set(audio_meta)
    if any(set(items) != keys for items in (audio, *side.values(), timbre, identity, locator)):
        raise RuntimeError('pilot TARs have different sample keys')
    names = target_names()
    example_scores = next(d['scores'] for d in side['empathic'].values() if d['scores'])
    mapped = {('Jealousy_&_Envy' if n == 'emo_Jealousy_and_Envy' else n[4:])
              for n in names[:40]}
    extra_names = sorted(set(example_scores) - mapped)
    if len(extra_names) != 19:
        raise RuntimeError(f'expected 19 EIV extras, found {len(extra_names)}')
    attr_names = sorted(next(d['voiceclap_attributes'] for d in
                             side['voiceclap'].values() if d['voiceclap_attributes']))
    if len(attr_names) != 61:
        raise RuntimeError(f'expected 61 VoiceCLAP attributes, found {len(attr_names)}')
    rows = []
    for key in sorted(keys):
        wav, sr = sf.read(io.BytesIO(audio[key]), dtype='float32')
        if sr != 16000 or wav.ndim != 1:
            raise RuntimeError(f'{key}: wrong audio format')
        target = adapt(audio_meta[key], {kind: items[key] for kind, items in side.items()},
                       names, extra_names, locator[key])
        score = ([target['ladder_scores'][name] for name in names] +
                 [target['extra_eiv_scores'][name] for name in extra_names] +
                 [target['audiobox_scores'][name] for name in ('CE', 'CU', 'PC', 'PQ')] +
                 [target['dnsmos_scores'][name] for name in
                  ('SIG_raw', 'BAK_raw', 'OVRL_raw', 'SIG', 'BAK', 'OVRL', 'P808_MOS')] +
                 [target['burst_count_log1p']] +
                 [target['voiceclap_attribute_scores'].get(name)
                  if target['voiceclap_attributes_valid_for_speech'] else None
                  for name in attr_names])
        if len(score) != 192:
            raise RuntimeError(f'{key}: wrong target count {len(score)}')
        frame, frame_mask = rasterize_burst_frames(target['burst_frame_target'])
        rows.append({'key': key, 'audio': wav,
                     'scores': np.array([np.nan if x is None else x for x in score], np.float32),
                     'frame': np.array(frame, np.float32),
                     'frame_mask': np.array(frame_mask, np.float32),
                     'timbre': np.load(io.BytesIO(timbre[key])).astype(np.float32),
                     'identity': np.load(io.BytesIO(identity[key])).astype(np.float32),
                     'speaker_valid': target['orange_timbre']['valid_single_speaker']})
    return rows


class PilotData(Dataset):
    def __init__(self, rows: list[dict], indices: list[int], mean: np.ndarray, std: np.ndarray):
        self.rows, self.indices, self.mean, self.std = rows, indices, mean, std

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        row = self.rows[self.indices[idx]]
        finite = np.isfinite(row['scores'])
        target = np.where(finite, (row['scores'] - self.mean) / self.std, 0).astype(np.float32)
        return row['audio'], target, finite, row['frame'], row['frame_mask'], \
               row['timbre'], row['identity'], row['speaker_valid']


def collator(model_path: Path):
    extractor = WhisperFeatureExtractor.from_pretrained(model_path, local_files_only=True)

    def collate(batch):
        waves = [x[0] for x in batch]
        feat = extractor(waves, sampling_rate=16000, padding='longest', truncation=True,
                         max_length=480000, return_attention_mask=True, return_tensors='np')
        mel = feat['input_features'].astype(np.float32)
        mask = feat['attention_mask'].astype(np.int64)
        if mel.shape[-1] % 2:
            mel = np.pad(mel, ((0, 0), (0, 0), (0, 1)))
            mask = np.pad(mask, ((0, 0), (0, 1)))
        length = mel.shape[-1] // 2
        frame = np.zeros((len(batch), length), np.float32)
        frame_mask = np.zeros_like(frame)
        for i, item in enumerate(batch):
            n = min(length, len(item[3]))
            frame[i, :n] = item[3][:n]
            frame_mask[i, :n] = item[4][:n]
        return tuple(torch.from_numpy(x) for x in
                     (mel, mask, np.stack([x[1] for x in batch]),
                      np.stack([x[2] for x in batch]), frame, frame_mask,
                      np.stack([x[5] for x in batch]),
                      np.stack([x[6] for x in batch]),
                      np.array([x[7] for x in batch], np.bool_)))
    return collate


class MultiTaskWhisper(nn.Module):
    def __init__(self, model_path: Path):
        super().__init__()
        config = WhisperConfig.from_pretrained(model_path, local_files_only=True)
        self.encoder = WhisperEncoder(config)
        with safe_open(model_path / 'model.safetensors', framework='pt', device='cpu') as file:
            prefix = 'model.encoder.'
            state = {key[len(prefix):]: file.get_tensor(key) for key in file.keys()
                     if key.startswith(prefix)}
        self.encoder.load_state_dict(state, strict=True)
        dim = config.d_model
        def clip_head(outputs):
            return nn.Sequential(nn.LayerNorm(dim * 4), nn.Linear(dim * 4, 512),
                                 nn.GELU(), nn.Linear(512, outputs))
        self.score_head = clip_head(192)
        self.timbre_head = clip_head(128)
        self.identity_head = clip_head(250)
        self.frame_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim // 2),
                                        nn.GELU(), nn.Conv1d(dim // 2, dim // 2, 7,
                                                               padding=3), nn.GELU(),
                                        nn.Conv1d(dim // 2, 1, 1))

    def forward(self, mel, mel_mask):
        enc = self.encoder
        h = F.gelu(enc.conv1(mel))
        h = F.gelu(enc.conv2(h)).transpose(1, 2)
        n = h.shape[1]
        h = h + enc.embed_positions(torch.arange(n, device=h.device))
        valid = torch.arange(n, device=h.device)[None, :] < \
                ((mel_mask.sum(1) + 1) // 2)[:, None]
        attention = torch.zeros((len(h), 1, 1, n), device=h.device, dtype=h.dtype)
        attention.masked_fill_(~valid[:, None, None, :], torch.finfo(h.dtype).min)
        for layer in enc.layers:
            h = layer(h, attention)
        h = enc.layer_norm(h).float()
        mask = valid[:, :, None]
        count = mask.sum(1).clamp_min(1)
        mean = (h * mask).sum(1) / count
        var = ((h - mean[:, None, :]).square() * mask).sum(1) / count
        minimum = h.masked_fill(~mask, torch.inf).amin(1)
        maximum = h.masked_fill(~mask, -torch.inf).amax(1)
        pooled = torch.cat((mean, minimum, maximum, var.clamp_min(1e-8).sqrt()), dim=-1)
        x = self.frame_head[0](h)
        x = self.frame_head[1](x)
        x = self.frame_head[2](x).transpose(1, 2)
        logits = self.frame_head[3:](x).squeeze(1)
        return {'scores': self.score_head(pooled), 'frame': logits,
                'timbre': F.normalize(self.timbre_head(pooled).float(), dim=-1),
                'identity': F.normalize(self.identity_head(pooled).float(), dim=-1)}


def loss_terms(output, batch):
    target, score_mask, frame, frame_mask, timbre, identity, speaker_mask = batch[2:]
    pred = output['scores'].float()
    terms = {}
    for name, (start, stop, weight) in GROUPS.items():
        mask = score_mask[:, start:stop].float()
        if mask.sum() > 0:
            huber = F.huber_loss(pred[:, start:stop], target[:, start:stop],
                                 delta=1.0, reduction='none')
            terms[name] = weight * (huber * mask).sum() / mask.sum()
    if frame_mask.sum() > 0:
        bce = F.binary_cross_entropy_with_logits(output['frame'].float(), frame,
                                                  reduction='none', pos_weight=torch.tensor(5.0,
                                                                                           device=frame.device))
        terms['frame_bce'] = .5 * (bce * frame_mask).sum() / frame_mask.sum()
        prob = output['frame'].float().sigmoid() * frame_mask
        intersection = (prob * frame).sum()
        dice = 1 - (2 * intersection + 1) / (prob.sum() + (frame * frame_mask).sum() + 1)
        terms['frame_dice'] = .25 * dice
    if speaker_mask.any():
        for name, vector in (('timbre', timbre), ('identity', identity)):
            pred_vec = output[name][speaker_mask]
            target_vec = F.normalize(vector[speaker_mask].float(), dim=-1)
            terms[name] = .2 * ((1 - (pred_vec * target_vec).sum(-1)).mean() +
                                .1 * F.huber_loss(pred_vec, target_vec, delta=.1))
    else:
        # Keep both heads in the graph on DDP ranks whose local batch has no
        # valid single-speaker target. The zero contribution is still masked.
        terms['timbre'] = output['timbre'].sum() * 0
        terms['identity'] = output['identity'].sum() * 0
    return terms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=['tiny', 'base'], required=True)
    ap.add_argument('--epochs', type=int, default=2)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--audio-tar', type=Path, default=ROOT / 'p3-pilot-00000.tar')
    ap.add_argument('--sidecar-dir', type=Path)
    ap.add_argument('--locator-audit', type=Path, default=ROOT / 'audit_locator.json')
    args = ap.parse_args()
    assert torch.cuda.is_available()
    torch.set_num_threads(2)
    torch.manual_seed(640913)
    sidecar_paths = ({name: args.sidecar_dir / f'{args.audio_tar.stem}-{name}.tar'
                      for name in SIDECARS} if args.sidecar_dir else SIDECAR_PATHS)
    rows = make_rows(args.audio_tar, sidecar_paths, args.locator_audit)
    train_idx = [i for i in range(len(rows)) if i % 10 < 8]
    val_idx = [i for i in range(len(rows)) if i % 10 >= 8]
    train_scores = np.stack([rows[i]['scores'] for i in train_idx])
    finite = np.isfinite(train_scores)
    counts = finite.sum(axis=0)
    safe_counts = np.maximum(counts, 1)
    clean = np.where(finite, train_scores, 0)
    mean = clean.sum(axis=0) / safe_counts
    variance = np.where(finite, (clean - mean) ** 2, 0).sum(axis=0) / safe_counts
    std = np.where(counts > 0, np.maximum(np.sqrt(variance), .25), 1)
    mean = mean.astype(np.float32)
    std = std.astype(np.float32)
    model_path = snapshot(args.model)
    collate = collator(model_path)
    train_loader = DataLoader(PilotData(rows, train_idx, mean, std), batch_size=8,
                              shuffle=True, collate_fn=collate, num_workers=0)
    val_loader = DataLoader(PilotData(rows, val_idx, mean, std), batch_size=8,
                            shuffle=False, collate_fn=collate, num_workers=0)
    model = MultiTaskWhisper(model_path).cuda()
    heads = list(model.score_head.parameters()) + list(model.timbre_head.parameters()) + \
            list(model.identity_head.parameters()) + list(model.frame_head.parameters())
    optimizer = torch.optim.AdamW([{'params': model.encoder.parameters(), 'lr': 2e-5},
                                   {'params': heads, 'lr': 3e-4}], weight_decay=.01)
    metrics = []
    for epoch in range(args.epochs):
        model.train()
        train_loss = []
        for batch in train_loader:
            batch = tuple(x.cuda() for x in batch)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                output = model(batch[0], batch[1])
                terms = loss_terms(output, batch)
                loss = sum(terms.values())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss.append(float(loss.detach()))
        model.eval()
        val_loss = []
        sums = {}
        with torch.no_grad():
            for batch in val_loader:
                batch = tuple(x.cuda() for x in batch)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    terms = loss_terms(model(batch[0], batch[1]), batch)
                val_loss.append(float(sum(terms.values())))
                for name, val in terms.items():
                    sums.setdefault(name, []).append(float(val))
        item = {'model': args.model, 'epoch': epoch + 1,
                'train_loss': float(np.mean(train_loss)),
                'val_loss': float(np.mean(val_loss)),
                'val_terms': {name: float(np.mean(v)) for name, v in sums.items()},
                'train_rows': len(train_idx), 'val_rows': len(val_idx),
                'disclaimer': '100-example architecture smoke; no generalization estimate'}
        metrics.append(item)
        print(json.dumps(item), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(metrics, indent=2) + '\n')


if __name__ == '__main__':
    main()
