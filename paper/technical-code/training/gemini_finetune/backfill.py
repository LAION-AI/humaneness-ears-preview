#!/usr/bin/env python3
"""Offline missing-teacher annotation, exact-audio hashes, resumable TAR sidecars.

Runs on compute GPUs. Waveforms are decoded once per batch and reused by the
requested teachers. Never creates per-clip audio/embedding files.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import tarfile
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from common import (CODE, HOME, ROOT, TarReader, add_member, jsonl, read_json, write_json)


class Teachers:
    def __init__(self, names):
        import torch
        self.torch = torch
        self.models = {}
        self.names = set(names)
        self.timing_s = {}
        sys.path.insert(0, str(HOME / 'vocal_burst_pool/p3'))
        sys.path.insert(0, str(CODE))
        if 'orange' in names:
            from timbre_targets import OrangeTimbreTeacher, OrangeIdentityTeacher
            self.models['timbre'] = OrangeTimbreTeacher().to('cuda').eval()
            self.models['identity'] = OrangeIdentityTeacher().to('cuda').eval()
        if 'dnsmos' in names:
            import annotate_dnsmos as dns
            self.dns = dns
            self.dns_workers = int(os.environ.get('TEACHER_DNSMOS_WORKERS', '1'))
            self.dns_threads = int(os.environ.get('TEACHER_DNSMOS_THREADS', '2'))
            if self.dns_workers < 1 or self.dns_threads < 1:
                raise ValueError('DNSMOS workers and threads must be positive')
            self._dns_local = threading.local()
            self.dns_pool = None
            if self.dns_workers == 1:
                self.models['dnsmos'] = self._new_dns_scorer()
            else:
                self.dns_pool = ThreadPoolExecutor(max_workers=self.dns_workers,
                                                   initializer=self._init_dns_scorer)
        if 'audiobox' in names:
            import annotate_audiobox as box
            self.box = box
            self.models['audiobox'] = box.AesMultiOutput.from_pretrained(str(box.MODEL)).to('cuda').eval()
        if 'voiceclap' in names:
            import annotate_voiceclap as voice
            self.voice = voice
            predictor = voice.VoiceNetPredictor(root=str(voice.VN), device='cuda')
            self.models['voiceclap'] = predictor
            self.gen = voice.weights(voice.GEN / 'genuineness_head.pt', 'genuineness')
            self.blend = voice.weights(voice.BLEND / 'blend_head_commercial.pt', 'blend')
            bundle = torch.load(voice.ATTR / 'heads.pt', map_location='cpu', weights_only=False)
            self.attr_mu, self.attr_sd = bundle['emb_mu'].to('cuda').float(), bundle['emb_sd'].to('cuda').float()
            self.attr_heads = {n: voice.build_attribute_head(state).to('cuda').eval() for n, state in bundle['heads'].items()}
        if 'empathic' in names:
            import annotate_empathic as eiv
            self.eiv = eiv
            self.models['eiv_feature'] = eiv.WhisperFeatureExtractor.from_pretrained(eiv.BUD, local_files_only=True)
            full = eiv.WhisperModel.from_pretrained(eiv.BUD, local_files_only=True, low_cpu_mem_usage=True)
            self.models['eiv_encoder'] = full.get_encoder().to('cuda').eval()
            del full
            self.models['eiv_experts'] = eiv.models()

    def _new_dns_scorer(self):
        return self.dns.ComputeScore(str(self.dns.CODE / 'DNSMOS/sig_bak_ovr.onnx'),
                                    str(self.dns.CODE / 'DNSMOS/model_v8.onnx'), threads=self.dns_threads)

    def _init_dns_scorer(self):
        self._dns_local.model = self._new_dns_scorer()

    def _score_dns(self, wav):
        scorer = self._dns_local.model if self.dns_pool is not None else self.models['dnsmos']
        return self.dns.score_wav(scorer, wav)

    def close(self):
        if getattr(self, 'dns_pool', None) is not None:
            self.dns_pool.shutdown(wait=True)

    def annotate(self, rows, wavs):
        import numpy as np
        torch = self.torch
        result = [{'sha256_mp3': row['sha256_mp3'], 'scores': {}, 'embeddings': {}, 'teachers': {},
                   'schema': 'gemini-teacher-sidecar-v1', 'audio_tar': row['audio_tar'],
                   'audio_member': row['audio_member'], 'annotated_utc': datetime.now(timezone.utc).isoformat(),
                   'input': 'exact_selected_audio_sha256; mono16k; no teacher-specific source substitution'} for row in rows]
        arrays = [{} for _ in rows]
        with torch.inference_mode():
            for kind in sorted(self.names):
                positions = [i for i, row in enumerate(rows) if kind in row['missing_teachers']]
                if not positions:
                    continue
                teacher_start = time.monotonic()
                part = [wavs[i] for i in positions]
                docs = [result[i] for i in positions]
                if kind == 'orange':
                    for name, dim in [('timbre', 128), ('identity', 250)]:
                        emb = self.models[name].encode_waveforms(part, torch.device('cuda'), batch_size=2)
                        if emb.shape != (len(part), dim):
                            raise ValueError(f'Wrong Orange {name} dimensions: {emb.shape}')
                        for i, value in zip(positions, emb):
                            arrays[i][name] = np.asarray(value, dtype=np.float16)
                    for i in positions:
                        result[i]['teachers']['orange'] = {
                            'timbre': 'Orange/Speaker-wavLM-tbr@b8d2608d56f18e2b6e27ba566cf69132e2c2ad6c',
                            'identity': 'Orange/Speaker-wavLM-id@abbb3c7b8d220ceebc33d9bd3bb5aedb580342c7',
                            'valid_single_speaker_target': rows[i].get('speaker_target_valid', False),
                            'speaker_validity_source': rows[i].get('speaker_validity_source', 'unknown'),
                        }
                elif kind == 'dnsmos':
                    values = (list(self.dns_pool.map(self._score_dns, part))
                              if self.dns_pool is not None else [self._score_dns(wav) for wav in part])
                    for i, (score, windows) in zip(positions, values):
                        result[i]['scores'].update({'dnsmos:' + n: float(v) for n, v in score.items()})
                        result[i]['teachers']['dnsmos'] = {'model': 'Microsoft DNSMOS non-personalized + P808',
                            'code': str(self.dns.CODE), 'valid_for_speech': rows[i].get('speech_target_valid', False),
                            'window_count': windows,
                            'short_audio_policy': 'repeat to 9.01 s; one-second window hop'}
                elif kind == 'audiobox':
                    values = self.box.infer(self.models['audiobox'], [(str(i), w) for i, w in zip(positions, part)])
                    for doc, value in zip(docs, values):
                        doc['scores'].update({'audiobox:' + n: float(v) for n, v in value.items()})
                        doc['teachers']['audiobox'] = {'model': 'facebook/audiobox-aesthetics', 'revision': self.box.MODEL_REVISION}
                elif kind == 'voiceclap':
                    predictor = self.models[kind]
                    tensor = torch.zeros(len(part), 480000, dtype=torch.float32, device='cuda')
                    for i, wav in enumerate(part):
                        tensor[i, :len(wav)] = torch.from_numpy(wav).to('cuda')
                    emb = torch.nn.functional.normalize(predictor.embedder.encode_waveform(tensor).float(), dim=-1)
                    if emb.shape != (len(part), 768):
                        raise ValueError('VoiceCLAP embedding dimension differs')
                    for dim in predictor.dims:
                        reg = predictor.reg[dim]
                        value = reg['net']((emb - reg['mu']) / reg['sd']).reshape(-1).cpu().numpy()
                        for doc, score in zip(docs, value):
                            doc['scores']['vn_' + dim.upper() + '_reg'] = float(score)
                    for key, (net, mean, std), upper in [('genuineness_0_6', self.gen, 6), ('blend_0_10', self.blend, 10)]:
                        values = net((emb - mean) / std).reshape(-1).clamp(0, upper).cpu().numpy()
                        for doc, value in zip(docs, values):
                            doc['scores'][key] = float(value)
                    attr_input = (emb - self.attr_mu) / self.attr_sd
                    for name, net in self.attr_heads.items():
                        values = net(attr_input).reshape(-1).cpu().numpy()
                        for doc, value in zip(docs, values):
                            doc['scores']['voiceclap_attribute:' + name] = float(value)
                    for i, value in zip(positions, emb.cpu().numpy()):
                        arrays[i]['voiceclap'] = value.astype(np.float16)
                        result[i]['teachers']['voiceclap'] = {**self.voice.REVISIONS, 'speech_domain_valid': rows[i].get('speech_target_valid', False)}
                elif kind == 'empathic':
                    features = self.models['eiv_feature'](part, sampling_rate=16000, return_tensors='pt').input_features.to('cuda')
                    with torch.autocast('cuda', torch.bfloat16):
                        hidden = self.models['eiv_encoder'](features).last_hidden_state.float()
                    pooled = torch.cat([hidden.mean(1), hidden.min(1).values, hidden.max(1).values, hidden.std(1)], 1)
                    for name, (net, quality) in self.models['eiv_experts'].items():
                        values = net(pooled if quality else hidden).reshape(-1).cpu().numpy()
                        for doc, value in zip(docs, values):
                            doc['scores']['emo_' + name] = float(value)
                            doc['scores']['eiv_extra:' + name] = float(value)
                    for i, value in zip(positions, pooled.cpu().numpy()):
                        arrays[i]['bud_e_pooled'] = value.astype(np.float16)
                        result[i]['teachers']['empathic'] = {'model': 'laion/Empathic-Insight-Voice-Plus',
                            'revision': self.eiv.EIV_SHA, 'encoder_revision': self.eiv.BUD_SHA,
                            'speech_domain_valid': rows[i].get('speech_target_valid', False)}
                self.timing_s[kind] = self.timing_s.get(kind, 0.0) + time.monotonic() - teacher_start
        for doc, stored in zip(result, arrays):
            for name, value in stored.items():
                if not np.isfinite(value).all():
                    raise ValueError(f'Non-finite teacher embedding {name}')
                doc['embeddings'][name] = {'member': doc['sha256_mp3'] + '.' + name + '.npy', 'dim': len(value), 'dtype': 'float16'}
            if any(not np.isfinite(v) for v in doc['scores'].values()):
                raise ValueError('Non-finite teacher score')
        return result, arrays


def completed(root):
    result = {}
    for path in root.glob('*.tar'):
        with tarfile.open(path, 'r:') as archive:
            for member in archive:
                if member.name.endswith('.teachers.json'):
                    doc = json.load(archive.extractfile(member))
                    result.setdefault(doc['sha256_mp3'], set()).update(doc['teachers'])
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--queue', type=Path, default=ROOT / 'prepared/teacher_backfill_queue.jsonl')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--shard-size', type=int, default=128)
    parser.add_argument('--rank', type=int, default=int(os.environ.get('SLURM_PROCID', 0)))
    parser.add_argument('--world', type=int, default=int(os.environ.get('SLURM_NTASKS', 1)))
    args = parser.parse_args()
    if args.world < 1 or not 0 <= args.rank < args.world:
        parser.error('rank must be in [0, world)')
    if args.batch_size < 1 or args.shard_size < 1:
        parser.error('batch and shard sizes must be positive')
    import numpy as np
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Teacher backfill must run on an allocated GPU')
    # Slurm assigns one visible GPU per task.
    torch.set_num_threads(2)
    output = ROOT / 'teacher_sidecars'
    output.mkdir(exist_ok=True)
    old = completed(output)
    queue = list(jsonl(args.queue))
    if args.limit:
        queue = queue[:args.limit]
    queue.sort(key=lambda row: (row['audio_tar'], row['audio_member']))
    queue = [row for i, row in enumerate(queue) if i % args.world == args.rank]
    for row in queue:
        row['missing_teachers'] = [n for n in row['missing_teachers'] if n not in old.get(row['sha256_mp3'], set())]
    queue = [row for row in queue if row['missing_teachers']]
    job = os.environ.get('SLURM_JOB_ID', str(os.getpid()))
    progress_path = ROOT / f'backfill-{job}-r{args.rank:02d}-status.json'
    progress = {'job_id': job, 'rank': args.rank, 'world': args.world,
                'queue': str(args.queue), 'queue_clips_for_rank': len(queue),
                'already_cached_clips_global': len(old),
                'dnsmos_workers': int(os.environ.get('TEACHER_DNSMOS_WORKERS', '1')),
                'dnsmos_threads_per_worker': int(os.environ.get('TEACHER_DNSMOS_THREADS', '2')),
                'completed': 0, 'remaining': len(queue)}
    def update(state, **extra):
        progress.update(state=state, updated_utc=datetime.now(timezone.utc).isoformat(), **extra)
        write_json(progress_path, progress)
    if not queue:
        update('COMPLETE')
        write_json(ROOT / f'backfill-{job}-r{args.rank:02d}-complete.json', {'clips': 0, 'teachers': [], 'limit': args.limit})
        print('No missing targets for this rank', flush=True)
        return
    names = set(n for row in queue for n in row['missing_teachers'])
    update('LOADING_TEACHERS', teachers=sorted(names))
    teachers = Teachers(names)
    reader = TarReader()
    started = time.monotonic()
    update('ANNOTATING')
    for start in range(0, len(queue), args.shard_size):
        shard = queue[start:start + args.shard_size]
        path = output / f'teachers-{job}-r{args.rank:02d}-b{start:07d}.tar'
        temporary = path.with_suffix('.tar.incomplete')
        with tarfile.open(temporary, 'w') as archive:
            for batch_start in range(0, len(shard), args.batch_size):
                part = shard[batch_start:batch_start + args.batch_size]
                wavs = [reader.audio(row, verify=True) for row in part]
                docs, arrays = teachers.annotate(part, wavs)
                for doc, array in zip(docs, arrays):
                    for name, value in array.items():
                        buffer = io.BytesIO()
                        np.save(buffer, value, allow_pickle=False)
                        add_member(archive, doc['embeddings'][name]['member'], buffer.getvalue())
                    add_member(archive, doc['sha256_mp3'] + '.teachers.json', json.dumps(doc, ensure_ascii=False, allow_nan=False).encode())
        os.replace(temporary, path)
        count = start + len(shard)
        elapsed = time.monotonic() - started
        update('ANNOTATING', completed=count, remaining=len(queue) - count,
               inference_elapsed_s=round(elapsed, 3),
               clips_per_second=round(count / max(elapsed, .001), 4),
               teacher_elapsed_s={n: round(v, 3) for n, v in teachers.timing_s.items()},
               last_committed_sidecar=str(path))
        print(json.dumps({'rank': args.rank, 'completed': start + len(shard), 'remaining': len(queue) - start - len(shard), 'sidecar': str(path)}), flush=True)
    reader.close()
    teachers.close()
    write_json(ROOT / f'backfill-{job}-r{args.rank:02d}-complete.json', {'clips': len(queue), 'teachers': sorted(names), 'limit': args.limit})
    update('COMPLETE')


if __name__ == '__main__':
    main()
