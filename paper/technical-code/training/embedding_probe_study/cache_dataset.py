"""Read cached native features and attach the selected target regime."""
import io
import json
import math
import os
from collections import OrderedDict
from pathlib import Path
import numpy as np
import torch
from study_paths import ROOT, GEMINI, RELEASE, read

_COMMITTED_INDEX_ROWS = {}


class FeatureData:
    def __init__(self, model, domains=None, split=None, phase='legacy', stage=None):
        self.rows = []
        self.fds = OrderedDict()
        self.projection = None
        self.phase = phase
        self.flash = None
        self.norm = read(RELEASE / 'training_normalization.json')
        if phase == 'gemini':
            if not (GEMINI / 'prepared/TRAINING_READY.json').exists():
                raise RuntimeError('Final Gemini training gate is not ready')
            self.flash = {r['sha256_mp3']: r for r in map(json.loads, (GEMINI / 'prepared/targets.jsonl').open())
                          if r['ready_for_whisper_training'] and (split is None or r['split'] == split)}
        feature_root=ROOT/'features'/model
        domain_key=tuple(sorted(domains)) if domains else ()
        cache_key=(model,domain_key)
        committed=bool(domains) and all((feature_root/(d+'-rank'+str(r)+'-COMPLETE.json')).exists()
                                      for d in domains for r in range(4))
        # The same immutable index serves train/validation/test and S8/S9/S10.
        # Decode its metadata once per process; keep the original split filters.
        indexed=_COMMITTED_INDEX_ROWS.get(cache_key) if committed else None
        if indexed is None:
            indexed=[]
            for path in sorted(feature_root.glob('*.jsonl')):
                if domains and not any(path.name.startswith(d+'-rank') for d in domains):
                    continue
                with path.open() as stream:
                    indexed.extend(row for row in map(json.loads,stream)
                                   if not domains or row['source'] in domains)
            if committed:_COMMITTED_INDEX_ROWS[cache_key]=indexed
        for row in indexed:
            if phase == 'gemini':
                if row['source'] != 'gemini' or row['key'] not in self.flash:
                    continue
            elif split and row.get('split') != split:
                continue
            if stage and stage not in row.get('stages', []):
                continue
            self.rows.append(row)

    def __len__(self):
        return len(self.rows)

    def payload(self, row):
        path = row['feature_tar']
        if path not in self.fds:
            if len(self.fds) >= 24:
                _, old = self.fds.popitem(last=False)
                os.close(old)
            self.fds[path] = os.open(path, os.O_RDONLY)
        self.fds.move_to_end(path)
        raw = os.pread(self.fds[path], row['feature_size'], row['feature_offset'])
        if len(raw) != row['feature_size']:
            raise RuntimeError('Short feature cache read')
        with np.load(io.BytesIO(raw), allow_pickle=False) as data:
            return {k: data[k] for k in data.files}

    def __getstate__(self):
        state = {**self.__dict__, 'fds': OrderedDict()}
        state.pop('teacher_reader', None)
        return state

    def __getitem__(self, index):
        row = self.rows[index]
        value = self.payload(row)
        embedding = value['embedding'].astype(np.float32)
        if self.projection:
            p = self.projection
            embedding = ((embedding - p['mean']) @ p['components'].T) / p['scale']
            embedding = np.pad(embedding, (0, 256 - len(embedding))).astype(np.float32)
        frames = int(math.ceil(float(value['duration_s']) * 50))
        # The model sees interpolated native features on the shared 20-ms grid.
        # Native timestamps/count are retained for resolution-aware reporting.
        ticks = (np.arange(frames) + .5) / 50
        native = value['frame_features'].astype(np.float32)
        times = value['frame_times_s']
        value['frame_features'] = np.stack([np.interp(ticks, times, native[:, i]) for i in range(64)], axis=1).astype(np.float32)
        value['embedding'] = embedding
        value['feature_key'] = row['feature_key']
        if self.phase == 'gemini':
            # Targets are prepared without decoding the cached full audio.
            r = self.flash[row['key']]
            norm = self.norm
            raw = np.asarray([r['raw_scores'].get(n, np.nan) for n in norm['score_names']], np.float32)
            valid = np.isfinite(raw)
            teachers = r.get('teacher_metadata', {})
            if not teachers.get('dnsmos', {}).get('valid_for_speech', False):
                valid[123:130] = False
            if not teachers.get('empathic', {}).get('speech_domain_valid', False):
                valid[100:119] = False
            if not teachers.get('voiceclap', {}).get('speech_domain_valid', False):
                for i in range(131, 192):
                    if r['score_provenance'].get(norm['score_names'][i]) != 'gemini-3.8-flash':
                        valid[i] = False
            value.update(scores=np.where(valid, (raw - norm['score_mean']) / norm['score_std'], 0).astype(np.float32),
                         score_mask=valid, frame=np.zeros(frames, np.float32),
                         frame_mask=np.full(frames, r['burst_frames_valid'], np.float32))
            events = r['vocal_bursts']
            starts = np.asarray([max(0, min(frames-1, int(e['start'] * 50))) for e in events], np.int64)
            ends = np.asarray([min(frames, max(int(a)+1, math.ceil(e['end'] * 50))) for a,e in zip(starts,events)], np.int64)
            for a,b in zip(starts,ends):
                value['frame'][a:b] = 1
            cps = r['characters_per_second']
            value.update(event_starts=starts, event_ends=ends,
                         event_classes=np.asarray([e['checkpoint_class_id'] for e in events], np.int64),
                         cps=np.float32((cps-norm['cps_mean'])/norm['cps_std']) if cps is not None else np.float32(0),
                         cps_valid=np.bool_(cps is not None))
            import sys
            from study_paths import CODE
            sys.path.insert(0, str(CODE / 'gemini_finetune'))
            from common import TarReader
            if not hasattr(self, 'teacher_reader'):
                self.teacher_reader = TarReader()
            for name,dim in [('timbre',128),('identity',250)]:
                reference = r['embeddings'].get(name)
                value[name] = np.zeros(dim,np.float32)
                if reference:
                    value[name] = np.load(io.BytesIO(self.teacher_reader.read(reference['tar'],reference['member'])),allow_pickle=False).astype(np.float32)
                value[name+'_valid'] = np.bool_(reference and teachers.get('orange',{}).get('valid_single_speaker_target'))
        return value


def collate(items):
    width = max(len(x['frame_features']) for x in items)
    events = max(1, max(len(x.get('event_starts', [])) for x in items))
    arrays = {}
    for name in ('embedding','scores','score_mask','timbre','identity','timbre_valid','identity_valid','cps','cps_valid'):
        arrays[name] = torch.from_numpy(np.stack([x[name] for x in items]))
    for name in ('frame','frame_mask','frame_features'):
        source = [np.pad(x[name], ((0,width-len(x[name])),(0,0)) if x[name].ndim == 2 else (0,width-len(x[name]))) for x in items]
        arrays[name] = torch.from_numpy(np.stack(source))
    for name in ('event_starts','event_ends','event_classes'):
        arrays[name] = torch.from_numpy(np.stack([np.pad(x[name], (0,events-len(x[name]))) for x in items]))
    mask = np.stack([np.arange(events)<len(x['event_starts']) for x in items])
    arrays['event_valid'] = torch.from_numpy(mask)
    arrays['event_span_valid'] = arrays['event_valid'].clone()
    arrays['keys'] = [x['feature_key'] for x in items]
    return arrays
