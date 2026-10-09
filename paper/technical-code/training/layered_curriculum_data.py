#!/usr/bin/env python3
"""CPS and event-class adapters for the existing S1→S10 Ladder/P3 readers."""
from __future__ import annotations

import bisect
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from full_ladder_dataset import LadderAudioDataset, MixedCollator
from train_full_curriculum import LadderVal


LADDER_ROOT=Path('/e/scratch/reformo/schuhmann1_moss/whisper_score_regression/full_ladder')
P3_ROOT=Path('/e/scratch/reformo/schuhmann1_moss/vocalburst_p3/frozen_1120k')
MAX_EVENTS=32
ALIASES={
    'Coughing':'Cough',
    'Hiccups':'Hiccup',
    'Clears Throat':'Throat Clearing',
    'Tongue Clicking':'Tongue Click',
    'Clicks Tongue':'Tongue Click',
    'Lip Smacking':'Lip Smack',
    'Smacks Lips':'Lip Smack',
    'Kissing Noises':'Kissing Sound',
    'Kissing Sounds':'Kissing Sound',
    'Gulps':'Gulp',
    'Sobs':'Sob',
    'Panting':'Pant',
    'Soft Hum':'Hum',
    'Resonant Hum':'Hum',
    'Humming':'Hum',
    'Deep Breath':'Breath',
    'Deep Breathing':'Breath',
    'Fast Breathing':'Breath',
    'Heavy Breathing':'Breath',
    'Slow Breathing':'Breath',
    'Breath':'Breath',
    'Normal Breathing':'Breath',
}


def taxonomy(min_events: int = 200) -> dict:
    ladder=json.loads((LADDER_ROOT/'ladder_aux_schema.json').read_text())
    p3=json.loads((P3_ROOT/'event_class_schema.json').read_text())
    counts=Counter()
    for name,count in ladder['class_counts'].items():
        counts[ALIASES.get(name,name)]+=int(count)
    p3_ids=np.load(P3_ROOT/'event_class_ids.npy',mmap_mode='r')
    train_positions=np.load(P3_ROOT/'train_indices.npy',mmap_mode='r')
    p3_train_counts=np.bincount(p3_ids[train_positions].ravel(),
                                minlength=len(p3['classes'])+1)
    for raw_id,name in enumerate(p3['classes'],1):
        counts[ALIASES.get(name,name)]+=int(p3_train_counts[raw_id])
    frequent=sorted(name for name,count in counts.items() if count>=min_events)
    names=['Other / rare burst']+frequent
    to_id={name:i for i,name in enumerate(names)}
    def lookup(source):
        result=np.zeros(len(source['classes'])+1,np.int64)
        for raw_id,name in enumerate(source['classes'],1):
            canonical=ALIASES.get(name,name)
            result[raw_id]=to_id.get(canonical,0)
        return result.tolist()
    canonical_counts=Counter({name:0 for name in names})
    for name,count in counts.items():
        canonical_counts[name if name in to_id else names[0]]+=count
    weight=np.asarray([1/math.sqrt(max(1,canonical_counts[name])) for name in names],np.float32)
    weight/=weight.mean()
    weight=np.clip(weight,.25,4.)
    return {'schema_version':'burst-canonical-v1','min_training_events':min_events,
            'names':names,'class_counts':dict(canonical_counts),
            'class_weights':weight.tolist(),
            'ladder_raw_to_canonical':lookup(ladder),
            'p3_raw_to_canonical':lookup(p3),
            'aliases':ALIASES,'raw_p3_classes':p3['classes'],
            'raw_ladder_classes':ladder['classes']}


class LadderAuxDataset(LadderAudioDataset):
    def __init__(self, mean: np.ndarray, std: np.ndarray):
        super().__init__(mean,std)
        self.aux_path=LADDER_ROOT/'ladder_aux.npy'
        self.aux=np.load(self.aux_path,mmap_mode='r')
        if len(self.aux)!=len(self):
            raise RuntimeError('Ladder auxiliary index has wrong row count')

    def __getstate__(self):
        state=super().__getstate__()
        state['aux']=None
        return state

    def __setstate__(self,state):
        self.__dict__.update(state)
        self.aux=np.load(self.aux_path,mmap_mode='r')

    def __getitem__(self,index: int) -> dict:
        result=super().__getitem__(index)
        shard=bisect.bisect_right(self.ends,int(index))
        start=0 if shard==0 else int(self.ends[shard-1])
        row=self._row(shard,int(index)-start)
        count=int(row['event_count'])
        result['events_frames']=[(max(0,int(float(row['starts'][j])*50)),
                                  min(1500,math.ceil(float(row['ends'][j])*50)))
                                 for j in range(count)]
        result['event_raw_class_ids']=np.asarray(self.aux['class_id'][index,:count],np.uint8)
        result['cps_raw']=float(self.aux['cps'][index])
        return result


class LadderAuxVal(LadderVal):
    def __init__(self,mean: np.ndarray,std: np.ndarray):
        super().__init__(mean,std)
        self.cps=np.load(LADDER_ROOT/'val_cps.npy',mmap_mode='r')
        if len(self.cps)!=len(self):
            raise RuntimeError('Ladder validation CPS length mismatch')

    def __getitem__(self,index: int) -> dict:
        result=super().__getitem__(index)
        result['cps_raw']=float(self.cps[index])
        result['events_frames']=[]
        result['event_raw_class_ids']=np.zeros(0,np.uint8)
        return result


class LayeredCollator(MixedCollator):
    def __init__(self,model_path: Path,*,cps_mean: float,cps_std: float,
                 ladder_lookup: list[int],p3_lookup: list[int],
                 cps_min: float = .5,cps_max: float = 60.):
        super().__init__(model_path)
        self.cps_mean=cps_mean
        self.cps_std=cps_std
        self.cps_min=cps_min
        self.cps_max=cps_max
        self.ladder_lookup=np.asarray(ladder_lookup,np.int64)
        self.p3_lookup=np.asarray(p3_lookup,np.int64)

    def __getstate__(self):
        state=self.__dict__.copy()
        state['extractor']=None
        return state

    def __call__(self,items: list[dict]) -> dict:
        result=super().__call__(items)
        b=len(items)
        n_frames=result['mel'].shape[-1]//2
        max_events=max(1,max(len(item.get('events_frames',item.get('events_samples',[])))
                             for item in items))
        if max_events>MAX_EVENTS:
            raise RuntimeError('too many events in one clip')
        starts=np.zeros((b,max_events),np.int64)
        ends=np.zeros((b,max_events),np.int64)
        classes=np.zeros((b,max_events),np.int64)
        event_valid=np.zeros((b,max_events),np.bool_)
        event_span_valid=np.zeros((b,max_events),np.bool_)
        cps=np.zeros(b,np.float32)
        cps_raw=np.full(b,np.nan,np.float32)
        cps_valid=np.zeros(b,np.bool_)
        for i,item in enumerate(items):
            enabled=item.get('enabled',True)
            raw_cps=item.get('cps_raw')
            if raw_cps is None:
                raw_cps=item.get('auxiliary_targets',{}).get('characters_per_second')
            if (enabled and raw_cps is not None and np.isfinite(raw_cps)
                    and self.cps_min<=float(raw_cps)<=self.cps_max):
                cps_raw[i]=float(raw_cps)
                cps[i]=(float(raw_cps)-self.cps_mean)/self.cps_std
                cps_valid[i]=True
            spans=(item['events_frames'] if 'events_frames' in item else
                   [(int(start)//320,(int(end)+319)//320)
                    for start,end in item.get('events_samples',[])])
            raw_ids=item.get('event_raw_class_ids')
            if raw_ids is None:
                raw_ids=np.zeros(len(spans),np.uint8)
            if len(raw_ids)<len(spans):
                raise RuntimeError('event class IDs shorter than intervals')
            lookup=self.ladder_lookup if item['domain']=='ladder' else self.p3_lookup
            for j,(start,end) in enumerate(spans):
                start=max(0,min(int(start),n_frames))
                end=max(0,min(int(end),n_frames))
                raw=int(raw_ids[j])
                if raw>=len(lookup):
                    raise RuntimeError('event class ID exceeds vocabulary')
                starts[i,j]=start
                ends[i,j]=end
                classes[i,j]=int(lookup[raw])
                event_span_valid[i,j]=bool(enabled and end>start)
                event_valid[i,j]=bool(event_span_valid[i,j] and raw)
        result.update(cps=torch.from_numpy(cps),cps_raw=torch.from_numpy(cps_raw),
                      cps_valid=torch.from_numpy(cps_valid),
                      event_starts=torch.from_numpy(starts),
                      event_ends=torch.from_numpy(ends),
                      event_classes=torch.from_numpy(classes),
                      event_valid=torch.from_numpy(event_valid),
                      event_span_valid=torch.from_numpy(event_span_valid))
        return result
