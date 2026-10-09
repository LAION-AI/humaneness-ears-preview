#!/usr/bin/env python3
"""Bounded real-cache training/evaluation contract in the production probe env."""
import hashlib
import os
import torch
from study_paths import ROOT, RELEASE, read, write
from cache_dataset import FeatureData, collate
from probe_model import Probe
from train_probes import evaluate, loss_terms
import numpy as np


def main():
    torch.set_num_threads(2);device=torch.device('cuda:0')
    classes=read(RELEASE/'classes.json')['names']
    datasets=[FeatureData('clapv2_xxs',['ladder'],split) for split in ('train','validation')]
    for data in datasets:
        if len(data)<8:raise RuntimeError('Need eight committed clips in each split')
        data.rows=data.rows[:8]
        width=data.payload(data.rows[0])['embedding'].shape[0]
        data.projection={'mean':np.zeros(width,np.float32),'components':np.eye(width,dtype=np.float32)[:256],
                         'scale':np.ones(min(width,256),np.float32)}
    first=datasets[0].rows[0]
    raw=os.pread(os.open(first['feature_tar'],os.O_RDONLY),first['feature_size'],first['feature_offset'])
    assert hashlib.sha256(raw).hexdigest()==first['feature_sha256'], 'TAR offset/SHA mismatch'
    assert not set(r['feature_key'] for r in datasets[0].rows)&set(r['feature_key'] for r in datasets[1].rows)
    results={}
    for variant in ('linear','mlp'):
        model=Probe(len(classes),variant=='linear').to(device)
        batch={k:v.to(device) if torch.is_tensor(v) else v for k,v in collate([datasets[0][i] for i in range(8)]).items()}
        before=next(model.scalars.parameters()).detach().clone()
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
        out=model(batch['embedding'],batch['frame_features'],batch['event_starts'],batch['event_ends'])
        loss=sum(loss_terms(out,batch,torch.ones(len(classes),device=device)).values())
        loss.backward()
        assert torch.isfinite(loss) and all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        optimizer.step()
        assert not torch.equal(before,next(model.scalars.parameters())), 'No scalar parameter update'
        metrics=evaluate(model,datasets[1],device,torch.ones(len(classes),device=device),True)
        assert len(metrics['per_score'])==192 and metrics['clips']==8
        results[variant]={'optimizer_updated':True,'finite_loss_and_gradients':True,
                          'evaluation_targets':len(metrics['per_score']),'heldout_contract_clips':metrics['clips']}
    write(ROOT/'TRAINING_PIPELINE_CONTRACT.json',{'passed':True,'source':'Committed real Ladder feature TARs',
          'scope':'Software contract only; eight train and eight separate validation clips; these are not benchmark results',
          'tar_offset_sha_verified':True,'variants':results})
    print('TRAINING_PIPELINE_CONTRACT_PASS',flush=True)


if __name__=='__main__':
    main()
