#!/usr/bin/env python3
"""Apply selected probes and the established, explicitly labeled benchmark metrics."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch
from study_paths import ROOT, CODE, RELEASE, read, write
from cache_dataset import FeatureData
from probe_model import Probe


def score_metrics(predictions, model_id, out):
    norm=read(RELEASE/'training_normalization.json')
    sys.path.insert(0,str(CODE));import score_public_benchmarks as metric
    metric.data=lambda kind,_model:predictions[kind]
    result={kind:(metric.emonet(model_id,norm['score_names']) if kind=='emonet' else
                  metric.emolia(kind,model_id,norm['score_names']) if kind.startswith('emolia') else
                  metric.acted(kind,model_id,norm['score_names'],np.asarray(norm['score_mean']),np.asarray(norm['score_std'])))
            for kind in predictions}
    write(out/'public_metrics.json',result)


def run(model_id,variant,phase):
    out=ROOT/'probes'/phase/model_id/variant
    checkpoint=torch.load(out/'best.pt',map_location='cpu',weights_only=False)
    classes=read(RELEASE/'classes.json')['names'];norm=read(RELEASE/'training_normalization.json')
    device=torch.device('cuda',int(__import__('os').environ.get('LOCAL_RANK',0)))
    core=Probe(len(classes),variant=='linear').to(device).eval();core.load_state_dict(checkpoint['model'])
    predictions={}
    with torch.inference_mode():
        for kind in read(ROOT/'study.json')['benchmarks']:
            data=FeatureData(model_id,[kind]);data.projection=checkpoint['pca']
            scores=[];rows=[]
            for index in range(0,len(data),32):
                values=[data[j] for j in range(index,min(index+32,len(data)))]
                # Scalar benchmark inference uses only the cached pooled feature.
                x=torch.from_numpy(np.stack([v['embedding'] for v in values])).to(device)
                score=core.scalars(x)[:,:192].cpu().numpy()*np.asarray(norm['score_std'])+np.asarray(norm['score_mean'])
                scores.append(score);rows.extend(data.rows[index:index+len(values)])
            scores=np.concatenate(scores).astype(np.float32)
            np.savez_compressed(out/(kind+'_predictions.npz'),scores=scores)
            (out/(kind+'_predictions.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
            predictions[kind]=(scores,rows)
    score_metrics(predictions,model_id,out)
    print('public_metrics',model_id,variant,phase,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);ap.add_argument('--variant',choices=['linear','mlp'],required=True);ap.add_argument('--phase',choices=['legacy','gemini'],required=True)
    args=ap.parse_args();run(args.model,args.variant,args.phase)
