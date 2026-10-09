#!/usr/bin/env python3
"""Train matched linear/MLP probes and report independent cached holdout scores."""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy.stats import pearsonr, spearmanr
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA
from study_paths import ROOT, CODE, RELEASE, SEED, read, write
from cache_dataset import FeatureData, collate
from probe_model import Probe

sys.path.insert(0, str(CODE))
from train_layered_curriculum import loss_terms


def projection(data, output):
    path = output / 'pca.npz'
    if path.exists():
        with np.load(path) as value:
            return {k: value[k] for k in value.files}
    indices = np.random.default_rng(SEED).choice(len(data), min(50000,len(data)), replace=False)
    x = np.stack([data.payload(data.rows[i])['embedding'].astype(np.float32) for i in indices])
    pca = PCA(n_components=min(256, x.shape[1], len(x)-1), svd_solver='randomized', random_state=SEED)
    z = pca.fit_transform(x)
    result = {'mean': pca.mean_.astype(np.float32), 'components': pca.components_.astype(np.float32),
              'scale': np.maximum(z.std(0),1e-5).astype(np.float32)}
    np.savez(path, **result)
    write(output / 'pca_fit.json', {'training_clips':len(data), 'fit_clips':len(indices),
                                   'training_key_sha256':__import__('hashlib').sha256('\n'.join(data.rows[i]['feature_key'] for i in indices).encode()).hexdigest(),
                                   'native_dim':x.shape[1], 'retained_dim':len(result['scale']), 'output_dim':256,
                                   'scope':'Training split only; validation and benchmark features never used'})
    return result


def loader(data, shuffle=False):
    return DataLoader(data,batch_size=256,shuffle=shuffle,num_workers=4,
                      multiprocessing_context='spawn',persistent_workers=True,collate_fn=collate)


def gpu(item,device):
    return {k:v.to(device) if torch.is_tensor(v) else v for k,v in item.items()}


def predict(model,batch):
    return model(batch['embedding'],batch['frame_features'],batch['event_starts'],batch['event_ends'])


@torch.no_grad()
def evaluate(model,data,device,classes,full=False,make_loader=None,predictor=None):
    model.eval()
    total=count=0
    predictions=[]; targets=[]; masks=[]
    speaker={'timbre':[],'identity':[]}; cps=[]; y_cps=[]
    burst_tp=burst_fp=burst_fn=0
    event_correct=event_count=0
    span_counts={threshold:[0,0,0] for threshold in (.1,.3,.5)}
    starts_error=[];ends_error=[];matched_class_correct=matched_class_total=0
    for item in (make_loader or loader)(data):
        batch=gpu(item,device)
        output=(predictor or predict)(model,batch)
        loss=sum(loss_terms(output,batch,classes).values())
        total+=float(loss)*len(batch['scores']);count+=len(batch['scores'])
        if not full:continue
        predictions.append(output['scores'].float().cpu().numpy())
        targets.append(batch['scores'].float().cpu().numpy())
        masks.append(batch['score_mask'].cpu().numpy())
        valid=batch['frame_mask'].bool();truth=batch['frame']>0;decision=output['frame']>=0
        burst_tp+=int((valid&truth&decision).sum());burst_fp+=int((valid&~truth&decision).sum());burst_fn+=int((valid&truth&~decision).sum())
        for name in speaker:
            mask=batch[name+'_valid'].bool()
            reference=torch.nn.functional.normalize(batch[name][mask].float(),dim=-1)
            speaker[name].extend((output[name][mask].float()*reference).sum(-1).cpu().tolist())
        valid_cps=batch['cps_valid'].bool()
        cps.extend(output['cps'][valid_cps].cpu().tolist());y_cps.extend(batch['cps'][valid_cps].cpu().tolist())
        event_mask=batch['event_valid'].bool()
        event_correct+=int((output['event_class'].argmax(-1)[event_mask]==batch['event_classes'][event_mask]).sum());event_count+=int(event_mask.sum())
        for i in range(len(batch['scores'])):
            length=int(batch['frame_mask'][i].bool().sum())
            if not length:continue
            prob=output['onset'][i,:length].float().sigmoid().cpu().numpy()
            peaks=[j for j in range(length) if prob[j]>=.5 and (j==0 or prob[j]>=prob[j-1]) and (j==length-1 or prob[j]>prob[j+1])]
            chosen=[]
            for j in sorted(peaks,key=lambda j:-prob[j]):
                if all(abs(j-k)>=3 for k in chosen):chosen.append(j)
                if len(chosen)==32:break
            predicted=[]
            for j in sorted(chosen):
                duration=max(1,round(math.expm1(float(output['log_duration'][i,j].clamp(0,8)))))
                predicted.append((j,min(length,j+duration)))
            if 'predicted_event_valid' in output:
                keep=output['predicted_event_valid'][i]
                predicted=list(zip(output['predicted_event_starts'][i,keep].cpu().tolist(),
                                   output['predicted_event_ends'][i,keep].cpu().tolist()))
            selected=batch['event_valid'][i].bool()
            reference=list(zip(batch['event_starts'][i,selected].cpu().tolist(),batch['event_ends'][i,selected].cpu().tolist()))
            overlap=np.zeros((len(reference),len(predicted)))
            for a,(s,e) in enumerate(reference):
                for b,(t,u) in enumerate(predicted):
                    intersection=max(0,min(e,u)-max(s,t));overlap[a,b]=intersection/max(1,e-s+u-t-intersection)
            matches=list(zip(*linear_sum_assignment(-overlap))) if overlap.size else []
            for threshold,counts in span_counts.items():
                tp=sum(bool(overlap[a,b]>=threshold) for a,b in matches)
                counts[0]+=tp;counts[1]+=len(predicted)-tp;counts[2]+=len(reference)-tp
            for a,b in matches:
                if overlap[a,b]<.1:continue
                s,e=reference[a];t,u=predicted[b]
                starts_error.append(abs(s-t)*.02);ends_error.append(abs(e-u)*.02)
                if 'predicted_event_class' in output:
                    guess=int(output['predicted_event_class'][i,output['predicted_event_valid'][i]][b].argmax())
                else:
                    local=batch['frame_features'][i,t:u].mean(0)
                    guess=int(model.event_head(local).argmax())
                target=int(batch['event_classes'][i,selected][a])
                matched_class_correct+=guess==target;matched_class_total+=1
    result={'loss':total/max(1,count),'clips':count}
    if full:
        p=np.concatenate(predictions);y=np.concatenate(targets);mask=np.concatenate(masks)
        norm=read(RELEASE/'training_normalization.json');per=[]
        for i,name in enumerate(norm['score_names']):
            take=mask[:,i];a=p[take,i];b=y[take,i]
            nonconstant=len(a)>2 and a.std()>1e-8 and b.std()>1e-8
            per.append({'score':name,'n':int(take.sum()),'raw_mae':float(np.abs(a-b).mean()*norm['score_std'][i]) if len(a) else None,
                        'normalized_mae':float(np.abs(a-b).mean()) if len(a) else None,
                        'pearson':float(pearsonr(a,b).statistic) if nonconstant else None,
                        'spearman':float(spearmanr(a,b).statistic) if nonconstant else None})
        result.update(per_score=per,speaker_cosine={k:{'n':len(v),'mean':float(np.mean(v)) if v else None} for k,v in speaker.items()},
                      frame_f1=2*burst_tp/max(1,2*burst_tp+burst_fp+burst_fn),frame_threshold=.5,
                      cps_raw_mae=float(np.abs(np.asarray(cps)-y_cps).mean()*norm['cps_std']) if cps else None,
                      burst_class_oracle_span_accuracy=event_correct/max(1,event_count),burst_class_events=event_count,
                      burst_localization={str(t):{'precision':tp/max(1,tp+fp),'recall':tp/max(1,tp+fn),'f1':2*tp/max(1,2*tp+fp+fn),'matched':tp} for t,(tp,fp,fn) in span_counts.items()},
                      burst_start_mae_s=float(np.mean(starts_error)) if starts_error else None,
                      burst_end_mae_s=float(np.mean(ends_error)) if ends_error else None,
                      burst_matched_event_class_accuracy=matched_class_correct/max(1,matched_class_total),
                      burst_matched_event_class_n=matched_class_total,
                      limitations='Oracle-span and predicted-span classification reported separately. Boundaries use fixed 0.5 onset threshold and native-feature interpolation; IoU matching uses Hungarian assignment.')
    return result


def run(model_id,variant,phase):
    cfg=read(ROOT/'study.json');device=torch.device('cuda',int(os.environ.get('LOCAL_RANK',0)))
    torch.cuda.set_device(device);torch.set_num_threads(2);torch.manual_seed(SEED)
    out=ROOT/'probes'/phase/model_id/variant;out.mkdir(parents=True,exist_ok=True)
    if (out/'COMPLETE.json').exists():return
    domains=['gemini'] if phase=='gemini' else ['ladder','p3_train','p3_validation','p3_test']
    train=FeatureData(model_id,domains,'train',phase)
    val=FeatureData(model_id,domains,'validation',phase)
    test=FeatureData(model_id,domains,'test',phase)
    if not train or not val or not test:raise RuntimeError('Training and two independent holdout partitions required')
    previous=None
    if phase=='gemini':
        previous=torch.load(ROOT/'probes/legacy'/model_id/variant/'best.pt',map_location='cpu',weights_only=False)
        pca=previous['pca'];np.savez(out/'pca.npz',**pca)
    else:pca=projection(train,out)
    for data in (train,val,test):data.projection=pca
    classes=read(RELEASE/'classes.json')['names']
    model=Probe(len(classes),variant=='linear').to(device)
    if previous:
        model.load_state_dict(previous['model'],strict=True)
        write(out/'pre_finetune_test_metrics.json',evaluate(model,test,device,torch.ones(len(classes),device=device),True))
        del previous
    weights=torch.ones(len(classes),device=device)
    # Stage P3 exposure is 10% of training examples at each stage.
    stages=['Gemini'] if phase=='gemini' else cfg['stages']
    training=[]
    for stage in stages:
        if phase=='gemini':data=train
        else:
            data=FeatureData(model_id,domains,'train',phase,stage);data.projection=pca
            ladder=[r for r in data.rows if r['source']=='ladder'];p3=[r for r in data.rows if r['source']=='p3_train']
            # Hash order avoids sample-ID blocks biasing the later small stages
            # toward one synthetic subtype or the positive/negative generation order.
            p3=sorted(p3,key=lambda r:__import__('hashlib').sha256((str(SEED)+r['feature_key']).encode()).digest())[:round(len(ladder)/9)]
            data.rows=ladder+p3
        training.append(data)
    epochs=cfg['gemini_probe_epochs'] if phase=='gemini' else cfg['probe_epochs_per_stage']
    updates=sum(math.ceil(len(data)/cfg['probe_batch'])*epochs for data in training)
    warm=max(1,round(updates*cfg['warmup_ratio']))
    def scale(u):
        if u<warm:return (u+1)/warm
        progress=min(1,(u-warm)/max(1,updates-warm))
        return .1+.9*.5*(1+math.cos(math.pi*progress))
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['probe_lr'],weight_decay=cfg['weight_decay'])
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,scale)
    best=float('inf');update=0;started=time.monotonic();completed=None
    if (out/'latest.pt').exists():
        saved=torch.load(out/'latest.pt',map_location='cpu',weights_only=False)
        if saved['variant']!=variant or saved['stage'] not in stages:
            raise RuntimeError('Probe resume checkpoint does not match the requested run')
        model.load_state_dict(saved['model'],strict=True)
        optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
        update=saved['update'];completed=(stages.index(saved['stage']),saved['epoch'])
        best=saved['best_validation_loss']
        if 'rng' in saved:
            torch.set_rng_state(saved['rng']['torch']);torch.cuda.set_rng_state(saved['rng']['cuda'],device)
            np.random.set_state(saved['rng']['numpy'])
        del saved
        print(model_id,variant,phase,'RESUME',completed,'update',update,flush=True)
    write(out/'config.json',{'model':model_id,'phase':phase,'variant':variant,'stages':stages,'epochs_per_stage':epochs,
                             'train_clips':len(train),'validation_clips':len(val),'test_clips':len(test),
                             'stage_training_clips':[len(d) for d in training], 'parameters':sum(p.numel() for p in model.parameters()),
                             'scalar_parameters_each':257 if variant=='linear' else 16513,
                             'backbone_trainable_parameters':0,'seed':SEED,'normalization':train.norm})
    for stage_index,(stage,data) in enumerate(zip(stages,training)):
        for epoch in range(epochs):
            if completed and (stage_index<completed[0] or (stage_index==completed[0] and epoch<completed[1])):
                continue
            model.train()
            for item in loader(data,True):
                batch=gpu(item,device)
                optimizer.zero_grad(set_to_none=True)
                output=predict(model,batch)
                loss=sum(loss_terms(output,batch,weights).values());loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                optimizer.step();scheduler.step();update+=1
                if update%50==0:
                    print(model_id,variant,phase,stage,'update',update,'of',updates,'loss',float(loss.detach()),flush=True)
            metrics=evaluate(model,val,device,weights)
            improved=metrics['loss']<best;best=min(best,metrics['loss'])
            state={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
                   'stage':stage,'epoch':epoch+1,'update':update,'variant':variant,'classes':classes,'pca':pca,'normalization':train.norm,
                   'best_validation_loss':best,'rng':{'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state(device),'numpy':np.random.get_state()}}
            temporary=out/'latest.pt.tmp';torch.save(state,temporary);os.replace(temporary,out/'latest.pt')
            if improved:
                __import__('shutil').copyfile(out/'latest.pt',out/'best.pt')
            __import__('shutil').copyfile(out/'latest.pt',out/(stage+'_epoch'+str(epoch+1)+'.pt'))
            with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps({'stage':stage,'epoch':epoch+1,'update':update,'validation':metrics})+'\n')
            print(model_id,variant,phase,stage,epoch+1,metrics,flush=True)
    model.load_state_dict(torch.load(out/'best.pt',map_location=device,weights_only=False)['model'])
    write(out/'test_metrics.json',evaluate(model,test,device,weights,True))
    write(out/'COMPLETE.json',{'updates':update,'best_validation_loss':best,'elapsed_s':time.monotonic()-started})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--model');parser.add_argument('--variant',choices=['linear','mlp']);parser.add_argument('--phase',choices=['legacy','gemini'],required=True);parser.add_argument('--all',action='store_true');parser.add_argument('--models',nargs='+')
    args=parser.parse_args()
    if args.all or args.models:
        jobs=[(m['id'],v) for m in read(ROOT/'study.json')['models'] if not args.models or m['id'] in args.models for v in ('linear','mlp')]
        rank=int(os.environ.get('LOCAL_RANK',0));world=int(os.environ.get('WORLD_SIZE',1))
        for model,variant in jobs[rank::world]:run(model,variant,args.phase)
    else:
        if not args.model or not args.variant:parser.error('--model and --variant required without --all')
        run(args.model,args.variant,args.phase)
