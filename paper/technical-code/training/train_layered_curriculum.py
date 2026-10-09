#!/usr/bin/env python3
"""Base/Small S1→S10 training with layer residuals, event classes and CPS."""
from __future__ import annotations

import argparse
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
from torch.utils.data import DataLoader, Subset

from full_ladder_dataset import MixedDataset, StageBatches
from layered_curriculum_data import (LADDER_ROOT, P3_ROOT, LadderAuxDataset,
                                     LadderAuxVal, LayeredCollator, taxonomy)
from layered_multitask_model import LayeredMultiTaskWhisper
from multitask_p3_smoke import GROUPS
from p3_frozen_loader import P3FrozenDataset
from timbre_targets import TimbreIndex
from train import ROOT as OLD_ROOT, snapshot
from train_full_curriculum import (IDENTITY_INDEX, TIMBRE_INDEX, P3Val,
                                   digest, fill_precomputed, lr_factor, normalization)


STOP=False


def request_stop(_signal,_frame):
    global STOP
    STOP=True


def gpu_batch(batch: dict,device: torch.device) -> dict:
    fields=('mel','mel_mask','scores','score_mask','frame','frame_mask',
            'timbre','identity','timbre_valid','identity_valid','cps','cps_raw',
            'cps_valid','event_starts','event_ends','event_classes','event_valid',
            'event_span_valid')
    return {name:batch[name].to(device,non_blocking=True) for name in fields}


def loss_terms(out: dict,batch: dict,class_weights: torch.Tensor) -> dict[str,torch.Tensor]:
    pred=out['scores'].float()
    terms={}
    for name,(start,stop,weight) in GROUPS.items():
        mask=batch['score_mask'][:,start:stop].float()
        huber=F.huber_loss(pred[:,start:stop],batch['scores'][:,start:stop].float(),
                           delta=1.,reduction='none')
        terms[name]=weight*(huber*mask).sum()/mask.sum().clamp_min(1)
    valid=batch['frame_mask'].float()
    frame=batch['frame'].float()
    logits=out['frame'].float()
    bce=F.binary_cross_entropy_with_logits(logits,frame,
                                            pos_weight=torch.tensor(5.,device=frame.device),
                                            reduction='none')
    terms['frame_bce']=.5*(bce*valid).sum()/valid.sum().clamp_min(1)
    prob=logits.sigmoid()*valid
    overlap=(prob*frame).sum()
    terms['frame_dice']=.25*(1-(2*overlap+1)/
                                (prob.sum()+(frame*valid).sum()+1))
    onset_target=torch.zeros_like(out['onset'].float())
    span_mask=batch['event_span_valid'].bool()
    if span_mask.any():
        row=torch.arange(len(span_mask),device=span_mask.device)[:,None].expand_as(span_mask)
        start=batch['event_starts'].clamp(0,onset_target.shape[1]-1)
        onset_target[row[span_mask],start[span_mask]]=1.
        selected=out['log_duration'][row[span_mask],start[span_mask]].float()
        duration=(batch['event_ends']-batch['event_starts']).clamp_min(1)
        expected=duration[span_mask].float().log1p()
        terms['duration_huber']=.1*F.huber_loss(selected,expected,delta=.5)
    else:
        terms['duration_huber']=out['log_duration'].sum()*0
    onset_bce=F.binary_cross_entropy_with_logits(
        out['onset'].float(),onset_target,
        pos_weight=torch.tensor(100.,device=frame.device),reduction='none')
    terms['onset_bce']=.2*(onset_bce*valid).sum()/valid.sum().clamp_min(1)
    for name in ('timbre','identity'):
        mask=batch[f'{name}_valid'].bool()
        if mask.any():
            predicted=out[name][mask].float()
            reference=F.normalize(batch[name][mask].float(),dim=-1)
            terms[name]=.2*((1-(predicted*reference).sum(-1)).mean()+
                             .1*F.huber_loss(predicted,reference,delta=.1))
        else:
            terms[name]=out[name].sum()*0
    cps_mask=batch['cps_valid'].bool()
    terms['cps_huber']=(.2*F.huber_loss(out['cps'][cps_mask].float(),
                                      batch['cps'][cps_mask].float(),delta=1.)
                        if cps_mask.any() else out['cps'].sum()*0)
    event_mask=batch['event_valid'].bool()
    terms['event_class_ce']=(.2*F.cross_entropy(
        out['event_class'][event_mask].float(),batch['event_classes'][event_mask],
        weight=class_weights,label_smoothing=.02)
        if event_mask.any() else out['event_class'].sum()*0)
    return terms


@torch.no_grad()
def validate(core: LayeredMultiTaskWhisper,sources: dict,device: torch.device,
             world: int,indices: dict,class_weights: torch.Tensor,cps_mean: float,
             cps_std: float,n_classes: int) -> dict:
    core.eval()
    result={}
    names=list(GROUPS)+['frame_bce','frame_dice','timbre','identity',
                         'cps_huber','event_class_ce','onset_bce','duration_huber']
    for domain,loader in sources.items():
        sums=torch.zeros((len(names),2),dtype=torch.float64,device=device)
        extra=torch.zeros(7,dtype=torch.float64,device=device)
        confusion=torch.zeros((n_classes,n_classes),dtype=torch.int64,device=device)
        for raw in loader:
            fill_precomputed(raw,indices)
            batch=gpu_batch(raw,device)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                out=core(batch['mel'],batch['mel_mask'],
                         batch['event_starts'],batch['event_ends'])
                terms=loss_terms(out,batch,class_weights)
            for i,name in enumerate(names):
                if name in GROUPS:
                    a,b,_=GROUPS[name]
                    count=batch['score_mask'][:,a:b].any(dim=1).sum().double()
                elif name.startswith('frame_') or name=='onset_bce':
                    count=batch['frame_mask'].any(dim=1).sum().double()
                elif name in ('timbre','identity'):
                    count=batch[f'{name}_valid'].sum().double()
                elif name=='cps_huber':
                    count=batch['cps_valid'].sum().double()
                elif name=='duration_huber':
                    count=batch['event_span_valid'].sum().double()
                else:
                    count=batch['event_valid'].sum().double()
                sums[i,0]+=terms[name].double()*count
                sums[i,1]+=count
            frame_valid=batch['frame_mask'].bool()
            frame_truth=batch['frame'].bool()&frame_valid
            frame_pred=(out['frame'].float().sigmoid()>=.5)&frame_valid
            extra[0]+=(frame_pred&frame_truth).sum()
            extra[1]+=(frame_pred&~frame_truth).sum()
            extra[2]+=(~frame_pred&frame_truth).sum()
            cps_mask=batch['cps_valid']
            if cps_mask.any():
                raw_pred=out['cps'][cps_mask].float()*cps_std+cps_mean
                extra[3]+=(raw_pred-batch['cps_raw'][cps_mask]).abs().sum()
                extra[4]+=cps_mask.sum()
            event_mask=batch['event_valid']
            if event_mask.any():
                truth=batch['event_classes'][event_mask]
                prediction=out['event_class'][event_mask].float().argmax(-1)
                hist=torch.bincount(truth*n_classes+prediction,minlength=n_classes*n_classes)
                confusion+=hist.reshape(n_classes,n_classes)
                extra[5]+=(truth==prediction).sum()
                extra[6]+=truth.numel()
        if world>1:
            for item in (sums,extra,confusion):
                torch.distributed.all_reduce(item)
        metrics={name:(float(sums[i,0]/sums[i,1]) if sums[i,1] else None)
                 for i,name in enumerate(names)}
        metrics['weighted_loss_sum']=sum(x for x in metrics.values() if x is not None)
        tp,fp,fn=extra[:3].tolist()
        metrics['frame_f1_at_0.5']=2*tp/max(1,2*tp+fp+fn)
        metrics['cps_mae_raw']=float(extra[3]/extra[4]) if extra[4] else None
        metrics['cps_valid_clips']=int(extra[4])
        metrics['event_class_accuracy']=float(extra[5]/extra[6]) if extra[6] else None
        metrics['event_class_valid_events']=int(extra[6])
        if extra[6]:
            tp_class=confusion.diag().double()
            support=confusion.sum(1).double()
            predicted=confusion.sum(0).double()
            f1=2*tp_class/(support+predicted).clamp_min(1)
            metrics['event_class_macro_f1']=float(f1[support>0].mean())
        else:
            metrics['event_class_macro_f1']=None
        result[domain]=metrics
    core.train()
    return result


def local_rng():
    return {'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state(),
            'numpy':np.random.get_state()}


def gather_rng(world: int):
    mine=local_rng()
    if world==1:
        return [mine]
    gathered=[None]*world
    torch.distributed.all_gather_object(gathered,mine)
    return gathered


def save_checkpoint(path: Path,core,optimizer,*,world: int,
                    stage_index: int,batch_index: int,update: int,
                    best_by_stage: dict,hashes: dict,args):
    states=gather_rng(world)
    if torch.distributed.is_initialized() and torch.distributed.get_rank()!=0:
        return
    state={'model':core.state_dict(),'optimizer':optimizer.state_dict(),
           'scheduler':{'kind':'single_cosine','completed_updates':update,
                        'min_lr_ratio':args.min_lr_ratio,'warmup_ratio':args.warmup_ratio},
           'stage_index':stage_index,'batch_index':batch_index,'update':update,
           'best_by_stage':best_by_stage,'hashes':hashes,'config':vars(args),
           'rng_by_rank':states}
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.incomplete')
    torch.save(state,temp)
    os.replace(temp,path)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',choices=('base','small'),required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--resume',type=Path)
    ap.add_argument('--batch-per-gpu',type=int,default=16)
    ap.add_argument('--accum',type=int,default=1)
    ap.add_argument('--workers',type=int,default=4)
    ap.add_argument('--encoder-lr',type=float,default=None)
    ap.add_argument('--head-lr',type=float,default=3e-4)
    ap.add_argument('--min-lr-ratio',type=float,default=.10)
    ap.add_argument('--warmup-ratio',type=float,default=.02)
    ap.add_argument('--checkpoint-every',type=int,default=5000)
    ap.add_argument('--validation-every',type=int,default=5000)
    ap.add_argument('--minimum-stage-validations',type=int,default=4)
    ap.add_argument('--min-class-events',type=int,default=200)
    ap.add_argument('--max-updates',type=int,default=0,help='Smoke limit; zero means full run')
    args=ap.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('Booster GPU required')
    if args.encoder_lr is None:
        args.encoder_lr=2e-5 if args.model=='base' else 1.5e-5
    world=int(os.environ.get('WORLD_SIZE',os.environ.get('SLURM_NTASKS','1')))
    rank=int(os.environ.get('RANK',os.environ.get('SLURM_PROCID','0')))
    if world>1:
        os.environ.setdefault('RANK',str(rank))
        os.environ.setdefault('WORLD_SIZE',str(world))
        torch.distributed.init_process_group('nccl')
    torch.cuda.set_device(0)
    torch.set_num_threads(2)
    torch.manual_seed(640913+rank)
    np.random.seed(640913+rank)
    signal.signal(signal.SIGUSR1,request_stop)
    device=torch.device('cuda:0')

    schedule_path=P3_ROOT/'curriculum_mix_schedule.json'
    schedule=json.loads(schedule_path.read_text())
    if schedule['schema_version']!='p3-frozen-curriculum-v2':
        raise RuntimeError('unexpected S1→S10 schedule')
    hashes={name:digest(path) for name,path in {
        'schedule':schedule_path,'ladder_index':LADDER_ROOT/'manifest.json',
        'ladder_aux_schema':LADDER_ROOT/'ladder_aux_schema.json',
        'p3_index':P3_ROOT/'training_index_report.json',
        'p3_event_schema':P3_ROOT/'event_class_schema.json',
        'p3_score_normalization':P3_ROOT/'score_normalization.json',
        'ladder_score_normalization':OLD_ROOT/'manifest.json'}.items()}
    classes=taxonomy(args.min_class_events)
    mean,std,score_names=normalization()
    cps_schema=json.loads((LADDER_ROOT/'ladder_aux_schema.json').read_text())
    cps_mean=float(cps_schema['cps_filtered_train_mean'])
    cps_std=max(.1,float(cps_schema['cps_filtered_train_std']))
    cps_filter=cps_schema['cps_training_filter']
    cps_min=float(cps_filter['minimum'])
    cps_max=float(cps_filter['maximum'])
    ladder=LadderAuxDataset(mean,std)
    event_sidecar=P3_ROOT/'event_class_ids.npy'
    p3_train=P3FrozenDataset('train',score_mean=mean,score_std=std,
                             event_class_path=event_sidecar)
    if len(p3_train)!=schedule['p3_train_unique']:
        raise RuntimeError('P3 split changed')
    mixed=MixedDataset(ladder,p3_train)
    path=snapshot(args.model)
    def collator():
        return LayeredCollator(path,cps_mean=cps_mean,cps_std=cps_std,
                               ladder_lookup=classes['ladder_raw_to_canonical'],
                               p3_lookup=classes['p3_raw_to_canonical'],
                               cps_min=cps_min,cps_max=cps_max)
    def loader_options():
        options={'num_workers':args.workers,'pin_memory':True,
                 'persistent_workers':args.workers>0,'collate_fn':collator()}
        if args.workers:
            options['multiprocessing_context']='spawn'
        return options
    val_ladder=LadderAuxVal(mean,std)
    val_p3=P3Val(P3FrozenDataset('validation',score_mean=mean,score_std=std,
                                 event_class_path=event_sidecar))
    validation={
        'ladder':DataLoader(Subset(val_ladder,range(rank,len(val_ladder),world)),
                            batch_size=args.batch_per_gpu,**loader_options()),
        'p3':DataLoader(Subset(val_p3,range(rank,len(val_p3),world)),
                        batch_size=args.batch_per_gpu,**loader_options())}
    indices={'timbre':TimbreIndex(TIMBRE_INDEX,expected_dim=128),
             'identity':TimbreIndex(IDENTITY_INDEX,expected_dim=250)}
    core=LayeredMultiTaskWhisper(path,len(classes['names'])).to(device)
    model=DDP(core,device_ids=[0],broadcast_buffers=False) if world>1 else core
    heads=[parameter for name,parameter in core.named_parameters()
           if not name.startswith('encoder.')]
    optimizer=torch.optim.AdamW([
        {'params':core.encoder.parameters(),'lr':args.encoder_lr,'base_lr':args.encoder_lr},
        {'params':heads,'lr':args.head_lr,'base_lr':args.head_lr}],weight_decay=.01)
    class_weights=torch.tensor(classes['class_weights'],device=device,dtype=torch.float32)
    width=world*args.batch_per_gpu
    total_updates=sum(math.ceil((math.ceil(stage['ladder']/width)+
                                 math.ceil(stage['p3']/width))/args.accum)
                      for stage in schedule['stages'])
    warmup=round(total_updates*args.warmup_ratio)
    args.out.mkdir(parents=True,exist_ok=True)
    start_stage=start_batch=update=0
    best_by_stage={}
    if args.resume:
        state=torch.load(args.resume,map_location='cpu',weights_only=False)
        if state['hashes']!=hashes or state['config']['model']!=args.model:
            raise RuntimeError('resume checkpoint model or source hashes differ')
        settings=('batch_per_gpu','accum','encoder_lr','head_lr','min_lr_ratio',
                  'warmup_ratio','min_class_events','validation_every',
                  'minimum_stage_validations')
        if any(state['config'][name]!=getattr(args,name) for name in settings):
            raise RuntimeError('resume checkpoint hyperparameters differ')
        if len(state['rng_by_rank'])!=world:
            raise RuntimeError('resume checkpoint distributed world size differs')
        core.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        start_stage=int(state['stage_index'])
        start_batch=int(state['batch_index'])
        update=int(state['update'])
        best_by_stage=state['best_by_stage']
        rng=state.get('rng_by_rank',[None]*world)[rank]
        if rng:
            torch.set_rng_state(rng['torch'])
            torch.cuda.set_rng_state(rng['cuda'])
            np.random.set_state(rng['numpy'])
    if rank==0:
        (args.out/'classes.json').write_text(json.dumps(classes,indent=2)+'\n')
        (args.out/'config.json').write_text(json.dumps({**vars(args),'world':world,
            'total_updates':total_updates,'warmup_updates':warmup,
            'score_names':score_names,'score_mean':mean.tolist(),
            'score_std':std.tolist(),'cps_mean':cps_mean,'cps_std':cps_std,
            'cps_min':cps_min,'cps_max':cps_max,
            'hashes':hashes,'selection':'sum of fixed Ladder and P3 weighted validation losses',
            'stage_best':'lowest selection at validation checks within each stage'},
            indent=2,default=str)+'\n')
        print('CONFIG',args.model,'world',world,'updates',total_updates,
              'resume',start_stage,start_batch,update,'classes',len(classes['names']),flush=True)

    def evaluate(stage_index: int,next_stage: int,batch_index: int):
        nonlocal best_by_stage
        report=validate(core,validation,device,world,indices,class_weights,
                        cps_mean,cps_std,len(classes['names']))
        selection=report['ladder']['weighted_loss_sum']+report['p3']['weighted_loss_sum']
        if not math.isfinite(selection):
            raise RuntimeError(f'nonfinite validation selection at {stage_index}:{update}')
        stage_name=schedule['stages'][stage_index]['stage']
        old=float(best_by_stage.get(stage_name,math.inf))
        improved=selection<old
        if improved:
            best_by_stage[stage_name]=selection
        if rank==0:
            row={'stage':stage_name,'update':update,'batch_index':batch_index,
                 'validation':report,'selection_loss':selection,
                 'stage_best':best_by_stage[stage_name],'improved':improved}
            with (args.out/'metrics.jsonl').open('a') as stream:
                stream.write(json.dumps(row)+'\n')
            print('VALIDATION',json.dumps(row),flush=True)
        if improved:
            save_checkpoint(args.out/'best'/f'{stage_name}.pt',core,optimizer,
                            world=world,stage_index=stage_index,batch_index=batch_index,
                            update=update,best_by_stage=best_by_stage,hashes=hashes,args=args)
        save_checkpoint(args.out/'latest.pt',core,optimizer,world=world,
                        stage_index=next_stage,batch_index=0 if next_stage!=stage_index else batch_index,
                        update=update,best_by_stage=best_by_stage,hashes=hashes,args=args)
        if world>1:
            torch.distributed.barrier()

    for stage_index in range(start_stage,len(schedule['stages'])):
        stage=schedule['stages'][stage_index]
        sampler=StageBatches(ladder,schedule,stage_index,
                             batch_per_rank=args.batch_per_gpu,world=world,rank=rank,
                             start_batch=start_batch if stage_index==start_stage else 0)
        stage_updates=math.ceil(sampler.total_batches/args.accum)
        stage_val_interval=max(1,min(args.validation_every,
                                     math.ceil(stage_updates/(args.minimum_stage_validations+1))))
        loader=DataLoader(mixed,batch_sampler=sampler,**loader_options())
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for local_step,raw in enumerate(loader):
            batch_index=sampler.start_batch+local_step
            fill_precomputed(raw,indices)
            batch=gpu_batch(raw,device)
            final_batch=batch_index+1==sampler.total_batches
            sync=(batch_index+1)%args.accum==0 or final_batch
            group_start=(batch_index//args.accum)*args.accum
            group_size=min(args.accum,sampler.total_batches-group_start)
            context=nullcontext() if sync or world==1 else model.no_sync()
            with context:
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    output=model(batch['mel'],batch['mel_mask'],
                                 batch['event_starts'],batch['event_ends'])
                    terms=loss_terms(output,batch,class_weights)
                    loss=sum(terms.values())/group_size
                loss.backward()
            if not sync:
                continue
            factor=lr_factor(update,total_updates,warmup,args.min_lr_ratio)
            for group in optimizer.param_groups:
                group['lr']=group['base_lr']*factor
            nn.utils.clip_grad_norm_(core.parameters(),1.)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            update+=1
            if rank==0 and (update%100==0 or update==1):
                print('STEP',stage['stage'],update,'batch',batch_index+1,
                      'loss',float(loss.detach()*group_size),
                      'lr_encoder',optimizer.param_groups[0]['lr'],flush=True)
            requested=torch.tensor(int(STOP or bool(args.max_updates and
                                                     update>=args.max_updates)),
                                   dtype=torch.int32,device=device)
            if world>1:
                torch.distributed.all_reduce(requested,op=torch.distributed.ReduceOp.MAX)
            stop_now=bool(requested.item())
            if stop_now:
                save_checkpoint(args.out/'latest.pt',core,optimizer,world=world,
                                stage_index=stage_index,batch_index=batch_index+1,
                                update=update,best_by_stage=best_by_stage,hashes=hashes,args=args)
                if world>1:
                    torch.distributed.barrier()
                print('STOP',rank,stage['stage'],update,flush=True)
                if world>1:
                    torch.distributed.destroy_process_group()
                return
            stage_update=math.ceil((batch_index+1)/args.accum)
            if not final_batch and stage_update%stage_val_interval==0:
                evaluate(stage_index,stage_index,batch_index+1)
            elif not final_batch and update%args.checkpoint_every==0:
                save_checkpoint(args.out/'latest.pt',core,optimizer,world=world,
                                stage_index=stage_index,batch_index=batch_index+1,
                                update=update,best_by_stage=best_by_stage,hashes=hashes,args=args)
        evaluate(stage_index,stage_index+1,sampler.total_batches)
        start_batch=0
    ladder.close()
    p3_train.close()
    val_p3.p3.close()
    if rank==0:
        temp=args.out/'COMPLETE.json.incomplete'
        temp.write_text(json.dumps({'status':'complete','model':args.model,
                                    'updates':update,'hashes':hashes,
                                    'best_by_stage':best_by_stage},indent=2)+'\n')
        os.replace(temp,args.out/'COMPLETE.json')
    if world>1:
        torch.distributed.destroy_process_group()


if __name__=='__main__':main()
