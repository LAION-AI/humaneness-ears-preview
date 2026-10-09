#!/usr/bin/env python3
"""Advance per-model probes; give full Whisper FT an independent GPU queue."""
import fcntl
import os
import socket
import subprocess
import time
from datetime import datetime, timezone
from study_paths import ROOT, HERE, CODE, GEMINI, read, write

ACTIVE={'PENDING','RUNNING','CONFIGURING','COMPLETING','SUSPENDED','REQUEUED'}
RESUMABLE={'TIMEOUT','PREEMPTED','NODE_FAIL','BOOT_FAIL'}


def retry_allocations(attempts):
    # A cancelled held script or a diagnosed/repaired startup/export failure
    # does not consume the three-allocation limit for interruption recovery.
    # Concrete failures still require a repair below before any retry.
    return sum(not (a.get('state') in {'FAILED','CANCELLED'} and
                    (a.get('repair') or a.get('replacement_reason'))) for a in attempts)


def state(job):
    r=subprocess.run(['sacct','-j',str(job),'-n','-X','--format=State','--parsable2'],
                     capture_output=True,text=True,check=True,timeout=15)
    lines=r.stdout.strip().splitlines()
    return lines[0].split('|')[0].split()[0].rstrip('+') if lines else 'UNKNOWN'


def submit(script,env,name,extra=()):
    r=subprocess.run(['sbatch','--parsable','--job-name='+name,
                      '--export=ALL,'+','.join(k+'='+str(v) for k,v in env.items()),
                      *extra,str(script)],capture_output=True,text=True,check=True,timeout=20)
    return int(r.stdout.strip().split(';')[0])


def domains_done(model,domains):
    return all((ROOT/'features'/model/(d+'-rank'+str(r)+'-COMPLETE.json')).exists()
               for d in domains for r in range(4))


def done_cache(model,cfg):
    return domains_done(model,['ladder','p3_train','p3_validation','p3_test','gemini',*cfg['benchmarks']])


def probe_done(phase,model,evaluation=False):
    out=ROOT/'probes'/phase/model
    if evaluation:
        return all((out/v/'public_metrics.json').exists() and
                   all((out/v/(k+'_matched_adapter.json')).exists() for k in ('crema','ravdess'))
                   for v in ('linear','mlp'))
    return all((out/v/'COMPLETE.json').exists() for v in ('linear','mlp'))


def tick(w,cfg):
    stamp=datetime.now(timezone.utc).isoformat()
    w.update(updated_utc=stamp,host=socket.gethostname(),pid=os.getpid())
    w['maximum_active_study_nodes']=5
    known={}
    for group in ('tasks','cache_jobs'):
        for name,attempts in w.get(group,{}).items():
            if attempts:
                a=attempts[-1];a['state']=state(a['id']);known[a['id']]=a['state']
    gate=(GEMINI/'prepared/TRAINING_READY.json').exists()
    # Release the already submitted four-GPU FT jobs immediately on final data.
    # Cache/probe capacity cannot delay this release; held jobs reserve no node.
    for model in ('base','small'):
        name='whisper-gemini-'+model;attempts=w['tasks'].get(name,[])
        if not attempts:raise RuntimeError('Missing pre-submitted full-FT chain: '+name)
        a=attempts[-1];out=GEMINI/'training'/('whisper_'+model)
        if a.get('held_for_targets') and gate and a['state']=='PENDING':
            subprocess.run(['scontrol','release',str(a['id'])],check=True,capture_output=True,text=True,timeout=20)
            a.update(held_for_targets=False,released_utc=stamp);write(ROOT/'workflow.json',w)
        elif a['state'] not in ACTIVE|{'UNKNOWN'} and not (out/'COMPLETE.json').exists():
            if retry_allocations(attempts)>=3:
                w.setdefault('errors',{})[name]='Three incomplete FT allocations; inspect saved logs.'
            elif a['state'] in RESUMABLE:
                env={'WHISPER_MODEL':model,'RESUME_CHECKPOINT':str(out/'latest.pt') if (out/'latest.pt').exists() else ''}
                job=submit(CODE/'gemini_finetune/train_whisper.sbatch',env,'flash38-fullft-'+model)
                attempts.append({'id':job,'state':'PENDING','submitted_utc':stamp,'resumed_from':a['id']})
                write(ROOT/'workflow.json',w)
            else:w.setdefault('errors',{})[name]='FT ended '+a['state']+'; repair the concrete error before another allocation.'
    parents=[w['tasks']['whisper-gemini-'+m][-1]['id'] for m in ('base','small')]
    eval_attempts=w['tasks'].setdefault('whisper-eval-all',[])
    done_whisper=all((ROOT/'whisper/gemini'/m/'COMPLETE.json').exists() for m in ('base','small'))
    # Reconnect evaluation to resumed FT jobs, rather than leaving a broken
    # afterok dependency attached to an obsolete wall-time-limited allocation.
    if not done_whisper:
        a=eval_attempts[-1] if eval_attempts else None
        if a and a['state']=='PENDING' and a.get('wait_for_ids')!=parents:
            subprocess.run(['scontrol','update','JobId='+str(a['id']),
                            'Dependency=afterok:'+':'.join(map(str,parents))],
                           check=True,capture_output=True,text=True,timeout=20)
            a['wait_for_ids']=parents;write(ROOT/'workflow.json',w)
        elif not a or a['state'] not in ACTIVE|{'UNKNOWN'}:
            if retry_allocations(eval_attempts)<3 and (not a or a['state'] in RESUMABLE|{'CANCELLED','COMPLETED'}):
                job=submit(HERE/'whisper_eval.sbatch',{'WHISPER_MODEL':'all'},'flash38-fullft-evaluation',
                           ['--dependency=afterok:'+':'.join(map(str,parents)),'--kill-on-invalid-dep=yes'])
                eval_attempts.append({'id':job,'state':'PENDING','wait_for_ids':parents,'submitted_utc':stamp})
                write(ROOT/'workflow.json',w)
            else:w.setdefault('errors',{})['whisper-eval-all']='Evaluation incomplete; inspect its concrete error.'
    # One probe/evaluation node runs alongside two cache nodes and two dedicated
    # FT nodes. Evaluation with an unmet dependency consumes no node or slot.
    busy_aux=False;active_aux=0
    for name,attempts in w['tasks'].items():
        if name.startswith('whisper-gemini-') or not attempts:continue
        a=attempts[-1]
        if a['state'] not in ACTIVE|{'UNKNOWN'}:continue
        if a.get('wait_for_ids') and a['state']=='PENDING' and not all(known.get(j)=='COMPLETED' for j in a['wait_for_ids']):continue
        busy_aux=True;active_aux+=1
    def probe_task(phase,models,evaluation):
        nonlocal busy_aux,active_aux
        if busy_aux or not models:return
        models=models[:2];name=('eval-' if evaluation else 'train-')+phase+'-'+'-'.join(models)
        attempts=w['tasks'].setdefault(name,[])
        if attempts and attempts[-1]['state'] in ACTIVE|{'UNKNOWN'}:return
        if retry_allocations(attempts)>=3 or (attempts and attempts[-1]['state'] not in RESUMABLE|{'COMPLETED'}):
            w.setdefault('errors',{})[name]='Incomplete allocation; repair its concrete error before retrying.';return
        env={'PROBE_PHASE':phase,'PROBE_MODELS':':'.join(models),'PROBE_EVAL':0}
        job=submit(HERE/('probe_eval.sbatch' if evaluation else 'train.sbatch'),env,name)
        attempts.append({'id':job,'state':'PENDING','submitted_utc':stamp,'models':models,'phase':phase,
                         'operation':'evaluation' if evaluation else 'training'})
        busy_aux=True;active_aux+=1;write(ROOT/'workflow.json',w)
    for phase in ('gemini','legacy'):
        ready=[m['id'] for m in cfg['models'] if probe_done(phase,m['id']) and not probe_done(phase,m['id'],True)
               and domains_done(m['id'],cfg['benchmarks'])]
        probe_task(phase,ready,True)
    ready=[m['id'] for m in cfg['models'] if gate and domains_done(m['id'],['gemini'])
           and probe_done('legacy',m['id']) and not probe_done('gemini',m['id'])]
    probe_task('gemini',ready,False)
    legacy_domains=['ladder','p3_train','p3_validation','p3_test']
    ready=[m['id'] for m in cfg['models'] if domains_done(m['id'],legacy_domains) and not probe_done('legacy',m['id'])]
    probe_task('legacy',ready,False)
    active_cache=sum(bool(a) and a[-1]['state'] in ACTIVE|{'UNKNOWN'} for a in w['cache_jobs'].values())
    active_ft=sum(w['tasks']['whisper-gemini-'+m][-1]['state'] in ACTIVE|{'UNKNOWN'} for m in ('base','small'))
    ft_finished=all((GEMINI/'training'/('whisper_'+m)/'COMPLETE.json').exists() for m in ('base','small'))
    # Keep FT capacity available until both full runs finish. Then reuse those
    # slots for remaining frozen encoders without exceeding five study nodes.
    cache_limit=min(4,max(0,5-active_ft-active_aux)) if ft_finished else 2
    w['cache_node_limit_now']=cache_limit
    w['scheduling_policy']['cache_after_full_ft']='Up to four; total study allocations stay at most five'
    pilot=state(w['pilot_job']);w['pilot_state']=pilot
    pilot_ok=pilot=='COMPLETED' and all((ROOT/'smoke/clapv2_xxs'/('CONTRACT-rank'+str(r)+'.json')).exists() for r in range(4))
    if pilot_ok:
        for m in cfg['models']:
            mid=m['id']
            if active_cache>=cache_limit:break
            if done_cache(mid,cfg):continue
            if m.get('repo') and not ((ROOT/'models'/mid/'MODEL_READY.json').exists() or (ROOT/'MODELS_READY.json').exists()):continue
            attempts=w['cache_jobs'].setdefault(mid,[])
            if attempts and attempts[-1]['state'] in ACTIVE|{'UNKNOWN'}:continue
            if retry_allocations(attempts)>=3 or (attempts and attempts[-1]['state'] not in RESUMABLE):
                w.setdefault('errors',{})[mid]='Encoder cache failed; inspect smoke/full-cache logs.';continue
            job=submit(HERE/'cache.sbatch',{'PROBE_MODEL':mid,'PROBE_SMOKE':0},'probe-cache-'+mid)
            attempts.append({'id':job,'state':'PENDING','submitted_utc':stamp});active_cache+=1
            write(ROOT/'workflow.json',w)
    else:w.setdefault('errors',{})['pilot']='Encoder/head smoke contract has not passed.'
    w['cache_states']={}
    for m in cfg['models']:
        attempts=w['cache_jobs'].get(m['id'],[])
        w['cache_states'][m['id']]='COMPLETE' if done_cache(m['id'],cfg) else attempts[-1]['state'] if attempts else 'NOT_SUBMITTED'
    complete=done_whisper and all(probe_done(p,m['id'],True) for m in cfg['models'] for p in ('legacy','gemini'))
    w['state']='COMPLETE' if complete else 'RUNNING_WITH_ERRORS' if w.get('errors') else 'RUNNING_OR_WAITING_FOR_DATA'
    write(ROOT/'workflow.json',w)
    return complete


def main():
    with (ROOT/'watcher.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return
        w=read(ROOT/'workflow.json');last_report=0
        while True:
            try:
                complete=tick(w,read(ROOT/'study.json'));w.pop('watcher_error',None)
                if complete or time.monotonic()-last_report>300:
                    subprocess.run(['python3',str(HERE/'write_report.py')],check=True,timeout=30)
                    r=subprocess.run(['bash',str(HERE/'publish.sh')],capture_output=True,text=True,timeout=120)
                    if r.returncode:w['publication_error']=r.stderr[-1200:]
                    else:w.pop('publication_error',None)
                    write(ROOT/'workflow.json',w);last_report=time.monotonic()
                if complete:return
            except Exception as error:
                w['watcher_error']=str(error)[-1800:];write(ROOT/'workflow.json',w)
            time.sleep(30)


if __name__=='__main__':
    main()
