#!/usr/bin/env python3
"""Pre-submit full FT on hold; queue four-rank evaluation after both runs."""
import fcntl
import subprocess
from datetime import datetime, timezone
from study_paths import ROOT, HERE, CODE, read, write


def queue(script,name,env,extra):
    r=subprocess.run(['sbatch','--parsable','--job-name='+name,
                      '--export=ALL,'+','.join(k+'='+str(v) for k,v in env.items()),
                      *extra,str(script)],capture_output=True,text=True,check=True,timeout=20)
    return int(r.stdout.strip().split(';')[0])


def main():
    with (ROOT/'watcher.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        w=read(ROOT/'workflow.json');ids=[]
        for model in ('base','small'):
            attempts=w.setdefault('tasks',{}).setdefault('whisper-gemini-'+model,[])
            if not attempts:
                job=queue(CODE/'gemini_finetune/train_whisper.sbatch','flash38-fullft-'+model,
                          {'WHISPER_MODEL':model,'RESUME_CHECKPOINT':''},['--hold'])
                attempts.append({'id':job,'state':'PENDING','held_for_targets':True,
                                 'submitted_utc':datetime.now(timezone.utc).isoformat(),
                                 'reason':'Release on final TRAINING_READY; full encoder and heads, two epochs, four GPUs'})
                write(ROOT/'workflow.json',w)
            ids.append(attempts[-1]['id'])
        attempts=w.setdefault('tasks',{}).setdefault('whisper-eval-all',[])
        if not attempts:
            job=queue(HERE/'whisper_eval.sbatch','flash38-fullft-evaluation',{'WHISPER_MODEL':'all'},
                      ['--dependency=afterok:'+':'.join(map(str,ids)),'--kill-on-invalid-dep=yes'])
            attempts.append({'id':job,'state':'PENDING','wait_for_ids':ids,
                             'submitted_utc':datetime.now(timezone.utc).isoformat(),
                             'reason':'Fine-tuned Base/Small on Flash holdout and all benchmarks; matched original Whisper MLPs'})
            write(ROOT/'workflow.json',w)
        w.update(scheduling_policy={'cache_nodes':2,'whisper_full_ft_nodes':2,'probe_or_evaluation_nodes':1,
                                    'whisper_release':'Independent of cache/probe capacity; release on final Flash export',
                                    'probe_start':'Individual model training domains complete'},
                 priority_updated_utc=datetime.now(timezone.utc).isoformat())
        write(ROOT/'workflow.json',w)
        print('WHISPER_FULL_FT',ids,'EVALUATION',attempts[-1]['id'])


if __name__=='__main__':
    main()
