#!/usr/bin/env python3
"""Update the existing authorized HF report Space, preserving unrelated files."""
import hashlib
import re
from pathlib import Path
from huggingface_hub import HfApi, CommitOperationAdd, hf_hub_download
from study_paths import ROOT, HERE, read, write
from write_report import OUT


def main():
    cfg=read(ROOT/'study.json');repo=cfg['report_space'];api=HfApi()
    # Read the current remote card immediately before applying a narrow update.
    info=api.repo_info(repo_id=repo,repo_type='space')
    original=Path(hf_hub_download(repo,'README.md',repo_type='space',revision=info.sha)).read_text()
    begin,end='<!-- EMBEDDING_PROBES_20261006_START -->','<!-- EMBEDDING_PROBES_20261006_END -->'
    section=begin+'\n## Frozen embedding comparison study\n\n[English methods, execution status and measured comparisons](https://laion-whisper-base-small-emotion-voice-burst.static.hf.space/embedding-probes.html). '
    section+='Nine frozen audio backbones with matched linear/MLP heads; S8–S10 then Gemini target tuning, plus two-epoch Whisper Base/Small fine-tuning. '
    section+='CREMA-D/RAVDESS use identical actor-held-out score MLPs, reported separately from paper zero-shot results. Pending experiments are explicitly labeled.\n'+end
    card=re.sub(re.escape(begin)+'.*?'+re.escape(end),lambda _:section,original,flags=re.S) if begin in original else original.rstrip()+'\n\n'+section+'\n'
    local=ROOT/'publication';local.mkdir(exist_ok=True);(local/'README.md').write_text(card)
    operations=[CommitOperationAdd(path_in_repo=cfg['report_path'],path_or_fileobj=str(OUT)),
                CommitOperationAdd(path_in_repo='embedding-probes-study.json',path_or_fileobj=str(ROOT/'study.json')),
                CommitOperationAdd(path_in_repo='README.md',path_or_fileobj=str(local/'README.md'))]
    for path in sorted(HERE.iterdir()):
        if path.suffix in ('.py','.sh','.sbatch','.txt'):
            operations.append(CommitOperationAdd(path_in_repo='embedding-probe-code/'+path.name,path_or_fileobj=str(path)))
    from study_paths import CODE
    for name in ('common.py','data.py','train_whisper.py','train_whisper.sbatch'):
        path=CODE/'gemini_finetune'/name
        operations.append(CommitOperationAdd(path_in_repo='gemini-full-ft-code/'+name,path_or_fileobj=str(path)))
    for model in ('base','small'):
        path=CODE/'gemini_finetune/configs'/('whisper_'+model+'.json')
        operations.append(CommitOperationAdd(path_in_repo='gemini-full-ft-code/configs/'+path.name,path_or_fileobj=str(path)))
    # Only measured evaluation artifacts are published; no audio, credentials,
    # unfinished checkpoints or unrelated Space content enter this commit.
    for tree in ('probes','whisper'):
        for path in sorted((ROOT/tree).rglob('*.json')):
            if path.name.endswith(('metrics.json','matched_adapter.json')) or path.name=='COMPLETE.json':
                operations.append(CommitOperationAdd(path_in_repo='embedding-probe-results/'+str(path.relative_to(ROOT)),path_or_fileobj=str(path)))
    for name in ('TRAINING_PIPELINE_CONTRACT.json','WHISPER_FULL_FT_CONTRACT.json','DISTRIBUTED_PIPELINE_CONTRACT.json'):
        path=ROOT/name
        if path.exists():operations.append(CommitOperationAdd(path_in_repo='embedding-probe-results/software-checks/'+name,path_or_fileobj=str(path)))
    result=api.create_commit(repo_id=repo,repo_type='space',operations=operations,
                             commit_message='Update frozen audio embedding comparison methods and available results',parent_commit=info.sha)
    write(ROOT/'publication/latest.json',{'commit':result.oid,'url':result.commit_url,
          'report_url':'https://laion-whisper-base-small-emotion-voice-burst.static.hf.space/embedding-probes.html',
          'html_sha256':hashlib.sha256(OUT.read_bytes()).hexdigest()})
    print(result.commit_url)


if __name__=='__main__':
    main()
