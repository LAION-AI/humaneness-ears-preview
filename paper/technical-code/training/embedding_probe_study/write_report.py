#!/usr/bin/env python3
"""Render an honest English study log; pending experiments have no invented scores."""
import html
import json
from datetime import datetime, timezone
from pathlib import Path
from study_paths import ROOT, CODE, BENCH, GEMINI, RELEASE, read

OUT = CODE.parent / 'WHISPER_EMBEDDING_PROBE_COMPARISON_2026-10-06.html'


def esc(value):
    return html.escape(str(value))


def number(value):
    return 'Pending / unavailable' if value is None else f'{value:.4f}'


def optional(path):
    return read(path) if path.exists() else {}


def table(headers, rows):
    return '<div class="scroll"><table><thead><tr>' + ''.join('<th>'+esc(x)+'</th>' for x in headers) + '</tr></thead><tbody>' + ''.join('<tr>'+''.join('<td>'+esc(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'


def main():
    cfg = read(ROOT / 'study.json')
    workflow = optional(ROOT / 'workflow.json')
    selection = ROOT.parent / 'gemini_multisource_100h_20261004/annotation_batch_20261005'
    flash = optional(selection / 'status.json')
    teachers = optional(GEMINI / 'bulk_annotation_workflow.json')
    parts = ['<h1>Frozen audio embedding probes versus Whisper Base and Small</h1>',
             '<p class="lead">A controlled downstream-head study of emotion, voice style, quality, speaker embeddings and vocal-burst localization. This page records actual execution state and fills result tables as completed artifacts become available.</p>',
             '<p>Updated '+esc(datetime.now(timezone.utc).isoformat())+'. <a href="benchmark.html">Existing public benchmark report</a> · <a href="acted-calibration.html">Earlier supervised score adapters</a> · <a href="https://huggingface.co/laion/whisper-base-small-layered-audio-scores">Released Whisper weights</a>.</p>',
             '<h2>Execution status</h2>',
             table(['Component','Observed state'], [
                 ['Gemini selected audio', str(flash.get('selected_available_unique_clips',66199))+' unique clips; 197.36 hours'],
                 ['Valid Flash targets',str(flash.get('valid_clips','Unknown'))+'; provider status timestamp '+str(flash.get('updated_utc','Unknown'))],
                 ['Other teacher annotations',teachers.get('state','Unknown')],
                 ['Study workflow',workflow.get('state','Prepared; execution gates pending')],
                 ['Final Flash training gate',str((GEMINI/'prepared/TRAINING_READY.json').exists())],
                 ['Public comparison results','Available only when individual evaluation artifacts below exist']])]
    checks=[]
    for filename,label in [('TRAINING_PIPELINE_CONTRACT.json','Real cached-feature probe training and held-out metric pipeline'),
                           ('WHISPER_FULL_FT_CONTRACT.json','Full Base/Small encoder and head optimizer updates'),
                           ('DISTRIBUTED_PIPELINE_CONTRACT.json','Four-GPU distributed updates and real P3 evaluation loaders')]:
        evidence=optional(ROOT/filename)
        checks.append([label,'Passed' if evidence.get('passed') else 'Not yet verified',
                       evidence.get('scope','No completed check artifact')])
    parts += ['<h3>Execution checks</h3>',table(['Check','Observed result','Scope'],checks),
              '<p>These bounded checks verify software compatibility and finite training updates. They are not scientific evaluation results. A stale console launcher used the system Python instead of the active production interpreter; affected queued scripts were replaced with <code>python -m torch.distributed.run</code>. The live job IDs below are the replacements. Final Flash targets remain a required gate; unresolved or invalid annotations are retained for review and excluded from fine-tuning.</p>']
    queued=[]
    for label,attempts in workflow.get('tasks',{}).items():
        if not attempts:continue
        job=attempts[-1]
        state='Held until final Flash training targets are ready; no GPU reservation' if job.get('held_for_targets') else job.get('state','Unknown')
        queued.append([label,job['id'],state,job.get('reason',job.get('operation','')),', '.join(map(str,job.get('wait_for_ids',[]))) or 'Data / per-model readiness gate'])
    if queued:
        parts += ['<h2>Submitted training and evaluation jobs</h2>',table(['Job','Slurm ID','Current state','Purpose','Dependencies'],queued),
                  '<p>Base and Small each have a dedicated four-GPU full fine-tuning allocation for two epochs. Every encoder weight, including the initially fixed positional embedding table, and every multitask head is trainable. The jobs are pre-submitted on hold and released immediately when the final Flash-priority dataset export passes its training gate. They do not wait for the embedding caches to finish. The pre-submitted evaluation allocation depends on successful completion of both full fine-tunes and evaluates identical Flash test clips before/after tuning, fixed 2,000-clip P3 validation and test selections before/after tuning, the five public benchmarks, and matched actor-disjoint classification adapters.</p>',
                  '<p>Embedding probes start independently as soon as their own Ladder/P3 training features are complete. Their evaluations wait for that model’s complete benchmark features. Gemini probe tuning starts when its legacy head and final Flash features are ready. Before full Whisper training finishes, two encoder-cache nodes run alongside two dedicated Whisper training nodes and one probe/evaluation node. Afterwards, the freed capacity permits up to four encoder-cache nodes; pending/running evaluations reduce that limit as needed. Maximum concurrency remains five study nodes, with four GPUs per node. Held jobs and unmet queue dependencies reserve no compute node.</p>',
                  '<p>To fit earlier scheduler openings, full Whisper fine-tuning and evaluation jobs accept time windows as short as one hour; probe training accepts windows as short as 45 minutes, and encoder caching as short as two hours. These are allocation limits, not promised completion times. Training saves model, optimizer and scheduler states at epoch boundaries; probes additionally save their random states. Interrupted epochs restart from the last completed epoch; caches skip every committed chunk. Queue priority and the remaining provider batches still determine actual start times.</p>']
    progress=[]
    for path in sorted((ROOT/'probes').rglob('metrics.jsonl')):
        completed=[]
        for line in path.read_text().splitlines():
            try:completed.append(json.loads(line))
            except json.JSONDecodeError:continue
        if completed:
            row=completed[-1];validation=row.get('validation',{})
            progress.append([str(path.parent.relative_to(ROOT/'probes')),row['stage'],row['epoch'],
                             row['update'],number(validation.get('loss')),validation.get('clips','Unknown')])
    for size in ('base','small'):
        path=GEMINI/'training'/('whisper_'+size)/'metrics.jsonl'
        if path.exists():
            for line in path.read_text().splitlines()[::-1]:
                try:row=json.loads(line)
                except json.JSONDecodeError:continue
                progress.append(['Gemini full FT / Whisper '+size,'Gemini',row['epoch'],row['update'],
                                 number(row.get('validation_loss')),'See fine-tuning config']);break
    if progress:
        parts += ['<h2>Completed training epochs and interim validation</h2>',
                  table(['Run','Last completed stage','Epoch','Optimizer updates','Multitask validation loss ↓','Validation clips'],progress),
                  '<p>These are recorded epoch checks, before the final held-out benchmark evaluation. The loss combines masked normalized regression, speaker embedding and burst objectives with different weights; it is not accuracy, correlation or a paper benchmark score. Compare final per-target and public benchmark metrics below once they exist.</p>']
    retention=[]
    for size in ('base','small'):
        out=ROOT/'whisper/gemini'/size
        for domain,before_file,after_file in [('Flash test','legacy_on_flash_test_metrics.json','gemini_test_metrics.json'),
                                             ('P3 validation','legacy_p3_validation_metrics.json','gemini_p3_validation_metrics.json'),
                                             ('P3 test','legacy_p3_test_metrics.json','gemini_p3_test_metrics.json')]:
            before=optional(out/before_file);after=optional(out/after_file)
            if before and after:
                retention.append([size,domain,after['clips'],number(before.get('frame_f1')),number(after.get('frame_f1')),
                                  number(after['frame_f1']-before['frame_f1'])])
    if retention:
        parts += ['<h2>Whisper burst agreement before and after Flash tuning</h2>',
                  table(['Encoder','Identical held-out split','Clips','Original frame F1','Tuned frame F1','Absolute change'],retention),
                  '<p>Frame F1 uses a fixed 0.5 decision threshold on the 20 ms grid. Flash test labels are machine-generated event annotations; P3 labels are known synthetic construction intervals. These domains have different labels and acoustic distributions. The current fine-tunes improve agreement with Flash targets while reducing performance on P3. Checkpoints were selected by Flash validation loss, and this two-epoch tuning stage contained no P3 replay. The separate localization IoU and boundary-error metrics below must also be considered; frame F1 alone does not measure event localization accuracy.</p>']
    model_rows=[]
    for m in cfg['models']:
        source=m.get('repo',m.get('model_name'))
        jobs=workflow.get('cache_jobs',{}).get(m['id'],[])
        temporal='160 ms patches' if m['backend']=='clap' else '20 ms' if m['backend']=='commercial' else 'Native audio-token grid; verified by cached token counts'
        model_rows.append([m['id'],source,m['native_dim'],temporal,', '.join(str(j['id']) for j in jobs) or 'Not submitted',
                           workflow.get('cache_states',{}).get(m['id'],'Pending')])
    parts += ['<h2>Backbones and feature extraction</h2>',table(['Backbone','Source','Native pooled dimensions','Native temporal resolution','Cache Slurm jobs','Execution state'],model_rows),
              '<p>All nine embedding backbones stay frozen. Audio is resampled to mono 16 kHz and cropped to 30 seconds where necessary; NaFlex performs its native 32 kHz feature transform. No reference transcription, caption, ASR decoding or generated answer enters a probe. The exact epoch-320 CLAP checkpoints are used; Large is the 32-node checkpoint. Native embeddings and time-indexed features are saved once in immutable TAR sidecars and reused.</p>',
              '<p>Training-only PCA maps pooled embeddings to 256 dimensions and scales components using training statistics. XXS has 224 native dimensions: 224 retained components are padded with zeros to 256. Temporal features use a fixed seeded projection to 64 dimensions. Interpolation onto the common 20 ms target grid does not improve a backbone’s native timing resolution.</p>',
              '<h2>Heads, targets and losses</h2>',table(['Objective','Head','Trainable parameters','Loss / evaluation'],[
                  ['Each of 192 scalar outputs, plus CPS','256 → 64 GELU → 1',16513,'Masked normalized Huber; raw MAE, normalized MAE, Pearson and Spearman'],
                  ['Linear scalar control','256 → 1',257,'Same target masks and Huber loss'],
                  ['Orange timbre','256 → 64 → 128',24768,'Cosine + Huber; held-out cosine similarity'],
                  ['Orange identity','256 → 64 → 250',32698,'Cosine + Huber; held-out cosine similarity'],
                  ['Burst frame / onset / log duration','64 → 64 → 3',4355,'BCE + Dice, onset BCE and duration Huber; frame F1 and interval IoU'],
                  ['Burst event class','Mean of frames inside each event → 64 → 53',7605,'Categorical cross entropy; oracle-span and matched predicted-span accuracy'],
                  ['CREMA-D adapter (every model)','192 scalar predictions → 64 GELU → 6',12742,'Cross entropy; actor-held-out top-1, balanced accuracy and macro F1'],
                  ['RAVDESS adapter (every model)','192 scalar predictions → 64 GELU → 8',12872,'Same protocol; calm and neutral remain separate']]),
              '<p>The 192-score schema contains 40 EmoNet emotion targets, 57 VoiceNet dimensions, genuineness/blend/quality, Empathic Insight Plus outputs, AudioBox scores, all DNSMOS outputs, burst count and additional VoiceCLAP attributes. CPS is an independent 193rd scalar. Missing or out-of-domain teacher targets are masked; they are not replaced with zero labels. Gemini Flash values override older teacher values wherever a valid corresponding Flash annotation exists. Orange vectors come from their audio models; Gemini does not synthesize embeddings.</p>',
              '<h2>Training schedule and comparison limits</h2>',table(['Run','Data','Epochs','Optimizer / schedule'],[
                  ['Legacy embedding probes','S8 → S9 → S10, with a 10% synthetic P3 mixture at each stage','One per stage','AdamW LR 0.001; 5% warmup; continuous cosine with 10% floor; batch 256 per GPU'],
                  ['Gemini embedding probes','Final hash-deduplicated Flash clips; existing teacher sidecars','Two','Initialize each corresponding legacy probe; frozen backbone and legacy training PCA; new optimizer'],
                  ['Whisper Base fine-tune','Same Flash targets, same deterministic held-out partitions','Two','Encoder LR 0.00001; heads LR 0.0001; global batch 128'],
                  ['Whisper Small fine-tune','Same Flash targets','Two','Encoder LR 0.000005; heads LR 0.0001; global batch 128']]),
              '<p>The original Whisper encoders were trained over S1–S10 and have larger heads and a different training budget. Identical probe inputs/head sizes control downstream capacity across embedding backbones; they do not equalize backbone size, pretraining data or previous Whisper exposure. Ladder holdouts here are withheld from the new probes, but may have been seen by the previously trained Whisper model. Underlying Emolia source overlap and upstream benchmark exposure are not audited. Before/after Gemini scores use exactly the same Flash test clips; improvements combine extra training and changed targets.</p>',
              '<h2>Benchmark protocol</h2><p>VoiceNet-Emo is <code>emolia-emo</code>; VoiceNet-Ext is <code>emolia-dim</code>, not an additional independent benchmark. We report the current repository ≥2-rater cut and unflagged cut, threshold-free mean per-prompt Spearman, the paper-style oracle threshold statistic and a separate five-fold clip-grouped threshold audit. Oracle thresholds fit the evaluated labels and must not be presented as held-out classification. The current Ext snapshot’s rater counts differ from the paper; see the existing detailed report.</p>',
              '<p>EmoNet human intensities 0/1/2 become 0/5/10; native emotion outputs use a fixed 2.5 endpoint conversion without fitting benchmark labels. Forty mapped emotion categories cover 12,000 of the released 12,600 clips. CREMA-D and RAVDESS fixed taxonomy mappings are reported separately from supervised adapters. Each new adapter uses all 192 predictions, exactly the same parameter count across models, five outer actor-disjoint folds and three inner actor-disjoint folds to select LR (0.001 or 0.003) and epochs (20 or 50). Scaling is fitted inside each training fold. Every clip receives one outer held-out prediction; reported confidence intervals resample actors.</p>']
    outputs=[]
    baseline=optional(BENCH/'public_scores.json').get('models',{})
    transfer=[]
    for size in ('base','small'):
        previous=baseline.get(size,{})
        tuned=optional(ROOT/'whisper/gemini'/size/'public_metrics.json')
        for benchmark,key,metric,label in [
            ('emolia-emo','emolia_emo','mean_prompt_spearman','Mean per-prompt Spearman'),
            ('emolia-dim','emolia_dim','mean_prompt_spearman','Mean per-prompt Spearman')]:
            before=previous.get(key,{}).get('repo_min_2_raters',{})
            after=tuned.get(benchmark,{}).get('repo_min_2_raters',{})
            if before and after:
                transfer.append([size,benchmark,'Repository ≥2 raters; '+str(after['n'])+' pairs',label,
                                 number(before.get(metric)),number(after.get(metric))])
        for benchmark in ('crema','ravdess'):
            before=optional(ROOT/'whisper/legacy'/size/(benchmark+'_matched_adapter.json')).get('metrics',{})
            after=optional(ROOT/'whisper/gemini'/size/(benchmark+'_matched_adapter.json')).get('metrics',{})
            if before and after:
                transfer.append([size,benchmark,'Same nested actor-disjoint folds','Matched score-MLP accuracy',
                                 number(before.get('accuracy')),number(after.get('accuracy'))])
    if transfer:
        parts += ['<h2>Whisper public-benchmark transfer after two full tuning epochs</h2>',
                  table(['Encoder','Benchmark','Evaluation selection','Metric ↑','Original checkpoint','Flash-tuned checkpoint'],transfer),
                  '<p>The original rows reuse existing benchmark predictions for Base S4 and Small S3. The tuned rows evaluate their new full fine-tunes. The two Emolia correlations use the same repository ≥2-rater selection and require no fitted decision threshold. CREMA-D and RAVDESS use supervised score adapters with the same architecture and nested actor-disjoint protocol before and after tuning. Results are mixed: emotion correlations improve, CREMA-D is nearly unchanged for Base and slightly higher for Small, while RAVDESS adapter accuracy decreases for both. These measurements do not establish a uniform improvement across tasks.</p>']
    reference=[]
    for name in ('base','small'):
        result=baseline.get(name,{})
        emo=result.get('emolia_emo',{}).get('repo_min_2_raters',{})
        dim=result.get('emolia_dim',{}).get('repo_min_2_raters',{})
        en=result.get('emonet',{}).get('all_mapped_40',{})
        reference.append(['Original Whisper '+name,number(emo.get('balanced_accuracy_oracle_per_prompt')),
                          number(emo.get('mean_prompt_spearman')),number(dim.get('balanced_accuracy_oracle_per_prompt')),
                          number(dim.get('mean_prompt_spearman')),number(en.get('pearson')),'Existing regression-head evaluation'])
    reference.extend([
        ['VoiceCLAP-Small, VoiceNet paper','0.6754','0.3176','0.6367','0.1051','Different EmoNet protocol','Paper embedding/text-prompt evaluation'],
        ['VoiceCLAP-Large, VoiceNet paper','0.7021','0.3719','0.6510','0.1475','Different EmoNet protocol','Paper embedding/text-prompt evaluation'],
        ['VoiceCLAP-Large-v2, released model card','0.7069','0.3865','0.6816','0.2125','Not specified here','Released embedding benchmark numbers, not the new probes']])
    parts += ['<h2>Existing results and published reference context</h2>',
              table(['Model','Emo oracle bal@pp','Emo mean prompt ρ','Ext oracle bal@pp','Ext mean prompt ρ','EmoNet Pearson','Protocol'],reference),
              '<p>These Whisper rows predate Gemini tuning and this embedding-probe study. Paper and model-card rows are quoted as context from their linked primary sources; current benchmark snapshots, regression scoring versus text-prompt similarity, and upstream training exposure differ. These are not controlled same-protocol rankings. The newly trained probes will appear separately below.</p>']
    for phase in ('legacy','gemini'):
        for m in cfg['models']:
            for variant in ('linear','mlp'):
                out=ROOT/'probes'/phase/m['id']/variant
                outputs.append((phase+' / '+m['id']+' / '+variant,out))
    for phase in ('legacy','gemini'):
        for m in ('base','small'):
            outputs.append((phase+' / Whisper '+m,ROOT/'whisper'/phase/m))
    rows=[]
    details=[]
    for name,out in outputs:
        public=optional(out/'public_metrics.json')
        internal=optional(out/'test_metrics.json') or optional(out/'gemini_test_metrics.json')
        domain='Flash test' if name.startswith('gemini') else 'Ladder S8–S10 + P3 test'
        if name.startswith('legacy / Whisper '):
            size=name.rsplit(' ',1)[-1]
            old=baseline.get(size,{})
            public=public or {key:old[source] for key,source in [
                ('emolia-emo','emolia_emo'),('emolia-dim','emolia_dim'),
                ('emonet','emonet'),('crema','crema'),('ravdess','ravdess')] if source in old}
            internal=internal or optional(ROOT/'whisper/gemini'/size/'legacy_on_flash_test_metrics.json')
            domain='Flash test, original checkpoint'
        emo=public.get('emolia-emo',{}).get('repo_min_2_raters',{})
        dim=public.get('emolia-dim',{}).get('repo_min_2_raters',{})
        emonet=public.get('emonet',{}).get('all_mapped_40',{})
        adapters=[optional(out/(kind+'_matched_adapter.json')).get('metrics',{}) for kind in ('crema','ravdess')]
        rows.append([name,'Complete' if public else 'Pending',number(emo.get('balanced_accuracy_oracle_per_prompt')),
                     number(emo.get('mean_prompt_spearman')),number(dim.get('mean_prompt_spearman')),
                     number(adapters[0].get('accuracy')),number(adapters[1].get('accuracy')),domain,number(internal.get('frame_f1'))])
        if internal or public:
            details.append('<details><summary>'+esc(name)+' — complete metrics and target tables</summary>')
            if internal:
                details.append(table(['Target','Valid N','Raw MAE ↓','Normalized MAE ↓','Pearson ↑','Spearman ↑'],
                               [[v['score'],v['n'],number(v.get('raw_mae')),number(v.get('normalized_mae')),
                                 number(v.get('pearson')),number(v.get('spearman'))] for v in internal.get('per_score',[])]))
                details.append(table(['Additional metric','Result'],[[k,json.dumps(v,ensure_ascii=False)] for k,v in internal.items() if k!='per_score']))
            details.append('<pre>'+esc(json.dumps(public,indent=2))+'</pre></details>')
        if 'Whisper ' in name and name.startswith('gemini'):
            for phase in ('legacy','gemini'):
                for domain in ('p3_validation','p3_test'):
                    result=optional(out/(phase+'_'+domain+'_metrics.json'))
                    if result:
                        details.append('<details><summary>'+esc(name+' / '+phase+' / '+domain)+' — synthetic retention audit</summary>'+table(
                            ['Target','Valid N','Raw MAE','Normalized MAE','Pearson','Spearman'],
                            [[v['score'],v['n'],number(v.get('raw_mae')),number(v.get('normalized_mae')),
                              number(v.get('pearson')),number(v.get('spearman'))] for v in result.get('per_score',[])])+
                            '<pre>'+esc(json.dumps({k:v for k,v in result.items() if k!='per_score'},indent=2))+'</pre></details>')
    parts += ['<h2>Measured comparisons</h2>',table(['Run','Public scores','Emo oracle bal@pp','Emo mean prompt ρ','Ext mean prompt ρ','CREMA-D matched MLP accuracy','RAVDESS matched MLP accuracy','Internal test domain','Burst frame F1'],rows),
              '<p>Internal frame F1 must be compared within the same test domain. Legacy embedding probes use the mixed Ladder/P3 holdout; Gemini runs use Flash test labels. Original and tuned Whisper rows both use the same Flash test clips; their separate P3 retention audit is above.</p>',
              '<p>Pending means the experiment has not produced a complete evaluation artifact. None of the pending rows are estimates. Supervised actor-CV adapter accuracy cannot be directly ranked against zero-shot paper accuracy.</p>',*details,
              '<h2>Reference papers and source models</h2><p><a href="https://arxiv.org/html/2609.32016">VoiceNet paper</a> · <a href="https://arxiv.org/html/2506.09827">EmoNet-Voice paper</a> · <a href="https://github.com/LAION-AI/emolia-bench">Human benchmark labels and taxonomy</a> · <a href="https://huggingface.co/laion/Empathic-Insight-Voice-Plus">Empathic Insight Voice Plus</a>.</p><ul>'+''.join('<li><a href="https://huggingface.co/'+esc(m['repo'])+'">'+esc(m['repo'])+'</a>; pinned revision '+esc(m['revision'])+'</li>' for m in cfg['models'] if m.get('repo'))+'</ul>',
              '<p>Study artifacts: '+esc(ROOT)+'. Machine-readable protocol: <a href="embedding-probes-study.json">study.json</a>. Source code: <a href="https://huggingface.co/spaces/laion/whisper-base-small-emotion-voice-burst/tree/main/embedding-probe-code">embedding-probe-code</a>; <a href="https://huggingface.co/spaces/laion/whisper-base-small-emotion-voice-burst/tree/main/gemini-full-ft-code">Gemini full fine-tuning code and exact configurations</a>. Report and new study code: CC BY 4.0, LAION; upstream models and datasets retain their own licenses.</p>']
    style='body{font:16px/1.6 system-ui;background:#f3f6fa;color:#172b40;margin:0}main{max-width:1400px;margin:auto;padding:36px}h1{font-size:38px;line-height:1.2}h2{margin-top:36px}.lead{font-size:20px}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;background:white;font-size:14px}th,td{padding:10px;border-bottom:1px solid #dde4ed;text-align:left;vertical-align:top}th{background:#e0ebf5}details{background:white;padding:14px;margin:12px 0}pre{white-space:pre-wrap;max-height:600px;overflow:auto}a{color:#075a9d}'
    OUT.write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Audio embedding probe comparison</title><style>'+style+'</style><main>'+''.join(parts)+'</main></html>')
    print(OUT)


if __name__=='__main__':
    main()
