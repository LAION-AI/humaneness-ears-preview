"""Consolidate completed probe and Whisper measurements into the main report.

Only saved measurements enter these tables. This module uses the standard
library and publishes no audio, model weights, predictions or credentials.
"""
from __future__ import annotations

import gzip
import html
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('/e/scratch/reformo/schuhmann1_moss/whisper_score_regression/embedding_probe_study_20261006')
GEMINI = ROOT.parent / 'gemini_finetune_preparation_20261005'
BENCH = ROOT.parent / 'bench_eval_20261004'
RELEASE = ROOT.parent / 'hf_layered_release/model'
HOME = Path('/e/home/jusers/schuhmann1/jupiter')
BUNDLE = HOME / 'WHISPER_ALL_BENCHMARK_RESULTS_2026-10-07.json.gz'
MANIFEST = ROOT / 'consolidated_benchmark/result_manifest.json'
PUBLIC_RESULTS = 'embedding-probe-results/'
LABELS = {'embeddinggemma2': 'EmbeddingGemma 2', 'ovis3b': 'Ovis Omni 3B',
          'voiceclap_large_v2': 'VoiceCLAP Large v2', 'voiceclap_commercial': 'VoiceCLAP Commercial',
          'clapv2_xxs': 'CLAPv2 XXS', 'clapv2_xs': 'CLAPv2 XS', 'clapv2_s': 'CLAPv2 S',
          'clapv2_m': 'CLAPv2 M', 'clapv2_l': 'CLAPv2 L'}


def esc(value):
    return html.escape(str(value), quote=True)


def read(path):
    return json.loads(Path(path).read_text())


def optional(path):
    return read(path) if Path(path).exists() else {}


def num(value, digits=4):
    return 'Not recorded / undefined' if value is None else f'{value:.{digits}f}'


def pct(value):
    return 'Not recorded / undefined' if value is None else f'{100 * value:.2f}%'


def table(headers, rows):
    return '<div class="tablewrap"><table class="sortable"><thead><tr>' + ''.join(
        '<th>' + esc(h) + '</th>' for h in headers) + '</tr></thead><tbody>' + ''.join(
        '<tr>' + ''.join('<td>' + esc(cell) + '</td>' for cell in row) + '</tr>'
        for row in rows) + '</tbody></table></div>'


def raw_link(path):
    return '<a href="' + esc(PUBLIC_RESULTS + str(path.relative_to(ROOT))) + '">Exact metric JSON</a>'


def score_table(metrics):
    return table(['Target (exact model key)', 'Valid clips N', 'Raw MAE ↓', 'Normalized MAE ↓',
                  'Pearson r ↑', 'Spearman ρ ↑'],
                 [[v['score'], v['n'], num(v.get('raw_mae')), num(v.get('normalized_mae')),
                   num(v.get('pearson')), num(v.get('spearman'))] for v in metrics.get('per_score', [])])


def internal_metrics(metrics):
    cps_raw = metrics.get('cps_raw_mae')
    cps_normalized = cps_raw / read(RELEASE / 'training_normalization.json')['cps_std'] if cps_raw is not None else None
    rows = [['Evaluated clips', metrics.get('clips')], ['Multitask loss ↓', num(metrics.get('loss'))],
            ['CPS raw MAE (characters/second) ↓', num(metrics.get('cps_raw_mae'))],
            ['CPS normalized MAE, derived using frozen training SD ↓', num(cps_normalized)],
            ['Frame F1 @ probability 0.5 ↑', pct(metrics.get('frame_f1'))],
            ['Oracle-span class accuracy ↑', pct(metrics.get('burst_class_oracle_span_accuracy'))],
            ['Oracle-span labeled events', metrics.get('burst_class_events')],
            ['Detected-span class accuracy, matched at IoU ≥0.1 ↑', pct(metrics.get('burst_matched_event_class_accuracy'))],
            ['Detected-span matched labeled events', metrics.get('burst_matched_event_class_n')],
            ['Start-time MAE, matches at IoU ≥0.1 (seconds) ↓', num(metrics.get('burst_start_mae_s'))],
            ['End-time MAE, matches at IoU ≥0.1 (seconds) ↓', num(metrics.get('burst_end_mae_s'))]]
    for name, vector in metrics.get('speaker_cosine', {}).items():
        rows += [[name.title() + ' mean cosine ↑', num(vector.get('mean'))],
                 [name.title() + ' valid single-speaker clips', vector.get('n')]]
    result = table(['Measurement', 'Recorded result'], rows)
    localization = metrics.get('burst_localization', {})
    result += table(['Interval IoU required', 'Precision ↑', 'Recall ↑', 'Event F1 ↑', 'Matched events'],
                    [[level, pct(v.get('precision')), pct(v.get('recall')), pct(v.get('f1')), v.get('matched')]
                     for level, v in localization.items()])
    return result + score_table(metrics)


def records(cfg, old_public):
    runs = []
    for phase in ('legacy', 'gemini'):
        for model in cfg['models']:
            for variant in ('linear', 'mlp'):
                out = ROOT / 'probes' / phase / model['id'] / variant
                runs.append({'id': 'probe/' + phase + '/' + model['id'] + '/' + variant,
                             'label': LABELS[model['id']] + ' / ' + variant.upper(), 'phase': phase,
                             'kind': 'frozen probe', 'head': variant, 'out': out,
                             'public': read(out / 'public_metrics.json'),
                             'internal': read(out / 'test_metrics.json'),
                             'domain': 'Flash test' if phase == 'gemini' else 'Ladder/P3 test',
                             'config': read(out / 'config.json'),
                             'completion': read(out / 'COMPLETE.json')})
    for phase in ('legacy', 'gemini'):
        for size in ('base', 'small'):
            out = ROOT / 'whisper' / phase / size
            public = (old_public[size] if phase == 'legacy' else read(out / 'public_metrics.json'))
            if phase == 'legacy':
                public = {key: public[source] for key, source in [
                    ('emolia-emo', 'emolia_emo'), ('emolia-dim', 'emolia_dim'),
                    ('emonet', 'emonet'), ('crema', 'crema'), ('ravdess', 'ravdess')]}
            internal_path = ROOT / 'whisper/gemini' / size / (
                'legacy_on_flash_test_metrics.json' if phase == 'legacy' else 'gemini_test_metrics.json')
            runs.append({'id': 'whisper/' + phase + '/' + size, 'label': 'Whisper ' + size.title(),
                         'phase': phase, 'kind': 'full encoder tuning' if phase == 'gemini' else 'original encoder',
                         'head': 'layer-pooled multitask heads', 'out': out,
                         'public': public, 'internal': read(internal_path), 'internal_path': internal_path,
                         'domain': 'Flash test', 'config': {}, 'completion': optional(out / 'COMPLETE.json')})
    for run in runs:
        run['adapters'] = {kind: read(run['out'] / (kind + '_matched_adapter.json'))
                           for kind in ('crema', 'ravdess')}
        run['validation'] = optional(run['out'] / 'validation_metrics.json')
    return runs


def benchmark_summary(runs):
    rows = []
    for run in runs:
        p = run['public']; emo = p['emolia-emo']['repo_min_2_raters']; dim = p['emolia-dim']['repo_min_2_raters']
        en = p['emonet']['all_mapped_40']
        rows.append([run['label'], run['phase'], num(en.get('pearson')), num(en.get('spearman')),
                     num(en.get('mae')), num(emo.get('mean_prompt_spearman')),
                     num(emo.get('balanced_accuracy_oracle_per_prompt')),
                     num(emo.get('balanced_accuracy_5fold_clip_grouped')),
                     num(dim.get('mean_prompt_spearman')),
                     num(dim.get('balanced_accuracy_oracle_per_prompt')),
                     num(dim.get('balanced_accuracy_5fold_clip_grouped')),
                     pct(run['adapters']['crema']['metrics']['accuracy']),
                     pct(run['adapters']['ravdess']['metrics']['accuracy'])])
    return table(['Model / head', 'Training phase', 'EmoNet Pearson ↑', 'EmoNet Spearman ↑', 'EmoNet MAE ↓',
                  'VoiceNet-Emo mean ρ ↑', 'Emo oracle bal@pp ↑', 'Emo held-clip threshold bal ↑',
                  'VoiceNet-Ext mean ρ ↑', 'Ext oracle bal@pp ↑', 'Ext held-clip threshold bal ↑',
                  'CREMA-D matched MLP accuracy ↑', 'RAVDESS matched MLP accuracy ↑'], rows)


def burst_summary(runs):
    rows = []
    for run in runs:
        x = run['internal']; loc = x['burst_localization']; speaker = x['speaker_cosine']
        rows.append([run['label'], run['phase'], run['domain'], x['clips'], pct(x.get('frame_f1')),
                     pct(loc['0.1']['f1']), pct(loc['0.3']['f1']), pct(loc['0.5']['f1']),
                     num(x.get('burst_start_mae_s')), num(x.get('burst_end_mae_s')),
                     pct(x.get('burst_class_oracle_span_accuracy')), pct(x.get('burst_matched_event_class_accuracy')),
                     num(speaker['timbre'].get('mean')), speaker['timbre']['n'],
                     num(speaker['identity'].get('mean')), speaker['identity']['n'], num(x.get('cps_raw_mae'))])
    return table(['Model / head', 'Phase', 'Test domain', 'Clips', 'Frame F1 ↑',
                  'Event F1 @ IoU .1 ↑', 'Event F1 @ IoU .3 ↑', 'Event F1 @ IoU .5 ↑',
                  'Start MAE s ↓', 'End MAE s ↓', 'Oracle-span class acc ↑', 'Detected-span class acc ↑',
                  'Timbre cosine ↑', 'Timbre N', 'Identity cosine ↑', 'Identity N', 'CPS MAE ↓'], rows)


def score_mean(run, prefix, field):
    values = [x[field] for x in run['internal']['per_score']
              if x['score'].startswith(prefix) and x.get(field) is not None]
    return sum(values) / len(values) if values else None


def task_leaders(runs):
    frozen = [r for r in runs if r['kind'] == 'frozen probe']
    flash = [r for r in frozen if r['phase'] == 'gemini']
    whisper = {r['label']: r for r in runs if r['id'].startswith('whisper/gemini/')}
    public_tasks = [
        ('EmoNet emotion intensity', 'Human labels; 12,000 clips; Pearson r ↑',
         lambda r: r['public']['emonet']['all_mapped_40']['pearson'], frozen, True),
        ('VoiceNet-Emo / emolia-emo', 'Human labels; ≥2 raters; mean prompt ρ ↑',
         lambda r: r['public']['emolia-emo']['repo_min_2_raters']['mean_prompt_spearman'], frozen, True),
        ('VoiceNet-Ext / emolia-dim', 'Human labels; ≥2 raters; mean prompt ρ ↑',
         lambda r: r['public']['emolia-dim']['repo_min_2_raters']['mean_prompt_spearman'], frozen, True),
        ('CREMA-D', 'Supervised matched actor-CV MLP; accuracy ↑',
         lambda r: r['adapters']['crema']['metrics']['accuracy'], frozen, True),
        ('RAVDESS speech', 'Supervised matched actor-CV MLP; accuracy ↑',
         lambda r: r['adapters']['ravdess']['metrics']['accuracy'], frozen, True),
        ('Orange timbre128 prediction', 'Same Flash test; mean cosine ↑',
         lambda r: r['internal']['speaker_cosine']['timbre']['mean'], flash, True),
        ('Orange identity250 prediction', 'Same Flash test; mean cosine ↑',
         lambda r: r['internal']['speaker_cosine']['identity']['mean'], flash, True),
        ('Characters per second', 'Same Flash test; raw MAE ↓',
         lambda r: r['internal']['cps_raw_mae'], flash, False),
        ('Burst frame detection', 'Same Flash test; frame F1 ↑',
         lambda r: r['internal']['frame_f1'], flash, True),
        ('Burst interval localization', 'Same Flash test; event F1 @ IoU .5 ↑',
         lambda r: r['internal']['burst_localization']['0.5']['f1'], flash, True),
        ('Burst class with true intervals', 'Same Flash test; oracle-span accuracy ↑',
         lambda r: r['internal']['burst_class_oracle_span_accuracy'], flash, True),
        ('40 emotion-target regressions', 'Same Flash test; mean normalized MAE ↓',
         lambda r: score_mean(r, 'emo_', 'normalized_mae'), flash, False),
        ('57 VoiceNet-target regressions', 'Same Flash test; mean normalized MAE ↓',
         lambda r: score_mean(r, 'vn_', 'normalized_mae'), flash, False),
        ('Genuineness (0–6)', 'Same Flash test; normalized MAE ↓',
         lambda r: score_mean(r, 'genuineness_0_6', 'normalized_mae'), flash, False),
        ('Vocal-burst blend (0–10)', 'Same Flash test; valid blend labels only; normalized MAE ↓',
         lambda r: score_mean(r, 'blend_0_10', 'normalized_mae'), flash, False),
        ('AudioBox aesthetics, four axes', 'Same Flash test; mean normalized MAE ↓',
         lambda r: score_mean(r, 'audiobox:', 'normalized_mae'), flash, False),
        ('DNSMOS, seven outputs', 'Same Flash test; mean normalized MAE ↓',
         lambda r: score_mean(r, 'dnsmos:', 'normalized_mae'), flash, False)]
    rows = []
    leaders = []
    for label, metric, extract, candidates, higher in public_tasks:
        best = (max if higher else min)(candidates, key=extract)
        b = whisper['Whisper Base']; s = whisper['Whisper Small']
        win = (max if higher else min)([best, b, s], key=extract)
        display = pct if 'accuracy' in metric or 'F1' in metric else num
        phase = lambda r: 'Flash' if r['phase'] == 'gemini' else 'S8–S10'
        rows.append([label, metric, best['label'] + ' / ' + phase(best), display(extract(best)),
                     display(extract(b)), display(extract(s)), win['label'] + ' / ' + phase(win)])
        leaders.append({'task': label, 'metric': metric, 'best_frozen_run': best['id'],
                        'best_frozen_value': extract(best), 'whisper_base_flash': extract(b),
                        'whisper_small_flash': extract(s), 'highest_measured_run': win['id']})
    result = table(['Task', 'Metric and selection', 'Best frozen embedding probe', 'Probe result',
                    'Whisper Base full FT', 'Whisper Small full FT', 'Best observed result'], rows)
    return result.replace('class="sortable"', 'class="sortable task-leaders"', 1), leaders


def quality_summary(runs):
    flash = [r for r in runs if r['phase'] == 'gemini']
    rows = []
    for r in flash:
        rows.append([r['label'], num(score_mean(r, 'emo_', 'normalized_mae')),
                     num(score_mean(r, 'vn_', 'normalized_mae')),
                     num(score_mean(r, 'audiobox:', 'normalized_mae')),
                     num(score_mean(r, 'audiobox:', 'spearman')),
                     num(score_mean(r, 'dnsmos:', 'normalized_mae')),
                     num(score_mean(r, 'dnsmos:', 'spearman')),
                     num(score_mean(r, 'eiv_extra:', 'normalized_mae')),
                     num(score_mean(r, 'voiceclap_attribute:', 'normalized_mae'))])
    return table(['Flash-tuned model / head', '40 emotion mean norm MAE ↓', '57 VoiceNet mean norm MAE ↓',
                  'AudioBox mean norm MAE ↓', 'AudioBox mean ρ ↑', 'DNSMOS mean norm MAE ↓',
                  'DNSMOS mean ρ ↑', 'Empathic extras mean norm MAE ↓', 'VoiceCLAP attributes mean norm MAE ↓'], rows)


def quality_axes(runs):
    flash = [r for r in runs if r['phase'] == 'gemini']
    frozen = [r for r in flash if r['kind'] == 'frozen probe']
    whisper = {r['label']: r for r in flash if r['id'].startswith('whisper/')}
    lookup = {r['id']: {s['score']: s for s in r['internal']['per_score']} for r in flash}
    names = [s['score'] for s in whisper['Whisper Base']['internal']['per_score']
             if s['score'].startswith(('audiobox:', 'dnsmos:')) or s['score'] in
             ('eiv_extra:score_content_enjoyment', 'eiv_extra:score_speech_quality',
              'eiv_extra:score_background_quality', 'eiv_extra:score_overall_quality',
              'eiv_extra:Background_Noise', 'eiv_extra:Recording_Quality', 'R_quality',
              'genuineness_0_6', 'blend_0_10')]
    rows = []
    for name in names:
        candidates = [r for r in frozen if lookup[r['id']][name].get('normalized_mae') is not None]
        best = min(candidates, key=lambda r: lookup[r['id']][name]['normalized_mae']) if candidates else None
        b = lookup[whisper['Whisper Base']['id']][name]; s = lookup[whisper['Whisper Small']['id']][name]
        p = lookup[best['id']][name] if best else {}
        rows.append([name, b['n'], s['n'], best['label'] if best else 'No valid targets',
                     num(p.get('normalized_mae')), num(b.get('raw_mae')), num(b.get('normalized_mae')),
                     num(b.get('spearman')), num(s.get('raw_mae')), num(s.get('normalized_mae')), num(s.get('spearman'))])
    return table(['Quality / enjoyment / genuineness target', 'Base valid N', 'Small valid N',
                  'Best Flash frozen probe by norm MAE', 'Probe norm MAE ↓', 'Base raw MAE ↓',
                  'Base norm MAE ↓', 'Base Spearman ↑', 'Small raw MAE ↓', 'Small norm MAE ↓', 'Small Spearman ↑'], rows)


def public_details(run):
    public = run['public']; parts = []
    en = public['emonet']
    parts.append('<h4>EmoNet intensity: all recorded cuts</h4>' + table(
        ['Cut', 'N', 'MAE ↓', 'RMSE ↓', 'Pearson ↑', 'Spearman ↑'],
        [[cut, en[cut]['n'], num(en[cut].get('mae')), num(en[cut].get('rmse')),
          num(en[cut].get('pearson')), num(en[cut].get('spearman'))]
         for cut in ('all_mapped_40', 'strict_unanimous', 'untruncated_mapped_40')]))
    parts.append('<details><summary>All 40 EmoNet emotion categories</summary>' + table(
        ['Emotion', 'N', 'MAE ↓', 'RMSE ↓', 'Pearson ↑', 'Spearman ↑'],
        [[r['category'], r['n'], num(r.get('mae')), num(r.get('rmse')), num(r.get('pearson')), num(r.get('spearman'))]
         for r in en['per_emotion']]) + '</details>')
    for kind in ('emolia-emo', 'emolia-dim'):
        x = public[kind]
        parts.append('<h4>' + esc(kind) + ': all label-selection audits</h4>' + table(
            ['Cut', 'Question pairs N', 'Prompts', 'Positive fraction', 'Mean prompt Spearman ↑',
             'Oracle bal@pp ↑', 'Held-clip threshold balanced accuracy ↑'],
            [[cut, x[cut]['n'], x[cut]['prompts'], pct(x[cut].get('positive_rate')),
              num(x[cut].get('mean_prompt_spearman')), num(x[cut].get('balanced_accuracy_oracle_per_prompt')),
              num(x[cut].get('balanced_accuracy_5fold_clip_grouped'))]
             for cut in ('all_repo_rows', 'repo_min_2_raters', 'unflagged_min_2_raters', 'untruncated_min_2_raters')]))
        for cut in ('all_repo_rows', 'repo_min_2_raters', 'unflagged_min_2_raters', 'untruncated_min_2_raters'):
            z = x[cut]
            parts.append('<details><summary>' + esc(kind + ' / ' + cut) + ': every dimension and prompt</summary>')
            parts.append(table(['Emotion / dimension', 'Questions N', 'Mean prompt ρ ↑', 'Oracle balanced accuracy ↑'],
                               [[r['family'], r['n'], num(r.get('mean_prompt_rho')), num(r.get('balanced_accuracy_oracle'))]
                                for r in z.get('per_family', [])]))
            parts.append('<details><summary>All individual prompts in this cut</summary>' + table(
                ['Prompt (dimension | ordinal level)', 'Questions N', 'Spearman ρ ↑', 'Oracle balanced accuracy ↑'],
                [[r['prompt'], r['n'], num(r.get('rho')), num(r.get('balanced_accuracy'))]
                 for r in z.get('per_prompt', [])]) + '</details></details>')
    parts.append('<h4>CREMA-D and RAVDESS: fixed mappings versus matched supervised MLPs</h4>' + table(
        ['Corpus', 'Protocol', 'Clips N', 'Accuracy ↑', 'Balanced accuracy ↑', 'Macro F1 ↑', 'Actor-bootstrap accuracy 95% CI'],
        [[kind, protocol, x['n'], pct(x.get('accuracy')), pct(x.get('balanced_accuracy')), pct(x.get('macro_f1')),
          '–'.join(pct(v) for v in x.get('accuracy_actor_bootstrap_95ci', [])) or 'Not recorded']
         for kind in ('crema', 'ravdess')
         for protocol, x in [('Fixed taxonomy; no corpus labels fitted', public[kind]),
                             ('Nested actor-disjoint matched MLP', run['adapters'][kind]['metrics'])]]))
    for kind in ('crema', 'ravdess'):
        adapter = run['adapters'][kind]
        parts.append('<details><summary>' + esc(kind) + ': every class, confusion matrix and matched-MLP fold</summary>')
        for protocol, x in [('Fixed mapping', public[kind]), ('Matched MLP', adapter['metrics'])]:
            parts.append('<h4>' + esc(protocol) + '</h4>' + table(['Class', 'Support N', 'Recall ↑', 'Precision ↑'],
                [[r['class'], r['support'], pct(r.get('recall')), pct(r.get('precision'))] for r in x['per_class']]))
            classes = list(x['confusion'])
            parts.append(table(['True label / predicted label'] + classes,
                               [[c] + [x['confusion'][c][other] for other in classes] for c in classes]))
        parts.append('<p>' + esc(adapter['protocol']) + '; ' + str(adapter['parameters']) + ' trainable parameters.</p>')
        parts.append(table(['Outer fold', 'Train clips', 'Test clips', 'Train actors', 'Test actors',
                            'Selected LR', 'Selected epochs', 'Inner accuracy', 'Outer accuracy'],
                           [[x['fold'], x['train_clips'], x['held_clips'], len(x['train_actors']), len(x['held_actors']),
                             x['selected']['lr'], x['selected']['epochs'], pct(x['selected']['inner_accuracy']),
                             pct(x['outer_accuracy'])] for x in adapter['folds']]))
        trials = [[x['fold'], t['lr'], t['epochs'], pct(t['inner_accuracy'])]
                  for x in adapter['folds'] for t in x['trials']]
        parts.append('<details><summary>Every inner-fold hyperparameter trial</summary>' + table(
            ['Outer fold', 'Candidate LR', 'Candidate epochs', 'Mean inner held-actor accuracy'], trials) + '</details>')
        parts.append(raw_link(run['out'] / (kind + '_matched_adapter.json')) + '</details>')
    return ''.join(parts)


def make_sections():
    cfg = read(ROOT / 'study.json'); workflow = read(ROOT / 'workflow.json')
    old = read(BENCH / 'public_scores.json')['models']; runs = records(cfg, old)
    audit = read(ROOT / 'consolidated_benchmark/job_audit.json')
    latest_ids = {workflow['pilot_job']} | {a[-1]['id'] for group in ('cache_jobs', 'tasks')
                                           for a in workflow[group].values()}
    by_id = {int(x['JobIDRaw']): x for x in audit['jobs']}
    if any(by_id[j]['State'] != 'COMPLETED' or by_id[j]['ExitCode'] != '0:0' for j in latest_ids):
        raise RuntimeError('A final study allocation is not successfully completed')
    if workflow['state'] != 'COMPLETE' or workflow.get('errors'):
        raise RuntimeError('Consolidation requires a complete error-free study')
    for run in runs:
        if len(run['internal']['per_score']) != 192:
            raise RuntimeError('Incomplete 192-target evaluation: ' + run['id'])
    stamp = datetime.now(timezone.utc).isoformat()
    gpu_hours = sum(int(x['ElapsedRaw']) * 4 / 3600 for x in audit['jobs'] if 'gres/gpu=4' in x['AllocTRES'])
    leader_table, leaders = task_leaders(runs)
    validation_job = optional(ROOT / 'consolidated_benchmark/validation_job.json')
    validation_count = sum(bool(r['validation']) for r in runs)
    result = ['<section id="study-overview"><div class="eyebrow">Completed comparison · 7 October 2026</div>',
              '<h2>What we trained, in plain language</h2>',
              '<p>We want one audio model to describe how a voice sounds: its emotions, speaking style, sound quality and non-speech vocal events such as laughter or crying. It also predicts two speaker-characteristic vectors and how many transcript characters are spoken per second. Every detected event has a start and an end time. These models use the Whisper audio encoder only; they do not transcribe speech, generate captions or identify a person by name.</p>',
              '<p>We tested two approaches. First, we kept nine existing audio embedding models fixed and trained small prediction heads on their saved audio features. Second, we trained all weights of our existing Whisper Base and Small multitask models for two further epochs using the final Gemini Flash 3.8 annotations. All models predict the same target schema; frozen embeddings use equal head capacities within each linear/MLP variant.</p>',
              '<p>An embedding is a list of numbers that represents an audio recording. A prediction head maps those numbers to an emotion or another target. We tested a single linear mapping and a small two-layer neural network, called an MLP. Full fine-tuning updates both the audio encoder and its output networks. An epoch is one pass through the training clips.</p>',
              '<p>The frozen models are EmbeddingGemma 2, Ovis Omni 3B, VoiceCLAP Large v2, VoiceCLAP Commercial, and five CLAPv2 sizes: XXS, XS, S, M and L. We tested a linear head and a small MLP for each, on both the older S8–S10 targets and the final Flash targets. Repository links and exact versions are in the training section.</p>',
              '<p>Human-rated benchmarks test emotion and voice-style agreement. Separate audio holdouts test agreement with the Flash and other teacher annotations for sound quality, speaker vectors and burst times. A model can lead on one task and lose on another, so the table below identifies the best measured result for each task.</p>',
              '<p>EmoNet-Voice measures emotion intensity in synthetic speech. VoiceNet-Emo / emolia-emo measures 40 emotions in real speech. VoiceNet-Ext / emolia-dim measures 57 voice-style dimensions: EXT and DIM are two names for the same benchmark. CREMA-D and RAVDESS test six and eight basic acted-emotion classes respectively.</p>',
              '<p>This is the current consolidated report. The original S1–S10 Whisper evaluation and paper context remain below as historical measurements. New results are read directly from completed experiment artifacts, including all measurements in <a href="embedding-probes.html">the embedding-probe report</a>.</p>',
              '<h3>Which model is best for which task?</h3>',
              '<p>For accuracy, F1, correlations and cosine, higher is better. For mean absolute error (MAE), lower is better. Correlation 1 means a perfect linear relationship or ordering; cosine 1 means the same vector direction; MAE 0 means no error. A normalized MAE of 0.20 means an average error of 0.20 training standard deviations. Each row uses one stated metric; a model can rank differently on another metric.</p>', leader_table,
              '<p class="note">Public-benchmark probe winners are selected from all 36 measured probe configurations; Flash-target winners are selected from the 18 Flash-tuned probes on the same 3,544 test clips. Genuineness and vocal-burst blend are compared against valid Flash annotations, not independent human ratings. A null blend on a clip without bursts is excluded rather than treated as zero; per-target sample counts, raw MAE and correlations are in the detailed model panels. Picking the largest test score is descriptive and does not demonstrate statistical superiority. Family MAEs are unweighted means over valid targets. Only matched nested actor-CV classifier results are compared in the acted-speech rows.</p>',
              '<h3>How our Whisper fine-tunes compare</h3>',
              '<p><strong>Whisper Small is the strongest measured choice for speaker-vector regression, the AudioBox/DNSMOS target families and EmoNet intensity correlation.</strong> Base has the highest measured VoiceNet-Emo rank correlation and the lowest CPS error. Both full fine-tunes reproduce the new Flash burst membership labels more closely than the frozen probes. Their advantages on teacher-scored holdouts are not independent human quality judgments.</p>',
              '<p><strong>Frozen VoiceCLAP Large v2 probes are stronger on CREMA-D and RAVDESS after matched supervised actor adaptation.</strong> The best frozen probe also has better Flash event-interval F1 than either full Whisper fine-tune, despite lower frame F1. VoiceNet-Ext correlations remain low for every model and are close enough that a universal style winner would be misleading. The full Whisper fine-tunes also lose accuracy on the original P3 burst domain.</p>',
              '<h3>Study completion and data</h3>',
              table(['Component', 'Verified outcome'], [
                  ['Frozen embedding backbones', '9; all feature caches complete'],
                  ['Probe configurations', '36 = 9 backbones × linear/MLP × legacy/Flash; all trained and evaluated'],
                  ['Whisper full fine-tunes', 'Base and Small; two epochs each; all encoder weights and heads trainable'],
                  ['Public benchmarks per configuration', 'EmoNet-Voice, VoiceNet-Emo, VoiceNet-Ext, CREMA-D, RAVDESS'],
                  ['Final Slurm allocations', str(len(latest_ids)) + ' including smoke pilot; every final state COMPLETED with exit 0:0'],
                  ['Workflow completion', workflow['updated_utc'] + ' (UTC)'],
                  ['Flash dataset', '65,163 valid annotations; 58,380 train / 3,239 validation / 3,544 test'],
                  ['Excluded annotations', '1,036 invalid or blocked; retained for review, excluded from tuning'],
                  ['Tracked study allocation usage', num(gpu_hours, 2) + ' GH200 GPU-hours; entire four-GPU allocations counted'],
                  ['Original training and benchmark jobs', 'None pending or running'],
                  ['Supplementary full validation evaluation', str(validation_count) + '/40 configurations evaluated; ' +
                   ('all metrics complete' if validation_count == 40 else 'Slurm job ' + str(validation_job.get('id', 'not submitted yet')) + ' processes the saved best checkpoints')]]),
              '<p class="note">Allocation usage includes the tracked encoder caches, head training, full fine-tuning, evaluation and repaired failed attempts. It excludes earlier S1–S10 pretraining, Gemini API charges, teacher backfills, model downloads and other projects. Superseded held jobs used no GPU time. Other account jobs are outside this study.</p>',
              '<h3>What changed, and what did not improve</h3><ul>',
              '<li>Full Whisper tuning improves agreement with Flash targets, including emotion ranking, speaker-vector regression and CPS. Flash labels are machine annotations, not an independent human-label gold standard.</li>',
              '<li>Whisper Base/Small frame F1 rises from 0.307/0.317 to 0.693/0.689 on the same Flash test clips. Event F1 at IoU 0.5 remains approximately 0.194/0.195: high frame agreement does not imply reliable event localization.</li>',
              '<li>P3 test frame F1 decreases from 0.962/0.969 to 0.805/0.839 after full tuning. This stage used no P3 replay and selected checkpoints by Flash validation loss.</li>',
              '<li>The strongest frozen-probe acted-speech result is VoiceCLAP Large v2 with linear heads on legacy targets: CREMA-D accuracy 72.41%, RAVDESS 76.94%, with supervised nested actor-disjoint score MLPs. These are not paper-style zero-shot results.</li>',
              '<li>Model and dataset size do not produce a consistent ranking across tasks. Compare the same target, test selection and head protocol; there is no single overall winner.</li></ul>',
              '<p>Report generated ' + esc(stamp) + '. <a href="benchmark-results-2026-10-07.json.gz">Download every exact result, configuration, fold and job receipt (compressed JSON)</a>.</p></section>']
    result += ['<section id="study-protocol"><h2>Protocols and metric definitions</h2>',
               table(['Experiment', 'Audio / targets', 'Training and checkpoint selection'], [
                   ['Legacy frozen probes', 'S8 → S9 → S10; 10% P3 at each stage', 'One epoch per stage; backbone frozen; validation-selected best head'],
                   ['Flash frozen probes', 'Same 58,380 final Flash train clips as Whisper tuning', 'Two epochs from corresponding legacy best; backbone and legacy PCA frozen'],
                   ['Original Whisper comparison', 'Base S4 and Small S3, selected stage checkpoints from the S1–S10 campaign', 'Original fixed encoder and layered heads; re-evaluated on identical Flash/P3 selections'],
                   ['Full Whisper Flash tuning', 'Same Flash split; missing values masked; Flash targets override older scores', 'Two epochs; entire encoder plus heads optimized; Flash validation selects best'],
                   ['Matched acted-emotion MLP', '192 raw output scores, not hidden representations', 'Same capacity for all models; five actor-disjoint outer folds; three inner folds select LR/epochs']]),
               '<p>Legacy probe test contains 12,059 mixed Ladder/P3 clips; Flash test contains 3,544 clips. These domains have different teacher targets. Probe PCA uses only legacy training features, maps pooled vectors to 256 dimensions and scales training components; XXS retains 224 components and pads zeros. Temporal features use a fixed seeded projection to 64 dimensions. Native CLAP patches are 160 ms; interpolation to 20 ms does not add timing information. Backbones receive audio only; reference captions, transcripts and ASR outputs do not enter inference.</p>',
               table(['Metric', 'Meaning / interpretation'], [
                   ['Raw MAE ↓', 'Mean absolute error in the target’s native units. burst_count_log1p is log(1 + count), not count error.'],
                   ['Normalized MAE ↓', 'MAE after frozen training mean/SD normalization; benchmark/test labels never fit these statistics.'],
                   ['Pearson r / Spearman ρ ↑', 'Linear agreement / ordering. Undefined if the reference or predictions are constant or too few labels exist.'],
                   ['Emolia mean prompt ρ ↑', 'Mean per-query ranking agreement against human yes-vote share; threshold-free.'],
                   ['Oracle bal@pp ↑', 'Per-prompt balanced accuracy with threshold fitted on the same evaluated labels; optimistic separability statistic.'],
                   ['Held-clip threshold accuracy ↑', 'Five-fold audit; threshold fitted on other clips. Ext exact audio duplicates are grouped by hash.'],
                   ['Acted matched-MLP accuracy ↑', 'Supervised outer-fold actor-held-out classification; not zero-shot. Same MLP architecture for all compared models.'],
                   ['Speaker cosine ↑', 'Cosine against Orange model vectors: timbre128 / identity250. Only valid single-speaker targets are scored.'],
                   ['CPS MAE ↓', 'Error in literal Unicode transcript characters per second; an independent 193rd scalar output.'],
                   ['Frame F1 ↑', 'Binary vocal-burst membership on the 20 ms grid at fixed probability threshold 0.5.'],
                   ['Event F1 @ IoU ↑', 'One-to-one Hungarian assignment of predicted start/end intervals to references at the stated overlap threshold.'],
                   ['Start/end MAE ↓', 'Mean boundary error in seconds, conditional on a matched event at IoU ≥0.1; missed bursts are not included.'],
                   ['Oracle-span class accuracy ↑', '53-class event recognition given reference start/end times; only pooled frames inside the event are used.'],
                   ['Detected-span class accuracy ↑', 'Recognition on predicted intervals matched at IoU ≥0.1; conditional, not end-to-end class F1.']]),
               '<p class="warn">The new study decodes onset/duration event proposals (up to 32 per clip); its event metric differs from the original report’s continuous-positive-frame regions and greedy matching. Compare old and new event scores only with this distinction in mind. Both use a fixed 0.5 operating threshold, but event decoding and matching are different. The original 9,693-clip P3 test and the new fixed 2,000-clip P3 audit also differ in coverage.</p>',
               '<p>VoiceNet-Ext and emolia-dim name the same benchmark, not two separate tests. Current ≥2-rater coverage is 7,986 Emo questions and 13,917 Ext questions, with an additional unflagged audit. EmoNet evaluates 12,000 mapped clips; Arousal and Authenticity are unsupported by the 40-emotion head. Scores use the fixed 0–4 → 0–10 endpoint conversion; clips longer than 30 seconds use the first 30 seconds. Source overlap with upstream pretraining/Emolia and the ladder is not exhaustively audited. Original Whisper training budgets and head sizes differ from frozen probes. The original campaign ran through S10; evaluation restores the earlier validation-selected S4/S3 checkpoints rather than the final S10 weights.</p></section>']
    result += ['<section id="study-public"><h2>All 40 model/phase combinations: public benchmark results</h2>',
               '<p>legacy = S8–S10 head training for frozen probes, or original S1–S10-trained Whisper. gemini = two further epochs on final Flash targets. All acted accuracy columns below use the same 192 → 64 GELU → class MLP; the older 80-feature logistic adapters remain separately labeled below.</p>',
               benchmark_summary(runs), '</section>',
               '<section id="study-audio-targets"><h2>Speaker vectors, CPS and vocal-burst timing for every run</h2>',
               '<p>Compare within a test domain. All Whisper rows and all gemini probe rows use the same 3,544 Flash test clips; legacy probe rows use the mixed Ladder/P3 holdout. Valid speaker count is shown beside cosine; only 1,718 Flash test clips satisfy the valid single-speaker target mask. Boundary and detected-class metrics include matched events only.</p>',
               burst_summary(runs), '</section>']
    result += ['<section id="study-quality"><h2>Emotion, voice-style, aesthetics and DNSMOS target agreement</h2>',
               '<p>All rows here use the same Flash test selection. A family mean averages its recorded valid targets equally, not its clips. The detailed native-unit errors and per-target counts appear in the model panels below. These are agreement scores against recorded annotations, not ratings from a fresh listening panel.</p>',
               quality_summary(runs),
               '<h3>Every aesthetics, DNSMOS and related quality axis</h3>', quality_axes(runs),
               '<p><a href="https://huggingface.co/facebook/audiobox-aesthetics">AudioBox Aesthetics</a> supplies Content Enjoyment (CE), Content Usefulness (CU), Production Complexity (PC) and Production Quality (PQ). <a href="https://github.com/microsoft/DNS-Challenge/tree/master/DNSMOS">DNSMOS</a> supplies speech signal (SIG), background (BAK), overall quality (OVRL), their raw variants and P808 MOS. Empathic extras include recording/background/speech quality, content enjoyment, valence/arousal and other voice attributes. MOS means mean opinion score; here its target is an automatic model prediction.</p></section>']
    retention = []; audits = []
    for size in ('base', 'small'):
        out = ROOT / 'whisper/gemini' / size
        for phase in ('legacy', 'gemini'):
            for domain in ('p3_validation', 'p3_test'):
                path = out / (phase + '_' + domain + '_metrics.json'); x = read(path)
                audits.append((size, phase, domain, path, x))
                retention.append([size.title(), phase, domain, x['clips'], pct(x['frame_f1']),
                                  pct(x['burst_localization']['0.5']['f1']), num(x['burst_start_mae_s']), num(x['burst_end_mae_s']),
                                  num(x['speaker_cosine']['timbre']['mean']), num(x['speaker_cosine']['identity']['mean'])])
    result += ['<section id="study-retention"><h2>Full Whisper tuning: P3 retention before and after</h2>',
               table(['Encoder', 'Phase', 'Identical P3 selection', 'Clips', 'Frame F1 ↑', 'Event F1 @ IoU .5 ↑',
                      'Start MAE s ↓', 'End MAE s ↓', 'Timbre cosine ↑', 'Identity cosine ↑'], retention),
               '<p>P3 validation and test selections each contain 2,000 clips. Within each split, exactly the same audio and masks are evaluated before/after tuning. P3 has constructed burst intervals, while Flash targets are model-generated event annotations. The two domains measure different kinds of agreement. Synthetic sample partitions are disjoint; source-pool disjointness for this new cached selection is not fully audited.</p>']
    for size, phase, domain, path, x in audits:
        result.append('<details><summary>' + esc('Whisper ' + size.title() + ' / ' + phase + ' / ' + domain) +
                      ': all 192 scores and event metrics</summary>' + internal_metrics(x) + raw_link(path) + '</details>')
    result.append('</section>')
    result += ['<section id="study-targets"><h2>Complete per-run results: every score, emotion, dimension, class and fold</h2>',
               '<p>Each panel contains the 192 native scalar targets, all recorded benchmark cuts, 40 EmoNet emotions, every VoiceNet dimension and ordinal prompt, speaker vectors, CPS, burst boundaries/classes and acted-emotion confusions. Missing/undefined measurements are explicit. Expand a model to read its tables; headings in all tables can be clicked to sort.</p>']
    extras = []
    for run in runs:
        result.append('<details class="study-run"><summary>' + esc(run['label'] + ' / ' + run['phase'] + ' / ' + run['domain']) + '</summary>')
        result.append('<h3>Held-out audio targets</h3>' + internal_metrics(run['internal']))
        path = run.get('internal_path', run['out'] / 'test_metrics.json')
        result.append(raw_link(path))
        if run['validation']:
            validation_note = ('The original Whisper checkpoint was selected using Ladder validation. This row re-evaluates it on the same Flash validation clips as its tuned counterpart.'
                               if run['kind'] == 'original encoder' else
                               'This split selected the checkpoint. Its metrics are validation measurements, not an untouched test.')
            result.append('<details><summary>Full selection-validation metrics: every score and class objective</summary>' +
                          '<p>' + esc(validation_note) + '</p>' +
                          internal_metrics(run['validation']) + raw_link(run['out'] / 'validation_metrics.json') + '</details>')
        previous = run['out'] / 'pre_finetune_test_metrics.json'
        if previous.exists():
            pre = read(previous); extras.append((str(previous.relative_to(ROOT)), pre))
            result.append('<details><summary>Same Flash test: legacy head before its two tuning epochs</summary>' +
                          internal_metrics(pre) + '</details>')
        result.append(public_details(run) + '</details>')
    result.append('</section>')
    training = []
    validation = []
    for run in runs:
        c = run['config']
        if c:
            training.append([run['label'], run['phase'], ', '.join(c['stages']), c['epochs_per_stage'],
                             c['train_clips'], c['validation_clips'], c['test_clips'],
                             ', '.join(map(str, c['stage_training_clips'])), c['parameters'], c['scalar_parameters_each'],
                             run['completion']['updates'], num(run['completion']['best_validation_loss'])])
            for line in (run['out'] / 'metrics.jsonl').read_text().splitlines():
                row = json.loads(line)
                validation.append([run['label'], run['phase'], row['stage'], row['epoch'], row['update'],
                                   row['validation']['clips'], num(row['validation']['loss'])])
    for size in ('base', 'small'):
        for line in (GEMINI / 'training' / ('whisper_' + size) / 'metrics.jsonl').read_text().splitlines():
            row = json.loads(line)
            validation.append(['Whisper ' + size.title(), 'gemini full FT', 'Flash', row['epoch'],
                               row['update'], 3239, num(row['validation_loss'])])
    result += ['<section id="study-reproduction"><h2>Executed training details and reproducibility</h2>',
               table(['Model / head', 'Phase', 'Stages', 'Epochs per stage', 'Unique train pool', 'Validation clips',
                      'Test clips', 'Actual stage exposure counts', 'Total head parameters', 'Parameters per scalar',
                      'Optimizer updates', 'Best validation loss ↓'], training),
               '<p>Probe heads use AdamW LR 0.001, weight decay 0.01, batch 256 per GPU, 5% warmup and continuous cosine to a 10% floor. MLP scalars use 256 → 64 GELU → 1 (16,513 parameters each); linear scalars use 256 → 1 (257 each). Other heads are capped at 50,000 parameters. In the linear control, an oversized speaker head is factorized through a 64-dimensional linear layer. The backbone has zero trainable parameters for every probe.</p>',
               '<p>Full Whisper Base uses encoder LR 0.00001; Small uses 0.000005; both use head LR 0.0001, global batch 128, two epochs/914 optimizer updates and a four-GPU allocation. Every encoder parameter, including positional embeddings, and every head is trainable. Losses combine masked normalized Huber, cosine/Huber speaker targets, frame BCE/Dice, onset BCE, duration Huber and event categorical cross entropy. Checkpoints store model, optimizer, schedule and random states. Flash tuning has no P3 replay.</p>',
               '<h3>Every recorded validation result</h3>',
               '<p>The training loop recorded multitask validation loss, not a scalar-regression “accuracy”. This combined loss selected checkpoints and cannot be interpreted as a percentage; compare losses only within the same phase and target masks. The supplementary evaluation below measures actual validation class accuracy, frame/event F1, speaker cosine and CPS from the saved best checkpoint. Recomputed holdout losses use unit burst-class weights; full Whisper training selected checkpoints using its configured class-balancing weights, so those loss values differ. P3 validation is in the retention table; acted-classifier inner/outer accuracies are in each model panel.</p>',
               table(['Model / head', 'Phase', 'Stage', 'Completed epoch', 'Optimizer updates', 'Validation clips',
                      'Recorded multitask validation loss ↓'], validation),
               '<h3>Full validation quality of the selected best checkpoint</h3>',
               '<p>These existing validation clips participated in checkpoint selection. Accuracy is reported for the burst class head; continuous regression targets use MAE and correlation. The complete per-target validation tables are inside each model panel. Original Whisper rows here use the same Flash validation audio as tuned Whisper.</p>',
               burst_summary([{**r, 'internal': r['validation'], 'domain': 'Flash validation' if r['domain'] == 'Flash test' else 'Ladder/P3 validation'}
                              for r in runs if r['validation']]),
               '<p>The matched acted MLP uses 192 raw scores, training-fold StandardScaler, 64 GELU hidden units and class logits: 12,742 parameters for six-class CREMA-D; 12,872 for eight-class RAVDESS. Five outer actor folds and three inner folds select LR 0.001/0.003 and 20/50 epochs. Outer test actors never fit the scaler or MLP. Confidence intervals resample actors. This exploratory supervised adaptation is different from paper zero-shot comparison and from the earlier 80-score logistic adapter.</p>',
               '<details><summary>Backbones, exact revisions and native feature widths</summary>' + table(
                   ['Backbone', 'Source', 'Revision / original checkpoint', 'Native pooled dimensions'],
                   [[LABELS[m['id']], m.get('repo', m.get('model_name')), m.get('revision', m.get('checkpoint')), m['native_dim']]
                    for m in cfg['models']]) + '</details>',
               '<h3>Models, teachers and training data repositories</h3><ul>' + ''.join(
                   '<li><a href="https://huggingface.co/' + esc(m['repo']) + '">' + esc(m['repo']) + '</a></li>'
                   for m in cfg['models'] if m.get('repo')) +
               '<li><a href="https://huggingface.co/laion/whisper-base-small-layered-audio-scores">Our original Whisper checkpoint repository</a>; initialized from <a href="https://huggingface.co/openai/whisper-base">OpenAI Whisper Base</a> and <a href="https://huggingface.co/openai/whisper-small">OpenAI Whisper Small</a>.</li>'
               '<li><a href="https://huggingface.co/datasets/laion/tts-scaling-ladder-de-en">DE/EN Scaling Ladder</a>; final Flash targets from its balanced selections plus Talent, 40×3k and POC source subsets.</li>'
               '<li><a href="https://huggingface.co/laion/Empathic-Insight-Voice-Plus">Empathic Insight Voice Plus</a>; <a href="https://github.com/LAION-AI/voicenet/blob/main/taxonomy/emonet_taxonomy.md">40-emotion taxonomy</a>; <a href="https://github.com/LAION-AI/voicenet/blob/main/taxonomy/voicenet_taxonomy.md">57 VoiceNet dimensions</a>; <a href="https://huggingface.co/laion/voicenet-dimension-predictors-commercial">VoiceNet teacher predictors</a>.</li>'
               '<li><a href="https://huggingface.co/laion/voiceclap-commercial-genuineness">Genuineness teacher</a>; <a href="https://huggingface.co/laion/voiceclap-commercial-vocalburst-blend">Vocal-burst blend teacher</a>; <a href="https://huggingface.co/laion/voiceclap-commercial-attribute-heads">VoiceCLAP attribute heads</a>.</li>'
               '<li><a href="https://huggingface.co/Orange/Speaker-wavLM-tbr">Orange timbre128 teacher</a>; <a href="https://huggingface.co/Orange/Speaker-wavLM-id">Orange identity250 teacher</a>; <a href="https://github.com/LAION-AI/emotion-annotations/blob/main/generate_timbre_embeddings.py">Timbre generation code</a>.</li>'
               '<li><a href="https://huggingface.co/laion/vocalburst-locator">Vocal-burst locator</a>; <a href="https://github.com/LAION-AI/voice-taxonomies">event taxonomies</a>.</li></ul>',
               '<details><summary>Canonical 53 vocal-burst class labels</summary>' + table(
                   ['Class index', 'Canonical label'], enumerate(read(RELEASE / 'classes.json')['names'])) + '</details>',
               '<p>The supplementary validation produced all 40 metric files successfully. Its original trailing upload step failed because the compute node has no external network route. Publication was recovered on the login node; the Slurm receipt retains that post-evaluation failure for an accurate audit.</p>' if validation_job.get('publication_recovery') else '',
               '<details><summary>Every Slurm attempt: repaired failures, replacements and final receipts</summary>' + table(
                   ['Slurm job', 'Name', 'Final study job?', 'State', 'Exit', 'Allocated elapsed seconds', 'Resources'],
                   [[x['JobIDRaw'], x['JobName'], 'Yes' if int(x['JobIDRaw']) in latest_ids else
                     'Supplementary validation; publication recovered on login' if int(x['JobIDRaw']) == validation_job.get('id') else 'Superseded attempt',
                     x['State'], x['ExitCode'], x['ElapsedRaw'], x['AllocTRES']] for x in audit['jobs']]) + '</details>',
               '<p>Machine-readable <a href="benchmark-result-manifest.json">result inventory</a> and <a href="benchmark-results-2026-10-07.json.gz">complete compressed result bundle</a>. Source: <a href="embedding-probe-code/benchmark_supplement.py">consolidation code</a>, <a href="https://huggingface.co/spaces/laion/whisper-base-small-emotion-voice-burst/tree/main/embedding-probe-code">probe training and scoring code</a>, <a href="https://huggingface.co/spaces/laion/whisper-base-small-emotion-voice-burst/tree/main/gemini-full-ft-code">full Whisper tuning code</a>. Original normalization: <a href="https://huggingface.co/laion/whisper-base-small-layered-audio-scores/blob/main/training_normalization.json">training_normalization.json</a>.</p>',
               '<p>Study code and this report: CC BY 4.0, LAION. Upstream models and source datasets retain their own licenses. Published paper tables and earlier adapter studies follow in the historical sections; their original protocols and coverage remain visible.</p></section>']
    bundle_runs = []
    for run in runs:
        value = {k: v for k, v in run.items() if k not in ('out', 'internal_path', 'config')}
        value['config'] = {k: v for k, v in run['config'].items() if k != 'normalization'}
        bundle_runs.append(value)
    payload = {'schema': 'consolidated-audio-benchmarks-v1', 'generated_utc': stamp,
               'study': cfg, 'workflow': workflow, 'job_audit': audit,
               'normalization': read(RELEASE / 'training_normalization.json'), 'runs': bundle_runs,
               'task_leaders': leaders, 'validation_history': validation, 'supplementary_validation_job': validation_job,
               'whisper_p3_audits': [{'size': s, 'phase': p, 'domain': d, 'metrics': x}
                                    for s, p, d, _, x in audits],
               'probe_before_flash_tuning': dict(extras)}
    BUNDLE.write_bytes(gzip.compress(json.dumps(payload, ensure_ascii=False, allow_nan=False).encode(), mtime=0))
    manifest = {'schema': payload['schema'], 'generated_utc': stamp, 'study_completed_utc': workflow['updated_utc'],
                'probe_runs': 36, 'whisper_runs': 4, 'all_run_ids': [x['id'] for x in runs],
                'scalar_target_count': 192, 'extra_scalar': 'CPS', 'recorded_job_attempts': len(audit['jobs']),
                'latest_successful_jobs_including_pilot': len(latest_ids), 'tracked_gh200_gpu_hours': gpu_hours,
                'result_bundle': 'benchmark-results-2026-10-07.json.gz',
                'p3_retention_evaluations': len(audits), 'probe_pre_flash_evaluations': len(extras)}
    manifest['full_validation_configurations'] = validation_count
    MANIFEST.parent.mkdir(exist_ok=True); MANIFEST.write_text(json.dumps(manifest, indent=2) + '\n')
    return result
