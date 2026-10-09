#!/usr/bin/env python3
"""Freeze backbone revisions, head budgets and comparison protocols."""
from pathlib import Path
from study_paths import ROOT, SEED, write, CLAP_CODE, CLAP_BENCH, TRANSFORMERS_REVISION


def main():
    models = [
        {'id': 'embeddinggemma2', 'repo': 'google/embeddinggemma-2', 'revision': '914f7f89142e33e77833254d9c9b90c3cef7303b', 'backend': 'gemma', 'native_dim': 768, 'batch': 8},
        {'id': 'ovis3b', 'repo': 'ATH-MaaS/Ovis-Omni-Embedding-3B', 'revision': '08547b8479edc10edc9878597438ebd076f74dff', 'subfolder': 'model', 'backend': 'omni', 'native_dim': 2048, 'batch': 4},
        {'id': 'voiceclap_large_v2', 'repo': 'laion/voiceclap-large-v2', 'revision': 'edd7959017dd3f2315d147b629204229a18f6c59', 'backend': 'omni', 'native_dim': 3584, 'batch': 2},
        {'id': 'voiceclap_commercial', 'repo': 'laion/voiceclap-commercial', 'revision': '4e0170bf9a7e29704dd3a26ca92855b3cdd8706c', 'backend': 'commercial', 'native_dim': 768, 'batch': 16},
    ]
    base = Path('/e/data1/datasets/playground/mmlaion/shared/experiments/autoexp/polygon/scaling_law_grid_clap_v3')
    for size in ('XXS', 'XS', 'S', 'M', 'L'):
        nodes = 32 if size == 'L' else 16
        matches = [p / 'checkpoints/epoch_320.pt' for p in base.iterdir()
                   if f'_model=CLAPNaFlex-v3-{size}_' in p.name and '_epochs=320_' in p.name and f'_nodes={nodes}_' in p.name]
        if len(matches) != 1 or not matches[0].is_file():
            raise RuntimeError(f'Missing/ambiguous epoch 320 checkpoint for {size}')
        import json
        config = json.loads((CLAP_CODE / 'src/open_clip/model_configs' / f'CLAPNaFlex-v3-{size}.json').read_text())
        models.append({'id': 'clapv2_' + size.lower(), 'backend': 'clap', 'model_name': 'CLAPNaFlex-v3-' + size,
                       'checkpoint': str(matches[0]), 'checkpoint_bytes': matches[0].stat().st_size,
                       'native_dim': config['embed_dim'], 'nodes_in_original_run': nodes, 'batch': 16})
    study = {'schema': 'frozen-audio-probe-study-v1', 'seed': SEED, 'models': models,
             'clap_code': str(CLAP_CODE), 'clap_benchmark_code': str(CLAP_BENCH),
             'transformers_revision': TRANSFORMERS_REVISION,
             'backbone_policy': 'Frozen audio backbones; inference only; no reference captions or transcripts in inputs',
             'stages': ['S8', 'S9', 'S10'], 'stage_levels': {'S8': 't2', 'S9': 't1', 'S10': 't0'},
             'p3_fraction': .10, 'common_pool_dim': 256, 'common_frame_dim': 64,
             'pool_projection': 'Train-only PCA, retain min(256,native_dim) components; zero-pad smaller models',
             'frame_projection': 'Fixed seeded orthogonal projection; no labels or validation data used',
             'head_hidden': 64, 'head_dropout': .1, 'maximum_parameters_per_head': 50000,
             'probe_epochs_per_stage': 1, 'gemini_probe_epochs': 2, 'whisper_gemini_epochs': 2,
             'probe_batch': 256, 'probe_lr': .001, 'minimum_lr_ratio': .1, 'warmup_ratio': .05,
             'weight_decay': .01,
             'losses': 'Masked normalized Huber for 192 scalars and CPS; cosine+Huber Orange; frame BCE+Dice; event CE',
             'benchmarks': ['emonet', 'emolia-emo', 'emolia-dim', 'crema', 'ravdess'],
             'acted_adapter': {'input': '192 scalar predictions from every compared model', 'hidden': 64,
                               'outer_actor_folds': 5, 'inner_actor_folds': 3, 'epochs': [20, 50], 'lr': [.001, .003],
                               'classes': {'crema': 6, 'ravdess': 8},
                               'parameters': {'crema': 12742, 'ravdess': 12872},
                               'reporting': 'Supervised nested actor-disjoint adaptation; separate from fixed-taxonomy / paper zero-shot'},
             'report_space': 'laion/whisper-base-small-emotion-voice-burst',
             'report_path': 'embedding-probes.html'}
    for model in models:
        model['local_path'] = str(ROOT / 'models' / model['id']) if model.get('repo') else None
    write(ROOT / 'study.json', study)
    print(ROOT / 'study.json')


if __name__ == '__main__':
    main()
