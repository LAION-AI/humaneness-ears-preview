#!/usr/bin/env python3
"""Pin the best validation checkpoints and create independent training configs."""
import json
from common import (HERE, MOSS_REVISION, RELEASE, ROOT, UPSTREAM_REVISION, read_json, write_json)


def main():
    root = ROOT / 'prepared'
    for model in ('base', 'small'):
        candidates = [read_json(p) for p in (RELEASE / model).glob('S*/selection.json')]
        selected = min(candidates, key=lambda row: row['selection_loss'])
        stage = selected['stage']
        cfg = {'model': model, 'stage': stage, 'selection': selected,
            'checkpoint': str(RELEASE / model / stage / 'checkpoint.pt'),
            'checkpoint_sha256': selected['checkpoint_sha256'],
            'model_config': str(RELEASE / model), 'data_root': str(root),
            'output': str(ROOT / 'training' / ('whisper_' + model)),
            'epochs': 2, 'encoder_lr': 1e-5 if model == 'base' else 5e-6,
            'head_lr': 1e-4, 'minimum_lr_ratio': .10, 'warmup_ratio': .05,
            'batch_per_gpu': 32 if model == 'base' else 16, 'accumulation': 1 if model == 'base' else 2,
            'workers': 4, 'weight_decay': .01, 'seed': 20261005,
            'normalization': 'checkpoint_normalization.json',
            'optimizer_policy': 'fresh fine-tune AdamW/cosine; best curriculum weights restored strictly',
            'loss_policy': 'existing masked Huber192+CPS, Orange cosine+Huber, frames BCE+Dice, onset BCE, duration Huber, event CE',
            'save_policy': 'best and latest epoch checkpoint, optimizer/scheduler and per-rank RNG; epoch-boundary resume'}
        write_json(HERE / 'configs' / f'whisper_{model}.json', cfg)
    for task in ('caption', 'director_caption', 'burst_caption'):
        cfg = {'task': task, 'model_id': 'OpenMOSS-Team/MOSS-Audio-8B-Instruct',
            'model_revision': MOSS_REVISION, 'upstream_revision': UPSTREAM_REVISION,
            'model_dir': str(ROOT / 'models/MOSS-Audio-8B-Instruct'), 'data_root': str(root),
            'output': str(ROOT / 'training' / ('moss_' + task)),
            'epochs': 2, 'learning_rate': 1e-4, 'warmup_ratio': .05,
            'weight_decay': .01, 'lora_rank': 32, 'lora_alpha': 64, 'lora_dropout': .05,
            'microbatch_per_gpu': 1, 'accumulation': 16, 'workers': 4,
            'maximum_tokens': 8192, 'attention_backend': 'sdpa',
            'audio_time_markers': task == 'burst_caption',
            'gradient_checkpointing': False, 'seed': 20261005,
            'frozen': ['audio_encoder', 'audio_adapter', 'deepstack_audio_merger_list', 'base_language_model'],
            'loss': 'assistant caption text only; system/user/audio labels -100; no thinking trace',
            'burst_input': 'full original audio + start and end in seconds; no event label or reference caption in input',
            'save_policy': 'best adapter plus full Trainer resume states at each epoch'}
        write_json(HERE / 'configs' / f'moss_{task}.json', cfg)
    print(json.dumps({'configs': str(HERE / 'configs'), 'whisper_checkpoints': ['Base S4', 'Small S3'],
        'moss_tasks': ['caption', 'director_caption', 'burst_caption'], 'training_started': False}))


if __name__ == '__main__':
    main()
