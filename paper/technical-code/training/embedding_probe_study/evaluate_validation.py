#!/usr/bin/env python3
"""Evaluate saved best checkpoints on their existing selection-validation split."""
import os
import torch

from study_paths import ROOT, RELEASE, GEMINI, CODE, read, write
from cache_dataset import FeatureData
from probe_model import Probe
from train_probes import evaluate
from evaluate_whisper import LayeredMultiTaskWhisper, GeminiDataset, HoldoutLoader, forward


def run_probe(model_id, variant, phase, device, classes):
    out = ROOT / 'probes' / phase / model_id / variant
    target = out / 'validation_metrics.json'
    if target.exists():
        return
    saved = torch.load(out / 'best.pt', map_location='cpu', weights_only=False)
    domains = ['gemini'] if phase == 'gemini' else ['ladder', 'p3_train', 'p3_validation', 'p3_test']
    data = FeatureData(model_id, domains, 'validation', phase)
    data.projection = saved['pca']
    model = Probe(len(classes), variant == 'linear').to(device)
    model.load_state_dict(saved['model'], strict=True)
    del saved
    metrics = evaluate(model, data, device, torch.ones(len(classes), device=device), True)
    metrics.update(split='validation', phase=phase, domain='Flash' if phase == 'gemini' else 'Ladder/P3',
                   split_scope='Existing checkpoint-selection validation split; not a fresh independent test')
    write(target, metrics)
    print('VALIDATED', phase, model_id, variant, 'clips', metrics['clips'], flush=True)
    del model, data
    torch.cuda.empty_cache()


def run_whisper(size, phase, device, classes):
    out = ROOT / 'whisper' / phase / size
    target = out / 'validation_metrics.json'
    if target.exists():
        return
    cfg = read(CODE / 'gemini_finetune/configs' / ('whisper_' + size + '.json'))
    checkpoint = cfg['checkpoint'] if phase == 'legacy' else GEMINI / 'training' / ('whisper_' + size) / 'best.pt'
    norm = read(RELEASE / 'training_normalization.json')
    model = LayeredMultiTaskWhisper(RELEASE / size, len(classes), initialize_pretrained=False).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=False)['model'], strict=True)
    data = GeminiDataset(GEMINI / 'prepared', 'validation')
    metrics = evaluate(model, data, device, torch.ones(len(classes), device=device), True,
                       HoldoutLoader(RELEASE / size, norm), forward)
    metrics.update(split='validation', phase=phase, domain='Flash',
                   split_scope='Same Flash validation audio before/after; used to select tuned checkpoint')
    write(target, metrics)
    print('VALIDATED Whisper', size, phase, 'clips', metrics['clips'], flush=True)
    del model, data
    torch.cuda.empty_cache()


def main():
    rank = int(os.environ.get('LOCAL_RANK', 0)); device = torch.device('cuda', rank)
    torch.cuda.set_device(device); torch.set_num_threads(2)
    classes = read(RELEASE / 'classes.json')['names']
    size, phase = [('base', 'legacy'), ('base', 'gemini'), ('small', 'legacy'), ('small', 'gemini')][rank]
    run_whisper(size, phase, device, classes)
    jobs = [(m['id'], v, p) for p in ('legacy', 'gemini')
            for m in read(ROOT / 'study.json')['models'] for v in ('linear', 'mlp')]
    for index, job in enumerate(jobs):
        if index % 4 == rank:
            run_probe(*job, device, classes)
    write(ROOT / 'consolidated_benchmark' / ('VALIDATION_COMPLETE-rank' + str(rank) + '.json'),
          {'rank': rank, 'probe_configurations': len(jobs[rank::4]), 'whisper_configurations': 1})


if __name__ == '__main__':
    main()
