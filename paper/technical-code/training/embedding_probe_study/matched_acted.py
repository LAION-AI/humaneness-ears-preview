"""Identical score MLPs; nested actor-held-out CREMA-D/RAVDESS evaluation."""
import json
import os
import sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
from study_paths import CODE, ROOT, SEED, read, write

sys.path.insert(0, str(CODE))
from calibrate_acted import actor_splits, metrics
from score_public_benchmarks import ACTED_CLASSES


def fit(x, y, train, test, n_classes, config, seed):
    # Scale only within this training fold, including every inner fit.
    mean = x[train].mean(0)
    scale = np.maximum(x[train].std(0), 1e-5)
    device = torch.device('cuda', int(os.environ.get('LOCAL_RANK', 0)))
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(192, 64), nn.GELU(), nn.Linear(64, n_classes)).to(device)
    features = torch.as_tensor((x - mean) / scale, dtype=torch.float32, device=device)
    labels = torch.as_tensor(y, dtype=torch.long, device=device)
    tr = torch.as_tensor(train, device=device)
    te = torch.as_tensor(test, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=.01)
    for _ in range(config['epochs']):
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(model(features[tr]), labels[tr])
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        logits = model(features[te]).cpu().numpy()
    artifact = {'model': {k: v.cpu() for k, v in model.state_dict().items()},
                'mean': mean, 'scale': scale, 'config': config, 'seed': seed}
    return logits, artifact


def run(kind, scores, rows, output):
    path = output / (kind + '_matched_adapter.json')
    if path.exists():
        return read(path)
    cfg = read(ROOT / 'study.json')['acted_adapter']
    classes = list(ACTED_CLASSES[kind])
    x = np.asarray(scores, np.float32)
    if x.shape != (len(rows), 192) or not np.isfinite(x).all():
        raise ValueError('Adapter requires finite aligned 192-score predictions')
    groups = np.asarray([Path(r['audio_name']).stem.split('-')[-1] if kind == 'ravdess'
                         else Path(r['audio_name']).name.split('_')[0] for r in rows])
    if len(set(r['audio_name'] for r in rows)) != len(rows):
        raise ValueError('Repeated benchmark clip')
    y = np.asarray([classes.index(r['emotion'].lower()) for r in rows])
    parameters = (192 + 1) * 64 + (64 + 1) * len(classes)
    assert parameters == cfg['parameters'][kind]
    candidates = [{'lr': lr, 'epochs': epochs} for lr in cfg['lr'] for epochs in cfg['epochs']]
    prediction = np.full(len(y), -1, np.int64)
    fold_ids = np.full(len(y), -1, np.int64)
    folds = []
    for fold, (train, test) in enumerate(actor_splits(y, groups, 5, SEED)):
        inner = actor_splits(y[train], groups[train], 3, SEED + fold + 1)
        trials = []
        for candidate in candidates:
            correct = count = 0
            for j, (inside, held) in enumerate(inner):
                logits, _ = fit(x, y, train[inside], train[held], len(classes), candidate, SEED + fold * 10 + j)
                correct += int((logits.argmax(1) == y[train[held]]).sum())
                count += len(held)
            trials.append({**candidate, 'inner_accuracy': correct / count})
        chosen = max(trials, key=lambda v: (v['inner_accuracy'], -v['epochs'], -v['lr']))
        logits, artifact = fit(x, y, train, test, len(classes), chosen, SEED + 100 + fold)
        prediction[test], fold_ids[test] = logits.argmax(1), fold
        torch.save({**artifact, 'classes': classes, 'train_actors': sorted(set(groups[train])),
                    'held_actors': sorted(set(groups[test]))}, output / (kind + '_adapter_fold' + str(fold) + '.pt'))
        folds.append({'fold': fold, 'train_clips': len(train), 'held_clips': len(test),
                      'train_actors': sorted(set(groups[train])), 'held_actors': sorted(set(groups[test])),
                      'trials': trials, 'selected': chosen,
                      'outer_accuracy': float((prediction[test] == y[test]).mean())})
    assert np.all(prediction >= 0)
    result = {'dataset': kind, 'parameters': parameters, 'classes': classes, 'seed': SEED,
              'protocol': '192 raw predictions -> training-fold StandardScaler -> 64 GELU -> class logits; '
                          'five actor-disjoint outer folds; three inner folds select LR/epochs; full-batch AdamW',
              'comparison': 'Supervised benchmark adaptation; distinct from paper zero-shot',
              'metrics': metrics(y, prediction, groups, classes), 'folds': folds}
    with (output / (kind + '_matched_oof.jsonl')).open('w') as stream:
        for i, row in enumerate(rows):
            stream.write(json.dumps({'audio_name': row['audio_name'], 'actor': str(groups[i]),
                                     'outer_fold': int(fold_ids[i]), 'truth': classes[y[i]],
                                     'prediction': classes[prediction[i]]}) + '\n')
    write(path, result)
    return result
