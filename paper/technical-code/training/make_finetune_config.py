#!/usr/bin/env python3
"""Relocate recorded fine-tuning paths without changing hyperparameters."""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--model', choices=['base', 'small'], required=True)
parser.add_argument('--data-root', type=Path, required=True)
parser.add_argument('--initial-checkpoint', type=Path, required=True)
parser.add_argument('--output-dir', type=Path, required=True)
parser.add_argument('--config', type=Path, required=True)
args = parser.parse_args()
training = Path(__file__).resolve().parent
release = training.parent
cfg = json.loads((training / 'configs' / (args.model + '_recorded_run.json')).read_text())
cfg.update(checkpoint=str(args.initial_checkpoint.resolve()), model_config=str((release / args.model).resolve()),
           data_root=str(args.data_root.resolve()), output=str(args.output_dir.resolve()))
for key in ['world', 'updates', 'train_clips', 'validation_clips', 'event_class_counts',
            'trainable_encoder_parameters', 'trainable_total_parameters', 'checkpoint_sha256', 'selection']:
    cfg.pop(key, None)
if not (args.data_root / 'TRAINING_READY.json').exists():
    raise RuntimeError('Provide the complete prepared target/sidecar dataset, including TRAINING_READY.json')
if not args.initial_checkpoint.exists():
    raise FileNotFoundError(args.initial_checkpoint)
args.config.parent.mkdir(parents=True, exist_ok=True)
args.config.write_text(json.dumps(cfg, indent=2) + '\n')
print(args.config)
