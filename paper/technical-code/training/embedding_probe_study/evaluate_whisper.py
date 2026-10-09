#!/usr/bin/env python3
"""Evaluate Gemini-tuned Whisper and the original weights on the same Flash holdout."""
import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import WhisperFeatureExtractor
from study_paths import ROOT, CODE, RELEASE, GEMINI, BENCH, read, write
from predict_benchmarks import score_metrics
from matched_acted import run as acted
from train_probes import evaluate

sys.path.insert(0, str(CODE / 'gemini_finetune'))
from data import GeminiDataset, GeminiCollator
sys.path.insert(0, str(CODE))
from layered_multitask_model import LayeredMultiTaskWhisper
from evaluate_public_benchmarks import decode, parquet_samples, emolia_samples


class P3WhisperData:
    """Exactly the same deterministic 2k P3 split selections as the probes."""
    def __init__(self,split):
        from study_data import SourceData
        self.source=SourceData('p3_'+split)

    def __len__(self):return len(self.source)

    def __getitem__(self,index):
        key,wave,meta,targets=self.source[index]
        return {**targets,'key':key,'uid':key,'domain':'gemini','enabled':True,'audio':wave,'cps':None,
                'timbre_valid':bool(targets['timbre_valid']),'identity_valid':bool(targets['identity_valid']),
                'events_frames':list(zip(targets['event_starts'],targets['event_ends'])),
                'event_class_ids':targets['event_classes']}


class HoldoutLoader:
    def __init__(self, model_path, norm):
        self.collator = GeminiCollator(model_path, norm)

    def __call__(self, data):
        return DataLoader(data, batch_size=32, num_workers=4, multiprocessing_context='spawn',
                          collate_fn=self.collator)


def forward(model, batch):
    with torch.autocast('cuda', torch.bfloat16):
        oracle = model(batch['mel'], batch['mel_mask'], batch['event_starts'], batch['event_ends'])
        predicted = model(batch['mel'], batch['mel_mask'], predict_events=True, max_pred_events=32)
    for key in ('predicted_event_starts', 'predicted_event_ends', 'predicted_event_valid'):
        oracle[key] = predicted[key]
    oracle['predicted_event_class'] = predicted['event_class']
    return oracle


def run(size):
    device = torch.device('cuda', int(os.environ.get('LOCAL_RANK', 0)))
    torch.cuda.set_device(device)
    torch.set_num_threads(2)
    classes = read(RELEASE / 'classes.json')['names']
    cfg = read(CODE / 'gemini_finetune/configs' / ('whisper_' + size + '.json'))
    checkpoint = GEMINI / 'training' / ('whisper_' + size) / 'best.pt'
    out = ROOT / 'whisper' / 'gemini' / size
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'COMPLETE.json').exists():
        return
    norm = read(RELEASE / 'training_normalization.json')
    model = LayeredMultiTaskWhisper(RELEASE / size, len(classes), initialize_pretrained=False).to(device)
    holdout = GeminiDataset(GEMINI / 'prepared', 'test')
    datasets={'flash':holdout,'p3_validation':P3WhisperData('validation'),'p3_test':P3WhisperData('test')}
    for phase,path in [('legacy',cfg['checkpoint']),('gemini',checkpoint)]:
        model.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['model'],strict=True)
        for domain,data in datasets.items():
            filename=('legacy_on_flash_test_metrics.json' if phase=='legacy' else 'gemini_test_metrics.json') if domain=='flash' else phase+'_'+domain+'_metrics.json'
            target=out/filename
            if not target.exists():
                metrics=evaluate(model,data,device,torch.ones(len(classes),device=device),True,
                                 HoldoutLoader(RELEASE/size,norm),forward)
                metrics.update(domain=domain,phase=phase,split_scope='Fixed identical held-out audio selection before and after fine-tuning')
                write(target,metrics)
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=False)['model'], strict=True)
    model.eval()
    extractor = WhisperFeatureExtractor.from_pretrained(RELEASE / size, local_files_only=True)
    predictions = {}
    for kind in read(ROOT / 'study.json')['benchmarks']:
        stem = out / (kind + '_predictions')
        if stem.with_suffix('.npz').exists() and stem.with_suffix('.jsonl').exists():
            scores = np.load(stem.with_suffix('.npz'))['scores']
            rows = [json.loads(line) for line in stem.with_suffix('.jsonl').open()]
        else:
            samples = parquet_samples(kind) if kind in ('emonet', 'crema', 'ravdess') else emolia_samples(kind)
            rows, scores = [], []
            with ThreadPoolExecutor(4) as pool, torch.inference_mode():
                for offset in range(0, len(samples), 32):
                    selected = samples[offset:offset + 32]
                    audio = list(pool.map(decode, selected))
                    features = extractor([w for w,_,_ in audio], sampling_rate=16000, padding='longest',
                                         truncation=True, max_length=480000, return_attention_mask=True, return_tensors='np')
                    mel, mask = features['input_features'], features['attention_mask']
                    if mel.shape[-1] % 2:
                        mel = np.pad(mel, ((0,0),(0,0),(0,1)))
                        mask = np.pad(mask, ((0,0),(0,1)))
                    with torch.autocast('cuda', torch.bfloat16):
                        result = model(torch.from_numpy(mel).to(device), torch.from_numpy(mask).to(device))
                    scores.append(result['scores'].float().cpu().numpy() * np.asarray(norm['score_std']) + np.asarray(norm['score_mean']))
                    rows.extend({**s.metadata, 'key': s.key, 'duration_s': d, 'truncated_30s': t}
                                for s,(_,d,t) in zip(selected,audio))
            scores = np.concatenate(scores).astype(np.float32)
            np.savez_compressed(stem.with_suffix('.npz'), scores=scores)
            stem.with_suffix('.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
        predictions[kind] = scores, rows
        if kind in ('crema', 'ravdess'):
            acted(kind, scores, rows, out)
    score_metrics(predictions, 'whisper_' + size, out)
    write(out / 'COMPLETE.json', {'checkpoint': str(checkpoint), 'test_clips': len(holdout),
                                'p3_validation_clips':len(datasets['p3_validation']),'p3_test_clips':len(datasets['p3_test']),
                                'benchmarks': list(predictions)})


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--model', choices=['base','small','all'], required=True)
    args=ap.parse_args()
    if args.model!='all':
        run(args.model)
    else:
        rank=int(os.environ.get('LOCAL_RANK',0))
        if rank<2:
            run(('base','small')[rank])
        else:
            size=('base','small')[rank-2];stage='S4' if size=='base' else 'S3'
            out=ROOT/'whisper/legacy'/size;out.mkdir(parents=True,exist_ok=True)
            for kind in ('crema','ravdess'):
                stem=BENCH/'predictions'/(kind+'_'+size+'_'+stage)
                acted(kind,np.load(stem.with_suffix('.npz'))['scores'],
                      [json.loads(line) for line in stem.with_suffix('.jsonl').open()],out)
