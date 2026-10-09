#!/usr/bin/env python3
"""Bounded distributed full-FT and real P3 evaluation compatibility check."""
import os
import sys
import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Subset
from study_paths import CODE, GEMINI, ROOT, RELEASE, read, write

sys.path.insert(0, str(CODE / 'gemini_finetune'))
from data import GeminiDataset, GeminiCollator
sys.path.insert(0, str(CODE))
from layered_multitask_model import LayeredMultiTaskWhisper
from train_layered_curriculum import gpu_batch, loss_terms


def main():
    rank = int(os.environ['LOCAL_RANK'])
    device = torch.device('cuda', rank)
    torch.cuda.set_device(device)
    torch.set_num_threads(2)
    torch.distributed.init_process_group('nccl')
    canary = GeminiDataset(GEMINI / 'parser_canary_complete', 'train', require_complete=False)
    classes = read(RELEASE / 'classes.json')['names']
    cfg = read(CODE / 'gemini_finetune/configs/whisper_base.json')
    core = LayeredMultiTaskWhisper(cfg['model_config'], len(classes), initialize_pretrained=False)
    core.load_state_dict(torch.load(cfg['checkpoint'], map_location='cpu', weights_only=False)['model'], strict=True)
    core.to(device).train().requires_grad_(True)
    model = DistributedDataParallel(core, device_ids=[rank], find_unused_parameters=False)
    optimizer = torch.optim.AdamW([
        {'params': list(core.encoder.parameters()), 'lr': cfg['encoder_lr']},
        {'params': [p for n,p in core.named_parameters() if not n.startswith('encoder.')], 'lr': cfg['head_lr']}])
    data = Subset(canary, [(rank + step * 4) % len(canary) for step in range(2)])
    loader = DataLoader(data, batch_size=1, num_workers=1, multiprocessing_context='spawn',
                        collate_fn=GeminiCollator(cfg['model_config'], canary.norm))
    weights = torch.ones(len(classes), device=device)
    for raw in loader:
        batch = gpu_batch(raw, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', torch.bfloat16):
            output = model(batch['mel'], batch['mel_mask'], batch['event_starts'], batch['event_ends'])
            loss = sum(loss_terms(output, batch, weights).values())
        loss.backward()
        assert torch.isfinite(loss) and all(p.grad is None or torch.isfinite(p.grad).all() for p in core.parameters())
        optimizer.step()
    torch.distributed.barrier()
    del model, core, optimizer, output, loss, batch, loader
    torch.cuda.empty_cache()
    # Check the new P3 wrapper, spawned audio loader and predicted event path.
    local_result = {'rank': rank, 'ddp_iterations': 2, 'finite_loss_and_gradients': True}
    if rank < 2:
        from evaluate_whisper import P3WhisperData, HoldoutLoader, forward
        from train_probes import evaluate
        size = ('base', 'small')[rank]
        cfg = read(CODE / 'gemini_finetune/configs' / ('whisper_' + size + '.json'))
        core = LayeredMultiTaskWhisper(cfg['model_config'], len(classes), initialize_pretrained=False).to(device)
        core.load_state_dict(torch.load(cfg['checkpoint'], map_location='cpu', weights_only=False)['model'], strict=True)
        checked = {}
        for split in ('validation', 'test'):
            metrics = evaluate(core, Subset(P3WhisperData(split), [0]), device, weights, True,
                               HoldoutLoader(RELEASE / size, canary.norm), forward)
            assert metrics['clips'] == 1 and len(metrics['per_score']) == 192
            checked[split] = {'clips': 1, 'targets': 192, 'predicted_and_oracle_burst_paths': True}
        local_result.update(model=size, p3_checks=checked)
    results = [None] * 4
    torch.distributed.all_gather_object(results, local_result)
    if rank == 0:
        write(ROOT / 'DISTRIBUTED_PIPELINE_CONTRACT.json', {
            'passed': True, 'scope': 'Two real canary updates per rank and one P3 clip per split/model; software check, not an evaluation result',
            'ranks': results})
        print('DISTRIBUTED_PIPELINE_CONTRACT_PASS', flush=True)
    torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
