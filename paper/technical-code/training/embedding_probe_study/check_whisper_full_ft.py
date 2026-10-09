#!/usr/bin/env python3
"""Verify full encoder/head updates using a real Flash/teacher canary clip."""
import sys
import torch
from study_paths import CODE, GEMINI, ROOT, read, write
sys.path.insert(0,str(CODE/'gemini_finetune'))
from data import GeminiDataset, GeminiCollator
sys.path.insert(0,str(CODE))
from layered_multitask_model import LayeredMultiTaskWhisper
from train_layered_curriculum import gpu_batch,loss_terms


def main():
    torch.set_num_threads(2)
    root=GEMINI/'parser_canary_complete'
    dataset=GeminiDataset(root,'train',require_complete=False)
    classes=read(root/'gemini_burst_mapping.json')['checkpoint_classes']
    results={}
    for size in ('base','small'):
        cfg=read(CODE/'gemini_finetune/configs'/('whisper_'+size+'.json'))
        core=LayeredMultiTaskWhisper(cfg['model_config'],len(classes),initialize_pretrained=False)
        core.load_state_dict(torch.load(cfg['checkpoint'],map_location='cpu',weights_only=False)['model'],strict=True)
        core.to('cuda').train().requires_grad_(True)
        assert all(p.requires_grad for p in core.parameters())
        batch=gpu_batch(GeminiCollator(cfg['model_config'],dataset.norm)([dataset[0]]),torch.device('cuda'))
        before=core.encoder.conv1.weight.detach().clone()
        positions=core.encoder.embed_positions.weight.detach().clone()
        optimizer=torch.optim.AdamW([
            {'params':list(core.encoder.parameters()),'lr':cfg['encoder_lr']},
            {'params':[p for n,p in core.named_parameters() if not n.startswith('encoder.')],'lr':cfg['head_lr']}],weight_decay=.01)
        with torch.autocast('cuda',torch.bfloat16):
            out=core(batch['mel'],batch['mel_mask'],batch['event_starts'],batch['event_ends'])
            loss=sum(loss_terms(out,batch,torch.ones(len(classes),device='cuda')).values())
        loss.backward()
        assert torch.isfinite(loss) and all(p.grad is None or torch.isfinite(p.grad).all() for p in core.parameters())
        optimizer.step()
        assert not torch.equal(before,core.encoder.conv1.weight)
        assert not torch.equal(positions,core.encoder.embed_positions.weight)
        results[size]={'all_parameters_trainable':True,'encoder_updated':True,'positional_embedding_updated':True,
                       'finite_loss_and_gradients':True,'scalar_outputs':out['scores'].shape[-1],
                       'encoder_parameters':sum(p.numel() for p in core.encoder.parameters()),'canary_clips':1}
        del core,batch,out,loss,optimizer;torch.cuda.empty_cache()
    write(ROOT/'WHISPER_FULL_FT_CONTRACT.json',{'passed':True,'scope':'One real canary optimizer step; not a training result','models':results})
    print('WHISPER_FULL_FT_CONTRACT_PASS',flush=True)


if __name__=='__main__':main()
