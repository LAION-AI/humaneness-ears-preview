#!/usr/bin/env python3
"""Encoder-only Whisper with layer-mean residual experts and event-local classes."""
from __future__ import annotations

from pathlib import Path

import torch
from safetensors import safe_open
from torch import nn
from torch.nn import functional as F
from transformers import WhisperConfig
from transformers.models.whisper.modeling_whisper import WhisperEncoder


SCORE_FAMILIES={
    'emotion':(0,40,64),
    'voicenet':(40,97,32),
    'genuineness_blend_quality':(97,100,32),
    'empathic_plus':(100,119,64),
    'audiobox':(119,123,32),
    'dnsmos':(123,130,32),
    'burst_count':(130,131,32),
    'voiceclap_attributes':(131,192,64),
}


class LayerMixture(nn.Module):
    """Project each layer's masked mean, then learn a sample-specific layer mix."""
    def __init__(self, dim: int, width: int, layers: int):
        super().__init__()
        self.project=nn.Sequential(nn.LayerNorm(dim),nn.Linear(dim,width),
                                   nn.GELU(),nn.Linear(width,width))
        self.gate=nn.Linear(width,1)
        self.layer_bias=nn.Parameter(torch.zeros(layers))
        self.out_norm=nn.LayerNorm(width)

    def forward(self, means: torch.Tensor) -> torch.Tensor:
        # means: [batch, encoder layers, hidden width]
        z=self.project(means)
        weights=(self.gate(z).squeeze(-1)+self.layer_bias).softmax(dim=1)
        return self.out_norm((z*weights.unsqueeze(-1)).sum(dim=1))


class LayeredMultiTaskWhisper(nn.Module):
    def __init__(self, model_path: str | Path, n_event_classes: int,
                 *, initialize_pretrained: bool = True):
        super().__init__()
        model_path=Path(model_path)
        config=WhisperConfig.from_pretrained(model_path,local_files_only=True)
        self.encoder=WhisperEncoder(config)
        if initialize_pretrained:
            with safe_open(model_path/'model.safetensors',framework='pt',device='cpu') as file:
                prefix='model.encoder.'
                pretrained={name[len(prefix):]:file.get_tensor(name) for name in file.keys()
                            if name.startswith(prefix)}
            self.encoder.load_state_dict(pretrained,strict=True)
        self.hidden_dim=config.d_model
        self.num_layers=config.encoder_layers
        self.n_event_classes=n_event_classes
        pooled_dim=config.d_model*4

        def clip_head(inputs: int, outputs: int):
            return nn.Sequential(nn.LayerNorm(inputs),nn.Linear(inputs,512),
                                 nn.GELU(),nn.Linear(512,outputs))

        # A direct final-layer route plus learned residuals from every layer.
        self.score_head=clip_head(pooled_dim,192)
        self.score_mix=nn.ModuleDict()
        self.score_delta=nn.ModuleDict()
        for family,(start,stop,width) in SCORE_FAMILIES.items():
            self.score_mix[family]=LayerMixture(config.d_model,width,self.num_layers)
            self.score_delta[family]=nn.Linear(width,stop-start)
            nn.init.zeros_(self.score_delta[family].weight)
            nn.init.zeros_(self.score_delta[family].bias)

        self.timbre_mix=LayerMixture(config.d_model,128,self.num_layers)
        self.identity_mix=LayerMixture(config.d_model,128,self.num_layers)
        self.timbre_head=clip_head(pooled_dim+128,128)
        self.identity_head=clip_head(pooled_dim+128,250)
        self.cps_mix=LayerMixture(config.d_model,32,self.num_layers)
        self.cps_head=clip_head(pooled_dim+32,1)

        self.frame_head=nn.Sequential(nn.LayerNorm(config.d_model),
                                      nn.Linear(config.d_model,config.d_model//2),
                                      nn.GELU(),
                                      nn.Conv1d(config.d_model//2,config.d_model//2,7,padding=3),
                                      nn.GELU(),nn.Conv1d(config.d_model//2,1,1))
        # Independent onset/duration proposals keep overlapping source events separate.
        self.proposal_head=nn.Conv1d(config.d_model//2,2,3,padding=1)
        self.event_mix=LayerMixture(config.d_model,64,self.num_layers)
        self.event_head=nn.Sequential(nn.LayerNorm(config.d_model+64),
                                      nn.Linear(config.d_model+64,256),nn.GELU(),
                                      nn.Linear(256,n_event_classes))

    @staticmethod
    def _event_means(h: torch.Tensor, starts: torch.Tensor,
                     ends: torch.Tensor) -> torch.Tensor:
        n=h.shape[1]
        starts=starts.clamp(0,n)
        ends=ends.clamp(0,n)
        prefix=F.pad(h.float().cumsum(dim=1),(0,0,1,0))
        width=h.shape[-1]
        left=prefix.gather(1,starts.unsqueeze(-1).expand(-1,-1,width))
        right=prefix.gather(1,ends.unsqueeze(-1).expand(-1,-1,width))
        return (right-left)/(ends-starts).clamp_min(1).unsqueeze(-1)

    def forward(self, mel: torch.Tensor, mel_mask: torch.Tensor,
                event_starts: torch.Tensor | None = None,
                event_ends: torch.Tensor | None = None,
                *, predict_events: bool = False,
                event_threshold: float = .5,
                max_pred_events: int = 10) -> dict[str,torch.Tensor]:
        enc=self.encoder
        h=F.gelu(enc.conv1(mel))
        h=F.gelu(enc.conv2(h)).transpose(1,2)
        n=h.shape[1]
        h=h+enc.embed_positions(torch.arange(n,device=h.device))
        valid=torch.arange(n,device=h.device)[None,:]<((mel_mask.sum(1)+1)//2)[:,None]
        attention=torch.zeros((len(h),1,1,n),device=h.device,dtype=h.dtype)
        attention.masked_fill_(~valid[:,None,None,:],torch.finfo(h.dtype).min)
        mask=valid.unsqueeze(-1)
        count=mask.sum(1).clamp_min(1)
        layer_means=[]
        event_layers=[]
        layer_frames=[]
        if (event_starts is None)!=(event_ends is None):
            raise ValueError('event starts and ends must be provided together')
        if predict_events and event_starts is not None:
            raise ValueError('Use ground-truth event spans or predicted spans, not both')
        for layer in enc.layers:
            h=layer(h,attention)
            layer_means.append((h.float()*mask).sum(1)/count)
            if event_starts is not None:
                event_layers.append(self._event_means(h,event_starts,event_ends))
            elif predict_events:
                layer_frames.append(h)
        h=enc.layer_norm(h).float()
        mean=(h*mask).sum(1)/count
        var=((h-mean[:,None,:]).square()*mask).sum(1)/count
        minimum=h.masked_fill(~mask,torch.inf).amin(1)
        maximum=h.masked_fill(~mask,-torch.inf).amax(1)
        pooled=torch.cat((mean,minimum,maximum,var.clamp_min(1e-8).sqrt()),dim=-1)
        means=torch.stack(layer_means,dim=1)

        deltas=torch.cat([self.score_delta[family](self.score_mix[family](means))
                          for family in SCORE_FAMILIES],dim=-1)
        score=self.score_head(pooled)+deltas
        timbre=F.normalize(self.timbre_head(torch.cat((pooled,self.timbre_mix(means)),dim=-1)).float(),dim=-1)
        identity=F.normalize(self.identity_head(torch.cat((pooled,self.identity_mix(means)),dim=-1)).float(),dim=-1)
        cps=self.cps_head(torch.cat((pooled,self.cps_mix(means)),dim=-1)).squeeze(-1)
        x=self.frame_head[0](h)
        x=self.frame_head[1](x)
        x=self.frame_head[2](x).transpose(1,2)
        frame=self.frame_head[3:](x).squeeze(1)
        proposal=self.proposal_head(x)
        onset=proposal[:,0,:]
        log_duration=proposal[:,1,:]
        output={'scores':score,'frame':frame,'timbre':timbre,
                'identity':identity,'cps':cps,
                'onset':onset,'log_duration':log_duration}
        if predict_events:
            if not 0<event_threshold<1 or max_pred_events<1:
                raise ValueError('Invalid event threshold or maximum event count')
            spans=[]
            with torch.no_grad():
                probabilities=onset.float().sigmoid()
                for i in range(len(h)):
                    active=(probabilities[i]>=event_threshold)&valid[i]
                    padded=F.pad(probabilities[i],(1,1),value=-1.)
                    peaks=active&(probabilities[i]>=padded[:-2])&(
                        probabilities[i]>padded[2:])
                    ranked=torch.nonzero(peaks).flatten().tolist()
                    ranked.sort(key=lambda start:float(probabilities[i,start]),reverse=True)
                    chosen=[]
                    for start in ranked:
                        if any(abs(start-old)<3 for old in chosen):
                            continue
                        chosen.append(start)
                        if len(chosen)>=max_pred_events:
                            break
                    row=[]
                    for start in chosen:
                        duration=int(round(float(log_duration[i,start].float().clamp(0,8).expm1())))
                        end=min(int(valid[i].sum()),start+max(1,duration))
                        row.append((start,end))
                    spans.append(sorted(row))
            width=max(1,max(len(row) for row in spans))
            event_starts=torch.zeros((len(h),width),device=h.device,dtype=torch.int64)
            event_ends=torch.zeros_like(event_starts)
            event_valid=torch.zeros_like(event_starts,dtype=torch.bool)
            for i,row in enumerate(spans):
                for j,(start,end) in enumerate(row):
                    event_starts[i,j]=start
                    event_ends[i,j]=end
                    event_valid[i,j]=True
            event_layers=[self._event_means(layer,event_starts,event_ends)
                          for layer in layer_frames]
            output.update(predicted_event_starts=event_starts,
                          predicted_event_ends=event_ends,
                          predicted_event_valid=event_valid)
        if event_starts is not None:
            local=self._event_means(h,event_starts,event_ends)
            per_layer=torch.stack(event_layers,dim=2)
            b,e,l,d=per_layer.shape
            mixed=self.event_mix(per_layer.reshape(b*e,l,d)).reshape(b,e,-1)
            output['event_class']=self.event_head(torch.cat((local,mixed),dim=-1))
        return output
