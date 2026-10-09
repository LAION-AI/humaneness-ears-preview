"""Frozen native audio encoders, pooled embeddings and aligned temporal features."""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from study_paths import ROOT, CLAP_CODE, SEED


class FrozenEncoder:
    def __init__(self, spec, device):
        self.spec, self.device = spec, device
        self.kind = spec['backend']
        self.frame_projection = None
        local = Path(spec.get('local_path') or '')
        if self.kind == 'clap':
            sys.path.insert(0, str(CLAP_CODE / 'src'))
            import open_clip
            from open_clip.audio.naflex_audio import AudioNaFlexCfg, AudioNaFlexPatchify
            self.model = open_clip.create_model(spec['model_name'], output_dict=True)
            ck = torch.load(spec['checkpoint'], map_location='cpu', weights_only=False)
            state = {k.removeprefix('module.'): v for k, v in ck.get('state_dict', ck).items()}
            self.model.load_state_dict(state, strict=True)
            del ck, state
            self.audio_cfg = AudioNaFlexCfg.from_clip_audio_cfg(self.model.audio.cfg)
            self.patchify = AudioNaFlexPatchify(self.audio_cfg, random_crop=False)
            self.tokens = None
            self.model.audio.encoder.vit.norm.register_forward_hook(self._capture)
        elif self.kind == 'commercial':
            from transformers import AutoModel
            self.model = AutoModel.from_pretrained(local, local_files_only=True, trust_remote_code=True)
            self.tokens = None
            self.model.audio_encoder.register_forward_hook(self._capture)
        elif self.kind == 'gemma':
            from transformers import AutoModel, AutoProcessor
            self.processor = AutoProcessor.from_pretrained(local, local_files_only=True)
            self.model = AutoModel.from_pretrained(local, local_files_only=True, torch_dtype=torch.bfloat16,
                                                  vision_config=None, attn_implementation='sdpa')
        elif self.kind == 'omni':
            from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerConfig, Qwen2_5OmniThinkerForConditionalGeneration
            import json
            local = local / spec.get('subfolder', '')
            config = json.loads((local / 'config.json').read_text())
            thinker = config.get('thinker_config', config)
            self.processor = Qwen2_5OmniProcessor.from_pretrained(local, local_files_only=True)
            self.model, loading = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
                local, config=Qwen2_5OmniThinkerConfig(**thinker), local_files_only=True,
                torch_dtype=torch.bfloat16, attn_implementation='sdpa', output_loading_info=True)
            missing = [k for k in loading.get('missing_keys', []) if not k.startswith(('visual.', 'lm_head.'))]
            if missing:
                raise RuntimeError('Missing trained audio/Thinker weights: ' + str(missing[:8]))
            self.model.lm_head = nn.Identity()  # No vocabulary logits or generation.
            self.tokens = None
            self.model.model.register_forward_hook(self._capture)
        else:
            raise ValueError(self.kind)
        self.model.to(device).eval()
        self.model.requires_grad_(False)

    def _capture(self, _module, _args, result):
        self.tokens = result.last_hidden_state if hasattr(result, 'last_hidden_state') else result
        if isinstance(self.tokens, tuple):
            self.tokens = self.tokens[0]

    def _compress_frames(self, tensor):
        width = tensor.shape[-1]
        if self.frame_projection is None:
            generator = torch.Generator(device='cpu').manual_seed(SEED + width)
            matrix = torch.randn(width, min(64, width), generator=generator)
            self.frame_projection = torch.linalg.qr(matrix, mode='reduced').Q.to(self.device)
        result = tensor.float() @ self.frame_projection
        return F.pad(result, (0, 64 - result.shape[-1]))

    @torch.inference_mode()
    def encode(self, waves):
        durations = [len(w) / 16000 for w in waves]
        frames, times = [], []
        with torch.autocast('cuda', dtype=torch.bfloat16):
            if self.kind == 'clap':
                patches = [self.patchify((torch.from_numpy(w).float()[None], 16000)) for w in waves]
                width = max(len(p['patches']) for p in patches)
                inputs = {}
                for name in ('patches', 'patch_coord', 'patch_valid'):
                    shape = (len(waves), width, *patches[0][name].shape[1:])
                    inputs[name] = torch.zeros(shape, dtype=patches[0][name].dtype, device=self.device)
                    for i, p in enumerate(patches):
                        inputs[name][i, :len(p[name])] = p[name].to(self.device)
                pooled = self.model.encode_audio(inputs, normalize=True)
                if self.tokens is None or self.tokens.ndim != 3 or self.tokens.shape[1] < width:
                    raise RuntimeError('NaFlex temporal hook did not expose the patch sequence')
                tokens = self.tokens[:, -width:]
                dt = self.audio_cfg.patch_time * self.audio_cfg.hop_size / self.audio_cfg.sample_rate
                for i, p in enumerate(patches):
                    coords = inputs['patch_coord'][i, :len(p['patches']), 1]
                    valid = inputs['patch_valid'][i, :len(p['patches'])].bool()
                    columns = coords[valid].unique(sorted=True)
                    frame = torch.stack([tokens[i, :len(coords)][valid & (coords == t)].mean(0) for t in columns])
                    frames.append(self._compress_frames(frame))
                    times.append((columns.float() + .5) * dt)
            elif self.kind == 'commercial':
                wave = torch.zeros(len(waves), 480000, device=self.device)
                for i, w in enumerate(waves):
                    wave[i, :len(w)] = torch.from_numpy(w).to(self.device)
                pooled = self.model.encode_waveform(wave)
                if self.tokens is None:
                    raise RuntimeError('Commercial audio encoder hook failed')
                for i, duration in enumerate(durations):
                    count = min(self.tokens.shape[1], int(np.ceil(duration * 50)))
                    frames.append(self._compress_frames(self.tokens[i, :count]))
                    times.append((torch.arange(count, device=self.device) + .5) / 50)
            else:
                if self.kind == 'gemma':
                    inputs = self.processor(audio=waves, padding=True, return_tensors='pt')
                else:
                    conversation = [{'role': 'user', 'content': [{'type': 'audio', 'audio': 'unused'}]}]
                    text = self.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
                    inputs = self.processor(text=[text] * len(waves), audio=waves, padding=True,
                                            sampling_rate=16000, return_tensors='pt')
                inputs = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in inputs.items()}
                output = self.model(**inputs, use_cache=False) if self.kind == 'omni' else self.model(**inputs)
                hidden = output.last_hidden_state if self.kind == 'gemma' else self.tokens
                if hidden is None:
                    raise RuntimeError('Native hidden state extraction failed')
                mask = inputs['attention_mask'].bool()
                if self.kind == 'gemma':
                    pooled = (hidden.float() * mask[:, :, None]).sum(1) / mask.sum(1)[:, None].clamp_min(1)
                else:
                    last = (mask * torch.arange(mask.shape[1], device=self.device)[None]).max(1).values
                    pooled = hidden[torch.arange(len(waves), device=self.device), last]
                config = self.model.config
                token = getattr(config, 'audio_token_id', None)
                if token is None:
                    token = getattr(config, 'audio_token_index', None)
                if token is None:
                    raise RuntimeError('Native audio token ID is unavailable')
                for i, duration in enumerate(durations):
                    audio_mask = (inputs['input_ids'][i] == token) & mask[i]
                    frame = hidden[i, audio_mask]
                    if not len(frame):
                        raise RuntimeError('No aligned audio-token features')
                    frames.append(self._compress_frames(frame))
                    # One contiguous complete audio input. Native token count
                    # defines the interval grid; no invented frame resolution.
                    times.append((torch.arange(len(frame), device=self.device) + .5) * duration / len(frame))
                pooled = F.normalize(pooled.float(), dim=-1)
        if pooled.shape != (len(waves), self.spec['native_dim']):
            raise RuntimeError('Native pooled embedding dimension mismatch')
        outputs = []
        for i, duration in enumerate(durations):
            embedding = pooled[i].float().cpu().numpy()
            frame = frames[i].float().cpu().numpy()
            time = times[i].float().cpu().numpy()
            keep = time < duration
            frame, time = frame[keep], time[keep]
            if not len(frame) or not np.isfinite(embedding).all() or not np.isfinite(frame).all():
                raise RuntimeError('Empty or non-finite frozen features')
            outputs.append({'embedding': embedding.astype(np.float16), 'frame_features': frame.astype(np.float32),
                            'frame_times_s': time.astype(np.float32), 'duration_s': np.float32(duration)})
        self.tokens = None
        return outputs
