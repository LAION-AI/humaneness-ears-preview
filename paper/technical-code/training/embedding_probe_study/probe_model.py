"""Matched small heads on cached frozen pooled and temporal features."""
import torch
from torch import nn
from torch.nn import functional as F


class ScalarHeads(nn.Module):
    def __init__(self, linear=False):
        super().__init__()
        self.linear = linear
        if linear:
            self.output = nn.Linear(256, 193)
        else:
            self.first = nn.Parameter(torch.empty(193, 256, 64))
            self.bias = nn.Parameter(torch.zeros(193, 64))
            self.second = nn.Parameter(torch.empty(193, 64))
            self.final_bias = nn.Parameter(torch.zeros(193))
            nn.init.normal_(self.first, std=.02)
            nn.init.normal_(self.second, std=.02)
            self.dropout = nn.Dropout(.1)

    def forward(self, features):
        if self.linear:
            return self.output(features)
        hidden = self.dropout(F.gelu(torch.einsum('bi,sih->bsh', features, self.first) + self.bias))
        return (hidden * self.second[None]).sum(-1) + self.final_bias


class Probe(nn.Module):
    def __init__(self, n_classes, linear=False):
        super().__init__()
        self.scalars = ScalarHeads(linear)
        def head(d, out):
            if linear:
                return nn.Linear(d, out) if (d + 1) * out <= 50000 else nn.Sequential(nn.Linear(d, 64), nn.Linear(64, out))
            return nn.Sequential(nn.Linear(d, 64), nn.GELU(), nn.Dropout(.1), nn.Linear(64, out))
        self.timbre_head, self.identity_head = head(256, 128), head(256, 250)
        self.frame_head = head(64, 3)
        self.event_head = head(64, n_classes)
        for name, module in [('timbre', self.timbre_head), ('identity', self.identity_head), ('frame', self.frame_head), ('event', self.event_head)]:
            if sum(p.numel() for p in module.parameters()) > 50000:
                raise ValueError('Head parameter cap exceeded: ' + name)

    def forward(self, features, frame_features, starts, ends):
        scalar = self.scalars(features)
        temporal = self.frame_head(frame_features)
        # Prefix sums pool only each event's frames without per-event CUDA sync.
        width = frame_features.shape[1]
        a = starts.clamp(0, width - 1)
        b = torch.maximum(ends, a + 1).clamp(max=width)
        prefix = F.pad(frame_features.cumsum(1), (0, 0, 1, 0))
        ai = a[..., None].expand(-1, -1, frame_features.shape[-1])
        bi = b[..., None].expand_as(ai)
        event = (prefix.gather(1, bi) - prefix.gather(1, ai)) / (b - a)[..., None]
        return {'scores': scalar[:, :192], 'cps': scalar[:, 192],
                'timbre': F.normalize(self.timbre_head(features), dim=-1),
                'identity': F.normalize(self.identity_head(features), dim=-1),
                'frame': temporal[:, :, 0], 'onset': temporal[:, :, 1], 'log_duration': temporal[:, :, 2],
                'event_class': self.event_head(event)}
