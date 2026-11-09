import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalTransformer(nn.Module):
    """Tokens (B,T,D) + learned month-of-year + sinusoidal position -> CLS embedding."""

    def __init__(self, d=128, layers=2, heads=4, T=72, drop=0.1):
        super().__init__()
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.month = nn.Embedding(12, d)
        pe = torch.zeros(T, d)
        pos = torch.arange(T).float()[:, None]
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer("pe", pe[None])
        self.register_buffer("months", (torch.arange(T) % 12)[None])
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, drop, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)
        self.norm = nn.LayerNorm(d)

    def forward(self, x):
        B, T, _ = x.shape
        x = x + self.pe[:, :T] + self.month(self.months[:, :T])
        x = torch.cat([self.cls.expand(B, -1, -1), x], 1)
        return self.norm(self.enc(x)[:, 0])


def head(d, n=3, drop=0.2):
    return nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Dropout(drop), nn.Linear(d, n))


# ------------------------------------------------------------------ M1: BiGRU on patch-mean series
class M1GRU(nn.Module):
    def __init__(self, c_in, d=128):
        super().__init__()
        self.inp = nn.Linear(c_in, d)
        self.gru = nn.GRU(d, d, 2, batch_first=True, bidirectional=True, dropout=0.1)
        self.att = nn.Linear(2 * d, 1)
        self.head = head(2 * d)

    def forward(self, s):  # s (B,T,C)
        h, _ = self.gru(F.gelu(self.inp(s)))
        w = torch.softmax(self.att(h), 1)
        return self.head((w * h).sum(1))


# ------------------------------------------------------------------ M2: frame CNN + temporal transformer
def cbr(i, o, s=1):
    return nn.Sequential(nn.Conv2d(i, o, 3, s, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class SmallCNN(nn.Module):
    """Frame encoder. stem_stride=2 for big windows. Returns global + centre-region pooled features."""

    def __init__(self, c_in, width=32, stem_stride=1, d=128):
        super().__init__()
        w = width
        self.net = nn.Sequential(cbr(c_in, w, stem_stride), cbr(w, w), nn.MaxPool2d(2),
                                 cbr(w, 2 * w), cbr(2 * w, 2 * w), nn.MaxPool2d(2),
                                 cbr(2 * w, 4 * w), cbr(4 * w, 4 * w))
        self.proj = nn.Linear(8 * w, d)

    def forward(self, x, k=1):
        f = self.net(x)
        g = f.mean((2, 3))
        if k > 1:  # centre patch region in feature map
            H, W = f.shape[2:]
            c = f[:, :, H * (k // 2) // k: H * (k // 2 + 1) // k, W * (k // 2) // k: W * (k // 2 + 1) // k].mean((2, 3))
        else:
            c = f.amax((2, 3))
        return self.proj(torch.cat([g, c], 1))


class ResNet18Enc(nn.Module):
    """torchvision resnet18, maxpool removed, input upsampled x2. Optionally SSL4EO-S12 MoCo weights.

    in_map: list, for each input channel the SSL4EO 13-band index to copy conv1 weights from (or -1 = new).
    """

    def __init__(self, c_in, in_map, pretrained, d=128):
        super().__init__()
        import torchvision
        m = torchvision.models.resnet18(weights=None)
        if pretrained:
            url = ("https://hf.co/torchgeo/resnet18_sentinel2_all_moco/resolve/"
                   "5b8cddc9a14f3844350b7f40b85bcd32aed75918/resnet18_sentinel2_all_moco-59bfdff9.pth")
            sd = torch.hub.load_state_dict_from_url(url, map_location="cpu", progress=False)
            sd = {k.replace("backbone.", "").replace("encoder_q.", ""): v for k, v in sd.items()}
            w13 = sd.pop("conv1.weight")  # (64,13,7,7)
            sd = {k: v for k, v in sd.items() if not k.startswith("fc.")}
            missing, unexpected = m.load_state_dict(sd, strict=False)
            print("ssl4eo load: missing", [k for k in missing if not k.startswith("conv1")], "unexpected", unexpected[:5], flush=True)
        conv1 = nn.Conv2d(c_in, 64, 7, 2, 3, bias=False)
        if pretrained:
            with torch.no_grad():
                nn.init.normal_(conv1.weight, std=0.01)
                for j, src in enumerate(in_map):
                    if src >= 0:
                        conv1.weight[:, j] = w13[:, src] * (13.0 / max(1, sum(s >= 0 for s in in_map)))
        m.conv1 = conv1
        m.maxpool = nn.Identity()
        m.fc = nn.Identity()
        self.m = m
        self.proj = nn.Linear(512, d)

    def forward(self, x, k=1):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.proj(self.m(x))


class M2(nn.Module):
    def __init__(self, encoder, d=128, layers=2):
        super().__init__()
        self.enc = encoder
        self.tt = TemporalTransformer(d, layers)
        self.head = head(d)

    def forward(self, x, k=1, chunk=1024):  # x (B,T,C,H,W)
        B, T = x.shape[:2]
        flat = x.flatten(0, 1)
        feats = torch.cat([self.enc(flat[i:i + chunk], k) for i in range(0, flat.shape[0], chunk)], 0)
        return self.head(self.tt(feats.view(B, T, -1)))


# ------------------------------------------------------------------ Presto (pretrained pixel-timeseries FM)
class PrestoClassifier(nn.Module):
    """Presto encoder over S sub-patch mean series; 72 months -> 3 chunks of 24; mean+std pooled."""

    def __init__(self, pretrained=True):
        super().__init__()
        from presto_vendor import Presto
        p = Presto.construct()
        if pretrained:
            path = os.path.join(os.path.dirname(__file__), "presto_default_model.pt")
            p.load_state_dict(torch.load(path, map_location="cpu"))
        self.enc = p.encoder
        self.enc.pos_embed.requires_grad_(False)
        self.enc.month_embed.requires_grad_(False)
        d = self.enc.embedding_size
        self.head = head(2 * 3 * d)

    def forward(self, x, latlon):
        # x: (B,S,72,17) presto-normalised; latlon (B,2)
        B, S, T, C = x.shape
        x = x.view(B * S, 3, 24, C).flatten(0, 1)  # (B*S*3,24,17)
        n = x.shape[0]
        dw = torch.full((n, 24), 9, dtype=torch.long, device=x.device)  # 9 = missing DW
        ll = latlon[:, None].expand(B, S * 3, 2).reshape(n, 2)
        mask = torch.zeros_like(x)
        mask[:, :, 5:8] = 1   # red edge  (unavailable)
        mask[:, :, 9:12] = 1  # B8A, SWIR (unavailable)
        mask[:, :, 12:16] = 1  # ERA5, SRTM (unavailable)
        e = self.enc(x, dynamic_world=dw, latlons=ll, mask=mask, month=0, eval_task=True)  # (n,d)
        e = e.view(B, S, 3 * e.shape[-1])
        return self.head(torch.cat([e.mean(1), e.std(1)], 1))
