"""1D ResNet over the raw trace (PyTorch), for the variable-key ASCAD set.

Architecture follows Karayalcin, Perin & Picek, "Resolving the Doubts: On the
Construction and Use of ResNets for Side-Channel Analysis" (Mathematics 2023),
which is the reference design for ResNet-SCA on ASCAD-r:

  stem:  Conv1d(k=stem_kernel) -> BN -> act
  block x N (N = floor(log2(input_length)) - 2 by default):
         Conv1d(k) -> BN -> act -> Conv1d(k) -> BN   (+ 1x1 conv shortcut)
         act(out + identity)
         AvgPool1d(2)                        <- every block halves the length
  head:  Flatten -> [Linear -> (BN) -> act -> (dropout)] x fc_units -> Linear(256)

Defaults vs the earlier version (see docs/resnet_improvement_research.md):
  kernel_size 11 (was 3), activation SELU + lecun-normal init (was ReLU),
  ~8 blocks for 1400 samples (was 6), flatten head (not global average pool).
The masked-SCA "initial plateau" means this needs ~100 epochs / batch 50 and
should be judged by validation guessing entropy, not by the first few epochs.
"""

from __future__ import annotations

import math

from torch import nn

from src.models.torch_common import TorchSCAModel

_ACT = {"selu": nn.SELU, "relu": nn.ReLU, "elu": nn.ELU}


def _act(name: str) -> nn.Module:
    return nn.SELU() if name == "selu" else _ACT[name](inplace=True)


class _ResBlock(nn.Module):
    """2x (Conv-BN-act) with a residual shortcut, then AvgPool(2)."""

    def __init__(self, c_in: int, c_out: int, k: int, activation: str, pool: bool = True):
        super().__init__()
        pad = k // 2
        self.conv1 = nn.Conv1d(c_in, c_out, k, padding=pad, bias=False)
        self.bn1 = nn.BatchNorm1d(c_out)
        self.conv2 = nn.Conv1d(c_out, c_out, k, padding=pad, bias=False)
        self.bn2 = nn.BatchNorm1d(c_out)
        self.act = _act(activation)
        self.skip = (
            nn.Sequential(nn.Conv1d(c_in, c_out, 1, bias=False), nn.BatchNorm1d(c_out))
            if c_in != c_out else nn.Identity()
        )
        self.pool = nn.AvgPool1d(2) if pool else nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.pool(self.act(out + identity))


class _ResNet1D(nn.Module):
    def __init__(self, n_classes, input_length, stem_filters, stem_kernel, k,
                 n_blocks, max_filters, fc_units, dropout, activation="selu",
                 head_bn=True, head_pool="flatten", stem_pool=1):
        super().__init__()
        stem_layers = [
            nn.Conv1d(1, stem_filters, stem_kernel, padding=stem_kernel // 2, bias=False),
            nn.BatchNorm1d(stem_filters),
            _act(activation),
        ]
        length = input_length
        if stem_pool and stem_pool > 1:
            # downsample once up front so the k=11 convs don't run over the full
            # 1400-length sequence (keeps VRAM / epoch time sane on an 8 GB card)
            stem_layers.append(nn.MaxPool1d(stem_pool))
            length //= stem_pool
        self.stem = nn.Sequential(*stem_layers)
        blocks, c_in = [], stem_filters
        for i in range(n_blocks):
            c_out = min(stem_filters * (2 ** i), max_filters)
            blocks.append(_ResBlock(c_in, c_out, k, activation, pool=(length // 2 >= 1)))
            c_in = c_out
            if length // 2 >= 1:
                length //= 2
        self.blocks = nn.Sequential(*blocks)

        if head_pool == "flatten":
            self.pool = nn.Identity()
            flat = c_in * max(length, 1)
        elif head_pool == "avg":
            self.pool = nn.AdaptiveAvgPool1d(1)
            flat = c_in
        else:
            n = int(head_pool)
            self.pool = nn.AdaptiveAvgPool1d(n)
            flat = c_in * n

        head, prev = [], flat
        for u in fc_units:
            head.append(nn.Linear(prev, u))
            if head_bn:
                head.append(nn.BatchNorm1d(u))
            head.append(_act(activation))
            if dropout > 0:
                head.append(nn.Dropout(dropout))
            prev = u
        head.append(nn.Linear(prev, n_classes))
        self.classifier = nn.Sequential(*head)
        self._init(activation)

    def _init(self, activation):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                if activation == "selu":  # lecun-normal
                    fan_in = (m.in_channels * m.kernel_size[0]
                              if isinstance(m, nn.Conv1d) else m.in_features)
                    nn.init.normal_(m.weight, 0.0, math.sqrt(1.0 / fan_in))
                else:
                    nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)


class ResNetModel(TorchSCAModel):
    name = "resnet"

    def build_network(self) -> nn.Module:
        hp = self.hparams
        stem_pool = int(hp.get("stem_pool", 1))
        eff_len = self.input_length // max(stem_pool, 1)
        default_blocks = max(2, int(math.log2(eff_len)) - 2)
        return _ResNet1D(
            n_classes=self.n_classes,
            input_length=self.input_length,
            stem_filters=int(hp.get("stem_filters", 16)),
            stem_kernel=int(hp.get("stem_kernel", 11)),
            k=int(hp.get("kernel_size", 11)),
            n_blocks=int(hp.get("n_blocks", default_blocks)),
            max_filters=int(hp.get("max_filters", 256)),
            fc_units=hp.get("fc_units", [200]),
            dropout=float(hp.get("dropout", 0.0)),
            activation=hp.get("activation", "selu"),
            head_bn=bool(hp.get("head_bn", True)),
            head_pool=hp.get("head_pool", "flatten"),
            stem_pool=stem_pool,
        )
