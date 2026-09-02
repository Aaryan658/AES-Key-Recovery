"""1D-CNN over the raw trace (PyTorch).

Two architectures, chosen by the ``arch`` hyperparameter:

* ``arch: zaid``  (default) -- the compact ASCAD CNN from Zaid et al.,
  "Methodology for Efficient CNN Architectures in Profiling Attacks" (CHES 2020).
  ``Conv1d(4, k=1) -> BN -> AvgPool(2) -> Flatten -> Dense(10) x3 -> Dense(256)``,
  SELU activations, Adam @ 5e-3. ~1.7e4 parameters: it cannot memorise 45k
  traces, which is exactly why it generalises on masked ASCAD where the big
  VGG-style net just overfits (train acc climbs, val loss diverges, key never
  ranks).

* ``arch: vgg``  -- the ASCAD "CNN_best" family (Benadjila et al. 2020):
  ``[Conv1d(k=11) -> BN -> ReLU -> AvgPool(2)] x5`` widths 64..512, then
  ``Dense x2``. Kept for comparison; needs RMSprop @ 1e-5 and long training.

Deviation from the Keras originals: BatchNorm after the conv(s). With SELU +
lecun-normal init the block is close to the paper's self-normalising design.
"""

from __future__ import annotations

import math

from torch import nn

from src.models.torch_common import TorchSCAModel

_ACT = {"selu": nn.SELU, "relu": nn.ReLU, "elu": nn.ELU}


def _make_act(name: str) -> nn.Module:
    return _ACT[name](inplace=True) if name != "selu" else nn.SELU()


class _ConvBlock(nn.Module):
    def __init__(self, c_in, c_out, k, pool, act, use_bn):
        super().__init__()
        layers = [nn.Conv1d(c_in, c_out, kernel_size=k, padding=k // 2)]
        if use_bn:
            layers.append(nn.BatchNorm1d(c_out))
        layers.append(_make_act(act))
        if pool and pool > 1:
            layers.append(nn.AvgPool1d(pool))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class _CNNNet(nn.Module):
    def __init__(self, input_length, n_classes, filters, k, pool, fc_units,
                 dropout, act="selu", use_bn=True):
        super().__init__()
        convs, c_in, length = [], 1, input_length
        for c_out in filters:
            convs.append(_ConvBlock(c_in, c_out, k, pool, act, use_bn))
            c_in = c_out
            if pool and pool > 1:
                length //= pool
        self.features = nn.Sequential(*convs)
        flat = c_in * max(length, 1)

        head, prev = [], flat
        for u in fc_units:
            head += [nn.Linear(prev, u), _make_act(act)]
            if dropout > 0:
                head.append(nn.Dropout(dropout))
            prev = u
        head.append(nn.Linear(prev, n_classes))
        self.classifier = nn.Sequential(*head)
        self._init_weights(act)

    def _init_weights(self, act):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                if act == "selu":  # lecun-normal, the SELU-recommended init
                    fan_in = (
                        m.in_channels * m.kernel_size[0]
                        if isinstance(m, nn.Conv1d) else m.in_features
                    )
                    nn.init.normal_(m.weight, 0.0, math.sqrt(1.0 / fan_in))
                else:
                    nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)


_PRESETS = {
    "zaid": dict(conv_filters=[4], kernel_size=1, pool_size=2,
                 fc_units=[10, 10, 10], dropout=0.0, activation="selu", use_bn=True),
    "vgg": dict(conv_filters=[64, 128, 256, 512, 512], kernel_size=11, pool_size=2,
                fc_units=[4096, 4096], dropout=0.0, activation="relu", use_bn=True),
}


class CNNModel(TorchSCAModel):
    name = "cnn"

    def build_network(self) -> nn.Module:
        hp = self.hparams
        preset = _PRESETS.get(hp.get("arch", "zaid"), _PRESETS["zaid"])
        g = lambda key: hp.get(key, preset[key])  # noqa: E731
        return _CNNNet(
            input_length=self.input_length,
            n_classes=self.n_classes,
            filters=g("conv_filters"),
            k=int(g("kernel_size")),
            pool=int(g("pool_size")),
            fc_units=g("fc_units"),
            dropout=float(g("dropout")),
            act=g("activation"),
            use_bn=bool(g("use_bn")),
        )
