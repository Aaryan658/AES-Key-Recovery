"""Multi-task 1D ResNet for masked-SCA (PyTorch).

A shared ResNet trunk (identical stem + residual blocks to
:mod:`src.models.resnet`) feeding several linear heads that are trained
*jointly*:

  * ``y``   - the unmasked first-round S-box output ``Sbox(p xor k)``  (256 cls)
  * ``r``   - the output mask share ``r_out`` = ``metadata['masks'][:, mask_index]``
  * ``yr``  - the masked S-box output ``Sbox(p xor k) xor r_out``       (256 cls)

Only the ``y`` head is used for key recovery (``predict_proba`` / val-GE); the
``r`` and ``yr`` heads exist purely to give back-propagation a non-flat gradient
during the masked-SCA "initial plateau". Marquet & Oswald (eprint 2023/006) show
this makes plateau escape *consistent* (< ~35 epochs) where single-task training
"converges around epoch 20, 30, or 70, or not at all". See
``docs/resnet_improvement_research.md`` section 4.5.

``y xor r == yr`` by construction, so the three tasks share all the information
the single-task net had - the auxiliary heads only change the optimisation path,
not the target.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from src.models.resnet import _act, _ResBlock
from src.models.torch_common import TorchSCAModel

#: order matters - column i of the (N, 3) label matrix maps to head i, and head 0
#: is the key-recovery head.
HEAD_ORDER = ("y", "r", "yr")


class _MultiTaskResNet1D(nn.Module):
    """Shared trunk (stem + N residual blocks, flatten) + one linear head per task."""

    def __init__(self, input_length, stem_filters, stem_kernel, k, n_blocks,
                 max_filters, fc_units, dropout, activation="selu", head_bn=True,
                 stem_pool=1, n_classes=256, heads=HEAD_ORDER):
        super().__init__()
        stem_layers = [
            nn.Conv1d(1, stem_filters, stem_kernel, padding=stem_kernel // 2, bias=False),
            nn.BatchNorm1d(stem_filters),
            _act(activation),
        ]
        length = input_length
        if stem_pool and stem_pool > 1:
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
        flat = c_in * max(length, 1)

        self._order = tuple(heads)
        self.heads = nn.ModuleDict(
            {name: self._make_head(flat, fc_units, dropout, activation, head_bn, n_classes)
             for name in self._order}
        )
        self._init(activation)

    @staticmethod
    def _make_head(flat, fc_units, dropout, activation, head_bn, n_classes):
        layers, prev = [], flat
        for u in fc_units:
            layers.append(nn.Linear(prev, u))
            if head_bn:
                layers.append(nn.BatchNorm1d(u))
            layers.append(_act(activation))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = u
        layers.append(nn.Linear(prev, n_classes))
        return nn.Sequential(*layers)

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
        z = self.blocks(self.stem(x)).flatten(1)
        return tuple(self.heads[name](z) for name in self._order)


class MultiTaskResNetModel(TorchSCAModel):
    """ResNet trunk with joint (y, r_out, y^r_out) heads; only ``y`` recovers the key.

    ``fit`` expects ``y`` of shape ``(N, 3)`` int64 - columns ``[y, r, yr]`` in
    :data:`HEAD_ORDER` order (build them with
    :func:`src.preprocessing.build_multitask_labels`). ``predict_proba`` returns
    the softmax of the ``y`` head only, so the rest of the pipeline
    (``src/evaluate.py``, ``src/key_rank.py``) is unchanged.
    """

    name = "multitask_resnet"

    def build_network(self) -> nn.Module:
        hp = self.hparams
        stem_pool = int(hp.get("stem_pool", 1))
        eff_len = self.input_length // max(stem_pool, 1)
        default_blocks = max(2, int(math.log2(eff_len)) - 2)
        return _MultiTaskResNet1D(
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
            stem_pool=stem_pool,
            n_classes=self.n_classes,
            heads=tuple(hp.get("heads", HEAD_ORDER)),
        )

    # --- multi-task training hooks (see TorchSCAModel) -------------------
    def _task_weights(self, n_heads: int):
        w = self.hparams.get("task_weights", [1.0] * n_heads)
        if len(w) != n_heads:
            raise ValueError(f"task_weights has {len(w)} entries, need {n_heads}")
        return [float(v) for v in w]

    def _compute_loss(self, out, yb, loss_fn) -> torch.Tensor:
        w = self._task_weights(len(out))
        return sum(wi * loss_fn(o, yb[:, i]) for i, (o, wi) in enumerate(zip(out, w)))

    def _num_correct(self, out, yb) -> torch.Tensor:
        # accuracy is tracked on the key-recovery head only
        return (out[0].detach().argmax(1) == yb[:, 0]).sum()

    def _proba_logits(self, out) -> torch.Tensor:
        return out[0]
