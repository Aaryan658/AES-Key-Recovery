"""Trace normalisation and label construction.

Two responsibilities:

1. ``TraceScaler`` - amplitude normalisation fitted on the *profiling* set only
   and then applied unchanged to the attack set (never fit on attack data - that
   would leak information the real attacker does not have).

2. ``build_labels`` - map (plaintext, key) to the 256-class first-round S-box
   label for a chosen key byte, on the profiling set. For the attack set there is
   no single label: every key guess implies a different label, which is what the
   key-rank evaluation in src/key_rank.py enumerates.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.aes import labels_from_plaintext_key
from src.data_loader import TraceSet


@dataclass
class TraceScaler:
    """Fit-on-profiling amplitude scaler. ``method`` in {standardize, minmax, none}."""

    method: str = "standardize"
    mean_: "np.ndarray | None" = None
    std_: "np.ndarray | None" = None
    min_: "np.ndarray | None" = None
    ptp_: "np.ndarray | None" = None

    def fit(self, x: np.ndarray) -> "TraceScaler":
        # float64 stats without a full float64 copy of x (variable-key ASCAD is
        # 200k x 1400 -> a float64 copy is ~2.2 GB). Accumulate over row chunks.
        x = np.asarray(x)
        n, d = x.shape
        if self.method == "standardize":
            ssum = np.zeros(d, np.float64)
            ssq = np.zeros(d, np.float64)
            for i in range(0, n, 20000):
                blk = x[i:i + 20000].astype(np.float64)
                ssum += blk.sum(axis=0)
                ssq += (blk * blk).sum(axis=0)
            self.mean_ = ssum / n
            var = np.maximum(ssq / n - self.mean_ ** 2, 0.0)
            self.std_ = np.sqrt(var)
            self.std_[self.std_ == 0] = 1.0
        elif self.method == "minmax":
            self.min_ = x.min(axis=0).astype(np.float64)
            self.ptp_ = (x.max(axis=0).astype(np.float64) - self.min_)
            self.ptp_[self.ptp_ == 0] = 1.0
        elif self.method == "none":
            pass
        else:
            raise ValueError(f"unknown scaler method {self.method!r}")
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if self.method == "standardize":
            return ((x - self.mean_) / self.std_).astype(np.float32)
        if self.method == "minmax":
            return ((x - self.min_) / self.ptp_).astype(np.float32)
        return x

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)


def build_labels(
    ts: TraceSet,
    target_byte: int,
    leakage_model: str = "ID",
    cross_check: bool = True,
) -> np.ndarray:
    """256-class (or 9-class for HW) label per profiling trace.

    ``label[i] = Sbox(plaintext[i, target_byte] XOR key[i, target_byte])`` under
    the ID model. Works for both fixed- and variable-key sets because the key is
    taken per trace from metadata.
    """
    pt = ts.plaintext_byte(target_byte)
    kb = ts.key_byte(target_byte)
    labels = labels_from_plaintext_key(pt, kb, leakage_model)

    if cross_check and leakage_model.upper() == "ID" and target_byte == 2:
        # ASCAD's own stored labels are byte 2 / ID; they must match exactly.
        if not np.array_equal(labels.astype(np.uint8), ts.stored_labels):
            n_bad = int((labels.astype(np.uint8) != ts.stored_labels).sum())
            raise AssertionError(
                f"derived byte-2 ID labels disagree with ASCAD stored labels "
                f"on {n_bad}/{len(labels)} traces - metadata parsing bug"
            )
    return labels


#: ANSSI ASCAD convention: metadata['masks'] is [r[2..15] (14 state masks),
#: r_in, r_out], so r_out is the last column. The masked-S-box value the device
#: actually manipulates at the targeted sample is Sbox(p^k) ^ r_out.
ASCAD_ROUT_INDEX = 15


def build_multitask_labels(
    ts: TraceSet,
    target_byte: int,
    mask_index: int = ASCAD_ROUT_INDEX,
) -> np.ndarray:
    """(N, 3) int64 label matrix for the multi-task ResNet.

    Columns, in :data:`src.models.multitask_resnet.HEAD_ORDER` order:

    ==== ================================================= ============
    col  meaning                                           head
    ==== ================================================= ============
    0    ``y  = Sbox(plaintext[b] ^ key[b])``  (unmasked)   ``y``  (key)
    1    ``r  = masks[:, mask_index]``  (r_out share)       ``r``
    2    ``yr = y ^ r``  (masked S-box output)              ``yr``
    ==== ================================================= ============

    Column 0 is exactly what :func:`build_labels` produces under the ID model, so
    key recovery from the ``y`` head is identical to the single-task path.
    """
    if ts.masks is None:
        raise ValueError(
            "multi-task labels need per-trace mask shares, but this TraceSet has "
            "no 'masks' metadata (is this a synthetic / non-ASCAD set?)"
        )
    if not 0 <= mask_index < ts.masks.shape[1]:
        raise ValueError(f"mask_index {mask_index} out of range for masks{ts.masks.shape}")

    y = build_labels(ts, target_byte, leakage_model="ID", cross_check=(target_byte == 2))
    r = ts.masks[:, mask_index].astype(np.int64)
    yr = np.bitwise_xor(y.astype(np.int64), r)
    return np.stack([y.astype(np.int64), r, yr], axis=1)


def prepare_xy(
    profiling: TraceSet,
    attack: TraceSet,
    target_byte: int,
    leakage_model: str = "ID",
    scaler_method: str = "standardize",
):
    """One-stop preprocessing.

    Returns
    -------
    x_prof, y_prof : scaled profiling traces (float32) and their labels (int64)
    x_attack       : scaled attack traces (float32)  [no single label]
    scaler         : the fitted TraceScaler (kept for reproducibility / saving)
    """
    scaler = TraceScaler(scaler_method).fit(profiling.traces)
    x_prof = scaler.transform(profiling.traces)
    x_attack = scaler.transform(attack.traces)
    y_prof = build_labels(profiling, target_byte, leakage_model)
    return x_prof, y_prof, x_attack, scaler
