"""Read the ASCAD HDF5 file into memory and split it into profiling / attack sets.

ASCAD.h5 layout (fixed-key, ATMEGA_AES_v1)::

    /Profiling_traces/traces     (50000, 700)  int8
    /Profiling_traces/labels     (50000,)      uint8   = Sbox(p[2] ^ k[2])
    /Profiling_traces/metadata   (50000,)      compound: plaintext[16] u8,
                                                         key[16] u8,
                                                         masks[16] u8,
                                                         desync  i4
    /Attack_traces/...            (10000, ...)  same layout

The ``labels`` dataset in the file is pre-computed for byte 2 under the ID
model. We re-derive labels ourselves from metadata (see src/preprocessing.py)
so the target byte and leakage model are configurable; the stored labels are
only used as a cross-check.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


@dataclass
class TraceSet:
    """One labelled set of traces plus the AES metadata needed to attack it."""

    traces: np.ndarray          # (n, n_samples) float/int
    plaintext: np.ndarray       # (n, 16) uint8
    key: np.ndarray             # (n, 16) uint8
    masks: "np.ndarray | None"  # (n, 16) uint8  (None if absent)
    desync: "np.ndarray | None" # (n,)    int32  (None if absent)
    stored_labels: np.ndarray   # (n,) uint8 - ASCAD's own byte-2 ID labels

    def __len__(self) -> int:
        return self.traces.shape[0]

    @property
    def n_samples(self) -> int:
        return self.traces.shape[1]

    def key_byte(self, index: int) -> np.ndarray:
        return self.key[:, index].astype(np.uint8)

    def plaintext_byte(self, index: int) -> np.ndarray:
        return self.plaintext[:, index].astype(np.uint8)

    def constant_key_byte(self, index: int) -> int:
        """Return the single key byte value, asserting it is constant (fixed-key set)."""
        col = self.key[:, index]
        uniq = np.unique(col)
        if uniq.size != 1:
            raise ValueError(
                f"key byte {index} is not constant across this set "
                f"({uniq.size} distinct values) - is this the variable-key dataset?"
            )
        return int(uniq[0])

    def take(self, sl) -> "TraceSet":
        """A new TraceSet over a slice / index array of the traces."""
        return TraceSet(
            traces=self.traces[sl],
            plaintext=self.plaintext[sl],
            key=self.key[sl],
            masks=None if self.masks is None else self.masks[sl],
            desync=None if self.desync is None else self.desync[sl],
            stored_labels=self.stored_labels[sl],
        )


@dataclass
class ASCADData:
    profiling: TraceSet
    attack: TraceSet
    source_path: Path


def _read_group(g: h5py.Group, n: "int | None") -> TraceSet:
    sl = slice(None) if n is None else slice(0, n)
    traces = np.asarray(g["traces"][sl])
    meta = g["metadata"][sl]
    fields = meta.dtype.names or ()

    plaintext = np.asarray(meta["plaintext"]) if "plaintext" in fields else None
    key = np.asarray(meta["key"]) if "key" in fields else None
    masks = np.asarray(meta["masks"]) if "masks" in fields else None
    desync = np.asarray(meta["desync"]) if "desync" in fields else None
    stored_labels = np.asarray(g["labels"][sl]) if "labels" in g else np.zeros(len(traces), np.uint8)

    if plaintext is None or key is None:
        raise RuntimeError("ASCAD metadata is missing 'plaintext' or 'key' fields")

    return TraceSet(
        traces=traces,
        plaintext=plaintext.astype(np.uint8),
        key=key.astype(np.uint8),
        masks=None if masks is None else masks.astype(np.uint8),
        desync=None if desync is None else desync.astype(np.int32),
        stored_labels=stored_labels.astype(np.uint8),
    )


def load_ascad(
    h5_path: "str | Path",
    n_profiling: "int | None" = None,
    n_attack: "int | None" = None,
) -> ASCADData:
    """Load the ASCAD HDF5 file.

    Parameters
    ----------
    h5_path : path to ASCAD.h5
    n_profiling, n_attack : optionally cap the number of traces loaded from each
        set (useful for quick smoke tests). ``None`` loads all of them.
    """
    h5_path = Path(h5_path)
    if not h5_path.exists():
        raise FileNotFoundError(
            f"{h5_path} not found. Run:  python -m data.download_ascad"
        )

    with h5py.File(h5_path, "r") as f:
        missing = [k for k in ("Profiling_traces", "Attack_traces") if k not in f]
        if missing:
            raise RuntimeError(f"{h5_path} is missing group(s) {missing}")
        profiling = _read_group(f["Profiling_traces"], n_profiling)
        attack = _read_group(f["Attack_traces"], n_attack)

    return ASCADData(profiling=profiling, attack=attack, source_path=h5_path)


def summarise(data: ASCADData) -> str:
    p, a = data.profiling, data.attack
    lines = [
        f"ASCAD  <{data.source_path}>",
        f"  profiling : {len(p):>6d} traces x {p.n_samples} samples "
        f"({p.traces.dtype}, range [{p.traces.min()}, {p.traces.max()}])",
        f"  attack    : {len(a):>6d} traces x {a.n_samples} samples",
    ]
    try:
        kb2 = p.constant_key_byte(2)
        lines.append(f"  profiling key byte 2 = 0x{kb2:02x} (fixed)")
    except ValueError as exc:  # variable-key set
        lines.append(f"  {exc}")
    try:
        kb2a = a.constant_key_byte(2)
        lines.append(f"  attack    key byte 2 = 0x{kb2a:02x} (the value to recover)")
    except ValueError:
        lines.append("  attack key byte 2 varies per trace")
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Sanity-check the ASCAD HDF5 file.")
    ap.add_argument("--h5-path", default="data/raw/ASCAD.h5")
    ap.add_argument("--n-profiling", type=int, default=2000)
    ap.add_argument("--n-attack", type=int, default=2000)
    args = ap.parse_args()

    d = load_ascad(args.h5_path, args.n_profiling, args.n_attack)
    print(summarise(d))
