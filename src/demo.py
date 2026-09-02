"""Plain-language demonstration of the recovered AES key byte.

    python -m src.demo                 # fixed-key ASCAD, byte 2 (the default target)
    python -m src.demo --ckpt results/<run>/model.pt --byte 2

What it shows, for a reader who has not seen the rest of the project:

  * the full 16-byte AES key that the target device was using;
  * which single byte of it this attack recovers, and its true value;
  * exactly what the attacker is given (power traces + known plaintext) and
    what they are NOT given (the key);
  * the attack's guess for that byte as more traces are used, ending in the
    recovered value and whether it is correct.

If ``data/raw/ASCAD.h5`` and a trained model are present it runs the attack
live. Otherwise it prints the result recorded earlier in
``results/m3_fullkey_fixed.json``. Either way it also writes
``results/demo_output.txt`` and a picture ``results/recovered_key.png``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CKPT = ROOT / "results" / "20260901-085316_cnn_byte2_milestone1_final" / "model.pt"
DEFAULT_H5 = ROOT / "data" / "raw" / "ASCAD.h5"
RECORDED = ROOT / "results" / "m3_fullkey_fixed.json"


def _key_row(key_bytes, hi):
    """A 16-cell hex row with byte ``hi`` marked."""
    cells = []
    for i, b in enumerate(key_bytes):
        cells.append(f"[{b:02x}]" if i == hi else f" {b:02x} ")
    idx = "".join(f" {i:>2} " if i != hi else f"[{i:>2}]" for i in range(16))
    return idx + "\n" + "".join(cells)


def _live_attack(ckpt: Path, h5: Path, byte: int, checkpoints):
    """Run the CNN on the attack traces; return recovered/true/key/plaintext/rows/ttr0."""
    from src.aes import AES_SBOX
    from src.data_loader import load_ascad
    from src.key_rank import guessing_entropy
    from src.models.cnn import CNNModel
    from src.preprocessing import TraceScaler

    data = load_ascad(str(h5), n_profiling=None, n_attack=None)
    scaler = TraceScaler("standardize").fit(data.profiling.traces)
    x_att = scaler.transform(data.attack.traces)
    pt_b = data.attack.plaintext_byte(byte)
    true_b = int(data.attack.key[0, byte])
    key_bytes = [int(v) for v in data.attack.key[0]]

    model = CNNModel.load(ckpt)
    proba = model.predict_proba(x_att).astype(np.float64)

    logp = np.log(np.clip(proba, 1e-40, None))
    guesses = np.arange(256, dtype=np.uint8)
    hyp = AES_SBOX[np.bitwise_xor(pt_b[:, None], guesses[None, :])]      # (N,256)
    per_trace = logp[np.arange(len(pt_b))[:, None], hyp]                 # (N,256)
    cum = np.cumsum(per_trace, axis=0)

    rows = []
    for n in checkpoints:
        n = min(n, len(cum))
        score = cum[n - 1]
        rows.append((n, int(np.argmax(score)), int((score > score[true_b]).sum())))

    ge = guessing_entropy(proba, pt_b, true_b, target_byte=byte,
                          max_traces=min(2000, len(proba)), n_experiments=100, step=5)
    recovered = int(np.argmax(cum[-1]))
    return (recovered, true_b, key_bytes,
            [int(v) for v in data.attack.plaintext[0]], rows, ge.traces_to_rank0)


def _recorded():
    d = json.loads(RECORDED.read_text(encoding="utf-8"))
    kb = bytes.fromhex(d["true_key_hex"])
    pb = d["per_byte"]["2"]
    return pb["recovered"], pb["true"], list(kb), None, None, pb["traces_to_rank0"]


def _figure(key_bytes, hi, rows, ttr0, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (a0, a1) = plt.subplots(1, 2, figsize=(11, 3.6), dpi=140,
                                 gridspec_kw={"width_ratios": [1.15, 1]})

    a0.set_xlim(0, 16); a0.set_ylim(0, 3); a0.axis("off")
    a0.set_title("The 16-byte AES key on the device", fontsize=11)
    for i, b in enumerate(key_bytes):
        hit = (i == hi)
        a0.add_patch(plt.Rectangle((i, 1), 1, 1, facecolor="#2e7d32" if hit else "#eceff1",
                                   edgecolor="#455a64"))
        a0.text(i + 0.5, 1.5, f"{b:02x}", ha="center", va="center",
                color="white" if hit else "#263238",
                fontsize=9, fontweight="bold" if hit else "normal")
        a0.text(i + 0.5, 0.7, str(i), ha="center", va="center", fontsize=7, color="#607d8b")
    a0.text(hi + 0.5, 2.2, "recovered", ha="center", fontsize=9, color="#2e7d32",
            fontweight="bold")
    a0.text(8, 0.15, "grey bytes: not attacked (their leakage is outside the trace window)",
            ha="center", fontsize=7.5, color="#607d8b")

    if rows:
        ns = [r[0] for r in rows]; rk = [r[2] for r in rows]
        a1.plot(ns, rk, "o-", color="#1565c0")
        a1.set_xscale("log")
        a1.axhline(0, color="k", lw=0.8, ls="--")
        a1.set_xlabel("number of attack traces used")
        a1.set_ylabel("rank of the true key byte\n(0 = fully recovered)")
        ttl = "Attack converges on the correct byte"
        if ttr0:
            ttl += f"\n(rank 0 by ~{ttr0} traces)"
        a1.set_title(ttl, fontsize=10)
        a1.grid(alpha=0.25)
    else:
        a1.axis("off")
        a1.text(0.5, 0.5, f"recorded result:\nrank 0 by ~{ttr0} traces",
                ha="center", va="center", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--h5-path", type=Path, default=DEFAULT_H5)
    ap.add_argument("--byte", type=int, default=2)
    args = ap.parse_args(argv)

    checkpoints = [10, 30, 100, 300, 1000, 2000]
    live = args.ckpt.exists() and args.h5_path.exists()
    if live:
        recovered, true_b, key_bytes, pt0, rows, ttr0 = _live_attack(
            args.ckpt, args.h5_path, args.byte, checkpoints)
    else:
        recovered, true_b, key_bytes, pt0, rows, ttr0 = _recorded()

    L = []
    w = L.append
    w("=" * 70)
    w("  AES KEY-RECOVERY SIDE-CHANNEL ATTACK  -  RESULT")
    w("=" * 70)
    w("")
    w("SETUP")
    w("  A small chip encrypts data with AES-128 using a fixed 16-byte secret")
    w("  key. While it encrypts, we record its electromagnetic emissions as a")
    w('  "trace" (700 numbers per encryption). We do this for many encryptions.')
    w("")
    w("WHAT THE ATTACKER IS GIVEN")
    w("  - a set of power traces (700 numbers each)")
    w("  - the plaintext that was encrypted each time (this is public)")
    if pt0 is not None:
        w("      e.g. first attack plaintext byte {}: 0x{:02x}".format(args.byte, pt0[args.byte]))
    w("  NOT given: the key. That is what we are trying to find.")
    w("")
    w("WHAT THIS ATTACK RECOVERS")
    w("  One byte of the key: byte #{} (bytes are numbered 0-15).".format(args.byte))
    w("  A trained neural network scores all 256 possible values for that byte;")
    w("  scores from many traces are added up and the values ranked.")
    w("")
    w("THE SECRET KEY ON THE DEVICE (normally unknown - shown here as ground truth)")
    w("  " + _key_row(key_bytes, args.byte).replace("\n", "\n  "))
    w("")
    w("  full key (hex): " + "".join(f"{b:02x}" for b in key_bytes))
    w("  target byte    : #{}".format(args.byte))
    w("  true value     : 0x{:02x}  ({})".format(true_b, true_b))
    w("")
    if rows:
        w("ATTACK PROGRESS  (top guess and the true byte's rank, vs traces used)")
        w("  {:>8}  {:>11}  {:>18}".format("traces", "top guess", "rank of true byte"))
        for n, top, rank in rows:
            flag = "   <- correct" if top == true_b else ""
            w("  {:>8}  {:>9}0x{:02x}  {:>18}{}".format(n, "", top, rank, flag))
        w("")
    w("RESULT")
    ok = (recovered == true_b)
    w("  recovered value: 0x{:02x}  ({})".format(recovered, recovered))
    w("  correct?       : {}".format("YES - matches the true key byte" if ok else "no"))
    if ttr0:
        w("  it took about {} attack traces for the guess to lock on and stay".format(ttr0))
    w("")
    w("  key with the recovered byte filled in:")
    filled = ["??"] * 16
    filled[args.byte] = f"{recovered:02x}"
    w("    " + " ".join(filled))
    w("=" * 70)
    text = "\n".join(L)
    print(text)

    out_txt = ROOT / "results" / "demo_output.txt"
    out_txt.write_text(text + "\n", encoding="utf-8")
    out_png = ROOT / "results" / "recovered_key.png"
    try:
        _figure(key_bytes, args.byte, rows, ttr0, out_png)
        print(f"\nwrote {out_txt}\nwrote {out_png}")
    except Exception as exc:  # noqa: BLE001
        print(f"\nwrote {out_txt}  (figure skipped: {exc})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
