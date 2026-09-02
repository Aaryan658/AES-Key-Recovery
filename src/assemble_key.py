"""Assemble a full AES-128 key from per-byte side-channel attacks.

Milestone 3 reframe. The standard extracted ASCAD databases (``ASCAD.h5``,
``ascad-variable.h5``) are a trace *window* around the masked S-box operation of
key byte 2 only, so a per-byte attack recovers **byte 2** and sits at guessing
entropy ~= chance for the other 15 bytes (empirically confirmed for bytes 3, 4 -
see ``results/*_m3_probe_b*``). Full 16-byte recovery needs the ~60 GB raw
``ATMega8515_raw_traces.h5`` plus a per-byte re-window; that is out of scope
here.

This tool still does the *assembly* step end-to-end so the pipeline is provably
byte-parametrised: give it one trained model per byte, it runs each model's
key-rank on the attack set, takes the rank-0 hypothesis as the recovered byte,
stitches the 16 bytes together and diffs against the true key from metadata.

Usage
-----
    # one --byte/--ckpt pair per attacked byte; models come from run_experiment
    python -m src.assemble_key --h5-path data/raw/ASCAD.h5 \
        --byte 2 --ckpt results/<cnn_byte2_run>/model.pt --model cnn \
        [--byte 3 --ckpt results/<run>/model.pt --model cnn] ...

    # or point at a directory of run dirs and let it match by name
    python -m src.assemble_key --h5-path data/raw/ASCAD.h5 --scan results/ --model cnn
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from src.aes import AES_SBOX
from src.data_loader import load_ascad
from src.key_rank import guessing_entropy
from src.preprocessing import TraceScaler

_MODEL_LOADERS = {}


def _loader(model: str):
    """Lazy import so a CPU-only / no-torch env can still assemble sklearn runs."""
    if model not in _MODEL_LOADERS:
        if model in ("cnn",):
            from src.models.cnn import CNNModel as C
        elif model in ("resnet",):
            from src.models.resnet import ResNetModel as C
        elif model in ("multitask_resnet", "mtl_resnet", "mtl"):
            from src.models.multitask_resnet import MultiTaskResNetModel as C
        elif model in ("random_forest", "rf"):
            from src.models.random_forest import RandomForestModel as C
        elif model in ("svm",):
            from src.models.svm import SVMModel as C
        else:
            raise KeyError(f"unknown model {model!r}")
        _MODEL_LOADERS[model] = C
    return _MODEL_LOADERS[model]


_RUN_RE = re.compile(r"_(?P<model>random_forest|svm|cnn|resnet|multitask_resnet)_byte(?P<byte>\d+)")


def _run_h5_basename(d: Path) -> "str | None":
    """The dataset a run was trained on, from its config.json (basename only)."""
    cfg = d / "config.json"
    if not cfg.exists():
        return None
    try:
        blob = json.loads(cfg.read_text(encoding="utf-8"))
        return Path(blob["config"]["dataset"]["h5_path"]).name
    except Exception:  # noqa: BLE001
        return None


def _scan_runs(root: Path, model_filter: "str | None",
               want_h5: "str | None" = None) -> dict:
    """Map target_byte -> newest matching run dir under ``root`` with a saved model.

    ``want_h5`` (a basename like ``ASCAD.h5``) restricts matches to runs trained
    on that dataset - without it a variable-key model can get picked for a
    fixed-key attack and the trace lengths won't match.
    """
    found: dict[int, Path] = {}
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        m = _RUN_RE.search(d.name)
        if not m:
            continue
        if model_filter and m.group("model") != model_filter:
            continue
        if not ((d / "model.pt").exists() or (d / "model.joblib").exists()):
            continue
        if want_h5 is not None:
            rh5 = _run_h5_basename(d)
            if rh5 is not None and rh5 != want_h5:
                continue
        b = int(m.group("byte"))
        found[b] = d  # sorted() ascending -> last write (newest timestamp) wins
    return found


def _predict_proba_for_byte(ckpt: Path, model: str, x_attack: np.ndarray) -> np.ndarray:
    cls = _loader(model)
    mdl = cls.load(ckpt)
    return mdl.predict_proba(x_attack)


def assemble(
    h5_path: str,
    byte_to_ckpt: dict,
    model: str,
    n_attack: "int | None" = None,
    max_traces: int = 3000,
    n_experiments: int = 100,
    scaler_method: str = "standardize",
) -> dict:
    data = load_ascad(h5_path, n_profiling=None, n_attack=n_attack)
    scaler = TraceScaler(scaler_method).fit(data.profiling.traces)
    x_attack = scaler.transform(data.attack.traces)

    true_key = data.attack.key[0].copy()
    key_is_constant = bool((data.attack.key == true_key).all())

    recovered = np.full(16, -1, dtype=int)
    per_byte = {}
    for b in range(16):
        ck = byte_to_ckpt.get(b)
        if ck is None:
            per_byte[b] = {"status": "not attempted"}
            continue
        try:
            proba = _predict_proba_for_byte(Path(ck), model, x_attack)
        except RuntimeError as exc:  # e.g. trace-length / dataset mismatch
            per_byte[b] = {"status": f"model error: {str(exc).splitlines()[-1][:120]}"}
            continue
        pt_b = data.attack.plaintext_byte(b)
        tkb = int(data.attack.key[:, b][0]) if key_is_constant else None
        ge = guessing_entropy(
            proba, pt_b, true_key_byte=(tkb if tkb is not None else 0),
            target_byte=b, leakage_model="ID",
            max_traces=min(max_traces, len(x_attack)),
            n_experiments=n_experiments, step=max(1, max_traces // 500),
        )
        # rank-0 hypothesis at the full trace budget = the recovered byte guess
        logp = np.log(np.clip(proba.astype(np.float64), 1e-40, None))
        guesses = np.arange(256, dtype=np.uint8)
        hyp = AES_SBOX[np.bitwise_xor(pt_b[:, None], guesses[None, :])]  # (N,256)
        scores = logp[np.arange(len(pt_b))[:, None], hyp].sum(axis=0)    # (256,)
        guess = int(np.argmax(scores))
        recovered[b] = guess
        per_byte[b] = {
            "status": "attempted",
            "true": tkb,
            "recovered": guess,
            "correct": (tkb is not None and guess == tkb),
            "final_mean_rank": float(ge.mean_rank[-1]),
            "traces_to_rank0": ge.traces_to_rank0,
        }

    attempted = [b for b in range(16) if per_byte[b]["status"] == "attempted"]
    correct = [b for b in attempted if per_byte[b]["correct"]]
    return {
        "h5_path": h5_path,
        "key_is_constant": key_is_constant,
        "true_key_hex": true_key.tobytes().hex() if key_is_constant else None,
        "recovered_key_hex": bytes(int(x) & 0xFF if x >= 0 else 0 for x in recovered).hex(),
        "recovered_mask": [int(x) for x in recovered],
        "bytes_attempted": attempted,
        "bytes_correct": correct,
        "per_byte": per_byte,
    }


def _format_report(res: dict) -> str:
    lines = [
        f"ASCAD           : {res['h5_path']}",
        f"attack key      : {'constant' if res['key_is_constant'] else 'varies per trace'}",
    ]
    if res["true_key_hex"]:
        lines.append(f"true key    (hex): {res['true_key_hex']}")
    lines.append("")
    lines.append(f"{'byte':>4}  {'status':<13} {'true':>5} {'rec':>5} {'ok':>3} "
                 f"{'final rank':>11} {'->rank0':>8}")
    for b in range(16):
        pb = res["per_byte"][b]
        if pb["status"] != "attempted":
            lines.append(f"{b:>4}  {pb['status']:<13}")
            continue
        t = "--" if pb["true"] is None else f"{pb['true']:02x}"
        lines.append(
            f"{b:>4}  {'attempted':<13} {t:>5} {pb['recovered']:5x} "
            f"{('Y' if pb['correct'] else 'n'):>3} {pb['final_mean_rank']:>11.1f} "
            f"{pb['traces_to_rank0']!s:>8}"
        )
    lines.append("")
    lines.append(f"bytes attempted : {res['bytes_attempted']}")
    lines.append(f"bytes recovered : {res['bytes_correct']}  "
                 f"({len(res['bytes_correct'])}/{len(res['bytes_attempted'])} of attempted)")
    lines.append(f"recovered key   : {res['recovered_key_hex']}")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--h5-path", default="data/raw/ASCAD.h5")
    ap.add_argument("--model", default="cnn",
                    help="model type of the checkpoints (cnn|resnet|multitask_resnet|svm|random_forest)")
    ap.add_argument("--byte", type=int, action="append", default=[], dest="bytes_",
                    help="target byte (repeatable); pair each with a --ckpt")
    ap.add_argument("--ckpt", type=Path, action="append", default=[],
                    help="model checkpoint for the matching --byte (repeatable)")
    ap.add_argument("--scan", type=Path, default=None,
                    help="directory of run dirs; auto-match target_byte -> newest model")
    ap.add_argument("--n-attack", type=int, default=None)
    ap.add_argument("--max-traces", type=int, default=3000)
    ap.add_argument("--n-experiments", type=int, default=100)
    ap.add_argument("--out", type=Path, default=None, help="write the JSON report here")
    args = ap.parse_args(argv)

    byte_to_ckpt: dict[int, Path] = {}
    if args.scan:
        byte_to_ckpt.update(
            {b: (d / ("model.pt" if (d / "model.pt").exists() else "model.joblib"))
             for b, d in _scan_runs(args.scan, args.model, Path(args.h5_path).name).items()}
        )
    if len(args.bytes_) != len(args.ckpt):
        ap.error("need one --ckpt per --byte")
    byte_to_ckpt.update(dict(zip(args.bytes_, args.ckpt)))
    if not byte_to_ckpt:
        ap.error("no models given: use --scan DIR or --byte/--ckpt pairs")

    print(f"assembling from {len(byte_to_ckpt)} per-byte model(s): "
          f"bytes {sorted(byte_to_ckpt)}")
    res = assemble(
        args.h5_path, byte_to_ckpt, args.model,
        n_attack=args.n_attack, max_traces=args.max_traces,
        n_experiments=args.n_experiments,
    )
    print("\n" + _format_report(res))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
