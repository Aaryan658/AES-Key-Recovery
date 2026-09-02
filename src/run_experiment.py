"""Single CLI entrypoint: train one model and emit its Key Rank curve.

    python -m src.run_experiment --model cnn
    python -m src.run_experiment --model random_forest --byte 2 --n-profiling 20000
    python -m src.run_experiment --model cnn --epochs 30 --lr 5e-5 --tag quicktest

Each run creates  results/<timestamp>_<model>_byte<byte>/  containing:
    config.json              - fully resolved config + CLI overrides + env info
    run.log                  - stdout/stderr transcript
    model.pt | model.joblib  - the fitted model
    training_history.png     - loss/acc vs epoch (deep models only)
    key_rank.csv / .png      - the guessing-entropy curve (paper's main result)
    key_rank_metrics.json    - accuracy, traces-to-rank-0, runtime, ...
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from src.data_loader import load_ascad, summarise
from src.evaluate import evaluate_model
from src.models import available_models, build_model
from src.plotting import plot_training_history
from src.preprocessing import prepare_xy

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"
TORCH_MODELS = {"cnn", "resnet", "multitask_resnet"}


class _Tee:
    """Mirror stdout/stderr into run.log."""

    def __init__(self, stream, fh):
        self.stream, self.fh = stream, fh

    def write(self, s):
        self.stream.write(s)
        self.fh.write(s)

    def flush(self):
        self.stream.flush()
        self.fh.flush()


def _parse_args(argv):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", required=True,
                    help=f"one of: {', '.join(available_models())}")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--h5-path", default=None, help="override dataset.h5_path")
    ap.add_argument("--byte", type=int, default=None, help="override dataset.target_byte")
    ap.add_argument("--leakage-model", default=None, choices=["ID", "HW"])
    ap.add_argument("--n-profiling", type=int, default=None)
    ap.add_argument("--n-attack", type=int, default=None)
    ap.add_argument("--scaler", default=None, choices=["standardize", "minmax", "none"])
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--shuffle-labels", action="store_true",
                    help="control experiment: randomly permute profiling labels. A "
                         "sound pipeline then yields key rank ~= chance (~128) - "
                         "guards against label-distribution bias false positives "
                         "(Rousselot et al. 2026).")
    # common deep-model overrides
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    # eval overrides
    ap.add_argument("--max-attack-traces", type=int, default=None)
    ap.add_argument("--n-experiments", type=int, default=None)
    ap.add_argument("--tag", default=None, help="suffix for the run directory name")
    ap.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results")
    return ap.parse_args(argv)


def _resolve_config(args) -> dict:
    with open(args.config, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    if args.h5_path is not None:
        cfg["dataset"]["h5_path"] = args.h5_path
    if args.byte is not None:
        cfg["dataset"]["target_byte"] = args.byte
    if args.leakage_model is not None:
        cfg["dataset"]["leakage_model"] = args.leakage_model
    if args.n_profiling is not None:
        cfg["dataset"]["n_profiling"] = args.n_profiling
    if args.n_attack is not None:
        cfg["dataset"]["n_attack"] = args.n_attack
    if args.scaler is not None:
        cfg["preprocess"]["scaler"] = args.scaler
    if args.seed is not None:
        cfg["train"]["seed"] = args.seed

    mkey = "random_forest" if args.model in ("rf", "random_forest") else args.model
    cfg.setdefault(mkey, {})
    for name in ("epochs", "batch_size", "lr"):
        val = getattr(args, name)
        if val is not None:
            cfg[mkey][name] = val

    if args.max_attack_traces is not None:
        cfg["evaluate"]["max_attack_traces"] = args.max_attack_traces
    if args.n_experiments is not None:
        cfg["evaluate"]["n_experiments"] = args.n_experiments
    return cfg


def main(argv=None) -> int:
    args = _parse_args(argv)
    cfg = _resolve_config(args)

    ds, pre, tr, ev = cfg["dataset"], cfg["preprocess"], cfg["train"], cfg["evaluate"]
    model_key = "random_forest" if args.model in ("rf", "random_forest") else args.model
    mcfg = dict(cfg.get(model_key, {}))
    byte = int(ds["target_byte"])
    leak = ds.get("leakage_model", "ID")
    seed = int(tr.get("seed", 0))
    np.random.seed(seed)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_name = f"{stamp}_{model_key}_byte{byte}" + (f"_{args.tag}" if args.tag else "")
    run_dir = Path(args.results_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    log_fh = open(run_dir / "run.log", "w", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, log_fh)
    sys.stderr = _Tee(sys.__stderr__, log_fh)

    print(f"=== run {run_name} ===")
    print(f"python {platform.python_version()} on {platform.platform()}")
    try:
        import torch

        print(f"torch {torch.__version__}  cuda_available={torch.cuda.is_available()}")
    except Exception:  # noqa: BLE001
        print("torch not importable")

    # ---- data ------------------------------------------------------------
    t_load = time.time()
    data = load_ascad(ds["h5_path"], ds.get("n_profiling"), ds.get("n_attack"))
    print(summarise(data))
    print(f"loaded in {time.time() - t_load:.1f}s")

    x_prof, y_prof, _x_attack, scaler = prepare_xy(
        data.profiling, data.attack, byte, leak, pre.get("scaler", "standardize")
    )
    n_classes = 256 if leak.upper() == "ID" else 9
    input_length = data.profiling.n_samples

    if model_key == "multitask_resnet":
        # (N, 3) labels: [Sbox(p^k), r_out, Sbox(p^k)^r_out]. Column 0 is the
        # single-task target, so key recovery is unchanged; the extra columns
        # only feed the auxiliary heads that break the masked-SCA plateau.
        from src.preprocessing import build_multitask_labels

        mask_index = int(mcfg.get("mask_index", 15))
        y_prof = build_multitask_labels(data.profiling, byte, mask_index=mask_index)
        print(f"multi-task labels {y_prof.shape}: [y, r=masks[:,{mask_index}], y^r]")

    if args.shuffle_labels:
        rng_sh = np.random.default_rng(seed + 1)
        y_prof = y_prof[rng_sh.permutation(len(y_prof))]
        print("!! CONTROL: profiling labels shuffled - key rank should stay ~chance")

    y_seen = y_prof if y_prof.ndim == 1 else y_prof[:, 0]
    print(f"profiling X {x_prof.shape}  y classes seen: {len(np.unique(y_seen))}/{n_classes}")

    # ---- model ---------------------------------------------------------------
    model = build_model(
        model_key, n_classes=n_classes, input_length=input_length,
        device=args.device, **mcfg,
    )
    print(f"model: {model.name}  hparams: {json.dumps(mcfg, default=str)}")

    # attack traces used for the FINAL key-rank evaluation (may be reduced below
    # so a disjoint slice can serve as the val-GE checkpoint set for variable-key)
    attack_for_eval = data.attack

    fit_kwargs = {"seed": seed}
    if model_key in TORCH_MODELS:
        val_split = float(tr.get("val_split", 0.1))
        n_val = int(len(x_prof) * val_split)
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(x_prof))
        val_idx, trn_idx = perm[:n_val], perm[n_val:]
        fit_kwargs.update(x_val=x_prof[val_idx], y_val=y_prof[val_idx])
        x_fit, y_fit = x_prof[trn_idx], y_prof[trn_idx]
        print(f"train/val split: {len(trn_idx)}/{len(val_idx)}")

        # SCA model selection = validation GUESSING ENTROPY (val CE loss is
        # anti-correlated with attack success early in training).
        try:
            vkey = data.profiling.constant_key_byte(byte)
            # fixed-key: held-out profiling slice has the same key as the attack set
            fit_kwargs["val_ge"] = {
                "x": x_prof[val_idx],
                "plaintext_byte": data.profiling.plaintext_byte(byte)[val_idx],
                "true_key_byte": vkey,
                "target_byte": byte, "leakage_model": leak,
                "max_traces": min(2000, n_val), "n_experiments": 30,
            }
            print(f"SCA checkpoint metric: val GE on {n_val} held-out profiling traces")
        except ValueError:
            # variable-key: profiling keys vary, so reserve a disjoint TAIL slice
            # of the (fixed-key) attack set for the val-GE metric.
            n_val_att = int(ev.get("val_attack_traces", 10000))
            n_val_att = min(n_val_att, len(data.attack) // 2)
            cut = len(data.attack) - n_val_att
            val_att = data.attack.take(slice(cut, None))
            attack_for_eval = data.attack.take(slice(0, cut))
            fit_kwargs["val_ge"] = {
                "x": scaler.transform(val_att.traces),
                "plaintext_byte": val_att.plaintext_byte(byte),
                "true_key_byte": val_att.constant_key_byte(byte),
                "target_byte": byte, "leakage_model": leak,
                "max_traces": min(3000, n_val_att), "n_experiments": 30,
            }
            print(f"variable-key: SCA checkpoint metric = val GE on {n_val_att} reserved "
                  f"attack traces; final eval on the other {cut}")
    else:
        x_fit, y_fit = x_prof, y_prof

    t_train = time.time()
    model.fit(x_fit, y_fit, **fit_kwargs)
    train_seconds = time.time() - t_train
    print(f"training finished in {train_seconds:.1f}s")

    y_score = y_prof[:5000] if y_prof.ndim == 1 else y_prof[:5000, 0]
    prof_acc = model.score(x_prof[:5000], y_score)
    print(f"profiling top-1 acc (5k sample): {prof_acc:.4f}  (chance {1 / n_classes:.4f})")

    # ---- persist model + training history ------------------------------
    ckpt = run_dir / ("model.pt" if model_key in TORCH_MODELS else "model.joblib")
    model.save(ckpt)
    hist_png = plot_training_history(
        model.history.as_dict(), run_dir / "training_history.png",
        title=f"{model.name} byte {byte}",
    )
    if hist_png:
        print(f"wrote {hist_png}")

    # ---- key rank ---------------------------------------------------------
    print("computing key rank (guessing entropy) on the attack set ...")
    t_eval = time.time()
    out = evaluate_model(
        model, attack_for_eval, scaler, byte, leak,
        out_dir=run_dir, label=model.name,
        max_attack_traces=int(ev.get("max_attack_traces", 2000)),
        n_experiments=int(ev.get("n_experiments", 100)),
        step=int(ev.get("step", 1)),
        seed=seed,
    )
    eval_seconds = time.time() - t_eval
    print(out.key_rank.summary())
    print(f"attack top-1 acc: {out.attack_accuracy:.4f}")
    print(f"key-rank eval in {eval_seconds:.1f}s")

    # ---- resolved config + runtime -----------------------------------
    resolved = {
        "run_name": run_name,
        "cli": {k: str(v) for k, v in vars(args).items()},
        "config": cfg,
        "env": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "runtime_seconds": {
            "train": train_seconds,
            "key_rank_eval": eval_seconds,
        },
        "model_history": model.history.as_dict(),
        "key_rank_metrics": out.metrics,
    }
    with open(run_dir / "config.json", "w", encoding="utf-8") as fh:
        json.dump(resolved, fh, indent=2, default=str)

    print(f"\nALL ARTIFACTS -> {run_dir}")
    for p in sorted(run_dir.iterdir()):
        print(f"  {p.name}")
    log_fh.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
