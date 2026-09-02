"""Turn a fitted model into the paper's Key Rank result.

Given a model with ``predict_proba`` and the ASCAD attack set:
  1. get (N_attack, 256) class probabilities,
  2. run the guessing-entropy computation (src/key_rank.py),
  3. write key_rank.csv + key_rank.png + a metrics dict.

Also usable standalone on a saved model:
    python -m src.evaluate --model cnn --ckpt results/<run>/model.pt --byte 2
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.data_loader import TraceSet, load_ascad
from src.key_rank import KeyRankResult, guessing_entropy
from src.plotting import plot_key_rank
from src.preprocessing import TraceScaler, build_labels


@dataclass
class EvalOutputs:
    key_rank: KeyRankResult
    attack_accuracy: float
    csv_path: Path
    plot_path: Path
    metrics: dict


def _write_csv(res: KeyRankResult, path: Path) -> None:
    p10 = np.percentile(res.all_ranks, 10, axis=0)
    p90 = np.percentile(res.all_ranks, 90, axis=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("n_traces,mean_rank,p10,p90\n")
        for n, m, a, b in zip(res.n_traces, res.mean_rank, p10, p90):
            fh.write(f"{int(n)},{m:.4f},{a:.4f},{b:.4f}\n")


def evaluate_model(
    model,
    attack: TraceSet,
    scaler: TraceScaler,
    target_byte: int,
    leakage_model: str = "ID",
    out_dir: "str | Path" = "results/_adhoc",
    label: str = "model",
    max_attack_traces: int = 2000,
    n_experiments: int = 100,
    step: int = 1,
    seed: int = 0,
) -> EvalOutputs:
    out_dir = Path(out_dir)
    x_attack = scaler.transform(attack.traces)
    proba = model.predict_proba(x_attack)

    true_key_byte = attack.constant_key_byte(target_byte)
    pt_bytes = attack.plaintext_byte(target_byte)

    # plain accuracy, reported but explicitly not the headline number
    y_true = build_labels(attack, target_byte, leakage_model, cross_check=False)
    acc = float(np.mean(np.argmax(proba, axis=1) == y_true))

    res = guessing_entropy(
        proba=proba,
        plaintext_bytes=pt_bytes,
        true_key_byte=true_key_byte,
        target_byte=target_byte,
        leakage_model=leakage_model,
        max_traces=max_attack_traces,
        n_experiments=n_experiments,
        step=step,
        rng=seed,
    )

    csv_path = out_dir / "key_rank.csv"
    plot_path = out_dir / "key_rank.png"
    _write_csv(res, csv_path)
    plot_key_rank(
        {label: res},
        plot_path,
        title=f"ASCAD fixed-key, byte {target_byte}: Key Rank vs traces ({label})",
    )

    metrics = {
        "label": label,
        "target_byte": target_byte,
        "leakage_model": leakage_model,
        "true_key_byte": true_key_byte,
        "attack_top1_accuracy": acc,
        "chance_accuracy": 1.0 / proba.shape[1],
        "n_attack_traces_used": int(res.n_traces[-1]),
        "n_experiments": n_experiments,
        "final_mean_key_rank": float(res.mean_rank[-1]),
        "traces_to_rank0": res.traces_to_rank0,
    }
    with open(out_dir / "key_rank_metrics.json", "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    return EvalOutputs(res, acc, csv_path, plot_path, metrics)


def _standalone(argv=None) -> int:
    import argparse

    from src.models.cnn import CNNModel
    from src.models.random_forest import RandomForestModel
    from src.models.resnet import ResNetModel
    from src.models.svm import SVMModel

    loaders = {
        "random_forest": RandomForestModel,
        "svm": SVMModel,
        "cnn": CNNModel,
        "resnet": ResNetModel,
    }

    ap = argparse.ArgumentParser(description="Evaluate a saved model -> key-rank curve.")
    ap.add_argument("--model", required=True, choices=list(loaders))
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--h5-path", default="data/raw/ASCAD.h5")
    ap.add_argument("--byte", type=int, default=2)
    ap.add_argument("--leakage-model", default="ID")
    ap.add_argument("--scaler", default="standardize")
    ap.add_argument("--n-attack", type=int, default=10000)
    ap.add_argument("--max-attack-traces", type=int, default=2000)
    ap.add_argument("--n-experiments", type=int, default=100)
    ap.add_argument("--out-dir", type=Path, default=Path("results/_adhoc"))
    args = ap.parse_args(argv)

    data = load_ascad(args.h5_path, n_profiling=None, n_attack=args.n_attack)
    scaler = TraceScaler(args.scaler).fit(data.profiling.traces)
    model = loaders[args.model].load(args.ckpt)

    out = evaluate_model(
        model, data.attack, scaler, args.byte, args.leakage_model,
        out_dir=args.out_dir, label=args.model,
        max_attack_traces=args.max_attack_traces, n_experiments=args.n_experiments,
    )
    print(out.key_rank.summary())
    print(f"wrote {out.csv_path}\nwrote {out.plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_standalone())
