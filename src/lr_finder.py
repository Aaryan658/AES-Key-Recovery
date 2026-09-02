"""Learning-rate range test (Smith 2017 / fastai style).

Trains a fresh model for a short run while increasing the LR exponentially from
``start_lr`` to ``end_lr`` per batch, records the (smoothed) loss, and suggests:
  * lr_min_grad  - LR at the steepest loss decrease (a good `max_lr`)
  * lr_div10     - lr_min_grad / 10 (a conservative constant LR / one-cycle init)

ASCAD deep models are extremely LR-sensitive (Berreby & Sauvage: "minute changes
in learning rate ... sometimes prevent convergence altogether"), so run this
before committing to a schedule.

    python -m src.lr_finder --model resnet --config configs/ascad_variable.yaml \
        --n-profiling 40000 --iters 400
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from src.data_loader import load_ascad  # noqa: E402
from src.models import build_model  # noqa: E402
from src.models.torch_common import _as_sequence, pick_device  # noqa: E402
from src.preprocessing import prepare_xy  # noqa: E402


def lr_range_test(model, x, y, *, start_lr=1e-7, end_lr=1.0, iters=400,
                  batch_size=128, beta=0.98, device="auto", seed=0):
    torch.manual_seed(seed)
    dev = pick_device(device)
    net = model.build_network().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=start_lr)
    loss_fn = nn.CrossEntropyLoss()

    ds = TensorDataset(_as_sequence(x), torch.from_numpy(np.asarray(y, np.int64)))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    mult = (end_lr / start_lr) ** (1 / max(iters - 1, 1))
    lr = start_lr
    avg_loss = 0.0
    best = float("inf")
    lrs, losses = [], []

    net.train()
    it = iter(dl)
    for i in range(iters):
        try:
            xb, yb = next(it)
        except StopIteration:
            it = iter(dl)
            xb, yb = next(it)
        xb, yb = xb.to(dev), yb.to(dev)
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad()
        loss = loss_fn(net(xb), yb)
        loss.backward()
        opt.step()

        li = loss.item()
        avg_loss = beta * avg_loss + (1 - beta) * li
        smooth = avg_loss / (1 - beta ** (i + 1))
        lrs.append(lr)
        losses.append(smooth)
        if smooth < best:
            best = smooth
        if i > 10 and (smooth > 4 * best or not np.isfinite(smooth)):
            print(f"  loss diverged at iter {i}, lr={lr:.2e} - stopping")
            break
        lr *= mult

    lrs, losses = np.array(lrs), np.array(losses)
    grads = np.gradient(losses, np.log10(lrs))
    lo = max(3, int(0.1 * len(lrs)))
    hi = int(0.95 * len(lrs))
    j = lo + int(np.argmin(grads[lo:hi]))
    return {"lrs": lrs, "losses": losses, "lr_min_grad": float(lrs[j]),
            "lr_div10": float(lrs[j] / 10.0)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", required=True, choices=["cnn", "resnet"])
    ap.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    ap.add_argument("--byte", type=int, default=None)
    ap.add_argument("--n-profiling", type=int, default=40000)
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--start-lr", type=float, default=1e-7)
    ap.add_argument("--end-lr", type=float, default=1.0)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--out", type=Path, default=Path("results/lr_finder.png"))
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    ds = cfg["dataset"]
    byte = args.byte if args.byte is not None else int(ds["target_byte"])
    leak = ds.get("leakage_model", "ID")
    mcfg = dict(cfg.get(args.model, {}))
    for k in ("epochs", "lr", "lr_schedule", "max_lr", "base_lr", "optimizer",
              "early_stopping_patience", "ge_eval_every"):
        mcfg.pop(k, None)

    data = load_ascad(ds["h5_path"], args.n_profiling, 2000)
    x, yv, _xa, _sc = prepare_xy(data.profiling, data.attack, byte, leak,
                                 cfg.get("preprocess", {}).get("scaler", "standardize"))
    n_classes = 256 if leak.upper() == "ID" else 9
    model = build_model(args.model, n_classes=n_classes,
                        input_length=data.profiling.n_samples, device=args.device, **mcfg)

    print(f"LR range test: {args.model}, {len(x)} traces, {args.iters} iters, "
          f"{args.start_lr:.0e} -> {args.end_lr:.0e}")
    r = lr_range_test(model, x, yv, start_lr=args.start_lr, end_lr=args.end_lr,
                      iters=args.iters, batch_size=args.batch_size, device=args.device)

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=130)
    ax.plot(r["lrs"], r["losses"])
    ax.axvline(r["lr_min_grad"], color="r", ls="--", label=f"steepest: {r['lr_min_grad']:.1e}")
    ax.axvline(r["lr_div10"], color="g", ls=":", label=f"/10: {r['lr_div10']:.1e}")
    ax.set_xscale("log")
    ax.set_xlabel("learning rate")
    ax.set_ylabel("smoothed loss")
    ax.set_title(f"LR range test - {args.model}")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    plt.close(fig)

    print(f"\n  suggested max_lr (steepest slope): {r['lr_min_grad']:.2e}")
    print(f"  suggested base_lr / constant lr : {r['lr_div10']:.2e}")
    print(f"  plot -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
