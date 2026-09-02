"""Old vs rebuilt ResNet: training-curve comparison.

Reads per-epoch history from each run's config.json (model_history) or, for runs
that were stopped before writing config.json, parses run.log. Produces a 3-panel
figure: cross-entropy loss, validation guessing entropy, and top-1 accuracy vs
epoch, contrasting the old kernel-3 / ReLU / one-cycle ResNet with the rebuilt
kernel-11 / SELU / cyclic-LR ("Karayalcin-style") ResNet.

    python -m src.plot_resnet_evolution
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RESULTS = Path("results")
OUT = RESULTS / "resnet_old_vs_new_training.png"

_EPOCH_RE = re.compile(
    r"epoch\s+(\d+)/\d+\s+train_loss=([\d.]+)\s+acc=([\d.]+)"
    r"(?:\s+val_loss=([\d.]+)\s+acc=([\d.]+))?(?:\s+val_GE=([\d.]+))?"
)


def from_config(run_dir: Path):
    h = json.loads((run_dir / "config.json").read_text())["model_history"]
    ep = list(range(1, len(h["train_loss"]) + 1))
    ge_ep = [e for e, _ in h.get("val_ge_epochs", [])]
    ge_val = [g for _, g in h.get("val_ge_epochs", [])]
    return {
        "epoch": ep,
        "train_loss": h["train_loss"], "val_loss": h.get("val_loss") or [],
        "train_acc": h["train_acc"], "val_acc": h.get("val_acc") or [],
        "ge_epoch": ge_ep, "ge": ge_val,
    }


def from_log(run_dir: Path):
    ep, tl, ta, vl, va, ge_e, ge = [], [], [], [], [], [], []
    for line in (run_dir / "run.log").read_text().splitlines():
        m = _EPOCH_RE.search(line)
        if not m:
            continue
        ep.append(int(m.group(1)))
        tl.append(float(m.group(2)))
        ta.append(float(m.group(3)))
        if m.group(4):
            vl.append(float(m.group(4)))
            va.append(float(m.group(5)))
        if m.group(6):
            ge_e.append(int(m.group(1)))
            ge.append(float(m.group(6)))
    return {"epoch": ep, "train_loss": tl, "val_loss": vl, "train_acc": ta,
            "val_acc": va, "ge_epoch": ge_e, "ge": ge}


def load(run_dir: Path):
    return from_config(run_dir) if (run_dir / "config.json").exists() else from_log(run_dir)


# (run dir, label, colour, linestyle)
RUNS = [
    ("20260901-092247_resnet_byte2_m2_varkey", "OLD (k3/ReLU) - variable-key", "#c0392b", ":"),
    ("20260901-090337_resnet_byte2_m2_fixedkey", "OLD (k3/ReLU) - fixed-key", "#e67e22", "--"),
    ("20260901-122446_resnet_byte2_m2_vk_kara_s0", "NEW (k11/SELU) - variable-key", "#1f77b4", "-"),
    ("20260901-130001_resnet_byte2_m2_fixedkey_kara", "NEW (k11/SELU) - fixed-key", "#2ca02c", "-"),
]

fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), dpi=130)
for name, label, colour, ls in RUNS:
    rd = RESULTS / name
    if not rd.exists():
        print(f"skip missing {name}")
        continue
    d = load(rd)
    axes[0].plot(d["epoch"], d["train_loss"], color=colour, ls=ls, lw=1.8, label=label)
    if d["val_loss"]:
        axes[0].plot(d["epoch"][:len(d["val_loss"])], d["val_loss"], color=colour,
                     ls=ls, lw=1.0, alpha=0.45)
    if d["ge"]:
        axes[1].plot(d["ge_epoch"], d["ge"], color=colour, ls=ls, lw=1.8,
                     marker="o", ms=3, label=label)
    axes[2].plot(d["epoch"], d["train_acc"], color=colour, ls=ls, lw=1.8, label=label)
    if d["val_acc"]:
        axes[2].plot(d["epoch"][:len(d["val_acc"])], d["val_acc"], color=colour,
                     ls=ls, lw=1.0, alpha=0.45)

axes[0].axhline(5.545, color="k", ls="--", lw=0.8, alpha=0.5)
axes[0].text(2, 5.56, "ln(256) = initial plateau", fontsize=8, alpha=0.7)
axes[0].set_title("Cross-entropy loss (bold = train, faint = val)")
axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss"); axes[0].grid(alpha=0.25)
axes[0].legend(fontsize=8)

axes[1].axhline(0, color="k", lw=0.8, alpha=0.5)
axes[1].axhline(127.5, color="grey", ls=":", lw=0.9, alpha=0.7)
axes[1].set_title("Validation guessing entropy (0 = key recovered)")
axes[1].set_xlabel("epoch"); axes[1].set_ylabel("mean key rank"); axes[1].grid(alpha=0.25)
axes[1].legend(fontsize=8)

axes[2].axhline(1 / 256, color="k", ls="--", lw=0.8, alpha=0.5)
axes[2].text(2, 1 / 256 + 0.01, "chance (1/256)", fontsize=8, alpha=0.7)
axes[2].set_title("Top-1 accuracy (bold = train, faint = val)")
axes[2].set_xlabel("epoch"); axes[2].set_ylabel("accuracy"); axes[2].grid(alpha=0.25)
axes[2].legend(fontsize=8)

fig.suptitle("ResNet: old kernel-3/ReLU/one-cycle  vs  rebuilt kernel-11/SELU/cyclic-LR", fontsize=12)
fig.tight_layout()
OUT.parent.mkdir(exist_ok=True)
fig.savefig(OUT)
plt.close(fig)
print(f"wrote {OUT}")
