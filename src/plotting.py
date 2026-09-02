"""Plots: the Key Rank curve (paper's main result) and training curves."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.key_rank import KeyRankResult  # noqa: E402


def plot_key_rank(
    results: "KeyRankResult | dict",
    out_path: "str | Path",
    title: "str | None" = None,
    logy: bool = False,
) -> Path:
    """Key rank (guessing entropy) vs number of attack traces.

    ``results`` is a single KeyRankResult or a ``{label: KeyRankResult}`` dict so
    several models can be overlaid on one axis (the paper's comparison figure).
    """
    if isinstance(results, KeyRankResult):
        results = {"model": results}

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=130)
    for label, res in results.items():
        ax.plot(res.n_traces, res.mean_rank, label=f"{label} (GE)", linewidth=1.8)
        lo = np.percentile(res.all_ranks, 10, axis=0)
        hi = np.percentile(res.all_ranks, 90, axis=0)
        ax.fill_between(res.n_traces, lo, hi, alpha=0.15)

    ax.axhline(0, color="k", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_xlabel("Number of attack traces")
    ax.set_ylabel("Key rank of true byte  (0 = recovered)")
    ax.set_title(title or "Key Rank vs. traces")
    if logy:
        ax.set_yscale("symlog")
    ax.set_ylim(bottom=-2)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_training_history(history: dict, out_path: "str | Path", title: "str | None" = None):
    """Loss/accuracy vs epoch. Returns None for models with no epoch log (sklearn)."""
    tl = history.get("train_loss") or []
    if not tl:
        return None
    vl = history.get("val_loss") or []
    ta = history.get("train_acc") or []
    va = history.get("val_acc") or []
    epochs = range(1, len(tl) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=130)
    axes[0].plot(epochs, tl, label="train")
    if vl:
        axes[0].plot(range(1, len(vl) + 1), vl, label="val")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("cross-entropy loss")
    axes[0].set_title("Loss"); axes[0].grid(alpha=0.25); axes[0].legend()

    axes[1].plot(epochs, ta, label="train")
    if va:
        axes[1].plot(range(1, len(va) + 1), va, label="val")
    axes[1].axhline(1 / 256, color="k", ls="--", alpha=0.5, label="chance (1/256)")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("top-1 accuracy")
    axes[1].set_title("Accuracy"); axes[1].grid(alpha=0.25); axes[1].legend()

    fig.suptitle(title or "Training history")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
