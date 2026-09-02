"""Overlay the Key Rank curves of several runs on one axis (the paper's figure).

    python -m src.compare_runs results/<run_a> results/<run_b> ... \
        --out results/comparison.png --title "ASCAD fixed-key, byte 2"

Each run dir must contain a key_rank.csv (n_traces,mean_rank,p10,p90) as written
by src/evaluate.py. Labels default to the model name parsed from the dir.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_NAME_RE = re.compile(r"\d{8}-\d{6}_(?P<model>[a-z_]+?)_byte(?P<byte>\d+)")


def _load_curve(run_dir: Path):
    csv_path = run_dir / "key_rank.csv"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    n, mean, p10, p90 = [], [], [], []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            n.append(int(row["n_traces"]))
            mean.append(float(row["mean_rank"]))
            p10.append(float(row["p10"]))
            p90.append(float(row["p90"]))
    return n, mean, p10, p90


def _label_for(run_dir: Path) -> str:
    m = _NAME_RE.search(run_dir.name)
    return m.group("model") if m else run_dir.name


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("run_dirs", nargs="+", type=Path)
    ap.add_argument("--labels", nargs="*", default=None, help="one label per run dir")
    ap.add_argument("--out", type=Path, default=Path("results/comparison.png"))
    ap.add_argument("--title", default="Key Rank vs. number of attack traces")
    ap.add_argument("--band", action="store_true", help="shade the 10-90 percentile band")
    args = ap.parse_args(argv)

    labels = args.labels or [_label_for(p) for p in args.run_dirs]
    if len(labels) != len(args.run_dirs):
        ap.error("need exactly one --labels entry per run dir")

    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=140)

    # reference lines first so the model curves draw on top of them
    ax.axhline(127.5, color="#c0392b", lw=1.4, ls="--", alpha=0.9,
               label="random guess (E[rank] = 127.5)")
    ax.axhline(0, color="k", lw=1.0, ls="-", alpha=0.5, label="key recovered (rank 0)")

    for run_dir, label in zip(args.run_dirs, labels):
        n, mean, p10, p90 = _load_curve(run_dir)
        line, = ax.plot(n, mean, linewidth=2.0, label=label, zorder=5)
        if args.band:
            ax.fill_between(n, p10, p90, color=line.get_color(), alpha=0.12, zorder=1)

    ax.set_xlabel("Number of attack traces")
    ax.set_ylabel("Mean key rank of true byte  (0 = recovered)")
    ax.set_title(args.title)
    ax.set_ylim(-6, 160)
    ax.grid(alpha=0.25)
    ax.legend(loc="center right", framealpha=0.92)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    plt.close(fig)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
