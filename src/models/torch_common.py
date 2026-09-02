"""Shared PyTorch plumbing for the CNN and ResNet models.

Keeps a single training loop, device selection, early stopping and softmax
inference in one place so the model files only describe architecture.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.base import SCAModel, TrainHistory


def pick_device(prefer: str = "auto") -> torch.device:
    if prefer not in ("auto", "cpu", "cuda"):
        raise ValueError(prefer)
    if prefer == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available() and prefer in ("auto", "cuda"):
        return torch.device("cuda")
    return torch.device("cpu")


def _as_sequence(x: np.ndarray) -> torch.Tensor:
    """(N, L) float32 -> (N, 1, L) tensor for 1D convolutions."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 2:
        x = x[:, None, :]
    return torch.from_numpy(np.ascontiguousarray(x))


class TorchSCAModel(SCAModel):
    """Base for torch models: subclasses only implement ``build_network``."""

    wants_sequence = True

    def __init__(self, n_classes: int = 256, input_length: int = 700, device: str = "auto", **hparams):
        super().__init__(n_classes, **hparams)
        self.input_length = input_length
        self.device = pick_device(device)
        self.net = None

    # --- to be provided by subclasses -----------------------------------
    def build_network(self) -> nn.Module:
        raise NotImplementedError

    # --- multi-task hooks --------------------------------------------------
    # Single-task defaults; MultiTaskResNetModel overrides these so the shared
    # training loop below stays model-agnostic. ``out`` is whatever the network
    # returns (a logits tensor, or a tuple of them); ``yb`` matches the y passed
    # to ``fit`` (int64 (B,) for single-task, int64 (B, K) for multi-task).
    def _compute_loss(self, out, yb, loss_fn) -> torch.Tensor:
        return loss_fn(out, yb)

    def _num_correct(self, out, yb) -> torch.Tensor:
        return (out.detach().argmax(1) == yb).sum()

    def _proba_logits(self, out) -> torch.Tensor:
        """The logits to softmax for predict_proba / val-GE (key-recovery head)."""
        return out

    # --- training -----------------------------------------------------------
    def fit(self, x: np.ndarray, y: np.ndarray, *, seed: int = 0,
            x_val: "np.ndarray | None" = None, y_val: "np.ndarray | None" = None,
            val_ge: "dict | None" = None) -> "TorchSCAModel":
        """Train the network.

        Model selection for side-channel analysis does NOT use validation
        cross-entropy loss: on masked datasets it is flat/rising for the first
        ~15-20 epochs *while the attack is still improving*, so picking the
        lowest-val-loss epoch returns a near-untrained network. Instead:

        * ``val_ge`` given  -> select the epoch with the best (lowest) validation
          *guessing entropy* on a held-out attack set with the same key. This is
          the correct SCA early-stopping metric.
        * ``val_ge`` absent -> keep the FINAL-epoch weights (what ASCAD/Zaid do).

        ``checkpoint_metric`` hparam ("val_ge" | "val_acc" | "none") overrides
        the default. ``early_stopping_patience`` only applies to val_ge/val_acc.

        ``val_ge`` dict: ``{plaintext_byte: (Nv,), true_key_byte: int,
        target_byte: int, leakage_model: str, max_traces: int, n_experiments: int}``
        """
        torch.manual_seed(seed)
        np.random.seed(seed)
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True  # fixed batch shape -> autotune convs

        hp = self.hparams
        epochs = int(hp.get("epochs", 50))
        batch = int(hp.get("batch_size", 128))
        lr = float(hp.get("lr", 1e-4))
        wd = float(hp.get("weight_decay", 0.0))
        patience = int(hp.get("early_stopping_patience", 20))
        ge_every = int(hp.get("ge_eval_every", 5))

        self.net = self.build_network().to(self.device)

        opt_name = hp.get("optimizer", "adam").lower()
        params = self.net.parameters()
        if opt_name == "adam":
            optim = torch.optim.Adam(params, lr=lr, weight_decay=wd)
        elif opt_name in ("rmsprop", "rms"):
            optim = torch.optim.RMSprop(params, lr=lr, weight_decay=wd)
        elif opt_name == "adamw":
            optim = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
        else:
            optim = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=wd)
        loss_fn = nn.CrossEntropyLoss()

        xt = _as_sequence(x)
        yt = torch.from_numpy(np.asarray(y, dtype=np.int64))
        n_train = xt.shape[0]
        steps_per_epoch = math.ceil(n_train / batch)

        # For a small model on a large trace set the num_workers=0 DataLoader is
        # the bottleneck (GPU sits at ~30% util). If the sequence tensor fits in
        # a modest VRAM budget, keep it resident on the GPU and index batches
        # directly - typically a 3-5x speed-up for the ASCAD ResNet/CNN.
        gpu_resident = (
            self.device.type == "cuda"
            and hp.get("gpu_resident", True)
            and xt.element_size() * xt.nelement() < int(hp.get("gpu_resident_max_bytes", 3_000_000_000))
        )
        if gpu_resident:
            xt = xt.to(self.device)
            yt = yt.to(self.device)
            print(f"  training set resident on {self.device} "
                  f"({xt.element_size() * xt.nelement() / 1e6:.0f} MB)", flush=True)
        else:
            dl = DataLoader(
                TensorDataset(xt, yt), batch_size=batch, shuffle=True, drop_last=False,
            )

        sched = None
        sched_per_batch = False
        sname = hp.get("lr_schedule", "none")
        if sname == "cosine":
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)
        elif sname == "onecycle":
            # Zaid et al. use one-cycle for the compact ASCAD CNN; it is what
            # takes GE to 0 in a few hundred traces rather than plateauing.
            sched = torch.optim.lr_scheduler.OneCycleLR(
                optim, max_lr=float(hp.get("max_lr", lr)),
                steps_per_epoch=steps_per_epoch, epochs=epochs,
                pct_start=float(hp.get("onecycle_pct_start", 0.3)),
            )
            sched_per_batch = True
        elif sname == "cyclic":
            # Smith (2017) triangular cyclic LR - what Zaid et al. and Karayalcin
            # et al. use for ASCAD CNN/ResNet. cycle_momentum must be off for Adam.
            half = int(hp.get("cyclic_step_size_epochs", 4)) * steps_per_epoch
            sched = torch.optim.lr_scheduler.CyclicLR(
                optim, base_lr=float(hp.get("base_lr", lr)),
                max_lr=float(hp.get("max_lr", 5e-3)),
                step_size_up=half, mode=hp.get("cyclic_mode", "triangular2"),
                cycle_momentum=False,
            )
            sched_per_batch = True

        has_val = x_val is not None and y_val is not None
        if has_val:
            xv = _as_sequence(x_val).to(self.device)
            yv = torch.from_numpy(np.asarray(y_val, dtype=np.int64)).to(self.device)

        metric = hp.get("checkpoint_metric", "val_ge" if val_ge is not None else "none")
        if metric == "val_ge" and val_ge is None:
            metric = "none"
        if metric == "val_acc" and not has_val:
            metric = "none"

        self.history = TrainHistory()
        best_score = float("inf")   # lower = better (GE) ; for val_acc we negate
        best_state = None
        best_epoch = 0
        bad_epochs = 0
        t0 = time.time()

        for ep in range(1, epochs + 1):
            self.net.train()
            # accumulate on-device; sync once per epoch (a per-batch .item() on a
            # tiny model makes training host-bound)
            run_loss = torch.zeros((), device=self.device)
            run_correct = torch.zeros((), device=self.device)
            run_n = 0
            if gpu_resident:
                perm = torch.randperm(n_train, device=self.device)
                batches = (
                    (xt[perm[i:i + batch]], yt[perm[i:i + batch]])
                    for i in range(0, n_train, batch)
                )
            else:
                batches = ((xb.to(self.device), yb.to(self.device)) for xb, yb in dl)
            for xb, yb in batches:
                optim.zero_grad()
                out = self.net(xb)
                loss = self._compute_loss(out, yb, loss_fn)
                loss.backward()
                optim.step()
                if sched_per_batch:
                    sched.step()
                run_loss += loss.detach() * xb.size(0)
                run_correct += self._num_correct(out, yb)
                run_n += xb.size(0)
            run_loss = run_loss.item()
            run_correct = run_correct.item()
            if sched is not None and not sched_per_batch:
                sched.step()
            tr_loss = run_loss / run_n
            tr_acc = run_correct / run_n
            self.history.train_loss.append(tr_loss)
            self.history.train_acc.append(tr_acc)

            msg = f"  epoch {ep:3d}/{epochs}  train_loss={tr_loss:.4f} acc={tr_acc:.4f}"
            if has_val:
                self.net.eval()
                with torch.no_grad():
                    vo = self.net(xv)
                    v_loss = self._compute_loss(vo, yv, loss_fn).item()
                    v_acc = self._num_correct(vo, yv).item() / len(yv)
                self.history.val_loss.append(v_loss)
                self.history.val_acc.append(v_acc)
                msg += f"  val_loss={v_loss:.4f} acc={v_acc:.4f}"

            do_ge = metric == "val_ge" and (ep % ge_every == 0 or ep == epochs)
            if do_ge:
                v_ge = self._val_guessing_entropy(val_ge)
                self.history.extra.setdefault("val_ge_epochs", []).append((ep, v_ge))
                msg += f"  val_GE={v_ge:.1f}"
                score = v_ge
            elif metric == "val_acc":
                score = -v_acc
            else:
                score = None

            if score is not None:
                if score < best_score - 1e-6:
                    best_score, best_state, best_epoch, bad_epochs = score, self._cpu_state(), ep, 0
                else:
                    bad_epochs += (ge_every if do_ge else 1)
            print(msg, flush=True)

            if metric in ("val_ge", "val_acc") and bad_epochs >= patience:
                print(f"  early stop at epoch {ep} ({metric} no improvement for ~{patience})",
                      flush=True)
                break

        if best_state is not None:
            print(f"  restoring best weights from epoch {best_epoch} "
                  f"({metric}={best_score if metric != 'val_acc' else -best_score:.3f})", flush=True)
            self.net.load_state_dict(best_state)
        else:
            print("  keeping final-epoch weights (no SCA checkpoint metric)", flush=True)
        self.history.extra["train_seconds"] = time.time() - t0
        self.history.extra["epochs_run"] = len(self.history.train_loss)
        self.history.extra["selected_epoch"] = best_epoch or len(self.history.train_loss)
        return self

    @torch.no_grad()
    def _val_guessing_entropy(self, vg: dict) -> float:
        """Mean final key rank on the held-out validation attack set."""
        from src.key_rank import guessing_entropy

        self.net.eval()
        xs = _as_sequence(vg["x"])
        out = []
        for i in range(0, len(xs), 4096):
            logits = self._proba_logits(self.net(xs[i:i + 4096].to(self.device)))
            out.append(torch.softmax(logits, 1).cpu().numpy())
        proba = np.concatenate(out).astype(np.float64)
        ge = guessing_entropy(
            proba, vg["plaintext_byte"], vg["true_key_byte"],
            target_byte=vg.get("target_byte", 2), leakage_model=vg.get("leakage_model", "ID"),
            max_traces=vg.get("max_traces", 2000), n_experiments=vg.get("n_experiments", 30),
            step=vg.get("step", 100), rng=0,
        )
        return float(ge.mean_rank[-1])

    def _cpu_state(self):
        return {k: v.detach().cpu().clone() for k, v in self.net.state_dict().items()}

    # --- inference --------------------------------------------------------
    @torch.no_grad()
    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self.net is None:
            raise RuntimeError("model not fitted")
        self.net.eval()
        out = []
        xs = _as_sequence(x)
        bs = int(self.hparams.get("infer_batch_size", 2048))
        for i in range(0, len(xs), bs):
            xb = xs[i : i + bs].to(self.device)
            logits = self._proba_logits(self.net(xb))
            out.append(torch.softmax(logits, dim=1).cpu().numpy())
        return np.concatenate(out, axis=0).astype(np.float64)

    # --- persistence ----------------------------------------------------
    def save(self, path: "str | Path") -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "class": type(self).__name__,
                "state_dict": self._cpu_state(),
                "n_classes": self.n_classes,
                "input_length": self.input_length,
                "hparams": self.hparams,
                "history": self.history.as_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path: "str | Path") -> "TorchSCAModel":
        blob = torch.load(path, map_location="cpu", weights_only=False)
        obj = cls(n_classes=blob["n_classes"], input_length=blob["input_length"], **blob["hparams"])
        obj.net = obj.build_network()
        obj.net.load_state_dict(blob["state_dict"])
        obj.net.to(obj.device)
        return obj
