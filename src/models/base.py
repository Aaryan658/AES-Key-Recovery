"""Common interface every model in this project implements.

The key-rank evaluation only needs one thing from a model: a calibrated
``predict_proba(X) -> (N, n_classes)``. Everything else (how it trains, whether
it is sklearn or torch) is hidden behind this base class so
``src/run_experiment.py`` and ``src/evaluate.py`` stay model-agnostic.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class TrainHistory:
    """Per-epoch training log (deep models). Empty for sklearn models."""

    train_loss: list = field(default_factory=list)
    val_loss: list = field(default_factory=list)
    train_acc: list = field(default_factory=list)
    val_acc: list = field(default_factory=list)
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "train_acc": self.train_acc,
            "val_acc": self.val_acc,
            **self.extra,
        }


class SCAModel(abc.ABC):
    """Abstract side-channel model."""

    name: str = "base"
    #: does this model consume a (N, 1, L) sequence (True) or a flat (N, L) matrix?
    wants_sequence: bool = False

    def __init__(self, n_classes: int = 256, **hparams):
        self.n_classes = n_classes
        self.hparams = hparams
        self.history = TrainHistory()

    @abc.abstractmethod
    def fit(self, x: np.ndarray, y: np.ndarray, *, seed: int = 0) -> "SCAModel":
        ...

    @abc.abstractmethod
    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Return an (N, n_classes) array of class probabilities (rows sum to 1)."""
        ...

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(x), axis=1)

    def score(self, x: np.ndarray, y: np.ndarray) -> float:
        """Plain top-1 accuracy (reported for completeness; not the real metric)."""
        return float(np.mean(self.predict(x) == np.asarray(y)))

    @abc.abstractmethod
    def save(self, path: "str | Path") -> None:
        ...

    @classmethod
    @abc.abstractmethod
    def load(cls, path: "str | Path") -> "SCAModel":
        ...
