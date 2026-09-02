"""Random Forest baseline (scikit-learn) - the paper's weak baseline.

Operates on flat (N, 700) standardised traces. No feature selection: the forest
does its own via ``max_features``. This is expected to be poor on ID-model
ASCAD (the masking hides first-order leakage from a bagged tree ensemble); it is
here as the floor to beat.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.models.base import SCAModel


class RandomForestModel(SCAModel):
    name = "random_forest"
    wants_sequence = False

    def __init__(self, n_classes: int = 256, **hparams):
        super().__init__(n_classes, **hparams)
        self._clf = None

    def fit(self, x: np.ndarray, y: np.ndarray, *, seed: int = 0) -> "RandomForestModel":
        from sklearn.ensemble import RandomForestClassifier

        hp = self.hparams
        self._clf = RandomForestClassifier(
            n_estimators=hp.get("n_estimators", 300),
            max_depth=hp.get("max_depth", None),
            max_features=hp.get("max_features", "sqrt"),
            min_samples_leaf=hp.get("min_samples_leaf", 5),
            class_weight=hp.get("class_weight", None),
            n_jobs=hp.get("n_jobs", -1),
            random_state=seed,
            verbose=hp.get("verbose", 0),
        )
        self._clf.fit(np.asarray(x, dtype=np.float32), np.asarray(y))
        self.history.extra["train_acc_final"] = float(
            self._clf.score(np.asarray(x, dtype=np.float32), np.asarray(y))
        )
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self._clf is None:
            raise RuntimeError("model not fitted")
        proba = self._clf.predict_proba(np.asarray(x, dtype=np.float32))
        # RF only emits columns for classes it saw; re-expand to full n_classes.
        if proba.shape[1] != self.n_classes:
            full = np.zeros((proba.shape[0], self.n_classes), dtype=np.float64)
            full[:, self._clf.classes_.astype(int)] = proba
            proba = full
        return proba

    def save(self, path: "str | Path") -> None:
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"clf": self._clf, "n_classes": self.n_classes, "hparams": self.hparams}, path)

    @classmethod
    def load(cls, path: "str | Path") -> "RandomForestModel":
        import joblib

        blob = joblib.load(path)
        obj = cls(n_classes=blob["n_classes"], **blob["hparams"])
        obj._clf = blob["clf"]
        return obj
