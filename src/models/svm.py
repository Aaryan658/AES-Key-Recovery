"""SVM baseline (scikit-learn) - the paper's second baseline.

Pipeline: StandardScaler -> PCA (whiten) -> RBF SVC with probability=True.

Deviations from a naive reading of the paper (documented in README):
  * RBF-SVM training is O(n^2 .. n^3). The full 50 000-trace profiling set with
    256 classes is impractical, so we subsample to ``max_train_samples`` (10 000
    by default) with a class-stratified draw.
  * ``probability=True`` uses Platt scaling (internal 5-fold CV) so we get the
    calibrated per-class scores the key-rank metric needs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.models.base import SCAModel


class SVMModel(SCAModel):
    name = "svm"
    wants_sequence = False

    def __init__(self, n_classes: int = 256, **hparams):
        super().__init__(n_classes, **hparams)
        self._pipe = None

    def _subsample(self, x, y, seed):
        cap = self.hparams.get("max_train_samples", 10000)
        if cap is None or cap >= len(x):
            return x, y
        rng = np.random.default_rng(seed)
        # class-stratified: take an equal-ish share per present class
        idx = []
        classes, _counts = np.unique(y, return_counts=True)
        per = max(1, cap // len(classes))
        for c in classes:
            ci = np.where(y == c)[0]
            take = min(per, ci.size)
            idx.append(rng.choice(ci, size=take, replace=False))
        idx = np.concatenate(idx)
        rng.shuffle(idx)
        return x[idx], y[idx]

    def fit(self, x: np.ndarray, y: np.ndarray, *, seed: int = 0) -> "SVMModel":
        from sklearn.decomposition import PCA
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC

        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y)
        x, y = self._subsample(x, y, seed)

        hp = self.hparams
        self._pipe = Pipeline(
            [
                ("scale", StandardScaler()),
                ("pca", PCA(n_components=hp.get("pca_components", 50), whiten=True, random_state=seed)),
                (
                    "svc",
                    SVC(
                        kernel=hp.get("kernel", "rbf"),
                        C=hp.get("C", 10.0),
                        gamma=hp.get("gamma", "scale"),
                        probability=True,
                        class_weight=hp.get("class_weight", None),
                        random_state=seed,
                        cache_size=hp.get("cache_size", 1000),
                    ),
                ),
            ]
        )
        self._pipe.fit(x, y)
        self.history.extra["n_train_used"] = int(len(x))
        self.history.extra["train_acc_final"] = float(self._pipe.score(x, y))
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self._pipe is None:
            raise RuntimeError("model not fitted")
        proba = self._pipe.predict_proba(np.asarray(x, dtype=np.float32))
        svc = self._pipe.named_steps["svc"]
        if proba.shape[1] != self.n_classes:
            full = np.zeros((proba.shape[0], self.n_classes), dtype=np.float64)
            full[:, svc.classes_.astype(int)] = proba
            proba = full
        return proba

    def save(self, path: "str | Path") -> None:
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"pipe": self._pipe, "n_classes": self.n_classes, "hparams": self.hparams}, path)

    @classmethod
    def load(cls, path: "str | Path") -> "SVMModel":
        import joblib

        blob = joblib.load(path)
        obj = cls(n_classes=blob["n_classes"], **blob["hparams"])
        obj._pipe = blob["pipe"]
        return obj
