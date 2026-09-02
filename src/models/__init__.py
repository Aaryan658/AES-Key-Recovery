"""Model registry.

``build_model("cnn", n_classes=256, input_length=700, **hparams)`` returns an
unfitted :class:`~src.models.base.SCAModel`. ``run_experiment`` and
``evaluate`` only ever touch models through this factory + the base interface.
"""

from __future__ import annotations

from src.models.base import SCAModel
from src.models.cnn import CNNModel
from src.models.multitask_resnet import MultiTaskResNetModel
from src.models.random_forest import RandomForestModel
from src.models.resnet import ResNetModel
from src.models.svm import SVMModel

_REGISTRY = {
    "random_forest": RandomForestModel,
    "rf": RandomForestModel,
    "svm": SVMModel,
    "cnn": CNNModel,
    "resnet": ResNetModel,
    "multitask_resnet": MultiTaskResNetModel,
}

#: models that are part of milestone 1 (baseline review gate)
MILESTONE1_MODELS = ("random_forest", "cnn")

#: torch models need input_length / device kwargs; sklearn models ignore them
_TORCH_MODELS = {"cnn", "resnet", "multitask_resnet"}


def available_models() -> list:
    return sorted(set(_REGISTRY) - {"rf"})


def build_model(name: str, *, n_classes: int = 256, input_length: int = 700,
                device: str = "auto", **hparams) -> SCAModel:
    key = name.lower()
    if key not in _REGISTRY:
        raise KeyError(f"unknown model {name!r}; choose from {available_models()}")
    cls = _REGISTRY[key]
    if key in _TORCH_MODELS:
        return cls(n_classes=n_classes, input_length=input_length, device=device, **hparams)
    return cls(n_classes=n_classes, **hparams)


__all__ = [
    "SCAModel",
    "RandomForestModel",
    "SVMModel",
    "CNNModel",
    "ResNetModel",
    "MultiTaskResNetModel",
    "build_model",
    "available_models",
    "MILESTONE1_MODELS",
]
