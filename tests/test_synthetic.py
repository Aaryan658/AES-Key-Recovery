"""End-to-end pipeline test on synthetic leakage - no ASCAD download needed.

Builds fake traces where exactly one sample point carries
``HW(Sbox(plaintext ^ key)) + N(0, sigma)`` and the rest is noise, then checks
that the whole chain (label mapping -> model -> guessing entropy) drives the
true key byte to rank 0. This exercises every module used by milestone 1.

Run:  pytest -q            or      python -m tests.test_synthetic
"""

from __future__ import annotations

import numpy as np

from src.aes import AES_SBOX, hamming_weight
from src.data_loader import TraceSet
from src.key_rank import guessing_entropy
from src.models import build_model
from src.preprocessing import TraceScaler, build_labels


def _make_traceset(n, n_samples, key_byte, sigma, rng, leak_at=17):
    plaintext = rng.integers(0, 256, size=(n, 16), dtype=np.uint8)
    key = np.full((n, 16), 0, dtype=np.uint8)
    key[:, 2] = key_byte
    v = AES_SBOX[np.bitwise_xor(plaintext[:, 2], key[:, 2])]
    leak = hamming_weight(v).astype(np.float32)

    traces = rng.normal(0, 1.0, size=(n, n_samples)).astype(np.float32)
    traces[:, leak_at] += (leak - leak.mean()) / (leak.std() + 1e-9) * (1.0 / sigma)
    stored = v.astype(np.uint8)  # byte-2 ID labels, matches build_labels cross-check
    return TraceSet(
        traces=traces, plaintext=plaintext, key=key,
        masks=None, desync=None, stored_labels=stored,
    )


def run(n_prof=6000, n_att=1500, n_samples=40, key_byte=0xA7, sigma=1.5, seed=0):
    rng = np.random.default_rng(seed)
    prof = _make_traceset(n_prof, n_samples, key_byte, sigma, rng)
    att = _make_traceset(n_att, n_samples, key_byte, sigma, rng)

    scaler = TraceScaler("standardize").fit(prof.traces)
    x_prof = scaler.transform(prof.traces)
    y_prof = build_labels(prof, target_byte=2, leakage_model="ID")
    x_att = scaler.transform(att.traces)

    # held-out slice for the CNN's validation-guessing-entropy checkpoint metric
    n_val = 1500
    val_ge = {
        "x": x_prof[:n_val],
        "plaintext_byte": prof.plaintext_byte(2)[:n_val],
        "true_key_byte": key_byte,
        "target_byte": 2, "leakage_model": "ID",
        "max_traces": n_val, "n_experiments": 20, "step": 50,
    }

    results = {}
    for name, kw in [
        ("random_forest", dict(n_estimators=120, min_samples_leaf=3)),
        ("cnn", dict(epochs=20, batch_size=128, lr=1e-3, max_lr=1e-3,
                     lr_schedule="onecycle", conv_filters=[16, 32],
                     kernel_size=5, fc_units=[128], activation="relu",
                     ge_eval_every=2, early_stopping_patience=40)),
    ]:
        model = build_model(name, n_classes=256, input_length=n_samples, device="cpu", **kw)
        if name == "cnn":
            model.fit(x_prof[n_val:], y_prof[n_val:], seed=seed,
                      x_val=x_prof[:n_val], y_val=y_prof[:n_val], val_ge=val_ge)
        else:
            model.fit(x_prof, y_prof, seed=seed)
        proba = model.predict_proba(x_att)
        ge = guessing_entropy(
            proba, att.plaintext_byte(2), true_key_byte=key_byte,
            target_byte=2, max_traces=n_att, n_experiments=30, step=25, rng=seed,
        )
        results[name] = ge
        print(f"{name:14s} final mean rank = {ge.mean_rank[-1]:.2f}  "
              f"traces_to_rank0 = {ge.traces_to_rank0}")
    return results


def test_pipeline_recovers_key():
    results = run()
    # RF nails the easy synthetic signal outright.
    assert results["random_forest"].mean_rank[-1] == 0, "RF should reach rank 0"
    # The CNN path (train -> val-GE checkpoint -> predict_proba -> key rank) must
    # carry real signal, i.e. far below the 128 chance rank. This guards against
    # the "restore near-untrained weights" regression in torch_common.fit().
    assert results["cnn"].mean_rank[-1] <= 100, (
        f"CNN pipeline produced ~chance key rank ({results['cnn'].mean_rank[-1]}); "
        "model selection may be restoring untrained weights (regression guard, "
        "not a performance benchmark -- real ASCAD convergence is checked by a "
        "full run, not this fast synthetic test)"
    )


if __name__ == "__main__":
    run()
    print("OK")
