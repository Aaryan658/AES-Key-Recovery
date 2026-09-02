"""Key Rank / Guessing Entropy - the metric that actually matters here.

Raw class-accuracy on ASCAD is ~0.4-1% (chance is 1/256 = 0.39%); a model can
look "useless" by accuracy yet still recover the key in a few hundred traces,
because the tiny per-trace bias accumulates. The standard SCA metric captures
this:

For a set of ``N`` attack traces, and for every key-byte hypothesis
``g in 0..255``:

    score(g) = sum_i  log( p_i[ Sbox(plaintext_i XOR g) ] )

where ``p_i`` is the model's 256-way softmax output for attack trace ``i``.
Sort hypotheses by descending score; the **rank** of the true key byte is how
many hypotheses beat it (0 == key fully recovered).

``guessing_entropy`` repeats this over many random orderings / subsets of the
attack traces and averages the rank, as a function of the number of traces
used - this is the curve the paper plots.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.aes import key_hypothesis_labels

_EPS = 1e-40


@dataclass
class KeyRankResult:
    n_traces: np.ndarray          # (T,) x-axis: number of attack traces used
    mean_rank: np.ndarray         # (T,) guessing entropy averaged over experiments
    all_ranks: np.ndarray         # (n_experiments, T) raw ranks
    true_key_byte: int
    target_byte: int
    n_experiments: int

    @property
    def traces_to_rank0(self) -> "int | None":
        """Smallest trace count at which the *mean* rank first reaches 0 (and stays)."""
        below = np.where(self.mean_rank <= 0.5)[0]
        if below.size == 0:
            return None
        for idx in below:
            if np.all(self.mean_rank[idx:] <= 0.5):
                return int(self.n_traces[idx])
        return None

    def summary(self) -> str:
        ttr = self.traces_to_rank0
        tail = self.mean_rank[-1]
        got = f"rank 0 after {ttr} traces" if ttr is not None else f"final mean rank {tail:.1f}"
        return (
            f"KeyRank(byte={self.target_byte}, true=0x{self.true_key_byte:02x}, "
            f"{self.n_experiments} experiments): {got}"
        )


def _per_trace_logprob_by_hypothesis(
    proba: np.ndarray,
    plaintext_bytes: np.ndarray,
    leakage_model: str,
) -> np.ndarray:
    """(N, 256) matrix: log p_i under each key hypothesis for each attack trace.

    ``out[i, g] = log( proba[i, Sbox(plaintext_i XOR g)] )``
    """
    proba = np.asarray(proba, dtype=np.float64)
    logp = np.log(np.clip(proba, _EPS, None))                    # (N, C)
    hyp = key_hypothesis_labels(plaintext_bytes, leakage_model)  # (N, 256) class idx
    rows = np.arange(hyp.shape[0])[:, None]
    return logp[rows, hyp]                                       # (N, 256)


def guessing_entropy(
    proba: np.ndarray,
    plaintext_bytes: np.ndarray,
    true_key_byte: int,
    target_byte: int = 2,
    leakage_model: str = "ID",
    max_traces: "int | None" = None,
    n_experiments: int = 100,
    step: int = 1,
    rng: "np.random.Generator | int | None" = 0,
) -> KeyRankResult:
    """Average key rank vs. number of attack traces.

    Parameters
    ----------
    proba : (N, C) model class probabilities for the attack traces.
    plaintext_bytes : (N,) target plaintext byte per attack trace.
    true_key_byte : the real value of the attacked key byte (0..255).
    max_traces : cap on the x-axis (defaults to N).
    n_experiments : number of random trace-order permutations to average over.
    step : evaluate the rank every ``step`` accumulated traces.
    """
    if not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(rng)

    logp_hyp = _per_trace_logprob_by_hypothesis(proba, plaintext_bytes, leakage_model)
    n_total = logp_hyp.shape[0]
    max_traces = n_total if max_traces is None else min(max_traces, n_total)

    idx_axis = np.arange(step - 1, max_traces, step)
    if idx_axis.size == 0 or idx_axis[-1] != max_traces - 1:
        idx_axis = np.append(idx_axis, max_traces - 1)
    n_axis = idx_axis + 1  # convert 0-based positions to trace counts

    all_ranks = np.empty((n_experiments, idx_axis.size), dtype=np.int32)
    for e in range(n_experiments):
        perm = rng.permutation(n_total)[:max_traces]
        cum = np.cumsum(logp_hyp[perm], axis=0)            # (max_traces, 256)
        snap = cum[idx_axis]                                # (T, 256)
        # rank = number of hypotheses with strictly greater accumulated score
        true_scores = snap[:, true_key_byte][:, None]
        all_ranks[e] = np.sum(snap > true_scores, axis=1)

    return KeyRankResult(
        n_traces=n_axis,
        mean_rank=all_ranks.mean(axis=0),
        all_ranks=all_ranks,
        true_key_byte=int(true_key_byte),
        target_byte=int(target_byte),
        n_experiments=n_experiments,
    )
