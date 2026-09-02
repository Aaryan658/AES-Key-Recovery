# Improving the 1D ResNet for ASCAD variable-key SCA — Research Report

*Generated 2026-09-01 | Sources: 12 | Confidence: High for the diagnosis and the
Karayalçin recipe; Medium for exact trace-count targets (they vary by run/seed).*

## Executive summary

Our 1D ResNet stalls on `ascad-variable.h5`: training loss sits at `ln(256) ≈
5.545` and never drops. The literature is unanimous that this **initial plateau
is the normal, expected behaviour of any gradient-trained model on a masked
dataset** — not a bug — and that it is broken by (a) training *far* longer than
we did (the loss drop typically happens around **epoch 30**; our probes were
killed at epochs 4–11), and (b) using the specific architecture/optimizer recipe
that the ResNet-for-SCA literature converged on. Our ResNet also diverges from
that recipe in three concrete ways that each independently hurt: **kernel size 3
instead of 11**, **ReLU instead of SELU**, and **too few residual blocks** (we
used 6; the guideline for 1400 samples is 8). The single most reliable
plateau-breaker in the literature is **multi-task learning** (jointly predict the
mask and the S-box output), which makes convergence consistent where single-task
training "converges around epoch 20, 30, or 70, or not at all."

---

## 1. The plateau is expected, and it is beaten mostly by *time*

Every serious ASCAD reproduction reports the same thing.

- Berreby & Sauvage, reproducing the original ASCAD networks: *"even with the
  authors' original code, the categorical cross-entropy loss goes from 5.5451 …
  at epoch 5, to 5.5448 … at epoch 20. After prolonging training runs, we found
  that the original network usually experienced **a sharp drop in the loss around
  the 30th epoch**, in spite of the learning rate remaining unchanged."*
  ([Investigating Efficient DL Architectures for SCA on AES](https://arxiv.org/html/2309.13170))
- Masure et al. (via Marquet & Oswald's summary): the plateau happens because
  *"no single point in the trace gives up information about the target … This
  leads to very weak feedback from back-propagation … the complexity of passing
  the plateau is **exponential with the number of shares** … does not come from
  the choice of hyperparameters but rather **the number of steps needed by
  gradient descent**."*
  ([Exploring multi-task learning …](https://eprint.iacr.org/2023/006))
- NeurIPS 2025 interpretability work: for the ASCAD ID model there is *"a sudden
  transition to positive perceived information from epochs 8–12 … improvement
  stops at epoch 25."*
  ([Interpreting Emergent Features in DL-based SCA](https://papers.nips.cc/paper_files/paper/2025/file/878147474808c35add04cf4852d2c996-Paper-Conference.pdf))
- Rousselot et al. (TCHES 2025, *Scoop*): the plateau corresponds to *"a
  prominent saddle point"* in the loss landscape and *"the magnitudes of the
  gradients decrease as the order of masking increases."*
  ([Scoop](https://tches.iacr.org/index.php/TCHES/article/view/12210))

**Implication for us:** our longest ResNet probe reached epoch 11. We very likely
never gave it the chance to break the plateau. Minimum viable change: **100
epochs, batch size 50**, patience measured in *tens* of epochs, and judge by
validation guessing entropy, not by whether the loss moved in the first 10
epochs.

---

## 2. The reference ResNet recipe (Karayalçin, Perin, Picek)

[*"Resolving the Doubts: On the Construction and Use of ResNets for
Side-Channel Analysis"*, Mathematics 11(15):3265, 2023](https://doi.org/10.3390/math11153265)
([eprint 2022/963](https://eprint.iacr.org/2022/963)) is the definitive study.
Their ASCAD-r (variable-key, 1400-sample) configuration:

| Element | Value |
|---|---|
| Residual block | **2 conv layers + one pooling layer** that halves the feature-map length |
| Kernel size | **11** |
| Filters, block *i* | grow as powers of two, **capped at 256** |
| Activation | **SELU** |
| Optimizer | **Adam** with a **cyclic learning-rate schedule** (Smith 2017 triangular) |
| Loss | categorical cross-entropy |
| Epochs / batch | **100 / 50** |
| # residual blocks | **⌊log₂(n_features)⌋ − 2** → for 1400 samples = **8 blocks** |
| Classifier head | 2 FC layers of **200 units** for ASCAD-r (vs 10 for fixed-key — the extra profiling data lets the head be wider without overfitting) |
| Model selection | best validation loss, **averaged over 5 training runs** |

Findings that matter for us:

- *"adding residual blocks … improves the attacking performance up to a point.
  From 4 to 7 residual blocks the best models steadily improve … 7 to 9 is
  negligible."* → **go deep (7–8 blocks), not 3.**
- *"ResNets work especially well when the number of profiling traces and features
  in a trace is large."* → the variable-key set (200k traces, 1400 samples) is
  the *ideal* case for a ResNet; if anything it should be easier than fixed-key.
- Reported ASCAD-r result: **~34–37 attack traces to GE = 1** on the standard
  1400-sample window (fewer on larger windows), with a ResNet of ~490k
  parameters.

The paper we're replicating (Poudel & Rahimi) uses a *simpler* variant that also
works for them: **4 residual blocks, kernel 11, filters 64/128/256/512,
`AvgPool1d(2)` after each block, then Flatten → Dense(4096) → Dense(4096) →
Dense(256)**, ReLU. Their `ResidualBlock` is `conv(k,pad=k//2)→BN→ReLU→conv(k)→BN`
with a `1×1` conv shortcut when channels change, `out += res; relu(out)`. Their
public code:
[ThorOdinson246/AES-Key-Recovery-using-Machine-Learning](https://github.com/ThorOdinson246/AES-Key-Recovery-using-Machine-Learning)
(`cnn_resnet_unified.py`). They report **key recovery in ~20–30 traces** on
ASCAD-v after ~110 on fixed-key.

---

## 3. How our ResNet differs from both references (ranked by likely impact)

| # | Ours | Reference | Why it matters |
|---|---|---|---|
| 1 | **kernel_size = 3** | 11 (both papers) | receptive field per block is tiny; with few blocks the network can't span the leak. Karayalçin only drop to k=3 for the *noisier hardware* AES_HD set. |
| 2 | trained ≤ 11 epochs | 100 epochs, batch 50 | plateau breaks ~epoch 30; we never got there. |
| 3 | **ReLU** | SELU (+ lecun-normal init) | our *working* CNN uses SELU; ReLU units can die on the near-zero-gradient plateau. |
| 4 | 3 stages / 6 blocks | 8 residual blocks for 1400 samples | "deeper is better up to ~7"; we're well below. |
| 5 | one-cycle, `max_lr/25 ≈ 2e-5` warmup | triangular cyclic LR, or LR-range test first | *"minute changes in learning rate … sometimes prevent convergence altogether."* The tiny warmup LR starves a 1.7 M-param net. |
| 6 | `weight_decay` 1e-4–5e-4 | ~none / small | competes with the already-tiny warmup LR early on. |
| 7 | tried global-avg-pool head | flatten + FC(200) | global average pooling over ~87 positions dilutes a leak localised to a few samples. |

Items 1–4 are architecture; 5–7 are training. **Fixing kernel size, activation,
depth, and epoch budget together is the high-probability path.**

---

## 4. Training-methodology upgrades (independent of architecture)

1. **Batch normalisation in the conv blocks *and* the classifier head.** Berreby
   & Sauvage: their JAX reimplementation did *not* show the epoch-30 loss drop
   *"until we added batch normalization within convolutional blocks and in the
   final classifier."* (Ours already has BN in blocks; add it in the FC head.)

2. **Run a learning-rate range test**, then a cyclic or exponentially-decayed
   cosine schedule. Berreby & Sauvage found an *exponentially-decayed cosine*
   (period = ⅕ of training, half-life = ½) gave *"~50 % lower training and test
   loss … and accordingly lower guessing entropy"* than one-cycle. They also warn
   that plain cosine's peaks *"hurt the network's performance."*

3. **Keep selecting on validation guessing entropy** (already implemented). We
   independently hit the exact phenomenon the literature documents: *"networks
   can keep improving with regards to guessing entropy while the validation loss
   goes up and the training loss goes down."* Karayalçin used best-val-loss and
   still succeeded (SELU + cyclic LR keeps val loss better-behaved), but val-GE
   is strictly safer. A [Guessing-Entropy-based training framework](https://dl.acm.org/doi/10.1109/TIFS.2023.3273169)
   (IEEE TIFS 2023) shows selecting/optimising on GE beats NLL outright.

4. **Train 3–5 seeds and keep the best.** Convergence is partly luck of
   initialisation (Wu et al.; Marquet & Oswald: single-task *"converges around
   epoch 20, 30, or 70, or not at all"*). Every ResNet-SCA paper reports
   best-of-5.

5. **Multi-task learning — the most reliable plateau-breaker.** Marquet & Oswald
   ([eprint 2023/006](https://eprint.iacr.org/2023/006)): a shared feature trunk
   with separate heads predicting the **mask `r`**, the **masked S-box output**,
   and the **unmasked S-box output** (optionally several key bytes) *"breaks
   through the initial plateau more consistently"* — multi-task models
   *"converge consistently under 35 epochs"* vs single-task's 20/30/70/never. The
   ASCAD metadata we already load (`masks`, `plaintext`, `key`) has everything
   needed to build these auxiliary labels. This is a larger code change but is
   the best-evidenced fix if the architecture fixes alone are inconsistent.

6. **Advanced optimisers (optional, research-grade).** *Scoop* (second-order +
   sparse mirror descent, [TCHES 2025](https://tches.iacr.org/index.php/TCHES/article/view/12210))
   is purpose-built to escape the plateau saddle point and is SOTA on ASCADv1.
   *CNN-Mamba* ([Li et al., 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12386059/))
   reports GE = 1 in < 100 traces within 25 epochs.

---

## 5. One caveat on evaluation

Rousselot et al., *"Is it Really Broken?"* ([eprint 2026/338](https://eprint.iacr.org/2026/338)):
a converging guessing entropy can be a **false positive** when the model latches
onto a bias in the intermediate-value distribution rather than real leakage. For
ASCAD-v1 with the ID model and roughly uniform labels this risk is low, but a
cheap control is worth adding: **train once on shuffled labels** and confirm GE
stays flat at ~128.

---

## Key takeaways — recommended plan

**Step 1 — realign the architecture with the literature** (edit
`configs/ascad_variable.yaml` / `src/models/resnet.py`):

- `kernel_size: 11` (was 3)
- `activation: selu` with lecun-normal init (add SELU support to the ResNet head,
  mirroring `src/models/cnn.py`)
- `block_filters: [16, 32, 64, 128, 256, 256, 256, 256]` → 8 residual blocks,
  each halving the length (⌊log₂ 1400⌋ − 2)
- head: `head_pool: flatten`, `fc_units: [200]`, add BN in the head
- `epochs: 100`, `batch_size: 50`
- LR: run an LR-range test; start with a triangular cyclic schedule, `base_lr
  1e-4`, `max_lr 5e-3` (Zaid/Karayalçin range), `weight_decay 0`

**Step 2 — run it long and don't kill it early.** Judge only by `val_GE` at
epochs 30/50/75/100. Run 3 seeds.

**Step 3 — if still inconsistent, add multi-task heads** (mask `r`, masked
S-box output `Sbox(p⊕k)⊕r`, unmasked `Sbox(p⊕k)`) on a shared trunk. Highest
evidence for consistent convergence.

**Step 4 — sanity control:** one shuffled-label run must leave GE ≈ 128.

Expected outcome if Steps 1–2 succeed: GE → 0 on `ascad-variable` in the low tens
of attack traces, matching Karayalçin (~35) and Poudel & Rahimi (~20–30).

---

## Sources

1. [Investigating Efficient Deep Learning Architectures For Side-Channel Attacks on AES](https://arxiv.org/html/2309.13170) — Berreby & Sauvage. Plateau reproduction, epoch-30 loss drop, BN requirement, LR sensitivity, cosine-vs-onecycle, CE-loss critique.
2. [Resolving the Doubts: On the Construction and Use of ResNets for Side-Channel Analysis](https://doi.org/10.3390/math11153265) — Karayalçin, Perin, Picek (Mathematics 2023). The reference ResNet-SCA recipe and depth guideline; ASCAD-r ~35 traces to GE=1.
3. [eprint 2022/963](https://eprint.iacr.org/2022/963) — preprint of (2), with block-construction ablation and per-window trace counts.
4. [Machine Learning-Based AES Key Recovery via Side-Channel Analysis on the ASCAD Dataset](https://arxiv.org/html/2508.11817v1) — Poudel & Rahimi (the paper being replicated). Their ResNet: 4 blocks, k=11, filters 64–512, AvgPool, flatten→4096→4096. ~20–30 traces on ASCAD-v.
5. [ThorOdinson246/AES-Key-Recovery-using-Machine-Learning](https://github.com/ThorOdinson246/AES-Key-Recovery-using-Machine-Learning) — the paper's companion code; exact `ResidualBlock` / `ResNet_SCA_CNN` PyTorch classes.
6. [Exploring multi-task learning in the context of two masked AES implementations](https://eprint.iacr.org/2023/006) — Marquet & Oswald. Plateau theory (Masure); multi-task learning breaks the plateau consistently (<35 epochs vs 20/30/70/never).
7. [A Comprehensive Study of Deep Learning for Side-Channel Analysis](https://eprint.iacr.org/2019/439.pdf) — Masure, Dumas, Prouff. NLL ≈ Perceived Information; early-epoch instability while PI ≈ 0.
8. [Interpreting Emergent Features in Deep Learning-based SCA](https://papers.nips.cc/paper_files/paper/2025/file/878147474808c35add04cf4852d2c996-Paper-Conference.pdf) — NeurIPS 2025. ASCAD ID phase transition at epochs 8–25; learn simple leakage first, then build on it.
9. [Scoop: An Optimization Algorithm for Profiling Attacks against Higher-Order Masking](https://tches.iacr.org/index.php/TCHES/article/view/12210) — Rousselot et al., TCHES 2025. Plateau = saddle point + vanishing gradients; second-order + sparse mirror descent.
10. [A Guessing Entropy-Based Framework for Deep Learning-Assisted Side-Channel Analysis](https://dl.acm.org/doi/10.1109/TIFS.2023.3273169) — IEEE TIFS 2023. Train/select on GE (via GEEA) rather than NLL.
11. [Is it Really Broken? The Failure of DL-SCA Scoring Metrics under Non-Uniform Priors](https://eprint.iacr.org/2026/338) — Rousselot et al. 2026. GE can be a false positive from label-distribution bias; use a shuffled-label control.
12. [Efficient AES Side-Channel Attacks Based on Residual Mamba](https://pmc.ncbi.nlm.nih.gov/articles/PMC12386059/) — Li et al. 2025. Residual-Mamba + MLP; GE=1 in <100 traces, <25 epochs.

## Methodology

Four Exa web searches (ResNet-for-ASCAD architectures; DL-SCA training plateau /
stuck loss; Karayalçin hyperparameters; plateau-escape epoch counts). ~12 unique
sources analysed, prioritising IACR ePrint / TCHES / peer-reviewed venues and the
replicated paper's own code. Sub-questions: (a) what ResNet designs work on
ASCAD-v, (b) why deep nets stall at `ln(256)` and how that is fixed, (c) exact
reference hyperparameters, (d) training-methodology upgrades, (e) evaluation
pitfalls.
