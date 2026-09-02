# AES Key Recovery via Side-Channel Analysis (ASCAD)

Replication of **Poudel & Rahimi, "Machine Learning-Based AES Key Recovery via
Side-Channel Analysis on the ASCAD Dataset"** (arXiv:2508.11817, 2025).

The task is framed as **256-class classification of the first-round AES S-box
output** `Sbox(plaintext_byte XOR key_byte)` from electromagnetic power traces,
followed by a **Key Rank / Guessing Entropy** evaluation that accumulates
log-likelihoods over many attack traces to recover one key byte.

> **What was recovered:** key byte **#2** of the AES-128 key
> `4d fb e0 f2 72 21 fe 10 a7 8d 4a dc 8e 49 04 69`, true value **`0xE0`**, from
> ~10,000 power traces plus the (public) plaintexts and *not* the key. See
> **[RESULTS.md](RESULTS.md)** for the plain-language version, or run
> `python -m src.demo`.

---

## Project layout

```
data/
  download_ascad.py      download ASCAD_data.zip and extract ASCAD.h5
  raw/                    ASCAD.h5 lands here (gitignored)
src/
  aes.py                 S-box table, leakage models, key-hypothesis matrices
  data_loader.py         read the ASCAD HDF5 -> profiling / attack TraceSets
  preprocessing.py       amplitude scaler (fit on profiling only) + label mapping
  key_rank.py            guessing-entropy / key-rank metric
  plotting.py            key-rank curve + training-history plots
  evaluate.py            fitted model -> key_rank.csv / .png / metrics.json
  run_experiment.py      >>> single CLI entrypoint <<<
  models/
    base.py              SCAModel interface (fit / predict_proba / save / load)
    random_forest.py     RandomForestClassifier   (baseline)
    svm.py               PCA + RBF-SVC            (baseline)
    cnn.py               1D CNN, ASCAD "CNN_best" family (PyTorch)
    resnet.py            1D ResNet, residual blocks (PyTorch) -- milestone 2
    torch_common.py      shared training loop / device / checkpoint code
notebooks/
  01_explore_ascad.ipynb exploratory analysis (SNR, class balance, leakage)
results/                 one sub-dir per run (gitignored except .gitkeep)
tests/
  test_synthetic.py      full pipeline on synthetic leakage (no download)
configs/default.yaml     all hyperparameters; every field is CLI-overridable
```

---

## Setup

Developed on Python 3.14, Windows 11, PyTorch 2.13.0+cu126 (CUDA 12.6),
NVIDIA RTX 4060. CUDA is optional -- the CNN also trains on CPU (slower).

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows;  source .venv/bin/activate on *nix
pip install -r requirements.txt
```

If `pip install torch` gives you a CPU-only build and you have a CUDA GPU, follow
the selector at <https://pytorch.org/get-started/locally/> for the matching
`+cuXXX` wheel. **Do not** install a CUDA toolkit system-wide just for this.

Quick check that everything imports and the metric works (no dataset needed,
~90 s):

```bash
python -m pytest -q
```

---

## Stage 1 -- Data acquisition

```bash
python -m data.download_ascad
```

Downloads `ASCAD_data.zip` (~4.4 GB) from the official data.gouv.fr mirror,
extracts **only** `ASCAD.h5` (~250 MB) into `data/raw/`, verifies it opens, and
prints its structure. The download is resumable -- rerun if it drops.

Options:

| flag | meaning |
|---|---|
| `--h5-url <URL>` | download a standalone `ASCAD.h5` mirror instead of the 4.4 GB zip |
| `--zip-path X.zip` | use an archive you already downloaded |
| `--from-h5 X.h5` | you already have `ASCAD.h5`; just copy + verify it |
| `--with-desync` | also extract `ASCAD_desync50/100.h5` (for later robustness tests) |
| `--keep-zip` | keep the archive after extraction |

`ASCAD.h5` structure:

```
/Profiling_traces/traces     (50000, 700) int8
/Profiling_traces/labels     (50000,)     uint8    = Sbox(p[2]^k[2])  [byte 2, ID]
/Profiling_traces/metadata   (50000,)     {plaintext[16] u8, key[16] u8,
                                           masks[16] u8, desync i4}
/Attack_traces/...            (10000, ...)  same layout
```

> **Note on "700".** The ASCAD fixed-key set has **50 000 + 10 000 traces of 700
> samples each** -- the 700 is the trace length (a window around the masked S-box
> op), not a trace count.

Sanity-check the file directly:

```bash
python -m src.data_loader --h5-path data/raw/ASCAD.h5 --n-profiling 2000 --n-attack 2000
```

---

## Stage 2 -- Preprocessing (automatic inside a run)

* **Amplitude scaling** (`preprocess.scaler`): `standardize` (default), `minmax`,
  or `none`. Fitted on the **profiling set only** and applied unchanged to the
  attack set.
* **Label mapping** (`src/preprocessing.build_labels`): for `target_byte`,
  `label[i] = Sbox(plaintext[i, byte] XOR key[i, byte])` (ID model, 256 classes;
  `HW` model = 9 classes also available). On byte 2 / ID the derived labels are
  cross-checked against ASCAD's stored `labels` and must match exactly.
* The **attack set gets no single label** -- every one of the 256 key guesses
  implies a different label, which is what the key-rank step enumerates.

---

## Stage 3 -- Train + evaluate a model

```bash
python -m src.run_experiment --model random_forest        # weak baseline
python -m src.run_experiment --model cnn                   # 1D CNN
python -m src.run_experiment --model svm                   # PCA + RBF-SVC
python -m src.run_experiment --model resnet                # milestone 2
```

Useful overrides (all optional, they layer on top of `configs/default.yaml`):

```bash
python -m src.run_experiment --model cnn --epochs 30 --lr 5e-5 --byte 2 \
    --n-profiling 50000 --max-attack-traces 3000 --n-experiments 100 \
    --device cuda --tag baseline
```

Each run writes `results/<UTC-timestamp>_<model>_byte<byte>[_<tag>]/`:

| file | contents |
|---|---|
| `config.json` | fully resolved config, CLI args, env, **runtime seconds** |
| `run.log` | full stdout/stderr transcript |
| `model.pt` / `model.joblib` | the fitted model |
| `training_history.png` | loss & accuracy vs epoch (deep models) |
| `key_rank.csv` | `n_traces, mean_rank, p10, p90` |
| `key_rank.png` | **Key Rank vs. #traces -- the paper's main figure** |
| `key_rank_metrics.json` | top-1 acc, chance, `traces_to_rank0`, final rank |

Re-evaluate a saved model without retraining:

```bash
python -m src.evaluate --model cnn --ckpt results/<run>/model.pt --byte 2
```

---

## Evaluation -- why not accuracy

Top-1 accuracy on ASCAD ID-model is ~0.4-1 % (chance = 1/256 = 0.39 %). A model
can look useless by accuracy yet still recover the key, because a tiny per-trace
bias accumulates. **Key Rank / Guessing Entropy** is the real metric:

for `N` attack traces and every key hypothesis `g in 0..255`,

```
score(g) = sum_i  log P_model( class = Sbox(plaintext_i XOR g) | trace_i )
```

rank the true key byte among the 256 sorted scores; **rank 0 = recovered**.
`src/key_rank.py` averages this over `n_experiments` random attack-trace
orderings and reports rank as a function of the number of traces used.

### The same logic applies to deep-model *training*

Validation cross-entropy loss is **not** a valid model-selection metric for
masked SCA: for the first ~20 epochs it is flat or rising (the net gets a little
more confident before it gets discriminative) *while the attack is already
improving*. Selecting the lowest-val-loss epoch returns a near-untrained
network. `src/models/torch_common.py` therefore selects the checkpoint by
**validation guessing entropy** on a held-out slice of the profiling set (same
fixed key), computed every `ge_eval_every` epochs. With `arch: zaid` +
`lr_schedule: onecycle` the validation GE reaches 0 around epoch 25.

Overlay several runs' Key Rank curves (the paper's comparison figure):

```bash
python -m src.compare_runs results/<rf_run> results/<cnn_run> \
    --labels "Random Forest" "CNN" --band --out results/comparison.png
```

---

## Deviations from the paper (and why)

| # | Paper / naive reading | Here | Reason |
|---|---|---|---|
| 1 | "700 traces" | 700 **samples** per trace; 50k/10k traces | ASCAD fixed-key dataset definition |
| 2 | target key byte unspecified | default `--byte 2` | ASCAD only masks bytes 2-15; byte 2 is the community standard |
| 3 | SVM on full profiling set | subsample to 10k (`svm.max_train_samples`), PCA-50 first | RBF-SVM is O(n^2)*256-class; full set is impractical |
| 4 | CNN = ASCAD `CNN_best` (60M-param VGG, Keras, SELU) | default `arch: zaid` -- the compact ~17k-param ASCAD CNN (Zaid et al., CHES 2020), one-cycle LR. `arch: vgg` still available. | the VGG head memorises masked ASCAD and never ranks the key without heavy tuning; the compact net converges reliably |
| 5 | model selection on val loss / accuracy | selection on validation **guessing entropy** | val CE loss is anti-correlated with attack success early in training (see Evaluation section) |
| 6 | ResNet architecture not fully specified | basic (non-bottleneck) 1D blocks, 3 stages | short inputs (<=1400 samples): depth > width |
| 7 | -- | milestone gate: RF + CNN first, then review | requested in the brief |
| 8 | "recover the AES key" | recover key **byte 2** only | the extracted ASCAD `.h5` is a trace window around byte 2's masked S-box; bytes 3-15 leak outside it (milestone 3b) |

---

## Milestones

1. **DONE** -- RF + CNN end-to-end on ATMEGA_AES_v1 fixed-key, byte 2
   (true = `0xe0`), Key Rank plot. Results:

   | model | attack top-1 acc | key rank @ 2000 traces | traces to rank 0 |
   |---|---|---|---|
   | Random Forest (300 trees) | 0.46 % (chance 0.39 %) | ~63 | did not converge |
   | CNN (zaid, 1D, one-cycle) | 0.67 % | **0** | ~1360 (mean); band at 0 by ~700 |

   Plot: `results/milestone1_keyrank_rf_vs_cnn.png`. RF is the weak baseline
   (slow, non-converging); the CNN recovers the byte. *Stopped here for review.*
2. **DONE (1 seed)** -- ResNet + variable-key set (`ascad-variable.h5`,
   200k/100k traces, 1400 samples). Run with `--config configs/ascad_variable.yaml`.

   | run | result |
   |---|---|
   | CNN (compact), variable-key, byte 2 (true `0x22`) | **key rank 0 after ~516 traces**, attack acc 0.88% |
   | ResNet (Karayalcin-style), variable-key, byte 2 | **key rank 0 after ~276 traces**, attack acc 0.89%; plateau broke at epoch 15 |
   | ResNet (Karayalcin-style), fixed-key, byte 2 (true `0xe0`) | **key rank 0 after ~868 traces**, attack acc 0.73%; val-GE checkpoint caught epoch 25 before the net overfit (train acc later hits 1.0) |
   | CNN (compact), fixed-key, byte 2 | key rank 0 after ~1362 traces (milestone 1) |

   Plots: `results/milestone2_varkey_cnn_vs_resnet.png`,
   `results/milestone2_fixedkey_rf_cnn_resnet.png`,
   `results/milestone2_cnn_fixed_vs_variable.png`.

   The rebuilt ResNet recovers the key on **both** datasets, and more
   efficiently on the harder variable-key set (276 vs 868 traces) -- the paper's
   qualitative point. (The old kernel-3 / ReLU ResNet overfit fixed-key to
   chance and would not train on variable-key.)

   On variable-key the **ResNet beats the CNN** (276 vs 516 traces), matching the
   paper's qualitative finding that ResNets suit the harder dataset -- though our
   trace counts are ~10x the best in the literature (Karayalcin ~35, paper
   ~20-30), so there is tuning headroom (more seeds, depth, LR).

   Getting the ResNet to train at all required the fixes in
   `docs/resnet_improvement_research.md`: kernel 11 + SELU + `floor(log2(len))`-scaled
   depth + flatten head (was kernel 3 / ReLU / 6 blocks / global-avg-pool),
   triangular cyclic LR, ~80 epochs judged by validation guessing entropy (the
   masked-SCA "initial plateau" only broke at epoch 15), and a clean GPU (an
   earlier run was OOM-thrashing against another process).

   New tooling: `python -m src.lr_finder --model resnet --config <cfg>` (LR range
   test); GPU-resident training in `torch_common.py` (~5x faster for small
   models on large trace sets).
3. **DONE** -- the five milestone-3 threads. Runs are tagged `*_m3_*` under
   `results/`; the venv is now `.venv/` (see Setup).

   **a. SVM baseline** (`--model svm`, fixed-key byte 2): subsample 10k + PCA-50 +
   RBF, 233 s. Attack top-1 acc 0.42 % (chance 0.39 %), **final mean key rank
   137.8 -- no recovery**. A weak baseline, like RF; only the CNN / ResNet
   family ranks the key.

   **b. Multi-byte / full-key recovery -- data-limited.** The extracted ASCAD
   databases (`ASCAD.h5`, `ascad-variable.h5`) are a trace *window* around key
   byte 2's masked S-box, so a per-byte attack only works for byte 2. Confirmed:
   a CNN on byte 3 (fixed rank ~169, variable ~chance) and byte 4 (fixed rank
   162) never leaves chance. True 16-byte recovery needs the ~60 GB raw
   `ATMega8515_raw_traces.h5` + a per-byte re-window (out of scope). The
   assembly step is still shipped end-to-end:

   ```bash
   python -m src.assemble_key --h5-path data/raw/ASCAD.h5 --scan results/ --model cnn
   ```

   `results/m3_fullkey_fixed.json`: byte 2 recovered (`0xe0`, rank 0 by ~1364
   traces), bytes 3-4 flagged at chance, true key
   `4dfbe0f27221fe10a78d4adc8e490469`.

   **c. Multi-task ResNet** (`--model multitask_resnet`, per
   `docs/resnet_improvement_research.md`): shared ResNet trunk + 3 joint heads
   `y = Sbox(p^k)` / `r = r_out` (`masks[:,15]`) / `yr = y^r`. Only the `y` head
   recovers the key; `r`/`yr` just keep back-prop's gradient non-flat through the
   masked-SCA plateau.

   | run | traces to rank 0 | vs single-task |
   |---|---|---|
   | multi-task ResNet, fixed-key byte 2 | **~270** | CNN 1362, ResNet 868 -- **3x better** |
   | multi-task ResNet, variable-key byte 2 (40 ep) | ~3374 | ResNet 276 -- **worse** |

   On fixed-key the auxiliary heads drive validation GE to 0 by **epoch 10**
   (single-task needs ~25) -- the Marquet & Oswald plateau-break, reproduced. On
   variable-key the plateau is not the bottleneck and the extra heads add noise;
   the compact single-task ResNet stays ahead there.

   **d. Desync robustness** (compact CNN, fixed key, `--h5-path
   data/raw/ASCAD_desync{50,100}.h5`):

   | model | clean | desync50 | desync100 |
   |---|---|---|---|
   | compact CNN, final mean key rank | **0** (@1362) | 109 | 176 |
   | ResNet, final mean key rank | 0 (@868) | 155 | -- |

   Neither model has shift-invariance; +-50-sample jitter drops attack accuracy
   to chance and the key never ranks. Recovering desynchronised traces needs
   shift augmentation or an alignment pre-step -- not attempted here.

   **e. Extra ResNet seed(s)** (variable-key byte 2): seed 0 = **276** traces
   (milestone 2), seed 1 = **1540** traces. A 5x spread across two seeds -- the
   276 figure was partly lucky, and any comparison to the literature's ~35-trace
   mark needs a proper multi-seed average, not a single run. Deferred.

   **f. Shuffled-label control** (`--shuffle-labels`, CNN variable-key): profiling
   labels randomly permuted. Attack top-1 acc **0.388 % = exactly 1/256**, key
   rank stays 91-128 and never reaches 0 -- the pipeline is not exploiting a
   label-distribution artefact (Rousselot et al. 2026 sanity check).

   Plots: `results/milestone3_fixedkey_multitask.png` (multi-task wins),
   `results/milestone3_varkey_multitask.png` (multi-task loses),
   `results/milestone3_desync_cnn.png` (desync collapse).

   New tooling: `src/assemble_key.py`, `src/run_experiment.py --shuffle-labels`,
   the `multitask_resnet` model + config blocks, and a project `.venv/`.
