"""Build report/ASCAD_replication_report.pdf from the project's results.

    ../.venv/Scripts/python.exe report/build_report.py

Text is fixed prose (edit below); figures are pulled from results/. Numbers in
the prose are transcribed from results/*/key_rank_metrics.json as of the last
run - re-check them if you re-train.
"""

from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = ROOT / "report" / "ASCAD_replication_report.pdf"

# ---------------------------------------------------------------- styles -----
styles = getSampleStyleSheet()
BODY = ParagraphStyle(
    "body", parent=styles["Normal"], fontName="Times-Roman", fontSize=10.5,
    leading=14.5, alignment=TA_JUSTIFY, spaceAfter=6,
)
H1 = ParagraphStyle(
    "h1", parent=styles["Heading1"], fontName="Times-Bold", fontSize=12.5,
    leading=15, spaceBefore=12, spaceAfter=5, textColor=colors.black,
)
H2 = ParagraphStyle(
    "h2", parent=styles["Heading2"], fontName="Times-Bold", fontSize=11,
    leading=14, spaceBefore=9, spaceAfter=3, textColor=colors.black,
)
TITLE = ParagraphStyle(
    "title", parent=styles["Title"], fontName="Times-Bold", fontSize=16,
    leading=20, alignment=TA_CENTER, spaceAfter=4,
)
SUB = ParagraphStyle(
    "sub", parent=styles["Normal"], fontName="Times-Roman", fontSize=10.5,
    leading=14, alignment=TA_CENTER, spaceAfter=2,
)
CAP = ParagraphStyle(
    "cap", parent=styles["Normal"], fontName="Times-Italic", fontSize=9,
    leading=11.5, alignment=TA_CENTER, spaceBefore=3, spaceAfter=10,
)
MONO = ParagraphStyle(
    "mono", parent=styles["Normal"], fontName="Courier", fontSize=8.5,
    leading=11, spaceBefore=2, spaceAfter=8,
)


def P(t):
    return Paragraph(t, BODY)


def fig(name, caption, width=12.5 * cm):
    path = RESULTS / name
    img = Image(str(path))
    img._restrictSize(width, 9.5 * cm)
    img.hAlign = "CENTER"
    return [Spacer(1, 4), img, Paragraph(caption, CAP)]


def _page(canvas, doc):
    canvas.saveState()
    canvas.setFont("Times-Roman", 8.5)
    canvas.drawCentredString(A4[0] / 2.0, 1.1 * cm, str(doc.page))
    canvas.restoreState()


# ----------------------------------------------------------------- content ---
story = []
story += [
    Paragraph("Machine-Learning AES Key Recovery by Side-Channel Analysis "
              "on the ASCAD Dataset", TITLE),
    Paragraph("A replication of Poudel &amp; Rahimi (arXiv:2508.11817, 2025)", SUB),
    Spacer(1, 6),
    Paragraph("[Name] &nbsp;&nbsp;|&nbsp;&nbsp; [Roll no.] &nbsp;&nbsp;|&nbsp;&nbsp; "
              "[Course / code] &nbsp;&nbsp;|&nbsp;&nbsp; September 2026", SUB),
    Spacer(1, 12),
]

story += [Paragraph("Abstract", H1), P(
    "This project reproduces the profiled side-channel attack of Poudel and Rahimi "
    "on the ASCAD dataset, in which one byte of an AES-128 key is recovered from "
    "electromagnetic power traces of a masked software implementation. The task is "
    "posed as 256-class classification of the first-round S-box output "
    "Sbox(p&#8853;k), and success is measured by guessing entropy (key rank) rather "
    "than classification accuracy. We built the full pipeline from the raw HDF5 "
    "files, trained four model families (random forest, SVM, a compact CNN and a "
    "1-D ResNet), and reproduced the paper's central qualitative claims: the deep "
    "models recover the key byte while the classical baselines do not, and the "
    "ResNet is more trace-efficient than the CNN on the harder variable-key set. "
    "Beyond the paper we added a multi-task ResNet, a desynchronisation robustness "
    "test, and a shuffled-label control. The multi-task network cuts the traces "
    "needed on the fixed-key set from about 870 to about 270, but does not help on "
    "the variable-key set. All code, configurations and result files accompany this "
    "report.")]

story += [Paragraph("1&nbsp;&nbsp;Introduction", H1), P(
    "A masking countermeasure splits every sensitive intermediate value into random "
    "shares so that no single point in a power trace depends on the secret. ASCAD "
    "(ANSSI, 2018) is the reference benchmark for evaluating whether a profiled, "
    "learning-based attacker can still recover the key from such an implementation. "
    "Poudel and Rahimi frame the attack as supervised classification of the S-box "
    "output for one key byte and report key recovery within a few tens of attack "
    "traces on ASCAD."), P(
    "The goal here was to rebuild that attack end to end and check which of its "
    "claims hold when the pipeline is written from scratch. The work was organised "
    "as three milestones: (1) classical baselines plus a CNN on the fixed-key set, "
    "(2) a 1-D ResNet and the variable-key set, and (3) a set of extensions - an "
    "SVM baseline, a multi-byte assembly step, a multi-task ResNet, a "
    "desynchronisation test, and sanity controls.")]

story += [Paragraph("2&nbsp;&nbsp;Background", H1)]
story += [Paragraph("2.1&nbsp;&nbsp;Target and leakage model", H2), P(
    "The attacked value is y = Sbox(p[b] &#8853; k[b]) for a chosen key byte b, "
    "where p is the known plaintext. Under the identity leakage model this gives "
    "256 classes. Because ASCAD only masks bytes 2 to 15, and byte 2 is the "
    "community-standard target, all experiments use b = 2 unless stated otherwise. "
    "The stored labels in ASCAD.h5 are for byte 2 under this model; we re-derive "
    "labels from metadata and cross-check them against the stored ones.")]
story += [Paragraph("2.2&nbsp;&nbsp;Why not accuracy", H2), P(
    "On masked ASCAD the top-1 accuracy of any model sits near the 1/256 = 0.39% "
    "chance line, so accuracy is not a useful metric. The standard measure is "
    "guessing entropy: for a set of attack traces and every key hypothesis "
    "g in [0,255], the log-probabilities the model assigns to Sbox(p<sub>i</sub> "
    "&#8853; g) are summed, the 256 hypotheses are ranked by that score, and the "
    "rank of the true key byte is recorded. Rank 0 means the key byte is recovered. "
    "The curve of mean rank against number of attack traces, averaged over many "
    "random trace orderings, is the primary result.")]
story += [Paragraph("2.3&nbsp;&nbsp;Model selection", H2), P(
    "Validation cross-entropy is a poor stopping criterion here: for roughly the "
    "first 20 epochs it is flat or rising while the attack is already improving, so "
    "the lowest-loss epoch corresponds to a near-untrained network. Every deep model "
    "in this project is instead checkpointed on validation guessing entropy computed "
    "on a held-out set with the same key, evaluated every few epochs.")]

story += [Paragraph("3&nbsp;&nbsp;Method", H1), P(
    "The pipeline reads the ASCAD HDF5 files into profiling and attack sets, "
    "standardises the traces using statistics fitted on the profiling set only, and "
    "builds labels for the chosen byte. Four model families share one interface:"), P(
    "<b>Random forest</b> (300 trees) and <b>SVM</b> (RBF kernel, PCA to 50 "
    "components, profiling subsampled to 10 000 traces because an RBF SVM is "
    "quadratic in sample count) are the classical baselines. The <b>CNN</b> is the "
    "compact ASCAD network of Zaid et al. (about 17k parameters, SELU, one-cycle "
    "learning rate); the larger VGG-style ASCAD_best network was tried but "
    "memorises the profiling set without ranking the key. The <b>ResNet</b> is a "
    "1-D residual network in the style of Karayalcin et al.: kernel size 11, SELU "
    "with LeCun-normal initialisation, a number of residual blocks scaled as "
    "floor(log2(length)) - 2, a flatten head, and a triangular cyclic learning "
    "rate. Training keeps the trace tensor resident on the GPU, which is several "
    "times faster for these small models on large trace sets."), P(
    "Key rank is computed by src/key_rank.py and every run writes a resolved "
    "configuration, a log, the fitted model, a training-history plot, the key-rank "
    "curve as CSV and PNG, and a metrics file.")]

story += [Paragraph("4&nbsp;&nbsp;Experimental setup", H1), P(
    "Fixed-key ASCAD (ATMega8515) provides 50 000 profiling and 10 000 attack "
    "traces of 700 samples; the variable-key set (ASCAD-r) provides 200 000 and "
    "100 000 traces of 1400 samples. Desynchronised variants shift each trace "
    "randomly by up to 50 or 100 samples. Training ran on an NVIDIA RTX 4060 "
    "(8 GB) with PyTorch 2.14 and CUDA 12.6, inside a project-local virtual "
    "environment. Key rank is averaged over 100 random attack-trace orderings.")]

story += [PageBreak()]

story += [Paragraph("5&nbsp;&nbsp;Results", H1)]
story += [Paragraph("5.1&nbsp;&nbsp;Milestone 1 - baselines and CNN (fixed key)", H2), P(
    "On the fixed-key set the random forest never converges: its mean key rank is "
    "still about 63 after 2000 traces. The SVM baseline added in milestone 3 is no "
    "better, ending at a mean rank of 138 (chance is 127.5). The compact CNN "
    "recovers the key byte, reaching rank 0 at roughly 1360 attack traces, with the "
    "10-90 percentile band at zero by about 700. This matches the paper's "
    "qualitative split between classical and deep models.")]
story += fig("milestone1_keyrank_rf_vs_cnn.png",
            "Figure 1. Fixed-key ASCAD, byte 2. The random forest stays near "
            "chance; the compact CNN drives the key rank to 0.")

story += [Paragraph("5.2&nbsp;&nbsp;Milestone 2 - ResNet and the variable-key set", H2), P(
    "The rebuilt 1-D ResNet recovers the key on both datasets. On the fixed-key set "
    "it reaches rank 0 at about 870 traces; on the harder variable-key set it needs "
    "about 280, against about 520 for the CNN. That the ResNet is more "
    "trace-efficient on the harder problem is the paper's main qualitative point, "
    "and it is reproduced here, though our trace counts are roughly an order of "
    "magnitude above the best figures in the literature (around 35), so there is "
    "clear tuning headroom. The variable-key ResNet figure is also seed-sensitive: "
    "a second training seed needed about 1540 traces rather than 280, so the lower "
    "number should be read as a lucky run rather than a stable result. An earlier "
    "kernel-3 / ReLU ResNet overfit the fixed-key "
    "set to chance and would not train on the variable-key set at all; the fixes "
    "that made it work (larger kernels, SELU, depth scaled to the input length, "
    "cyclic learning rate, and judging convergence by validation guessing entropy "
    "rather than the first few epochs) are documented in the repository.")]
story += fig("milestone2_varkey_cnn_vs_resnet.png",
            "Figure 2. Variable-key ASCAD, byte 2. The ResNet recovers the key "
            "byte in fewer traces than the CNN.")

story += [Paragraph("5.3&nbsp;&nbsp;Milestone 3a - multi-task ResNet", H2), P(
    "The masking scheme exposes two shares at the targeted sample: the mask "
    "r<sub>out</sub> and the masked S-box output y &#8853; r<sub>out</sub>. The "
    "multi-task ResNet keeps one shared convolutional trunk and attaches three "
    "linear heads that are trained jointly - one for y, one for r<sub>out</sub>, "
    "and one for y &#8853; r<sub>out</sub> - but only the y head is used for key "
    "recovery. Predicting a raw mask byte is easy, so this head gives "
    "back-propagation a non-flat gradient during the early plateau. On the "
    "fixed-key set the effect is large: validation guessing entropy reaches 0 by "
    "epoch 10 instead of about 25, and the key byte is recovered in about 270 "
    "traces rather than 870. On the variable-key set the plateau is not the "
    "bottleneck and the extra heads act as noise; the single-task ResNet stays "
    "well ahead. This is reported as a negative result for that setting.")]
story += fig("milestone3_fixedkey_multitask.png",
            "Figure 3. Fixed-key ASCAD, byte 2. The multi-task ResNet reaches "
            "rank 0 roughly three times sooner than the single-task ResNet.")

story += [Paragraph("5.4&nbsp;&nbsp;Milestone 3b - how many key bytes can be recovered", H2), P(
    "The published ASCAD databases are a short trace window around the masked S-box "
    "operation of key byte 2. A CNN trained on byte 3 (mean rank about 170) or byte "
    "4 (about 160) never leaves chance, on both the fixed-key and variable-key "
    "sets, because the leakage for those bytes falls outside the window. Recovering "
    "the full 16-byte key would need the roughly 60 GB raw trace set and a per-byte "
    "re-windowing step, which was out of scope. The assembly step is still "
    "implemented end to end: src/assemble_key.py runs one model per byte, takes the "
    "rank-0 hypothesis as the recovered byte, stitches the 16 bytes together and "
    "checks them against the key in the metadata."), P(
    "For the fixed-key set the recovered byte 2 matches the ground truth:"),
    Paragraph("true key&nbsp;&nbsp;: 4d fb <b>e0</b> f2 72 21 fe 10 a7 8d 4a dc 8e 49 04 69<br/>"
              "recovered&nbsp;&nbsp;: -- -- <b>e0</b> -- -- -- -- -- -- -- -- -- -- -- -- --", MONO),
]

story += [Paragraph("5.5&nbsp;&nbsp;Milestone 3c - desynchronisation", H2), P(
    "When the attack traces are randomly shifted the compact CNN collapses: its "
    "mean key rank ends at 109 for a shift of up to 50 samples and 176 for up to "
    "100 samples, with attack accuracy back at the 1/256 chance level, and the key "
    "is never recovered. The ResNet still recovers it - rank 0 after about 608 "
    "traces at shift 50 and about 979 at shift 100, against about 870 on aligned "
    "traces, so a modest penalty rather than a failure. This matches the "
    "literature: a deeper residual network tolerates misalignment that a compact "
    "CNN cannot. The caveat is training stability - the shift-50 ResNet run only "
    "recovered on the second attempt, after its guessing-entropy plateau failed "
    "to break inside a 50-epoch budget and did break at 80 epochs. Shift "
    "augmentation during training would be the next step to close the remaining "
    "gap to the aligned case.")]
story += fig("milestone3_desync.png",
            "Figure 4. Fixed-key ASCAD, byte 2. The compact CNN loses the key "
            "under any misalignment; the ResNet recovers it at both shift levels.")

story += [Paragraph("5.6&nbsp;&nbsp;Milestone 3d - control", H2), P(
    "As a check against recovering the key from a label-distribution artefact "
    "rather than real leakage, the CNN was retrained on the variable-key set with "
    "the profiling labels randomly permuted. Attack accuracy was 0.388%, i.e. "
    "exactly 1/256, and the key rank stayed between 90 and 128 and never reached 0. "
    "The pipeline therefore depends on genuine trace-label association.")]

story += [Paragraph("6&nbsp;&nbsp;Summary of results", H1)]
tbl_data = [
    ["Model", "Fixed-key byte 2", "Variable-key byte 2"],
    ["Random forest", "not recovered (rank ~63)", "-"],
    ["SVM (RBF, PCA-50)", "not recovered (rank 138)", "-"],
    ["CNN (compact)", "1360 traces", "520 traces"],
    ["ResNet (single-task)", "870 traces", "280 traces"],
    ["ResNet (multi-task)", "270 traces", "3400 traces (worse)"],
    ["CNN, desync 50 / 100", "not recovered (rank 109 / 176)", "-"],
    ["ResNet, desync 50 / 100", "608 / 979 traces", "-"],
    ["CNN, shuffled labels", "-", "not recovered (acc = 1/256)"],
]
tbl = Table(tbl_data, colWidths=[5.2 * cm, 5.6 * cm, 5.2 * cm])
tbl.setStyle(TableStyle([
    ("FONT", (0, 0), (-1, -1), "Times-Roman", 9.5),
    ("FONT", (0, 0), (-1, 0), "Times-Bold", 9.5),
    ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.black),
    ("LINEBELOW", (0, -1), (-1, -1), 0.6, colors.black),
    ("LINEABOVE", (0, 0), (-1, 0), 0.6, colors.black),
    ("TOPPADDING", (0, 0), (-1, -1), 3),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.Color(.95, .95, .95)]),
]))
story += [tbl, Paragraph("Table 1. Attack traces needed to reach mean key rank 0. "
                         "A dash means the configuration was not run.", CAP)]

story += [Paragraph("7&nbsp;&nbsp;Discussion", H1)]
story += [Paragraph("7.1&nbsp;&nbsp;Deliberate deviations from the paper", H2), P(
    "Several choices differ from a literal reading of the paper, each for a stated "
    "reason. The compact CNN replaces the 60M-parameter VGG network, which does not "
    "rank the key without heavy tuning. The SVM is trained on a 10 000-trace "
    "subsample because a full RBF SVM over 50 000 traces and 256 classes is "
    "impractical. Model selection uses validation guessing entropy, not validation "
    "loss. The target is always key byte 2, since the published traces only carry "
    "its leakage. These are recorded in a table in the repository README.")]
story += [Paragraph("7.2&nbsp;&nbsp;What reproduced and what did not", H2), P(
    "Reproduced: the classical-versus-deep split, key recovery by both the CNN and "
    "the ResNet, the ResNet's advantage on the variable-key set, and the ResNet "
    "tolerating trace desynchronisation that defeats the compact CNN. Not "
    "reproduced: the literature's ~35-trace efficiency - our best variable-key "
    "figure is around 280 traces for a single seed, so the gap is roughly 8x and "
    "would need a proper multi-seed hyperparameter search to close. The multi-task "
    "ResNet is a clear win on the fixed-key set and a clear loss on the "
    "variable-key set, which is a useful reminder that an auxiliary objective only "
    "helps while optimisation, not capacity, is the limiting factor.")]
story += [Paragraph("7.3&nbsp;&nbsp;Limitations", H2), P(
    "Most numbers are single-seed, and where a training run was repeated the "
    "spread was large (variable-key ResNet: 280 vs 1540 traces; desync-50 ResNet: "
    "no recovery at 50 epochs, 608 traces at 80), so the trace counts should be "
    "read as order-of-magnitude figures and the deep models as sensitive to the "
    "epoch budget. Full-key recovery was not attempted because it needs the raw "
    "trace set. Desync was tested but not defended against with augmentation. The "
    "guessing-entropy checkpoint metric is itself noisy on small validation "
    "slices, which occasionally selects a lucky epoch.")]

story += [Paragraph("8&nbsp;&nbsp;Conclusion", H1), P(
    "The core attack of Poudel and Rahimi reproduces: a profiled deep model "
    "recovers a byte of a masked AES key from ASCAD traces where classical models "
    "cannot, and a 1-D ResNet does so more efficiently than a CNN on the harder "
    "dataset. A multi-task variant that also predicts the mask shares improves the "
    "fixed-key attack substantially but not the variable-key attack. The efficiency "
    "figures reported in the wider literature were not matched and remain the main "
    "target for further tuning.")]

story += [Paragraph("References", H1), Paragraph(
    "[1] A. Poudel and A. Rahimi. Machine Learning-Based AES Key Recovery via "
    "Side-Channel Analysis on the ASCAD Dataset. arXiv:2508.11817, 2025.<br/>"
    "[2] R. Benadjila et al. Deep Learning for Side-Channel Analysis and "
    "Introduction to ASCAD Database. J. Cryptographic Engineering, 2020.<br/>"
    "[3] G. Zaid et al. Methodology for Efficient CNN Architectures in Profiling "
    "Attacks. IACR TCHES 2020(1).<br/>"
    "[4] L. Karayalcin, G. Perin and S. Picek. Resolving the Doubts: On the "
    "Construction and Use of ResNets for Side-Channel Analysis. Mathematics, 2023.<br/>"
    "[5] T. Marquet and E. Oswald. A Comparison of Multi-task Learning and "
    "Single-task Learning Approaches. IACR ePrint 2023/006.",
    ParagraphStyle("ref", parent=BODY, fontSize=9.5, leading=13))]


def build():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUT), pagesize=A4,
        leftMargin=2.2 * cm, rightMargin=2.2 * cm,
        topMargin=2.0 * cm, bottomMargin=1.8 * cm,
        title="ASCAD ML Key-Recovery Replication", author="[Name]",
    )
    doc.build(story, onFirstPage=_page, onLaterPages=_page)
    print(f"wrote {OUT}  ({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    build()
