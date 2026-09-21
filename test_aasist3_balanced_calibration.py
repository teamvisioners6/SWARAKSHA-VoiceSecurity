import os
import warnings
import numpy as np
import librosa
import onnxruntime as ort

from collections import Counter
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    confusion_matrix
)


# ============================================================
# VIGILVOICE
# SPECTRA-AASIST3 BALANCED DEV CALIBRATION
#
# IMPORTANT:
# - Official ASVspoof2019 LA Dev protocol
# - 500 bonafide + 500 spoof
# - Fixed random seed
# - Calibration only
# - NOT an official full-Dev performance claim
# ============================================================

ROOT = r"C:\VIGILVOICE\data\voice_dataset"

DEV_AUDIO = os.path.join(
    ROOT,
    "ASVspoof2019_LA_dev",
    "flac"
)

DEV_PROTOCOL = os.path.join(
    ROOT,
    "ASVspoof2019_LA_cm_protocols",
    "ASVspoof2019.LA.cm.dev.trl.txt"
)

MODEL_PATH = (
    r"C:\VIGILVOICE\models\spectra_aasist3"
    r"\spectra-aasist3.onnx"
)

RESULT_PATH = (
    r"C:\VIGILVOICE\models"
    r"\vigilvoice_aasist3_balanced_calibration.npz"
)

SEED = 42

N_REAL = 500
N_SPOOF = 500

SR = 16000
TARGET_LENGTH = 64600
HOP = TARGET_LENGTH // 2


# ============================================================
# HEADER
# ============================================================

print("=" * 75)
print("VIGILVOICE - AASIST3 BALANCED DEV CALIBRATION")
print("=" * 75)

print()
print("Target:")
print(f"  Bonafide : {N_REAL}")
print(f"  Spoof    : {N_SPOOF}")
print(f"  Total    : {N_REAL + N_SPOOF}")
print(f"  Seed     : {SEED}")


# ============================================================
# LOAD ONNX
# ============================================================

print()
print("Loading Spectra-AASIST3...")

try:
    ort.preload_dlls(directory="")
except Exception:
    pass

available = ort.get_available_providers()

if "CUDAExecutionProvider" in available:
    providers = [
        "CUDAExecutionProvider",
        "CPUExecutionProvider"
    ]
else:
    providers = [
        "CPUExecutionProvider"
    ]

session = ort.InferenceSession(
    MODEL_PATH,
    providers=providers
)

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

print("Available providers:", available)
print("Actual providers:", session.get_providers())


# ============================================================
# SOFTMAX
# ============================================================

def softmax(x):

    x = np.asarray(
        x,
        dtype=np.float32
    )

    x = x - np.max(
        x,
        axis=-1,
        keepdims=True
    )

    e = np.exp(x)

    return (
        e /
        np.sum(
            e,
            axis=-1,
            keepdims=True
        )
    )


# ============================================================
# PREPROCESS
# ============================================================

def preprocess(audio):

    audio = np.asarray(
        audio,
        dtype=np.float32
    )

    if len(audio) == 0:
        raise ValueError("Empty audio")

    # Spectra-AASIST3 preemphasis
    emphasized = np.empty_like(audio)

    emphasized[0] = audio[0]

    if len(audio) > 1:
        emphasized[1:] = (
            audio[1:]
            -
            0.97 * audio[:-1]
        )

    audio = emphasized

    # Tile short audio
    if len(audio) < TARGET_LENGTH:

        repeats = int(
            np.ceil(
                TARGET_LENGTH /
                len(audio)
            )
        )

        audio = np.tile(
            audio,
            repeats
        )

    return np.asarray(
        audio[:TARGET_LENGTH],
        dtype=np.float32
    )


# ============================================================
# SINGLE WINDOW
# ============================================================

def predict_segment(audio):

    audio = preprocess(audio)

    batch = audio.reshape(
        1,
        TARGET_LENGTH
    ).astype(np.float32)

    output = session.run(
        [output_name],
        {
            input_name: batch
        }
    )[0]

    probabilities = softmax(output)[0]

    # Spectra-AASIST3
    # index 0 = spoof
    # index 1 = bona fide

    return float(
        probabilities[0]
    )


# ============================================================
# FILE PREDICTION
# ============================================================

def predict_file(path):

    with warnings.catch_warnings():

        warnings.simplefilter(
            "ignore"
        )

        audio, _ = librosa.load(
            path,
            sr=SR,
            mono=True
        )

    audio = np.asarray(
        audio,
        dtype=np.float32
    )

    if len(audio) == 0:
        return 0.0, 0

    # --------------------------------------------------------
    # SHORT FILE
    # --------------------------------------------------------

    if len(audio) <= TARGET_LENGTH:

        score = predict_segment(
            audio
        )

        return score, 1

    # --------------------------------------------------------
    # LONG FILE
    # --------------------------------------------------------

    starts = list(
        range(
            0,
            len(audio)
            - TARGET_LENGTH
            + 1,
            HOP
        )
    )

    last_start = (
        len(audio)
        - TARGET_LENGTH
    )

    if starts[-1] != last_start:
        starts.append(last_start)

    scores = []

    for start in starts:

        segment = audio[
            start:
            start + TARGET_LENGTH
        ]

        scores.append(
            predict_segment(
                segment
            )
        )

    scores = np.asarray(
        scores,
        dtype=np.float32
    )

    avg_spoof = np.mean(
        scores
    )

    median_spoof = np.median(
        scores
    )

    strong_ratio = np.mean(
        scores >= 0.80
    )

    moderate_ratio = np.mean(
        scores >= 0.50
    )

    # Current VIGILVOICE aggregation
    final_score = (
        0.45 * avg_spoof
        +
        0.30 * median_spoof
        +
        0.15 * strong_ratio
        +
        0.10 * moderate_ratio
    )

    return (
        float(final_score),
        len(scores)
    )


# ============================================================
# READ DEV PROTOCOL
# ============================================================

print()
print("Reading official Dev protocol...")

bonafide = []
spoof = []

with open(
    DEV_PROTOCOL,
    "r",
    encoding="utf-8"
) as f:

    for line in f:

        parts = line.strip().split()

        if len(parts) < 5:
            continue

        file_id = parts[1]

        label_text = parts[-1].lower()

        path = os.path.join(
            DEV_AUDIO,
            file_id + ".flac"
        )

        if not os.path.exists(path):
            continue

        if label_text == "bonafide":
            bonafide.append(
                (
                    path,
                    0,
                    file_id
                )
            )

        elif label_text == "spoof":
            spoof.append(
                (
                    path,
                    1,
                    file_id
                )
            )


print()
print("Available:")
print("  Bonafide:", len(bonafide))
print("  Spoof   :", len(spoof))


# ============================================================
# STRATIFIED RANDOM SAMPLE
# ============================================================

rng = np.random.default_rng(
    SEED
)

if len(bonafide) < N_REAL:
    raise RuntimeError(
        "Not enough bonafide files."
    )

if len(spoof) < N_SPOOF:
    raise RuntimeError(
        "Not enough spoof files."
    )


real_indices = rng.choice(
    len(bonafide),
    size=N_REAL,
    replace=False
)

spoof_indices = rng.choice(
    len(spoof),
    size=N_SPOOF,
    replace=False
)


selected_real = [
    bonafide[i]
    for i in real_indices
]

selected_spoof = [
    spoof[i]
    for i in spoof_indices
]


entries = (
    selected_real
    +
    selected_spoof
)


# Shuffle combined set
rng.shuffle(entries)


print()
print("=" * 75)
print("SELECTED CALIBRATION SET")
print("=" * 75)

print(
    "Bonafide:",
    len(selected_real)
)

print(
    "Spoof   :",
    len(selected_spoof)
)

print(
    "Total   :",
    len(entries)
)


# ============================================================
# RUN MODEL
# ============================================================

scores = []
labels = []
file_ids = []
segment_counts = []

print()
print("Running AASIST3...")
print(
    "This is intentionally limited to 1,000 files."
)
print()


for i, (
    path,
    label,
    file_id
) in enumerate(
    entries,
    1
):

    score, segments = predict_file(
        path
    )

    scores.append(score)
    labels.append(label)
    file_ids.append(file_id)
    segment_counts.append(segments)

    if (
        i % 25 == 0
        or
        i == len(entries)
    ):

        print(
            f"\rProcessed "
            f"{i}/{len(entries)}",
            end="",
            flush=True
        )


print()


# ============================================================
# ARRAYS
# ============================================================

y_true = np.asarray(
    labels,
    dtype=np.int32
)

y_score = np.asarray(
    scores,
    dtype=np.float32
)

file_ids = np.asarray(
    file_ids
)

segment_counts = np.asarray(
    segment_counts,
    dtype=np.int32
)


# ============================================================
# ROC / AUC
# ============================================================

auc = roc_auc_score(
    y_true,
    y_score
)

fpr, tpr, thresholds = roc_curve(
    y_true,
    y_score
)

fnr = 1.0 - tpr


# ============================================================
# EER
# ============================================================

eer_idx = np.argmin(
    np.abs(
        fpr - fnr
    )
)

eer = (
    fpr[eer_idx]
    +
    fnr[eer_idx]
) / 2

eer_threshold = float(
    thresholds[eer_idx]
)


# ============================================================
# HEADER RESULTS
# ============================================================

print()
print("=" * 75)
print("AASIST3 BALANCED DEV RESULTS")
print("=" * 75)

print(
    f"ROC-AUC       : {auc:.5f}"
)

print(
    f"EER           : {eer * 100:.2f}%"
)

print(
    f"EER Threshold : {eer_threshold:.6f}"
)


# ============================================================
# LOW FPR THRESHOLDS
# ============================================================

print()
print("=" * 75)
print("LOW-FPR OPERATING POINTS")
print("=" * 75)


selected_thresholds = {}


for target_fpr in [
    0.001,
    0.002,
    0.005,
    0.010,
    0.020,
    0.030,
    0.050,
    0.100
]:

    valid = np.where(
        fpr <= target_fpr
    )[0]

    if len(valid) == 0:
        continue

    idx = valid[
        np.argmax(
            tpr[valid]
        )
    ]

    threshold = float(
        thresholds[idx]
    )

    actual_fpr = float(
        fpr[idx]
    )

    recall = float(
        tpr[idx]
    )

    selected_thresholds[
        target_fpr
    ] = threshold

    print(
        f"FPR <= {target_fpr * 100:5.2f}%"
        f" | Threshold {threshold:.6f}"
        f" | Actual FPR {actual_fpr * 100:6.3f}%"
        f" | Spoof Recall {recall * 100:6.2f}%"
    )


# ============================================================
# FIXED THRESHOLD TEST
# ============================================================

print()
print("=" * 75)
print("FIXED THRESHOLD PERFORMANCE")
print("=" * 75)


for threshold in [
    0.30,
    0.40,
    0.50,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.90
]:

    pred = (
        y_score >= threshold
    ).astype(int)

    cm = confusion_matrix(
        y_true,
        pred,
        labels=[0, 1]
    )

    tn, fp, fn, tp = (
        cm.ravel()
    )

    accuracy = (
        (tn + tp)
        /
        len(y_true)
    )

    precision = (
        tp /
        (tp + fp)
        if tp + fp > 0
        else 0
    )

    recall = (
        tp /
        (tp + fn)
        if tp + fn > 0
        else 0
    )

    false_positive = (
        fp /
        (fp + tn)
        if fp + tn > 0
        else 0
    )

    print(
        f"Threshold {threshold:.2f}"
        f" | Acc {accuracy * 100:6.2f}%"
        f" | Precision {precision * 100:6.2f}%"
        f" | Recall {recall * 100:6.2f}%"
        f" | FPR {false_positive * 100:6.2f}%"
    )


# ============================================================
# SCORE DISTRIBUTION
# ============================================================

real_scores = y_score[
    y_true == 0
]

spoof_scores = y_score[
    y_true == 1
]


print()
print("=" * 75)
print("SCORE DISTRIBUTION")
print("=" * 75)

print()
print("BONAFIDE")

print(
    f"Mean   : {np.mean(real_scores):.6f}"
)

print(
    f"Median : {np.median(real_scores):.6f}"
)

print(
    f"P95    : {np.percentile(real_scores, 95):.6f}"
)

print(
    f"Max    : {np.max(real_scores):.6f}"
)


print()
print("SPOOF")

print(
    f"Mean   : {np.mean(spoof_scores):.6f}"
)

print(
    f"Median : {np.median(spoof_scores):.6f}"
)

print(
    f"P05    : {np.percentile(spoof_scores, 5):.6f}"
)

print(
    f"Min    : {np.min(spoof_scores):.6f}"
)


# ============================================================
# FINAL 1% FPR OPERATING POINT
# ============================================================

if 0.01 in selected_thresholds:

    threshold_1pct = (
        selected_thresholds[0.01]
    )

else:

    threshold_1pct = 0.50


pred = (
    y_score >= threshold_1pct
).astype(int)


cm = confusion_matrix(
    y_true,
    pred,
    labels=[0, 1]
)

tn, fp, fn, tp = (
    cm.ravel()
)

accuracy = (
    (tn + tp)
    /
    len(y_true)
)

precision = (
    tp /
    (tp + fp)
    if tp + fp > 0
    else 0
)

recall = (
    tp /
    (tp + fn)
    if tp + fn > 0
    else 0
)

actual_fpr = (
    fp /
    (fp + tn)
    if fp + tn > 0
    else 0
)


print()
print("=" * 75)
print("RECOMMENDED LOW-FPR OPERATING POINT")
print("=" * 75)

print(
    f"Threshold      : {threshold_1pct:.6f}"
)

print(
    f"Accuracy       : {accuracy * 100:.2f}%"
)

print(
    f"Precision      : {precision * 100:.2f}%"
)

print(
    f"Spoof Recall   : {recall * 100:.2f}%"
)

print(
    f"False Positive : {actual_fpr * 100:.2f}%"
)

print()
print("Confusion Matrix")
print(
    "[[TN, FP],")
print(
    " [FN, TP]]"
)

print(cm)


# ============================================================
# SAVE
# ============================================================

np.savez(
    RESULT_PATH,

    file_ids=file_ids,

    y_true=y_true,

    y_score=y_score,

    segment_counts=segment_counts,

    fpr=fpr,

    tpr=tpr,

    thresholds=thresholds,

    auc=auc,

    eer=eer,

    eer_threshold=eer_threshold,

    threshold_1pct=threshold_1pct
)


print()
print("=" * 75)
print("CALIBRATION SAVED")
print("=" * 75)

print(
    RESULT_PATH
)

print()
print("IMPORTANT:")
print(
    "These are balanced-sample calibration results."
)

print(
    "They are NOT full official Dev-set accuracy."
)

print()
print("DONE.")
print("=" * 75)