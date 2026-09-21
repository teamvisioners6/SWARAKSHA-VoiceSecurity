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
# VIGILVOICE - SPECTRA-AASIST3 OFFICIAL DEV CALIBRATION
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

CACHE_PATH = (
    r"C:\VIGILVOICE\models"
    r"\vigilvoice_aasist3_dev_calibration.npz"
)


# ============================================================
# MODEL SETTINGS
# ============================================================

SR = 16000
TARGET_LENGTH = 64600
HOP = TARGET_LENGTH // 2


# ============================================================
# HEADER
# ============================================================

print("=" * 75)
print("VIGILVOICE - SPECTRA-AASIST3 OFFICIAL DEV CALIBRATION")
print("=" * 75)


# ============================================================
# ONNX
# ============================================================

print()
print("Loading Spectra-AASIST3...")

try:
    ort.preload_dlls(
        directory=""
    )
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

print(
    "Available providers:",
    available
)

print(
    "Actual providers:",
    session.get_providers()
)


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

        raise ValueError(
            "Empty audio."
        )

    # Spectra-AASIST3 preemphasis
    emphasized = np.empty_like(
        audio
    )

    emphasized[0] = audio[0]

    if len(audio) > 1:

        emphasized[1:] = (
            audio[1:]
            -
            0.97 * audio[:-1]
        )

    audio = emphasized


    # Tile short utterances
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


    # Deterministic first window
    audio = audio[
        :TARGET_LENGTH
    ]

    return np.asarray(
        audio,
        dtype=np.float32
    )


# ============================================================
# SINGLE AASIST3 WINDOW
# ============================================================

def predict_segment(audio):

    processed = preprocess(
        audio
    )

    batch = processed.reshape(
        1,
        TARGET_LENGTH
    ).astype(np.float32)

    output = session.run(
        [output_name],
        {
            input_name: batch
        }
    )[0]

    probabilities = softmax(
        output
    )[0]

    # Spectra-AASIST3:
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
    # SHORT AUDIO
    # --------------------------------------------------------

    if len(audio) <= TARGET_LENGTH:

        score = predict_segment(
            audio
        )

        return score, 1


    # --------------------------------------------------------
    # LONG AUDIO
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

        starts.append(
            last_start
        )


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


    # --------------------------------------------------------
    # SAME ROBUST AGGREGATION AS CURRENT DETECTOR
    # --------------------------------------------------------

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
# PROTOCOL
# ============================================================

entries = []

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

        label_text = (
            parts[-1].lower()
        )

        label = (
            1
            if label_text == "spoof"
            else 0
        )

        path = os.path.join(
            DEV_AUDIO,
            file_id + ".flac"
        )

        if os.path.exists(path):

            entries.append(
                (
                    path,
                    label,
                    file_id
                )
            )


print()
print(
    "Dev utterances:",
    len(entries)
)

print(
    "Distribution:",
    Counter(
        x[1]
        for x in entries
    )
)


# ============================================================
# CACHE
# ============================================================

if os.path.exists(
    CACHE_PATH
):

    print()
    print(
        "Existing AASIST3 cache found."
    )

    cache = np.load(
        CACHE_PATH,
        allow_pickle=True
    )

    file_ids = cache[
        "file_ids"
    ]

    y_true = cache[
        "y_true"
    ]

    y_score = cache[
        "y_score"
    ]

    print(
        "Loaded:",
        len(y_true),
        "cached scores"
    )


else:

    print()
    print(
        "Evaluating official Dev set..."
    )

    print(
        "AASIST3 is currently running on:",
        session.get_providers()[0]
    )

    print()

    file_ids = []
    y_true = []
    y_score = []


    for i, (
        path,
        label,
        file_id
    ) in enumerate(
        entries,
        1
    ):

        score, segments = (
            predict_file(
                path
            )
        )

        file_ids.append(
            file_id
        )

        y_true.append(
            label
        )

        y_score.append(
            score
        )


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


        # ----------------------------------------------------
        # CHECKPOINT EVERY 100 FILES
        # ----------------------------------------------------

        if i % 100 == 0:

            np.savez(
                CACHE_PATH,
                file_ids=np.asarray(
                    file_ids
                ),
                y_true=np.asarray(
                    y_true
                ),
                y_score=np.asarray(
                    y_score,
                    dtype=np.float32
                )
            )


    print()

    file_ids = np.asarray(
        file_ids
    )

    y_true = np.asarray(
        y_true
    )

    y_score = np.asarray(
        y_score,
        dtype=np.float32
    )


    np.savez(
        CACHE_PATH,
        file_ids=file_ids,
        y_true=y_true,
        y_score=y_score
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

fnr = (
    1.0 - tpr
)


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

eer_threshold = (
    thresholds[eer_idx]
)


# ============================================================
# RESULTS
# ============================================================

print()
print("=" * 75)
print("AASIST3 OFFICIAL DEV RESULTS")
print("=" * 75)

print(
    f"ROC-AUC       : {auc:.5f}"
)

print(
    f"EER           : {eer * 100:.2f}%"
)

print(
    f"EER Threshold : {eer_threshold:.5f}"
)


# ============================================================
# LOW-FPR THRESHOLDS
# ============================================================

print()
print("=" * 75)
print("AASIST3 LOW-FPR THRESHOLD ANALYSIS")
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

    recall = float(
        tpr[idx]
    )

    actual_fpr = float(
        fpr[idx]
    )


    selected_thresholds[
        str(target_fpr)
    ] = threshold


    print(
        f"FPR <= {target_fpr * 100:6.2f}%"
        f" | Threshold {threshold:.6f}"
        f" | Actual FPR {actual_fpr * 100:6.3f}%"
        f" | Spoof Recall {recall * 100:6.2f}%"
    )


# ============================================================
# STANDARD THRESHOLDS
# ============================================================

print()
print("=" * 75)
print("FIXED THRESHOLD ANALYSIS")
print("=" * 75)


for threshold in [
    0.30,
    0.40,
    0.50,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80
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
        if (tp + fp) > 0
        else 0
    )


    recall = (
        tp /
        (tp + fn)
        if (tp + fn) > 0
        else 0
    )


    false_positive = (
        fp /
        (fp + tn)
        if (fp + tn) > 0
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
# CONFUSION MATRIX @ 1% FPR OPERATING POINT
# ============================================================

if "0.01" in selected_thresholds:

    threshold_1pct = (
        selected_thresholds["0.01"]
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

false_positive = (
    fp /
    (fp + tn)
    if fp + tn > 0
    else 0
)


print()
print("=" * 75)
print(
    "AASIST3 @ <=1% DEV FPR"
)
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
    f"False Positive : {false_positive * 100:.2f}%"
)

print()
print(
    "Confusion Matrix:"
)

print(cm)


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
print("BONAFIDE / REAL")

print(
    "Mean   :",
    f"{np.mean(real_scores):.6f}"
)

print(
    "Median :",
    f"{np.median(real_scores):.6f}"
)

print(
    "P95    :",
    f"{np.percentile(real_scores, 95):.6f}"
)

print(
    "Max    :",
    f"{np.max(real_scores):.6f}"
)


print()
print("SPOOF")

print(
    "Mean   :",
    f"{np.mean(spoof_scores):.6f}"
)

print(
    "Median :",
    f"{np.median(spoof_scores):.6f}"
)

print(
    "P05    :",
    f"{np.percentile(spoof_scores, 5):.6f}"
)

print(
    "Min    :",
    f"{np.min(spoof_scores):.6f}"
)


# ============================================================
# SAVE CALIBRATION
# ============================================================

np.savez(
    CACHE_PATH,
    file_ids=file_ids,
    y_true=y_true,
    y_score=y_score,
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
    CACHE_PATH
)

print()
print("DONE.")
print("=" * 75)