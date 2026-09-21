import os
import joblib
import numpy as np
import librosa

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)


# ============================================================
# CONFIG
# ============================================================

BASE = r"C:\VIGILVOICE\data\voice_dataset"

DEV_AUDIO = os.path.join(
    BASE,
    "ASVspoof2019_LA_dev",
    "flac"
)

PROTOCOL = os.path.join(
    BASE,
    "ASVspoof2019_LA_cm_protocols",
    "ASVspoof2019.LA.cm.dev.trl.txt"
)

MODEL_PATH = r"C:\VIGILVOICE\models\voice_authenticity_model.pkl"

SR = 16000

# Set to None to evaluate the entire development set.
# For a faster first test, you can use 2000.
MAX_FILES = None


# ============================================================
# FEATURE EXTRACTION
# MUST MATCH TRAINING FEATURES
# ============================================================

def extract_features(path):

    try:

        audio, sr = librosa.load(
            path,
            sr=SR,
            mono=True,
            duration=4.0
        )

        if len(audio) < 1000:
            return None

        audio = audio.astype(np.float32)

        # ----------------------------------------------------
        # MFCC
        # ----------------------------------------------------

        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=20
        )

        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)

        # ----------------------------------------------------
        # Spectral features
        # ----------------------------------------------------

        centroid = librosa.feature.spectral_centroid(
            y=audio,
            sr=sr
        )

        bandwidth = librosa.feature.spectral_bandwidth(
            y=audio,
            sr=sr
        )

        rolloff = librosa.feature.spectral_rolloff(
            y=audio,
            sr=sr
        )

        zcr = librosa.feature.zero_crossing_rate(
            audio
        )

        # ----------------------------------------------------
        # Energy
        # ----------------------------------------------------

        rms = librosa.feature.rms(
            y=audio
        )[0]

        # ----------------------------------------------------
        # Pitch
        # ----------------------------------------------------

        try:

            f0, _, _ = librosa.pyin(
                audio,
                fmin=librosa.note_to_hz("C2"),
                fmax=librosa.note_to_hz("C7"),
                sr=sr
            )

            valid_f0 = f0[np.isfinite(f0)]

            if len(valid_f0) > 0:

                pitch_mean = np.mean(valid_f0)
                pitch_std = np.std(valid_f0)

            else:

                pitch_mean = 0
                pitch_std = 0

        except Exception:

            pitch_mean = 0
            pitch_std = 0

        # ----------------------------------------------------
        # Combine
        # ----------------------------------------------------

        features = np.concatenate([

            mfcc_mean,
            mfcc_std,

            [
                np.mean(centroid),
                np.std(centroid),

                np.mean(bandwidth),
                np.std(bandwidth),

                np.mean(rolloff),
                np.std(rolloff),

                np.mean(zcr),
                np.std(zcr),

                np.mean(rms),
                np.std(rms),

                pitch_mean,
                pitch_std
            ]

        ])

        return features

    except Exception as e:

        print(
            f"Feature extraction failed: "
            f"{os.path.basename(path)} -> {e}"
        )

        return None


# ============================================================
# LOAD DEV PROTOCOL
# ============================================================

def load_protocol():

    files = []
    labels = []

    with open(PROTOCOL, "r") as f:

        for line in f:

            parts = line.strip().split()

            if len(parts) < 5:
                continue

            filename = parts[1]
            label = parts[-1]

            audio_path = os.path.join(
                DEV_AUDIO,
                filename + ".flac"
            )

            if not os.path.exists(audio_path):
                continue

            if label == "bonafide":

                files.append(audio_path)
                labels.append(0)

            elif label == "spoof":

                files.append(audio_path)
                labels.append(1)

            if (
                MAX_FILES is not None
                and len(files) >= MAX_FILES
            ):
                break

    return files, labels


# ============================================================
# MAIN
# ============================================================

print("=" * 60)
print("VIGILVOICE - UNSEEN ASVSPOOF EVALUATION")
print("=" * 60)

# ------------------------------------------------------------
# Check model
# ------------------------------------------------------------

print("\nLoading trained model...")

if not os.path.exists(MODEL_PATH):

    print("ERROR: Model not found:")
    print(MODEL_PATH)
    raise SystemExit

model = joblib.load(MODEL_PATH)

print("Model:", type(model).__name__)
print("Classes:", model.classes_)

# ------------------------------------------------------------
# Load development protocol
# ------------------------------------------------------------

print("\nLoading ASVspoof development protocol...")

files, labels = load_protocol()

print("Files found:", len(files))

if len(files) == 0:

    print("\nERROR: No development files found.")
    print("Check:")
    print(DEV_AUDIO)
    print(PROTOCOL)
    raise SystemExit

print(
    "REAL:",
    labels.count(0)
)

print(
    "SPOOF:",
    labels.count(1)
)

# ------------------------------------------------------------
# Extract features
# ------------------------------------------------------------

print("\nExtracting features...")

X = []
y = []

total = len(files)

for i, (path, label) in enumerate(
    zip(files, labels),
    start=1
):

    features = extract_features(path)

    if features is not None:

        X.append(features)
        y.append(label)

    if i % 100 == 0 or i == total:

        print(
            f"Processed {i}/{total}"
        )


X = np.array(X)
y = np.array(y)

print("\nFeature matrix:", X.shape)

if len(X) == 0:

    print("ERROR: No usable features.")
    raise SystemExit

# ------------------------------------------------------------
# Prediction
# ------------------------------------------------------------

print("\nRunning predictions...")

predictions = model.predict(X)

probabilities = model.predict_proba(X)

# ------------------------------------------------------------
# Metrics
# ------------------------------------------------------------

accuracy = accuracy_score(
    y,
    predictions
)

precision = precision_score(
    y,
    predictions,
    pos_label=1,
    zero_division=0
)

recall = recall_score(
    y,
    predictions,
    pos_label=1,
    zero_division=0
)

f1 = f1_score(
    y,
    predictions,
    pos_label=1,
    zero_division=0
)

cm = confusion_matrix(
    y,
    predictions
)

# ------------------------------------------------------------
# Results
# ------------------------------------------------------------

print("\n")
print("=" * 60)
print("UNSEEN DATASET RESULTS")
print("=" * 60)

print(
    f"\nAccuracy : {accuracy * 100:.2f}%"
)

print(
    f"Precision: {precision * 100:.2f}%"
)

print(
    f"Recall   : {recall * 100:.2f}%"
)

print(
    f"F1 Score : {f1 * 100:.2f}%"
)

print("\nClassification Report:")
print(
    classification_report(
        y,
        predictions,
        target_names=[
            "REAL",
            "AI/SPOOF"
        ],
        digits=4,
        zero_division=0
    )
)

# ------------------------------------------------------------
# Confusion Matrix
# ------------------------------------------------------------

print("Confusion Matrix:")
print()

print(
    "                 Predicted"
)

print(
    "                 REAL   SPOOF"
)

print(
    f"Actual REAL     {cm[0][0]:5d}  {cm[0][1]:5d}"
)

print(
    f"Actual SPOOF    {cm[1][0]:5d}  {cm[1][1]:5d}"
)

# ------------------------------------------------------------
# Probability statistics
# ------------------------------------------------------------

spoof_prob = probabilities[:, 1]

print("\nProbability statistics:")

print(
    f"Average spoof probability: "
    f"{np.mean(spoof_prob):.4f}"
)

print(
    f"Minimum spoof probability: "
    f"{np.min(spoof_prob):.4f}"
)

print(
    f"Maximum spoof probability: "
    f"{np.max(spoof_prob):.4f}"
)

print("\n" + "=" * 60)
print("EVALUATION COMPLETE")
print("=" * 60)