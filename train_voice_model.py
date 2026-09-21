import os
import numpy as np
import librosa
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from lightgbm import LGBMClassifier


# ============================================================
# PATHS
# ============================================================

BASE = r"C:\VIGILVOICE\data\voice_dataset"

TRAIN_AUDIO_DIR = os.path.join(
    BASE,
    "ASVspoof2019_LA_train",
    "flac"
)

PROTOCOL_FILE = os.path.join(
    BASE,
    "ASVspoof2019_LA_cm_protocols",
    "ASVspoof2019.LA.cm.train.trn.txt"
)

MODEL_DIR = r"C:\VIGILVOICE\models"

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "voice_authenticity_lgbm.pkl"
)

SAMPLE_RATE = 16000

MAX_REAL = 2500
MAX_SPOOF = 2500


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(audio_path):

    try:

        y, sr = librosa.load(
            audio_path,
            sr=SAMPLE_RATE,
            mono=True
        )

        if y is None or len(y) < 1000:
            return None

        # Normalize
        peak = np.max(np.abs(y))

        if peak > 0:
            y = y / peak

        features = []

        # ----------------------------------------------------
        # MFCC
        # ----------------------------------------------------

        mfcc = librosa.feature.mfcc(
            y=y,
            sr=sr,
            n_mfcc=20
        )

        features.extend(np.mean(mfcc, axis=1))
        features.extend(np.std(mfcc, axis=1))

        # ----------------------------------------------------
        # Spectral Centroid
        # ----------------------------------------------------

        centroid = librosa.feature.spectral_centroid(
            y=y,
            sr=sr
        )

        features.append(float(np.mean(centroid)))
        features.append(float(np.std(centroid)))

        # ----------------------------------------------------
        # Spectral Bandwidth
        # ----------------------------------------------------

        bandwidth = librosa.feature.spectral_bandwidth(
            y=y,
            sr=sr
        )

        features.append(float(np.mean(bandwidth)))
        features.append(float(np.std(bandwidth)))

        # ----------------------------------------------------
        # Spectral Rolloff
        # ----------------------------------------------------

        rolloff = librosa.feature.spectral_rolloff(
            y=y,
            sr=sr
        )

        features.append(float(np.mean(rolloff)))
        features.append(float(np.std(rolloff)))

        # ----------------------------------------------------
        # Zero Crossing Rate
        # ----------------------------------------------------

        zcr = librosa.feature.zero_crossing_rate(y)

        features.append(float(np.mean(zcr)))
        features.append(float(np.std(zcr)))

        # ----------------------------------------------------
        # RMS Energy
        # ----------------------------------------------------

        rms = librosa.feature.rms(y=y)

        features.append(float(np.mean(rms)))
        features.append(float(np.std(rms)))

        # ----------------------------------------------------
        # Pitch
        # ----------------------------------------------------

        try:

            f0, voiced_flag, voiced_prob = librosa.pyin(
                y,
                fmin=librosa.note_to_hz("C2"),
                fmax=librosa.note_to_hz("C7"),
                sr=sr
            )

            f0 = f0[np.isfinite(f0)]

            if len(f0) > 0:

                features.append(float(np.mean(f0)))
                features.append(float(np.std(f0)))

            else:

                features.append(0.0)
                features.append(0.0)

        except Exception:

            features.append(0.0)
            features.append(0.0)

        features = np.asarray(
            features,
            dtype=np.float32
        )

        return features

    except Exception as e:

        print(
            f"Feature extraction failed: "
            f"{os.path.basename(audio_path)} -> {e}"
        )

        return None


# ============================================================
# READ PROTOCOL
# ============================================================

def load_protocol():

    real_files = []
    spoof_files = []

    print("\nReading protocol...")
    print(PROTOCOL_FILE)

    with open(
        PROTOCOL_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            parts = line.strip().split()

            if len(parts) < 5:
                continue

            file_id = parts[1]
            label = parts[-1]

            audio_path = os.path.join(
                TRAIN_AUDIO_DIR,
                file_id + ".flac"
            )

            if not os.path.exists(audio_path):
                continue

            # ASVspoof:
            # bonafide = real
            # spoof = AI
            if label.lower() == "bonafide":

                real_files.append(audio_path)

            elif label.lower() == "spoof":

                spoof_files.append(audio_path)

    return real_files, spoof_files


# ============================================================
# MAIN TRAINING
# ============================================================

def main():

    print("=" * 70)
    print("VIGILVOICE - LIGHTGBM VOICE AUTHENTICITY TRAINING")
    print("=" * 70)

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    real_files, spoof_files = load_protocol()

    print("\nAvailable data:")
    print("REAL  :", len(real_files))
    print("SPOOF :", len(spoof_files))

    # Limit dataset
    real_files = real_files[:MAX_REAL]
    spoof_files = spoof_files[:MAX_SPOOF]

    print("\nUsing:")
    print("REAL  :", len(real_files))
    print("SPOOF :", len(spoof_files))

    X = []
    y = []

    # --------------------------------------------------------
    # REAL
    # --------------------------------------------------------

    print("\nExtracting REAL features...")

    for i, audio_path in enumerate(real_files):

        features = extract_features(audio_path)

        if features is not None:

            X.append(features)
            y.append(0)

        if (i + 1) % 100 == 0:

            print(
                f"REAL: {i + 1}/{len(real_files)}"
            )

    # --------------------------------------------------------
    # SPOOF
    # --------------------------------------------------------

    print("\nExtracting SPOOF features...")

    for i, audio_path in enumerate(spoof_files):

        features = extract_features(audio_path)

        if features is not None:

            X.append(features)
            y.append(1)

        if (i + 1) % 100 == 0:

            print(
                f"SPOOF: {i + 1}/{len(spoof_files)}"
            )

    X = np.asarray(
        X,
        dtype=np.float32
    )

    y = np.asarray(
        y,
        dtype=np.int32
    )

    print("\nFeature matrix:")
    print("Shape:", X.shape)

    print("\nLabels:")
    print("REAL  :", np.sum(y == 0))
    print("SPOOF :", np.sum(y == 1))

    # --------------------------------------------------------
    # TRAIN / TEST SPLIT
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("\nTraining samples:", len(X_train))
    print("Testing samples :", len(X_test))

    # --------------------------------------------------------
    # LIGHTGBM
    # --------------------------------------------------------

    print("\nCreating LightGBM model...")

    model = LGBMClassifier(

        objective="binary",

        n_estimators=300,

        learning_rate=0.05,

        num_leaves=31,

        max_depth=-1,

        min_child_samples=20,

        subsample=0.9,

        colsample_bytree=0.9,

        reg_alpha=0.1,

        reg_lambda=0.1,

        random_state=42,

        n_jobs=-1,

        verbosity=-1
    )

    print("\nTraining LightGBM...")

    model.fit(
        X_train,
        y_train
    )

    # --------------------------------------------------------
    # EVALUATION
    # --------------------------------------------------------

    print("\nEvaluating model...")

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print("\n" + "=" * 70)
    print("LIGHTGBM RESULTS")
    print("=" * 70)

    print(
        f"\nAccuracy: {accuracy * 100:.2f}%"
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            target_names=[
                "REAL",
                "AI/SPOOF"
            ]
        )
    )

    print("\nConfusion Matrix:")

    print(
        confusion_matrix(
            y_test,
            predictions
        )
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    joblib.dump(
        model,
        MODEL_PATH
    )

    print("\n" + "=" * 70)
    print("MODEL SAVED")
    print("=" * 70)

    print(MODEL_PATH)

    print("\nClasses:")
    print(model.classes_)

    print("\nTraining completed successfully.")


if __name__ == "__main__":
    main()