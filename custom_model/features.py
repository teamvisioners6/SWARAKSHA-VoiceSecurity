from pathlib import Path

import librosa
import numpy as np
import torch


# ============================================================
# SWARAKSHA - Audio Feature Extraction V2
# ============================================================

SAMPLE_RATE = 16000

DURATION = 4.0

NUM_SAMPLES = int(
    SAMPLE_RATE * DURATION
)

N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 128

FMIN = 20
FMAX = 8000


# ============================================================
# Load audio
# ============================================================

def load_audio(audio_path: str) -> np.ndarray:
    """
    Load audio as mono 16 kHz floating-point waveform.
    """

    audio, _ = librosa.load(
        audio_path,
        sr=SAMPLE_RATE,
        mono=True,
    )

    return audio.astype(
        np.float32
    )


# ============================================================
# Fix audio length
# ============================================================

def fix_length(
    audio: np.ndarray,
    num_samples: int = NUM_SAMPLES,
) -> np.ndarray:
    """
    Convert audio to exactly 4 seconds.

    Short audio:
        Zero padded.

    Long audio:
        Center cropped.
    """

    if len(audio) == num_samples:
        return audio

    if len(audio) < num_samples:

        padded = np.zeros(
            num_samples,
            dtype=np.float32,
        )

        padded[:len(audio)] = audio

        return padded

    start = (
        len(audio) - num_samples
    ) // 2

    return audio[
        start:start + num_samples
    ].astype(np.float32)


# ============================================================
# Audio normalization
# ============================================================

def normalize_audio(
    audio: np.ndarray,
) -> np.ndarray:
    """
    Peak-normalize waveform.
    """

    peak = np.max(
        np.abs(audio)
    )

    if peak > 1e-8:

        audio = audio / peak

    return audio.astype(
        np.float32
    )


# ============================================================
# Log-Mel from waveform
# ============================================================

def waveform_to_log_mel(
    audio: np.ndarray,
) -> np.ndarray:
    """
    Convert an already-loaded waveform into
    a standardized log-Mel spectrogram.

    Expected input:
        16 kHz mono waveform

    Output:
        (N_MELS, TIME_FRAMES)
    """

    # --------------------------------------------------------
    # Ensure fixed length
    # --------------------------------------------------------

    audio = fix_length(
        audio
    )

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    audio = normalize_audio(
        audio
    )

    # --------------------------------------------------------
    # Mel spectrogram
    # --------------------------------------------------------

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
        power=2.0,
    )

    # --------------------------------------------------------
    # Power → dB
    # --------------------------------------------------------

    log_mel = librosa.power_to_db(
        mel,
        ref=np.max,
    )

    # --------------------------------------------------------
    # Per-sample standardization
    # --------------------------------------------------------

    mean = np.mean(
        log_mel
    )

    std = np.std(
        log_mel
    )

    log_mel = (
        log_mel - mean
    ) / (
        std + 1e-8
    )

    return log_mel.astype(
        np.float32
    )


# ============================================================
# Standard inference feature extraction
# ============================================================

def extract_log_mel(
    audio_path: str,
) -> np.ndarray:
    """
    Standard feature extraction.

    Used for:
        validation
        testing
        inference

    NO augmentation is applied here.
    """

    audio = load_audio(
        audio_path
    )

    return waveform_to_log_mel(
        audio
    )


# ============================================================
# Training feature extraction
# ============================================================

def extract_training_feature_tensor(
    audio_path: str,
) -> torch.Tensor:
    """
    Extract training features.

    Workflow:

        audio file
            ↓
        waveform
            ↓
        optional augmentation
            ↓
        fixed length
            ↓
        normalization
            ↓
        log-Mel
            ↓
        tensor
    """

    audio = load_audio(
        audio_path
    )

    # --------------------------------------------------------
    # Import augmentation only when training
    # --------------------------------------------------------

    from augmentation import (
        apply_random_augmentation
    )

    audio = apply_random_augmentation(
        audio
    )

    feature = waveform_to_log_mel(
        audio
    )

    tensor = torch.from_numpy(
        feature
    ).unsqueeze(0)

    return tensor.float()


# ============================================================
# Standard tensor extraction
# ============================================================

def extract_feature_tensor(
    audio_path: str,
) -> torch.Tensor:
    """
    Standard feature extraction for
    validation and inference.

    NO augmentation.
    """

    feature = extract_log_mel(
        audio_path
    )

    tensor = torch.from_numpy(
        feature
    ).unsqueeze(0)

    return tensor.float()


# ============================================================
# Quick test
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("SWARAKSHA - FEATURE EXTRACTION V2 TEST")
    print("=" * 60)

    manifest = (
        Path(__file__).parent
        / "manifests"
        / "train_manifest.csv"
    )

    if not manifest.exists():

        print()
        print(
            "ERROR: train_manifest.csv not found."
        )

        print(
            f"Expected: {manifest}"
        )

        raise SystemExit(1)

    import csv

    with open(
        manifest,
        "r",
        encoding="utf-8",
    ) as f:

        reader = csv.DictReader(f)

        first_record = next(reader)

    audio_path = first_record[
        "audio_path"
    ]

    print()
    print("Test audio:")
    print(audio_path)

    print()
    print("Label:")
    print(first_record["label"])

    # --------------------------------------------------------
    # Standard extraction
    # --------------------------------------------------------

    print()
    print(
        "Testing standard extraction..."
    )

    feature = extract_feature_tensor(
        audio_path
    )

    print(
        f"Standard shape: {feature.shape}"
    )

    print(
        f"Standard dtype: {feature.dtype}"
    )

    # --------------------------------------------------------
    # Training extraction
    # --------------------------------------------------------

    print()
    print(
        "Testing training extraction..."
    )

    augmented_feature = (
        extract_training_feature_tensor(
            audio_path
        )
    )

    print(
        f"Training shape: "
        f"{augmented_feature.shape}"
    )

    print(
        f"Training dtype: "
        f"{augmented_feature.dtype}"
    )

    print()
    print("=" * 60)
    print(
        "FEATURE EXTRACTION V2 TEST PASSED"
    )
    print("=" * 60)