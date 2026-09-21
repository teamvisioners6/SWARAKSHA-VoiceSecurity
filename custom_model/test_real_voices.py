from pathlib import Path

import torch
import torch.nn.functional as F

from features import extract_feature_tensor
from model import SwarakshaCNN


# ============================================================
# SWARAKSHA - REAL WORLD VOICE TEST
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR
    / "checkpoints"
    / "swaraksha_cnn_best.pt"
)

TEST_DIR = (
    BASE_DIR.parent
    / "data"
    / "voice_dataset"
    / "real_voice_test_wav"
)

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# Load model
# ============================================================

print("=" * 70)
print("SWARAKSHA - REAL WORLD VOICE EVALUATION")
print("=" * 70)

print()
print("Device:")
print(DEVICE)

print()
print("Model:")
print(MODEL_PATH)

if not MODEL_PATH.exists():

    raise FileNotFoundError(
        f"Model not found:\n{MODEL_PATH}"
    )


model = SwarakshaCNN(
    num_classes=2
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
)

# ------------------------------------------------------------
# Handle checkpoint format
# ------------------------------------------------------------

if "model_state_dict" in checkpoint:

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

else:

    model.load_state_dict(
        checkpoint
    )


model = model.to(
    DEVICE
)

model.eval()


# ============================================================
# Find test files
# ============================================================

if not TEST_DIR.exists():

    raise FileNotFoundError(
        f"Test directory not found:\n{TEST_DIR}"
    )


audio_extensions = {
    ".wav",
    ".mp3",
    ".m4a",
    ".ogg",
    ".flac",
    ".aac",
    ".mpeg",
    ".mpg",
}


audio_files = sorted(
    [
        p
        for p in TEST_DIR.iterdir()
        if p.is_file()
        and p.suffix.lower()
        in audio_extensions
    ]
)


if len(audio_files) == 0:

    raise RuntimeError(
        f"No audio files found in:\n{TEST_DIR}"
    )


print()
print("Test directory:")
print(TEST_DIR)

print()
print(
    f"Real-world recordings found: "
    f"{len(audio_files)}"
)


# ============================================================
# Prediction function
# ============================================================

def predict_audio(
    audio_path: Path,
):

    feature = extract_feature_tensor(
        str(audio_path)
    )

    # [1, 128, time]
    feature = feature.unsqueeze(0)

    # [1, 1, 128, time]
    feature = feature.to(
        DEVICE
    )

    with torch.no_grad():

        logits = model(
            feature
        )

        probabilities = F.softmax(
            logits,
            dim=1,
        )[0]

    real_probability = (
        probabilities[0]
        .item()
    )

    ai_probability = (
        probabilities[1]
        .item()
    )

    if ai_probability >= real_probability:

        verdict = "AI SPOOF"

    else:

        verdict = "REAL"

    return (
        real_probability,
        ai_probability,
        verdict,
    )


# ============================================================
# Run evaluation
# ============================================================

results = []


print()
print("=" * 70)
print("INDIVIDUAL RESULTS")
print("=" * 70)


for index, audio_path in enumerate(
    audio_files,
    start=1,
):

    print()
    print(
        f"[{index}/{len(audio_files)}] "
        f"{audio_path.name}"
    )

    try:

        (
            real_probability,
            ai_probability,
            verdict,
        ) = predict_audio(
            audio_path
        )

        real_percent = (
            real_probability
            * 100.0
        )

        ai_percent = (
            ai_probability
            * 100.0
        )

        print(
            f"REAL probability : "
            f"{real_percent:.2f}%"
        )

        print(
            f"AI probability   : "
            f"{ai_percent:.2f}%"
        )

        print(
            f"VERDICT          : "
            f"{verdict}"
        )

        results.append(
            {
                "file": audio_path.name,
                "real_probability":
                    real_probability,
                "ai_probability":
                    ai_probability,
                "verdict": verdict,
            }
        )

    except Exception as e:

        print(
            f"ERROR: {e}"
        )


# ============================================================
# Summary
# ============================================================

print()
print("=" * 70)
print("REAL-WORLD TEST SUMMARY")
print("=" * 70)


successful_results = len(
    results
)

real_predictions = sum(
    1
    for result in results
    if result["verdict"] == "REAL"
)

ai_predictions = sum(
    1
    for result in results
    if result["verdict"] == "AI SPOOF"
)


print()
print(
    f"Files tested       : "
    f"{successful_results}"
)

print(
    f"Predicted REAL     : "
    f"{real_predictions}"
)

print(
    f"Predicted AI       : "
    f"{ai_predictions}"
)


if successful_results > 0:

    real_detection_rate = (
        real_predictions
        / successful_results
        * 100.0
    )

    print()
    print(
        f"REAL detection rate: "
        f"{real_detection_rate:.2f}%"
    )


print()
print("=" * 70)
print("EVALUATION COMPLETE")
print("=" * 70)