import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from features import extract_feature_tensor
from model import SwarakshaCNN


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "swaraksha_cnn_best.pt"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 60)
print("SWARAKSHA - CUSTOM CNN INFERENCE")
print("=" * 60)

print()
print(f"Device : {DEVICE}")

if DEVICE.type == "cuda":
    print(
        f"GPU    : {torch.cuda.get_device_name(0)}"
    )

print()
print(f"Model  : {MODEL_PATH}")


if not MODEL_PATH.exists():

    raise FileNotFoundError(
        f"Model not found:\n{MODEL_PATH}"
    )


model = SwarakshaCNN(
    num_classes=2
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

# Training script saved model_state_dict
if "model_state_dict" in checkpoint:

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

else:

    # Fallback if raw state_dict was saved
    model.load_state_dict(
        checkpoint
    )


model.to(DEVICE)
model.eval()


print("Model loaded successfully.")


# ============================================================
# PREDICT
# ============================================================

def predict(audio_path):

    audio_path = Path(audio_path)

    if not audio_path.exists():

        raise FileNotFoundError(
            f"Audio file not found:\n{audio_path}"
        )

    print()
    print("-" * 60)
    print(f"Audio: {audio_path.name}")
    print("-" * 60)

    start = time.perf_counter()

    # --------------------------------------------------------
    # Feature extraction
    # --------------------------------------------------------

    feature = extract_feature_tensor(
        str(audio_path)
    )

    if not isinstance(feature, torch.Tensor):

        feature = torch.from_numpy(feature)

    feature = feature.float()

    # Add batch dimension
    if feature.ndim == 3:

        feature = feature.unsqueeze(0)

    feature = feature.to(DEVICE)

    # --------------------------------------------------------
    # Model inference
    # --------------------------------------------------------

    with torch.no_grad():

        if DEVICE.type == "cuda":

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
            ):

                logits = model(feature)

        else:

            logits = model(feature)

        probabilities = F.softmax(
            logits,
            dim=1
        )

    real_probability = (
        probabilities[0, 0]
        .item()
    )

    spoof_probability = (
        probabilities[0, 1]
        .item()
    )

    # --------------------------------------------------------
    # Verdict
    # --------------------------------------------------------

    if spoof_probability >= 0.5:

        verdict = "AI SPOOF"

    else:

        verdict = "REAL / BONAFIDE"

    elapsed = (
        time.perf_counter()
        - start
    )

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    print()
    print(
        f"REAL probability : "
        f"{real_probability * 100:.2f}%"
    )

    print(
        f"AI probability   : "
        f"{spoof_probability * 100:.2f}%"
    )

    print()
    print(
        f"VERDICT          : {verdict}"
    )

    print(
        f"Processing time  : "
        f"{elapsed:.3f} seconds"
    )

    return {
        "audio": str(audio_path),
        "verdict": verdict,
        "real_probability": real_probability,
        "spoof_probability": spoof_probability,
        "processing_time": elapsed,
    }


# ============================================================
# COMMAND LINE
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) < 2:

        print()
        print("Usage:")
        print(
            "python .\\custom_model\\predict.py "
            "\"path_to_audio\""
        )

        print()
        print("Example:")
        print(
            "python .\\custom_model\\predict.py "
            "\"C:\\VIGILVOICE\\synthetic.mp3\""
        )

        sys.exit(1)

    audio_file = sys.argv[1]

    predict(audio_file)