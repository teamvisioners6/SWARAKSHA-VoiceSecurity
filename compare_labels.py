from datasets import load_dataset
import soundfile as sf
from backend.intelligence.voice_detector import VoiceSpoofDetector

print("Loading dataset...")

ds = load_dataset(
    "SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA",
    split="test",
    streaming=True
)

detector = VoiceSpoofDetector()

found = {}

for sample in ds:
    label = sample["label"]

    if label not in found:
        found[label] = sample

    if 0 in found and 1 in found:
        break

for label in [0, 1]:
    sample = found[label]

    filename = f"sample_label_{label}.wav"

    sf.write(
        filename,
        sample["audio"]["array"],
        sample["audio"]["sampling_rate"]
    )

    result = detector.predict(filename)

    print("\n==============================")
    print("DATASET LABEL :", label)
    print("FILE          :", sample["path"])
    print("LOGITS        :", result["logits"])
    print("PROBABILITIES :", result["probabilities"])
    print("==============================")