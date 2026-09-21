import sounddevice as sd
import soundfile as sf

SAMPLE_RATE = 16000
DURATION = 5

print("====================================")
print("       VIGILVOICE AUDIO TEST")
print("====================================")
print()
print("Recording for 5 seconds...")
print("Please speak naturally.")
print()

audio = sd.rec(
    int(DURATION * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype="float32"
)

sd.wait()

sf.write(
    "test.wav",
    audio,
    SAMPLE_RATE
)

print()
print("Recording complete!")
print("Saved: test.wav")