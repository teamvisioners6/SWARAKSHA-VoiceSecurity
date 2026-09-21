import pyttsx3

OUTPUT = "synthetic.wav"

engine = pyttsx3.init()

# Use Microsoft David
voices = engine.getProperty("voices")
for voice in voices:
    if "David" in voice.name:
        engine.setProperty("voice", voice.id)
        break

engine.setProperty("rate", 155)
engine.setProperty("volume", 1.0)

text = (
    "This is a security verification test for VigilVoice. "
    "Please verify the caller before approving any sensitive transaction."
)

print("Generating synthetic speech...")
engine.save_to_file(text, OUTPUT)
engine.runAndWait()

print(f"Synthetic audio saved: {OUTPUT}")