import subprocess
from pathlib import Path

INPUT = Path(r"..\my_real_voice.wav")
OUTPUT_DIR = Path("real_segments")
OUTPUT_DIR.mkdir(exist_ok=True)

subprocess.run([
    "ffmpeg",
    "-y",
    "-i", str(INPUT),
    "-f", "segment",
    "-segment_time", "4",
    "-ar", "16000",
    "-ac", "1",
    "-c:a", "pcm_s16le",
    str(OUTPUT_DIR / "segment_%02d.wav")
], check=True)

print("\nCreated segments:")
for f in sorted(OUTPUT_DIR.glob("*.wav")):
    print(f)