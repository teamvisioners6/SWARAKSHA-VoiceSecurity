import os
import random
import numpy as np
import librosa

# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

ASVSPOOF_BONAFIDE_DIR = r"C:\VIGILVOICE\data\voice_dataset\ASVspoof2019_LA_dev\flac"

REAL_WORLD_DIR = r"C:\VIGILVOICE\data\voice_dataset\real_voice_test_wav"

SAMPLE_RATE = 16000
N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 128

# Number of ASVspoof genuine samples to compare
NUM_ASVSPOOF_SAMPLES = 20

random.seed(42)


# ---------------------------------------------------------
# AUDIO LOADING
# ---------------------------------------------------------

def load_audio(path):
    audio, sr = librosa.load(
        path,
        sr=SAMPLE_RATE,
        mono=True
    )

    audio = audio.astype(np.float32)

    if len(audio) == 0:
        raise ValueError("Empty audio")

    return audio, sr


# ---------------------------------------------------------
# AUDIO STATISTICS
# ---------------------------------------------------------

def audio_statistics(audio):
    rms = float(np.sqrt(np.mean(audio ** 2)))

    peak = float(np.max(np.abs(audio)))

    zcr = float(
        np.mean(
            librosa.feature.zero_crossing_rate(
                audio,
                frame_length=1024,
                hop_length=256
            )
        )
    )

    centroid = float(
        np.mean(
            librosa.feature.spectral_centroid(
                y=audio,
                sr=SAMPLE_RATE,
                n_fft=N_FFT,
                hop_length=HOP_LENGTH
            )
        )
    )

    bandwidth = float(
        np.mean(
            librosa.feature.spectral_bandwidth(
                y=audio,
                sr=SAMPLE_RATE,
                n_fft=N_FFT,
                hop_length=HOP_LENGTH
            )
        )
    )

    rolloff = float(
        np.mean(
            librosa.feature.spectral_rolloff(
                y=audio,
                sr=SAMPLE_RATE,
                n_fft=N_FFT,
                hop_length=HOP_LENGTH,
                roll_percent=0.85
            )
        )
    )

    # Same Mel configuration used by SWARAKSHA CNN
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=20,
        fmax=8000,
        power=2.0
    )

    log_mel = librosa.power_to_db(
        mel,
        ref=np.max
    )

    mel_mean = float(np.mean(log_mel))
    mel_std = float(np.std(log_mel))
    mel_min = float(np.min(log_mel))
    mel_max = float(np.max(log_mel))

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SAMPLE_RATE,
        n_mfcc=20,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )

    mfcc_mean = float(np.mean(mfcc))
    mfcc_std = float(np.std(mfcc))

    duration = len(audio) / SAMPLE_RATE

    return {
        "duration": duration,
        "rms": rms,
        "peak": peak,
        "zcr": zcr,
        "spectral_centroid": centroid,
        "spectral_bandwidth": bandwidth,
        "spectral_rolloff": rolloff,
        "mel_mean": mel_mean,
        "mel_std": mel_std,
        "mel_min": mel_min,
        "mel_max": mel_max,
        "mfcc_mean": mfcc_mean,
        "mfcc_std": mfcc_std,
    }


# ---------------------------------------------------------
# FIND ASVSPOOF BONAFIDE FILES
# ---------------------------------------------------------

def find_asvspoof_files():
    if not os.path.exists(ASVSPOOF_BONAFIDE_DIR):
        raise FileNotFoundError(
            f"ASVspoof directory not found:\n{ASVSPOOF_BONAFIDE_DIR}"
        )

    files = [
        os.path.join(ASVSPOOF_BONAFIDE_DIR, f)
        for f in os.listdir(ASVSPOOF_BONAFIDE_DIR)
        if f.lower().endswith(".flac")
    ]

    random.shuffle(files)

    return files[:NUM_ASVSPOOF_SAMPLES]


# ---------------------------------------------------------
# FIND REAL-WORLD FILES
# ---------------------------------------------------------

def find_real_files():
    if not os.path.exists(REAL_WORLD_DIR):
        raise FileNotFoundError(
            f"Real-world directory not found:\n{REAL_WORLD_DIR}"
        )

    files = [
        os.path.join(REAL_WORLD_DIR, f)
        for f in os.listdir(REAL_WORLD_DIR)
        if f.lower().endswith(".wav")
    ]

    return sorted(files)


# ---------------------------------------------------------
# PRINT INDIVIDUAL RESULT
# ---------------------------------------------------------

def print_result(name, stats):
    print(f"\n{name}")
    print("-" * 60)

    print(f"Duration              : {stats['duration']:.2f} sec")
    print(f"RMS                   : {stats['rms']:.6f}")
    print(f"Peak                  : {stats['peak']:.6f}")
    print(f"Zero Crossing Rate    : {stats['zcr']:.6f}")

    print(
        f"Spectral Centroid     : "
        f"{stats['spectral_centroid']:.2f} Hz"
    )

    print(
        f"Spectral Bandwidth    : "
        f"{stats['spectral_bandwidth']:.2f} Hz"
    )

    print(
        f"Spectral Rolloff      : "
        f"{stats['spectral_rolloff']:.2f} Hz"
    )

    print(f"Mel Mean              : {stats['mel_mean']:.4f}")
    print(f"Mel Std               : {stats['mel_std']:.4f}")
    print(f"Mel Min               : {stats['mel_min']:.4f}")
    print(f"Mel Max               : {stats['mel_max']:.4f}")

    print(f"MFCC Mean             : {stats['mfcc_mean']:.4f}")
    print(f"MFCC Std              : {stats['mfcc_std']:.4f}")


# ---------------------------------------------------------
# GROUP AVERAGE
# ---------------------------------------------------------

def group_average(results):
    keys = results[0].keys()

    return {
        key: float(np.mean([r[key] for r in results]))
        for key in keys
    }


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    print("=" * 70)
    print("SWARAKSHA DOMAIN DIAGNOSTIC")
    print("=" * 70)

    print("\nPurpose:")
    print(
        "Compare ASVspoof genuine speech against "
        "real-world recordings using the same audio domain."
    )

    # -----------------------------------------------------
    # ASVSPOOF
    # -----------------------------------------------------

    print("\n\n[1] Loading ASVspoof genuine samples...")

    asv_files = find_asvspoof_files()

    print(f"Selected {len(asv_files)} ASVspoof files.")

    asv_results = []

    for path in asv_files:

        try:
            audio, sr = load_audio(path)
            stats = audio_statistics(audio)

            asv_results.append(stats)

            print_result(
                os.path.basename(path),
                stats
            )

        except Exception as e:
            print(
                f"\nERROR: {os.path.basename(path)}"
                f"\n{e}"
            )

    # -----------------------------------------------------
    # REAL WORLD
    # -----------------------------------------------------

    print("\n\n[2] Loading real-world recordings...")

    real_files = find_real_files()

    print(f"Found {len(real_files)} real-world WAV files.")

    real_results = []

    for path in real_files:

        try:
            audio, sr = load_audio(path)
            stats = audio_statistics(audio)

            real_results.append(stats)

            print_result(
                os.path.basename(path),
                stats
            )

        except Exception as e:
            print(
                f"\nERROR: {os.path.basename(path)}"
                f"\n{e}"
            )

    # -----------------------------------------------------
    # GROUP COMPARISON
    # -----------------------------------------------------

    if not asv_results or not real_results:
        print("\nNot enough data for comparison.")
        return

    asv_avg = group_average(asv_results)
    real_avg = group_average(real_results)

    print("\n\n" + "=" * 70)
    print("GROUP COMPARISON")
    print("=" * 70)

    print(
        f"\n{'Feature':<25}"
        f"{'ASVspoof Genuine':>20}"
        f"{'Real World':>20}"
        f"{'Difference %':>18}"
    )

    print("-" * 83)

    features = [
        "duration",
        "rms",
        "peak",
        "zcr",
        "spectral_centroid",
        "spectral_bandwidth",
        "spectral_rolloff",
        "mel_mean",
        "mel_std",
        "mel_min",
        "mel_max",
        "mfcc_mean",
        "mfcc_std",
    ]

    for feature in features:

        a = asv_avg[feature]
        r = real_avg[feature]

        if abs(a) > 1e-10:
            diff = abs(r - a) / abs(a) * 100
        else:
            diff = 0.0

        print(
            f"{feature:<25}"
            f"{a:>20.6f}"
            f"{r:>20.6f}"
            f"{diff:>17.2f}%"
        )

    # -----------------------------------------------------
    # INTERPRETATION
    # -----------------------------------------------------

    print("\n\n" + "=" * 70)
    print("DIAGNOSTIC NOTES")
    print("=" * 70)

    print("""
This diagnostic does NOT prove the cause of the model's
real-world failure by itself.

We are looking for evidence of domain differences such as:

1. Recording loudness / microphone differences
2. Spectral bandwidth differences
3. Telephone / mobile recording characteristics
4. Background noise
5. Different speech characteristics
6. Different recording environments
7. Feature-distribution mismatch

If several features show large differences, we should address
the training-data domain before changing the CNN architecture.

IMPORTANT:
Do not use the five real-world recordings as the final test
set after using them to tune the model.
""")

    print("\nDiagnostic completed.")


if __name__ == "__main__":
    main()