import numpy as np
import librosa


class ProsodyAnalyzer:

    TARGET_SR = 16000

    def analyze(self, audio_path):

        audio, sr = librosa.load(
            audio_path,
            sr=self.TARGET_SR,
            mono=True
        )

        audio = audio.astype(np.float32)

        if len(audio) == 0:
            return {
                "pitch_mean": 0.0,
                "pitch_std": 0.0,
                "pitch_variation": 0.0,
                "energy_mean": 0.0,
                "energy_std": 0.0,
                "energy_variation": 0.0,
                "zero_crossing_rate": 0.0,
                "spectral_centroid": 0.0,
                "spectral_bandwidth": 0.0,
                "spectral_rolloff": 0.0
            }

        # Remove DC offset
        audio = audio - np.mean(audio)

        # -------------------------------------------------
        # 1. Pitch / F0
        # -------------------------------------------------

        f0, voiced_flag, voiced_prob = librosa.pyin(
            audio,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=sr
        )

        valid_f0 = f0[np.isfinite(f0)]

        if len(valid_f0) > 0:
            pitch_mean = float(np.mean(valid_f0))
            pitch_std = float(np.std(valid_f0))
        else:
            pitch_mean = 0.0
            pitch_std = 0.0

        if pitch_mean > 0:
            pitch_variation = pitch_std / pitch_mean
        else:
            pitch_variation = 0.0

        # -------------------------------------------------
        # 2. Energy
        # -------------------------------------------------

        rms = librosa.feature.rms(y=audio)[0]

        energy_mean = float(np.mean(rms))
        energy_std = float(np.std(rms))

        if energy_mean > 0:
            energy_variation = energy_std / energy_mean
        else:
            energy_variation = 0.0

        # -------------------------------------------------
        # 3. Zero Crossing Rate
        # -------------------------------------------------

        zcr = librosa.feature.zero_crossing_rate(audio)[0]

        zero_crossing_rate = float(np.mean(zcr))

        # -------------------------------------------------
        # 4. Spectral Features
        # -------------------------------------------------

        spectral_centroid = librosa.feature.spectral_centroid(
            y=audio,
            sr=sr
        )[0]

        spectral_bandwidth = librosa.feature.spectral_bandwidth(
            y=audio,
            sr=sr
        )[0]

        spectral_rolloff = librosa.feature.spectral_rolloff(
            y=audio,
            sr=sr
        )[0]

        return {
            "pitch_mean": pitch_mean,
            "pitch_std": pitch_std,
            "pitch_variation": float(pitch_variation),

            "energy_mean": energy_mean,
            "energy_std": energy_std,
            "energy_variation": float(energy_variation),

            "zero_crossing_rate": zero_crossing_rate,

            "spectral_centroid": float(np.mean(spectral_centroid)),
            "spectral_bandwidth": float(np.mean(spectral_bandwidth)),
            "spectral_rolloff": float(np.mean(spectral_rolloff))
        }


if __name__ == "__main__":

    import sys

    if len(sys.argv) < 2:
        print("Usage: python prosody.py <audio.wav>")
        sys.exit(1)

    analyzer = ProsodyAnalyzer()

    result = analyzer.analyze(sys.argv[1])

    print("\n========== ACOUSTIC ANALYSIS ==========")

    for key, value in result.items():
        print(f"{key:25}: {value:.6f}")

    print("========================================")