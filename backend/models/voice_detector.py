import os
import joblib
import numpy as np
import librosa


# ============================================================
# VIGILVOICE - FAST VOICE SPOOF DETECTOR
# Random Forest Acoustic Model
# ============================================================

class VoiceSpoofDetector:

    def __init__(self):

        print("=" * 60)
        print("VIGILVOICE - FAST VOICE AUTHENTICITY MODEL")
        print("=" * 60)

        self.sample_rate = 16000

        # ----------------------------------------------------
        # MODEL PATH
        # ----------------------------------------------------

        base_dir = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                ".."
            )
        )

        self.model_path = os.path.join(
            base_dir,
            "models",
            "voice_authenticity_model.pkl"
        )

        print("Loading Random Forest model...")

        if not os.path.exists(self.model_path):

            raise FileNotFoundError(
                f"Model not found: {self.model_path}"
            )

        self.model = joblib.load(
            self.model_path
        )

        print(
            "Model:",
            type(self.model).__name__
        )

        print(
            "Classes:",
            self.model.classes_
        )

        print(
            "Model loaded successfully."
        )

        print("=" * 60)

    # ========================================================
    # FEATURE EXTRACTION
    # EXACT SAME 52 FEATURES USED DURING TRAINING
    # ========================================================

    def extract_features(self, audio_path):

        audio, sr = librosa.load(
            audio_path,
            sr=self.sample_rate,
            mono=True,
            duration=4.0
        )

        if len(audio) < 1000:

            raise ValueError(
                "Audio is too short for analysis."
            )

        audio = audio.astype(
            np.float32
        )

        # ----------------------------------------------------
        # MFCC
        # 20 mean + 20 std = 40
        # ----------------------------------------------------

        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=20
        )

        mfcc_mean = np.mean(
            mfcc,
            axis=1
        )

        mfcc_std = np.std(
            mfcc,
            axis=1
        )

        # ----------------------------------------------------
        # SPECTRAL CENTROID
        # ----------------------------------------------------

        centroid = librosa.feature.spectral_centroid(
            y=audio,
            sr=sr
        )

        # ----------------------------------------------------
        # SPECTRAL BANDWIDTH
        # ----------------------------------------------------

        bandwidth = librosa.feature.spectral_bandwidth(
            y=audio,
            sr=sr
        )

        # ----------------------------------------------------
        # SPECTRAL ROLLOFF
        # ----------------------------------------------------

        rolloff = librosa.feature.spectral_rolloff(
            y=audio,
            sr=sr
        )

        # ----------------------------------------------------
        # ZERO CROSSING RATE
        # ----------------------------------------------------

        zcr = librosa.feature.zero_crossing_rate(
            audio
        )

        # ----------------------------------------------------
        # RMS ENERGY
        # ----------------------------------------------------

        rms = librosa.feature.rms(
            y=audio
        )[0]

        # ----------------------------------------------------
        # PITCH
        # ----------------------------------------------------

        try:

            f0, _, _ = librosa.pyin(
                audio,
                fmin=librosa.note_to_hz("C2"),
                fmax=librosa.note_to_hz("C7"),
                sr=sr
            )

            valid_f0 = f0[
                np.isfinite(f0)
            ]

            if len(valid_f0) > 0:

                pitch_mean = float(
                    np.mean(valid_f0)
                )

                pitch_std = float(
                    np.std(valid_f0)
                )

            else:

                pitch_mean = 0.0
                pitch_std = 0.0

        except Exception:

            pitch_mean = 0.0
            pitch_std = 0.0

        # ----------------------------------------------------
        # FINAL 52 FEATURES
        # ----------------------------------------------------

        features = np.concatenate([

            mfcc_mean,

            mfcc_std,

            [
                np.mean(centroid),
                np.std(centroid),

                np.mean(bandwidth),
                np.std(bandwidth),

                np.mean(rolloff),
                np.std(rolloff),

                np.mean(zcr),
                np.std(zcr),

                np.mean(rms),
                np.std(rms),

                pitch_mean,
                pitch_std
            ]

        ])

        return features.astype(
            np.float32
        )

    # ========================================================
    # PREDICT
    # ========================================================

    def predict(self, audio_path):

        print()
        print("-" * 60)

        print(
            "Analyzing:",
            audio_path
        )

        if not os.path.exists(audio_path):

            raise FileNotFoundError(
                f"Audio file not found: {audio_path}"
            )

        # ----------------------------------------------------
        # AUDIO INFORMATION
        # ----------------------------------------------------

        try:

            duration = librosa.get_duration(
                path=audio_path
            )

        except Exception:

            duration = 0.0

        print(
            f"Duration: {duration:.2f} seconds"
        )

        # ----------------------------------------------------
        # EXTRACT FEATURES
        # ----------------------------------------------------

        print(
            "Extracting 52 acoustic features..."
        )

        features = self.extract_features(
            audio_path
        )

        print(
            "Feature vector:",
            features.shape
        )

        # ----------------------------------------------------
        # MODEL INPUT
        # ----------------------------------------------------

        X = features.reshape(
            1,
            -1
        )

        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        prediction = int(
            self.model.predict(X)[0]
        )

        probabilities = (
            self.model.predict_proba(X)[0]
        )

        # ----------------------------------------------------
        # CLASS MAPPING
        #
        # 0 = REAL
        # 1 = AI/SPOOF
        # ----------------------------------------------------

        class_to_probability = {
            int(cls): float(prob)
            for cls, prob
            in zip(
                self.model.classes_,
                probabilities
            )
        }

        real_probability = (
            class_to_probability.get(
                0,
                0.0
            )
        )

        spoof_probability = (
            class_to_probability.get(
                1,
                0.0
            )
        )

        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        verdict = (
            "SPOOF"
            if prediction == 1
            else "REAL"
        )

        print()
        print(
            "Random Forest RESULT"
        )

        print(
            f"Real probability : "
            f"{real_probability:.6f}"
        )

        print(
            f"Spoof probability: "
            f"{spoof_probability:.6f}"
        )

        print(
            "Prediction       :",
            verdict
        )

        print("-" * 60)

        return {

            "real_probability":
                round(
                    real_probability,
                    6
                ),

            "spoof_probability":
                round(
                    spoof_probability,
                    6
                ),

            "rf_real_probability":
                round(
                    real_probability,
                    6
                ),

            "rf_spoof_probability":
                round(
                    spoof_probability,
                    6
                ),

            "prediction":
                prediction,

            "model":
                "VIGILVOICE Random Forest",

            "duration":
                round(
                    duration,
                    3
                ),

            "sample_rate":
                self.sample_rate,

            # Compatibility fields
            "max_spoof_probability":
                round(
                    spoof_probability,
                    6
                ),

            "strong_spoof_segments":
                1 if spoof_probability >= 0.70 else 0,

            "num_segments":
                1
        }


# ============================================================
# COMMAND LINE TEST
# ============================================================

if __name__ == "__main__":

    import sys

    if len(sys.argv) < 2:

        print()
        print(
            "Usage:"
        )

        print(
            'python backend\\intelligence\\voice_detector.py "audio.wav"'
        )

        sys.exit(1)

    audio_file = sys.argv[1]

    detector = VoiceSpoofDetector()

    result = detector.predict(
        audio_file
    )

    print()
    print("=" * 60)
    print("FINAL RESULT")
    print("=" * 60)

    print(result)