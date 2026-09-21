import os
import warnings
import numpy as np
import librosa
import onnxruntime as ort


class VoiceDetector:
    """
    VIGILVOICE - Spectra-AASIST3 detector

    Spectra-AASIST3:
        input  : wav [batch, 64600] float32
        output : logits [batch, 2]

    Model documentation:
        index 0 = spoof
        index 1 = bona fide / real

    Higher index-1 score = more bona fide.
    """

    SAMPLE_RATE = 16000
    TARGET_LENGTH = 64600

    def __init__(self):
        self.model_path = (
            r"C:\VIGILVOICE\models\spectra_aasist3"
            r"\spectra-aasist3.onnx"
        )

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Spectra-AASIST3 model not found:\n{self.model_path}"
            )

        # Load NVIDIA CUDA/cuDNN DLLs when available.
        try:
            ort.preload_dlls(directory="")
        except Exception:
            pass

        available = ort.get_available_providers()

        if "CUDAExecutionProvider" in available:
            providers = [
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ]
        else:
            providers = ["CPUExecutionProvider"]

        print("\n========================================")
        print(" VIGILVOICE - Spectra-AASIST3")
        print("========================================")
        print("Model:", self.model_path)
        print("Available providers:", available)
        print("Selected providers:", providers)

        self.session = ort.InferenceSession(
            self.model_path,
            providers=providers,
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        print("Input :", self.input_name)
        print("Output:", self.output_name)
        print("Actual providers:", self.session.get_providers())
        print("========================================\n")

    # ---------------------------------------------------------
    # AUDIO LOADING
    # ---------------------------------------------------------

    def load_audio(self, file_path):
        """
        Load audio as mono float32 at 16 kHz.
        """

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            audio, sr = librosa.load(
                file_path,
                sr=self.SAMPLE_RATE,
                mono=True,
            )

        audio = np.asarray(audio, dtype=np.float32)

        if audio.size == 0:
            raise ValueError("Audio file contains no samples.")

        # Remove NaN / Inf
        audio = np.nan_to_num(
            audio,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        return audio

    # ---------------------------------------------------------
    # PREPROCESSING
    # ---------------------------------------------------------

    def preprocess(self, audio):
        """
        Match Spectra-AASIST3 documented preprocessing:

        1. float32 mono 16 kHz
        2. preemphasis = 0.97
        3. deterministic first 64600 samples
        4. tile-repeat if shorter
        """

        audio = np.asarray(audio, dtype=np.float32)

        if len(audio) == 0:
            raise ValueError("Empty audio.")

        # Preemphasis
        emphasized = np.empty_like(audio)

        emphasized[0] = audio[0]

        if len(audio) > 1:
            emphasized[1:] = (
                audio[1:] - 0.97 * audio[:-1]
            )

        audio = emphasized

        # Exactly 64600 samples.
        if len(audio) < self.TARGET_LENGTH:

            repeats = int(
                np.ceil(
                    self.TARGET_LENGTH / len(audio)
                )
            )

            audio = np.tile(audio, repeats)

        # Deterministic FIRST window.
        audio = audio[:self.TARGET_LENGTH]

        # Final float32
        audio = np.asarray(
            audio,
            dtype=np.float32,
        )

        return audio

    # ---------------------------------------------------------
    # LOGIT → PROBABILITY
    # ---------------------------------------------------------

    @staticmethod
    def softmax(logits):
        """
        Numerically stable softmax.
        """

        logits = np.asarray(
            logits,
            dtype=np.float32,
        )

        logits = logits - np.max(
            logits,
            axis=-1,
            keepdims=True,
        )

        exp_logits = np.exp(logits)

        return exp_logits / np.sum(
            exp_logits,
            axis=-1,
            keepdims=True,
        )

    # ---------------------------------------------------------
    # SINGLE INFERENCE
    # ---------------------------------------------------------

    def predict_segment(self, audio):
        """
        Returns:

            spoof_probability
            bona_fide_probability
            raw logits
        """

        processed = self.preprocess(audio)

        batch = processed.reshape(
            1,
            self.TARGET_LENGTH,
        ).astype(np.float32)

        outputs = self.session.run(
            [self.output_name],
            {
                self.input_name: batch
            },
        )

        logits = np.asarray(
            outputs[0],
            dtype=np.float32,
        )

        probabilities = self.softmax(logits)[0]

        spoof_probability = float(
            probabilities[0]
        )

        bona_fide_probability = float(
            probabilities[1]
        )

        return {
            "spoof_probability": spoof_probability,
            "bona_fide_probability": bona_fide_probability,
            "logits": logits[0].tolist(),
        }

    # ---------------------------------------------------------
    # FULL AUDIO ANALYSIS
    # ---------------------------------------------------------

    def predict(self, file_path):
        """
        Analyze the uploaded audio.

        For long audio, evaluate deterministic 4.04-second
        chunks. Each chunk is independently processed using
        the official Spectra-AASIST3 preprocessing.

        Final decision is based on the distribution of
        spoof probabilities across chunks.
        """

        audio = self.load_audio(file_path)

        duration = len(audio) / self.SAMPLE_RATE

        # For short audio, one official model window.
        if len(audio) <= self.TARGET_LENGTH:

            result = self.predict_segment(audio)

            spoof = result["spoof_probability"]
            real = result["bona_fide_probability"]

            prediction = (
                "SPOOF"
                if spoof >= real
                else "REAL"
            )

            return {
                "prediction": prediction,
                "spoof_probability": round(
                    spoof,
                    6,
                ),
                "real_probability": round(
                    real,
                    6,
                ),
                "confidence": round(
                    max(spoof, real),
                    6,
                ),
                "duration": round(
                    duration,
                    3,
                ),
                "segments": 1,
                "segment_results": [result],
                "model": "Spectra-AASIST3",
                "provider": self.session.get_providers()[0],
            }

        # -----------------------------------------------------
        # LONG AUDIO
        # -----------------------------------------------------

        # 4.04 second windows with 50% overlap.
        hop = self.TARGET_LENGTH // 2

        starts = list(
            range(
                0,
                len(audio) - self.TARGET_LENGTH + 1,
                hop,
            )
        )

        # Make sure the end of the recording is covered.
        last_start = len(audio) - self.TARGET_LENGTH

        if starts[-1] != last_start:
            starts.append(last_start)

        segment_results = []

        for start in starts:

            segment = audio[
                start:start + self.TARGET_LENGTH
            ]

            result = self.predict_segment(
                segment
            )

            result["start_time"] = round(
                start / self.SAMPLE_RATE,
                3,
            )

            result["end_time"] = round(
                (start + self.TARGET_LENGTH)
                / self.SAMPLE_RATE,
                3,
            )

            segment_results.append(result)

        spoof_scores = np.array(
            [
                x["spoof_probability"]
                for x in segment_results
            ],
            dtype=np.float32,
        )

        real_scores = np.array(
            [
                x["bona_fide_probability"]
                for x in segment_results
            ],
            dtype=np.float32,
        )

        # Robust aggregation.
        avg_spoof = float(
            np.mean(spoof_scores)
        )

        median_spoof = float(
            np.median(spoof_scores)
        )

        max_spoof = float(
            np.max(spoof_scores)
        )

        avg_real = float(
            np.mean(real_scores)
        )

        strong_spoof_count = int(
            np.sum(spoof_scores >= 0.80)
        )

        moderate_spoof_count = int(
            np.sum(spoof_scores >= 0.50)
        )

        total_segments = len(
            segment_results
        )

        strong_ratio = (
            strong_spoof_count
            / total_segments
        )

        moderate_ratio = (
            moderate_spoof_count
            / total_segments
        )

        # Main score.
        #
        # Average is the strongest signal.
        # Median prevents one abnormal segment
        # from dominating.
        #
        # Strong segment ratio provides evidence
        # when a substantial portion is spoofed.
        segment_score = (
            avg_spoof * 0.45
            + median_spoof * 0.30
            + strong_ratio * 0.15
            + moderate_ratio * 0.10
        )

        # Final decision.
        #
        # We intentionally keep a three-level output.
        if segment_score >= 0.65:
            prediction = "SPOOF"
        elif segment_score >= 0.40:
            prediction = "SUSPICIOUS"
        else:
            prediction = "REAL"

        confidence = max(
            segment_score,
            1.0 - segment_score,
        )

        return {
            "prediction": prediction,

            "spoof_probability": round(
                avg_spoof,
                6,
            ),

            "real_probability": round(
                avg_real,
                6,
            ),

            "median_spoof_probability": round(
                median_spoof,
                6,
            ),

            "max_spoof_probability": round(
                max_spoof,
                6,
            ),

            "segment_score": round(
                segment_score,
                6,
            ),

            "confidence": round(
                confidence,
                6,
            ),

            "duration": round(
                duration,
                3,
            ),

            "segments": total_segments,

            "strong_spoof_segments":
                strong_spoof_count,

            "moderate_spoof_segments":
                moderate_spoof_count,

            "strong_spoof_ratio": round(
                strong_ratio,
                6,
            ),

            "moderate_spoof_ratio": round(
                moderate_ratio,
                6,
            ),

            "segment_results":
                segment_results,

            "model":
                "Spectra-AASIST3",

            "provider":
                self.session.get_providers()[0],
        }