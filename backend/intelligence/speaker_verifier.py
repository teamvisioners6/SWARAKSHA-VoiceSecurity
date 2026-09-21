import os
import torch
import torchaudio
import torch.nn.functional as F

from speechbrain.inference.speaker import EncoderClassifier


class SpeakerVerifier:
    """
    VIGILVOICE Speaker Verification Engine.

    Uses SpeechBrain ECAPA-TDNN to:
    1. Create speaker embeddings
    2. Compare incoming voice with a protected speaker profile
    3. Calculate cosine similarity
    4. Determine speaker match
    """

    # --------------------------------------------------------
    # Speaker matching threshold
    # --------------------------------------------------------
    MATCH_THRESHOLD = 0.75

    def __init__(self):

        self.device = (
            "cuda:0"
            if torch.cuda.is_available()
            else "cpu"
        )

        print(
            f"[SpeakerVerifier] Device: {self.device}"
        )

        print(
            "[SpeakerVerifier] Loading ECAPA-TDNN..."
        )

        self.encoder = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=r"C:\VIGILVOICE\models\ecapa",
            run_opts={
                "device": self.device
            }
        )

        print(
            "[SpeakerVerifier] ECAPA-TDNN loaded successfully"
        )

    # ========================================================
    # LOAD AUDIO
    # ========================================================

    def load_audio(self, audio_path: str):

        if not os.path.exists(audio_path):

            raise FileNotFoundError(
                f"Audio file not found: {audio_path}"
            )

        waveform, sample_rate = torchaudio.load(
            audio_path
        )

        # ----------------------------------------------------
        # Stereo → Mono
        # ----------------------------------------------------

        if waveform.shape[0] > 1:

            waveform = waveform.mean(
                dim=0,
                keepdim=True
            )

        # ----------------------------------------------------
        # Resample → 16 kHz
        # ----------------------------------------------------

        if sample_rate != 16000:

            waveform = torchaudio.functional.resample(
                waveform,
                sample_rate,
                16000
            )

        # ----------------------------------------------------
        # Move to selected device
        # ----------------------------------------------------

        waveform = waveform.to(
            self.device
        )

        return waveform

    # ========================================================
    # CREATE SPEAKER EMBEDDING
    # ========================================================

    def create_embedding(self, audio_path: str):

        """
        Generate a normalized 192-dimensional
        ECAPA-TDNN speaker embedding.
        """

        waveform = self.load_audio(
            audio_path
        )

        with torch.no_grad():

            embedding = self.encoder.encode_batch(
                waveform
            )

        # Expected shape:
        # [1, 1, 192]

        embedding = embedding.squeeze()

        # ----------------------------------------------------
        # L2 normalization
        # ----------------------------------------------------

        embedding = F.normalize(
            embedding,
            p=2,
            dim=0
        )

        return embedding.detach().cpu().numpy()

    # ========================================================
    # COSINE SIMILARITY
    # ========================================================

    @staticmethod
    def cosine_similarity(
        embedding_a,
        embedding_b
    ):

        """
        Calculate cosine similarity between
        two speaker embeddings.
        """

        a = torch.tensor(
            embedding_a,
            dtype=torch.float32
        )

        b = torch.tensor(
            embedding_b,
            dtype=torch.float32
        )

        # Normalize both vectors

        a = F.normalize(
            a,
            p=2,
            dim=0
        )

        b = F.normalize(
            b,
            p=2,
            dim=0
        )

        similarity = torch.dot(
            a,
            b
        ).item()

        return float(similarity)

    # ========================================================
    # SIMILARITY → UI PERCENTAGE
    # ========================================================

    @staticmethod
    def similarity_to_percentage(
        similarity: float
    ):

        """
        Convert cosine similarity into a
        human-readable similarity indicator.

        IMPORTANT:
        This is NOT speaker accuracy
        and NOT probability.
        """

        similarity = max(
            -1.0,
            min(1.0, similarity)
        )

        percentage = (
            (similarity + 1.0)
            / 2.0
        ) * 100.0

        return round(
            percentage,
            2
        )

    # ========================================================
    # VERIFY SPEAKER
    # ========================================================

    def verify_speaker(
        self,
        audio_path: str,
        speaker_profile: dict
    ):

        """
        Compare incoming audio against a protected
        speaker profile stored in MongoDB.
        """

        # ----------------------------------------------------
        # Validate profile
        # ----------------------------------------------------

        if not speaker_profile:

            raise ValueError(
                "Speaker profile is empty"
            )

        stored_embedding = (
            speaker_profile.get(
                "embedding"
            )
        )

        if not stored_embedding:

            raise ValueError(
                "Speaker profile does not contain an embedding"
            )

        # ----------------------------------------------------
        # Create embedding for incoming voice
        # ----------------------------------------------------

        incoming_embedding = (
            self.create_embedding(
                audio_path
            )
        )

        # ----------------------------------------------------
        # Compare embeddings
        # ----------------------------------------------------

        similarity = (
            self.cosine_similarity(
                incoming_embedding,
                stored_embedding
            )
        )

        # ----------------------------------------------------
        # Determine speaker match
        # ----------------------------------------------------

        speaker_match = (
            similarity
            >= self.MATCH_THRESHOLD
        )

        # ----------------------------------------------------
        # Human-readable indicator
        # ----------------------------------------------------

        similarity_percentage = (
            self.similarity_to_percentage(
                similarity
            )
        )

        # ----------------------------------------------------
        # Verdict
        # ----------------------------------------------------

        if speaker_match:

            verdict = "SAME SPEAKER"

        else:

            verdict = "DIFFERENT SPEAKER"

        # ----------------------------------------------------
        # Return complete result
        # ----------------------------------------------------

        return {

            "speaker_id":
                speaker_profile.get(
                    "speaker_id"
                ),

            "speaker_name":
                speaker_profile.get(
                    "name"
                ),

            "cosine_similarity":
                round(
                    similarity,
                    4
                ),

            "similarity_indicator":
                similarity_percentage,

            "match_threshold":
                self.MATCH_THRESHOLD,

            "speaker_match":
                speaker_match,

            "verdict":
                verdict
        }


# ============================================================
# SINGLETON INSTANCE
# ============================================================

speaker_verifier = SpeakerVerifier()