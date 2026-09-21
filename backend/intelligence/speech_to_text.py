from faster_whisper import WhisperModel


class SpeechToText:

    def __init__(self):

        print("\nLoading Whisper AI...")

        self.model = WhisperModel(
            "base",
            device="cpu",
            compute_type="int8"
        )

        print("Whisper AI ready.")

    def transcribe(self, audio_path):

        segments, info = self.model.transcribe(
            audio_path,
            beam_size=5,
            vad_filter=True
        )

        text_parts = []

        for segment in segments:
            text_parts.append(
                segment.text.strip()
            )

        text = " ".join(text_parts)

        return {
            "text": text,
            "language": info.language,
            "language_probability": round(
                info.language_probability,
                4
            )
        }