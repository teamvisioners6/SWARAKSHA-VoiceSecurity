import torch
import torch.nn as nn
import librosa
import numpy as np


class VoiceCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(256, 128),

            nn.ReLU(),

            nn.Dropout(0.35),

            nn.Linear(128, 2)
        )

    def forward(self, x):

        return self.classifier(
            self.features(x)
        )


class TrainedVoiceDetector:

    def __init__(
        self,
        model_path="models/vigilvoice_cnn.pth"
    ):

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        checkpoint = torch.load(
            model_path,
            map_location=self.device
        )

        self.model = VoiceCNN().to(
            self.device
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.model.eval()

        self.sr = checkpoint.get(
            "sample_rate",
            16000
        )

        self.n_mels = checkpoint.get(
            "n_mels",
            64
        )

        self.n_fft = checkpoint.get(
            "n_fft",
            1024
        )

        self.hop_length = checkpoint.get(
            "hop_length",
            256
        )

        print(
            "VIGILVOICE trained model loaded"
        )

    def predict(self, audio_path):

        audio, sr = librosa.load(
            audio_path,
            sr=self.sr,
            mono=True
        )

        audio = audio.astype(
            np.float32
        )

        audio = audio - np.mean(audio)

        peak = np.max(
            np.abs(audio)
        )

        if peak > 0:
            audio = audio / peak

        samples = self.sr * 4

        if len(audio) < samples:

            audio = np.pad(
                audio,
                (0, samples - len(audio))
            )

        else:

            audio = audio[:samples]

        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=20,
            fmax=self.sr // 2
        )

        mel = librosa.power_to_db(
            mel,
            ref=np.max
        )

        mel = (
            mel - mel.mean()
        ) / (
            mel.std() + 1e-6
        )

        x = torch.tensor(
            mel,
            dtype=torch.float32
        )

        x = x.unsqueeze(0).unsqueeze(0)

        x = x.to(self.device)

        with torch.no_grad():

            logits = self.model(x)

            probabilities = torch.softmax(
                logits,
                dim=1
            )[0]

        real_probability = float(
            probabilities[0]
        )

        spoof_probability = float(
            probabilities[1]
        )

        if spoof_probability >= 0.5:

            verdict = "SPOOF"

        else:

            verdict = "REAL"

        return {
            "verdict": verdict,
            "real_probability":
                real_probability,
            "spoof_probability":
                spoof_probability
        }


if __name__ == "__main__":

    import sys

    if len(sys.argv) < 2:

        print(
            "Usage: python trained_detector.py audio.wav"
        )

        raise SystemExit

    detector = TrainedVoiceDetector()

    result = detector.predict(
        sys.argv[1]
    )

    print()
    print("========== VIGILVOICE ==========")
    print(
        "Verdict:",
        result["verdict"]
    )

    print(
        "Real:",
        round(
            result["real_probability"] * 100,
            2
        ),
        "%"
    )

    print(
        "AI/Spoof:",
        round(
            result["spoof_probability"] * 100,
            2
        ),
        "%"
    )

    print("================================")