import os
import random
import numpy as np
import librosa
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ============================================================
# CONFIG
# ============================================================

DATASET_DIR = r"C:\VIGILVOICE\data\voice_dataset"

AUDIO_DIR = os.path.join(
    DATASET_DIR,
    "ASVspoof2019_LA_train",
    "flac"
)

PROTOCOL_FILE = os.path.join(
    DATASET_DIR,
    "ASVspoof2019_LA_cm_protocols",
    "ASVspoof2019.LA.cm.train.trn.txt"
)

MODEL_DIR = r"C:\VIGILVOICE\models"

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "stft_lf_cnn.pth"
)

SAMPLE_RATE = 16000
DURATION = 2.0
NUM_SAMPLES = int(SAMPLE_RATE * DURATION)

MAX_REAL = 2500
MAX_SPOOF = 2500

BATCH_SIZE = 32
EPOCHS = 12
LEARNING_RATE = 0.001

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("VIGILVOICE - STFT + LINEAR FILTERBANK CNN")
print("=" * 70)

print(f"Device: {DEVICE}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ============================================================
# LINEAR FILTERBANK
# ============================================================

def create_linear_filterbank(
    sample_rate=16000,
    n_fft=512,
    n_filters=40,
    min_freq=50,
    max_freq=7600
):
    """
    Creates triangular filters spaced linearly in frequency.
    """

    freqs = np.linspace(
        0,
        sample_rate / 2,
        n_fft // 2 + 1
    )

    filter_points = np.linspace(
        min_freq,
        max_freq,
        n_filters + 2
    )

    filterbank = np.zeros(
        (n_filters, len(freqs)),
        dtype=np.float32
    )

    for i in range(n_filters):

        left = filter_points[i]
        center = filter_points[i + 1]
        right = filter_points[i + 2]

        left_indices = np.where(
            (freqs >= left) &
            (freqs <= center)
        )[0]

        right_indices = np.where(
            (freqs >= center) &
            (freqs <= right)
        )[0]

        if center > left:
            filterbank[i, left_indices] = (
                freqs[left_indices] - left
            ) / (center - left)

        if right > center:
            filterbank[i, right_indices] = (
                right - freqs[right_indices]
            ) / (right - center)

    return filterbank


FILTERBANK = create_linear_filterbank()


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_stft_lf(audio):
    """
    STFT -> power spectrum -> linear filterbank -> log energy
    """

    stft = librosa.stft(
        audio,
        n_fft=512,
        hop_length=160,
        win_length=400,
        window="hann",
        center=True
    )

    magnitude = np.abs(stft)

    power = magnitude ** 2

    filtered = np.dot(
        FILTERBANK,
        power
    )

    filtered = np.log(
        filtered + 1e-8
    )

    # Normalize per utterance
    mean = filtered.mean()
    std = filtered.std() + 1e-6

    filtered = (filtered - mean) / std

    return filtered.astype(np.float32)


# ============================================================
# AUDIO LOADING
# ============================================================

def load_audio(path):

    audio, sr = librosa.load(
        path,
        sr=SAMPLE_RATE,
        mono=True
    )

    if len(audio) < NUM_SAMPLES:

        repeat_count = int(
            np.ceil(NUM_SAMPLES / len(audio))
        )

        audio = np.tile(
            audio,
            repeat_count
        )

    # Random 2-second crop
    if len(audio) > NUM_SAMPLES:

        start = random.randint(
            0,
            len(audio) - NUM_SAMPLES
        )

        audio = audio[
            start:start + NUM_SAMPLES
        ]

    else:
        audio = audio[:NUM_SAMPLES]

    # Prevent numerical problems
    audio = audio.astype(np.float32)

    max_value = np.max(
        np.abs(audio)
    )

    if max_value > 0:
        audio = audio / max_value

    return audio


# ============================================================
# READ PROTOCOL
# ============================================================

def read_protocol():

    real_files = []
    spoof_files = []

    print("\nReading protocol...")

    with open(
        PROTOCOL_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            parts = line.strip().split()

            if len(parts) < 2:
                continue

            file_id = parts[1]

            label = parts[-1].lower()

            audio_path = os.path.join(
                AUDIO_DIR,
                file_id + ".flac"
            )

            if not os.path.exists(audio_path):
                continue

            if label == "bonafide":
                real_files.append(audio_path)

            elif label == "spoof":
                spoof_files.append(audio_path)

    random.shuffle(real_files)
    random.shuffle(spoof_files)

    real_files = real_files[:MAX_REAL]
    spoof_files = spoof_files[:MAX_SPOOF]

    print(f"Real samples : {len(real_files)}")
    print(f"Spoof samples: {len(spoof_files)}")

    files = real_files + spoof_files
    labels = (
        [0] * len(real_files) +
        [1] * len(spoof_files)
    )

    return files, labels


# ============================================================
# DATASET
# ============================================================

class VoiceDataset(Dataset):

    def __init__(
        self,
        files,
        labels
    ):

        self.files = files
        self.labels = labels

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):

        path = self.files[index]
        label = self.labels[index]

        try:

            audio = load_audio(path)

            feature = extract_stft_lf(
                audio
            )

            feature = torch.tensor(
                feature,
                dtype=torch.float32
            )

            # CNN expects:
            # [channel, frequency, time]

            feature = feature.unsqueeze(0)

            label = torch.tensor(
                label,
                dtype=torch.long
            )

            return feature, label

        except Exception as e:

            print(
                f"Error loading {path}: {e}"
            )

            # fallback zero feature
            dummy_audio = np.zeros(
                NUM_SAMPLES,
                dtype=np.float32
            )

            feature = extract_stft_lf(
                dummy_audio
            )

            feature = torch.tensor(
                feature,
                dtype=torch.float32
            ).unsqueeze(0)

            label = torch.tensor(
                label,
                dtype=torch.long
            )

            return feature, label


# ============================================================
# CNN MODEL
# ============================================================

class STFTLFCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(
                1,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(32),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(64),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(128),

            nn.ReLU(),

            nn.AdaptiveAvgPool2d(
                (1, 1)
            )
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                128,
                64
            ),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(
                64,
                2
            )
        )

    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x


# ============================================================
# TRAINING
# ============================================================

def main():

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    files, labels = read_protocol()

    print("\nSplitting dataset...")

    train_files, val_files, train_labels, val_labels = train_test_split(
        files,
        labels,
        test_size=0.20,
        random_state=SEED,
        stratify=labels
    )

    print(
        f"Training samples  : {len(train_files)}"
    )

    print(
        f"Validation samples: {len(val_files)}"
    )

    train_dataset = VoiceDataset(
        train_files,
        train_labels
    )

    val_dataset = VoiceDataset(
        val_files,
        val_labels
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = STFTLFCNN().to(DEVICE)

    print("\nModel:")
    print(model)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    best_accuracy = 0.0

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    for epoch in range(EPOCHS):

        model.train()

        running_loss = 0.0

        train_predictions = []
        train_targets = []

        print(
            f"\nEpoch {epoch + 1}/{EPOCHS}"
        )

        for batch_idx, (x, y) in enumerate(
            train_loader
        ):

            x = x.to(
                DEVICE,
                non_blocking=True
            )

            y = y.to(
                DEVICE,
                non_blocking=True
            )

            optimizer.zero_grad()

            outputs = model(x)

            loss = criterion(
                outputs,
                y
            )

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            train_predictions.extend(
                predictions.detach()
                .cpu()
                .numpy()
            )

            train_targets.extend(
                y.detach()
                .cpu()
                .numpy()
            )

            if (batch_idx + 1) % 20 == 0:

                print(
                    f"  Batch {batch_idx + 1}/"
                    f"{len(train_loader)} "
                    f"Loss: {loss.item():.4f}"
                )

        train_accuracy = accuracy_score(
            train_targets,
            train_predictions
        )

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        model.eval()

        val_predictions = []
        val_targets = []

        val_loss = 0.0

        with torch.no_grad():

            for x, y in val_loader:

                x = x.to(DEVICE)
                y = y.to(DEVICE)

                outputs = model(x)

                loss = criterion(
                    outputs,
                    y
                )

                val_loss += loss.item()

                predictions = torch.argmax(
                    outputs,
                    dim=1
                )

                val_predictions.extend(
                    predictions.cpu().numpy()
                )

                val_targets.extend(
                    y.cpu().numpy()
                )

        val_accuracy = accuracy_score(
            val_targets,
            val_predictions
        )

        avg_train_loss = (
            running_loss /
            len(train_loader)
        )

        avg_val_loss = (
            val_loss /
            len(val_loader)
        )

        print(
            f"\nTrain Loss: {avg_train_loss:.4f}"
        )

        print(
            f"Train Acc : {train_accuracy * 100:.2f}%"
        )

        print(
            f"Val Loss  : {avg_val_loss:.4f}"
        )

        print(
            f"Val Acc   : {val_accuracy * 100:.2f}%"
        )

        # ----------------------------------------------------
        # SAVE BEST MODEL
        # ----------------------------------------------------

        if val_accuracy > best_accuracy:

            best_accuracy = val_accuracy

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "validation_accuracy":
                        best_accuracy,

                    "sample_rate":
                        SAMPLE_RATE,

                    "duration":
                        DURATION,

                    "n_fft":
                        512,

                    "hop_length":
                        160,

                    "n_filters":
                        40,

                    "class_mapping":
                        {
                            "0": "REAL",
                            "1": "SPOOF"
                        }
                },
                MODEL_PATH
            )

            print(
                f"\n*** BEST MODEL SAVED ***"
            )

            print(
                f"Path: {MODEL_PATH}"
            )

    # ========================================================
    # FINAL RESULTS
    # ========================================================

    print("\n" + "=" * 70)
    print("FINAL VALIDATION RESULTS")
    print("=" * 70)

    print(
        f"Best validation accuracy: "
        f"{best_accuracy * 100:.2f}%"
    )

    print("\nConfusion Matrix:")

    print(
        confusion_matrix(
            val_targets,
            val_predictions
        )
    )

    print("\nClassification Report:")

    print(
        classification_report(
            val_targets,
            val_predictions,
            target_names=[
                "REAL",
                "SPOOF"
            ],
            digits=4
        )
    )

    print(
        f"\nModel saved to:"
        f"\n{MODEL_PATH}"
    )


if __name__ == "__main__":
    main()