import os
import random
import numpy as np
import librosa
import torch
import torch.nn as nn

from scipy.fftpack import dct

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)


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
    "lfcc_cnn.pth"
)

SAMPLE_RATE = 16000

# 2 seconds
NUM_SAMPLES = SAMPLE_RATE * 2

# ASVspoof LFCC settings from reference
N_FFT = 512
WIN_LENGTH = int(0.020 * SAMPLE_RATE)   # 20 ms
HOP_LENGTH = int(0.010 * SAMPLE_RATE)   # 10 ms

N_FILTERS = 20
N_LFCC = 20

MAX_REAL = 2500
MAX_SPOOF = 2500

BATCH_SIZE = 32
EPOCHS = 12
LEARNING_RATE = 0.0005

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("VIGILVOICE - PROPER LFCC + CNN")
print("=" * 70)

print(f"Device: {DEVICE}")

if torch.cuda.is_available():
    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )


# ============================================================
# LINEAR FILTERBANK
# ============================================================

def create_linear_filterbank():

    frequencies = np.linspace(
        0,
        SAMPLE_RATE / 2,
        N_FFT // 2 + 1
    )

    # 20 linearly spaced filters
    points = np.linspace(
        0,
        SAMPLE_RATE / 2,
        N_FILTERS + 2
    )

    filterbank = np.zeros(
        (
            N_FILTERS,
            N_FFT // 2 + 1
        ),
        dtype=np.float32
    )

    for i in range(N_FILTERS):

        left = points[i]
        center = points[i + 1]
        right = points[i + 2]

        left_idx = np.where(
            (frequencies >= left) &
            (frequencies <= center)
        )[0]

        right_idx = np.where(
            (frequencies >= center) &
            (frequencies <= right)
        )[0]

        if center > left:

            filterbank[
                i,
                left_idx
            ] = (
                frequencies[left_idx] - left
            ) / (
                center - left
            )

        if right > center:

            filterbank[
                i,
                right_idx
            ] = (
                right - frequencies[right_idx]
            ) / (
                right - center
            )

    return filterbank


FILTERBANK = create_linear_filterbank()


# ============================================================
# DELTA FEATURES
# ============================================================

def compute_delta(features):

    """
    features:
        [coefficients, frames]

    returns:
        same shape
    """

    delta = librosa.feature.delta(
        features,
        width=5,
        order=1
    )

    return delta


# ============================================================
# LFCC EXTRACTION
# ============================================================

def extract_lfcc(audio):

    # --------------------------------------------------------
    # STFT
    # --------------------------------------------------------

    stft = librosa.stft(
        audio,
        n_fft=N_FFT,
        win_length=WIN_LENGTH,
        hop_length=HOP_LENGTH,
        window="hamming",
        center=False
    )

    # Power spectrum
    power = np.abs(stft) ** 2

    # --------------------------------------------------------
    # Linear filterbank
    # --------------------------------------------------------

    filter_energy = np.dot(
        FILTERBANK,
        power
    )

    # Avoid log(0)
    filter_energy = np.maximum(
        filter_energy,
        1e-10
    )

    log_energy = np.log(
        filter_energy
    )

    # --------------------------------------------------------
    # DCT → LFCC
    # --------------------------------------------------------

    lfcc = dct(
        log_energy,
        type=2,
        axis=0,
        norm="ortho"
    )

    # Keep 20 static LFCC coefficients
    lfcc = lfcc[
        :N_LFCC,
        :
    ]

    # --------------------------------------------------------
    # Delta
    # --------------------------------------------------------

    delta = compute_delta(
        lfcc
    )

    # --------------------------------------------------------
    # Delta-delta
    # --------------------------------------------------------

    delta_delta = librosa.feature.delta(
        lfcc,
        width=5,
        order=2
    )

    # --------------------------------------------------------
    # Stack
    # --------------------------------------------------------

    features = np.concatenate(
        [
            lfcc,
            delta,
            delta_delta
        ],
        axis=0
    )

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    mean = features.mean(
        axis=1,
        keepdims=True
    )

    std = features.std(
        axis=1,
        keepdims=True
    ) + 1e-6

    features = (
        features - mean
    ) / std

    return features.astype(
        np.float32
    )


# ============================================================
# AUDIO PREPARATION
# ============================================================

def prepare_audio(
    path,
    training=False
):

    audio, sr = librosa.load(
        path,
        sr=SAMPLE_RATE,
        mono=True
    )

    # --------------------------------------------------------
    # Too short → repeat
    # --------------------------------------------------------

    if len(audio) < NUM_SAMPLES:

        repeats = int(
            np.ceil(
                NUM_SAMPLES / len(audio)
            )
        )

        audio = np.tile(
            audio,
            repeats
        )

    # --------------------------------------------------------
    # Longer → crop
    # --------------------------------------------------------

    if len(audio) > NUM_SAMPLES:

        if training:

            # Random crop ONLY for training
            start = random.randint(
                0,
                len(audio) - NUM_SAMPLES
            )

        else:

            # Deterministic center crop
            start = (
                len(audio) - NUM_SAMPLES
            ) // 2

        audio = audio[
            start:start + NUM_SAMPLES
        ]

    else:

        audio = audio[
            :NUM_SAMPLES
        ]

    # --------------------------------------------------------
    # Normalize waveform
    # --------------------------------------------------------

    audio = audio.astype(
        np.float32
    )

    peak = np.max(
        np.abs(audio)
    )

    if peak > 0:

        audio = audio / peak

    return audio


# ============================================================
# PROTOCOL
# ============================================================

def read_protocol():

    real_files = []
    spoof_files = []

    print("\nReading ASVspoof protocol...")

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

            if not os.path.exists(
                audio_path
            ):
                continue

            if label == "bonafide":

                real_files.append(
                    audio_path
                )

            elif label == "spoof":

                spoof_files.append(
                    audio_path
                )

    random.shuffle(
        real_files
    )

    random.shuffle(
        spoof_files
    )

    real_files = real_files[
        :MAX_REAL
    ]

    spoof_files = spoof_files[
        :MAX_SPOOF
    ]

    print(
        f"Real samples : {len(real_files)}"
    )

    print(
        f"Spoof samples: {len(spoof_files)}"
    )

    files = (
        real_files +
        spoof_files
    )

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
        labels,
        training=False
    ):

        self.files = files
        self.labels = labels
        self.training = training

    def __len__(self):

        return len(
            self.files
        )

    def __getitem__(
        self,
        index
    ):

        path = self.files[index]

        label = self.labels[index]

        try:

            audio = prepare_audio(
                path,
                training=self.training
            )

            features = extract_lfcc(
                audio
            )

            # [60, time]
            features = torch.tensor(
                features,
                dtype=torch.float32
            )

            # CNN:
            # [channel, features, time]

            features = features.unsqueeze(
                0
            )

            label = torch.tensor(
                label,
                dtype=torch.long
            )

            return (
                features,
                label
            )

        except Exception as e:

            print(
                f"Error: {path}"
            )

            print(e)

            dummy = np.zeros(
                NUM_SAMPLES,
                dtype=np.float32
            )

            features = extract_lfcc(
                dummy
            )

            features = torch.tensor(
                features,
                dtype=torch.float32
            ).unsqueeze(0)

            label = torch.tensor(
                label,
                dtype=torch.long
            )

            return (
                features,
                label
            )


# ============================================================
# CNN
# ============================================================

class LFCCCNN(nn.Module):

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

            nn.MaxPool2d(
                kernel_size=(2, 2)
            ),

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(64),

            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=(2, 2)
            ),

            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(128),

            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=(2, 2)
            ),

            nn.Conv2d(
                128,
                256,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(256),

            nn.ReLU(),

            nn.AdaptiveAvgPool2d(
                (1, 1)
            )
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                256,
                128
            ),

            nn.ReLU(),

            nn.Dropout(
                0.35
            ),

            nn.Linear(
                128,
                2
            )
        )

    def forward(
        self,
        x
    ):

        x = self.features(
            x
        )

        x = self.classifier(
            x
        )

        return x


# ============================================================
# TRAIN
# ============================================================

def main():

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    files, labels = read_protocol()

    # --------------------------------------------------------
    # SPLIT ON FILES
    # --------------------------------------------------------

    (
        train_files,
        val_files,
        train_labels,
        val_labels
    ) = train_test_split(
        files,
        labels,
        test_size=0.20,
        random_state=SEED,
        stratify=labels
    )

    print("\nDataset split:")

    print(
        f"Training   : {len(train_files)}"
    )

    print(
        f"Validation : {len(val_files)}"
    )

    # --------------------------------------------------------
    # DATASETS
    # --------------------------------------------------------

    train_dataset = VoiceDataset(
        train_files,
        train_labels,
        training=True
    )

    val_dataset = VoiceDataset(
        val_files,
        val_labels,
        training=False
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

    model = LFCCCNN().to(
        DEVICE
    )

    print("\nModel:")

    print(model)

    # --------------------------------------------------------
    # LOSS
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2
    )

    best_accuracy = 0.0

    # ========================================================
    # EPOCHS
    # ========================================================

    for epoch in range(
        EPOCHS
    ):

        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        model.train()

        running_loss = 0.0

        train_predictions = []
        train_targets = []

        print(
            f"\nEpoch {epoch + 1}/{EPOCHS}"
        )

        for batch_idx, (
            x,
            y
        ) in enumerate(
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

            outputs = model(
                x
            )

            loss = criterion(
                outputs,
                y
            )

            loss.backward()

            optimizer.step()

            running_loss += (
                loss.item()
            )

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

            if (
                batch_idx + 1
            ) % 20 == 0:

                print(
                    f"  Batch "
                    f"{batch_idx + 1}/"
                    f"{len(train_loader)} "
                    f"Loss: "
                    f"{loss.item():.4f}"
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

                x = x.to(
                    DEVICE
                )

                y = y.to(
                    DEVICE
                )

                outputs = model(
                    x
                )

                loss = criterion(
                    outputs,
                    y
                )

                val_loss += (
                    loss.item()
                )

                predictions = torch.argmax(
                    outputs,
                    dim=1
                )

                val_predictions.extend(
                    predictions.cpu()
                    .numpy()
                )

                val_targets.extend(
                    y.cpu()
                    .numpy()
                )

        val_accuracy = accuracy_score(
            val_targets,
            val_predictions
        )

        scheduler.step(
            val_accuracy
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
            f"\nTrain Loss: "
            f"{avg_train_loss:.4f}"
        )

        print(
            f"Train Acc : "
            f"{train_accuracy * 100:.2f}%"
        )

        print(
            f"Val Loss  : "
            f"{avg_val_loss:.4f}"
        )

        print(
            f"Val Acc   : "
            f"{val_accuracy * 100:.2f}%"
        )

        print(
            f"LR        : "
            f"{optimizer.param_groups[0]['lr']:.6f}"
        )

        # ----------------------------------------------------
        # SAVE BEST
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

                    "num_samples":
                        NUM_SAMPLES,

                    "n_fft":
                        N_FFT,

                    "win_length":
                        WIN_LENGTH,

                    "hop_length":
                        HOP_LENGTH,

                    "n_filters":
                        N_FILTERS,

                    "n_lfcc":
                        N_LFCC,

                    "class_mapping":
                        {
                            "0": "REAL",
                            "1": "SPOOF"
                        }
                },
                MODEL_PATH
            )

            print(
                "\n*** BEST LFCC MODEL SAVED ***"
            )

            print(
                f"Path: {MODEL_PATH}"
            )

    # ========================================================
    # FINAL
    # ========================================================

    print("\n" + "=" * 70)

    print(
        "FINAL LFCC VALIDATION RESULTS"
    )

    print("=" * 70)

    print(
        f"Best validation accuracy: "
        f"{best_accuracy * 100:.2f}%"
    )

    print(
        "\nFinal epoch confusion matrix:"
    )

    print(
        confusion_matrix(
            val_targets,
            val_predictions
        )
    )

    print(
        "\nClassification report:"
    )

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
        f"\nModel saved:"
        f"\n{MODEL_PATH}"
    )


if __name__ == "__main__":

    main()