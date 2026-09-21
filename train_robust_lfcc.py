import os
import random
import numpy as np
import librosa
import torch
import torch.nn as nn

from scipy.fftpack import dct
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


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
    "robust_lfcc_cnn.pth"
)

SAMPLE_RATE = 16000
NUM_SAMPLES = SAMPLE_RATE * 2

N_FFT = 512
WIN_LENGTH = int(0.020 * SAMPLE_RATE)
HOP_LENGTH = int(0.010 * SAMPLE_RATE)

N_FILTERS = 20
N_LFCC = 20

MAX_REAL = 2500
MAX_SPOOF = 2500

BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 0.0003

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
print("VIGILVOICE - ROBUST AUGMENTED LFCC CNN")
print("=" * 70)

print(f"Device: {DEVICE}")

if torch.cuda.is_available():
    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )


# ============================================================
# LINEAR FILTERBANK
# ============================================================

def create_filterbank():

    frequencies = np.linspace(
        0,
        SAMPLE_RATE / 2,
        N_FFT // 2 + 1
    )

    points = np.linspace(
        0,
        SAMPLE_RATE / 2,
        N_FILTERS + 2
    )

    bank = np.zeros(
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

            bank[
                i,
                left_idx
            ] = (
                frequencies[left_idx] - left
            ) / (
                center - left
            )

        if right > center:

            bank[
                i,
                right_idx
            ] = (
                right - frequencies[right_idx]
            ) / (
                right - center
            )

    return bank


FILTERBANK = create_filterbank()


# ============================================================
# AUGMENTATION
# ============================================================

def augment_audio(audio):

    choice = random.randint(
        0,
        5
    )

    # --------------------------------------------------------
    # 0 = clean
    # --------------------------------------------------------

    if choice == 0:

        return audio

    # --------------------------------------------------------
    # 1 = additive noise
    # --------------------------------------------------------

    elif choice == 1:

        noise_level = random.uniform(
            0.002,
            0.015
        )

        noise = np.random.randn(
            len(audio)
        ).astype(
            np.float32
        )

        audio = (
            audio +
            noise_level * noise
        )

    # --------------------------------------------------------
    # 2 = gain variation
    # --------------------------------------------------------

    elif choice == 2:

        gain = random.uniform(
            0.6,
            1.4
        )

        audio = audio * gain

    # --------------------------------------------------------
    # 3 = mild speed perturbation
    # --------------------------------------------------------

    elif choice == 3:

        rate = random.uniform(
            0.90,
            1.10
        )

        try:

            audio = librosa.effects.time_stretch(
                audio,
                rate=rate
            )

        except Exception:

            pass

    # --------------------------------------------------------
    # 4 = low-pass filtering
    # --------------------------------------------------------

    elif choice == 4:

        cutoff = random.randint(
            3500,
            7500
        )

        try:

            audio = librosa.effects.preemphasis(
                audio,
                coef=0.0
            )

            fft = np.fft.rfft(
                audio
            )

            freqs = np.fft.rfftfreq(
                len(audio),
                1 / SAMPLE_RATE
            )

            fft[
                freqs > cutoff
            ] *= 0.25

            audio = np.fft.irfft(
                fft,
                n=len(audio)
            )

        except Exception:

            pass

    # --------------------------------------------------------
    # 5 = clipping / distortion
    # --------------------------------------------------------

    elif choice == 5:

        threshold = random.uniform(
            0.65,
            0.90
        )

        audio = np.clip(
            audio,
            -threshold,
            threshold
        )

    # --------------------------------------------------------
    # Fix length after augmentation
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

    if len(audio) > NUM_SAMPLES:

        start = random.randint(
            0,
            len(audio) - NUM_SAMPLES
        )

        audio = audio[
            start:start + NUM_SAMPLES
        ]

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
# LFCC
# ============================================================

def extract_lfcc(audio):

    stft = librosa.stft(
        audio,
        n_fft=N_FFT,
        win_length=WIN_LENGTH,
        hop_length=HOP_LENGTH,
        window="hamming",
        center=False
    )

    power = np.abs(
        stft
    ) ** 2

    energy = np.dot(
        FILTERBANK,
        power
    )

    energy = np.maximum(
        energy,
        1e-10
    )

    log_energy = np.log(
        energy
    )

    lfcc = dct(
        log_energy,
        type=2,
        axis=0,
        norm="ortho"
    )

    lfcc = lfcc[
        :N_LFCC,
        :
    ]

    delta = librosa.feature.delta(
        lfcc,
        width=5,
        order=1
    )

    delta_delta = librosa.feature.delta(
        lfcc,
        width=5,
        order=2
    )

    features = np.concatenate(
        [
            lfcc,
            delta,
            delta_delta
        ],
        axis=0
    )

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

    audio, _ = librosa.load(
        path,
        sr=SAMPLE_RATE,
        mono=True
    )

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

    if len(audio) > NUM_SAMPLES:

        if training:

            start = random.randint(
                0,
                len(audio) - NUM_SAMPLES
            )

        else:

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

    audio = audio.astype(
        np.float32
    )

    peak = np.max(
        np.abs(audio)
    )

    if peak > 0:

        audio = audio / peak

    if training:

        audio = augment_audio(
            audio
        )

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

            path = os.path.join(
                AUDIO_DIR,
                file_id + ".flac"
            )

            if not os.path.exists(path):
                continue

            if label == "bonafide":

                real_files.append(path)

            elif label == "spoof":

                spoof_files.append(path)

    random.shuffle(real_files)
    random.shuffle(spoof_files)

    real_files = real_files[:MAX_REAL]
    spoof_files = spoof_files[:MAX_SPOOF]

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

        return len(self.files)

    def __getitem__(self, index):

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

            x = torch.tensor(
                features,
                dtype=torch.float32
            )

            x = x.unsqueeze(0)

            y = torch.tensor(
                label,
                dtype=torch.long
            )

            return x, y

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

            x = torch.tensor(
                features,
                dtype=torch.float32
            ).unsqueeze(0)

            y = torch.tensor(
                label,
                dtype=torch.long
            )

            return x, y


# ============================================================
# CNN
# ============================================================

class RobustLFCCCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(
                1,
                32,
                3,
                padding=1
            ),

            nn.BatchNorm2d(32),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                32,
                64,
                3,
                padding=1
            ),

            nn.BatchNorm2d(64),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                64,
                128,
                3,
                padding=1
            ),

            nn.BatchNorm2d(128),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                128,
                256,
                3,
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

            nn.Dropout(0.40),

            nn.Linear(
                128,
                2
            )
        )

    def forward(self, x):

        x = self.features(x)

        return self.classifier(x)


# ============================================================
# MAIN
# ============================================================

def main():

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    files, labels = read_protocol()

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

    model = RobustLFCCCNN().to(
        DEVICE
    )

    criterion = nn.CrossEntropyLoss(
        label_smoothing=0.05
    )

    optimizer = torch.optim.AdamW(
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
    # TRAINING
    # ========================================================

    for epoch in range(EPOCHS):

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
        ) in enumerate(train_loader):

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

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                5.0
            )

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

        print(
            f"\nTrain Loss: "
            f"{running_loss / len(train_loader):.4f}"
        )

        print(
            f"Train Acc : "
            f"{train_accuracy * 100:.2f}%"
        )

        print(
            f"Val Loss  : "
            f"{val_loss / len(val_loader):.4f}"
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
                "\n*** BEST ROBUST LFCC MODEL SAVED ***"
            )

            print(
                f"Path: {MODEL_PATH}"
            )

    # ========================================================
    # FINAL REPORT
    # ========================================================

    print("\n" + "=" * 70)
    print("ROBUST LFCC TRAINING COMPLETE")
    print("=" * 70)

    print(
        f"Best validation accuracy: "
        f"{best_accuracy * 100:.2f}%"
    )

    print("\nFinal epoch confusion matrix:")

    print(
        confusion_matrix(
            val_targets,
            val_predictions
        )
    )

    print("\nFinal epoch classification report:")

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
        f"\nSaved model:"
        f"\n{MODEL_PATH}"
    )


if __name__ == "__main__":
    main()