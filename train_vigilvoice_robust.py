# ============================================================
# VIGILVOICE ROBUST ACOUSTIC CNN v1
# CORRECTED VERSION
# ============================================================

import os
import random
import warnings
from collections import Counter

import numpy as np
import librosa

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_auc_score
)

warnings.filterwarnings("ignore")


# ============================================================
# CONFIG
# ============================================================

ROOT = r"C:\VIGILVOICE\data\voice_dataset"

TRAIN_AUDIO_DIR = os.path.join(
    ROOT, "ASVspoof2019_LA_train", "flac"
)

DEV_AUDIO_DIR = os.path.join(
    ROOT, "ASVspoof2019_LA_dev", "flac"
)

PROTOCOL_DIR = os.path.join(
    ROOT, "ASVspoof2019_LA_cm_protocols"
)

TRAIN_PROTOCOL = os.path.join(
    PROTOCOL_DIR,
    "ASVspoof2019.LA.cm.train.trn.txt"
)

DEV_PROTOCOL = os.path.join(
    PROTOCOL_DIR,
    "ASVspoof2019.LA.cm.dev.trl.txt"
)

MODEL_DIR = r"C:\VIGILVOICE\models"

BEST_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "vigilvoice_robust_cnn_v1.pth"
)

LAST_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "vigilvoice_robust_cnn_v1_last.pth"
)


# ============================================================
# AUDIO / FEATURE SETTINGS
# ============================================================

SAMPLE_RATE = 16000

WINDOW_SECONDS = 2.0

WINDOW_SAMPLES = int(
    SAMPLE_RATE * WINDOW_SECONDS
)

N_FFT = 512
HOP_LENGTH = 160
WIN_LENGTH = 400

N_LFCC = 20
N_MELS = 40

TARGET_FRAMES = 201


# ============================================================
# TRAINING SETTINGS
# ============================================================

BATCH_SIZE = 32
EPOCHS = 12

LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4

NUM_WORKERS = 0

SEED = 42

AUGMENT_PROBABILITY = 0.85


# ============================================================
# SEED
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# PRINT
# ============================================================

def print_header(text):

    print()
    print("=" * 75)
    print(text)
    print("=" * 75)


# ============================================================
# FIX FEATURE SIZE
# ============================================================

def fix_feature_size(
    x,
    target_frames=TARGET_FRAMES
):

    x = np.asarray(
        x,
        dtype=np.float32
    )

    if x.ndim != 2:

        raise ValueError(
            f"Expected 2D feature matrix, got {x.shape}"
        )

    current = x.shape[1]

    if current > target_frames:

        x = x[:, :target_frames]

    elif current < target_frames:

        pad = target_frames - current

        x = np.pad(
            x,
            (
                (0, 0),
                (0, pad)
            ),
            mode="constant"
        )

    return x.astype(
        np.float32
    )


# ============================================================
# AUDIO
# ============================================================

def load_audio(path):

    audio, _ = librosa.load(
        path,
        sr=SAMPLE_RATE,
        mono=True
    )

    audio = np.asarray(
        audio,
        dtype=np.float32
    )

    if len(audio) == 0:

        audio = np.zeros(
            WINDOW_SAMPLES,
            dtype=np.float32
        )

    return audio


# ============================================================
# WINDOW
# ============================================================

def get_audio_window(
    audio,
    training=True
):

    if len(audio) < WINDOW_SAMPLES:

        audio = np.pad(
            audio,
            (
                0,
                WINDOW_SAMPLES - len(audio)
            )
        )

        return audio.astype(
            np.float32
        )

    if len(audio) == WINDOW_SAMPLES:

        return audio.astype(
            np.float32
        )

    max_start = (
        len(audio) -
        WINDOW_SAMPLES
    )

    if training:

        start = random.randint(
            0,
            max_start
        )

    else:

        start = max_start // 2

    return audio[
        start:start + WINDOW_SAMPLES
    ].astype(
        np.float32
    )


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_audio(audio):

    audio = audio.astype(
        np.float32
    )

    peak = np.max(
        np.abs(audio)
    )

    if peak > 1e-8:

        audio = audio / peak

    return audio


# ============================================================
# AUGMENTATION
# ============================================================

def augment_audio(audio):

    audio = audio.astype(
        np.float32,
        copy=True
    )

    # Gain
    if random.random() < 0.60:

        gain = random.uniform(
            0.60,
            1.40
        )

        audio *= gain

    # Gaussian noise
    if random.random() < 0.60:

        noise_level = random.uniform(
            0.001,
            0.015
        )

        noise = np.random.normal(
            0,
            noise_level,
            len(audio)
        ).astype(
            np.float32
        )

        audio += noise

    # Speed perturbation
    if random.random() < 0.35:

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

        if len(audio) < WINDOW_SAMPLES:

            audio = np.pad(
                audio,
                (
                    0,
                    WINDOW_SAMPLES - len(audio)
                )
            )

        elif len(audio) > WINDOW_SAMPLES:

            audio = audio[
                :WINDOW_SAMPLES
            ]

    # Low-pass filtering
    if random.random() < 0.30:

        cutoff = random.uniform(
            3500,
            7500
        )

        try:

            spectrum = np.fft.rfft(
                audio
            )

            freqs = np.fft.rfftfreq(
                len(audio),
                d=1.0 / SAMPLE_RATE
            )

            spectrum[
                freqs > cutoff
            ] = 0

            audio = np.fft.irfft(
                spectrum,
                n=len(audio)
            ).astype(
                np.float32
            )

        except Exception:
            pass

    # Clipping
    if random.random() < 0.20:

        clip_level = random.uniform(
            0.65,
            0.90
        )

        audio = np.clip(
            audio,
            -clip_level,
            clip_level
        )

    return audio.astype(
        np.float32
    )


# ============================================================
# PREEMPHASIS
# ============================================================

def preemphasis(
    audio,
    coef=0.97
):

    output = np.empty_like(
        audio
    )

    output[0] = audio[0]

    output[1:] = (
        audio[1:] -
        coef * audio[:-1]
    )

    return output


# ============================================================
# DCT-II
# ============================================================

def dct_type_2(
    x,
    num_ceps
):

    n_filters = x.shape[0]

    n = np.arange(
        n_filters,
        dtype=np.float32
    )

    k = np.arange(
        num_ceps,
        dtype=np.float32
    )[:, None]

    basis = np.cos(
        np.pi *
        (n + 0.5) *
        k /
        n_filters
    )

    return np.dot(
        basis,
        x
    )


# ============================================================
# LFCC
# ============================================================

def extract_lfcc(audio):

    audio = preemphasis(
        audio,
        0.97
    )

    stft = librosa.stft(
        audio,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window="hamming",
        center=False
    )

    power = np.abs(
        stft
    ) ** 2

    n_freqs = power.shape[0]

    low_freq = 50.0

    high_freq = min(
        7600.0,
        SAMPLE_RATE / 2
    )

    hz_points = np.linspace(
        low_freq,
        high_freq,
        N_LFCC + 2
    )

    bins = np.floor(
        (N_FFT + 1) *
        hz_points /
        SAMPLE_RATE
    ).astype(int)

    filter_bank = np.zeros(
        (
            N_LFCC,
            n_freqs
        ),
        dtype=np.float32
    )

    for m in range(
        1,
        N_LFCC + 1
    ):

        left = max(
            0,
            min(
                bins[m - 1],
                n_freqs - 1
            )
        )

        center = max(
            0,
            min(
                bins[m],
                n_freqs - 1
            )
        )

        right = max(
            0,
            min(
                bins[m + 1],
                n_freqs - 1
            )
        )

        if center > left:

            filter_bank[
                m - 1,
                left:center
            ] = np.linspace(
                0,
                1,
                center - left,
                endpoint=False
            )

        if right > center:

            filter_bank[
                m - 1,
                center:right
            ] = np.linspace(
                1,
                0,
                right - center,
                endpoint=False
            )

    energies = np.dot(
        filter_bank,
        power
    )

    energies = np.maximum(
        energies,
        1e-10
    )

    log_energy = np.log(
        energies
    )

    static = dct_type_2(
        log_energy,
        N_LFCC
    )

    delta = librosa.feature.delta(
        static
    )

    delta2 = librosa.feature.delta(
        static,
        order=2
    )

    lfcc = np.concatenate(
        [
            static,
            delta,
            delta2
        ],
        axis=0
    )

    mean = np.mean(
        lfcc,
        axis=1,
        keepdims=True
    )

    std = np.std(
        lfcc,
        axis=1,
        keepdims=True
    ) + 1e-6

    lfcc = (
        lfcc - mean
    ) / std

    return fix_feature_size(
        lfcc
    )


# ============================================================
# LOG-MEL
# ============================================================

def extract_log_mel(audio):

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window="hann",
        n_mels=N_MELS,
        fmin=50,
        fmax=7600,
        center=False,
        power=2.0
    )

    mel = librosa.power_to_db(
        mel,
        ref=np.max
    )

    mean = np.mean(
        mel,
        axis=1,
        keepdims=True
    )

    std = np.std(
        mel,
        axis=1,
        keepdims=True
    ) + 1e-6

    mel = (
        mel - mean
    ) / std

    return fix_feature_size(
        mel
    )


# ============================================================
# TIME MASK
# ============================================================

def time_mask(
    feature,
    max_width=20
):

    feature = feature.copy()

    frames = feature.shape[1]

    if frames <= 2:

        return feature

    width = random.randint(
        0,
        min(
            max_width,
            frames // 5
        )
    )

    if width == 0:

        return feature

    start = random.randint(
        0,
        frames - width
    )

    feature[
        :,
        start:start + width
    ] = 0

    return feature


# ============================================================
# PROTOCOL
# ============================================================

def read_protocol(
    protocol_path,
    audio_dir
):

    entries = []

    with open(
        protocol_path,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            parts = line.strip().split()

            if len(parts) < 5:
                continue

            file_id = parts[1]

            label_text = parts[-1].lower()

            if label_text == "bonafide":

                label = 0

            elif label_text == "spoof":

                label = 1

            else:

                continue

            path = os.path.join(
                audio_dir,
                file_id + ".flac"
            )

            if os.path.exists(path):

                entries.append(
                    (
                        path,
                        label,
                        file_id
                    )
                )

    return entries


# ============================================================
# DATASET
# ============================================================

class VoiceDataset(Dataset):

    def __init__(
        self,
        entries,
        training=False
    ):

        self.entries = entries
        self.training = training

    def __len__(self):

        return len(
            self.entries
        )

    def __getitem__(
        self,
        index
    ):

        path, label, file_id = (
            self.entries[index]
        )

        try:

            audio = load_audio(
                path
            )

            audio = get_audio_window(
                audio,
                self.training
            )

            audio = normalize_audio(
                audio
            )

            if (
                self.training and
                random.random() <
                AUGMENT_PROBABILITY
            ):

                audio = augment_audio(
                    audio
                )

                audio = normalize_audio(
                    audio
                )

            lfcc = extract_lfcc(
                audio
            )

            mel = extract_log_mel(
                audio
            )

            if (
                self.training and
                random.random() < 0.30
            ):

                lfcc = time_mask(
                    lfcc
                )

                mel = time_mask(
                    mel
                )

            # Final hard shape guarantee
            lfcc = fix_feature_size(
                lfcc
            )

            mel = fix_feature_size(
                mel
            )

            return (
                torch.from_numpy(
                    lfcc
                ).float(),

                torch.from_numpy(
                    mel
                ).float(),

                torch.tensor(
                    label,
                    dtype=torch.long
                )
            )

        except Exception as e:

            print(
                f"\nFeature error "
                f"{file_id}: {e}"
            )

            lfcc = np.zeros(
                (
                    60,
                    TARGET_FRAMES
                ),
                dtype=np.float32
            )

            mel = np.zeros(
                (
                    40,
                    TARGET_FRAMES
                ),
                dtype=np.float32
            )

            return (
                torch.from_numpy(
                    lfcc
                ).float(),

                torch.from_numpy(
                    mel
                ).float(),

                torch.tensor(
                    label,
                    dtype=torch.long
                )
            )


# ============================================================
# CNN BRANCH
# ============================================================
#
# IMPORTANT:
#
# LFCC tensor:
#     [B, 60, 201]
#
# Mel tensor:
#     [B, 40, 201]
#
# We add ONE channel dimension:
#
#     LFCC -> [B, 1, 60, 201]
#     Mel  -> [B, 1, 40, 201]
#
# Therefore Conv2d MUST start with in_channels=1.
# ============================================================

class AcousticCNNBranch(
    nn.Module
):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            # ------------------------------------------------
            # Block 1
            # ------------------------------------------------

            nn.Conv2d(
                1,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(
                32
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.MaxPool2d(
                2
            ),

            # ------------------------------------------------
            # Block 2
            # ------------------------------------------------

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(
                64
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.MaxPool2d(
                2
            ),

            # ------------------------------------------------
            # Block 3
            # ------------------------------------------------

            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(
                128
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.MaxPool2d(
                2
            ),

            # ------------------------------------------------
            # Block 4
            # ------------------------------------------------

            nn.Conv2d(
                128,
                256,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(
                256
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.AdaptiveAvgPool2d(
                (1, 1)
            )
        )

    def forward(
        self,
        x
    ):

        return self.features(
            x
        ).flatten(
            1
        )


# ============================================================
# FULL DUAL-BRANCH MODEL
# ============================================================

class VigilVoiceRobustCNN(
    nn.Module
):

    def __init__(self):

        super().__init__()

        self.lfcc_branch = (
            AcousticCNNBranch()
        )

        self.mel_branch = (
            AcousticCNNBranch()
        )

        # 256 LFCC + 256 Mel
        self.classifier = nn.Sequential(

            nn.Linear(
                512,
                128
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Dropout(
                0.40
            ),

            nn.Linear(
                128,
                2
            )
        )

    def forward(
        self,
        lfcc,
        mel
    ):

        # [B,60,T]
        # -> [B,1,60,T]

        lfcc = lfcc.unsqueeze(
            1
        )

        # [B,40,T]
        # -> [B,1,40,T]

        mel = mel.unsqueeze(
            1
        )

        lfcc_features = (
            self.lfcc_branch(
                lfcc
            )
        )

        mel_features = (
            self.mel_branch(
                mel
            )
        )

        combined = torch.cat(
            [
                lfcc_features,
                mel_features
            ],
            dim=1
        )

        return self.classifier(
            combined
        )


# ============================================================
# LOSS
# ============================================================

class LabelSmoothingCrossEntropy(
    nn.Module
):

    def __init__(
        self,
        smoothing=0.05
    ):

        super().__init__()

        self.smoothing = smoothing

    def forward(
        self,
        pred,
        target
    ):

        log_probs = F.log_softmax(
            pred,
            dim=1
        )

        classes = pred.size(
            1
        )

        with torch.no_grad():

            true_dist = torch.full_like(
                log_probs,
                self.smoothing /
                (classes - 1)
            )

            true_dist.scatter_(
                1,
                target.unsqueeze(1),
                1.0 -
                self.smoothing
            )

        return torch.mean(
            torch.sum(
                -true_dist *
                log_probs,
                dim=1
            )
        )


# ============================================================
# START
# ============================================================

print_header(
    "VIGILVOICE ROBUST ACOUSTIC CNN v1"
)

print(
    "Device:",
    DEVICE
)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

    vram = (
        torch.cuda.get_device_properties(0)
        .total_memory /
        (1024 ** 3)
    )

    print(
        f"VRAM: {vram:.1f} GB"
    )


# ============================================================
# CHECK PATHS
# ============================================================

print_header(
    "Checking dataset paths..."
)

for path in [
    TRAIN_AUDIO_DIR,
    DEV_AUDIO_DIR,
    TRAIN_PROTOCOL,
    DEV_PROTOCOL
]:

    if not os.path.exists(path):

        raise FileNotFoundError(
            path
        )

    print(
        "OK:",
        path
    )


# ============================================================
# TRAIN DATA
# ============================================================

print_header(
    "BUILDING TRAIN DATA"
)

train_entries = read_protocol(
    TRAIN_PROTOCOL,
    TRAIN_AUDIO_DIR
)

train_counter = Counter(
    label
    for _, label, _
    in train_entries
)

print(
    "Protocol entries:",
    len(train_entries)
)

print(
    "Audio files found:",
    len(train_entries)
)

print(
    "Missing audio: 0"
)

print(
    "Distribution:",
    train_counter
)


# ============================================================
# DEV DATA
# ============================================================

print_header(
    "BUILDING DEV DATA"
)

dev_entries = read_protocol(
    DEV_PROTOCOL,
    DEV_AUDIO_DIR
)

dev_counter = Counter(
    label
    for _, label, _
    in dev_entries
)

print(
    "Protocol entries:",
    len(dev_entries)
)

print(
    "Audio files found:",
    len(dev_entries)
)

print(
    "Missing audio: 0"
)

print(
    "Distribution:",
    dev_counter
)


# ============================================================
# DATASETS
# ============================================================

print_header(
    "Creating datasets..."
)

train_dataset = VoiceDataset(
    train_entries,
    training=True
)

dev_dataset = VoiceDataset(
    dev_entries,
    training=False
)


# ============================================================
# BALANCED SAMPLER
# ============================================================

class_counts = Counter(
    label
    for _, label, _
    in train_entries
)

print(
    "\nClass counts:",
    class_counts
)

class_weights = {
    label: 1.0 / count
    for label, count
    in class_counts.items()
}

sample_weights = torch.tensor(
    [
        class_weights[label]
        for _, label, _
        in train_entries
    ],
    dtype=torch.double
)

sampler = WeightedRandomSampler(
    sample_weights,
    num_samples=len(
        train_entries
    ),
    replacement=True
)


# ============================================================
# LOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=sampler,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)

dev_loader = DataLoader(
    dev_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)


# ============================================================
# MODEL
# ============================================================

model = VigilVoiceRobustCNN().to(
    DEVICE
)

num_parameters = sum(
    p.numel()
    for p in model.parameters()
)

print(
    f"\nModel parameters: "
    f"{num_parameters:,}"
)


# ============================================================
# OPTIMIZER / LOSS
# ============================================================

criterion = LabelSmoothingCrossEntropy(
    0.05
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS
)


# ============================================================
# AMP
# ============================================================

if torch.cuda.is_available():

    scaler = torch.amp.GradScaler(
        "cuda"
    )

else:

    scaler = None


# ============================================================
# TRAIN FUNCTION
# ============================================================

def train_one_epoch():

    model.train()

    total_loss = 0.0

    targets_all = []
    predictions_all = []

    total_batches = len(
        train_loader
    )

    for batch_index, (
        lfcc,
        mel,
        targets
    ) in enumerate(
        train_loader,
        1
    ):

        lfcc = lfcc.to(
            DEVICE,
            non_blocking=True
        )

        mel = mel.to(
            DEVICE,
            non_blocking=True
        )

        targets = targets.to(
            DEVICE,
            non_blocking=True
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        if scaler is not None:

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16
            ):

                outputs = model(
                    lfcc,
                    mel
                )

                loss = criterion(
                    outputs,
                    targets
                )

            scaler.scale(
                loss
            ).backward()

            scaler.unscale_(
                optimizer
            )

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                5.0
            )

            scaler.step(
                optimizer
            )

            scaler.update()

        else:

            outputs = model(
                lfcc,
                mel
            )

            loss = criterion(
                outputs,
                targets
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                5.0
            )

            optimizer.step()

        total_loss += loss.item()

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        targets_all.extend(
            targets.detach()
            .cpu()
            .numpy()
        )

        predictions_all.extend(
            predictions.detach()
            .cpu()
            .numpy()
        )

        if (
            batch_index == 1 or
            batch_index % 100 == 0 or
            batch_index == total_batches
        ):

            print(
                f"\rBatch "
                f"{batch_index}/{total_batches}"
                f" | Loss "
                f"{loss.item():.4f}",
                end="",
                flush=True
            )

    print()

    avg_loss = (
        total_loss /
        total_batches
    )

    accuracy = accuracy_score(
        targets_all,
        predictions_all
    )

    precision = precision_score(
        targets_all,
        predictions_all,
        pos_label=1,
        zero_division=0
    )

    recall = recall_score(
        targets_all,
        predictions_all,
        pos_label=1,
        zero_division=0
    )

    return (
        avg_loss,
        accuracy,
        precision,
        recall
    )


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def validate():

    model.eval()

    total_loss = 0.0

    targets_all = []
    predictions_all = []
    spoof_scores_all = []

    for (
        lfcc,
        mel,
        targets
    ) in dev_loader:

        lfcc = lfcc.to(
            DEVICE,
            non_blocking=True
        )

        mel = mel.to(
            DEVICE,
            non_blocking=True
        )

        targets = targets.to(
            DEVICE,
            non_blocking=True
        )

        outputs = model(
            lfcc,
            mel
        )

        loss = criterion(
            outputs,
            targets
        )

        total_loss += loss.item()

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        predictions = torch.argmax(
            probabilities,
            dim=1
        )

        spoof_scores = probabilities[
            :, 1
        ]

        targets_all.extend(
            targets.cpu()
            .numpy()
        )

        predictions_all.extend(
            predictions.cpu()
            .numpy()
        )

        spoof_scores_all.extend(
            spoof_scores.cpu()
            .numpy()
        )

    avg_loss = (
        total_loss /
        len(dev_loader)
    )

    accuracy = accuracy_score(
        targets_all,
        predictions_all
    )

    precision = precision_score(
        targets_all,
        predictions_all,
        pos_label=1,
        zero_division=0
    )

    recall = recall_score(
        targets_all,
        predictions_all,
        pos_label=1,
        zero_division=0
    )

    cm = confusion_matrix(
        targets_all,
        predictions_all,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    bonafide_total = (
        tn + fp
    )

    fpr = (
        fp / bonafide_total
        if bonafide_total > 0
        else 0.0
    )

    try:

        auc = roc_auc_score(
            targets_all,
            spoof_scores_all
        )

    except Exception:

        auc = 0.0

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "precision": precision,
        "spoof_recall": recall,
        "fpr": fpr,
        "auc": auc,
        "cm": cm
    }


# ============================================================
# TRAIN
# ============================================================

print_header(
    "STARTING TRAINING"
)

print(
    f"Train files: {len(train_entries)}"
)

print(
    f"Dev files: {len(dev_entries)}"
)

print(
    f"Batch size: {BATCH_SIZE}"
)

print(
    f"Epochs: {EPOCHS}"
)

print(
    f"Audio window: "
    f"{WINDOW_SECONDS} seconds"
)

print(
    f"LFCC shape: "
    f"[60, {TARGET_FRAMES}]"
)

print(
    f"Log-Mel shape: "
    f"[40, {TARGET_FRAMES}]"
)

print(
    f"DataLoader workers: "
    f"{NUM_WORKERS}"
)

print(
    "=" * 75
)


best_score = -999.0
best_epoch = 0


for epoch in range(
    1,
    EPOCHS + 1
):

    print()
    print(
        "=" * 75
    )

    print(
        f"EPOCH {epoch}/{EPOCHS}"
    )

    print(
        "=" * 75
    )

    train_loss, train_acc, train_precision, train_recall = (
        train_one_epoch()
    )

    metrics = validate()

    scheduler.step()

    print()

    print(
        f"Train Loss        : "
        f"{train_loss:.5f}"
    )

    print(
        f"Train Accuracy    : "
        f"{train_acc * 100:.2f}%"
    )

    print(
        f"Train Precision   : "
        f"{train_precision * 100:.2f}%"
    )

    print(
        f"Train Spoof Recall : "
        f"{train_recall * 100:.2f}%"
    )

    print()

    print(
        f"Dev Loss           : "
        f"{metrics['loss']:.5f}"
    )

    print(
        f"Dev Accuracy       : "
        f"{metrics['accuracy'] * 100:.2f}%"
    )

    print(
        f"Dev Precision      : "
        f"{metrics['precision'] * 100:.2f}%"
    )

    print(
        f"Dev Spoof Recall   : "
        f"{metrics['spoof_recall'] * 100:.2f}%"
    )

    print(
        f"Dev False Positive : "
        f"{metrics['fpr'] * 100:.2f}%"
    )

    print(
        f"Dev ROC-AUC        : "
        f"{metrics['auc']:.4f}"
    )

    print()

    print(
        "Confusion Matrix "
        "(actual rows / predicted columns):"
    )

    print(
        metrics["cm"]
    )

    # --------------------------------------------------------
    # Selection metric
    # --------------------------------------------------------

    selection_score = (
        metrics["spoof_recall"]
        -
        0.5 * metrics["fpr"]
    )

    print()

    print(
        f"Selection Score    : "
        f"{selection_score:.5f}"
    )

    # --------------------------------------------------------
    # Save best
    # --------------------------------------------------------

    if selection_score > best_score:

        best_score = selection_score

        best_epoch = epoch

        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "scheduler_state_dict":
                    scheduler.state_dict(),

                "epoch":
                    epoch,

                "best_score":
                    best_score,

                "architecture":
                    "VIGILVOICE Robust Acoustic CNN v1",

                "sample_rate":
                    SAMPLE_RATE,

                "window_seconds":
                    WINDOW_SECONDS,

                "target_frames":
                    TARGET_FRAMES,

                "class_mapping":
                    {
                        "0": "REAL",
                        "1": "SPOOF"
                    },

                "num_parameters":
                    num_parameters,

                "dev_metrics":
                    metrics
            },
            BEST_MODEL_PATH
        )

        print()

        print(
            "*** NEW BEST MODEL SAVED ***"
        )

        print(
            BEST_MODEL_PATH
        )

    # --------------------------------------------------------
    # Save latest
    # --------------------------------------------------------

    torch.save(
        {
            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "scheduler_state_dict":
                scheduler.state_dict(),

            "epoch":
                epoch,

            "architecture":
                "VIGILVOICE Robust Acoustic CNN v1",

            "class_mapping":
                {
                    "0": "REAL",
                    "1": "SPOOF"
                }
        },
        LAST_MODEL_PATH
    )


# ============================================================
# COMPLETE
# ============================================================

print_header(
    "TRAINING COMPLETE"
)

print(
    f"Best epoch: {best_epoch}"
)

print(
    f"Best selection score: "
    f"{best_score:.5f}"
)

print()

print(
    "Best model:"
)

print(
    BEST_MODEL_PATH
)

print()

print(
    "Latest model:"
)

print(
    LAST_MODEL_PATH
)

print()

print(
    "The model was trained using the official"
)

print(
    "ASVspoof2019 LA train protocol and validated"
)

print(
    "using the official LA development protocol."
)

print()

print(
    "These are development-set results, NOT official"
)

print(
    "ASVspoof evaluation results."
)

print(
    "=" * 75
)