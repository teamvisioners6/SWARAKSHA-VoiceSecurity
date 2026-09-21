import csv
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

from features import (
    extract_feature_tensor,
    extract_training_feature_tensor,
)
from model import SwarakshaCNN


# ============================================================
# SWARAKSHA - Custom Voice Spoof Detection
# Training V2 - Robust Audio Augmentation
# Dataset: ASVspoof 2019 LA
# ============================================================


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

TRAIN_MANIFEST = (
    BASE_DIR
    / "manifests"
    / "train_manifest.csv"
)

DEV_MANIFEST = (
    BASE_DIR
    / "manifests"
    / "dev_manifest.csv"
)

CHECKPOINT_DIR = (
    BASE_DIR
    / "checkpoints"
)

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

BEST_CHECKPOINT = (
    CHECKPOINT_DIR
    / "swaraksha_cnn_best.pt"
)

LAST_CHECKPOINT = (
    CHECKPOINT_DIR
    / "swaraksha_cnn_last.pt"
)

HISTORY_FILE = (
    CHECKPOINT_DIR
    / "training_history.csv"
)


# ============================================================
# Reproducibility
# ============================================================

SEED = 42


def set_seed(seed: int = SEED):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


set_seed()


# ============================================================
# Device
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# Training configuration
# ============================================================

BATCH_SIZE = 32

EPOCHS = 20

LEARNING_RATE = 1e-3

WEIGHT_DECAY = 1e-4

PATIENCE = 5

NUM_WORKERS = 0

PIN_MEMORY = (
    DEVICE.type == "cuda"
)

USE_AMP = (
    DEVICE.type == "cuda"
)


# ============================================================
# Dataset
# ============================================================

class ASVspoofDataset(Dataset):

    def __init__(
        self,
        manifest_path,
        training=False,
    ):

        self.manifest_path = Path(
            manifest_path
        )

        self.training = training

        if not self.manifest_path.exists():

            raise FileNotFoundError(
                f"Manifest not found: "
                f"{self.manifest_path}"
            )

        self.records = []

        with open(
            self.manifest_path,
            "r",
            encoding="utf-8",
        ) as f:

            reader = csv.DictReader(f)

            for row in reader:

                audio_path = row[
                    "audio_path"
                ]

                raw_label = row["label"].strip().lower()

                if raw_label in ("bonafide", "real", "0"):
                    label = 0

                elif raw_label in ("spoof", "ai", "fake", "1"):
                    label = 1

                else:
                    raise ValueError(
                        f"Unknown label '{row['label']}' "
                        f"in {self.manifest_path}"
                    )

                self.records.append(
                    {
                        "audio_path": audio_path,
                        "label": label,
                    }
                )

        if len(self.records) == 0:

            raise RuntimeError(
                f"No records found in "
                f"{self.manifest_path}"
            )

        print()
        print(
            f"Loaded dataset: "
            f"{self.manifest_path.name}"
        )

        print(
            f"Samples: {len(self.records):,}"
        )

        print(
            f"Training augmentation: "
            f"{self.training}"
        )

    def __len__(self):

        return len(
            self.records
        )

    def __getitem__(
        self,
        index,
    ):

        record = self.records[
            index
        ]

        audio_path = record[
            "audio_path"
        ]

        label = record[
            "label"
        ]

        # ----------------------------------------------------
        # Training
        # ----------------------------------------------------

        if self.training:

            feature = (
                extract_training_feature_tensor(
                    audio_path
                )
            )

        # ----------------------------------------------------
        # Validation / evaluation
        # ----------------------------------------------------

        else:

            feature = (
                extract_feature_tensor(
                    audio_path
                )
            )

        return (
            feature,
            torch.tensor(
                label,
                dtype=torch.long,
            ),
        )


# ============================================================
# Load datasets
# ============================================================

print("=" * 70)
print(
    "SWARAKSHA - ROBUST VOICE SPOOF DETECTOR"
)
print("=" * 70)

print()
print("Device:")
print(DEVICE)

if DEVICE.type == "cuda":

    print()
    print(
        "GPU:"
    )

    print(
        torch.cuda.get_device_name(0)
    )

    print()
    print(
        "CUDA available:"
    )

    print(
        torch.cuda.is_available()
    )


train_dataset = ASVspoofDataset(
    TRAIN_MANIFEST,
    training=True,
)

dev_dataset = ASVspoofDataset(
    DEV_MANIFEST,
    training=False,
)


# ============================================================
# Data loaders
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    drop_last=False,
)

dev_loader = DataLoader(
    dev_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    drop_last=False,
)


# ============================================================
# Calculate class distribution
# ============================================================

train_labels = np.array(
    [
        record["label"]
        for record in train_dataset.records
    ]
)

real_count = int(
    np.sum(train_labels == 0)
)

spoof_count = int(
    np.sum(train_labels == 1)
)

print()
print("=" * 70)
print("CLASS DISTRIBUTION")
print("=" * 70)

print(
    f"REAL / BONAFIDE : {real_count:,}"
)

print(
    f"SPOOF / AI      : {spoof_count:,}"
)


# ============================================================
# Class weights
# ============================================================

total_samples = (
    real_count
    + spoof_count
)

class_weight_real = (
    total_samples
    / (2.0 * real_count)
)

class_weight_spoof = (
    total_samples
    / (2.0 * spoof_count)
)

class_weights = torch.tensor(
    [
        class_weight_real,
        class_weight_spoof,
    ],
    dtype=torch.float32,
    device=DEVICE,
)

print()
print(
    "Class weights:"
)

print(
    f"REAL  : {class_weight_real:.6f}"
)

print(
    f"SPOOF : {class_weight_spoof:.6f}"
)


# ============================================================
# Model
# ============================================================

model = SwarakshaCNN(
    num_classes=2
)

model = model.to(
    DEVICE
)


# ============================================================
# Parameter count
# ============================================================

trainable_parameters = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

total_parameters = sum(
    p.numel()
    for p in model.parameters()
)

print()
print("=" * 70)
print("MODEL")
print("=" * 70)

print(
    f"Total parameters: "
    f"{total_parameters:,}"
)

print(
    f"Trainable parameters: "
    f"{trainable_parameters:,}"
)


# ============================================================
# Loss
# ============================================================

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)


# ============================================================
# Optimizer
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)


# ============================================================
# Learning rate scheduler
# ============================================================

scheduler = (
    torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
    )
)


# ============================================================
# AMP
# ============================================================

scaler = GradScaler(
    enabled=USE_AMP
)


# ============================================================
# Validation function
# ============================================================

def evaluate(
    model,
    loader,
):

    model.eval()

    total_loss = 0.0

    all_labels = []

    all_predictions = []

    all_probabilities = []

    with torch.no_grad():

        for features, labels in loader:

            features = features.to(
                DEVICE,
                non_blocking=True,
            )

            labels = labels.to(
                DEVICE,
                non_blocking=True,
            )

            with autocast(
                enabled=USE_AMP
            ):

                outputs = model(
                    features
                )

                loss = criterion(
                    outputs,
                    labels,
                )

            probabilities = torch.softmax(
                outputs,
                dim=1,
            )

            predictions = torch.argmax(
                outputs,
                dim=1,
            )

            total_loss += (
                loss.item()
                * labels.size(0)
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            # Probability of spoof class
            all_probabilities.extend(
                probabilities[
                    :, 1
                ]
                .cpu()
                .numpy()
            )

    avg_loss = (
        total_loss
        / len(loader.dataset)
    )

    accuracy = accuracy_score(
        all_labels,
        all_predictions,
    )

    precision = precision_score(
        all_labels,
        all_predictions,
        zero_division=0,
    )

    recall = recall_score(
        all_labels,
        all_predictions,
        zero_division=0,
    )

    f1 = f1_score(
        all_labels,
        all_predictions,
        zero_division=0,
    )

    try:

        roc_auc = roc_auc_score(
            all_labels,
            all_probabilities,
        )

    except ValueError:

        roc_auc = float("nan")

    cm = confusion_matrix(
        all_labels,
        all_predictions,
        labels=[0, 1],
    )

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": cm,
    }


# ============================================================
# Training
# ============================================================

print()
print("=" * 70)
print("TRAINING")
print("=" * 70)

print()
print(
    "Training augmentation:"
)

print(
    "  Random gain"
)

print(
    "  Background noise"
)

print(
    "  Channel / low-pass variation"
)

print(
    "  Time shift"
)

print(
    "  Mild reverberation"
)

print()
print(
    "Validation augmentation:"
)

print(
    "  NONE"
)

print()
print(
    "Best checkpoint:"
)

print(
    BEST_CHECKPOINT
)

print()


history = []

best_f1 = -1.0

best_epoch = 0

epochs_without_improvement = 0


training_start = time.time()


for epoch in range(
    1,
    EPOCHS + 1,
):

    epoch_start = time.time()

    model.train()

    running_loss = 0.0

    correct = 0

    total = 0

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------

    for batch_index, (
        features,
        labels,
    ) in enumerate(
        train_loader,
        start=1,
    ):

        features = features.to(
            DEVICE,
            non_blocking=True,
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        with autocast(
            enabled=USE_AMP
        ):

            outputs = model(
                features
            )

            loss = criterion(
                outputs,
                labels,
            )

        scaler.scale(
            loss
        ).backward()

        scaler.step(
            optimizer
        )

        scaler.update()

        running_loss += (
            loss.item()
            * labels.size(0)
        )

        predictions = torch.argmax(
            outputs,
            dim=1,
        )

        correct += (
            predictions
            == labels
        ).sum().item()

        total += labels.size(0)

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            batch_index % 100 == 0
            or batch_index
            == len(train_loader)
        ):

            progress = (
                batch_index
                / len(train_loader)
            ) * 100.0

            print(
                f"\rEpoch "
                f"{epoch:02d}/{EPOCHS} | "
                f"Batch "
                f"{batch_index:04d}/"
                f"{len(train_loader):04d} "
                f"({progress:5.1f}%) | "
                f"Loss "
                f"{loss.item():.4f}",
                end="",
                flush=True,
            )

    print()

    # --------------------------------------------------------
    # Training metrics
    # --------------------------------------------------------

    train_loss = (
        running_loss
        / len(train_loader.dataset)
    )

    train_accuracy = (
        correct
        / total
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    validation = evaluate(
        model,
        dev_loader,
    )

    val_loss = validation[
        "loss"
    ]

    val_accuracy = validation[
        "accuracy"
    ]

    val_precision = validation[
        "precision"
    ]

    val_recall = validation[
        "recall"
    ]

    val_f1 = validation[
        "f1"
    ]

    val_roc_auc = validation[
        "roc_auc"
    ]

    cm = validation[
        "confusion_matrix"
    ]

    # --------------------------------------------------------
    # Scheduler
    # --------------------------------------------------------

    scheduler.step(
        val_f1
    )

    current_lr = (
        optimizer.param_groups[0]["lr"]
    )

    epoch_time = (
        time.time()
        - epoch_start
    )

    # --------------------------------------------------------
    # Print epoch result
    # --------------------------------------------------------

    print()
    print(
        "-" * 70
    )

    print(
        f"Epoch {epoch}/{EPOCHS}"
    )

    print(
        f"Time          : "
        f"{epoch_time:.2f}s"
    )

    print(
        f"Learning Rate : "
        f"{current_lr:.8f}"
    )

    print(
        f"Train Loss    : "
        f"{train_loss:.4f}"
    )

    print(
        f"Train Accuracy: "
        f"{train_accuracy * 100:.2f}%"
    )

    print(
        f"Val Loss      : "
        f"{val_loss:.4f}"
    )

    print(
        f"Val Accuracy  : "
        f"{val_accuracy * 100:.2f}%"
    )

    print(
        f"Val Precision : "
        f"{val_precision * 100:.2f}%"
    )

    print(
        f"Val Recall    : "
        f"{val_recall * 100:.2f}%"
    )

    print(
        f"Val F1        : "
        f"{val_f1 * 100:.2f}%"
    )

    print(
        f"Val ROC-AUC   : "
        f"{val_roc_auc:.4f}"
    )

    print()
    print(
        "Confusion Matrix:"
    )

    print(
        "              Predicted"
    )

    print(
        "              REAL  SPOOF"
    )

    print(
        f"Actual REAL   "
        f"{cm[0, 0]:5d} "
        f"{cm[0, 1]:5d}"
    )

    print(
        f"Actual SPOOF  "
        f"{cm[1, 0]:5d} "
        f"{cm[1, 1]:5d}"
    )

    # --------------------------------------------------------
    # Save history
    # --------------------------------------------------------

    history.append(
        {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "val_precision": val_precision,
            "val_recall": val_recall,
            "val_f1": val_f1,
            "val_roc_auc": val_roc_auc,
            "learning_rate": current_lr,
            "epoch_time_seconds": epoch_time,
        }
    )

    history_df = pd.DataFrame(
        history
    )

    history_df.to_csv(
        HISTORY_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Save last checkpoint
    # --------------------------------------------------------

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict":
                model.state_dict(),
            "optimizer_state_dict":
                optimizer.state_dict(),
            "scheduler_state_dict":
                scheduler.state_dict(),
            "scaler_state_dict":
                scaler.state_dict(),
            "val_f1": val_f1,
            "val_accuracy":
                val_accuracy,
            "val_precision":
                val_precision,
            "val_recall":
                val_recall,
            "val_roc_auc":
                val_roc_auc,
        },
        LAST_CHECKPOINT,
    )

    # --------------------------------------------------------
    # Best checkpoint
    # --------------------------------------------------------

    if val_f1 > best_f1:

        best_f1 = val_f1

        best_epoch = epoch

        epochs_without_improvement = 0

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict":
                    model.state_dict(),
                "optimizer_state_dict":
                    optimizer.state_dict(),
                "scheduler_state_dict":
                    scheduler.state_dict(),
                "scaler_state_dict":
                    scaler.state_dict(),
                "best_val_f1":
                    best_f1,
                "val_accuracy":
                    val_accuracy,
                "val_precision":
                    val_precision,
                "val_recall":
                    val_recall,
                "val_roc_auc":
                    val_roc_auc,
            },
            BEST_CHECKPOINT,
        )

        print()
        print(
            f"*** NEW BEST MODEL "
            f"(F1 = "
            f"{best_f1 * 100:.2f}%) ***"
        )

    else:

        epochs_without_improvement += 1

        print()
        print(
            f"No improvement. "
            f"Patience: "
            f"{epochs_without_improvement}/"
            f"{PATIENCE}"
        )

    # --------------------------------------------------------
    # Early stopping
    # --------------------------------------------------------

    if (
        epochs_without_improvement
        >= PATIENCE
    ):

        print()
        print(
            "=" * 70
        )

        print(
            "EARLY STOPPING"
        )

        print(
            f"No F1 improvement for "
            f"{PATIENCE} epochs."
        )

        print(
            "=" * 70
        )

        break


# ============================================================
# Training complete
# ============================================================

total_training_time = (
    time.time()
    - training_start
)


print()
print("=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print()
print(
    f"Best Epoch : {best_epoch}"
)

print(
    f"Best Val F1: "
    f"{best_f1 * 100:.2f}%"
)

print(
    f"Total Time : "
    f"{total_training_time / 60:.2f} minutes"
)

print()
print(
    "Best checkpoint:"
)

print(
    BEST_CHECKPOINT
)

print()
print(
    "Last checkpoint:"
)

print(
    LAST_CHECKPOINT
)

print()
print(
    "Training history:"
)

print(
    HISTORY_FILE
)

print()
print("=" * 70)
print("SWARAKSHA TRAINING V2 FINISHED")
print("=" * 70)