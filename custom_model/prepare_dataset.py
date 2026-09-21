from pathlib import Path
import csv


# ============================================================
# SWARAKSHA - ASVspoof 2019 LA Dataset Preparation
# ============================================================

PROJECT_ROOT = Path(r"C:\VIGILVOICE")

DATASET_ROOT = PROJECT_ROOT / "data" / "voice_dataset"

TRAIN_AUDIO_DIR = (
    DATASET_ROOT / "ASVspoof2019_LA_train" / "flac"
)

DEV_AUDIO_DIR = (
    DATASET_ROOT / "ASVspoof2019_LA_dev" / "flac"
)

PROTOCOL_DIR = (
    DATASET_ROOT / "ASVspoof2019_LA_cm_protocols"
)

TRAIN_PROTOCOL = (
    PROTOCOL_DIR / "ASVspoof2019.LA.cm.train.trn.txt"
)

DEV_PROTOCOL = (
    PROTOCOL_DIR / "ASVspoof2019.LA.cm.dev.trl.txt"
)

OUTPUT_DIR = PROJECT_ROOT / "custom_model" / "manifests"


# ============================================================
# Read protocol
# ============================================================

def read_protocol(protocol_path: Path):
    records = []

    with open(protocol_path, "r", encoding="utf-8") as f:

        for line_number, line in enumerate(f, start=1):

            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) < 5:
                print(
                    f"[WARNING] Invalid line {line_number}: {line}"
                )
                continue

            speaker_id = parts[0]
            audio_id = parts[1]
            label = parts[-1].lower()

            if label not in {"bonafide", "spoof"}:
                print(
                    f"[WARNING] Unknown label at line "
                    f"{line_number}: {label}"
                )
                continue

            records.append(
                {
                    "speaker_id": speaker_id,
                    "audio_id": audio_id,
                    "label": label,
                }
            )

    return records


# ============================================================
# Build manifest
# ============================================================

def build_manifest(
    records,
    audio_directory: Path,
    output_file: Path,
    dataset_name: str,
):

    valid_records = []
    missing_files = []

    for record in records:

        audio_path = (
            audio_directory /
            f"{record['audio_id']}.flac"
        )

        if audio_path.exists():

            valid_records.append(
                {
                    "audio_path": str(audio_path.resolve()),
                    "audio_id": record["audio_id"],
                    "speaker_id": record["speaker_id"],
                    "label": record["label"],
                }
            )

        else:
            missing_files.append(record["audio_id"])

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        output_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "audio_path",
                "audio_id",
                "speaker_id",
                "label",
            ],
        )

        writer.writeheader()
        writer.writerows(valid_records)

    bonafide_count = sum(
        1
        for r in valid_records
        if r["label"] == "bonafide"
    )

    spoof_count = sum(
        1
        for r in valid_records
        if r["label"] == "spoof"
    )

    print()
    print("=" * 60)
    print(f"{dataset_name.upper()} MANIFEST")
    print("=" * 60)

    print(f"Protocol records : {len(records)}")
    print(f"Valid audio      : {len(valid_records)}")
    print(f"Missing audio    : {len(missing_files)}")
    print(f"Bonafide         : {bonafide_count}")
    print(f"Spoof            : {spoof_count}")
    print(f"Output           : {output_file}")

    if missing_files:

        print()
        print("First missing files:")

        for audio_id in missing_files[:10]:
            print(f"  {audio_id}.flac")

    return valid_records


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 60)
    print("SWARAKSHA - DATASET PREPARATION")
    print("ASVspoof 2019 LA")
    print("=" * 60)

    print()
    print("Dataset root:")
    print(DATASET_ROOT)

    print()
    print("Reading training protocol...")

    train_records = read_protocol(
        TRAIN_PROTOCOL
    )

    print(
        f"Training protocol records: "
        f"{len(train_records)}"
    )

    print()
    print("Reading development protocol...")

    dev_records = read_protocol(
        DEV_PROTOCOL
    )

    print(
        f"Development protocol records: "
        f"{len(dev_records)}"
    )

    # --------------------------------------------------------
    # Training manifest
    # --------------------------------------------------------

    train_output = (
        OUTPUT_DIR / "train_manifest.csv"
    )

    build_manifest(
        records=train_records,
        audio_directory=TRAIN_AUDIO_DIR,
        output_file=train_output,
        dataset_name="train",
    )

    # --------------------------------------------------------
    # Development manifest
    # --------------------------------------------------------

    dev_output = (
        OUTPUT_DIR / "dev_manifest.csv"
    )

    build_manifest(
        records=dev_records,
        audio_directory=DEV_AUDIO_DIR,
        output_file=dev_output,
        dataset_name="dev",
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("DATASET PREPARATION COMPLETE")
    print("=" * 60)

    print()
    print("Created files:")

    print(f"  {train_output}")
    print(f"  {dev_output}")

    print()
    print("Original ASVspoof files were NOT modified.")

    print()
    print("Ready for feature extraction.")


if __name__ == "__main__":
    main()