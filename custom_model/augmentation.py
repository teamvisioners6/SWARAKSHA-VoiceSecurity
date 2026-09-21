import random
import numpy as np
import librosa


# ============================================================
# SWARAKSHA V2 - REAL WORLD AUDIO AUGMENTATION
# ============================================================

SAMPLE_RATE = 16000


def random_gain(audio, min_db=-6.0, max_db=6.0):
    """
    Randomly change microphone recording volume.
    """

    gain_db = random.uniform(min_db, max_db)

    gain = 10 ** (gain_db / 20.0)

    audio = audio * gain

    return np.clip(audio, -1.0, 1.0)


def add_noise(audio, min_snr_db=15.0, max_snr_db=35.0):
    """
    Add controlled background noise.

    This simulates:
    - room noise
    - fan noise
    - environmental noise
    - imperfect phone microphones
    """

    signal_power = np.mean(audio ** 2)

    if signal_power < 1e-10:
        return audio

    snr_db = random.uniform(
        min_snr_db,
        max_snr_db
    )

    noise_power = signal_power / (
        10 ** (snr_db / 10.0)
    )

    noise = np.random.normal(
        0.0,
        np.sqrt(noise_power),
        size=audio.shape
    ).astype(np.float32)

    return np.clip(
        audio + noise,
        -1.0,
        1.0
    )


def apply_lowpass(audio):
    """
    Simulate limited microphone / telephone bandwidth.
    """

    cutoff = random.choice([
        3400,
        4000,
        6000,
        7000,
    ])

    audio = librosa.resample(
        audio,
        orig_sr=SAMPLE_RATE,
        target_sr=cutoff * 2
    )

    audio = librosa.resample(
        audio,
        orig_sr=cutoff * 2,
        target_sr=SAMPLE_RATE
    )

    return audio.astype(np.float32)


def time_shift(audio):
    """
    Small temporal shift.
    """

    max_shift = int(
        0.15 * SAMPLE_RATE
    )

    shift = random.randint(
        -max_shift,
        max_shift
    )

    if shift > 0:
        audio = np.concatenate([
            np.zeros(shift, dtype=np.float32),
            audio[:-shift]
        ])

    elif shift < 0:
        shift = abs(shift)

        audio = np.concatenate([
            audio[shift:],
            np.zeros(shift, dtype=np.float32)
        ])

    return audio


def apply_reverb(audio):
    """
    Lightweight synthetic room reverberation.
    """

    if random.random() > 0.25:
        return audio

    delay_ms = random.uniform(
        15,
        60
    )

    decay = random.uniform(
        0.1,
        0.35
    )

    delay = int(
        SAMPLE_RATE * delay_ms / 1000
    )

    if delay >= len(audio):
        return audio

    reverberated = audio.copy()

    reverberated[delay:] += (
        decay * audio[:-delay]
    )

    return np.clip(
        reverberated,
        -1.0,
        1.0
    )


def apply_random_augmentation(audio):
    """
    Apply a random combination of realistic
    recording/channel distortions.

    Used ONLY during training.
    """

    audio = audio.astype(
        np.float32,
        copy=True
    )

    # Random gain
    if random.random() < 0.50:
        audio = random_gain(audio)

    # Background noise
    if random.random() < 0.45:
        audio = add_noise(audio)

    # Telephone/channel bandwidth
    if random.random() < 0.25:
        audio = apply_lowpass(audio)

    # Small temporal shift
    if random.random() < 0.20:
        audio = time_shift(audio)

    # Room reverberation
    audio = apply_reverb(audio)

    # Final normalization
    peak = np.max(
        np.abs(audio)
    )

    if peak > 1e-8:
        audio = audio / peak

    return audio.astype(
        np.float32
    )