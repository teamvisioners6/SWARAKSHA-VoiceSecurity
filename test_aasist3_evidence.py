import os
import warnings
import numpy as np
import librosa


# ============================================================
# VIGILVOICE - AASIST3 SUPPORTING ACOUSTIC EVIDENCE
# ============================================================
#
# PURPOSE:
#   Analyze supporting acoustic / phonetic evidence for
#   REAL vs AI-generated speech.
#
# PRIMARY DETECTOR:
#   Spectra-AASIST3 remains the main detector.
#
# THIS SCRIPT DOES NOT:
#   - retrain AASIST3
#   - modify AASIST3
#   - replace AASIST3
#
# ============================================================


SR = 16000

# ============================================================
# TEST FILES
# ============================================================

TEST_FILES = [
    (
        r"C:\VIGILVOICE\my_real_voice.m4a",
        "REAL"
    ),
    (
        r"C:\VIGILVOICE\ai 5 40.mpeg",
        "AI"
    ),
    (
        r"C:\VIGILVOICE\synthetic.wav",
        "AI"
    ),
]


# ============================================================
# AUDIO
# ============================================================

def load_audio(path):

    with warnings.catch_warnings():

        warnings.simplefilter("ignore")

        audio, sr = librosa.load(
            path,
            sr=SR,
            mono=True
        )

    audio = np.asarray(
        audio,
        dtype=np.float32
    )

    audio = np.nan_to_num(
        audio
    )

    if len(audio) == 0:
        raise ValueError(
            "Empty audio."
        )

    # Peak normalization
    peak = np.max(
        np.abs(audio)
    )

    if peak > 1e-8:
        audio = audio / peak

    return audio


# ============================================================
# BASIC SIGNAL FEATURES
# ============================================================

def basic_features(audio):

    rms = librosa.feature.rms(
        y=audio,
        frame_length=1024,
        hop_length=160
    )[0]

    zcr = librosa.feature.zero_crossing_rate(
        audio,
        frame_length=1024,
        hop_length=160
    )[0]

    centroid = librosa.feature.spectral_centroid(
        y=audio,
        sr=SR,
        n_fft=1024,
        hop_length=160
    )[0]

    bandwidth = librosa.feature.spectral_bandwidth(
        y=audio,
        sr=SR,
        n_fft=1024,
        hop_length=160
    )[0]

    flatness = librosa.feature.spectral_flatness(
        y=audio,
        n_fft=1024,
        hop_length=160
    )[0]

    return {
        "rms_mean": float(np.mean(rms)),
        "rms_std": float(np.std(rms)),
        "zcr_mean": float(np.mean(zcr)),
        "zcr_std": float(np.std(zcr)),
        "centroid_mean": float(np.mean(centroid)),
        "centroid_std": float(np.std(centroid)),
        "bandwidth_mean": float(np.mean(bandwidth)),
        "bandwidth_std": float(np.std(bandwidth)),
        "flatness_mean": float(np.mean(flatness)),
        "flatness_std": float(np.std(flatness)),
    }


# ============================================================
# MFCC DYNAMICS
# ============================================================

def mfcc_features(audio):

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SR,
        n_mfcc=20,
        n_fft=1024,
        hop_length=160
    )

    delta = librosa.feature.delta(
        mfcc
    )

    delta2 = librosa.feature.delta(
        mfcc,
        order=2
    )

    # Temporal variation
    mfcc_var = np.mean(
        np.std(
            mfcc,
            axis=1
        )
    )

    delta_var = np.mean(
        np.std(
            delta,
            axis=1
        )
    )

    delta2_var = np.mean(
        np.std(
            delta2,
            axis=1
        )
    )

    # Frame-to-frame movement
    mfcc_motion = np.mean(
        np.abs(
            np.diff(
                mfcc,
                axis=1
            )
        )
    )

    return {
        "mfcc_variation": float(
            mfcc_var
        ),
        "mfcc_delta_variation": float(
            delta_var
        ),
        "mfcc_delta2_variation": float(
            delta2_var
        ),
        "mfcc_temporal_motion": float(
            mfcc_motion
        ),
    }


# ============================================================
# PITCH / PROSODY
# ============================================================

def pitch_features(audio):

    try:

        f0, voiced_flag, voiced_prob = (
            librosa.pyin(
                audio,
                fmin=librosa.note_to_hz("C2"),
                fmax=librosa.note_to_hz("C7"),
                sr=SR,
                frame_length=1024,
                hop_length=160
            )
        )

    except Exception:

        return {
            "f0_mean": 0.0,
            "f0_std": 0.0,
            "f0_cv": 0.0,
            "f0_range": 0.0,
            "f0_motion": 0.0,
            "voiced_ratio": 0.0,
            "pitch_direction_changes": 0.0,
        }

    valid = (
        np.isfinite(f0) &
        (f0 > 70) &
        (f0 < 800) &
        (voiced_prob > 0.30)
    )

    pitch = f0[valid]

    if len(pitch) < 3:

        return {
            "f0_mean": 0.0,
            "f0_std": 0.0,
            "f0_cv": 0.0,
            "f0_range": 0.0,
            "f0_motion": 0.0,
            "voiced_ratio": 0.0,
            "pitch_direction_changes": 0.0,
        }

    f0_mean = np.mean(
        pitch
    )

    f0_std = np.std(
        pitch
    )

    f0_range = (
        np.percentile(
            pitch,
            95
        )
        -
        np.percentile(
            pitch,
            5
        )
    )

    f0_cv = (
        f0_std /
        (f0_mean + 1e-8)
    )

    pitch_motion = np.diff(
        pitch
    )

    # Normalize jumps
    relative_motion = (
        np.abs(
            pitch_motion
        )
        /
        (pitch[:-1] + 1e-8)
    )

    # Direction changes
    signs = np.sign(
        pitch_motion
    )

    signs = signs[
        signs != 0
    ]

    if len(signs) >= 2:

        direction_changes = np.mean(
            signs[1:] != signs[:-1]
        )

    else:

        direction_changes = 0.0

    return {
        "f0_mean": float(
            f0_mean
        ),
        "f0_std": float(
            f0_std
        ),
        "f0_cv": float(
            f0_cv
        ),
        "f0_range": float(
            f0_range
        ),
        "f0_motion": float(
            np.mean(
                relative_motion
            )
        ),
        "voiced_ratio": float(
            np.mean(valid)
        ),
        "pitch_direction_changes": float(
            direction_changes
        ),
    }


# ============================================================
# HARMONIC / NOISE EVIDENCE
# ============================================================

def harmonic_features(audio):

    try:

        harmonic, percussive = (
            librosa.effects.hpss(
                audio
            )
        )

        harmonic_energy = np.mean(
            harmonic ** 2
        )

        total_energy = np.mean(
            audio ** 2
        )

        harmonic_ratio = (
            harmonic_energy /
            (total_energy + 1e-10)
        )

        residual = (
            audio - harmonic
        )

        noise_energy = np.mean(
            residual ** 2
        )

        harmonic_noise_ratio = (
            harmonic_energy /
            (noise_energy + 1e-10)
        )

    except Exception:

        harmonic_ratio = 0.0
        harmonic_noise_ratio = 0.0

    return {
        "harmonic_ratio": float(
            harmonic_ratio
        ),
        "harmonic_noise_ratio": float(
            harmonic_noise_ratio
        ),
    }


# ============================================================
# SPECTRAL DYNAMICS
# ============================================================

def spectral_dynamics(audio):

    centroid = librosa.feature.spectral_centroid(
        y=audio,
        sr=SR,
        n_fft=1024,
        hop_length=160
    )[0]

    bandwidth = librosa.feature.spectral_bandwidth(
        y=audio,
        sr=SR,
        n_fft=1024,
        hop_length=160
    )[0]

    rolloff = librosa.feature.spectral_rolloff(
        y=audio,
        sr=SR,
        roll_percent=0.85,
        n_fft=1024,
        hop_length=160
    )[0]

    centroid_diff = np.diff(
        centroid
    )

    bandwidth_diff = np.diff(
        bandwidth
    )

    rolloff_diff = np.diff(
        rolloff
    )

    return {
        "centroid_motion": float(
            np.mean(
                np.abs(
                    centroid_diff
                )
            )
        ),
        "bandwidth_motion": float(
            np.mean(
                np.abs(
                    bandwidth_diff
                )
            )
        ),
        "rolloff_motion": float(
            np.mean(
                np.abs(
                    rolloff_diff
                )
            )
        ),
    }


# ============================================================
# VOICED / UNVOICED STRUCTURE
# ============================================================

def voiced_unvoiced_features(audio):

    rms = librosa.feature.rms(
        y=audio,
        frame_length=1024,
        hop_length=160
    )[0]

    threshold = (
        np.percentile(
            rms,
            25
        )
    )

    voiced = (
        rms > threshold
    )

    voiced_ratio = np.mean(
        voiced
    )

    transitions = np.sum(
        voiced[1:] != voiced[:-1]
    )

    transition_rate = (
        transitions /
        max(
            len(voiced),
            1
        )
    )

    # Average run lengths
    runs = []

    if len(voiced) > 0:

        current = voiced[0]
        count = 1

        for value in voiced[1:]:

            if value == current:

                count += 1

            else:

                runs.append(
                    count
                )

                current = value
                count = 1

        runs.append(
            count
        )

    if runs:

        mean_run = np.mean(
            runs
        )

        run_std = np.std(
            runs
        )

    else:

        mean_run = 0.0
        run_std = 0.0

    return {
        "voiced_frame_ratio": float(
            voiced_ratio
        ),
        "voiced_unvoiced_transition_rate": float(
            transition_rate
        ),
        "voiced_run_mean": float(
            mean_run
        ),
        "voiced_run_std": float(
            run_std
        ),
    }


# ============================================================
# FEATURE COLLECTION
# ============================================================

def extract_all(audio):

    features = {}

    features.update(
        basic_features(
            audio
        )
    )

    features.update(
        mfcc_features(
            audio
        )
    )

    features.update(
        pitch_features(
            audio
        )
    )

    features.update(
        harmonic_features(
            audio
        )
    )

    features.update(
        spectral_dynamics(
            audio
        )
    )

    features.update(
        voiced_unvoiced_features(
            audio
        )
    )

    return features


# ============================================================
# NORMALIZED EVIDENCE SCORES
# ============================================================

def compute_evidence(features):

    """
    IMPORTANT:

    These are NOT trained probabilities.

    They are diagnostic evidence scores only.

    We intentionally avoid claiming that any individual
    feature means "AI".
    """

    evidence = {}

    # --------------------------------------------------------
    # Pitch structure
    # --------------------------------------------------------

    pitch_motion = features[
        "f0_motion"
    ]

    pitch_cv = features[
        "f0_cv"
    ]

    direction = features[
        "pitch_direction_changes"
    ]

    pitch_score = np.clip(
        (
            pitch_motion * 0.35
            +
            pitch_cv * 0.30
            +
            direction * 0.35
        ),
        0,
        1
    )

    evidence[
        "pitch_structure"
    ] = float(
        pitch_score
    )


    # --------------------------------------------------------
    # Spectral movement
    # --------------------------------------------------------

    spectral_motion = (
        features["centroid_motion"]
        /
        (
            features["centroid_mean"]
            + 1e-8
        )
    )

    spectral_score = np.clip(
        spectral_motion * 5.0,
        0,
        1
    )

    evidence[
        "spectral_structure"
    ] = float(
        spectral_score
    )


    # --------------------------------------------------------
    # Harmonic / noise
    # --------------------------------------------------------

    harmonic = features[
        "harmonic_ratio"
    ]

    harmonic_score = np.clip(
        harmonic,
        0,
        1
    )

    evidence[
        "harmonic_structure"
    ] = float(
        harmonic_score
    )


    # --------------------------------------------------------
    # MFCC dynamics
    # --------------------------------------------------------

    mfcc_motion = features[
        "mfcc_temporal_motion"
    ]

    mfcc_score = np.clip(
        mfcc_motion / 20.0,
        0,
        1
    )

    evidence[
        "cepstral_dynamics"
    ] = float(
        mfcc_score
    )


    # --------------------------------------------------------
    # Voiced/unvoiced
    # --------------------------------------------------------

    vu = features[
        "voiced_unvoiced_transition_rate"
    ]

    vu_score = np.clip(
        vu * 10.0,
        0,
        1
    )

    evidence[
        "voiced_unvoiced_structure"
    ] = float(
        vu_score
    )


    return evidence


# ============================================================
# MAIN
# ============================================================

print("=" * 75)
print("VIGILVOICE - SUPPORTING ACOUSTIC EVIDENCE")
print("=" * 75)

print()
print("AASIST3 remains the PRIMARY detector.")
print("Evidence scores below are diagnostic, not probabilities.")
print()


for path, expected in TEST_FILES:

    print("=" * 75)

    name = os.path.basename(
        path
    )

    print(
        "FILE:",
        name
    )

    print(
        "EXPECTED:",
        expected
    )

    if not os.path.exists(path):

        print(
            "STATUS: FILE NOT FOUND"
        )

        continue

    try:

        audio = load_audio(
            path
        )

        duration = (
            len(audio) /
            SR
        )

        features = extract_all(
            audio
        )

        evidence = compute_evidence(
            features
        )


        print()
        print(
            f"Duration: {duration:.2f} sec"
        )

        # ----------------------------------------------------
        # Raw acoustic features
        # ----------------------------------------------------

        print()
        print(
            "--- PITCH / PROSODY ---"
        )

        print(
            f"F0 mean                 : "
            f"{features['f0_mean']:.2f} Hz"
        )

        print(
            f"F0 std                  : "
            f"{features['f0_std']:.2f}"
        )

        print(
            f"F0 CV                   : "
            f"{features['f0_cv']:.4f}"
        )

        print(
            f"F0 range                : "
            f"{features['f0_range']:.2f}"
        )

        print(
            f"F0 motion               : "
            f"{features['f0_motion']:.4f}"
        )

        print(
            f"Voiced ratio            : "
            f"{features['voiced_ratio']:.4f}"
        )

        print(
            f"Pitch direction changes : "
            f"{features['pitch_direction_changes']:.4f}"
        )


        print()
        print(
            "--- SPECTRAL ---"
        )

        print(
            f"Centroid mean           : "
            f"{features['centroid_mean']:.2f}"
        )

        print(
            f"Centroid std            : "
            f"{features['centroid_std']:.2f}"
        )

        print(
            f"Bandwidth mean          : "
            f"{features['bandwidth_mean']:.2f}"
        )

        print(
            f"Bandwidth std           : "
            f"{features['bandwidth_std']:.2f}"
        )

        print(
            f"Spectral flatness       : "
            f"{features['flatness_mean']:.6f}"
        )


        print()
        print(
            "--- HARMONIC / NOISE ---"
        )

        print(
            f"Harmonic ratio          : "
            f"{features['harmonic_ratio']:.4f}"
        )

        print(
            f"Harmonic/noise ratio    : "
            f"{features['harmonic_noise_ratio']:.4f}"
        )


        print()
        print(
            "--- CEPSTRAL ---"
        )

        print(
            f"MFCC variation          : "
            f"{features['mfcc_variation']:.4f}"
        )

        print(
            f"MFCC temporal motion    : "
            f"{features['mfcc_temporal_motion']:.4f}"
        )


        print()
        print(
            "--- VOICED / UNVOICED ---"
        )

        print(
            f"Frame voiced ratio      : "
            f"{features['voiced_frame_ratio']:.4f}"
        )

        print(
            f"V/U transition rate     : "
            f"{features['voiced_unvoiced_transition_rate']:.4f}"
        )

        print(
            f"Voiced run mean         : "
            f"{features['voiced_run_mean']:.2f}"
        )


        print()
        print(
            "--- DIAGNOSTIC EVIDENCE ---"
        )

        for key, value in evidence.items():

            print(
                f"{key:30s}: "
                f"{value:.4f}"
            )


    except Exception as e:

        print()
        print(
            "ERROR:",
            repr(e)
        )


print()
print("=" * 75)
print("EVIDENCE ANALYSIS COMPLETE")
print("=" * 75)