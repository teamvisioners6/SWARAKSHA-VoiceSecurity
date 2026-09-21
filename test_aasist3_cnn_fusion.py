import os
import warnings
import numpy as np
import librosa
import torch
import torch.nn as nn
import onnxruntime as ort


# ============================================================
# VIGILVOICE - FAST AASIST3 + ROBUST CNN DIAGNOSTIC
# ============================================================

SR = 16000

CNN_WINDOW = 32000
CNN_HOP = 16000
TARGET_FRAMES = 201

AASIST_LENGTH = 64600
AASIST_HOP = AASIST_LENGTH // 2

CNN_MODEL_PATH = (
    r"C:\VIGILVOICE\models\vigilvoice_robust_cnn_v1.pth"
)

AASIST_MODEL_PATH = (
    r"C:\VIGILVOICE\models\spectra_aasist3"
    r"\spectra-aasist3.onnx"
)

# ============================================================
# TEST FILES
# ============================================================

TEST_FILES = [
    r"C:\VIGILVOICE\my_real_voice.m4a",
    r"C:\VIGILVOICE\aadu ai voice.mpeg",
    r"C:\VIGILVOICE\ai 5 40.mpeg",
    r"C:\VIGILVOICE\synthetic.wav",
]


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 75)
print("VIGILVOICE - FAST AASIST3 + ROBUST CNN DIAGNOSTIC")
print("=" * 75)

print("Device:", DEVICE)


# ============================================================
# ROBUST CNN
# EXACT TRAINED ARCHITECTURE
# ============================================================

class Branch(nn.Module):

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

    def forward(self, x):
        return self.features(x).flatten(1)


class VigilVoiceRobustCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.lfcc_branch = Branch()
        self.mel_branch = Branch()

        self.classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.40),
            nn.Linear(128, 2)
        )

    def forward(self, lfcc, mel):

        lfcc = lfcc.unsqueeze(1)
        mel = mel.unsqueeze(1)

        a = self.lfcc_branch(lfcc)
        b = self.mel_branch(mel)

        return self.classifier(
            torch.cat([a, b], dim=1)
        )


# ============================================================
# CNN FEATURES
# ============================================================

def fix_size(x):

    if x.shape[1] > TARGET_FRAMES:

        x = x[:, :TARGET_FRAMES]

    elif x.shape[1] < TARGET_FRAMES:

        x = np.pad(
            x,
            (
                (0, 0),
                (0, TARGET_FRAMES - x.shape[1])
            )
        )

    return x.astype(np.float32)


def preemphasis(x):

    y = np.empty_like(x)

    y[0] = x[0]

    if len(x) > 1:

        y[1:] = (
            x[1:] -
            0.97 * x[:-1]
        )

    return y


def dct2(x, n=20):

    nf = x.shape[0]

    k = np.arange(n)[:, None]
    m = np.arange(nf)[None, :]

    basis = np.cos(
        np.pi *
        (m + 0.5) *
        k /
        nf
    )

    return np.dot(
        basis,
        x
    )


def lfcc(audio):

    audio = preemphasis(audio)

    stft = librosa.stft(
        audio,
        n_fft=512,
        hop_length=160,
        win_length=400,
        window="hamming",
        center=False
    )

    power = np.abs(stft) ** 2

    nf = power.shape[0]

    points = np.linspace(
        50,
        7600,
        22
    )

    bins = np.floor(
        (513 * points) / SR
    ).astype(int)

    fb = np.zeros(
        (20, nf),
        dtype=np.float32
    )

    for m in range(1, 21):

        left = max(
            0,
            min(
                bins[m - 1],
                nf - 1
            )
        )

        center = max(
            0,
            min(
                bins[m],
                nf - 1
            )
        )

        right = max(
            0,
            min(
                bins[m + 1],
                nf - 1
            )
        )

        if center > left:

            fb[
                m - 1,
                left:center
            ] = np.linspace(
                0,
                1,
                center - left,
                endpoint=False
            )

        if right > center:

            fb[
                m - 1,
                center:right
            ] = np.linspace(
                1,
                0,
                right - center,
                endpoint=False
            )

    energy = np.maximum(
        np.dot(fb, power),
        1e-10
    )

    static = dct2(
        np.log(energy)
    )

    delta = librosa.feature.delta(
        static
    )

    delta2 = librosa.feature.delta(
        static,
        order=2
    )

    x = np.concatenate(
        [
            static,
            delta,
            delta2
        ],
        axis=0
    )

    x = (
        x -
        x.mean(
            axis=1,
            keepdims=True
        )
    ) / (
        x.std(
            axis=1,
            keepdims=True
        ) + 1e-6
    )

    return fix_size(x)


def mel(audio):

    x = librosa.feature.melspectrogram(
        y=audio,
        sr=SR,
        n_fft=512,
        hop_length=160,
        win_length=400,
        window="hann",
        n_mels=40,
        fmin=50,
        fmax=7600,
        center=False,
        power=2
    )

    x = librosa.power_to_db(
        x,
        ref=np.max
    )

    x = (
        x -
        x.mean(
            axis=1,
            keepdims=True
        )
    ) / (
        x.std(
            axis=1,
            keepdims=True
        ) + 1e-6
    )

    return fix_size(x)


# ============================================================
# LOAD CNN
# ============================================================

print()
print("[1/3] Loading Robust CNN...")

cnn = VigilVoiceRobustCNN().to(
    DEVICE
)

checkpoint = torch.load(
    CNN_MODEL_PATH,
    map_location=DEVICE,
    weights_only=False
)

cnn.load_state_dict(
    checkpoint["model_state_dict"]
)

cnn.eval()

print("Robust CNN loaded.")


# ============================================================
# LOAD AASIST3
# ============================================================

print()
print("[2/3] Loading Spectra-AASIST3...")

try:
    ort.preload_dlls(
        directory=""
    )
except Exception:
    pass

available = ort.get_available_providers()

if "CUDAExecutionProvider" in available:

    providers = [
        "CUDAExecutionProvider",
        "CPUExecutionProvider"
    ]

else:

    providers = [
        "CPUExecutionProvider"
    ]

aasist = ort.InferenceSession(
    AASIST_MODEL_PATH,
    providers=providers
)

AASIST_INPUT = (
    aasist.get_inputs()[0].name
)

AASIST_OUTPUT = (
    aasist.get_outputs()[0].name
)

print(
    "AASIST3 providers:",
    aasist.get_providers()
)


# ============================================================
# SOFTMAX
# ============================================================

def softmax(x):

    x = np.asarray(
        x,
        dtype=np.float32
    )

    x -= np.max(
        x,
        axis=-1,
        keepdims=True
    )

    e = np.exp(x)

    return (
        e /
        np.sum(
            e,
            axis=-1,
            keepdims=True
        )
    )


# ============================================================
# AASIST3 SINGLE SEGMENT
# ============================================================

def aasist_segment(audio):

    audio = np.asarray(
        audio,
        dtype=np.float32
    )

    emphasized = np.empty_like(
        audio
    )

    emphasized[0] = audio[0]

    if len(audio) > 1:

        emphasized[1:] = (
            audio[1:] -
            0.97 * audio[:-1]
        )

    audio = emphasized

    if len(audio) < AASIST_LENGTH:

        repeats = int(
            np.ceil(
                AASIST_LENGTH /
                len(audio)
            )
        )

        audio = np.tile(
            audio,
            repeats
        )

    audio = audio[
        :AASIST_LENGTH
    ]

    output = aasist.run(
        [AASIST_OUTPUT],
        {
            AASIST_INPUT:
            audio.reshape(
                1,
                AASIST_LENGTH
            ).astype(np.float32)
        }
    )[0]

    probability = softmax(
        output
    )[0]

    return float(
        probability[0]
    )


# ============================================================
# AASIST3 FILE
# ============================================================

def predict_aasist(path):

    with warnings.catch_warnings():

        warnings.simplefilter(
            "ignore"
        )

        audio, _ = librosa.load(
            path,
            sr=SR,
            mono=True
        )

    if len(audio) == 0:
        return 0.0, []


    if len(audio) <= AASIST_LENGTH:

        score = aasist_segment(
            audio
        )

        return score, [score]


    starts = list(
        range(
            0,
            len(audio) -
            AASIST_LENGTH + 1,
            AASIST_HOP
        )
    )

    last_start = (
        len(audio) -
        AASIST_LENGTH
    )

    if starts[-1] != last_start:

        starts.append(
            last_start
        )

    scores = []

    for start in starts:

        segment = audio[
            start:
            start + AASIST_LENGTH
        ]

        scores.append(
            aasist_segment(
                segment
            )
        )

    scores = np.asarray(
        scores,
        dtype=np.float32
    )

    avg = np.mean(scores)
    median = np.median(scores)

    strong = np.mean(
        scores >= 0.80
    )

    moderate = np.mean(
        scores >= 0.50
    )

    final = (
        avg * 0.45 +
        median * 0.30 +
        strong * 0.15 +
        moderate * 0.10
    )

    return float(final), scores.tolist()


# ============================================================
# CNN FILE
# ============================================================

@torch.no_grad()
def predict_cnn(path):

    audio, _ = librosa.load(
        path,
        sr=SR,
        mono=True
    )

    if len(audio) == 0:
        return 0.0, []


    peak = np.max(
        np.abs(audio)
    )

    if peak > 1e-8:

        audio = (
            audio / peak
        )


    if len(audio) <= CNN_WINDOW:

        audio = np.pad(
            audio,
            (
                0,
                max(
                    0,
                    CNN_WINDOW -
                    len(audio)
                )
            )
        )

        windows = [
            audio[:CNN_WINDOW]
        ]

    else:

        starts = list(
            range(
                0,
                len(audio) -
                CNN_WINDOW + 1,
                CNN_HOP
            )
        )

        last_start = (
            len(audio) -
            CNN_WINDOW
        )

        if starts[-1] != last_start:

            starts.append(
                last_start
            )

        windows = [
            audio[
                s:s + CNN_WINDOW
            ]
            for s in starts
        ]


    scores = []

    for w in windows:

        lf = torch.from_numpy(
            lfcc(w)
        ).unsqueeze(0).to(
            DEVICE
        )

        me = torch.from_numpy(
            mel(w)
        ).unsqueeze(0).to(
            DEVICE
        )

        output = cnn(
            lf,
            me
        )

        score = torch.softmax(
            output,
            dim=1
        )[0, 1].item()

        scores.append(
            score
        )


    scores = np.asarray(
        scores,
        dtype=np.float32
    )

    final = (
        0.50 * np.mean(scores) +
        0.30 * np.median(scores) +
        0.20 * np.max(scores)
    )

    return float(final), scores.tolist()


# ============================================================
# FUSION
# ============================================================

def fusion_score(aasist_score, cnn_score):

    # Conservative evidence fusion.
    #
    # AASIST3 is primary.
    # CNN is secondary.
    #
    # We intentionally do NOT call this
    # "calibrated" yet.

    return (
        0.70 * aasist_score +
        0.30 * cnn_score
    )


# ============================================================
# DECISION
# ============================================================

def decision(score):

    if score >= 0.65:
        return "SPOOF"

    elif score >= 0.40:
        return "SUSPICIOUS"

    return "REAL"


# ============================================================
# RUN
# ============================================================

print()
print("[3/3] Testing real-world samples...")
print()

for path in TEST_FILES:

    print("=" * 75)

    name = os.path.basename(
        path
    )

    print("FILE:", name)

    if not os.path.exists(path):

        print("STATUS: FILE NOT FOUND")
        print(path)
        continue


    try:

        aasist_score, aasist_segments = (
            predict_aasist(path)
        )

        cnn_score, cnn_segments = (
            predict_cnn(path)
        )

        fused = fusion_score(
            aasist_score,
            cnn_score
        )


        print()
        print(
            f"AASIST3 Score : {aasist_score:.6f}"
        )

        print(
            f"CNN Score     : {cnn_score:.6f}"
        )

        print(
            f"Fused Score   : {fused:.6f}"
        )

        print()
        print(
            f"AASIST3 Segments: "
            f"{len(aasist_segments)}"
        )

        print(
            f"CNN Windows    : "
            f"{len(cnn_segments)}"
        )

        print()
        print(
            "AASIST3 Verdict:",
            decision(aasist_score)
        )

        print(
            "CNN Verdict    :",
            decision(cnn_score)
        )

        print(
            "FUSED Verdict  :",
            decision(fused)
        )

        print()

        print(
            "AASIST3 segment scores:"
        )

        print(
            np.round(
                aasist_segments,
                4
            )
        )

        print()

        print(
            "CNN window scores:"
        )

        print(
            np.round(
                cnn_segments,
                4
            )
        )


    except Exception as e:

        print()
        print(
            "ERROR:",
            repr(e)
        )


print()
print("=" * 75)
print("DIAGNOSTIC COMPLETE")
print("=" * 75)