import torch
import torch.nn as nn


# ============================================================
# SWARAKSHA - Custom CNN
# Voice Deepfake Detection
# ============================================================


class SwarakshaCNN(nn.Module):
    """
    Lightweight CNN for classifying speech as:

        0 -> BONAFIDE / REAL
        1 -> SPOOF / AI

    Input:
        [batch, 1, 128, time_frames]
    """

    def __init__(self, num_classes=2):
        super().__init__()

        # ----------------------------------------------------
        # Convolution Block 1
        # ----------------------------------------------------
        self.block1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
            ),
        )

        # ----------------------------------------------------
        # Convolution Block 2
        # ----------------------------------------------------
        self.block2 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
            ),
        )

        # ----------------------------------------------------
        # Convolution Block 3
        # ----------------------------------------------------
        self.block3 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
            ),
        )

        # ----------------------------------------------------
        # Convolution Block 4
        # ----------------------------------------------------
        self.block4 = nn.Sequential(
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
            ),
        )

        # ----------------------------------------------------
        # Global feature aggregation
        # ----------------------------------------------------
        self.global_pool = nn.AdaptiveAvgPool2d(
            output_size=(1, 1)
        )

        # ----------------------------------------------------
        # Classifier
        # ----------------------------------------------------
        self.classifier = nn.Sequential(
            nn.Flatten(),

            nn.Linear(
                256,
                128,
            ),

            nn.ReLU(inplace=True),

            nn.Dropout(
                p=0.40
            ),

            nn.Linear(
                128,
                num_classes,
            ),
        )

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    def forward(self, x):

        x = self.block1(x)

        x = self.block2(x)

        x = self.block3(x)

        x = self.block4(x)

        x = self.global_pool(x)

        x = self.classifier(x)

        return x


# ============================================================
# Model information
# ============================================================

def count_parameters(model):

    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


# ============================================================
# Test model
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("SWARAKSHA - CUSTOM CNN TEST")
    print("=" * 60)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print("Device:")
    print(device)

    if device.type == "cuda":

        print()
        print("GPU:")
        print(torch.cuda.get_device_name(0))

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    model = SwarakshaCNN(
        num_classes=2
    ).to(device)

    print()
    print("Trainable parameters:")

    print(
        f"{count_parameters(model):,}"
    )

    # --------------------------------------------------------
    # Dummy Mel-spectrogram
    # --------------------------------------------------------

    dummy_input = torch.randn(
        4,
        1,
        128,
        251,
        device=device,
    )

    print()
    print("Input shape:")
    print(dummy_input.shape)

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    with torch.no_grad():

        output = model(
            dummy_input
        )

    print()
    print("Output shape:")
    print(output.shape)

    print()
    print("Output:")
    print(output)

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    probabilities = torch.softmax(
        output,
        dim=1,
    )

    predictions = torch.argmax(
        probabilities,
        dim=1,
    )

    print()
    print("Probabilities:")
    print(probabilities)

    print()
    print("Predicted classes:")
    print(predictions)

    # --------------------------------------------------------
    # Final check
    # --------------------------------------------------------

    expected_shape = (
        4,
        2,
    )

    if tuple(output.shape) == expected_shape:

        print()
        print("=" * 60)
        print("CUSTOM CNN TEST PASSED")
        print("=" * 60)

    else:

        print()
        print("ERROR: Unexpected output shape.")