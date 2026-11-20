import torch
import torch.nn as nn


class SolarPanelCNN(nn.Module):

    def __init__(self, num_classes=4):
        super(SolarPanelCNN, self).__init__()

        # Convolutional Block 1
        self.features = nn.Sequential(

            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # Convolutional Block 2
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # Convolutional Block 3
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # Convolutional Block 4
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        # Adaptive pooling allows the classifier to receive
        # a fixed-size feature representation.
        self.pool = nn.AdaptiveAvgPool2d((4, 4))

        # Fully connected classifier
        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, num_classes)
        )

    def forward(self, x):

        x = self.features(x)

        x = self.pool(x)

        x = self.classifier(x)

        return x


if __name__ == "__main__":

    print("=" * 60)
    print("SOLAR PANEL CUSTOM CNN MODEL TEST")
    print("=" * 60)

    # Select CPU for the current environment
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\nDevice:", device)

    # Create the model
    model = SolarPanelCNN(num_classes=4)
    model = model.to(device)

    print("\nModel:")
    print(model)

    # Create a dummy batch:
    # 4 images, 1 grayscale channel, 224 x 224
    dummy_input = torch.randn(4, 1, 224, 224).to(device)

    # Forward pass
    output = model(dummy_input)

    print("\nInput shape:")
    print(dummy_input.shape)

    print("\nOutput shape:")
    print(output.shape)

    print("\nExpected output:")
    print("(batch_size, number_of_classes)")

    print("\nNumber of trainable parameters:")

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(f"{trainable_parameters:,}")

    print("\n" + "=" * 60)
    print("CNN MODEL TEST COMPLETED SUCCESSFULLY")
    print("=" * 60)