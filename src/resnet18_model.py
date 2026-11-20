import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


def create_resnet18_model(num_classes=4, pretrained=True):
    """
    Create a ResNet18 transfer-learning model
    for grayscale ELPV solar panel images.
    """

    if pretrained:
        weights = ResNet18_Weights.DEFAULT
        model = resnet18(weights=weights)
    else:
        model = resnet18(weights=None)

    # Change the first convolution from 3-channel RGB
    # to 1-channel grayscale.
    original_conv = model.conv1

    model.conv1 = nn.Conv2d(
        in_channels=1,
        out_channels=original_conv.out_channels,
        kernel_size=original_conv.kernel_size,
        stride=original_conv.stride,
        padding=original_conv.padding,
        bias=False
    )

    # Initialize the grayscale convolution using
    # the average of the pretrained RGB filters.
    with torch.no_grad():
        model.conv1.weight.copy_(
            original_conv.weight.mean(dim=1, keepdim=True)
        )

    # Replace the ImageNet classifier with our 4-class classifier.
    num_features = model.fc.in_features

    model.fc = nn.Linear(
        num_features,
        num_classes
    )

    # Freeze the pretrained backbone.
    for param in model.parameters():
        param.requires_grad = False

    # Keep the new classifier trainable.
    for param in model.fc.parameters():
        param.requires_grad = True

    return model


if __name__ == "__main__":
    model = create_resnet18_model(
        num_classes=4,
        pretrained=True
    )

    print("=" * 60)
    print("RESNET18 TRANSFER LEARNING MODEL")
    print("=" * 60)

    print("Input channels:", model.conv1.in_channels)
    print("Output classes:", model.fc.out_features)

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    print("Total parameters:", total_params)
    print("Trainable parameters:", trainable_params)

    # Test grayscale input
    test_input = torch.randn(4, 1, 224, 224)

    with torch.no_grad():
        output = model(test_input)

    print("Test input shape:", test_input.shape)
    print("Output shape:", output.shape)

    print("=" * 60)