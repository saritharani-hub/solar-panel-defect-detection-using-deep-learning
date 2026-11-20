from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from dataset import create_datasets
from cnn_model import SolarPanelCNN


# --------------------------------------------------
# Project paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Configuration
# --------------------------------------------------

BATCH_SIZE = 32
NUM_CLASSES = 4
NUM_EPOCHS = 15
LEARNING_RATE = 0.001
PATIENCE = 5

RANDOM_SEED = 42

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# --------------------------------------------------
# Device
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("SOLAR PANEL DEFECT DETECTION - CNN TRAINING")
print("=" * 70)

print("\nDevice:", device)
print("Batch size:", BATCH_SIZE)
print("Number of epochs:", NUM_EPOCHS)
print("Learning rate:", LEARNING_RATE)


# --------------------------------------------------
# Load datasets
# --------------------------------------------------

print("\n" + "=" * 70)
print("LOADING DATASETS")
print("=" * 70)

train_dataset, validation_dataset, test_dataset = create_datasets()

print("\nTraining samples:", len(train_dataset))
print("Validation samples:", len(validation_dataset))
print("Test samples:", len(test_dataset))


# --------------------------------------------------
# DataLoaders
# --------------------------------------------------

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# --------------------------------------------------
# Calculate class weights
# --------------------------------------------------

print("\n" + "=" * 70)
print("CALCULATING CLASS WEIGHTS")
print("=" * 70)

train_labels = train_dataset.dataframe["label"].values

class_counts = np.bincount(
    train_labels,
    minlength=NUM_CLASSES
)

print("\nClass counts:")

for class_id, count in enumerate(class_counts):
    print(f"Class {class_id}: {count}")


# Inverse-frequency weighting
class_weights = len(train_labels) / (
    NUM_CLASSES * class_counts
)

print("\nClass weights:")

for class_id, weight in enumerate(class_weights):
    print(f"Class {class_id}: {weight:.4f}")


class_weights_tensor = torch.tensor(
    class_weights,
    dtype=torch.float32
).to(device)


# --------------------------------------------------
# Create CNN model
# --------------------------------------------------

print("\n" + "=" * 70)
print("CREATING CNN MODEL")
print("=" * 70)

model = SolarPanelCNN(
    num_classes=NUM_CLASSES
).to(device)

print("\nModel created successfully.")


# --------------------------------------------------
# Loss function
# --------------------------------------------------

criterion = nn.CrossEntropyLoss(
    weight=class_weights_tensor
)


# --------------------------------------------------
# Optimizer
# --------------------------------------------------

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# --------------------------------------------------
# Training history
# --------------------------------------------------

history = {
    "epoch": [],
    "train_loss": [],
    "validation_loss": [],
    "train_accuracy": [],
    "validation_accuracy": []
}


best_validation_loss = float("inf")
epochs_without_improvement = 0


# --------------------------------------------------
# Training loop
# --------------------------------------------------

print("\n" + "=" * 70)
print("STARTING CNN TRAINING")
print("=" * 70)


for epoch in range(NUM_EPOCHS):

    # ----------------------------------------------
    # Training
    # ----------------------------------------------

    model.train()

    running_train_loss = 0.0
    correct_train = 0
    total_train = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        # Clear previous gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Calculate loss
        loss = criterion(outputs, labels)

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        # Statistics
        running_train_loss += loss.item() * images.size(0)

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        correct_train += (
            predictions == labels
        ).sum().item()

        total_train += labels.size(0)


    train_loss = running_train_loss / total_train
    train_accuracy = correct_train / total_train


    # ----------------------------------------------
    # Validation
    # ----------------------------------------------

    model.eval()

    running_validation_loss = 0.0
    correct_validation = 0
    total_validation = 0

    with torch.no_grad():

        for images, labels in validation_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            running_validation_loss += (
                loss.item() * images.size(0)
            )

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            correct_validation += (
                predictions == labels
            ).sum().item()

            total_validation += labels.size(0)


    validation_loss = (
        running_validation_loss /
        total_validation
    )

    validation_accuracy = (
        correct_validation /
        total_validation
    )


    # ----------------------------------------------
    # Store history
    # ----------------------------------------------

    history["epoch"].append(epoch + 1)
    history["train_loss"].append(train_loss)
    history["validation_loss"].append(validation_loss)
    history["train_accuracy"].append(train_accuracy)
    history["validation_accuracy"].append(
        validation_accuracy
    )


    # ----------------------------------------------
    # Print epoch results
    # ----------------------------------------------

    print(
        f"\nEpoch [{epoch + 1}/{NUM_EPOCHS}]"
    )

    print(
        f"Train Loss: {train_loss:.4f} | "
        f"Train Accuracy: {train_accuracy:.4f}"
    )

    print(
        f"Validation Loss: {validation_loss:.4f} | "
        f"Validation Accuracy: "
        f"{validation_accuracy:.4f}"
    )


    # ----------------------------------------------
    # Save best model
    # ----------------------------------------------

    if validation_loss < best_validation_loss:

        best_validation_loss = validation_loss

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "num_classes": NUM_CLASSES,
                "class_weights": class_weights.tolist(),
                "epoch": epoch + 1,
                "validation_loss": validation_loss
            },
            MODEL_DIR / "best_cnn_model.pth"
        )

        print("✓ Best model saved.")

        epochs_without_improvement = 0

    else:

        epochs_without_improvement += 1

        print(
            f"No improvement for "
            f"{epochs_without_improvement} epoch(s)."
        )


    # ----------------------------------------------
    # Early stopping
    # ----------------------------------------------

    if epochs_without_improvement >= PATIENCE:

        print(
            "\nEarly stopping triggered."
        )

        break


# --------------------------------------------------
# Save training history
# --------------------------------------------------

history_df = pd.DataFrame(history)

history_file = (
    RESULTS_DIR /
    "cnn_training_history.csv"
)

history_df.to_csv(
    history_file,
    index=False
)


# --------------------------------------------------
# Plot training history
# --------------------------------------------------

plt.figure(figsize=(10, 5))

plt.plot(
    history_df["epoch"],
    history_df["train_loss"],
    label="Training Loss"
)

plt.plot(
    history_df["epoch"],
    history_df["validation_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("CNN Training and Validation Loss")
plt.legend()
plt.grid(True)

loss_plot = (
    RESULTS_DIR /
    "cnn_loss_curve.png"
)

plt.savefig(
    loss_plot,
    dpi=150,
    bbox_inches="tight"
)

plt.close()


# --------------------------------------------------
# Accuracy plot
# --------------------------------------------------

plt.figure(figsize=(10, 5))

plt.plot(
    history_df["epoch"],
    history_df["train_accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history_df["epoch"],
    history_df["validation_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("CNN Training and Validation Accuracy")
plt.legend()
plt.grid(True)

accuracy_plot = (
    RESULTS_DIR /
    "cnn_accuracy_curve.png"
)

plt.savefig(
    accuracy_plot,
    dpi=150,
    bbox_inches="tight"
)

plt.close()


# --------------------------------------------------
# Final summary
# --------------------------------------------------

print("\n" + "=" * 70)
print("CNN TRAINING COMPLETED")
print("=" * 70)

print("\nBest validation loss:")
print(f"{best_validation_loss:.4f}")

print("\nSaved model:")
print(
    MODEL_DIR / "best_cnn_model.pth"
)

print("\nSaved training history:")
print(history_file)

print("\nSaved loss curve:")
print(loss_plot)

print("\nSaved accuracy curve:")
print(accuracy_plot)

print("\n" + "=" * 70)