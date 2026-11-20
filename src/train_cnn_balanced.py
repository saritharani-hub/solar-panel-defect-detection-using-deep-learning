from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, WeightedRandomSampler

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
print("BALANCED CNN TRAINING - SOLAR PANEL DEFECT DETECTION")
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
# Calculate sampling weights
# --------------------------------------------------

print("\n" + "=" * 70)
print("CREATING BALANCED TRAINING SAMPLER")
print("=" * 70)

train_labels = train_dataset.dataframe["label"].values

class_counts = np.bincount(
    train_labels,
    minlength=NUM_CLASSES
)

print("\nOriginal class counts:")

for class_id, count in enumerate(class_counts):
    print(f"Class {class_id}: {count}")


# Inverse-frequency class weights
class_weights = 1.0 / class_counts

print("\nSampling class weights:")

for class_id, weight in enumerate(class_weights):
    print(f"Class {class_id}: {weight:.6f}")


# Assign each training sample the weight
# of its corresponding class
sample_weights = np.array([
    class_weights[label]
    for label in train_labels
])


sample_weights = torch.as_tensor(
    sample_weights,
    dtype=torch.double
)


sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)


# --------------------------------------------------
# DataLoaders
# --------------------------------------------------

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=sampler,
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


print("\nBalanced sampler created successfully.")


# --------------------------------------------------
# Create CNN
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
# We use normal CrossEntropyLoss because the
# WeightedRandomSampler already handles imbalance.

criterion = nn.CrossEntropyLoss()


# --------------------------------------------------
# Optimizer
# --------------------------------------------------

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# --------------------------------------------------
# Helper function:
# calculate macro F1 without sklearn
# --------------------------------------------------

def calculate_macro_f1(true_labels, predicted_labels):

    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)

    f1_scores = []

    for class_id in range(NUM_CLASSES):

        true_positive = np.sum(
            (true_labels == class_id) &
            (predicted_labels == class_id)
        )

        false_positive = np.sum(
            (true_labels != class_id) &
            (predicted_labels == class_id)
        )

        false_negative = np.sum(
            (true_labels == class_id) &
            (predicted_labels != class_id)
        )

        if (
            true_positive +
            false_positive
        ) > 0:

            precision = (
                true_positive /
                (
                    true_positive +
                    false_positive
                )
            )

        else:

            precision = 0.0


        if (
            true_positive +
            false_negative
        ) > 0:

            recall = (
                true_positive /
                (
                    true_positive +
                    false_negative
                )
            )

        else:

            recall = 0.0


        if precision + recall > 0:

            f1 = (
                2 *
                precision *
                recall /
                (precision + recall)
            )

        else:

            f1 = 0.0


        f1_scores.append(f1)


    return np.mean(f1_scores)


# --------------------------------------------------
# Training history
# --------------------------------------------------

history = {
    "epoch": [],
    "train_loss": [],
    "train_accuracy": [],
    "validation_loss": [],
    "validation_accuracy": [],
    "validation_macro_f1": []
}


best_macro_f1 = -1.0
epochs_without_improvement = 0


# --------------------------------------------------
# Training
# --------------------------------------------------

print("\n" + "=" * 70)
print("STARTING BALANCED CNN TRAINING")
print("=" * 70)


for epoch in range(NUM_EPOCHS):

    # --------------------------------------------------
    # Training
    # --------------------------------------------------

    model.train()

    running_train_loss = 0.0
    correct_train = 0
    total_train = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_train_loss += (
            loss.item() *
            images.size(0)
        )

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        correct_train += (
            predictions == labels
        ).sum().item()

        total_train += labels.size(0)


    train_loss = (
        running_train_loss /
        total_train
    )

    train_accuracy = (
        correct_train /
        total_train
    )


    # --------------------------------------------------
    # Validation
    # --------------------------------------------------

    model.eval()

    running_validation_loss = 0.0
    correct_validation = 0
    total_validation = 0

    validation_labels = []
    validation_predictions = []


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
                loss.item() *
                images.size(0)
            )

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            correct_validation += (
                predictions == labels
            ).sum().item()

            total_validation += labels.size(0)

            validation_labels.extend(
                labels.cpu().numpy()
            )

            validation_predictions.extend(
                predictions.cpu().numpy()
            )


    validation_loss = (
        running_validation_loss /
        total_validation
    )

    validation_accuracy = (
        correct_validation /
        total_validation
    )


    validation_macro_f1 = calculate_macro_f1(
        validation_labels,
        validation_predictions
    )


    # --------------------------------------------------
    # Store history
    # --------------------------------------------------

    history["epoch"].append(
        epoch + 1
    )

    history["train_loss"].append(
        train_loss
    )

    history["train_accuracy"].append(
        train_accuracy
    )

    history["validation_loss"].append(
        validation_loss
    )

    history["validation_accuracy"].append(
        validation_accuracy
    )

    history["validation_macro_f1"].append(
        validation_macro_f1
    )


    # --------------------------------------------------
    # Print results
    # --------------------------------------------------

    print(
        f"\nEpoch [{epoch + 1}/{NUM_EPOCHS}]"
    )

    print(
        f"Train Loss: {train_loss:.4f} | "
        f"Train Accuracy: {train_accuracy:.4f}"
    )

    print(
        f"Validation Loss: "
        f"{validation_loss:.4f} | "
        f"Validation Accuracy: "
        f"{validation_accuracy:.4f}"
    )

    print(
        f"Validation Macro F1: "
        f"{validation_macro_f1:.4f}"
    )


    # --------------------------------------------------
    # Save best model based on Macro F1
    # --------------------------------------------------

    if validation_macro_f1 > best_macro_f1:

        best_macro_f1 = validation_macro_f1

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "num_classes": NUM_CLASSES,
                "epoch": epoch + 1,
                "validation_loss": validation_loss,
                "validation_accuracy": validation_accuracy,
                "validation_macro_f1": validation_macro_f1
            },
            MODEL_DIR / "best_cnn_balanced_model.pth"
        )

        print("✓ Best balanced model saved.")

        epochs_without_improvement = 0

    else:

        epochs_without_improvement += 1

        print(
            f"No Macro F1 improvement for "
            f"{epochs_without_improvement} epoch(s)."
        )


    # --------------------------------------------------
    # Early stopping
    # --------------------------------------------------

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
    "cnn_balanced_training_history.csv"
)

history_df.to_csv(
    history_file,
    index=False
)


# --------------------------------------------------
# Plot loss
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

plt.title(
    "Balanced CNN Training and Validation Loss"
)

plt.legend()
plt.grid(True)

loss_plot = (
    RESULTS_DIR /
    "cnn_balanced_loss_curve.png"
)

plt.savefig(
    loss_plot,
    dpi=150,
    bbox_inches="tight"
)

plt.close()


# --------------------------------------------------
# Plot accuracy
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

plt.title(
    "Balanced CNN Training and Validation Accuracy"
)

plt.legend()
plt.grid(True)

accuracy_plot = (
    RESULTS_DIR /
    "cnn_balanced_accuracy_curve.png"
)

plt.savefig(
    accuracy_plot,
    dpi=150,
    bbox_inches="tight"
)

plt.close()


# --------------------------------------------------
# Plot validation Macro F1
# --------------------------------------------------

plt.figure(figsize=(10, 5))

plt.plot(
    history_df["epoch"],
    history_df["validation_macro_f1"],
    marker="o",
    label="Validation Macro F1"
)

plt.xlabel("Epoch")
plt.ylabel("Macro F1")

plt.title(
    "Balanced CNN Validation Macro F1"
)

plt.legend()
plt.grid(True)

f1_plot = (
    RESULTS_DIR /
    "cnn_balanced_macro_f1_curve.png"
)

plt.savefig(
    f1_plot,
    dpi=150,
    bbox_inches="tight"
)

plt.close()


# --------------------------------------------------
# Final summary
# --------------------------------------------------

print("\n" + "=" * 70)
print("BALANCED CNN TRAINING COMPLETED")
print("=" * 70)

print(
    "\nBest validation Macro F1:"
)

print(
    f"{best_macro_f1:.4f}"
)

print("\nSaved balanced model:")

print(
    MODEL_DIR /
    "best_cnn_balanced_model.pth"
)

print("\nSaved training history:")

print(history_file)

print("\nSaved loss curve:")

print(loss_plot)

print("\nSaved accuracy curve:")

print(accuracy_plot)

print("\nSaved Macro F1 curve:")

print(f1_plot)

print("\n" + "=" * 70)