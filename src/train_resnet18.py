from pathlib import Path
import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
import matplotlib.pyplot as plt

from dataset import create_datasets
from resnet18_model import create_resnet18_model


# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


# --------------------------------------------------
# Configuration
# --------------------------------------------------

BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 0.001
NUM_CLASSES = 4
PATIENCE = 5

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# --------------------------------------------------
# Macro F1
# --------------------------------------------------

def calculate_macro_f1(y_true, y_pred, num_classes=4):

    f1_scores = []

    for class_id in range(num_classes):

        tp = np.sum(
            (y_true == class_id) &
            (y_pred == class_id)
        )

        fp = np.sum(
            (y_true != class_id) &
            (y_pred == class_id)
        )

        fn = np.sum(
            (y_true == class_id) &
            (y_pred != class_id)
        )

        precision = (
            tp / (tp + fp)
            if (tp + fp) > 0
            else 0.0
        )

        recall = (
            tp / (tp + fn)
            if (tp + fn) > 0
            else 0.0
        )

        if precision + recall > 0:
            f1 = (
                2 * precision * recall
                / (precision + recall)
            )
        else:
            f1 = 0.0

        f1_scores.append(f1)

    return float(np.mean(f1_scores))


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

def evaluate_model(model, dataloader, criterion):

    model.eval()

    total_loss = 0.0
    all_labels = []
    all_predictions = []

    with torch.no_grad():

        for images, labels in dataloader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(outputs, labels)

            total_loss += (
                loss.item() * images.size(0)
            )

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

    avg_loss = (
        total_loss / len(dataloader.dataset)
    )

    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)

    accuracy = np.mean(
        all_labels == all_predictions
    )

    macro_f1 = calculate_macro_f1(
        all_labels,
        all_predictions,
        NUM_CLASSES
    )

    return avg_loss, accuracy, macro_f1


# --------------------------------------------------
# Main
# --------------------------------------------------

print("=" * 70)
print("RESNET18 TRANSFER LEARNING - SOLAR PANEL DEFECT DETECTION")
print("=" * 70)

print(f"Device: {DEVICE}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Number of epochs: {EPOCHS}")
print(f"Learning rate: {LEARNING_RATE}")


# --------------------------------------------------
# Load datasets
# --------------------------------------------------

train_dataset, validation_dataset, test_dataset = (
    create_datasets()
)

print("\nDataset sizes:")
print(f"Training:   {len(train_dataset)}")
print(f"Validation: {len(validation_dataset)}")
print(f"Test:       {len(test_dataset)}")


# --------------------------------------------------
# Create balanced sampler
# --------------------------------------------------

train_labels = train_dataset.dataframe["label"].values

class_counts = np.bincount(
    train_labels,
    minlength=NUM_CLASSES
)

class_weights = 1.0 / class_counts

sample_weights = np.array([
    class_weights[label]
    for label in train_labels
])

sample_weights = torch.DoubleTensor(
    sample_weights
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


# --------------------------------------------------
# Model
# --------------------------------------------------

model = create_resnet18_model(
    num_classes=NUM_CLASSES,
    pretrained=True
)

model = model.to(DEVICE)


# --------------------------------------------------
# Loss and optimizer
# --------------------------------------------------

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    filter(
        lambda parameter: parameter.requires_grad,
        model.parameters()
    ),
    lr=LEARNING_RATE
)


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
best_epoch = 0
best_model_state = None

epochs_without_improvement = 0


# --------------------------------------------------
# Training loop
# --------------------------------------------------

for epoch in range(1, EPOCHS + 1):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()
        optimizer.step()

        running_loss += (
            loss.item() * images.size(0)
        )

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        correct += (
            (predictions == labels)
            .sum()
            .item()
        )

        total += labels.size(0)

    train_loss = (
        running_loss /
        len(train_loader.dataset)
    )

    train_accuracy = correct / total


    # Validation

    validation_loss, validation_accuracy, validation_macro_f1 = (
        evaluate_model(
            model,
            validation_loader,
            criterion
        )
    )


    # Save history

    history["epoch"].append(epoch)
    history["train_loss"].append(train_loss)
    history["train_accuracy"].append(train_accuracy)
    history["validation_loss"].append(validation_loss)
    history["validation_accuracy"].append(validation_accuracy)
    history["validation_macro_f1"].append(validation_macro_f1)


    print(
        f"\nEpoch {epoch}/{EPOCHS}"
    )

    print(
        f"Train Loss: {train_loss:.4f} | "
        f"Train Accuracy: {train_accuracy:.4f}"
    )

    print(
        f"Val Loss: {validation_loss:.4f} | "
        f"Val Accuracy: {validation_accuracy:.4f} | "
        f"Val Macro F1: {validation_macro_f1:.4f}"
    )


    # --------------------------------------------------
    # Save best model based on Macro F1
    # --------------------------------------------------

    if validation_macro_f1 > best_macro_f1:

        best_macro_f1 = validation_macro_f1
        best_epoch = epoch

        best_model_state = copy.deepcopy(
            model.state_dict()
        )

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "validation_macro_f1": validation_macro_f1,
                "validation_accuracy": validation_accuracy,
                "validation_loss": validation_loss
            },
            MODEL_DIR / "best_resnet18_model.pth"
        )

        epochs_without_improvement = 0

        print(
            f"✓ Best model saved "
            f"(Macro F1: {best_macro_f1:.4f})"
        )

    else:

        epochs_without_improvement += 1


    # Early stopping

    if epochs_without_improvement >= PATIENCE:

        print(
            f"\nEarly stopping at epoch {epoch}."
        )

        break


# --------------------------------------------------
# Save training history
# --------------------------------------------------

history_df = pd.DataFrame(history)

history_file = (
    RESULTS_DIR /
    "resnet18_training_history.csv"
)

history_df.to_csv(
    history_file,
    index=False
)


# --------------------------------------------------
# Restore best model
# --------------------------------------------------

if best_model_state is not None:

    model.load_state_dict(
        best_model_state
    )


# --------------------------------------------------
# Plot loss
# --------------------------------------------------

plt.figure(figsize=(10, 6))

plt.plot(
    history["epoch"],
    history["train_loss"],
    label="Training Loss"
)

plt.plot(
    history["epoch"],
    history["validation_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title(
    "ResNet18 Training and Validation Loss"
)

plt.legend()
plt.grid(True)

plt.savefig(
    RESULTS_DIR /
    "resnet18_loss_curve.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close()


# --------------------------------------------------
# Plot accuracy
# --------------------------------------------------

plt.figure(figsize=(10, 6))

plt.plot(
    history["epoch"],
    history["train_accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history["epoch"],
    history["validation_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title(
    "ResNet18 Training and Validation Accuracy"
)

plt.legend()
plt.grid(True)

plt.savefig(
    RESULTS_DIR /
    "resnet18_accuracy_curve.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close()


# --------------------------------------------------
# Plot Macro F1
# --------------------------------------------------

plt.figure(figsize=(10, 6))

plt.plot(
    history["epoch"],
    history["validation_macro_f1"],
    marker="o",
    label="Validation Macro F1"
)

plt.xlabel("Epoch")
plt.ylabel("Macro F1")
plt.title(
    "ResNet18 Validation Macro F1"
)

plt.legend()
plt.grid(True)

plt.savefig(
    RESULTS_DIR /
    "resnet18_macro_f1_curve.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close()


# --------------------------------------------------
# Summary
# --------------------------------------------------

print("\n" + "=" * 70)
print("RESNET18 TRAINING COMPLETED")
print("=" * 70)

print(
    f"Best epoch: {best_epoch}"
)

print(
    f"Best validation Macro F1: "
    f"{best_macro_f1:.4f}"
)

print(
    f"\nSaved model:"
)

print(
    MODEL_DIR /
    "best_resnet18_model.pth"
)

print(
    "\nSaved training history:"
)

print(history_file)

print(
    "\nSaved curves:"
)

print(
    RESULTS_DIR /
    "resnet18_loss_curve.png"
)

print(
    RESULTS_DIR /
    "resnet18_accuracy_curve.png"
)

print(
    RESULTS_DIR /
    "resnet18_macro_f1_curve.png"
)

print("=" * 70)