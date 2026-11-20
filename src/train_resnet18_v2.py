
from pathlib import Path
import copy
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)
import matplotlib.pyplot as plt

from dataset import create_datasets
from resnet18_model import create_resnet18_model


# =========================================================
# 1. CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.0001
NUM_CLASSES = 4
PATIENCE = 6
SEED = 42

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print("=" * 70)
print("RESNET18 V2 - PARTIAL FINE-TUNING")
print("=" * 70)
print(f"Device: {DEVICE}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Maximum epochs: {EPOCHS}")
print(f"Learning rate: {LEARNING_RATE}")


# =========================================================
# 2. MACRO F1
# =========================================================

def calculate_macro_f1(y_true, y_pred):
    _, _, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1, 2, 3],
        average="macro",
        zero_division=0,
    )
    return float(macro_f1)


# =========================================================
# 3. EVALUATION
# =========================================================

def evaluate_model(model, loader, criterion):
    model.eval()

    total_loss = 0.0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

    y_true = np.asarray(all_labels)
    y_pred = np.asarray(all_predictions)

    average_loss = total_loss / len(loader.dataset)
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = calculate_macro_f1(y_true, y_pred)

    return average_loss, accuracy, macro_f1, y_true, y_pred


# =========================================================
# 4. LOAD DATASETS
# =========================================================

train_dataset, validation_dataset, test_dataset = (
    create_datasets()
)

print("\nDataset sizes:")
print("Training:", len(train_dataset))
print("Validation:", len(validation_dataset))
print("Test:", len(test_dataset))

train_labels = np.asarray(
    train_dataset.dataframe["label"].values,
    dtype=int,
)

class_counts = np.bincount(
    train_labels,
    minlength=NUM_CLASSES,
)

print("\nTraining class counts:")
for class_id, count in enumerate(class_counts):
    print(f"Class {class_id}: {count}")

if np.any(class_counts == 0):
    raise ValueError(
        "At least one class is missing from the training dataset."
    )


# =========================================================
# 5. BALANCED SAMPLING
# =========================================================

class_weights = 1.0 / class_counts

sample_weights = class_weights[train_labels]

sampler = WeightedRandomSampler(
    weights=torch.as_tensor(
        sample_weights, dtype=torch.double
    ),
    num_samples=len(sample_weights),
    replacement=True,
    generator=torch.Generator().manual_seed(SEED),
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=sampler,
    num_workers=0,
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)


# =========================================================
# 6. CREATE MODEL AND UNFREEZE LAYER4
# =========================================================

model = create_resnet18_model(
    num_classes=NUM_CLASSES,
    pretrained=True,
)

# The model factory freezes the backbone.
# Unfreeze only layer4 and the final classifier.
for parameter in model.layer4.parameters():
    parameter.requires_grad = True

for parameter in model.fc.parameters():
    parameter.requires_grad = True

model = model.to(DEVICE)

trainable_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)

total_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
)

print("\nModel configuration:")
print("Trainable layers: layer4 and fc")
print("Trainable parameters:", trainable_parameters)
print("Total parameters:", total_parameters)


# =========================================================
# 7. LOSS AND OPTIMIZER
# =========================================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    filter(
        lambda parameter: parameter.requires_grad,
        model.parameters(),
    ),
    lr=LEARNING_RATE,
    weight_decay=1e-4,
)

# Reduce the learning rate if validation performance plateaus.
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=2,
)


# =========================================================
# 8. TRAINING LOOP
# =========================================================

history = []

best_macro_f1 = -1.0
best_epoch = 0
epochs_without_improvement = 0

checkpoint_path = MODEL_DIR / "best_resnet18_v2_model.pth"

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
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    train_loss = running_loss / len(train_dataset)
    train_accuracy = correct / total

    (
        val_loss,
        val_accuracy,
        val_macro_f1,
        _,
        _,
    ) = evaluate_model(
        model,
        validation_loader,
        criterion,
    )

    scheduler.step(val_macro_f1)

    current_lr = optimizer.param_groups[0]["lr"]

    history.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "validation_loss": val_loss,
        "validation_accuracy": val_accuracy,
        "validation_macro_f1": val_macro_f1,
        "learning_rate": current_lr,
    })

    print(f"\nEpoch {epoch}/{EPOCHS}")
    print(
        f"Train Loss: {train_loss:.4f} | "
        f"Train Accuracy: {train_accuracy:.4f}"
    )
    print(
        f"Validation Loss: {val_loss:.4f} | "
        f"Validation Accuracy: {val_accuracy:.4f} | "
        f"Validation Macro F1: {val_macro_f1:.4f}"
    )
    print(f"Learning rate: {current_lr:.7f}")

    if val_macro_f1 > best_macro_f1:
        best_macro_f1 = val_macro_f1
        best_epoch = epoch
        epochs_without_improvement = 0

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": copy.deepcopy(
                    model.state_dict()
                ),
                "validation_macro_f1": val_macro_f1,
                "validation_accuracy": val_accuracy,
                "validation_loss": val_loss,
            },
            checkpoint_path,
        )

        print("Best V2 checkpoint saved.")
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= PATIENCE:
        print("\nEarly stopping triggered.")
        break


# =========================================================
# 9. SAVE TRAINING HISTORY AND CURVES
# =========================================================

history_df = pd.DataFrame(history)

history_path = RESULTS_DIR / "resnet18_v2_training_history.csv"
history_df.to_csv(history_path, index=False)

for metric, title, filename in [
    ("train_loss", "Training Loss", "resnet18_v2_loss.png"),
    ("validation_loss", "Validation Loss", "resnet18_v2_val_loss.png"),
    (
        "validation_accuracy",
        "Validation Accuracy",
        "resnet18_v2_accuracy.png",
    ),
    (
        "validation_macro_f1",
        "Validation Macro F1",
        "resnet18_v2_macro_f1.png",
    ),
]:
    plt.figure(figsize=(9, 5))
    plt.plot(
        history_df["epoch"],
        history_df[metric],
        marker="o",
    )
    plt.xlabel("Epoch")
    plt.ylabel(metric.replace("_", " ").title())
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        RESULTS_DIR / filename,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()


# =========================================================
# 10. RESTORE BEST CHECKPOINT
# =========================================================

checkpoint = torch.load(
    checkpoint_path,
    map_location=DEVICE,
    weights_only=True,
)

model.load_state_dict(checkpoint["model_state_dict"])


# =========================================================
# 11. FINAL TEST EVALUATION
# =========================================================

# Use the test set only after model selection is complete.
(
    test_loss,
    test_accuracy,
    test_macro_f1,
    y_true,
    y_pred,
) = evaluate_model(
    model,
    test_loader,
    criterion,
)

precision, recall, f1, support = (
    precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1, 2, 3],
        zero_division=0,
    )
)

print("\n" + "=" * 70)
print("RESNET18 V2 TEST RESULTS")
print("=" * 70)
print("Best epoch:", best_epoch)
print(f"Best validation Macro F1: {best_macro_f1:.4f}")
print(f"Test loss: {test_loss:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")
print(f"Test Macro F1: {test_macro_f1:.4f}")

class_names = [
    "Class 0 (0.00)",
    "Class 1 (0.33)",
    "Class 2 (0.66)",
    "Class 3 (1.00)",
]

report = classification_report(
    y_true,
    y_pred,
    labels=[0, 1, 2, 3],
    target_names=class_names,
    output_dict=True,
    zero_division=0,
)

report_path = RESULTS_DIR / "resnet18_v2_classification_report.csv"
pd.DataFrame(report).transpose().to_csv(report_path)

matrix = confusion_matrix(
    y_true,
    y_pred,
    labels=[0, 1, 2, 3],
)

matrix_path = RESULTS_DIR / "resnet18_v2_confusion_matrix.csv"

pd.DataFrame(
    matrix,
    index=class_names,
    columns=class_names,
).to_csv(matrix_path)

metrics_path = RESULTS_DIR / "resnet18_v2_test_metrics.csv"

pd.DataFrame([{
    "Model": "ResNet18 V2 Partial Fine-Tuning",
    "Accuracy": test_accuracy,
    "Macro Precision": float(np.mean(precision)),
    "Macro Recall": float(np.mean(recall)),
    "Macro F1": test_macro_f1,
    "Test Loss": test_loss,
}]).to_csv(metrics_path, index=False)

print("\nClassification report:")
print(classification_report(
    y_true,
    y_pred,
    labels=[0, 1, 2, 3],
    target_names=class_names,
    zero_division=0,
))

print("\nConfusion matrix:")
print(matrix)

print("\nSaved checkpoint:", checkpoint_path)
print("Saved history:", history_path)
print("Saved classification report:", report_path)
print("Saved confusion matrix:", matrix_path)
print("Saved metrics:", metrics_path)
print("\nRESNET18 V2 EXPERIMENT COMPLETED")
