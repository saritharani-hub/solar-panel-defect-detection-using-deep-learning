from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from dataset import create_datasets
from cnn_model import SolarPanelCNN


# --------------------------------------------------
# Project paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_FILE = (
    BASE_DIR /
    "models" /
    "best_cnn_balanced_model.pth"
)

RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# --------------------------------------------------
# Configuration
# --------------------------------------------------

BATCH_SIZE = 32
NUM_CLASSES = 4

CLASS_NAMES = [
    "Class 0 (0.0)",
    "Class 1 (0.33)",
    "Class 2 (0.66)",
    "Class 3 (1.0)"
]


# --------------------------------------------------
# Device
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("BALANCED CNN - TEST SET EVALUATION")
print("SOLAR PANEL DEFECT DETECTION")
print("=" * 70)

print("\nDevice:", device)


# --------------------------------------------------
# Check model
# --------------------------------------------------

if not MODEL_FILE.exists():

    raise FileNotFoundError(
        f"Balanced model not found:\n{MODEL_FILE}\n"
        "Please run train_cnn_balanced.py first."
    )


# --------------------------------------------------
# Load datasets
# --------------------------------------------------

print("\n" + "=" * 70)
print("LOADING TEST DATASET")
print("=" * 70)

train_dataset, validation_dataset, test_dataset = (
    create_datasets()
)

print("\nTraining samples:", len(train_dataset))
print("Validation samples:", len(validation_dataset))
print("Test samples:", len(test_dataset))


# --------------------------------------------------
# Test DataLoader
# --------------------------------------------------

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# --------------------------------------------------
# Create model
# --------------------------------------------------

model = SolarPanelCNN(
    num_classes=NUM_CLASSES
).to(device)


# --------------------------------------------------
# Load balanced model
# --------------------------------------------------

print("\n" + "=" * 70)
print("LOADING BEST BALANCED CNN MODEL")
print("=" * 70)

checkpoint = torch.load(
    MODEL_FILE,
    map_location=device,
    weights_only=False
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("\nBalanced model loaded successfully.")

if "epoch" in checkpoint:

    print(
        "Saved from epoch:",
        checkpoint["epoch"]
    )

if "validation_loss" in checkpoint:

    print(
        "Validation loss:",
        f"{checkpoint['validation_loss']:.4f}"
    )

if "validation_accuracy" in checkpoint:

    print(
        "Validation accuracy:",
        f"{checkpoint['validation_accuracy']:.4f}"
    )

if "validation_macro_f1" in checkpoint:

    print(
        "Validation Macro F1:",
        f"{checkpoint['validation_macro_f1']:.4f}"
    )


# --------------------------------------------------
# Generate predictions
# --------------------------------------------------

print("\n" + "=" * 70)
print("GENERATING TEST PREDICTIONS")
print("=" * 70)

all_labels = []
all_predictions = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(images)

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        all_labels.extend(
            labels.numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )


all_labels = np.array(all_labels)
all_predictions = np.array(all_predictions)


print("\nPredictions generated successfully.")

print(
    "Number of test labels:",
    len(all_labels)
)

print(
    "Number of predictions:",
    len(all_predictions)
)


# --------------------------------------------------
# Confusion matrix
# --------------------------------------------------

cm = np.zeros(
    (NUM_CLASSES, NUM_CLASSES),
    dtype=int
)

for true_label, predicted_label in zip(
    all_labels,
    all_predictions
):

    cm[
        int(true_label),
        int(predicted_label)
    ] += 1


# --------------------------------------------------
# Calculate metrics
# --------------------------------------------------

total_samples = len(all_labels)

correct_predictions = np.trace(cm)

accuracy = (
    correct_predictions /
    total_samples
)


class_precision = []
class_recall = []
class_f1 = []


for class_id in range(NUM_CLASSES):

    true_positive = cm[
        class_id,
        class_id
    ]

    false_positive = (
        cm[:, class_id].sum()
        - true_positive
    )

    false_negative = (
        cm[class_id, :].sum()
        - true_positive
    )


    # Precision

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


    # Recall

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


    # F1-score

    if (
        precision +
        recall
    ) > 0:

        f1 = (
            2 *
            precision *
            recall /
            (
                precision +
                recall
            )
        )

    else:

        f1 = 0.0


    class_precision.append(
        precision
    )

    class_recall.append(
        recall
    )

    class_f1.append(
        f1
    )


macro_precision = np.mean(
    class_precision
)

macro_recall = np.mean(
    class_recall
)

macro_f1 = np.mean(
    class_f1
)


# --------------------------------------------------
# Test performance
# --------------------------------------------------

print("\n" + "=" * 70)
print("BALANCED CNN TEST SET PERFORMANCE")
print("=" * 70)

print(
    f"\nAccuracy : {accuracy:.4f} "
    f"({accuracy * 100:.2f}%)"
)

print(
    f"Precision: {macro_precision:.4f}"
)

print(
    f"Recall   : {macro_recall:.4f}"
)

print(
    f"Macro F1 : {macro_f1:.4f}"
)


# --------------------------------------------------
# Class-wise report
# --------------------------------------------------

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

report_rows = []

for class_id in range(NUM_CLASSES):

    support = cm[
        class_id,
        :
    ].sum()

    report_rows.append({
        "Class": CLASS_NAMES[class_id],
        "Precision": class_precision[class_id],
        "Recall": class_recall[class_id],
        "F1-Score": class_f1[class_id],
        "Support": support
    })


report_df = pd.DataFrame(
    report_rows
)

print("\n")

print(
    report_df.to_string(
        index=False,
        formatters={
            "Precision":
                "{:.4f}".format,
            "Recall":
                "{:.4f}".format,
            "F1-Score":
                "{:.4f}".format
        }
    )
)


# --------------------------------------------------
# Confusion matrix
# --------------------------------------------------

print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print("\n")
print(cm)


# --------------------------------------------------
# Save classification report
# --------------------------------------------------

report_file = (
    RESULTS_DIR /
    "cnn_balanced_classification_report.csv"
)

report_df.to_csv(
    report_file,
    index=False
)


# --------------------------------------------------
# Save overall metrics
# --------------------------------------------------

metrics_df = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Macro Precision",
        "Macro Recall",
        "Macro F1"
    ],
    "Score": [
        accuracy,
        macro_precision,
        macro_recall,
        macro_f1
    ]
})

metrics_file = (
    RESULTS_DIR /
    "cnn_balanced_test_metrics.csv"
)

metrics_df.to_csv(
    metrics_file,
    index=False
)


# --------------------------------------------------
# Save confusion matrix CSV
# --------------------------------------------------

cm_df = pd.DataFrame(
    cm,
    index=CLASS_NAMES,
    columns=CLASS_NAMES
)

cm_csv_file = (
    RESULTS_DIR /
    "cnn_balanced_confusion_matrix.csv"
)

cm_df.to_csv(
    cm_csv_file
)


# --------------------------------------------------
# Plot confusion matrix
# --------------------------------------------------

fig, ax = plt.subplots(
    figsize=(8, 7)
)

image = ax.imshow(
    cm,
    cmap="Blues"
)

plt.colorbar(
    image,
    ax=ax
)

ax.set_xticks(
    range(NUM_CLASSES)
)

ax.set_yticks(
    range(NUM_CLASSES)
)

ax.set_xticklabels(
    CLASS_NAMES,
    rotation=25,
    ha="right"
)

ax.set_yticklabels(
    CLASS_NAMES
)

ax.set_xlabel(
    "Predicted Class"
)

ax.set_ylabel(
    "True Class"
)

ax.set_title(
    "Balanced CNN Confusion Matrix - ELPV Test Set"
)


# Numbers inside confusion matrix

for i in range(NUM_CLASSES):

    for j in range(NUM_CLASSES):

        ax.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center"
        )


plt.tight_layout()

cm_file = (
    RESULTS_DIR /
    "cnn_balanced_confusion_matrix.png"
)

plt.savefig(
    cm_file,
    dpi=150,
    bbox_inches="tight"
)

plt.close()


# --------------------------------------------------
# Save predictions
# --------------------------------------------------

predictions_df = (
    test_dataset.dataframe.copy()
)

predictions_df[
    "predicted_label"
] = all_predictions

predictions_df[
    "correct"
] = (
    predictions_df["label"]
    ==
    predictions_df["predicted_label"]
)


predictions_file = (
    RESULTS_DIR /
    "cnn_balanced_test_predictions.csv"
)

predictions_df.to_csv(
    predictions_file,
    index=False
)


# --------------------------------------------------
# Final summary
# --------------------------------------------------

print("\n" + "=" * 70)
print("BALANCED CNN TEST EVALUATION COMPLETED")
print("=" * 70)

print("\nSaved classification report:")
print(report_file)

print("\nSaved test metrics:")
print(metrics_file)

print("\nSaved confusion matrix CSV:")
print(cm_csv_file)

print("\nSaved confusion matrix image:")
print(cm_file)

print("\nSaved test predictions:")
print(predictions_file)

print("\n" + "=" * 70)