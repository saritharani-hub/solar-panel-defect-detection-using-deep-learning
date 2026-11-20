
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    auc
)

from dataset import create_datasets
from cnn_model import SolarPanelCNN


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_FILE = BASE_DIR / "models" / "best_cnn_model.pth"
RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# CONFIGURATION
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
# DEVICE
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("SOLAR PANEL DEFECT DETECTION - CNN EVALUATION")
print("=" * 70)

print("Device:", device)


# --------------------------------------------------
# CHECK MODEL
# --------------------------------------------------

if not MODEL_FILE.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_FILE}"
    )


# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

print("\nLoading datasets...")

train_dataset, validation_dataset, test_dataset = create_datasets()

print("Training samples:", len(train_dataset))
print("Validation samples:", len(validation_dataset))
print("Test samples:", len(test_dataset))


test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# --------------------------------------------------
# CREATE MODEL
# --------------------------------------------------

model = SolarPanelCNN(
    num_classes=NUM_CLASSES
).to(device)


# --------------------------------------------------
# LOAD TRAINED WEIGHTS
# --------------------------------------------------

print("\nLoading trained CNN model...")

checkpoint = torch.load(
    MODEL_FILE,
    map_location=device,
    weights_only=False
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("Model loaded successfully.")

if "epoch" in checkpoint:
    print("Best epoch:", checkpoint["epoch"])


# --------------------------------------------------
# GENERATE PREDICTIONS AND PROBABILITIES
# --------------------------------------------------

print("\nGenerating predictions...")

all_labels = []
all_predictions = []
all_probabilities = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(images)

        # Convert logits into probabilities
        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        predictions = torch.argmax(
            probabilities,
            dim=1
        )

        all_labels.extend(
            labels.cpu().numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_probabilities.extend(
            probabilities.cpu().numpy()
        )


all_labels = np.array(all_labels)
all_predictions = np.array(all_predictions)
all_probabilities = np.array(all_probabilities)

print("Predictions generated:", len(all_predictions))
print("Probability shape:", all_probabilities.shape)


# --------------------------------------------------
# VALIDATE OUTPUT
# --------------------------------------------------

if all_probabilities.shape != (
    len(all_labels), NUM_CLASSES
):
    raise ValueError("Unexpected probability array shape.")

if len(all_labels) != len(all_predictions):
    raise ValueError("Labels and predictions do not match.")

if not np.allclose(
    all_probabilities.sum(axis=1),
    1.0,
    atol=1e-5
):
    raise ValueError("Probability rows do not sum to 1.")


# --------------------------------------------------
# CONFUSION MATRIX
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
# CLASS-WISE METRICS
# --------------------------------------------------

class_precision = []
class_recall = []
class_f1 = []

for class_id in range(NUM_CLASSES):

    tp = cm[class_id, class_id]

    fp = cm[:, class_id].sum() - tp

    fn = cm[class_id, :].sum() - tp

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0 else 0.0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0 else 0.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )

    class_precision.append(precision)
    class_recall.append(recall)
    class_f1.append(f1)


accuracy = np.mean(all_labels == all_predictions)

macro_precision = np.mean(class_precision)
macro_recall = np.mean(class_recall)
macro_f1 = np.mean(class_f1)


# --------------------------------------------------
# MULTICLASS ROC-AUC
# --------------------------------------------------

print("\nCalculating ROC-AUC...")

try:

    macro_roc_auc = roc_auc_score(
        all_labels,
        all_probabilities,
        labels=np.arange(NUM_CLASSES),
        multi_class="ovr",
        average="macro"
    )

    print("Macro ROC-AUC:", round(macro_roc_auc, 4))

except ValueError as error:

    macro_roc_auc = np.nan

    print("Macro ROC-AUC unavailable:", error)


# --------------------------------------------------
# CLASS-WISE ROC-AUC
# --------------------------------------------------

class_auc = []

for class_id in range(NUM_CLASSES):

    binary_labels = (
        all_labels == class_id
    ).astype(int)

    if len(np.unique(binary_labels)) == 2:

        score = roc_auc_score(
            binary_labels,
            all_probabilities[:, class_id]
        )

    else:
        score = np.nan

    class_auc.append(score)


# --------------------------------------------------
# OVERALL PERFORMANCE
# --------------------------------------------------

print("\n" + "=" * 70)
print("CNN TEST PERFORMANCE")
print("=" * 70)

print(f"Accuracy          : {accuracy:.4f}")
print(f"Macro Precision   : {macro_precision:.4f}")
print(f"Macro Recall      : {macro_recall:.4f}")
print(f"Macro F1          : {macro_f1:.4f}")
print(f"Macro ROC-AUC     : {macro_roc_auc:.4f}")


# --------------------------------------------------
# CLASSIFICATION REPORT
# --------------------------------------------------

report_df = pd.DataFrame({

    "Class": CLASS_NAMES,

    "Precision": class_precision,

    "Recall": class_recall,

    "F1-Score": class_f1,

    "ROC-AUC": class_auc,

    "Support": cm.sum(axis=1)

})

print("\nCLASS-WISE PERFORMANCE")

print(
    report_df.to_string(index=False)
)


# --------------------------------------------------
# SAVE CLASSIFICATION REPORT
# --------------------------------------------------

report_file = (
    RESULTS_DIR / "cnn_classification_report.csv"
)

report_df.to_csv(
    report_file,
    index=False
)


# --------------------------------------------------
# SAVE OVERALL METRICS
# --------------------------------------------------

metrics_df = pd.DataFrame({

    "Metric": [
        "Accuracy",
        "Macro Precision",
        "Macro Recall",
        "Macro F1",
        "Macro ROC-AUC"
    ],

    "Score": [
        accuracy,
        macro_precision,
        macro_recall,
        macro_f1,
        macro_roc_auc
    ]

})

metrics_file = (
    RESULTS_DIR / "cnn_test_metrics.csv"
)

metrics_df.to_csv(
    metrics_file,
    index=False
)


# --------------------------------------------------
# SAVE CONFUSION MATRIX
# --------------------------------------------------

cm_df = pd.DataFrame(
    cm,
    index=CLASS_NAMES,
    columns=CLASS_NAMES
)

cm_file = (
    RESULTS_DIR / "cnn_confusion_matrix.csv"
)

cm_df.to_csv(cm_file)


# --------------------------------------------------
# PLOT CONFUSION MATRIX
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 7))

image = ax.imshow(
    cm,
    cmap="Blues"
)

plt.colorbar(image, ax=ax)

ax.set_xticks(range(NUM_CLASSES))
ax.set_yticks(range(NUM_CLASSES))

ax.set_xticklabels(
    CLASS_NAMES,
    rotation=25,
    ha="right"
)

ax.set_yticklabels(CLASS_NAMES)

ax.set_xlabel("Predicted Class")
ax.set_ylabel("True Class")

ax.set_title("CNN Confusion Matrix")

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

cm_image_file = (
    RESULTS_DIR / "cnn_confusion_matrix.png"
)

plt.savefig(
    cm_image_file,
    dpi=150,
    bbox_inches="tight"
)

plt.close()


# --------------------------------------------------
# MULTICLASS ROC CURVE
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(9, 7))

for class_id in range(NUM_CLASSES):

    binary_labels = (
        all_labels == class_id
    ).astype(int)

    if len(np.unique(binary_labels)) < 2:
        continue

    fpr, tpr, thresholds = roc_curve(
        binary_labels,
        all_probabilities[:, class_id]
    )

    class_roc_auc = auc(fpr, tpr)

    ax.plot(
        fpr,
        tpr,
        label=(
            f"{CLASS_NAMES[class_id]} "
            f"(AUC = {class_roc_auc:.3f})"
        )
    )

ax.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")

ax.set_title("Multiclass ROC Curves - CNN")

ax.legend(loc="lower right")

ax.grid(True)

plt.tight_layout()

roc_file = (
    RESULTS_DIR / "cnn_roc_curve.png"
)

plt.savefig(
    roc_file,
    dpi=150,
    bbox_inches="tight"
)

plt.close()


# --------------------------------------------------
# SAVE TEST PREDICTIONS AND PROBABILITIES
# --------------------------------------------------

predictions_df = test_dataset.dataframe.copy()

if len(predictions_df) != len(all_labels):
    raise ValueError(
        "Test dataframe and prediction lengths do not match."
    )

predictions_df["true_label"] = all_labels

predictions_df["predicted_label"] = all_predictions

predictions_df["correct"] = (
    all_labels == all_predictions
)

for class_id in range(NUM_CLASSES):

    predictions_df[
        f"probability_class_{class_id}"
    ] = all_probabilities[:, class_id]


predictions_file = (
    RESULTS_DIR / "cnn_test_predictions.csv"
)

predictions_df.to_csv(
    predictions_file,
    index=False
)


# --------------------------------------------------
# FINAL SUMMARY
# --------------------------------------------------

print("\n" + "=" * 70)
print("CNN EVALUATION COMPLETED")
print("=" * 70)

print("\nSaved files:")

print(report_file)
print(metrics_file)
print(cm_file)
print(cm_image_file)
print(roc_file)
print(predictions_file)

print("\nAll evaluation results saved successfully.")

print("=" * 70)
