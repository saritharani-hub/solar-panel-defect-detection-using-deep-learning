
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import create_datasets
from resnet18_model import create_resnet18_model


# --------------------------------------------------
# Paths and configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_FILE = BASE_DIR / "models" / "best_resnet18_model.pth"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 32
NUM_CLASSES = 4

CLASS_NAMES = [
    "Class 0 (0.0)",
    "Class 1 (0.33)",
    "Class 2 (0.66)",
    "Class 3 (1.0)",
]

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# --------------------------------------------------
# Manual classification metrics (no sklearn)
# --------------------------------------------------

def calculate_metrics(y_true, y_pred):
    confusion = np.zeros(
        (NUM_CLASSES, NUM_CLASSES),
        dtype=np.int64
    )

    for actual, predicted in zip(y_true, y_pred):
        confusion[int(actual), int(predicted)] += 1

    per_class = []

    for class_id in range(NUM_CLASSES):
        tp = confusion[class_id, class_id]
        fp = confusion[:, class_id].sum() - tp
        fn = confusion[class_id, :].sum() - tp
        support = confusion[class_id, :].sum()

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0

        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )

        per_class.append({
            "class_id": class_id,
            "class_name": CLASS_NAMES[class_id],
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "support": int(support),
        })

    accuracy = float(np.mean(y_true == y_pred))

    macro_precision = float(
        np.mean([item["precision"] for item in per_class])
    )
    macro_recall = float(
        np.mean([item["recall"] for item in per_class])
    )
    macro_f1 = float(
        np.mean([item["f1_score"] for item in per_class])
    )

    return (
        confusion,
        per_class,
        accuracy,
        macro_precision,
        macro_recall,
        macro_f1,
    )


# --------------------------------------------------
# Manual ROC-AUC (one-vs-rest, no sklearn/scipy)
# --------------------------------------------------

def binary_auc(y_binary, scores):
    """
    Calculate binary ROC-AUC using the rank-sum method.
    Handles tied scores by assigning average ranks.
    """
    y_binary = np.asarray(y_binary, dtype=int)
    scores = np.asarray(scores, dtype=float)

    positives = int(np.sum(y_binary == 1))
    negatives = int(np.sum(y_binary == 0))

    if positives == 0 or negatives == 0:
        return float("nan")

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=float)

    start = 0
    while start < len(scores):
        end = start + 1

        while (
            end < len(scores)
            and sorted_scores[end] == sorted_scores[start]
        ):
            end += 1

        # Ranks are 1-based; tied values receive average rank.
        average_rank = ((start + 1) + end) / 2.0
        ranks[order[start:end]] = average_rank
        start = end

    positive_rank_sum = float(np.sum(ranks[y_binary == 1]))

    auc = (
        positive_rank_sum
        - positives * (positives + 1) / 2.0
    ) / (positives * negatives)

    return float(auc)


def calculate_multiclass_auc(y_true, probabilities):
    """
    Calculate one-vs-rest ROC-AUC for each class and macro ROC-AUC.
    """
    class_aucs = []

    for class_id in range(NUM_CLASSES):
        binary_labels = (y_true == class_id).astype(int)

        auc = binary_auc(
            binary_labels,
            probabilities[:, class_id]
        )

        class_aucs.append({
            "class_id": class_id,
            "class_name": CLASS_NAMES[class_id],
            "roc_auc_ovr": auc,
        })

    valid_aucs = [
        item["roc_auc_ovr"]
        for item in class_aucs
        if np.isfinite(item["roc_auc_ovr"])
    ]

    macro_auc = (
        float(np.mean(valid_aucs))
        if valid_aucs else float("nan")
    )

    return class_aucs, macro_auc


# --------------------------------------------------
# Main evaluation
# --------------------------------------------------

def main():
    print("=" * 65)
    print("RESNET18 - TEST SET EVALUATION")
    print("SOLAR PANEL DEFECT DETECTION")
    print("=" * 65)
    print("Device:", DEVICE)

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Saved model not found: {MODEL_FILE}"
        )

    # Use the existing split and evaluation preprocessing.
    _, _, test_dataset = create_datasets()

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    print("Test samples:", len(test_dataset))

    model = create_resnet18_model(
        num_classes=NUM_CLASSES,
        pretrained=False,
    )

    checkpoint = torch.load(
        MODEL_FILE,
        map_location=DEVICE,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    print("Checkpoint epoch:", checkpoint["epoch"])
    print(
        "Best validation Macro F1:",
        f'{checkpoint["validation_macro_f1"]:.4f}',
    )

    all_labels = []
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)

            outputs = model(images)

            # Convert model outputs to class probabilities.
            probabilities = torch.softmax(outputs, dim=1)
            predictions = probabilities.argmax(dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(
                probabilities.cpu().numpy()
            )

    y_true = np.asarray(all_labels, dtype=int)
    y_pred = np.asarray(all_predictions, dtype=int)
    y_prob = np.asarray(all_probabilities, dtype=float)

    (
        confusion,
        per_class,
        accuracy,
        macro_precision,
        macro_recall,
        macro_f1,
    ) = calculate_metrics(y_true, y_pred)

    class_aucs, macro_auc = calculate_multiclass_auc(
        y_true, y_prob
    )

    print("\nTEST SET PERFORMANCE")
    print("-" * 45)
    print(f"Accuracy:         {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Macro Precision:  {macro_precision:.4f}")
    print(f"Macro Recall:     {macro_recall:.4f}")
    print(f"Macro F1:         {macro_f1:.4f}")
    print(f"Macro ROC-AUC:    {macro_auc:.4f}")

    # Save per-class classification report.
    report_df = pd.DataFrame(per_class)
    report_df.to_csv(
        RESULTS_DIR / "resnet18_classification_report.csv",
        index=False,
    )

    # Save overall metrics, now including macro ROC-AUC.
    metrics_df = pd.DataFrame([{
        "model": "ResNet18 Transfer Learning",
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "macro_roc_auc_ovr": macro_auc,
    }])

    metrics_df.to_csv(
        RESULTS_DIR / "resnet18_test_metrics.csv",
        index=False,
    )

    # Save confusion matrix.
    confusion_df = pd.DataFrame(
        confusion,
        index=CLASS_NAMES,
        columns=CLASS_NAMES,
    )

    confusion_df.to_csv(
        RESULTS_DIR / "resnet18_confusion_matrix.csv"
    )

    # Save actual labels, predicted labels, and all probabilities.
    prediction_data = {
        "actual_label": y_true,
        "predicted_label": y_pred,
    }

    for class_id in range(NUM_CLASSES):
        prediction_data[f"probability_class_{class_id}"] = (
            y_prob[:, class_id]
        )

    predictions_df = pd.DataFrame(prediction_data)

    predictions_df.to_csv(
        RESULTS_DIR / "resnet18_test_predictions.csv",
        index=False,
    )

    # Save ROC-AUC for each class.
    auc_df = pd.DataFrame(class_aucs)
    auc_df.to_csv(
        RESULTS_DIR / "resnet18_class_roc_auc.csv",
        index=False,
    )

    print("\nPER-CLASS CLASSIFICATION PERFORMANCE")
    print(report_df.to_string(index=False))

    print("\nPER-CLASS ROC-AUC (ONE-VS-REST)")
    print(auc_df.to_string(index=False))

    print("\nCONFUSION MATRIX")
    print("Rows = actual classes; columns = predicted classes")
    print(confusion_df.to_string())

    print("\nSaved evaluation files:")
    print(RESULTS_DIR / "resnet18_classification_report.csv")
    print(RESULTS_DIR / "resnet18_test_metrics.csv")
    print(RESULTS_DIR / "resnet18_confusion_matrix.csv")
    print(RESULTS_DIR / "resnet18_test_predictions.csv")
    print(RESULTS_DIR / "resnet18_class_roc_auc.csv")

    print("\nRESNET18 TEST EVALUATION COMPLETED")


if __name__ == "__main__":
    main()
