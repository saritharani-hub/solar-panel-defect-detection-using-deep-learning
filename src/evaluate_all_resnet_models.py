
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

from dataset import create_datasets
from resnet18_model import create_resnet18_model


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 32
NUM_CLASSES = 4

CLASS_NAMES = [
    "Class 0 (0.00)",
    "Class 1 (0.33)",
    "Class 2 (0.66)",
    "Class 3 (1.00)"
]

MODEL_FILES = {
    "ResNet18": "best_resnet18_model.pth",
    "ResNet18 V2": "best_resnet18_v2_model.pth"
}


def evaluate_model(model_name, model_file, test_loader):

    model_path = MODEL_DIR / model_file

    if not model_path.exists():
        raise FileNotFoundError(model_path)

    print(f"\nEvaluating {model_name}...")

    model = create_resnet18_model(
        num_classes=NUM_CLASSES,
        pretrained=False
    )

    checkpoint = torch.load(
        model_path,
        map_location=DEVICE,
        weights_only=False
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    y_true = []
    y_pred = []
    y_prob = []

    with torch.no_grad():
        for images, labels in test_loader:

            images = images.to(DEVICE)

            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)

            predictions = probabilities.argmax(dim=1)

            y_true.extend(labels.numpy())
            y_pred.extend(predictions.cpu().numpy())
            y_prob.extend(probabilities.cpu().numpy())

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)

    accuracy = accuracy_score(y_true, y_pred)

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=[0, 1, 2, 3],
            zero_division=0
        )
    )

    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_f1 = np.mean(f1)

    macro_auc = roc_auc_score(
        y_true,
        y_prob,
        labels=[0, 1, 2, 3],
        multi_class="ovr",
        average="macro"
    )

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro Precision: {macro_precision:.4f}")
    print(f"Macro Recall: {macro_recall:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Macro ROC-AUC: {macro_auc:.4f}")

    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1, 2, 3],
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0
    )

    pd.DataFrame(report).transpose().to_csv(
        RESULTS_DIR / f"{model_name.lower().replace(' ', '_')}_classification_report.csv"
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1, 2, 3]
    )

    pd.DataFrame(
        matrix,
        index=CLASS_NAMES,
        columns=CLASS_NAMES
    ).to_csv(
        RESULTS_DIR / f"{model_name.lower().replace(' ', '_')}_confusion_matrix.csv"
    )

    return {
        "Model": model_name,
        "Accuracy": accuracy,
        "Macro Precision": macro_precision,
        "Macro Recall": macro_recall,
        "Macro F1": macro_f1,
        "Macro ROC-AUC": macro_auc
    }


def main():

    print("=" * 65)
    print("RESNET18 MODEL COMPARISON - TEST EVALUATION")
    print("=" * 65)

    print("Device:", DEVICE)

    _, _, test_dataset = create_datasets()

    print("Test samples:", len(test_dataset))

    if len(test_dataset) != 394:
        raise ValueError(
            "Expected 394 test images. Check dataset_split.csv."
        )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    results = []

    for model_name, model_file in MODEL_FILES.items():

        result = evaluate_model(
            model_name,
            model_file,
            test_loader
        )

        results.append(result)

    comparison = pd.DataFrame(results)

    comparison.to_csv(
        RESULTS_DIR / "resnet18_models_test_comparison.csv",
        index=False
    )

    print("\nFINAL RESNET18 COMPARISON")
    print("=" * 65)
    print(comparison.to_string(index=False))

    print("\nSaved results to:", RESULTS_DIR)
    print("\nRESNET18 EVALUATION COMPLETED")


if __name__ == "__main__":
    main()
