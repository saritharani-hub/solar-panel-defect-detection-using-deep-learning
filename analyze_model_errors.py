

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"

matrix_files = {
    "Baseline CNN": "cnn_confusion_matrix.csv",
    "Balanced CNN": "cnn_balanced_confusion_matrix.csv",
    "ResNet18": "resnet18_confusion_matrix.csv",
}

class_names = ["0.00", "0.33", "0.66", "1.00"]
records = []

print("=" * 70)
print("CORRECTED CLASS-WISE MODEL COMPARISON")
print("=" * 70)

for model_name, filename in matrix_files.items():
    path = RESULTS_DIR / filename

    if not path.exists():
        print(f"Missing file: {filename}")
        continue

    df = pd.read_csv(path, index_col=0)
    matrix = df.apply(pd.to_numeric, errors="coerce").to_numpy()

    if matrix.shape != (4, 4) or pd.isna(matrix).any():
        print(f"Invalid confusion matrix: {filename}")
        continue

    matrix = matrix.astype(int)
    print(f"\n{model_name} confusion matrix:")
    print(matrix)

    for i, class_name in enumerate(class_names):
        tp = matrix[i, i]
        actual_count = matrix[i, :].sum()
        predicted_count = matrix[:, i].sum()

        precision = (
            tp / predicted_count if predicted_count else 0.0
        )
        recall = tp / actual_count if actual_count else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )

        records.append({
            "Model": model_name,
            "Defect Probability": class_name,
            "Precision": precision,
            "Recall": recall,
            "F1-score": f1,
            "Support": int(actual_count),
        })

if not records:
    raise ValueError(
        "No valid confusion matrices found. Check the filenames."
    )

comparison = pd.DataFrame(records)

csv_path = RESULTS_DIR / "class_wise_model_comparison.csv"
comparison.to_csv(csv_path, index=False)

print("\nCLASS-WISE METRICS")
print(comparison.round(4).to_string(index=False))
print(f"\nSaved: {csv_path}")

# Plot per-class F1-score
pivot = comparison.pivot(
    index="Defect Probability",
    columns="Model",
    values="F1-score",
)

ax = pivot.plot(kind="bar", figsize=(10, 6), ylim=(0, 1))
ax.set_title("Per-Class F1-Score Comparison")
ax.set_xlabel("Defect Probability")
ax.set_ylabel("F1-score")
ax.legend(title="Model")
plt.tight_layout()

chart_path = RESULTS_DIR / "class_wise_f1_comparison.png"
plt.savefig(chart_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved: {chart_path}")
print("\nCLASS-WISE COMPARISON COMPLETED")
