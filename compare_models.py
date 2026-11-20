
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Project directories
BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"

# Your existing combined metrics file
metrics_file = RESULTS_DIR / "model_comparison_test_metrics.csv"

if not metrics_file.exists():
    raise FileNotFoundError(
        f"Metrics file not found: {metrics_file}"
    )

# Load model comparison
df = pd.read_csv(metrics_file)
df.columns = df.columns.str.strip()

print("\n" + "=" * 65)
print("SOLAR PANEL DEFECT DETECTION - MODEL COMPARISON")
print("=" * 65)
print(df.to_string(index=False))

# Save a clean copy
output_csv = RESULTS_DIR / "final_model_comparison.csv"
df.to_csv(output_csv, index=False)
print(f"\nComparison saved: {output_csv}")

# Identify metric columns regardless of capitalization
column_lookup = {
    col.strip().lower().replace("_", " ").replace("-", " "): col
    for col in df.columns
}

accuracy_col = column_lookup.get("accuracy")
f1_col = column_lookup.get("macro f1")

if accuracy_col and f1_col:
    df[accuracy_col] = pd.to_numeric(
        df[accuracy_col], errors="coerce"
    )
    df[f1_col] = pd.to_numeric(
        df[f1_col], errors="coerce"
    )

    plot_df = df.dropna(
        subset=[accuracy_col, f1_col]
    ).copy()

    if not plot_df.empty:
        labels = plot_df["Model"].astype(str)

        ax = plot_df.set_index(labels)[
            [accuracy_col, f1_col]
        ].plot(
            kind="bar",
            figsize=(10, 6),
            ylim=(0, 1),
            rot=15
        )

        ax.set_title("Solar Panel Defect Detection: Model Comparison")
        ax.set_ylabel("Score")
        ax.set_xlabel("Model")
        ax.legend(["Accuracy", "Macro F1"])
        plt.tight_layout()

        chart_file = RESULTS_DIR / "final_model_comparison.png"
        plt.savefig(chart_file, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Comparison chart saved: {chart_file}")
else:
    print(
        "\nCould not identify Accuracy and Macro F1 columns. "
        "Check the column names printed above."
    )

print("\nNEXT: Review the class-wise classification reports.")
