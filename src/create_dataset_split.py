from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "elpv-dataset"

LABEL_FILE = DATA_DIR / "labels.csv"
RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Load labels
# --------------------------------------------------

labels = pd.read_csv(
    LABEL_FILE,
    sep=r"\s+",
    header=None,
    names=["image_path", "defect_probability", "cell_type"]
)

print("=" * 60)
print("CREATING TRAIN / VALIDATION / TEST SPLIT")
print("=" * 60)

print("\nOriginal dataset:", labels.shape)


# --------------------------------------------------
# Create 4-class label
# --------------------------------------------------

def create_class_label(probability):
    if probability == 0.0:
        return 0
    elif abs(probability - 0.333333) < 0.01:
        return 1
    elif abs(probability - 0.666667) < 0.01:
        return 2
    elif probability == 1.0:
        return 3
    else:
        raise ValueError(f"Unexpected defect probability: {probability}")


labels["label"] = labels["defect_probability"].apply(create_class_label)


# --------------------------------------------------
# Check class distribution
# --------------------------------------------------

print("\nClass distribution:")
print(labels["label"].value_counts().sort_index())


# --------------------------------------------------
# First split:
# 70% train
# 30% temporary
# --------------------------------------------------

train_df, temp_df = train_test_split(
    labels,
    test_size=0.30,
    random_state=42,
    stratify=labels["label"]
)


# --------------------------------------------------
# Second split:
# 15% validation
# 15% test
# --------------------------------------------------

validation_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42,
    stratify=temp_df["label"]
)


# --------------------------------------------------
# Add split information
# --------------------------------------------------

train_df = train_df.copy()
validation_df = validation_df.copy()
test_df = test_df.copy()

train_df["split"] = "train"
validation_df["split"] = "validation"
test_df["split"] = "test"


# --------------------------------------------------
# Combine
# --------------------------------------------------

dataset_split = pd.concat(
    [train_df, validation_df, test_df],
    ignore_index=True
)


# --------------------------------------------------
# Shuffle
# --------------------------------------------------

dataset_split = dataset_split.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)


# --------------------------------------------------
# Save
# --------------------------------------------------

output_file = RESULTS_DIR / "dataset_split.csv"

dataset_split.to_csv(
    output_file,
    index=False
)


# --------------------------------------------------
# Print results
# --------------------------------------------------

print("\nSplit sizes:")
print(dataset_split["split"].value_counts())

print("\nClass distribution by split:")
print(
    pd.crosstab(
        dataset_split["split"],
        dataset_split["label"]
    )
)

print("\nSaved:")
print(output_file)

print("\n" + "=" * 60)
print("DATASET SPLIT COMPLETED")
print("=" * 60)