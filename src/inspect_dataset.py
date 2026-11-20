from pathlib import Path
import pandas as pd
from PIL import Image


# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "elpv-dataset"

IMAGE_DIR = DATA_DIR / "images"
LABEL_FILE = DATA_DIR / "labels.csv"


# --------------------------------------------------
# Check paths
# --------------------------------------------------

print("=" * 60)
print("ELPV DATASET INSPECTION")
print("=" * 60)

print(f"Dataset directory: {DATA_DIR}")
print(f"Images directory:  {IMAGE_DIR}")
print(f"Labels file:      {LABEL_FILE}")

if not IMAGE_DIR.exists():
    raise FileNotFoundError("Images directory not found.")

if not LABEL_FILE.exists():
    raise FileNotFoundError("labels.csv not found.")


# --------------------------------------------------
# Count images
# --------------------------------------------------

image_files = sorted(IMAGE_DIR.glob("*.png"))

print("\nNumber of PNG images:", len(image_files))


# --------------------------------------------------
# Load labels
# --------------------------------------------------

labels = pd.read_csv(
    LABEL_FILE,
    sep=r"\s+",
    header=None,
    names=["image_path", "defect_probability", "cell_type"]
)

print("\nLabels shape:", labels.shape)

print("\nColumn names:")
print(labels.columns.tolist())

print("\nFirst 5 rows:")
print(labels.head())


# --------------------------------------------------
# Image inspection
# --------------------------------------------------

if image_files:
    sample_image = image_files[0]

    with Image.open(sample_image) as img:
        print("\nSample image:")
        print("File:", sample_image.name)
        print("Size:", img.size)
        print("Mode:", img.mode)


# --------------------------------------------------
# Dataset information
# --------------------------------------------------

print("\nDataset information:")
print(labels.info())


# --------------------------------------------------
# Defect probability distribution
# --------------------------------------------------

print("\nDefect probability distribution:")

for column in labels.columns:
    unique_values = labels[column].unique()

    if any(value in unique_values for value in [0, 0.33, 0.66, 1.0]):
        print(labels[column].value_counts(dropna=False))


# --------------------------------------------------
# Cell type distribution
# --------------------------------------------------

print("\nPossible cell-type distributions:")

for column in labels.columns:
    unique_values = labels[column].astype(str).str.lower().unique()

    if any(
        value in unique_values
        for value in ["mono", "poly", "monocrystalline", "polycrystalline"]
    ):
        print(f"\nColumn: {column}")
        print(labels[column].value_counts(dropna=False))


# --------------------------------------------------
# Missing values
# --------------------------------------------------

print("\nMissing values:")
print(labels.isnull().sum())

print("\n" + "=" * 60)
print("INSPECTION COMPLETED")
print("=" * 60)