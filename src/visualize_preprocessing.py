from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from dataset import preprocess_image, train_transform


# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data" / "elpv-dataset"
IMAGE_DIR = DATA_DIR / "images"
SPLIT_FILE = BASE_DIR / "results" / "dataset_split.csv"


# --------------------------------------------------
# Load dataset split
# --------------------------------------------------

df = pd.read_csv(SPLIT_FILE)

# Select one training image
sample_row = df[df["split"] == "train"].iloc[0]

image_name = Path(sample_row["image_path"]).name
image_path = IMAGE_DIR / image_name

label = int(sample_row["label"])
defect_probability = sample_row["defect_probability"]
cell_type = sample_row["cell_type"]


# --------------------------------------------------
# Load original image
# --------------------------------------------------

original_image = Image.open(image_path).convert("L")


# --------------------------------------------------
# Preprocess image
# --------------------------------------------------

preprocessed_image = preprocess_image(
    image_path,
    image_size=224
)


# --------------------------------------------------
# Create augmented image
# --------------------------------------------------

augmented_image = train_transform(
    preprocessed_image
)


# Convert tensor to displayable image
augmented_display = augmented_image.squeeze(0).numpy()

# Undo normalization:
# normalized = (image - 0.5) / 0.5
augmented_display = (
    augmented_display * 0.5
) + 0.5

augmented_display = augmented_display.clip(
    0,
    1
)


# --------------------------------------------------
# Plot comparison
# --------------------------------------------------

fig, axes = plt.subplots(
    1,
    3,
    figsize=(15, 5)
)


# Original
axes[0].imshow(
    original_image,
    cmap="gray"
)

axes[0].set_title(
    "Original Image"
)

axes[0].axis("off")


# Preprocessed
axes[1].imshow(
    preprocessed_image,
    cmap="gray"
)

axes[1].set_title(
    "Preprocessed Image"
)

axes[1].axis("off")


# Augmented
axes[2].imshow(
    augmented_display,
    cmap="gray"
)

axes[2].set_title(
    "Augmented Image"
)

axes[2].axis("off")


# --------------------------------------------------
# Overall title
# --------------------------------------------------

fig.suptitle(
    f"ELPV Preprocessing Visualization\n"
    f"Label: {label} | "
    f"Defect Probability: {defect_probability} | "
    f"Cell Type: {cell_type}",
    fontsize=12
)

plt.tight_layout()


# --------------------------------------------------
# Save result
# --------------------------------------------------

output_dir = BASE_DIR / "results"
output_dir.mkdir(
    parents=True,
    exist_ok=True
)

output_file = (
    output_dir /
    "preprocessing_visualization.png"
)

plt.savefig(
    output_file,
    dpi=150,
    bbox_inches="tight"
)

print("=" * 60)
print("PREPROCESSING VISUALIZATION")
print("=" * 60)

print("\nImage:", image_name)
print("Label:", label)
print("Defect probability:", defect_probability)
print("Cell type:", cell_type)

print("\nSaved visualization:")
print(output_file)

plt.show()