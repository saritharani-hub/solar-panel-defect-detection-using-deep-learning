from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data" / "elpv-dataset"
IMAGE_DIR = DATA_DIR / "images"
SPLIT_FILE = BASE_DIR / "results" / "dataset_split.csv"


# --------------------------------------------------
# Image preprocessing
# --------------------------------------------------

def preprocess_image(image_path, image_size=224):
    """
    Load and preprocess one ELPV grayscale image.

    Steps:
    1. Read grayscale image
    2. Mild denoising
    3. Contrast normalization
    4. Resize
    5. Convert to PIL image
    """

    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:
        raise FileNotFoundError(
            f"Could not read image: {image_path}"
        )

    # --------------------------------------------------
    # Mild denoising
    # --------------------------------------------------

    image = cv2.GaussianBlur(
        image,
        (3, 3),
        0
    )

    # --------------------------------------------------
    # Contrast normalization
    # CLAHE helps with inconsistent EL illumination
    # --------------------------------------------------

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    image = clahe.apply(image)

    # --------------------------------------------------
    # Resize
    # --------------------------------------------------

    image = cv2.resize(
        image,
        (image_size, image_size),
        interpolation=cv2.INTER_AREA
    )

    # Convert grayscale NumPy array to PIL image
    image = Image.fromarray(image)

    return image


# --------------------------------------------------
# Training augmentation
# --------------------------------------------------

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),

    transforms.RandomRotation(
        degrees=15
    ),

    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15
    ),

    transforms.ElasticTransform(
        alpha=20.0,
        sigma=5.0
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.5],
        std=[0.5]
    )
])


# --------------------------------------------------
# Validation / test transformation
# --------------------------------------------------

eval_transform = transforms.Compose([
    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.5],
        std=[0.5]
    )
])


# --------------------------------------------------
# ELPV Dataset
# --------------------------------------------------

class ELPVDataset(Dataset):

    def __init__(
        self,
        dataframe,
        transform=None,
        image_size=224
    ):

        self.dataframe = dataframe.reset_index(
            drop=True
        )

        self.transform = transform
        self.image_size = image_size

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):

        row = self.dataframe.iloc[index]

        image_name = Path(
            row["image_path"]
        ).name

        image_path = IMAGE_DIR / image_name

        image = preprocess_image(
            image_path,
            image_size=self.image_size
        )

        label = int(row["label"])

        if self.transform is not None:
            image = self.transform(image)

        return image, label


# --------------------------------------------------
# Load split data
# --------------------------------------------------

def load_split_data():

    dataframe = pd.read_csv(SPLIT_FILE)

    train_df = dataframe[
        dataframe["split"] == "train"
    ].copy()

    validation_df = dataframe[
        dataframe["split"] == "validation"
    ].copy()

    test_df = dataframe[
        dataframe["split"] == "test"
    ].copy()

    return (
        train_df,
        validation_df,
        test_df
    )


# --------------------------------------------------
# Create datasets
# --------------------------------------------------

def create_datasets():

    (
        train_df,
        validation_df,
        test_df
    ) = load_split_data()

    train_dataset = ELPVDataset(
        train_df,
        transform=train_transform
    )

    validation_dataset = ELPVDataset(
        validation_df,
        transform=eval_transform
    )

    test_dataset = ELPVDataset(
        test_df,
        transform=eval_transform
    )

    return (
        train_dataset,
        validation_dataset,
        test_dataset
    )


# --------------------------------------------------
# Test the pipeline
# --------------------------------------------------

if __name__ == "__main__":

    print("=" * 60)
    print("ELPV PREPROCESSING PIPELINE TEST")
    print("=" * 60)

    train_dataset, validation_dataset, test_dataset = (
        create_datasets()
    )

    print("\nDataset sizes:")
    print("Training:", len(train_dataset))
    print("Validation:", len(validation_dataset))
    print("Test:", len(test_dataset))

    # Test one training image
    image, label = train_dataset[0]

    print("\nSample training image:")
    print("Tensor shape:", image.shape)
    print("Label:", label)
    print("Data type:", image.dtype)

    # Test one validation image
    image_val, label_val = validation_dataset[0]

    print("\nSample validation image:")
    print("Tensor shape:", image_val.shape)
    print("Label:", label_val)

    print("\n" + "=" * 60)
    print("PREPROCESSING PIPELINE TEST COMPLETED")
    print("=" * 60)