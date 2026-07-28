"""
dataset.py — Bean crop dataset, splitting, and augmentation transforms.
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import torchvision.transforms as T
import torch

CROP_SIZE = 224
CROP_PAD  = 8     # pixel padding added around each annotated bbox


class BeanCropDataset(Dataset):
    """
    Bean-quality binary classification dataset.
    Each sample is a crop from a tray image, labeled Good (0) or Bad (1).
    """

    def __init__(self, records: pd.DataFrame, crops: list, transform=None):
        self.records   = records.reset_index(drop=True)
        self.crops     = crops
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        crop  = self.crops[idx]
        label = int(self.records.loc[idx, "label"])
        if self.transform:
            crop = self.transform(crop)
        return crop, label


def load_annotated_crops(image_dir: Path, csv_path: Path):
    """
    Load per-bean crops from exported LabelMe annotations.

    CSV columns: image_stem, x1, y1, x2, y2, label, tray_id, session_date, variety

    Returns:
        meta  — DataFrame with one row per crop (no pixel data)
        crops — list of uint8 numpy arrays, shape (CROP_SIZE, CROP_SIZE, 3)
    """
    df = pd.read_csv(csv_path)
    records, crops = [], []

    for _, row in df.iterrows():
        # Load the mask overlay image (LabelMe coloured-polygon export)
        # rather than the original RGB photo. The CROP_PAD border will
        # capture the polygon outline that LabelMe draws on the overlay,
        # providing boundary context to help locate each annotation.
        img_path = (Path(image_dir) / "mask_overlays"
                    / f"{row['image_stem']}_overlay.jpg")
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x1, y1, x2, y2 = (int(row.x1), int(row.y1),
                           int(row.x2), int(row.y2))
        H, W = img.shape[:2]
        crop = img[max(0, y1 - CROP_PAD): min(H, y2 + CROP_PAD),
                   max(0, x1 - CROP_PAD): min(W, x2 + CROP_PAD)]
        crop = cv2.resize(crop, (CROP_SIZE, CROP_SIZE))

        records.append({
            "label":        int(row.label),
            "tray_id":      row.tray_id,
            "session_date": row.session_date,
            "variety":      row.variety,
        })
        crops.append(crop)

    return pd.DataFrame(records), crops


def make_splits(meta: pd.DataFrame, crops: list):
    """
    Split crops into train and validation partitions.

    Returns:
        train_meta, val_meta, train_crops, val_crops
    """
    # Split at the crop level so every bean has an equal chance of
    # landing in either partition.
    train_idx, val_idx = train_test_split(
        meta.index, test_size=0.2,
        stratify=meta["label"], random_state=42
    )
    train_meta  = meta.loc[train_idx].reset_index(drop=True)
    val_meta    = meta.loc[val_idx].reset_index(drop=True)
    train_crops = [crops[i] for i in train_idx]
    val_crops   = [crops[i] for i in val_idx]
    return train_meta, val_meta, train_crops, val_crops


def get_train_transforms() -> T.Compose:
    return T.Compose([
        T.ToPILImage(),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.3),
        T.RandomRotation(degrees=20),
        T.ColorJitter(brightness=0.35, contrast=0.30, saturation=0.25),
        T.RandomResizedCrop(CROP_SIZE, scale=(0.7, 1.0)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])


def get_val_transforms() -> T.Compose:
    """Deterministic transforms — used at validation and inference."""
    return T.Compose([
        T.ToPILImage(),
        T.Resize(256),
        T.CenterCrop(CROP_SIZE),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])
