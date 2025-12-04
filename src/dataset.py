"""
Dataset and dataloader utilities for the Fish Segmentation project.

This module defines:
- A reproducible random seed helper.
- A dataset class that locates RGB images and their corresponding GT masks
  using a glob-based strategy compatible with the Kaggle fish dataset layout.
- A helper to create train/test DataLoaders.
"""

import os
import random
from glob import glob
from typing import Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T


def set_seed(seed: int = 42) -> None:
    """
    Fix all relevant random seeds to make experiments reproducible.

    Parameters
    ----------
    seed : int
        The seed value to use for Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    print(f"[INFO] Random seed fixed to {seed}")


class FishGlobDataset(Dataset):
    """
    Dataset for the Kaggle Large-Scale Fish Dataset using a glob-based pairing.

    Folder structure this class expects (simplified):

        root_dir/
            ClassA/
                ClassA/            # RGB images
                    001.png
                    002.png
                    ...
                ClassA GT/         # Ground-truth masks
                    001.png
                    002.png
                    ...
            ClassB/
                ClassB/
                ClassB GT/
            ...

    The dataset:
    - Collects all PNGs.
    - Separates mask files by looking for "* GT" directories.
    - For each RGB image, tries to dynamically construct the corresponding
      mask path (ClassName GT / filename). If that fails, falls back to
      the sorted mask list.
    """

    def __init__(self, root_dir: str, img_size: int = 448) -> None:
        """
        Parameters
        ----------
        root_dir : str
            Path to the dataset root (e.g. "fish_dataset/Fish_Dataset/Fish_Dataset").
        img_size : int
            Target size (square) for both images and masks.
        """
        super().__init__()
        self.root_dir = root_dir
        self.img_size = img_size

        # Collect all PNG files under class folders
        all_files = glob(os.path.join(root_dir, "*", "*", "*.png"))

        # Ground-truth masks live in directories that end with " GT"
        self.mask_files = sorted(
            glob(os.path.join(root_dir, "*", "* GT", "*.png"))
        )

        # Images are all PNGs minus the ones that belong to GT folders
        all_set = set(all_files)
        mask_set = set(self.mask_files)
        self.img_files = sorted(list(all_set - mask_set))

        print(f"[INFO] Found {len(self.img_files)} images and {len(self.mask_files)} masks.")

        # Image preprocessing: resize, convert to tensor, and normalize for ViT/DINOv3
        self.transform_img = T.Compose(
            [
                T.Resize((img_size, img_size)),
                T.ToTensor(),
                T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        # Mask preprocessing: nearest-neighbor resize and conversion to single-channel tensor
        self.transform_mask = T.Compose(
            [
                T.Resize((img_size, img_size), interpolation=Image.NEAREST),
                T.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.img_files)

    def __getitem__(self, idx: int):
        """
        Return a (image, mask) pair at the given index.

        The pairing is:
        - Try to construct the mask path by replacing the RGB folder with the
          corresponding "ClassName GT" folder.
        - If that file does not exist (any mismatch in the dataset), fall back to
          pairing by index with the sorted mask list.
        """
        img_path = self.img_files[idx]

        # Example:
        #   img_path     = .../Red Sea Bream/Red Sea Bream/001.png
        #   desired mask = .../Red Sea Bream/Red Sea Bream GT/001.png
        parent = os.path.dirname(img_path)        # .../ClassName/ClassName
        filename = os.path.basename(img_path)     # 001.png
        grandparent = os.path.dirname(parent)     # .../ClassName
        class_name = os.path.basename(parent)     # "Red Sea Bream"

        mask_path = os.path.join(grandparent, f"{class_name} GT", filename)

        if not os.path.exists(mask_path):
            # Fallback: rely on the sorted list alignment
            mask_path = self.mask_files[idx]

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        return self.transform_img(image), self.transform_mask(mask)


def create_dataloaders(
    root_dir: str,
    img_size: int = 448,
    batch_size: int = 8,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build train and test DataLoaders from the fish dataset.

    Parameters
    ----------
    root_dir : str
        Dataset root directory.
    img_size : int
        Square resolution to resize images and masks to.
    batch_size : int
        Batch size for both train and test loaders.
    train_ratio : float
        Fraction of the dataset to use for training; the rest is used for testing.
    seed : int
        Seed used for the random split.

    Returns
    -------
    train_loader : DataLoader
    test_loader : DataLoader
    """
    full_ds = FishGlobDataset(root_dir=root_dir, img_size=img_size)

    train_size = int(train_ratio * len(full_ds))
    test_size = len(full_ds) - train_size

    generator = torch.Generator().manual_seed(seed)
    train_ds, test_ds = random_split(full_ds, [train_size, test_size], generator=generator)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    print(
        f"[INFO] Dataset split into {len(train_ds)} train and {len(test_ds)} test samples."
    )

    return train_loader, test_loader
