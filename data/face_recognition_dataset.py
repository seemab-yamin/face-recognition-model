import os
import random
import shutil

import torch
from torch.utils.data import Dataset


def prepare_casia_webface(data_dir, train_val_split_ratio=0.8):
    """
    Download CASIA-WebFace if necessary and split identity directories
    into train and validation sets.

    Args:
        data_dir (str): Directory where the dataset should be stored.
        train_val_split_ratio (float): Fraction of identities assigned
            to the training split.
    """
    if not 0 < train_val_split_ratio < 1:
        raise ValueError("train_val_split_ratio must be between 0 and 1.")

    os.makedirs(data_dir, exist_ok=True)

    dataset_dir = os.path.join(data_dir, "webface_112x112")

    # check folder exist and has train and val sub folders with some files.
    if (
        not os.path.exists(dataset_dir)
        or not os.path.exists(os.path.join(dataset_dir, "train"))
        or not os.path.exists(os.path.join(dataset_dir, "val"))
    ):
        print(f"Downloading and preparing CASIA-WebFace dataset in {dataset_dir}...")

        os.system(
            f"""curl -C - -L -o {data_dir}/webface_112x112.zip https://www.kaggle.com/api/v1/datasets/download/yakhyokhuja/webface-112x112"""
        )

        os.system(f"unzip -q -o {data_dir}/webface_112x112.zip -d {data_dir}")

    train_dir = os.path.join(dataset_dir, "train")
    val_dir = os.path.join(dataset_dir, "val")

    # validate if the val and train folders exist
    if not os.path.exists(train_dir) or not os.path.exists(val_dir):

        # Get all unique identity folders
        all_dirs_path = [
            os.path.join(dataset_dir, f)
            for f in os.listdir(dataset_dir)
            if os.path.isdir(os.path.join(dataset_dir, f)) and not f.startswith(".")
        ]

        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        random.shuffle(all_dirs_path)

        train_size = int(train_val_split_ratio * len(all_dirs_path))

        # create deep copy of all_dirs_path for validation directories
        val_dirs_path = all_dirs_path[train_size:].copy()

        # move validation files to val_dir
        for src in val_dirs_path:
            if os.path.exists(src):
                dst = os.path.join(
                    val_dir,
                    os.path.basename(src),
                )
                shutil.move(src, dst)
                all_dirs_path.remove(src)

        # Move remaining directories to train_dir
        for src in all_dirs_path:
            if os.path.exists(src):
                dst = os.path.join(
                    train_dir,
                    os.path.basename(src),
                )
                shutil.move(src, dst)


class FaceRecognitionDataset(Dataset):
    """
    Dataset for loading images organized by identity.

    Expected directory structure:

        root/
        ├── person_001/
        │   ├── image1.jpg
        │   ├── image2.jpg
        │   └── ...
        ├── person_002/
        │   ├── image1.jpg
        │   └── ...
        └── ...

    Args:
        split (str):
            Dataset split identifier. Supported values are ``train``
            and ``val``. The value is stored for dataset metadata and
            does not change directory discovery behavior.

        root (str or pathlib.Path):
            Root directory containing identity directories.

        transform (callable, optional):
            Optional transform applied to each PIL image.
    """

    VALID_SPLITS = {"train", "val"}
    base_folder = "webface_112x112"

    def __init__(self, split, root, transform=None):
        if split not in self.VALID_SPLITS:
            raise ValueError(
                f"Unsupported split '{split}'. "
                f"Expected one of {sorted(self.VALID_SPLITS)}."
            )

        if not os.path.isdir(root):
            raise FileNotFoundError(f"Dataset directory does not exist: {root}")

        self.split_root = os.path.join(root, self.base_folder, split)
        self.transform = transform
        self.split = split
        self.image_paths = []
        self.labels = []

        if not os.path.isdir(self.split_root):
            raise FileNotFoundError(
                f"Dataset split directory does not exist: {self.split_root}"
            )

        self.classes = sorted(
            entry.name for entry in os.scandir(self.split_root) if entry.is_dir()
        )

        if not self.classes:
            raise RuntimeError(
                f"No class directories found in dataset: {self.split_root}"
            )

        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        for cls_name in self.classes:
            cls_dir = os.path.join(self.split_root, cls_name)
            for entry in os.scandir(cls_dir):
                if entry.is_file() and entry.name.lower().endswith(
                    (".png", ".jpg", ".jpeg")
                ):
                    self.image_paths.append(entry.path)
                    self.labels.append(self.class_to_idx[cls_name])

        if not self.image_paths:
            raise RuntimeError(f"No images found in dataset: {self.split_root}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        from PIL import Image

        image_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(
            label,
            dtype=torch.long,
        )
