import argparse
import os

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from utils.download_datasets import prepare_casia_webface
from utils.utils import seed_worker

# Dataset-specific configurations
DATASET_CONFIGS = {
    "casia-webface": {
        "prepare_function": prepare_casia_webface,
        "data_folder": "webface_112x112",
        "input_shape": (112, 122, 3),
        "mean": (0.5, 0.5, 0.5),
        "std": (0.5, 0.5, 0.5),
        "train_kwargs": {"train": True},
        "test_kwargs": {"train": False},
    }
}


class DatasetInfo:
    """Container for dataset metadata."""

    def __init__(self, num_classes, input_shape, mean, std, class_names):
        self.num_classes = num_classes
        self.input_shape = input_shape  # (C, H, W)
        self.mean = mean  # for normalization
        self.std = std  # for normalization
        self.class_names = class_names


class FaceRecognitionDataset(Dataset):
    def __init__(self, root_dir, transforms=None):
        self.root_dir = root_dir
        self.transforms = transforms
        self.image_paths = []
        self.labels = []

        # map class names to indices
        self.classes = sorted(
            entry.name for entry in os.scandir(root_dir) if entry.is_dir()
        )
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        # populate image_paths and labels
        for cls_name in self.classes:
            cls_dir = os.path.join(root_dir, cls_name)
            for entry in os.scandir(cls_dir):
                if entry.is_file() and entry.name.lower().endswith(
                    (".png", ".jpg", ".jpeg")
                ):
                    self.image_paths.append(entry.path)
                    self.labels.append(self.class_to_idx[cls_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load image
        from PIL import Image

        image = Image.open(image_path).convert("RGB")

        if self.transforms:
            image = self.transforms(image)

        return image, torch.tensor(label, dtype=torch.long)


def make_dataloaders(
    dataset_name: str,
    seed: int = 42,
    batch_size: int = 32,
    num_workers: int = 4,
    data_dir: str = "./data",
    augment: bool = True,
):
    """
    dataset factory to load datasets
    """
    dataset_name = dataset_name.lower()

    # Get dataset config
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    dataset_config = DATASET_CONFIGS[dataset_name]
    input_shape = dataset_config["input_shape"]
    mean, std = dataset_config["mean"], dataset_config["std"]

    dataset_config["prepare_function"](data_dir)
    print(f"Prepared dataset {dataset_name} in {data_dir}")

    # build transforms
    train_transforms = [transforms.ToTensor()]
    if augment:
        train_transforms = [transforms.RandomHorizontalFlip()] + train_transforms

    train_transform = transforms.Compose(
        [
            *train_transforms,
            transforms.Normalize(mean, std),
        ]
    )

    val_transforms = [transforms.ToTensor()]
    val_transform = transforms.Compose(
        [
            *val_transforms,
            transforms.Normalize(mean, std),
        ]
    )

    # Get dataset class using folder names in data_dir
    data_folder = os.path.join(data_dir, dataset_config["data_folder"])
    train_dir = os.path.join(data_folder, "train")
    val_dir = os.path.join(data_folder, "val")

    train_dataset = FaceRecognitionDataset(
        root_dir=train_dir, transforms=train_transform
    )
    val_dataset = FaceRecognitionDataset(root_dir=val_dir, transforms=val_transform)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataloader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": (num_workers > 0)
        and (device == "cuda"),  # default, will be set to True if device is CUDA
        "drop_last": True,  # drop last incomplete batch
        "worker_init_fn": seed_worker,
        "generator": torch.Generator().manual_seed(seed),  # for reproducibility
    }

    # read classes from the dataset directory
    dataset_config["num_classes"] = train_dataset.classes.__len__()
    dataset_config["class_names"] = train_dataset.classes

    # Build dataloaders
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **dataloader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **dataloader_kwargs,
    )
    if device == "cuda":
        print(
            f"Using CUDA with {num_workers} workers and pin_memory={dataloader_kwargs.get('pin_memory', False)}"
        )

    # Create info object
    info = DatasetInfo(
        num_classes=dataset_config["num_classes"],
        input_shape=input_shape,
        mean=mean,
        std=std,
        class_names=dataset_config["class_names"],
    )
    return train_loader, val_loader, info


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="Name of the dataset")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of workers")
    parser.add_argument("--augment", action="store_true", help="Enable augmentation")
    parser.add_argument(
        "--data-dir", type=str, default="./data", help="Data directory for the dataset"
    )
    args = parser.parse_args()

    train_loader, val_loader, info = make_dataloaders(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment=args.augment,
        data_dir=args.data_dir,
    )

    print(f"Dataset: {args.dataset}")
    print(f"Classes: {info.num_classes}")
    print(f"Input shape: {info.input_shape}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
