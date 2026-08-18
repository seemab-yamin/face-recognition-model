import argparse
import os

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from data.face_recognition_dataset import (
    LFWKaggleDataset,
    WebFaceKaggleDataset,
    prepare_casia_webface,
    prepare_lfw,
)
from utils.utils import seed_worker

# Dataset-specific configurations
DATASET_CONFIGS = {
    "casia-webface": {
        "prepare_function": prepare_casia_webface,
        "class": WebFaceKaggleDataset,
        "num_classes": 0,
        "class_names": "",
        "mean": (0.5202712416648865, 0.40445297956466675, 0.3465300500392914),
        "std": (0.28148382902145386, 0.24436739087104797, 0.23586858808994293),
        "train_kwargs": {"split": "train"},
        "test_kwargs": {"split": "val"},
    },
    "lfw": {
        "prepare_function": prepare_lfw,
        "class": LFWKaggleDataset,
        "num_classes": 0,
        "class_names": "",
        "mean": (0.464, 0.395, 0.347),
        "std": (0.252, 0.231, 0.223),
        "train_kwargs": {"split": "train"},
        "test_kwargs": {"split": "val"},
    },
}

# Model-specific requirements
MODEL_REQUIREMENTS = {
    "resnet18": {
        "input_shape": (224, 224),
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
    },
    "resnet50": {
        "input_shape": (224, 224),
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
    },
}


class DatasetInfo:
    def __init__(self, num_classes, input_shape, mean, std, class_names):
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.mean = mean
        self.std = std
        self.class_names = class_names


def make_dataloaders(
    dataset_name: str,
    seed: int = 42,
    batch_size: int = 32,
    num_workers: int = 4,
    data_dir: str = "./data",
    augment: bool = True,
    normalization: str = "dataset",
    arch: str = "resnet18",
    drop_last: bool = True,
):
    os.makedirs(data_dir, exist_ok=True)
    dataset_name = dataset_name.lower()

    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    if arch not in MODEL_REQUIREMENTS:
        raise ValueError(f"Unsupported architecture: {arch}")
    model_config = MODEL_REQUIREMENTS[arch]
    resize_size = model_config["input_shape"]

    dataset_config = DATASET_CONFIGS[dataset_name]
    if normalization not in {"dataset", "model"}:
        raise ValueError(
            f"Unsupported normalization: {normalization}. "
            "Expected 'dataset' or 'model'."
        )
    if normalization == "dataset":
        mean, std = dataset_config["mean"], dataset_config["std"]
    else:
        mean, std = model_config["mean"], model_config["std"]
    # Prepare the dataset if a prepare function is provided
    if dataset_config.get("prepare_function"):
        dataset_config["prepare_function"](data_dir)
        print(f"Dataset prepared: {dataset_name}")

    train_transforms = [
        transforms.Resize(resize_size),
        transforms.ToTensor(),
    ]

    if augment:
        train_transforms = [transforms.RandomHorizontalFlip()] + train_transforms
    train_transforms = transforms.Compose(
        [
            *train_transforms,
            transforms.Normalize(mean, std),
        ]
    )
    val_transforms = transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    dataset_class = dataset_config["class"]
    train_dataset_kwargs = {**dataset_config["train_kwargs"], "root": data_dir}
    test_dataset_kwargs = {**dataset_config["test_kwargs"], "root": data_dir}

    train_dataset = dataset_class(**train_dataset_kwargs, transform=train_transforms)
    val_dataset = dataset_class(**test_dataset_kwargs, transform=val_transforms)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataloader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": (num_workers > 0 and device == "cuda"),
        "drop_last": drop_last,
        "worker_init_fn": seed_worker,
        "generator": torch.Generator().manual_seed(seed),
    }

    num_classes = len(train_dataset.classes)
    class_names = train_dataset.classes

    train_loader = DataLoader(train_dataset, shuffle=True, **dataloader_kwargs)
    # remove drop_last from dataloader_kwargs after creating train_loader to avoid affecting val_loader
    dataloader_kwargs.pop("drop_last", None)
    val_loader = DataLoader(val_dataset, shuffle=False, **dataloader_kwargs)

    info = DatasetInfo(
        num_classes=num_classes,
        input_shape=resize_size,
        mean=mean,
        std=std,
        class_names=class_names,
    )

    # Single summary line
    print(
        f"Loaded {info.num_classes} classes, {len(train_loader.dataset)} train, {len(val_loader.dataset)} val samples"
    )
    return train_loader, val_loader, info


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default="lfw", help="Name of the dataset"
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--drop_last", action="store_true")
    parser.add_argument(
        "--normalization", type=str, default="model", choices=["dataset", "model"]
    )
    args = parser.parse_args()

    train_loader, val_loader, info = make_dataloaders(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment=args.augment,
        normalization=args.normalization,
        arch="resnet18",
        data_dir=args.data_dir,
        drop_last=args.drop_last,
    )

    # Quick validation check
    batch = next(iter(train_loader))
    print(
        f"Batch: images {batch[0].shape}, labels {batch[1].shape}, range {batch[1].min()}-{batch[1].max()}"
    )
