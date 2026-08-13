import argparse
import os

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from utils.download_datasets import prepare_casia_webface
from utils.utils import seed_worker

# dummy empty lambda function for prepare_casia_sample_webface, since we don't have a sample dataset to prepare
prepare_casia_sample_webface = lambda x: x

# Dataset-specific configurations
DATASET_CONFIGS = {
    "casia-webface": {
        "prepare_function": prepare_casia_webface,
        "data_folder": "webface_112x112",
        "mean": (0.5202712416648865, 0.40445297956466675, 0.3465300500392914),
        "std": (0.28148382902145386, 0.24436739087104797, 0.23586858808994293),
    },
    "casia-webface-sample": {
        "prepare_function": prepare_casia_sample_webface,
        "data_folder": "webface_112x112_sample",
        "mean": (0.5202712416648865, 0.40445297956466675, 0.3465300500392914),
        "std": (0.28148382902145386, 0.24436739087104797, 0.23586858808994293),
    },
}

# Model-specific requirements
MODEL_REQUIREMENTS = {
    "resnet": {
        "input_shape": (224, 224, 3),
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
        "needs_resize": True,
    },
}


class DatasetInfo:
    def __init__(self, num_classes, input_shape, mean, std, class_names):
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.mean = mean
        self.std = std
        self.class_names = class_names


class FaceRecognitionDataset(Dataset):
    def __init__(self, root_dir, transforms=None):
        self.root_dir = root_dir
        self.transforms = transforms
        self.image_paths = []
        self.labels = []

        self.classes = sorted(
            entry.name for entry in os.scandir(root_dir) if entry.is_dir()
        )
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

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
        from PIL import Image

        image_path = self.image_paths[idx]
        label = self.labels[idx]
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
    training_from_scratch: bool = True,
    arch: str = "resnet",
    drop_last: bool = True,
):
    os.makedirs(data_dir, exist_ok=True)
    dataset_name = dataset_name.lower()

    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    dataset_config = DATASET_CONFIGS[dataset_name]
    mean, std = dataset_config["mean"], dataset_config["std"]

    model_config = MODEL_REQUIREMENTS[arch]
    input_shape = model_config["input_shape"]

    if training_from_scratch:
        print(f"Using dataset normalization: mean={mean}, std={std}")
    else:
        mean, std = model_config["mean"], model_config["std"]
        print(f"Using model normalization: mean={mean}, std={std}")

    dataset_config["prepare_function"](data_dir)
    print(f"Dataset prepared: {dataset_name}")

    resize_size = (input_shape[0], input_shape[1])
    train_transforms = [transforms.ToTensor(), transforms.Resize(resize_size)]

    if augment:
        train_transforms = [transforms.RandomHorizontalFlip()] + train_transforms

    train_transform = transforms.Compose(
        [
            *train_transforms,
            transforms.Normalize(mean, std),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    data_folder = os.path.join(data_dir, dataset_config["data_folder"])
    train_dir = os.path.join(data_folder, "train")
    val_dir = os.path.join(data_folder, "val")

    train_dataset = FaceRecognitionDataset(train_dir, train_transform)
    val_dataset = FaceRecognitionDataset(val_dir, val_transform)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataloader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": (num_workers > 0 and device == "cuda"),
        "drop_last": drop_last,
        "worker_init_fn": seed_worker,
        "generator": torch.Generator().manual_seed(seed),
    }

    dataset_config["num_classes"] = len(train_dataset.classes)
    dataset_config["class_names"] = train_dataset.classes

    train_loader = DataLoader(train_dataset, shuffle=True, **dataloader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **dataloader_kwargs)

    info = DatasetInfo(
        num_classes=dataset_config["num_classes"],
        input_shape=input_shape,
        mean=mean,
        std=std,
        class_names=dataset_config["class_names"],
    )

    # Single summary line
    print(
        f"Loaded {info.num_classes} classes, {len(train_loader.dataset)} train, {len(val_loader.dataset)} val samples"
    )
    return train_loader, val_loader, info


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="casia-webface-sample")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--drop_last", action="store_true")
    args = parser.parse_args()

    train_loader, val_loader, info = make_dataloaders(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment=args.augment,
        training_from_scratch=True,
        arch="resnet",
        data_dir=args.data_dir,
        drop_last=args.drop_last,
    )

    # Quick validation check
    batch = next(iter(train_loader))
    print(
        f"Batch: images {batch[0].shape}, labels {batch[1].shape}, range {batch[1].min()}-{batch[1].max()}"
    )
