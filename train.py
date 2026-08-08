"""
Modular Image Classifier - Training Script
"""

import os

from utils.datasets import make_dataloaders
from utils.utils import parse_args_with_defaults, set_seed


# ============================================================
# BLOCK 1: CONFIGURATION & SETUP
# ============================================================
def setup():
    """Parse arguments, set seed, and create directories."""

    args = parse_args_with_defaults()
    set_seed(args.seed)

    # Create results directories
    os.makedirs(args.artifacts_dir, exist_ok=True)
    os.makedirs(os.path.join(args.artifacts_dir, "metrics"), exist_ok=True)
    os.makedirs(os.path.join(args.artifacts_dir, "exported_models"), exist_ok=True)
    os.makedirs(os.path.join(args.artifacts_dir, "checkpoints"), exist_ok=True)
    return args


def print_config(args):
    """Print configuration."""
    print("\n" + "=" * 60)
    print("CONFIGURATION")
    print("=" * 60)
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    print("=" * 60 + "\n")


# ============================================================
# BLOCK 2: DATA LOADING
# ============================================================
def load_data(args):
    """Create dataloaders and get dataset info."""

    train_loader, val_loader, info = make_dataloaders(
        dataset_name=args.dataset,
        seed=args.seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        data_dir=args.data_dir,
    )
    print(f"Created dataloaders with {info.num_classes} classes")
    print(f"Batch shape: {next(iter(train_loader))[0].shape}")
    return train_loader, val_loader, info


def main():
    """Main training function."""
    args = setup()
    print_config(args)
    train_loader, val_loader, info = load_data(args)

    print(f"Training on dataset: {args.dataset} with {info.num_classes} classes")


if __name__ == "__main__":
    main()
