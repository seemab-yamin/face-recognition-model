"""
Modular Image Classifier - Training Script
"""

import os

from utils.datasets import make_dataloaders
from utils.utils import parse_args_with_defaults, set_seed
from model_factory import make_model

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


# ============================================================
# BLOCK 3: MODEL CREATION
# ============================================================
def create_model(args, num_classes, device):
    """Create model, optimizer, and criterion."""
    import torch.nn as nn

    # load model architecture
    model = make_model(
        arch=args.arch,
        num_classes=num_classes,
        pretrained=args.pre_trained,
        freeze_backbone=args.freeze_backbone,
        device=device,
    )

    print(f"Selected model: {args.arch} with pretrained={args.pre_trained}")
    print(f"Created model:\n{str(model)}")

    # Select trainable parameters
    if args.pre_trained and args.freeze_backbone:
        update_params = [p for p in model.parameters() if p.requires_grad]
    else:
        update_params = model.parameters()

    if args.use_weight_decay:
        from torch.optim import AdamW

        print("Using weight decay for optimizer")
        optimizer = AdamW(
            params=update_params,
            lr=args.learning_rate,
            weight_decay=args.use_weight_decay,
        )
    else:
        from torch.optim import Adam

        optimizer = Adam(params=update_params, lr=args.learning_rate)

    if args.use_scheduler:
        from torch.optim.lr_scheduler import CosineAnnealingLR

        scheduler = CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.use_scheduler
        )
        print(f"Using CosineAnnealingLR scheduler with eta_min={args.use_scheduler}")
    else:
        scheduler = None

    criterion = nn.CrossEntropyLoss()
    return model, optimizer, criterion, scheduler


def main():
    """Main training function."""
    import torch

    # 1. Setup
    args = setup()
    print_config(args)

    # 2. Load data
    train_loader, val_loader, info = load_data(args)

    # 3. Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")

    # 4. Create model
    model, optimizer, criterion, scheduler = create_model(
        args, info.num_classes, device=device
    )


if __name__ == "__main__":
    main()
