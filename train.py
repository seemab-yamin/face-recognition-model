"""
Modular Image Classifier - Training Script
"""

import os

import torch

from model_factory import make_model
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
        augment=args.augment,
        training_from_scratch=args.training_from_scratch,
        arch=args.arch,
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
    print(f"Created model:\n{model}")

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


# ============================================================
# BLOCK 4: TRAINING
# ============================================================
def train_epoch(
    model, criterion, optimizer, train_loader, use_amp, device, use_grad_clip
):
    """Train one epoch and return metrics."""

    import time

    if use_amp:
        from torch.cuda.amp import GradScaler, autocast

        scaler = GradScaler()

    epoch_start = time.time()
    model.train()
    batch_times = []

    for batch_idx, (images, labels) in enumerate(train_loader):
        batch_start = time.time()
        # move to device
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        # forward pass with mixed precision if enabled
        if use_amp:
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            if use_grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=use_grad_clip
                )
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            if use_grad_clip:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=use_grad_clip
                )
            optimizer.step()

        batch_time = time.time() - batch_start
        batch_times.append(batch_time)

        if batch_idx % 10 == 0:
            print(f"  Batch {batch_idx}: total={batch_time * 1000:.1f}ms")

    epoch_time = time.time() - epoch_start
    avg_batch = sum(batch_times) / len(batch_times)

    return epoch_time, avg_batch


# ============================================================
# BLOCK 4: VALIDATE
# ============================================================
def validate(model, val_loader, device):
    import time

    model.eval()
    val_start = time.time()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            # move to device
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)

            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_time = time.time() - val_start
    val_acc = (
        100
        * sum(1 for pred, label in zip(all_preds, all_labels) if pred == label)
        / len(all_labels)
    )
    return all_preds, all_labels, val_time, val_acc


# ============================================================
# BLOCK 6: REPORTING
# ============================================================
def print_epoch_report(epoch, epoch_time, avg_batch, val_time, val_acc, device):
    """Print epoch summary."""

    print(f"\n{'=' * 60}")
    print(f"Epoch {epoch + 1}:")
    print(f"  Train time: {epoch_time:.2f}s ({avg_batch * 1000:.1f}ms/batch)")
    print(f"  Val time: {val_time:.2f}s")
    print(f"  Val Acc: {val_acc:.2f}%")

    if device == "cuda":
        print(
            f"  GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f}GB / "
            f"{torch.cuda.max_memory_allocated() / 1e9:.2f}GB"
        )
    print(f"{'=' * 60}\n")


def main():
    """Main training function."""

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

    # 5. Training Loop
    for epoch in range(args.epochs):
        epoch_time, avg_batch = train_epoch(
            model,
            criterion,
            optimizer,
            train_loader,
            args.use_amp,
            device,
            args.use_grad_clip,
        )

        # validate
        all_preds, all_labels, val_time, val_acc = validate(model, val_loader, device)
        print_epoch_report(epoch, epoch_time, avg_batch, val_time, val_acc, device)

        if scheduler:
            scheduler.step()


if __name__ == "__main__":
    main()
