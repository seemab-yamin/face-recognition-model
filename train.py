"""
Face Recognition Model - Training Script
"""

import os

import torch

from data import make_dataloaders
from losses import ArcFace
from model_factory import make_model
from utils import print_config, setup


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
        return_embeddings=args.return_embeddings,
        device=device,
    )

    print(f"Selected model: {args.arch} with pretrained={args.pre_trained}")
    print(f"Created model:\n{model}")

    # Create ArcFace
    arcface = ArcFace(
        in_features=model.in_features,
        num_classes=num_classes,
        s=args.arcface_s,
        m=args.arcface_m,
    ).to(device)

    # Select trainable parameters
    if args.pre_trained and args.freeze_backbone:
        update_params = list(arcface.parameters())
    else:
        update_params = list(model.parameters()) + list(arcface.parameters())

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

    return model, optimizer, criterion, arcface, scheduler


# ============================================================
# BLOCK 4: TRAINING
# ============================================================
def train_epoch(
    model, criterion, arcface, optimizer, train_loader, use_amp, device, use_grad_clip
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
                embeddings = model(images)  # Get 512-dim feature vectors
                logits = arcface(embeddings, labels)  # Apply ArcFace margin
                # print Training loss
                loss = criterion(logits, labels)  # Standard Cross-Entrop
                print(f"  Batch {batch_idx}: Loss = {loss.item():.4f}")
            scaler.scale(loss).backward()
            # compute gradient norm for logging
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.norm().item() ** 2
            total_norm = total_norm**0.5
            if use_grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=use_grad_clip
                )
            scaler.step(optimizer)
            scaler.update()
        else:
            embeddings = model(images)  # Get 512-dim feature vectors
            logits = arcface(embeddings, labels)  # Apply ArcFace margin
            loss = criterion(logits, labels)  # Standard Cross-Entropy
            print(f"  Batch {batch_idx}: Loss = {loss.item():.4f}")
            loss.backward()
            # compute gradient norm for logging
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.norm().item() ** 2
            total_norm = total_norm**0.5
            if use_grad_clip:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=use_grad_clip
                )
            optimizer.step()

        batch_time = time.time() - batch_start
        batch_times.append(batch_time)

        if batch_idx % 10 == 0:
            print(
                f"  Batch {batch_idx}: loss={loss.item():.4f}, grad_norm={total_norm:.4f}"
            )

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
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}\n")

    # Checkpoint path
    checkpoint_dir = os.path.join(args.artifacts_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(
        checkpoint_dir,
        f"{args.arch}_{args.dataset}_best.pth",
    )

    # 5. Resume or create fresh
    start_epoch = 0
    max_val_acc = 0.0

    # Restore model and optimizer
    model, optimizer, criterion, arcface, scheduler = create_model(
        args, info.num_classes, device=device
    )
    if args.resume and os.path.exists(checkpoint_path):
        print(f"🔄 Resuming from checkpoint: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
        except (FileNotFoundError, RuntimeError) as e:
            print(f"⚠️ Failed to load checkpoint: {e}. Starting fresh.")
            checkpoint = None

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        arcface.load_state_dict(checkpoint["arcface_state_dict"])

        start_epoch = checkpoint["epoch"]
        max_val_acc = checkpoint.get("val_acc", 0.0)

        print(f"   Resumed from epoch {start_epoch} with val_acc: {max_val_acc:.4f}")
    else:
        # 6. Create fresh model
        if args.resume:
            print(f"⚠️ Checkpoint not found at {checkpoint_path}. Starting fresh.")

    # 7. Training Loop
    for epoch in range(start_epoch, args.epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"{'='*50}")

        # Train
        epoch_time, avg_batch = train_epoch(
            model,
            criterion,
            arcface,
            optimizer,
            train_loader,
            args.use_amp,
            device,
            args.use_grad_clip,
        )

        # Validate
        all_preds, all_labels, val_time, val_acc = validate(model, val_loader, device)
        print_epoch_report(epoch, epoch_time, avg_batch, val_time, val_acc, device)

        # Update scheduler
        if scheduler:
            scheduler.step()

        # 8. Save checkpoint (best model only)
        if args.save_best and val_acc > max_val_acc:
            max_val_acc = val_acc
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "arcface_state_dict": arcface.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": (
                        scheduler.state_dict() if scheduler else None
                    ),
                    "val_acc": val_acc,
                    "args": vars(args),
                },
                checkpoint_path,
            )
            print(
                f"✅ Best checkpoint saved: {checkpoint_path} (val_acc: {val_acc:.4f})"
            )

        # Optional: Save periodic checkpoint every N epochs
        if args.save_every and (epoch + 1) % args.save_every == 0:
            periodic_path = checkpoint_path.replace("_best", f"_epoch{epoch+1}")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "arcface_state_dict": arcface.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": (
                        scheduler.state_dict() if scheduler else None
                    ),
                    "val_acc": val_acc,
                    "args": vars(args),
                },
                periodic_path,
            )
            print(f"📁 Periodic checkpoint saved: {periodic_path}")

    print("\n✅ Training complete!")


if __name__ == "__main__":
    main()
