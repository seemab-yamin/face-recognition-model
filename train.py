"""
Face Recognition Model - Training Script
"""

import os
import time
import torch

from data import make_dataloaders
from losses import ArcFace
from model_factory import make_model
from utils import print_config, setup


def compute_grad_norm(model):
    """Compute gradient norm across all parameters."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm**0.5


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

    model = make_model(
        arch=args.arch,
        num_classes=num_classes,
        pretrained=args.pre_trained,
        freeze_backbone=args.freeze_backbone,
        return_embeddings=args.return_embeddings,
        device=device,
    )

    print(f"Selected model: {args.arch} with pretrained={args.pre_trained}")

    if not args.freeze_backbone:
        for param in model.parameters():
            param.requires_grad = True

    if args.use_arcface:
        arcface = ArcFace(
            in_features=model.in_features,
            num_classes=num_classes,
            s=args.arcface_s,
            m=args.arcface_m,
        ).to(device)
        print(f"ArcFace enabled: s={args.arcface_s}, m={args.arcface_m}")
    else:
        arcface = None
        print("ArcFace disabled — using standard Cross-Entropy")

    if args.pre_trained and args.freeze_backbone:
        if args.use_arcface:
            update_params = list(arcface.parameters())
        else:
            update_params = [p for p in model.parameters() if p.requires_grad]
            if len(update_params) == 0:
                raise ValueError(
                    "No trainable parameters found! "
                    "Ensure freeze_backbone=False or enable ArcFace."
                )
    else:
        if args.use_arcface:
            update_params = list(model.parameters()) + list(arcface.parameters())
        else:
            update_params = list(model.parameters())

    total_trainable = sum(p.numel() for p in update_params)
    print(f"Trainable parameters: {total_trainable:,}")

    if args.use_weight_decay:
        from torch.optim import AdamW

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

    if use_amp:
        from torch.cuda.amp import GradScaler, autocast

        scaler = GradScaler()

    epoch_start = time.time()
    model.train()
    batch_times = []
    epoch_loss = 0.0
    num_batches = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        batch_start = time.time()
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        if use_amp:
            with autocast():
                embeddings = model(images)
                logits = (
                    arcface(embeddings, labels) if arcface is not None else embeddings
                )
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()

            if use_grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=use_grad_clip
                )

            scaler.step(optimizer)
            scaler.update()
        else:
            embeddings = model(images)
            logits = arcface(embeddings, labels) if arcface is not None else embeddings
            loss = criterion(logits, labels)
            loss.backward()

            if use_grad_clip:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=use_grad_clip
                )

            optimizer.step()

        batch_time = time.time() - batch_start
        batch_times.append(batch_time)
        epoch_loss += loss.item()
        num_batches += 1

        if batch_idx % 50 == 0:
            grad_norm = compute_grad_norm(model)
            print(
                f"  Batch {batch_idx}: loss={loss.item():.4f}, grad_norm={grad_norm:.4f}"
            )

    epoch_time = time.time() - epoch_start
    avg_batch = sum(batch_times) / len(batch_times) if batch_times else 0
    avg_loss = epoch_loss / num_batches if num_batches > 0 else 0

    return epoch_time, avg_batch, avg_loss


# ============================================================
# BLOCK 5: VALIDATE
# ============================================================
def validate(model, val_loader, device):
    model.eval()
    val_start = time.time()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_time = time.time() - val_start
    val_acc = (
        100 * sum(1 for p, l in zip(all_preds, all_labels) if p == l) / len(all_labels)
    )
    return all_preds, all_labels, val_time, val_acc


# ============================================================
# BLOCK 6: REPORTING
# ============================================================
def print_epoch_report(
    epoch, epoch_time, avg_batch, avg_loss, val_time, val_acc, device
):
    print(f"\n{'=' * 60}")
    print(f"Epoch {epoch + 1}:")
    print(f"  Train loss: {avg_loss:.4f}")
    print(f"  Train time: {epoch_time:.2f}s ({avg_batch * 1000:.1f}ms/batch)")
    print(f"  Val time: {val_time:.2f}s")
    print(f"  Val Acc: {val_acc:.2f}%")
    if device == "cuda":
        print(f"  GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f}GB")
    print(f"{'=' * 60}\n")


# ============================================================
# MAIN
# ============================================================
def main():
    args = setup()
    print_config(args)

    train_loader, val_loader, info = load_data(args)

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}\n")

    checkpoint_dir = os.path.join(args.artifacts_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(
        checkpoint_dir,
        f"{args.arch}_{args.dataset}_best.pth",
    )

    model, optimizer, criterion, arcface, scheduler = create_model(
        args, info.num_classes, device=device
    )

    start_epoch = 0
    max_val_acc = 0.0

    if args.resume and os.path.exists(checkpoint_path):
        print(f"🔄 Resuming from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        arcface.load_state_dict(checkpoint["arcface_state_dict"])
        start_epoch = checkpoint["epoch"]
        max_val_acc = checkpoint.get("val_acc", 0.0)
        print(f"   Resumed epoch {start_epoch}, best val_acc: {max_val_acc:.4f}")
    else:
        if args.resume:
            print(f"⚠️ No checkpoint at {checkpoint_path}. Starting fresh.")

    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        epoch_time, avg_batch, avg_loss = train_epoch(
            model,
            criterion,
            arcface,
            optimizer,
            train_loader,
            args.use_amp,
            device,
            args.use_grad_clip,
        )

        _, _, val_time, val_acc = validate(model, val_loader, device)
        print_epoch_report(
            epoch, epoch_time, avg_batch, avg_loss, val_time, val_acc, device
        )

        if scheduler:
            scheduler.step()

        if args.save_best and val_acc > max_val_acc:
            max_val_acc = val_acc
            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "args": vars(args),
            }
            if arcface is not None:
                checkpoint["arcface_state_dict"] = arcface.state_dict()
            if scheduler is not None:
                checkpoint["scheduler_state_dict"] = scheduler.state_dict()
            torch.save(checkpoint, checkpoint_path)
            print(f"✅ Best checkpoint saved (val_acc: {val_acc:.4f})")

        if args.save_every and (epoch + 1) % args.save_every == 0:
            periodic_path = checkpoint_path.replace("_best", f"_epoch{epoch + 1}")
            torch.save(checkpoint, periodic_path)
            print(f"📁 Periodic checkpoint: {periodic_path}")

    print("\n✅ Training complete!")


if __name__ == "__main__":
    main()
