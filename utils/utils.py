import argparse
import os
import random
import sys

import numpy as np
import torch
import yaml


def check_tensor(tensor, name):
    """Check for NaN/Inf in a tensor (silent by default)."""

    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        print(f"⚠️ Invalid values detected in {name}")


def setup():
    """Parse arguments, set seed, and create directories."""

    args = parse_args_with_defaults()
    set_seed(args.seed)

    os.makedirs(args.artifacts_dir, exist_ok=True)
    os.makedirs(os.path.join(args.artifacts_dir, "metrics"), exist_ok=True)
    os.makedirs(os.path.join(args.artifacts_dir, "exported_models"), exist_ok=True)
    os.makedirs(os.path.join(args.artifacts_dir, "checkpoints"), exist_ok=True)
    return args


def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"⚠️ Config '{config_path}' not found. Using defaults.")
        return {}
    except yaml.YAMLError as e:
        print(f"❌ Error parsing config: {e}")
        sys.exit(1)


def parse_args_with_defaults():
    """Parse CLI arguments with defaults from YAML config."""

    parser = argparse.ArgumentParser(description="Face Recognition Training")

    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file",
    )

    known_args, _ = parser.parse_known_args()
    defaults = load_config(known_args.config)

    # Core
    parser.add_argument("--arch", type=str, default=defaults.get("arch", "resnet18"))
    parser.add_argument("--seed", type=int, default=defaults.get("seed", 42))
    parser.add_argument("--epochs", type=int, default=defaults.get("epochs", 10))
    parser.add_argument("--dataset", type=str, default=defaults.get("dataset", "model"))
    parser.add_argument(
        "--normalization",
        type=str,
        default=defaults.get("normalization", "dataset"),
        choices=["dataset", "model"],
    )
    parser.add_argument(
        "--batch-size", type=int, default=defaults.get("batch_size", 32)
    )
    parser.add_argument(
        "--learning-rate", type=float, default=defaults.get("learning_rate", 0.001)
    )
    parser.add_argument(
        "--num-workers", type=int, default=defaults.get("num_workers", 4)
    )
    parser.add_argument(
        "--data-dir", type=str, default=defaults.get("data_dir", "./data")
    )
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default=defaults.get("artifacts_dir", "./artifacts"),
    )

    # Flags (store_true)
    bool_args = [
        ("--return-embeddings", "return_embeddings"),
        ("--augment", "augment"),
        ("--drop-last", "drop_last"),
        ("--freeze-backbone", "freeze_backbone"),
        ("--pre-trained", "pre_trained"),
        ("--use-amp", "use_amp"),
        ("--use-weight-decay", "use_weight_decay"),
        ("--save-best", "save_best"),
        ("--resume", "resume"),
        ("--use-arcface", "use_arcface"),
    ]
    for flag, dest in bool_args:
        parser.add_argument(
            flag,
            action="store_true",
            default=defaults.get(dest, False),
        )

    # Float args
    float_args = [
        ("--use-scheduler", "use_scheduler", 1e-6),
        ("--use-grad-clip", "use_grad_clip", 1.0),
        ("--arcface-m", "arcface_m", 0.2),
        ("--arcface-s", "arcface_s", 64.0),
    ]
    for flag, dest, default in float_args:
        parser.add_argument(
            flag,
            type=float,
            default=defaults.get(dest, default),
        )

    # Int args
    parser.add_argument(
        "--save-every",
        type=int,
        default=defaults.get("save_every", 0),
        help="Save checkpoint every N epochs (0 = disabled)",
    )

    args = parser.parse_args()
    args.config_path = known_args.config
    return args


def print_config(args):
    """Print configuration in a readable format."""
    print("\n" + "=" * 60)
    print("CONFIGURATION")
    print("=" * 60)
    for key, value in vars(args).items():
        print(f"  {key:20s}: {value}")
    print("=" * 60 + "\n")


def set_seed(seed: int = 42, deterministic: bool = True):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def seed_worker(worker_id):
    """Ensure DataLoader workers have reproducible seeds."""

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


if __name__ == "__main__":
    args = parse_args_with_defaults()
    print_config(args)
    set_seed(args.seed)
