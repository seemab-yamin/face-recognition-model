import argparse
import random
import sys

import numpy as np
import torch
import yaml


def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Warning: Config file '{config_path}' not found. Using defaults.")
        return {}
    except yaml.YAMLError as e:
        print(f"Error parsing config file '{config_path}': {e}")
        sys.exit(1)


def parse_args_with_defaults():
    """
    Parse CLI arguments with defaults from YAML config.
    CLI args override YAML defaults.
    """
    # Set up argument parser
    parser = argparse.ArgumentParser(description="face-recognition-model")

    # Config file path (parsed first, before loading defaults)
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file (default: config.yaml)",
    )

    # Parse only --config first to know which config file to load
    # We use parse_known_args to avoid errors for unknown args
    known_args, _ = parser.parse_known_args()
    config_path = known_args.config

    # Load defaults from YAML config file
    defaults = load_config(config_path)

    # Add all arguments with defaults from config file
    parser.add_argument(
        "--arch",
        type=str,
        default=defaults.get("arch", "custom"),
        help="Model architecture",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=defaults.get("seed", 42),
        help="Reproducibility seed",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=defaults.get("epochs", 10),
        help="Number of epochs to train",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=defaults.get("dataset", "casia-webface"),
        help="Name of the dataset",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=defaults.get("batch_size", 32),
        help="Batch size for dataloaders",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=defaults.get("learning_rate", 0.001),
        help="Learning rate for the optimizer",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=defaults.get("num_workers", 4),
        help="Number of workers for dataloaders",
    )
    parser.add_argument(
        "--use-amp",
        action="store_true",
        default=defaults.get("use_amp", False),
        help="Use automatic mixed precision (AMP) for training",
    )
    parser.add_argument(
        "--use-weight-decay",
        action="store_true",
        default=defaults.get("use_weight_decay", False),
        help="Use weight decay for the optimizer",
    )
    parser.add_argument(
        "--use-scheduler",
        type=float,
        default=defaults.get("use_scheduler", 1e-6),
        help="Use learning rate scheduler with specified minimum learning rate",
    )
    parser.add_argument(
        "--use-grad-clip",
        type=float,
        default=defaults.get("use_grad_clip", 1.0),
        help="Use gradient clipping with specified maximum norm",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=defaults.get("data_dir", "./data"),
        help="Data directory for the dataset",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default=defaults.get("artifacts_dir", "./artifacts"),
        help="Directory for storing artifacts",
    )
    parser.add_argument(
        "--save-best",
        action="store_true",
        default=defaults.get("save_best", True),
        help="Save the best model based on validation accuracy",
    )

    # Now parse all args with the fully built parser
    args = parser.parse_args()

    # Store config path in args for reference
    args.config_path = config_path

    return args


def set_seed(seed: int = 42, deterministic: bool = True):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def seed_worker(worker_id):
    """Ensure each DataLoader worker has different but reproducible seeds."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ==================== Usage Example ====================
if __name__ == "__main__":
    args = parse_args_with_defaults()

    print("=" * 50)
    print("Configuration")
    print("=" * 50)
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)} : type: {type(getattr(args, arg))}")
    print("=" * 50)

    # Set seeds
    set_seed(args.seed, deterministic=True)
