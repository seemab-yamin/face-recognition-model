import argparse
import os
import random
import shutil
import sys


def prepare_casia_webface(data_dir, val_size=200):
    dataset_dir = os.path.join(data_dir, "webface_112x112")
    if not os.path.exists(dataset_dir):
        os.system(
            f"""curl -C - -L -o {data_dir}/webface_112x112.zip https://www.kaggle.com/api/v1/datasets/download/yakhyokhuja/webface_112x112"""
        )
        os.system(f"unzip -o {data_dir}/webface_112x112.zip -d {data_dir}")

    train_dir = os.path.join(dataset_dir, "train")
    val_dir = os.path.join(dataset_dir, "val")
    # validate if the val and train folders exist
    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        # Get all unique identity folders
        all_dirs_path = [
            os.path.join(dataset_dir, f)
            for f in os.listdir(dataset_dir)
            if os.path.isdir(os.path.join(dataset_dir, f))
            and not f.startswith(".")  # Skip hidden folders
        ]
        all_files = sorted(
            [
                os.path.join(dirs, f)
                for dirs in all_dirs_path
                for f in os.listdir(dirs)
                if os.path.isfile(os.path.join(dirs, f))
                and not f.startswith(".")  # Skip hidden folders
            ]
        )
        random.shuffle(all_files)
        val_files = all_files[:val_size]

        # move validation files to val_dir
        for src in val_files:
            dst = os.path.join(val_dir, src.split(os.sep)[-2], os.path.basename(src))
            # Check if the folder exists in source before trying to move it
            if os.path.exists(src):
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.move(src, dst)

        # Move remaining directories to train_dir
        for src in all_dirs_path:
            if os.path.exists(src):
                dst = os.path.join(train_dir, os.path.basename(src))
                shutil.move(src, dst)

        # Remove empty directories after moving files
        for dirs in os.listdir(dataset_dir):
            dir_path = os.path.join(dataset_dir, dirs)
            if os.path.isdir(dir_path) and not os.listdir(dir_path):
                os.rmdir(dir_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="face-recognition-model")
    parser.add_argument(
        "--dataset",
        type=str,
        help="Dataset to use",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Data directory for the dataset",
    )

    # Now parse all args with the fully built parser
    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)

    if args.dataset.lower() == "casia-webface":
        prepare_casia_webface(args.data_dir)
    else:
        print(f"Dataset {args.dataset} not supported")
        sys.exit(1)
