#!/usr/bin/env python3
# Copyright 2025
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Download BAAI/TACO dataset from Hugging Face.

This script downloads the TACO dataset parquet files directly from Hugging Face,
bypassing the dataset loading script that is no longer supported.

Usage:
    python download_taco.py --output_dir data/taco
    python download_taco.py --output_dir data/taco --split test
    python download_taco.py --output_dir data/taco --split train --start_shard 0 --end_shard 2
"""

import argparse
import hashlib
import os
import sys
from pathlib import Path
from urllib.request import urlretrieve
from urllib.error import URLError, HTTPError

# TACO dataset file information
TACO_FILES = {
    "test": [
        {
            "name": "test-00000-of-00001.parquet",
            "url": "https://huggingface.co/datasets/BAAI/TACO/resolve/main/ALL/test-00000-of-00001.parquet",
            "size_mb": 246,
        },
    ],
    "train": [
        {
            "name": "train-00000-of-00009.parquet",
            "url": "https://huggingface.co/datasets/BAAI/TACO/resolve/main/ALL/train-00000-of-00009.parquet",
            "size_mb": 287,
        },
        {
            "name": "train-00001-of-00009.parquet",
            "url": "https://huggingface.co/datasets/BAAI/TACO/resolve/main/ALL/train-00001-of-00009.parquet",
            "size_mb": 327,
        },
        {
            "name": "train-00002-of-00009.parquet",
            "url": "https://huggingface.co/datasets/BAAI/TACO/resolve/main/ALL/train-00002-of-00009.parquet",
            "size_mb": 178,
        },
        {
            "name": "train-00003-of-00009.parquet",
            "url": "https://huggingface.co/datasets/BAAI/TACO/resolve/main/ALL/train-00003-of-00009.parquet",
            "size_mb": 179,
        },
        {
            "name": "train-00004-of-00009.parquet",
            "url": "https://huggingface.co/datasets/BAAI/TACO/resolve/main/ALL/train-00004-of-00009.parquet",
            "size_mb": 206,
        },
        {
            "name": "train-00005-of-00009.parquet",
            "url": "https://huggingface.co/datasets/BAAI/TACO/resolve/main/ALL/train-00005-of-00009.parquet",
            "size_mb": 272,
        },
        {
            "name": "train-00006-of-00009.parquet",
            "url": "https://huggingface.co/datasets/BAAI/TACO/resolve/main/ALL/train-00006-of-00009.parquet",
            "size_mb": 214,
        },
        {
            "name": "train-00007-of-00009.parquet",
            "url": "https://huggingface.co/datasets/BAAI/TACO/resolve/main/ALL/train-00007-of-00009.parquet",
            "size_mb": 260,
        },
        {
            "name": "train-00008-of-00009.parquet",
            "url": "https://huggingface.co/datasets/BAAI/TACO/resolve/main/ALL/train-00008-of-00009.parquet",
            "size_mb": 252,
        },
    ],
}


class DownloadProgressBar:
    """Simple progress bar for downloads."""

    def __init__(self, filename: str, total_size_mb: int):
        self.filename = filename
        self.total_size_mb = total_size_mb
        self.downloaded = 0
        self.last_percent = -1

    def __call__(self, block_num: int, block_size: int, total_size: int):
        self.downloaded += block_size
        if total_size > 0:
            percent = int(self.downloaded * 100 / total_size)
        else:
            percent = int(self.downloaded * 100 / (self.total_size_mb * 1024 * 1024))

        if percent != self.last_percent and percent % 5 == 0:
            downloaded_mb = self.downloaded / (1024 * 1024)
            bar_length = 30
            filled = int(bar_length * percent / 100)
            bar = "=" * filled + "-" * (bar_length - filled)
            print(f"\r  [{bar}] {percent}% ({downloaded_mb:.1f}MB / ~{self.total_size_mb}MB)", end="", flush=True)
            self.last_percent = percent


def download_file(url: str, output_path: Path, size_mb: int, force: bool = False) -> bool:
    """
    Download a file from URL to output path.

    Args:
        url: URL to download from.
        output_path: Path to save the file.
        size_mb: Expected file size in MB (for progress display).
        force: If True, overwrite existing files.

    Returns:
        True if download successful, False otherwise.
    """
    if output_path.exists() and not force:
        existing_size_mb = output_path.stat().st_size / (1024 * 1024)
        if abs(existing_size_mb - size_mb) < 10:  # Within 10MB tolerance
            print(f"  [SKIP] {output_path.name} already exists ({existing_size_mb:.1f}MB)")
            return True
        else:
            print(f"  [WARN] {output_path.name} exists but size mismatch ({existing_size_mb:.1f}MB vs expected {size_mb}MB)")

    print(f"  Downloading {output_path.name} (~{size_mb}MB)...")

    try:
        progress = DownloadProgressBar(output_path.name, size_mb)
        urlretrieve(url, output_path, reporthook=progress)
        print()  # New line after progress bar

        # Verify file size
        actual_size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"  [OK] Downloaded {actual_size_mb:.1f}MB")
        return True

    except HTTPError as e:
        print(f"\n  [ERROR] HTTP Error {e.code}: {e.reason}")
        return False
    except URLError as e:
        print(f"\n  [ERROR] URL Error: {e.reason}")
        return False
    except Exception as e:
        print(f"\n  [ERROR] Download failed: {e}")
        return False


def download_split(
    split: str,
    output_dir: Path,
    start_shard: int = 0,
    end_shard: int | None = None,
    force: bool = False,
) -> tuple[int, int]:
    """
    Download files for a specific split.

    Args:
        split: 'train' or 'test'.
        output_dir: Directory to save files.
        start_shard: Starting shard index (for partial downloads).
        end_shard: Ending shard index (exclusive, None for all).
        force: If True, overwrite existing files.

    Returns:
        Tuple of (successful_downloads, total_files).
    """
    if split not in TACO_FILES:
        print(f"[ERROR] Unknown split: {split}")
        return 0, 0

    files = TACO_FILES[split]
    if end_shard is not None:
        files = files[start_shard:end_shard]
    else:
        files = files[start_shard:]

    output_dir.mkdir(parents=True, exist_ok=True)

    successful = 0
    for file_info in files:
        output_path = output_dir / file_info["name"]
        if download_file(file_info["url"], output_path, file_info["size_mb"], force):
            successful += 1

    return successful, len(files)


def load_downloaded_data(data_dir: Path, split: str = "train", num_samples: int | None = None):
    """
    Load downloaded parquet files and return as list of dicts.

    Args:
        data_dir: Directory containing parquet files.
        split: 'train' or 'test'.
        num_samples: Maximum number of samples to load.

    Returns:
        List of dataset samples.
    """
    try:
        import pandas as pd
    except ImportError:
        print("[ERROR] pandas is required: pip install pandas pyarrow")
        return []

    # Find parquet files
    pattern = f"{split}-*.parquet"
    files = sorted(data_dir.glob(pattern))

    if not files:
        print(f"[ERROR] No {pattern} files found in {data_dir}")
        return []

    print(f"Loading {len(files)} parquet files...")

    samples = []
    for f in files:
        df = pd.read_parquet(f)
        samples.extend(df.to_dict(orient="records"))

        if num_samples and len(samples) >= num_samples:
            samples = samples[:num_samples]
            break

    print(f"Loaded {len(samples)} samples")
    return samples


def main():
    parser = argparse.ArgumentParser(
        description="Download BAAI/TACO dataset from Hugging Face",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/taco",
        help="Directory to save downloaded files",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="all",
        choices=["train", "test", "all"],
        help="Which split to download",
    )
    parser.add_argument(
        "--start_shard",
        type=int,
        default=0,
        help="Starting shard index (for partial downloads)",
    )
    parser.add_argument(
        "--end_shard",
        type=int,
        default=None,
        help="Ending shard index (exclusive, default: all)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing files",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify downloaded files by loading them",
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    print("=" * 60)
    print("TACO Dataset Downloader")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print(f"Split: {args.split}")
    print()

    # Calculate total size
    total_size_mb = 0
    splits_to_download = ["train", "test"] if args.split == "all" else [args.split]

    for split in splits_to_download:
        files = TACO_FILES[split]
        if args.end_shard is not None:
            files = files[args.start_shard:args.end_shard]
        else:
            files = files[args.start_shard:]
        total_size_mb += sum(f["size_mb"] for f in files)

    print(f"Total download size: ~{total_size_mb}MB ({total_size_mb/1024:.2f}GB)")
    print()

    # Download files
    total_successful = 0
    total_files = 0

    for split in splits_to_download:
        print(f"--- Downloading {split} split ---")
        successful, total = download_split(
            split=split,
            output_dir=output_dir,
            start_shard=args.start_shard,
            end_shard=args.end_shard,
            force=args.force,
        )
        total_successful += successful
        total_files += total
        print()

    print("=" * 60)
    print(f"Download Summary: {total_successful}/{total_files} files successful")
    print("=" * 60)

    # Verify if requested
    if args.verify and total_successful > 0:
        print("\nVerifying downloaded files...")
        for split in splits_to_download:
            samples = load_downloaded_data(output_dir, split, num_samples=5)
            if samples:
                print(f"\n{split} split sample:")
                sample = samples[0]
                print(f"  - question: {sample.get('question', 'N/A')[:100]}...")
                print(f"  - difficulty: {sample.get('difficulty', 'N/A')}")
                print(f"  - tags: {sample.get('tags', 'N/A')}")

    print("\nDone!")

    # Print usage hint
    print("\nTo use the downloaded data:")
    print(f"  from download_taco import load_downloaded_data")
    print(f"  samples = load_downloaded_data(Path('{output_dir}'), split='train')")


if __name__ == "__main__":
    main()
