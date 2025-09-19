#!/usr/bin/env python3

"""
inspect_parquet.py

A small utility to inspect the data format of a Parquet file:
- Prints file size, basic parquet metadata
- Shows schema with column names, types, and nullability
- Shows a few sample rows from the first row group (without loading the whole file)

Usage:
  python scripts/inspect_parquet.py --path ~/data/geo3k/test.parquet --rows 5
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def format_bytes(num_bytes: int) -> str:
    """Convert bytes into a human-friendly string."""
    step_to_units = [
        (1 << 40, "TB"),
        (1 << 30, "GB"),
        (1 << 20, "MB"),
        (1 << 10, "KB"),
    ]
    for factor, suffix in step_to_units:
        if num_bytes >= factor:
            return f"{num_bytes / factor:.2f} {suffix}"
    return f"{num_bytes} B"


def try_import_pyarrow() -> None:
    try:
        import pyarrow  # noqa: F401
        import pyarrow.parquet  # noqa: F401
    except Exception as exc:
        print("[ERROR] pyarrow is required. Install with: pip install pyarrow", file=sys.stderr)
        print(f"Details: {exc}", file=sys.stderr)
        sys.exit(1)


def print_basic_info(parquet_path: Path) -> None:
    file_size = parquet_path.stat().st_size
    print(f"File: {parquet_path}")
    print(f"Size: {format_bytes(file_size)}")


def print_parquet_metadata(parquet_path: Path) -> None:
    import pyarrow.parquet as pq

    parquet_file = pq.ParquetFile(str(parquet_path))
    metadata = parquet_file.metadata

    created_by = getattr(metadata, "created_by", None) or "Unknown"
    num_rows = metadata.num_rows
    num_row_groups = metadata.num_row_groups
    num_columns = metadata.num_columns

    print("\nParquet Metadata:")
    print(f"- Created by: {created_by}")
    print(f"- Rows: {num_rows}")
    print(f"- Row groups: {num_row_groups}")
    print(f"- Columns: {num_columns}")


def print_schema(parquet_path: Path) -> None:
    import pyarrow.parquet as pq

    parquet_file = pq.ParquetFile(str(parquet_path))
    # Prefer Arrow schema for consistent Field API across versions
    try:
        schema = parquet_file.schema_arrow  # pyarrow >= 9
    except Exception:
        # Fallback for older versions
        schema = parquet_file.schema.to_arrow_schema()

    print("\nSchema:")
    for field in schema:
        nullability = "nullable" if getattr(field, "nullable", False) else "non-null"
        print(f"- {field.name}: {field.type} ({nullability})")


def print_sample_rows(parquet_path: Path, num_rows: int) -> None:
    import pyarrow.parquet as pq

    parquet_file = pq.ParquetFile(str(parquet_path))
    if parquet_file.num_row_groups == 0:
        print("\nNo row groups found; file appears empty.")
        return

    print(f"\nSample rows (first {num_rows}):")
    # Read only the first row group to avoid loading the entire file
    table = parquet_file.read_row_group(0)
    if table.num_rows == 0:
        print("<empty>")
        return

    limited = table.slice(0, min(num_rows, table.num_rows))
    try:
        # Prefer pretty printing with pandas if available
        import pandas as pd  # noqa: F401
        df = limited.to_pandas()
        # Show dataframe using pandas' default formatting
        print(df)
    except Exception:
        # Fallback to Arrow table printing
        print(limited)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect Parquet file format and preview rows.")
    parser.add_argument(
        "--path",
        type=str,
        default=str(Path.home() / "data/iceberg/test.parquet"),
        help="Path to the Parquet file (default: ~/data/geo3k/test.parquet)",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=5,
        help="Number of sample rows to display from the first row group",
    )
    args = parser.parse_args()

    try_import_pyarrow()

    parquet_path = Path(os.path.expanduser(args.path)).resolve()
    if not parquet_path.exists():
        print(f"[ERROR] File not found: {parquet_path}", file=sys.stderr)
        sys.exit(1)

    print_basic_info(parquet_path)
    print_parquet_metadata(parquet_path)
    print_schema(parquet_path)
    print_sample_rows(parquet_path, args.rows)


if __name__ == "__main__":
    main()


