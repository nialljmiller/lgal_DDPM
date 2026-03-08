#!/usr/bin/env python3

import csv
import random
import argparse
from pathlib import Path


def sample_csv_streaming(input_file, output_file, total_rows=200, keep_first=5, seed=42):
    """
    Stream a huge CSV without loading it all into memory.

    Output contains:
    - header
    - first `keep_first` data rows
    - random sample from the remaining rows so total data rows = `total_rows`
    """

    if total_rows < keep_first:
        raise ValueError("--rows must be >= --keep-first")

    random.seed(seed)

    input_path = Path(input_file)
    output_path = Path(output_file)

    reservoir_size = total_rows - keep_first
    first_rows = []
    reservoir = []

    total_data_rows_seen = 0
    rows_after_first_seen = 0

    with input_path.open("r", newline="", encoding="utf-8") as infile:
        reader = csv.reader(infile)

        try:
            header = next(reader)
        except StopIteration:
            raise ValueError("Input CSV is empty.")

        for row in reader:
            total_data_rows_seen += 1

            if len(first_rows) < keep_first:
                first_rows.append(row)
                continue

            rows_after_first_seen += 1

            if reservoir_size <= 0:
                continue

            if len(reservoir) < reservoir_size:
                reservoir.append(row)
            else:
                j = random.randint(0, rows_after_first_seen - 1)
                if j < reservoir_size:
                    reservoir[j] = row

    sampled_data = first_rows + reservoir

    with output_path.open("w", newline="", encoding="utf-8") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(header)
        writer.writerows(sampled_data)

    print(f"Input rows (excluding header): {total_data_rows_seen}")
    print(f"Output rows (excluding header): {len(sampled_data)}")
    print(f"Saved sampled CSV to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Stream-sample a huge CSV for LLM use.")
    parser.add_argument("input_csv", help="Path to input CSV")
    parser.add_argument("output_csv", help="Path to output CSV")
    parser.add_argument(
        "--rows",
        type=int,
        default=200,
        help="Total number of data rows to keep (default: 200)",
    )
    parser.add_argument(
        "--keep-first",
        type=int,
        default=5,
        help="Number of initial data rows to always keep (default: 5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()

    sample_csv_streaming(
        input_file=args.input_csv,
        output_file=args.output_csv,
        total_rows=args.rows,
        keep_first=args.keep_first,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()