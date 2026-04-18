#!/usr/bin/env python3
"""
Run Batch Inference
A stub for running AnTiEnTRopY on multiple datasets sequentially.
"""

import argparse
import glob

def main():
    parser = argparse.ArgumentParser(description="Run batch inference on multiple CSVs")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing CSVs")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save reports")
    args = parser.parse_args()

    files = glob.glob(f"{args.input_dir}/*.csv")
    print(f"Found {len(files)} datasets.")
    for file in files:
        print(f"Stub: Running inference on {file}...")

    print("Batch inference complete.")

if __name__ == "__main__":
    main()
