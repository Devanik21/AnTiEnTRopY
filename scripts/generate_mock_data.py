#!/usr/bin/env python3
"""
Generate Mock Data
Creates a small mock dataset for testing AnTiEnTRopY.
"""

import pandas as pd
import numpy as np
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Generate mock methylation data")
    parser.add_argument("--samples", type=int, default=50, help="Number of samples")
    parser.add_argument("--cpgs", type=int, default=1000, help="Number of CpGs")
    parser.add_argument("--output", type=str, default="data/raw/mock_data.csv", help="Output file")
    args = parser.parse_args()

    ages = np.random.uniform(20, 80, args.samples)
    data = {"age": ages}

    # Simulate some drift
    drift = (ages - 20) / 60.0 * 0.2
    for i in range(args.cpgs):
        beta = np.random.uniform(0.1, 0.9, args.samples)
        # Adding simple age correlation to ~10% of CpGs
        if i < args.cpgs * 0.1:
            beta = np.clip(beta + drift * np.random.choice([-1, 1]), 0, 1)
        data[f"cg{i:08d}"] = beta

    df = pd.DataFrame(data)
    df.to_csv(args.output, index=False)
    print(f"Generated mock data at {args.output}")

if __name__ == "__main__":
    main()
