#!/usr/bin/env python3
"""
Top-level entry point matching the recommended submission structure
(PS Section 5). Thin wrapper around dataset/generator.py -- see that file
for implementation.

Usage:
    python generate_dataset.py --out_dir outputs --n_pairs 30 --seed 42
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset"))
from generator import generate_dataset

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="outputs")
    parser.add_argument("--n_pairs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate_dataset(args.out_dir, n_pairs=args.n_pairs, base_seed=args.seed)
