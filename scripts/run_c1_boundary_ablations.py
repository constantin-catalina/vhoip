#!/usr/bin/env python3
"""
Run C1 temporal boundary ablations for MPHOI-72, folds 0-4.

Order: for each fold, run kernel_size=3 then kernel_size=7.
Seed is fixed at 42 for reproducibility.
"""

import subprocess
import sys

FOLDS = [0, 1, 2, 3, 4]
KERNELS = [3, 7]
CONFIG = "configs/mphoi72.yaml"
SEED = 42
DEVICE = "cuda"
PYTHON = sys.executable


def main():
    for fold in FOLDS:
        for k in KERNELS:
            exp_name = f"c1-temporal-boundary-{k}"
            cmd = [
                PYTHON,
                "train.py",
                "--config", CONFIG,
                "--fold", str(fold),
                "--seed", str(SEED),
                "--experiment_name", exp_name,
                "--device", DEVICE,
                "--override", f"model.boundary_kernel_size={k}",
            ]
            print(f"\n{'=' * 80}")
            print(f"Running Fold {fold} | Kernel size {k} | Experiment: {exp_name}")
            print(f"{'=' * 80}\n")
            subprocess.run(cmd, check=False)


if __name__ == "__main__":
    main()
