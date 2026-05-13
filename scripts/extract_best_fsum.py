"""
Extract best FSUM per experiment and fold from checkpoints.

Looks for best_model.pth under checkpoints/ and prints the exact
FSUM value with all decimals, grouped by experiment name and fold.

Usage:
    python scripts/extract_best_fsum.py
    python scripts/extract_best_fsum.py --checkpoint_dir checkpoints
    python scripts/extract_best_fsum.py --dataset mphoi72
"""

import argparse
import os
import re
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Extract best FSUM per experiment and fold")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Root checkpoint directory")
    parser.add_argument("--dataset", type=str, default=None, help="Filter by dataset (e.g., mphoi72, cad120, bimanual)")
    parser.add_argument("--experiment", type=str, default=None, help="Filter by experiment name (e.g., c6, default)")
    parser.add_argument("--csv", type=str, default=None, help="Optional: also write results to a CSV file")
    return parser.parse_args()


def extract_from_checkpoint(path: Path) -> float:
    """Load a checkpoint and return the exact FSUM value."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    metrics = ckpt.get("metrics", {})
    fsum = metrics.get("fsum")
    if fsum is None:
        raise ValueError(f"'fsum' not found in checkpoint: {path}")
    return float(fsum)


def main():
    args = parse_args()
    root = Path(args.checkpoint_dir)

    if not root.exists():
        print(f"Checkpoint directory not found: {root}")
        return

    # Collect results: list of dicts
    results = []

    # Walk checkpoints/<dataset>_fold<N>/<experiment>/best_model.pth
    for fold_dir in root.iterdir():
        if not fold_dir.is_dir():
            continue

        # Parse fold directory name: <dataset>_fold<N>
        match = re.match(r"^(?P<dataset>\w+)_fold(?P<fold>\d+)$", fold_dir.name)
        if not match:
            continue

        dataset = match.group("dataset")
        fold = int(match.group("fold"))

        if args.dataset and dataset != args.dataset:
            continue

        for exp_dir in fold_dir.iterdir():
            if not exp_dir.is_dir():
                continue

            experiment = exp_dir.name
            if args.experiment and experiment != args.experiment:
                continue

            best_model = exp_dir / "best_model.pth"
            if not best_model.exists():
                continue

            try:
                fsum = extract_from_checkpoint(best_model)
                results.append({
                    "dataset": dataset,
                    "experiment": experiment,
                    "fold": fold,
                    "fsum": fsum,
                    "path": str(best_model),
                })
            except Exception as e:
                print(f"  [WARN] Skipping {best_model}: {e}")

    if not results:
        print("No best_model.pth checkpoints found.")
        return

    # Sort by dataset, experiment, fold
    results.sort(key=lambda r: (r["dataset"], r["experiment"], r["fold"]))

    # Print table
    print(f"{'Dataset':<12} {'Experiment':<12} {'Fold':<6} {'Best FSUM'}")
    print("-" * 60)
    for r in results:
        # Print fsum with all decimals using repr to avoid any formatting truncation
        print(f"{r['dataset']:<12} {r['experiment']:<12} {r['fold']:<6} {repr(r['fsum'])}")

    # Summary per experiment
    print("\n" + "=" * 60)
    print("Summary per experiment")
    print("=" * 60)
    from collections import defaultdict
    exp_groups = defaultdict(list)
    for r in results:
        exp_groups[(r["dataset"], r["experiment"])].append(r["fsum"])

    for (dataset, experiment), fsums in sorted(exp_groups.items()):
        mean_fsum = sum(fsums) / len(fsums)
        print(f"{dataset:<12} {experiment:<12}  folds={len(fsums)}  mean={repr(mean_fsum)}")

    # Optional CSV export
    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["dataset", "experiment", "fold", "fsum"])
            writer.writeheader()
            for r in results:
                writer.writerow({
                    "dataset": r["dataset"],
                    "experiment": r["experiment"],
                    "fold": r["fold"],
                    "fsum": repr(r["fsum"]),  # preserve full precision as string
                })
        print(f"\nResults also written to: {args.csv}")


if __name__ == "__main__":
    main()
