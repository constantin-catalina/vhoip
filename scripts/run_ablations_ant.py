"""
scripts/run_ablations_ant.py
Orchestrator for C6 ablation experiments with L_Ant always removed.

Runs ablations 2-5 from the base list, each combined with lambda_ant=0,
on folds 0-4. Logs to W&B under c6_<name>, extracts full metrics
(F1@10/25/50 + FSUM) from checkpoints, and produces a summary table + CSV.

Usage:
    python scripts/run_ablations_ant.py --config configs/mphoi72.yaml --folds 5
"""

import argparse
import csv
import os
import subprocess
import sys
import time
import torch
from pathlib import Path


ABLATIONS = [
    {
        "name": "w/o_L_Ant+w/o_L_PromptReg",
        "overrides": ["training.lambda_ant=0", "training.lambda4=0"],
    },
    {
        "name": "w/o_L_Ant+w/o_GeoBranch",
        "overrides": ["training.lambda_ant=0", "model.disable_geo_branch=true"],
    },
    {
        "name": "w/o_L_Ant+w/o_L_MI",
        "overrides": ["training.lambda_ant=0", "training.lambda2=0"],
    },
    {
        "name": "w/o_L_Ant+w/o_L_Cos",
        "overrides": ["training.lambda_ant=0", "training.lambda3=0"],
    },
]


def parse_args():
    parser = argparse.ArgumentParser(description="Ablation study orchestrator (L_Ant always removed)")
    parser.add_argument("--config", type=str, default="configs/mphoi72.yaml")
    parser.add_argument("--folds", type=int, default=5, help="Number of folds to run (0 .. folds-1)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--csv", type=str, default="ablation_ant_results.csv")
    parser.add_argument("--resume", action="store_true", help="Skip runs where best_model.pth already exists")
    return parser.parse_args()


def build_command(python_exe, work_dir, config, fold, seed, device, exp_name, overrides):
    cmd = [
        python_exe,
        "train.py",
        "--config", config,
        "--fold", str(fold),
        "--seed", str(seed),
        "--experiment_name", exp_name,
        "--device", device,
    ]
    for ov in overrides:
        cmd.extend(["--override", ov])
    return cmd


def find_checkpoint(dataset_name, fold, exp_name):
    # Path mirrors train.py logic:
    # checkpoints/<dataset>_fold<N>/<exp_name>/best_model.pth
    checkpoint_dir = Path("checkpoints") / f"{dataset_name}_fold{fold}" / exp_name
    best_path = checkpoint_dir / "best_model.pth"
    return best_path if best_path.exists() else None


def load_metrics_from_checkpoint(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    metrics = ckpt.get("metrics", {})
    return {
        "f1_10": metrics.get("f1_10", 0.0),
        "f1_25": metrics.get("f1_25", 0.0),
        "f1_50": metrics.get("f1_50", 0.0),
        "fsum": metrics.get("fsum", 0.0),
    }


def print_summary(results):
    # results: list of dicts with keys: ablation, fold, f1_10, f1_25, f1_50, fsum
    print("\n" + "=" * 80)
    print("ABLATION RESULTS (per fold)")
    print("=" * 80)
    print(f"{'Ablation':<25} {'Fold':>5} {'F1@10':>8} {'F1@25':>8} {'F1@50':>8} {'FSUM':>8}")
    print("-" * 80)
    for r in results:
        print(
            f"{r['ablation']:<25} {r['fold']:>5} "
            f"{r['f1_10']:>8.1f} {r['f1_25']:>8.1f} "
            f"{r['f1_50']:>8.1f} {r['fsum']:>8.1f}"
        )

    # Aggregate
    print("\n" + "=" * 80)
    print("ABLATION SUMMARY (mean +- std over folds)")
    print("=" * 80)
    print(f"{'Ablation':<25} {'F1@10':>12} {'F1@25':>12} {'F1@50':>12} {'FSUM':>12}")
    print("-" * 80)

    import numpy as np
    ablation_names = sorted({r["ablation"] for r in results})
    for name in ablation_names:
        vals = [r for r in results if r["ablation"] == name]
        f1_10s = [v["f1_10"] for v in vals]
        f1_25s = [v["f1_25"] for v in vals]
        f1_50s = [v["f1_50"] for v in vals]
        fsums = [v["fsum"] for v in vals]
        print(
            f"{name:<25} "
            f"{np.mean(f1_10s):>6.1f}+-{np.std(f1_10s):<4.1f} "
            f"{np.mean(f1_25s):>6.1f}+-{np.std(f1_25s):<4.1f} "
            f"{np.mean(f1_50s):>6.1f}+-{np.std(f1_50s):<4.1f} "
            f"{np.mean(fsums):>6.1f}+-{np.std(fsums):<4.1f}"
        )
    print("=" * 80)


def write_csv(results, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["ablation", "fold", "f1_10", "f1_25", "f1_50", "fsum"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to {path}")


def main():
    args = parse_args()
    python_exe = sys.executable
    work_dir = Path(__file__).parent.parent.resolve()

    # Infer dataset name from config path (e.g. configs/mphoi72.yaml -> mphoi72)
    dataset_name = Path(args.config).stem

    results = []
    total_runs = len(ABLATIONS) * args.folds
    run_idx = 0

    for ablation in ABLATIONS:
        exp_name = f"c6_{ablation['name']}"
        print(f"\n>>> Ablation: {exp_name}")

        for fold in range(args.folds):
            run_idx += 1
            print(f"\n[{run_idx}/{total_runs}] Fold {fold} -- {exp_name}")

            # Resume check
            ckpt_path = find_checkpoint(dataset_name, fold, exp_name)
            if ckpt_path and args.resume:
                print(f"  Found checkpoint: {ckpt_path} -- skipping training.")
                metrics = load_metrics_from_checkpoint(ckpt_path)
                results.append({
                    "ablation": ablation["name"],
                    "fold": fold,
                    **metrics,
                })
                continue

            # Build and run command
            cmd = build_command(
                python_exe,
                work_dir,
                args.config,
                fold,
                args.seed,
                args.device,
                exp_name,
                ablation["overrides"],
            )
            print(f"  Command: {' '.join(cmd)}")
            start = time.time()
            proc = subprocess.run(
                cmd,
                cwd=work_dir,
                capture_output=True,
                text=True,
            )
            elapsed = time.time() - start

            # Print last lines of output for progress tracking
            lines = (proc.stdout + proc.stderr).splitlines()
            for line in lines[-20:]:
                print(f"    {line}")

            if proc.returncode != 0:
                print(f"  [ERROR] Run failed with exit code {proc.returncode}")
                continue

            # Extract metrics from checkpoint
            ckpt_path = find_checkpoint(dataset_name, fold, exp_name)
            if not ckpt_path:
                print(f"  [WARN] Checkpoint not found after training.")
                continue

            metrics = load_metrics_from_checkpoint(ckpt_path)
            results.append({
                "ablation": ablation["name"],
                "fold": fold,
                **metrics,
            })
            print(
                f"  Completed in {elapsed:.1f}s -- "
                f"F1@10={metrics['f1_10']:.1f} F1@25={metrics['f1_25']:.1f} "
                f"F1@50={metrics['f1_50']:.1f} FSUM={metrics['fsum']:.1f}"
            )

    # Final summary
    print_summary(results)
    write_csv(results, args.csv)


if __name__ == "__main__":
    main()
