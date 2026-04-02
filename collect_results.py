"""
collect_results.py
Colecteaza rezultatele din toate fold-urile si afiseaza tabelul final.
Utilizare: python collect_results.py --config configs/mphoi72.yaml
"""
import os
import torch
import numpy as np
from omegaconf import OmegaConf
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    args = parser.parse_args()

    cfg = OmegaConf.merge(
        OmegaConf.load("configs/base.yaml"),
        OmegaConf.load(args.config),
    )
    dataset = cfg.dataset.name
    num_folds = 28 if dataset == "mphoi72" else (6 if dataset == "cad120" else 15)

    f1_10_all, f1_25_all, f1_50_all, fsum_all = [], [], [], []
    missing = []

    for fold in range(num_folds):
        fold_dir = os.path.join(args.checkpoint_dir, f"{dataset}_fold{fold}")
        best_path = os.path.join(fold_dir, "best_model.pth")

        if not os.path.exists(best_path):
            missing.append(fold)
            continue

        ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
        m = ckpt["metrics"]
        f1_10_all.append(m["f1_10"])
        f1_25_all.append(m["f1_25"])
        f1_50_all.append(m["f1_50"])
        fsum_all.append(m["fsum"])

        print(f"  Fold {fold:2d} | epoch {ckpt['epoch']:2d} | "
              f"F1@10={m['f1_10']:5.1f} F1@25={m['f1_25']:5.1f} "
              f"F1@50={m['f1_50']:5.1f} FSUM={m['fsum']:6.1f}")

    if missing:
        print(f"\nLipsesc fold-urile: {missing}")

    if not f1_10_all:
        print("Niciun fold complet.")
        return

    print("\n" + "=" * 65)
    print(f"  REZULTATE FINALE — {dataset.upper()} ({len(f1_10_all)}/{num_folds} folduri)")
    print("=" * 65)
    print(f"  F1@10 : {np.mean(f1_10_all):.1f} ± {np.std(f1_10_all):.1f}")
    print(f"  F1@25 : {np.mean(f1_25_all):.1f} ± {np.std(f1_25_all):.1f}")
    print(f"  F1@50 : {np.mean(f1_50_all):.1f} ± {np.std(f1_50_all):.1f}")
    print(f"  FSUM  : {np.mean(fsum_all):.1f}")
    print("=" * 65)
    print(f"\n  Target din paper:")
    print(f"  F1@10=70.3 ± 9.1 | F1@25=65.7 ± 8.9 | F1@50=52.6 ± 7.1 | FSUM=188.6")

if __name__ == "__main__":
    main()