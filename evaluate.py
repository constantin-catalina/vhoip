"""
evaluate.py
Evaluare completa cu cross-validare pentru VHOIP.
Ruleaza toate fold-urile si raporteaza media +/- std (ca in paper).

Utilizare:
    python evaluate.py --config configs/cad120.yaml --checkpoint checkpoints/best_model.pth
    python evaluate.py --config configs/cad120.yaml --checkpoint checkpoints/best_model.pth --all_folds
"""

import argparse
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from data.dataset import get_dataset, CAD120Dataset, MPHOI72Dataset, BimanualDataset
from models.vhoip import VHOIP
from utils.metrics import compute_f1_at_k, compute_metrics_epoch
from utils.checkpoint import load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluare VHOIP")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--all_folds", action="store_true",
                        help="Ruleaza evaluarea pe toate fold-urile si raporteaza media")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def get_label_names(dataset_name: str):
    mapping = {
        "cad120": CAD120Dataset.ACTIVITY_LABELS,
        "mphoi72": MPHOI72Dataset.ACTIVITY_LABELS,
        "bimanual": BimanualDataset.ACTIVITY_LABELS,
    }
    return mapping[dataset_name]


def get_num_folds(dataset_name: str) -> int:
    """Returneaza numarul de fold-uri per dataset (conform paper)."""
    return {"cad120": 4, "mphoi72": 28, "bimanual": 6}.get(dataset_name, 4)


@torch.no_grad()
def evaluate_fold(model, dataloader, device, iou_thresholds):
    """Evalueaza un singur fold si returneaza metrici detaliate."""
    model.set_inference_mode(True)

    all_preds, all_gts = [], []
    per_class_correct = {}

    for batch in dataloader:
        roi = batch["roi_features"].to(device)
        seg_labels = batch["seg_labels"]

        outputs = model(roi)
        pred_classes = outputs["segment_logits"].argmax(dim=-1).cpu()

        B, N = pred_classes.shape
        for b in range(B):
            preds = [(i, i + 1, pred_classes[b, i].item()) for i in range(N)]
            gts   = [(i, i + 1, seg_labels[b, i].item())   for i in range(N)]
            all_preds.append(preds)
            all_gts.append(gts)

            # Statistici per clasa
            for i in range(N):
                gt_cls = seg_labels[b, i].item()
                pr_cls = pred_classes[b, i].item()
                if gt_cls not in per_class_correct:
                    per_class_correct[gt_cls] = {"correct": 0, "total": 0}
                per_class_correct[gt_cls]["total"] += 1
                if gt_cls == pr_cls:
                    per_class_correct[gt_cls]["correct"] += 1

    metrics = compute_metrics_epoch(all_preds, all_gts, iou_thresholds)

    # Acuratete per clasa
    per_class_acc = {
        cls: v["correct"] / v["total"] * 100
        for cls, v in per_class_correct.items()
        if v["total"] > 0
    }
    metrics["per_class_acc"] = per_class_acc

    return metrics


def print_results_table(all_fold_metrics: list, dataset_name: str, label_names: list):
    """Afiseaza tabelul de rezultate ca in paper (media +/- std)."""

    print("\n" + "=" * 65)
    print(f"  Rezultate VHOIP pe {dataset_name.upper()}")
    print("=" * 65)
    print(f"  {'Fold':<6} {'F1@10':>8} {'F1@25':>8} {'F1@50':>8} {'FSUM':>8}")
    print("-" * 65)

    f1_10_vals, f1_25_vals, f1_50_vals, fsum_vals = [], [], [], []

    for fold, m in enumerate(all_fold_metrics):
        f1_10_vals.append(m["f1_10"])
        f1_25_vals.append(m["f1_25"])
        f1_50_vals.append(m["f1_50"])
        fsum_vals.append(m["fsum"])
        print(
            f"  {fold:<6} "
            f"{m['f1_10']:>7.1f}% "
            f"{m['f1_25']:>7.1f}% "
            f"{m['f1_50']:>7.1f}% "
            f"{m['fsum']:>8.1f}"
        )

    print("-" * 65)
    print(
        f"  {'Media':<6} "
        f"{np.mean(f1_10_vals):>6.1f}±{np.std(f1_10_vals):.1f} "
        f"{np.mean(f1_25_vals):>6.1f}±{np.std(f1_25_vals):.1f} "
        f"{np.mean(f1_50_vals):>6.1f}±{np.std(f1_50_vals):.1f} "
        f"{np.mean(fsum_vals):>8.1f}"
    )
    print("=" * 65)

    # Target din paper pentru comparatie
    paper_targets = {
        "cad120":   {"f1_10": 90.1, "f1_25": 86.6, "f1_50": 76.4, "fsum": 518.0},
        "mphoi72":  {"f1_10": 70.3, "f1_25": 65.7, "f1_50": 52.6, "fsum": 188.6},
        "bimanual": {"f1_10": 84.5, "f1_25": 81.6, "f1_50": 68.9, "fsum": 235.0},
    }

    if dataset_name in paper_targets:
        t = paper_targets[dataset_name]
        print(f"\n  Target din paper:")
        print(
            f"  {'Paper':<6} "
            f"{t['f1_10']:>8.1f} "
            f"{t['f1_25']:>8.1f} "
            f"{t['f1_50']:>8.1f} "
            f"{t['fsum']:>8.1f}"
        )

    # Acuratete per clasa (medie pe fold-uri)
    if label_names and all_fold_metrics[0].get("per_class_acc"):
        print(f"\n  Acuratete per clasa (media pe fold-uri):")
        for cls_idx, cls_name in enumerate(label_names):
            accs = [
                m["per_class_acc"].get(cls_idx, 0.0)
                for m in all_fold_metrics
            ]
            print(f"    {cls_name:<20} {np.mean(accs):>5.1f}%")


def main():
    args = parse_args()
    device = torch.device(args.device)

    cfg = OmegaConf.merge(
        OmegaConf.load("configs/base.yaml"),
        OmegaConf.load(args.config),
    )

    label_names = get_label_names(cfg.dataset.name)
    iou_thresholds = cfg.evaluation.iou_thresholds

    # Determina ce fold-uri sa evalueze
    if args.all_folds:
        folds = list(range(get_num_folds(cfg.dataset.name)))
        print(f"Evaluare pe {len(folds)} fold-uri...")
    else:
        folds = [args.fold]

    all_fold_metrics = []

    for fold in folds:
        print(f"\nFold {fold}...")

        # Incarca dataset
        val_ds = get_dataset(
            cfg.dataset.name, root=cfg.dataset.root, split="test", fold=fold
        )
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                                num_workers=cfg.data.num_workers)

        # Incarca model
        model = VHOIP(cfg, label_names, device=str(device)).to(device)
        load_checkpoint(args.checkpoint, model, device=str(device))

        # Evalueaza
        metrics = evaluate_fold(model, val_loader, device, iou_thresholds)
        all_fold_metrics.append(metrics)

        print(
            f"  F1@10={metrics['f1_10']:.1f}% "
            f"F1@25={metrics['f1_25']:.1f}% "
            f"F1@50={metrics['f1_50']:.1f}% "
            f"FSUM={metrics['fsum']:.1f}"
        )

    # Tabel final
    print_results_table(all_fold_metrics, cfg.dataset.name, label_names)


if __name__ == "__main__":
    main()