# quick_viz.py
import torch
import numpy as np
import argparse
import os
from omegaconf import OmegaConf
from data.mphoi72_dataset import MPHOI72ZarrDataset, collate_fn
from torch.utils.data import DataLoader
from models.vhoip import VHOIP
from utils.visualize import plot_segmentation, plot_multi_segmentation
from data.dataset import MPHOI72Dataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to best_model.pth")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--num_videos", type=int, default=5,
                        help="How many test videos to visualize")
    parser.add_argument("--entity", type=int, default=0,
                        help="Which entity index to visualize (0=first human)")
    parser.add_argument("--output_dir", type=str, default="outputs/viz")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    cfg = OmegaConf.merge(
        OmegaConf.load("configs/base.yaml"),
        OmegaConf.load("configs/mphoi72.yaml"),
    )
    label_names = MPHOI72Dataset.ACTIVITY_LABELS
    device = torch.device(args.device)

    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    model = VHOIP(cfg, label_names, device=str(device)).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.set_inference_mode(True)
    print(f"  Loaded from epoch {ckpt['epoch']} | "
          f"FSUM={ckpt['metrics'].get('fsum', 0):.1f}")

    # Load test set
    ds = MPHOI72ZarrDataset(cfg.dataset.root, split="test", fold=args.fold)
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_fn)
    print(f"Test set: {len(ds)} videos")

    results = []

    for i, batch in enumerate(loader):
        if i >= args.num_videos:
            break

        video_id = batch["video_ids"][0]
        S = batch["roi_features"].shape[1]
        M = batch["roi_features"].shape[2]

        with torch.no_grad():
            out = model(
                batch["roi_features"].to(device),
                batch["geo_features"].to(device),
                batch["entity_types"].to(device),
            )

        preds = out["segment_logits"].argmax(-1).cpu().squeeze(0)  # (S*M,)
        gt    = batch["seg_labels"].squeeze(0)                      # (S*M,)

        # Reshape to (S, M)
        preds_sm = preds.reshape(S, M)
        gt_sm    = gt.reshape(S, M)

        # Pick the entity to visualize
        e = min(args.entity, M - 1)
        preds_e = preds_sm[:, e].tolist()
        gt_e    = gt_sm[:, e].tolist()

        # Filter out padding (-1) from gt
        valid = [t for t in range(S) if gt_e[t] != -1]
        if not valid:
            print(f"  Video {video_id}: no valid labels, skipping")
            continue
        preds_e = [preds_e[t] for t in valid]
        gt_e    = [gt_e[t] for t in valid]

        # Per-video accuracy
        correct = sum(p == g for p, g in zip(preds_e, gt_e))
        acc = correct / len(gt_e) * 100

        print(f"  [{i+1}] {video_id} | entity {e} | "
              f"{len(gt_e)} frames | acc={acc:.1f}%")

        # Save individual plot
        save_path = os.path.join(
            args.output_dir, f"fold{args.fold}_{video_id}_e{e}.png"
        )
        plot_segmentation(
            ground_truth=gt_e,
            prediction=preds_e,
            label_names=label_names,
            title=f"{video_id} | entity {e} | acc={acc:.1f}%",
            save_path=save_path,
            show=False,
        )

        results.append({"title": f"{video_id} (acc={acc:.1f}%)",
                        "gt": gt_e, "pred": preds_e})

    # Save combined multi-video plot
    if results:
        combined_path = os.path.join(
            args.output_dir, f"fold{args.fold}_combined.png"
        )
        plot_multi_segmentation(
            results=results,
            label_names=label_names,
            save_path=combined_path,
            show=False,
        )
        print(f"\nCombined plot saved: {combined_path}")

    print("Done.")

if __name__ == "__main__":
    main()