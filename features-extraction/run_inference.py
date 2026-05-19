"""
VHOIP Inference on Extracted Features
======================================
Loads the .npz feature file produced by extract_features.py
and runs the VHOIP model to predict HOI labels.

Usage:
    python run_inference.py \
        --features features/my_video_features.npz \
        --config configs/mphoi72.yaml \
        --checkpoint checkpoints/mphoi72_fold0/c6/best_model.pth \
        --output results.json

    # Ensemble across folds:
    python run_inference.py \
        --features features/my_video_features.npz \
        --config configs/mphoi72.yaml \
        --checkpoint-pattern "checkpoints/mphoi72_fold{fold}/c6/best_model.pth" \
        --num-folds 28 \
        --output results.json
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from pathlib import Path

# Make VHOIP repo importable
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

from models.vhoip import VHOIP
from utils.checkpoint import load_checkpoint
from utils.video_annotation import get_label_names

from omegaconf import OmegaConf


# ──────────────────────────────────────────────────────────────────────────────
# Feature loader
# ──────────────────────────────────────────────────────────────────────────────

def load_features(npz_path: str, device: str = "cpu"):
    """
    Load feature .npz and convert to model-ready tensors.

    Returns dict with:
      roi_features  : (1, T, M, 2048)
      geo_features  : (1, T, J, 4)
      entity_types  : (1, M)
      clip_features : (1, T, M, 512)   (for G initialization)
      bboxes        : (T, M, 4)       numpy, for visualization
    """
    data = np.load(npz_path)

    roi  = data["roi_features"].astype(np.float32)   # (T, M, 2048)
    geo  = data["geo_features"].astype(np.float32)    # (T, J, 4)
    etypes = data["entity_types"].astype(np.int64)     # (M,)
    clip = data["clip_features"].astype(np.float32)    # (T, M, 512)
    bboxes = data["bboxes"].astype(np.float32)         # (T, M, 4)

    print(f"[Loader] Loaded features from {npz_path}")
    print(f"  roi_features   : {roi.shape}")
    print(f"  geo_features   : {geo.shape}")
    print(f"  entity_types   : {etypes}")
    print(f"  clip_features  : {clip.shape}")
    print(f"  bboxes         : {bboxes.shape}")

    return {
        "roi_features":  torch.FloatTensor(roi).unsqueeze(0).to(device),    # (1, T, M, 2048)
        "geo_features":  torch.FloatTensor(geo).unsqueeze(0).to(device),    # (1, T, J, 4)
        "entity_types":  torch.LongTensor(etypes).unsqueeze(0).to(device),  # (1, M)
        "clip_features": torch.FloatTensor(clip).unsqueeze(0).to(device),    # (1, T, M, 512)
        "bboxes":        bboxes,   # (T, M, 4) numpy
    }


# ──────────────────────────────────────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────────────────────────────────────

def run_single(model, features, checkpoint_path, device):
    """Run inference with a single checkpoint."""
    load_checkpoint(checkpoint_path, model, device=str(device))
    model.set_inference_mode(True)
    with torch.no_grad():
        out = model(
            roi_features=features["roi_features"],
            geo_features=features["geo_features"],
            entity_types=features["entity_types"],
        )
    return out["segment_logits"]


def run_ensemble(model, features, checkpoint_pattern, num_folds, device):
    """Run inference with ensemble over multiple fold checkpoints."""
    all_logits = []
    loaded = 0
    for fold in range(num_folds):
        ckpt = checkpoint_pattern.format(fold=fold)
        if not os.path.exists(ckpt):
            print(f"  [SKIP] Fold {fold}: {ckpt}")
            continue
        print(f"  [LOAD] Fold {fold}: {ckpt}")
        load_checkpoint(ckpt, model, device=str(device))
        model.set_inference_mode(True)
        with torch.no_grad():
            out = model(
                roi_features=features["roi_features"],
                geo_features=features["geo_features"],
                entity_types=features["entity_types"],
            )
            all_logits.append(out["segment_logits"])
        loaded += 1

    if loaded == 0:
        raise RuntimeError("No valid checkpoints found for ensemble.")
    avg_logits = torch.stack(all_logits).mean(dim=0)
    print(f"  Ensemble: averaged over {loaded}/{num_folds} folds.")
    return avg_logits


# ──────────────────────────────────────────────────────────────────────────────
# Results
# ──────────────────────────────────────────────────────────────────────────────

def decode_predictions(segment_logits, label_names, entity_types):
    """Decode logits to per-frame per-entity predictions."""
    B, N, C = segment_logits.shape
    probs = torch.softmax(segment_logits, dim=-1)
    pred_ids = probs.argmax(dim=-1)
    M = entity_types.shape[1]
    S = N // M

    pred_classes = pred_ids.squeeze(0).cpu().numpy().reshape(S, M)

    results = []
    for s in range(S):
        frame_entities = []
        for m in range(M):
            idx = s * M + m
            if idx >= N:
                break
            cls_id = pred_ids[0, idx].item()
            conf = probs[0, idx, cls_id].item()
            et = entity_types[0, m].item()
            frame_entities.append({
                "id": m,
                "type": "human" if et == 0 else "object",
                "predicted": label_names[cls_id] if cls_id < len(label_names) else f"class_{cls_id}",
                "confidence": round(conf, 4),
            })
        results.append({"frame": s, "entities": frame_entities})

    return results, pred_classes


def print_summary(pred_classes, label_names):
    """Print prediction distribution."""
    print("\nDistribution:")
    for cls_idx, cls_name in enumerate(label_names):
        count = np.sum(pred_classes == cls_idx)
        if count > 0:
            pct = count / pred_classes.size * 100
            print(f"  {cls_name:<20} {'#' * int(pct/2):<25} {pct:.1f}%")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Run VHOIP inference on extracted features.")
    p.add_argument("--features", required=True,
                   help="Path to .npz feature file from extract_features.py")
    p.add_argument("--config", required=True,
                   help="Path to config YAML (e.g. configs/mphoi72.yaml)")
    p.add_argument("--checkpoint", default="",
                   help="Path to single VHOIP checkpoint .pth")
    p.add_argument("--checkpoint-pattern", default="",
                   help="Pattern with {fold} for ensemble inference")
    p.add_argument("--num-folds", type=int, default=28,
                   help="Number of folds for ensemble (default 28)")
    p.add_argument("--output", default="inference_results.json",
                   help="Output JSON file path")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = torch.device(args.device)

    if not args.checkpoint and not args.checkpoint_pattern:
        raise ValueError("Specify --checkpoint or --checkpoint-pattern.")

    # Load config (resolve paths relative to the parent repo root)
    repo_root = str(Path(__file__).parent.parent.resolve())
    def resolve_cfg(path):
        if os.path.isabs(path):
            return path
        full = os.path.join(repo_root, path)
        if os.path.exists(full):
            return full
        return path
    cfg = OmegaConf.merge(
        OmegaConf.load(resolve_cfg(os.path.join("configs", "base.yaml"))),
        OmegaConf.load(resolve_cfg(args.config)),
    )
    label_names = get_label_names(cfg.dataset.name)
    print(f"Dataset: {cfg.dataset.name} | Classes: {label_names}")

    # Load features
    print(f"\n[1/3] Loading features...")
    features = load_features(args.features, str(device))

    # Initialize model
    print(f"\n[2/3] Loading model...")
    model = VHOIP(cfg, label_names, device=str(device)).to(device)

    # Run inference
    print(f"\n[3/3] Running inference...")
    if args.checkpoint_pattern:
        segment_logits = run_ensemble(
            model, features, args.checkpoint_pattern, args.num_folds, str(device))
    else:
        segment_logits = run_single(
            model, features, args.checkpoint, str(device))

    # Decode
    predictions, pred_classes = decode_predictions(
        segment_logits, label_names, features["entity_types"])

    # Save results
    B, N, C = segment_logits.shape
    M = features["entity_types"].shape[1]
    S = N // M

    result = {
        "input": args.features,
        "dataset": cfg.dataset.name,
        "num_frames": int(S),
        "num_entities": int(M),
        "predictions": predictions,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {args.output}")

    # Summary
    print_summary(pred_classes, label_names)