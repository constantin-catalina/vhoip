"""
demo_from_features.py
Runs VHOIP inference on pre-extracted features (downloaded from Colab).

Workflow:
  1. Run vhoip_vg_extractor.ipynb on Colab to extract VG features
  2. Download vhoip_features.zip and unzip it
  3. Run this script locally — no GPU required for inference

Usage:
    python demo_from_features.py \
        --config configs/mphoi72.yaml \
        --checkpoint checkpoints/mphoi72_fold0/best_model.pth \
        --features_dir vhoip_features \
        --video diet_15fps.mp4 \
        --output outputs/diet_vg_demo.mp4

Ensemble (average across folds):
    python demo_from_features.py \
        --config configs/mphoi72.yaml \
        --checkpoint_pattern "checkpoints/mphoi72_fold{fold}/best_model.pth" \
        --num_folds 5 \
        --features_dir vhoip_features \
        --video diet_15fps.mp4 \
        --output outputs/diet_vg_ensemble.mp4
"""

import argparse
import json
import os
import numpy as np
import torch
import torch.nn.functional as F
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from omegaconf import OmegaConf
from typing import List

from data.dataset import CAD120Dataset, MPHOI72Dataset, BimanualDataset
from data.preprocess import VideoReader
from models.vhoip import VHOIP
from utils.checkpoint import load_checkpoint


COLORS = [
    (255, 100, 100), (100, 255, 100), (100, 100, 255),
    (255, 255, 100), (255, 100, 255), (100, 255, 255),
    (200, 150, 100), (150, 200, 100), (100, 150, 200),
    (200, 100, 150), (150, 100, 200), (100, 200, 150),
    (220, 180, 80),  (80, 220, 180),
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",             type=str, required=True)
    parser.add_argument("--checkpoint",         type=str, default=None)
    parser.add_argument("--checkpoint_pattern", type=str, default=None,
                        help="e.g. checkpoints/mphoi72_fold{fold}/best_model.pth")
    parser.add_argument("--num_folds",          type=int, default=5)
    parser.add_argument("--features_dir",       type=str, required=True,
                        help="Directory with roi.npy, boxes.npy, geo.npy, entity_types.npy")
    parser.add_argument("--video",              type=str, required=True,
                        help="Original video for visual overlay")
    parser.add_argument("--output",             type=str, default="outputs/demo_vg.mp4")
    parser.add_argument("--device",             type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--ground_truth",       type=str, default=None,
                        help="Optional .npy file with GT class indices, shape (S,), "
                             "one label per frame for the primary person")
    return parser.parse_args()


def get_label_names(dataset_name):
    return {
        "cad120":   CAD120Dataset.ACTIVITY_LABELS,
        "mphoi72":  MPHOI72Dataset.ACTIVITY_LABELS,
        "bimanual": BimanualDataset.ACTIVITY_LABELS,
    }[dataset_name]


def load_features(features_dir):
    """Load pre-extracted features from directory."""
    roi   = np.load(os.path.join(features_dir, "roi.npy"))           # (S, M, 2048)
    boxes = np.load(os.path.join(features_dir, "boxes.npy"))         # (S, M, 4)
    geo   = np.load(os.path.join(features_dir, "geo.npy"))           # (S, M, 4)
    etypes = np.load(os.path.join(features_dir, "entity_types.npy")) # (M,)

    meta_path = os.path.join(features_dir, "meta.json")
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)

    print(f"Features loaded from: {features_dir}")
    print(f"  ROI shape:      {roi.shape}")
    print(f"  Boxes shape:    {boxes.shape}")
    print(f"  Entity types:   {etypes}  (0=person, 1=object)")
    if meta:
        print(f"  Extractor used: {meta.get('extractor', 'unknown')}")
        print(f"  Source video:   {meta.get('source_video', 'unknown')}")

    return roi, boxes, geo, etypes


def draw_predictions(frame, boxes, pred_labels, label_names, frame_idx, total_frames):
    vis = frame.copy()
    H, W = vis.shape[:2]
    for m in range(len(boxes)):
        x1, y1, x2, y2 = boxes[m].astype(int)
        if x2 - x1 < 5 or y2 - y1 < 5:
            continue
        cls_idx = int(pred_labels[m])
        color = COLORS[cls_idx % len(COLORS)]
        label = label_names[cls_idx] if cls_idx < len(label_names) else f"cls_{cls_idx}"
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
        cv2.rectangle(vis, (x1, y1 - 20), (x1 + text_size[0] + 4, y1), color, -1)
        cv2.putText(vis, label, (x1 + 2, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    progress = int((frame_idx / max(total_frames - 1, 1)) * W)
    cv2.rectangle(vis, (0, H - 8), (progress, H), (100, 200, 100), -1)
    cv2.putText(vis, f"Frame {frame_idx+1}/{total_frames} | VG features",
                (10, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return vis


def draw_timeline(all_predictions, label_names, width=800, row_height=30):
    S, M = all_predictions.shape
    height = (M + 1) * row_height + 20
    timeline = np.ones((height, width, 3), dtype=np.uint8) * 245
    cv2.putText(timeline, "VHOIP — Visual Genome features",
                (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 1)
    for m in range(M):
        y_top = (m + 1) * row_height
        cv2.putText(timeline, f"Entity {m+1}",
                    (5, y_top + row_height // 2 + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 80, 80), 1)
        fw = max(1, (width - 80) // S)
        for s in range(S):
            cls_idx = int(all_predictions[s, m])
            color = COLORS[cls_idx % len(COLORS)]
            x0 = 80 + s * fw
            cv2.rectangle(timeline, (x0, y_top + 2), (x0 + fw - 1, y_top + row_height - 2), color, -1)
    y_legend = (M + 1) * row_height + 5
    x = 80
    for i, name in enumerate(label_names[:min(len(label_names), 8)]):
        color = COLORS[i % len(COLORS)]
        cv2.rectangle(timeline, (x, y_legend - 10), (x + 15, y_legend + 2), color, -1)
        cv2.putText(timeline, name[:8], (x + 18, y_legend),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (50, 50, 50), 1)
        x += 90
        if x > width - 90:
            break
    return timeline


def save_timeline_png(pred_classes, entity_types, label_names, out_path, ground_truth=None):
    S, M = pred_classes.shape
    colors = plt.cm.tab20.colors

    n_rows = M + (1 if ground_truth is not None else 0)
    fig, axes = plt.subplots(n_rows, 1, figsize=(16, 2 * n_rows), sharex=True)
    if n_rows == 1:
        axes = [axes]

    row = 0
    if ground_truth is not None:
        gt = ground_truth[:S]
        for s in range(len(gt)):
            axes[row].barh(0, 1, left=s, color=colors[int(gt[s]) % 20], height=0.8)
        axes[row].set_ylabel("Ground truth", fontsize=9, color="darkgreen")
        axes[row].set_yticks([])
        axes[row].spines["left"].set_edgecolor("darkgreen")
        row += 1

    for m in range(M):
        entity_label = "Person" if entity_types[m] == 0 else "Object"
        seq = pred_classes[:, m]
        for s in range(S):
            axes[row].barh(0, 1, left=s, color=colors[seq[s] % 20], height=0.8)
        axes[row].set_ylabel(f"{entity_label} {m+1}", fontsize=9)
        axes[row].set_yticks([])
        row += 1

    axes[0].set_xlim(0, S)

    patches = [mpatches.Patch(color=colors[i % 20], label=label_names[i])
               for i in range(len(label_names))]
    fig.legend(handles=patches, loc="lower center", ncol=min(7, len(label_names)), fontsize=8)
    plt.xlabel("Frame")
    title = "VHOIP predictions vs ground truth — Visual Genome features" \
            if ground_truth is not None else "VHOIP predictions — Visual Genome features"
    plt.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Timeline saved: {out_path}")


@torch.no_grad()
def run(args):
    device = torch.device(args.device)

    cfg = OmegaConf.merge(
        OmegaConf.load("configs/base.yaml"),
        OmegaConf.load(args.config),
    )
    label_names = get_label_names(cfg.dataset.name)

    # Load pre-extracted features
    roi_all, boxes_all, geo_all, entity_types = load_features(args.features_dir)
    S, M, _ = roi_all.shape

    # Load original video frames for visual overlay
    print(f"Reading video frames: {args.video}")
    reader = VideoReader(args.video, max_frames=S)
    frames = reader.read_frames()
    if len(frames) < S:
        print(f"  Warning: video has {len(frames)} frames, features have {S}. Using min.")
        S = min(len(frames), S)
        roi_all   = roi_all[:S]
        boxes_all = boxes_all[:S]

    # Build tensors
    roi_tensor   = torch.FloatTensor(roi_all).unsqueeze(0).to(device)     # (1, S, M, 2048)
    types_tensor = torch.LongTensor(entity_types).unsqueeze(0).to(device) # (1, M)

    # Build model
    print("Loading model...")
    model = VHOIP(cfg, label_names, device=str(device)).to(device)
    model.set_inference_mode(True)

    # Inference — single checkpoint or ensemble
    if args.checkpoint_pattern:
        print(f"Ensemble mode: {args.num_folds} folds")
        all_softmax = []
        for fold in range(args.num_folds):
            ckpt = args.checkpoint_pattern.format(fold=fold)
            if not os.path.exists(ckpt):
                print(f"  Skipping fold {fold} — checkpoint not found: {ckpt}")
                continue
            print(f"  Fold {fold}: {ckpt}")
            load_checkpoint(ckpt, model, device=str(device))
            out = model(roi_features=roi_tensor, entity_types=types_tensor)
            all_softmax.append(F.softmax(out["segment_logits"], dim=-1))
        avg = torch.stack(all_softmax).mean(0)
        pred_classes = avg.argmax(dim=-1).squeeze(0).cpu().numpy()
    else:
        load_checkpoint(args.checkpoint, model, device=str(device))
        out = model(roi_features=roi_tensor, entity_types=types_tensor)
        pred_classes = out["segment_logits"].argmax(dim=-1).squeeze(0).cpu().numpy()

    if pred_classes.ndim == 1:
        pred_classes = pred_classes.reshape(S, M)

    # Write output video
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    H, W = frames[0].shape[:2]
    timeline = draw_timeline(pred_classes, label_names)
    tl_h = timeline.shape[0]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, 15, (W, H + tl_h))

    print(f"Writing output: {args.output}")
    for s in range(S):
        vis = draw_predictions(frames[s], boxes_all[s], pred_classes[s], label_names, s, S)
        tl = timeline.copy()
        cursor_x = 80 + int(s / max(S - 1, 1) * (W - 80))
        cv2.line(tl, (cursor_x, 0), (cursor_x, tl_h), (0, 0, 0), 2)
        tl_resized = cv2.resize(tl, (W, tl_h))
        writer.write(np.vstack([vis, tl_resized]))

    writer.release()
    print(f"\nSaved: {args.output}")

    # Load optional ground truth
    ground_truth = None
    if args.ground_truth:
        ground_truth = np.load(args.ground_truth)
        print(f"Ground truth loaded: {args.ground_truth}  shape={ground_truth.shape}")

    # Save timeline PNG
    timeline_png = args.output.replace(".mp4", "_timeline.png")
    save_timeline_png(pred_classes, entity_types, label_names, timeline_png,
                      ground_truth=ground_truth)

    # Print prediction distribution
    print("\nPrediction distribution:")
    for cls_idx, cls_name in enumerate(label_names):
        count = np.sum(pred_classes == cls_idx)
        if count > 0:
            pct = count / pred_classes.size * 100
            print(f"  {cls_name:<20} {'#' * int(pct/2):<25} {pct:.1f}%")


if __name__ == "__main__":
    args = parse_args()
    run(args)
