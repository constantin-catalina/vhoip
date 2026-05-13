"""
visualize.py
Script standalone pentru vizualizarea rezultatelor VHOIP.

Poate rula in doua moduri:
  1. Din date brute salvate de inference.py (--raw)
  2. Din JSON + video original (--results + --input)

Utilizare:
    # Din raw data (fara re-extractie)
    python visualize.py --raw inference_results_raw.npz \
                        --dataset mphoi72 \
                        --output viz.mp4

    # Din JSON + video (re-extrage bounding boxes)
    python visualize.py --results inference_results.json \
                        --input path/to/video.mp4 \
                        --dataset mphoi72 \
                        --output viz.mp4
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

import json
import argparse
from typing import List

import numpy as np
import cv2

from data.preprocess import VideoReader
from utils.video_annotation import (
    COLORS, get_label_names, draw_predictions_on_frame,
    draw_timeline, save_visualization_video,
)

# Vizualizare: vezi utils/viz_opencv.py


# get_label_names, draw_predictions_on_frame, draw_timeline, save_visualization_video: vezi utils/viz_opencv.py


def save_timeline_png(pred_classes, entity_types, label_names, out_path):
    """Salveaza timeline ca PNG cu matplotlib."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("Matplotlib nu este instalat. Skip timeline PNG.")
        return

    S, M = pred_classes.shape
    colors = plt.cm.tab20.colors

    fig, axes = plt.subplots(M, 1, figsize=(16, 2 * M), sharex=True)
    if M == 1:
        axes = [axes]

    for m in range(M):
        entity_label = "Person" if entity_types[m] == 0 else "Object"
        seq = pred_classes[:, m]
        for s in range(S):
            axes[m].barh(0, 1, left=s, color=colors[int(seq[s]) % 20], height=0.8)
        axes[m].set_ylabel(f"{entity_label} {m+1}", fontsize=9)
        axes[m].set_yticks([])

    axes[0].set_xlim(0, S)

    patches = [mpatches.Patch(color=colors[i % 20], label=label_names[i])
               for i in range(len(label_names))]
    fig.legend(handles=patches, loc="lower center", ncol=min(7, len(label_names)), fontsize=8)
    plt.xlabel("Frame")
    plt.suptitle("VHOIP predictions — timeline", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Timeline PNG salvat: {out_path}")


# ---------------------------------------------------------------------------
# Incarcare date
# ---------------------------------------------------------------------------

def load_from_raw(raw_path: str):
    """Incarca date din fisier .npz generat de inference.py."""
    data = np.load(raw_path, allow_pickle=True)
    frames = list(data["frames"])
    boxes = data["boxes"]
    pred_classes = data["pred_classes"]
    entity_types = data["entity_types"]
    print(f"Date brute incarcate: {raw_path}")
    print(f"  Frames: {len(frames)}, Boxes: {boxes.shape}, Preds: {pred_classes.shape}")
    return frames, boxes, pred_classes, entity_types


def load_from_json_results(results_path: str, input_path: str, max_frames: int = None):
    """
    Reconstruieste vizualizarea din JSON + video original.
    Necesita re-extractie bounding boxes (nu salvam boxes in JSON).
    """
    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    # Citeste frame-uri
    reader = VideoReader(input_path, max_frames=max_frames)
    frames = reader.read_frames()
    S = len(frames)

    # Parsam predictii din JSON
    predictions = results["predictions"]
    S_json = len(predictions)
    if S != S_json:
        print(f"[WARN] Video are {S} frame-uri, JSON are {S_json}. Folosesc min.")
        S = min(S, S_json)
        frames = frames[:S]
        predictions = predictions[:S]

    M = len(predictions[0]["entities"]) if predictions else 0
    pred_classes = np.zeros((S, M), dtype=np.int64)
    entity_types = np.zeros(M, dtype=np.int64)

    for s, frame_info in enumerate(predictions):
        for ent in frame_info["entities"]:
            m = ent["id"]
            # Mapare nume -> index
            # Nu avem mapare aici, folosim index 0 ca fallback
            # Acest mod este limitat pentru vizualizare pura
            pred_classes[s, m] = 0  # placeholder
            entity_types[m] = 0 if ent["type"] == "human" else 1

    # Pentru boxes, nu le avem in JSON — folosim dummy boxes
    # Acest mod este DOAR pentru timeline, nu pentru overlay de boxe
    boxes = np.zeros((S, M, 4), dtype=np.float32)
    print("[WARN] Mod JSON: bounding boxes nu sunt disponibile. Overlay de boxe dezactivat.")

    return frames, boxes, pred_classes, entity_types


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Vizualizare rezultate VHOIP")
    parser.add_argument("--raw", type=str, default=None,
                        help="Path catre fisierul .npz generat de inference.py")
    parser.add_argument("--results", type=str, default=None,
                        help="Path catre JSON rezultate")
    parser.add_argument("--input", type=str, default=None,
                        help="Video original (necesar doar cu --results)")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["cad120", "mphoi72", "bimanual"])
    parser.add_argument("--output", type=str, default="visualization.mp4",
                        help="Path video output")
    parser.add_argument("--timeline-png", type=str, default=None,
                        help="Path optional pentru timeline PNG")
    parser.add_argument("--fps", type=int, default=15)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.raw:
        frames, boxes, pred_classes, entity_types = load_from_raw(args.raw)
    elif args.results and args.input:
        frames, boxes, pred_classes, entity_types = load_from_json_results(args.results, args.input)
    else:
        raise ValueError("Specifica --raw sau ambele --results + --input.")

    label_names = get_label_names(args.dataset)

    # Genereaza video
    save_visualization_video(frames, boxes, pred_classes, label_names, args.output, fps=args.fps)

    # Genereaza timeline PNG
    png_path = args.timeline_png or args.output.replace(".mp4", "_timeline.png")
    save_timeline_png(pred_classes, entity_types, label_names, png_path)

    print("\nGata!")


if __name__ == "__main__":
    main()
