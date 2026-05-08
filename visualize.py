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
import json
import argparse
from typing import List

import numpy as np
import cv2

from data.dataset import CAD120Dataset, MPHOI72Dataset, BimanualDataset
from data.preprocess import VideoReader

# ---------------------------------------------------------------------------
# Culori (aceleasi cu inference.py)
# ---------------------------------------------------------------------------

COLORS = [
    (255, 100, 100), (100, 255, 100), (100, 100, 255),
    (255, 255, 100), (255, 100, 255), (100, 255, 255),
    (200, 150, 100), (150, 200, 100), (100, 150, 200),
    (200, 100, 150), (150, 100, 200), (100, 200, 150),
    (220, 180, 80),  (80, 220, 180),
]


def get_label_names(dataset_name: str):
    mapping = {
        "cad120": CAD120Dataset.ACTIVITY_LABELS,
        "mphoi72": MPHOI72Dataset.ACTIVITY_LABELS,
        "bimanual": BimanualDataset.ACTIVITY_LABELS,
    }
    return mapping[dataset_name]


# ---------------------------------------------------------------------------
# Functii vizualizare (replicate din inference.py)
# ---------------------------------------------------------------------------

def draw_predictions_on_frame(frame, boxes, pred_labels, label_names, frame_idx, total_frames, source="auto"):
    """Deseneaza bounding boxes + etichete pe un frame."""
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
    cv2.putText(vis, f"Frame {frame_idx+1}/{total_frames} | {source}",
                (10, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return vis


def draw_timeline(all_predictions, label_names, width=800, row_height=30):
    """Creeaza o banda timeline colorata cu predictii per entitate."""
    S, M = all_predictions.shape
    height = (M + 1) * row_height + 20
    timeline = np.ones((height, width, 3), dtype=np.uint8) * 245
    cv2.putText(timeline, "VHOIP — inference",
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


def save_visualization_video(frames, boxes, pred_classes, label_names, out_path, fps=15):
    """Scrie video annotat + timeline."""
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    S = len(frames)
    M = pred_classes.shape[1]
    H, W = frames[0].shape[:2]

    timeline = draw_timeline(pred_classes, label_names, width=W)
    tl_h = timeline.shape[0]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (W, H + tl_h))

    print(f"Scrie video: {out_path}")
    for s in range(S):
        vis = draw_predictions_on_frame(frames[s], boxes[s], pred_classes[s], label_names, s, S, source="auto")
        tl = timeline.copy()
        cursor_x = 80 + int(s / max(S - 1, 1) * (W - 80))
        cv2.line(tl, (cursor_x, 0), (cursor_x, tl_h), (0, 0, 0), 2)
        tl_resized = cv2.resize(tl, (W, tl_h))
        writer.write(np.vstack([vis, tl_resized]))

    writer.release()
    print(f"Video salvat: {out_path}")


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
