"""
demo.py
Inferenta VHOIP pe un video nou - vizualizeaza segmentarea HOI.

Utilizare:
    python demo.py --config configs/cad120.yaml --checkpoint checkpoints/best_model.pth --video path/to/video.mp4
    python demo.py --config configs/cad120.yaml --checkpoint checkpoints/best_model.pth --video path/to/frames_dir/
"""

import argparse
import os
import numpy as np
import torch
import cv2
from omegaconf import OmegaConf
from typing import List

from data.dataset import CAD120Dataset, MPHOI72Dataset, BimanualDataset
from data.preprocess import FasterRCNNExtractor, CLIPExtractor, VideoReader, get_imagenet_transform
from models.vhoip import VHOIP
from utils.checkpoint import load_checkpoint


# Culori per clasa (BGR pentru OpenCV)
COLORS = [
    (255, 100, 100), (100, 255, 100), (100, 100, 255),
    (255, 255, 100), (255, 100, 255), (100, 255, 255),
    (200, 150, 100), (150, 200, 100), (100, 150, 200),
    (200, 100, 150), (150, 100, 200), (100, 200, 150),
    (220, 180, 80),  (80, 220, 180),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Demo VHOIP")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--video", type=str, required=True,
                        help="Calea catre video sau director de frame-uri")
    parser.add_argument("--output", type=str, default="outputs/demo_output.mp4")
    parser.add_argument("--max_frames", type=int, default=None)
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


def draw_predictions(
    frame: np.ndarray,
    boxes: np.ndarray,        # (M, 4)
    pred_labels: np.ndarray,  # (M,)
    label_names: List[str],
    frame_idx: int,
    total_frames: int,
) -> np.ndarray:
    """Deseneaza bounding boxes si etichete pe frame."""
    vis = frame.copy()
    H, W = vis.shape[:2]

    for m in range(len(boxes)):
        x1, y1, x2, y2 = boxes[m].astype(int)
        if x2 - x1 < 5 or y2 - y1 < 5:
            continue

        cls_idx = int(pred_labels[m])
        color = COLORS[cls_idx % len(COLORS)]
        label = label_names[cls_idx] if cls_idx < len(label_names) else f"cls_{cls_idx}"

        # Bounding box
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

        # Label background
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
        cv2.rectangle(vis, (x1, y1 - 20), (x1 + text_size[0] + 4, y1), color, -1)
        cv2.putText(vis, label, (x1 + 2, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Progress bar
    progress = int((frame_idx / max(total_frames - 1, 1)) * W)
    cv2.rectangle(vis, (0, H - 8), (progress, H), (100, 200, 100), -1)
    cv2.putText(vis, f"Frame {frame_idx + 1}/{total_frames}",
                (10, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return vis


def draw_timeline(
    all_predictions: np.ndarray,   # (S, M)
    label_names: List[str],
    width: int = 800,
    row_height: int = 30,
) -> np.ndarray:
    """Genereaza o imagine cu timeline-ul segmentarii (ca Fig. 3 din paper)."""
    S, M = all_predictions.shape
    height = (M + 1) * row_height + 20

    timeline = np.ones((height, width, 3), dtype=np.uint8) * 245

    # Header
    cv2.putText(timeline, "Segmentare temporala VHOIP",
                (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 1)

    for m in range(M):
        y_top = (m + 1) * row_height
        label_text = f"Entity {m + 1}"
        cv2.putText(timeline, label_text,
                    (5, y_top + row_height // 2 + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 80, 80), 1)

        # Coloreaza fiecare frame
        frame_width = max(1, (width - 80) // S)
        for s in range(S):
            cls_idx = int(all_predictions[s, m])
            color = COLORS[cls_idx % len(COLORS)]
            x_start = 80 + s * frame_width
            x_end = x_start + frame_width - 1
            cv2.rectangle(timeline,
                          (x_start, y_top + 2),
                          (x_end, y_top + row_height - 2),
                          color, -1)

    # Legenda
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


@torch.no_grad()
def run_demo(args):
    device = torch.device(args.device)

    cfg = OmegaConf.merge(
        OmegaConf.load("configs/base.yaml"),
        OmegaConf.load(args.config),
    )

    label_names = get_label_names(cfg.dataset.name)

    # Incarca model
    print("Incarcare model...")
    model = VHOIP(cfg, label_names, device=str(device)).to(device)
    load_checkpoint(args.checkpoint, model, device=str(device))
    model.set_inference_mode(True)

    # Initializeaza extractoare
    print("Incarcare extractoare...")
    imagenet_transform = get_imagenet_transform()
    rcnn = FasterRCNNExtractor(device=str(device))
    clip_extractor = CLIPExtractor(device=str(device))

    # Citeste frame-uri
    print(f"Citire video: {args.video}")
    reader = VideoReader(args.video, args.max_frames)
    frames = reader.read_frames()

    if not frames:
        print("Niciun frame gasit!")
        return

    print(f"  {len(frames)} frame-uri gasite")
    S = len(frames)
    M = cfg.model.get("max_entities", 5)

    # Extrage features
    print("Extragere features...")
    roi_all   = np.zeros((S, M, 2048), dtype=np.float32)
    boxes_all = np.zeros((S, M, 4),    dtype=np.float32)

    for s, frame in enumerate(frames):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_tensor = imagenet_transform(frame_rgb)
        roi_feats, boxes = rcnn.detect_and_extract(frame_tensor, M)
        roi_all[s]   = roi_feats.numpy()
        boxes_all[s] = boxes.numpy()

        if (s + 1) % 10 == 0:
            print(f"  Frame {s + 1}/{S}")

    # Inferenta
    print("Inferenta...")
    roi_tensor = torch.FloatTensor(roi_all).unsqueeze(0).to(device)  # (1, S, M, 2048)
    outputs = model(roi_tensor)
    segment_logits = outputs["segment_logits"]  # (1, S*M, C)
    pred_classes = segment_logits.argmax(dim=-1).squeeze(0).cpu().numpy()  # (S*M,)
    pred_classes = pred_classes.reshape(S, M)   # (S, M)

    # Salveaza video cu vizualizare
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    H, W = frames[0].shape[:2]

    # Timeline
    timeline = draw_timeline(pred_classes, label_names)
    timeline_h = timeline.shape[0]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(args.output, fourcc, 15, (W, H + timeline_h))

    print(f"Salvare video: {args.output}")
    for s, frame in enumerate(frames):
        vis_frame = draw_predictions(
            frame, boxes_all[s], pred_classes[s], label_names, s, S
        )

        # Timeline cu cursor la frame-ul curent
        tl = timeline.copy()
        cursor_x = 80 + int(s / max(S - 1, 1) * (W - 80))
        cv2.line(tl, (cursor_x, 0), (cursor_x, timeline_h), (0, 0, 0), 2)

        # Resize timeline la latimea frame-ului
        tl_resized = cv2.resize(tl, (W, timeline_h))
        combined = np.vstack([vis_frame, tl_resized])
        out_writer.write(combined)

    out_writer.release()
    print(f"\nDemo salvat: {args.output}")

    # Statistici finale
    print("\nDistributie predictii:")
    for cls_idx, cls_name in enumerate(label_names):
        count = np.sum(pred_classes == cls_idx)
        if count > 0:
            pct = count / pred_classes.size * 100
            bar = "#" * int(pct / 2)
            print(f"  {cls_name:<20} {bar:<25} {pct:.1f}%")


if __name__ == "__main__":
    args = parse_args()
    run_demo(args)