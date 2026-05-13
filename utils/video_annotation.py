"""
utils/viz_opencv.py
Functii comune de vizualizare OpenCV folosite de inference.py si visualize.py.
"""

import os
from typing import List
import numpy as np
import cv2

from data.dataset import CAD120Dataset, MPHOI72Dataset, BimanualDataset


# ---------------------------------------------------------------------------
# Culori pentru vizualizare
# ---------------------------------------------------------------------------

COLORS = [
    (255, 100, 100), (100, 255, 100), (100, 100, 255),
    (255, 255, 100), (255, 100, 255), (100, 255, 255),
    (200, 150, 100), (150, 200, 100), (100, 150, 200),
    (200, 100, 150), (150, 100, 200), (100, 200, 150),
    (220, 180, 80),  (80, 220, 180),
]


# ---------------------------------------------------------------------------
# Mapare dataset -> nume clase
# ---------------------------------------------------------------------------

def get_label_names(dataset_name: str) -> List[str]:
    mapping = {
        "cad120": CAD120Dataset.ACTIVITY_LABELS,
        "mphoi72": MPHOI72Dataset.ACTIVITY_LABELS,
        "bimanual": BimanualDataset.ACTIVITY_LABELS,
    }
    return mapping[dataset_name]


# ---------------------------------------------------------------------------
# Desenare frame + timeline
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
