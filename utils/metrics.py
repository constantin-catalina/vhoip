"""
metrics.py
Metrici de evaluare din paper: F1@k (k=10,25,50) si FSUM.

O predictie e corecta daca IoU cu segmentul real >= k%.
"""

import numpy as np
from typing import List, Tuple


def compute_iou(pred_start: float, pred_end: float, gt_start: float, gt_end: float) -> float:
    """Calculeaza IoU intre doua intervale temporale."""
    intersection = max(0, min(pred_end, gt_end) - max(pred_start, gt_start))
    union = max(pred_end, gt_end) - min(pred_start, gt_start)
    if union <= 0:
        return 0.0
    return intersection / union


def compute_f1_at_k(
    predictions: List[Tuple[float, float, int]],
    ground_truths: List[Tuple[float, float, int]],
    iou_threshold: float,
) -> float:
    """
    Calculeaza F1@k pentru o singura secventa video.

    Args:
        predictions: lista de (start, end, class_id) - segmente prezise
        ground_truths: lista de (start, end, class_id) - segmente reale
        iou_threshold: pragul IoU (ex: 0.10, 0.25, 0.50)

    Returns:
        F1 score in [0, 1]
    """
    if not ground_truths:
        return 1.0 if not predictions else 0.0
    if not predictions:
        return 0.0

    matched_gt = set()
    true_positives = 0

    for pred in predictions:
        pred_start, pred_end, pred_cls = pred
        best_iou = 0.0
        best_gt_idx = -1

        for gt_idx, gt in enumerate(ground_truths):
            if gt_idx in matched_gt:
                continue
            gt_start, gt_end, gt_cls = gt

            if pred_cls != gt_cls:
                continue

            iou = compute_iou(pred_start, pred_end, gt_start, gt_end)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_threshold and best_gt_idx >= 0:
            true_positives += 1
            matched_gt.add(best_gt_idx)

    precision = true_positives / len(predictions) if predictions else 0.0
    recall = true_positives / len(ground_truths) if ground_truths else 0.0

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_fsum(f1_scores: dict) -> float:
    """
    FSUM = suma mediilor F1@10 + F1@25 + F1@50 (din paper).

    Args:
        f1_scores: {"f1_10": val, "f1_25": val, "f1_50": val}
    Returns:
        FSUM scalar
    """
    return f1_scores["f1_10"] + f1_scores["f1_25"] + f1_scores["f1_50"]


def compute_metrics_epoch(
    all_predictions: List,
    all_ground_truths: List,
    thresholds: List[float] = [0.10, 0.25, 0.50],
) -> dict:
    """
    Calculeaza toate metricile pentru un epoch intreg (media pe toate video-urile).

    Args:
        all_predictions: lista de liste de segmente prezise per video
        all_ground_truths: lista de liste de segmente reale per video
        thresholds: pragurile IoU
    Returns:
        dict cu f1_10, f1_25, f1_50, fsum (toate ca procente 0-100)
    """
    results = {f"f1_{int(t*100)}": [] for t in thresholds}

    for preds, gts in zip(all_predictions, all_ground_truths):
        for t in thresholds:
            key = f"f1_{int(t*100)}"
            score = compute_f1_at_k(preds, gts, t)
            results[key].append(score * 100)  # in procente, ca in paper

    # Media si std pentru cross-validation
    final = {}
    for key, vals in results.items():
        arr = np.array(vals)
        final[key] = float(np.mean(arr))
        final[f"{key}_std"] = float(np.std(arr))

    final["fsum"] = compute_fsum(final)
    return final