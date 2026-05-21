"""
inference.py
Script pentru inferenta VHOIP folosind features pre-extrase.

Ruleaza modelul pe features .npy (aceleasi folosite la training),
eliminand mismatch-urile intre pipeline-ul de training si inferenta.

Exemplu:
    python inference.py --config configs/mphoi72.yaml \
                        --video-id Subject12-task_1_cheering-take_0 \
                        --data-root data/mphoi72/ \
                        --checkpoint-pattern "checkpoints/mphoi72_fold{fold}/c6/best_model.pth" \
                        --num-folds 28 --output results.json --visualize

Pentru vizualizare, specifica --input <path_video> sau lasa scriptul
sa caute automat fisierul video in --data-root.
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

import json
import argparse
from typing import List, Dict
from pathlib import Path

import numpy as np
import torch
import cv2
from omegaconf import OmegaConf

# Proiect
from data.preprocess import VideoReader
from models.vhoip import VHOIP
from utils.checkpoint import load_checkpoint
from utils.video_annotation import (
    COLORS, get_label_names, draw_predictions_on_frame,
    draw_timeline, save_visualization_video,
)

# Vizualizare: vezi utils/viz_opencv.py


# ---------------------------------------------------------------------------
# Decodare rezultate
# ---------------------------------------------------------------------------

def decode_predictions(
    segment_logits: torch.Tensor,
    label_names: List[str],
    entity_types: torch.Tensor,
) -> List[Dict]:
    """
    Decodifica logit-urile in predictii per frame per entitate.
    """
    B, N, C = segment_logits.shape
    probs = torch.softmax(segment_logits, dim=-1)
    pred_ids = probs.argmax(dim=-1)
    M = entity_types.shape[1]
    S = N // M

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

    return results


# ---------------------------------------------------------------------------
# Vizualizare
# ---------------------------------------------------------------------------

# draw_predictions_on_frame, draw_timeline, save_visualization_video: vezi utils/viz_opencv.py


def save_raw_data_npz(out_path, frames, boxes, pred_classes, entity_types):
    """Salveaza date brute pentru re-vizualizare ulterioara."""
    # Converteste lista de frame-uri in array (poate fi mare)
    # Salvam ca lista de array-uri in npz
    np.savez_compressed(
        out_path,
        frames=np.array(frames),
        boxes=boxes,
        pred_classes=pred_classes,
        entity_types=entity_types,
    )
    print(f"Date brute salvate: {out_path}")


# ---------------------------------------------------------------------------
# Ensemble inference
# ---------------------------------------------------------------------------

def ensemble_inference(
    model: VHOIP,
    roi_features: torch.Tensor,
    geo_features: torch.Tensor,
    entity_types: torch.Tensor,
    checkpoint_pattern: str,
    num_folds: int,
    device: str,
) -> torch.Tensor:
    """
    Ruleaza inferenta cu ensemble peste mai multe fold-uri.
    Returneaza logit-urile mediate (nu softmax).
    """
    all_logits = []
    loaded = 0
    for fold in range(num_folds):
        ckpt = checkpoint_pattern.format(fold=fold)
        if not os.path.exists(ckpt):
            print(f"  [SKIP] Fold {fold}: checkpoint nu exista: {ckpt}")
            continue
        print(f"  [LOAD] Fold {fold}: {ckpt}")
        load_checkpoint(ckpt, model, device=device)
        model.set_inference_mode(True)
        with torch.no_grad():
            out = model(
                roi_features=roi_features,
                geo_features=geo_features,
                entity_types=entity_types,
            )
            all_logits.append(out["segment_logits"])
        loaded += 1

    if loaded == 0:
        raise RuntimeError("Niciun checkpoint valid gasit pentru ensemble.")

    avg_logits = torch.stack(all_logits).mean(dim=0)
    print(f"  Ensemble: mediat peste {loaded}/{num_folds} fold-uri.")
    return avg_logits


# ---------------------------------------------------------------------------
# Incarcare features pre-extrase din dataset (identic cu training)
# ---------------------------------------------------------------------------

def load_preextracted_features(
    video_id: str,
    data_root: str,
    clip_dim: int = 512,
) -> Dict[str, torch.Tensor]:
    """
    Incarca features pre-extrase din fisierele .npy generate de convert_zarr_to_npy().
    Acesta este acelasi pipeline folosit la training, eliminand mismatch-urile.

    Returns:
        dict cu: roi_features (1,S,M,2048), geo_features (1,S,J,4),
                 entity_types (1,M), clip_features (1,S,M,512),
                 bboxes (S,M,4)
    """
    feat_dir  = os.path.join(data_root, "features")
    label_dir = os.path.join(data_root, "labels")

    # ROI features
    roi_path = os.path.join(feat_dir, f"{video_id}_roi.npy")
    if not os.path.exists(roi_path):
        raise FileNotFoundError(f"ROI features nu exista: {roi_path}")
    roi = np.load(roi_path).astype(np.float32)  # (S, M, 2048)

    # Geometric features
    geo_path = os.path.join(feat_dir, f"{video_id}_geo.npy")
    if os.path.exists(geo_path):
        geo = np.load(geo_path).astype(np.float32)  # (S, J, 4)
    else:
        S, M = roi.shape[:2]
        J = 2 * 32 + (M - 2) * 4  # 2 humans * 32 joints + objects * 4 corners
        geo = np.zeros((S, J, 4), dtype=np.float32)

    # Entity types
    etypes_path = os.path.join(feat_dir, f"{video_id}_entity_types.npy")
    if os.path.exists(etypes_path):
        entity_types = np.load(etypes_path).astype(np.int64)  # (M,)
    else:
        M = roi.shape[1]
        entity_types = np.array([0, 0] + [1] * (M - 2), dtype=np.int64)

    # CLIP features (optional)
    clip_path = os.path.join(feat_dir, f"{video_id}_clip.npy")
    if os.path.exists(clip_path):
        clip = np.load(clip_path).astype(np.float32)  # (S, M, 512)
    else:
        clip = np.zeros((roi.shape[0], roi.shape[1], clip_dim), dtype=np.float32)

    # Bounding boxes (optional, pentru vizualizare)
    bbox_path = os.path.join(feat_dir, f"{video_id}_bbox.npy")
    bboxes = None
    if os.path.exists(bbox_path):
        bboxes = np.load(bbox_path).astype(np.float32)  # (S, M, 4)

    # Ground truth labels (optional, pentru comparare)
    seg_path = os.path.join(label_dir, f"{video_id}_seg.npy")
    gt_labels = None
    if os.path.exists(seg_path):
        gt_labels = np.load(seg_path).astype(np.int64)  # (N,)

    return {
        "roi_features":  torch.FloatTensor(roi).unsqueeze(0),     # (1, S, M, 2048)
        "geo_features":  torch.FloatTensor(geo).unsqueeze(0),      # (1, S, J, 4)
        "entity_types":  torch.LongTensor(entity_types).unsqueeze(0),  # (1, M)
        "clip_features": torch.FloatTensor(clip).unsqueeze(0),     # (1, S, M, 512)
        "bboxes":        bboxes,   # (S, M, 4) numpy, optional
        "gt_labels":     gt_labels, # (N,) numpy, optional
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Inferenta VHOIP pe video sau imagini")
    parser.add_argument("--config", type=str, required=True, help="Config YAML")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path checkpoint .pth (single)")
    parser.add_argument("--checkpoint-pattern", type=str, default=None,
                        help="Pattern cu {fold} pentru ensemble, ex: checkpoints/fold{fold}/best.pth")
    parser.add_argument("--num-folds", type=int, default=28,
                        help="Numar fold-uri pentru ensemble (default 28 pentru MPHOI-72)")
    parser.add_argument("--video-id", type=str, default=None, required=True,
                        help="Video ID din dataset (ex: Subject12-task_1_cheering-take_0). "
                             "Incarca features pre-extrase din data-root, identic cu training.")
    parser.add_argument("--data-root", type=str, default="data/mphoi72/",
                        help="Directorul radacina al dataset-ului (folosit cu --video-id)")
    parser.add_argument("--input", type=str, default=None,
                        help="Path video sau director imagini pentru vizualizare (optional). "
                             "Daca lipseste, scriptul cauta automat video in --data-root.")
    parser.add_argument("--output", type=str, default="inference_results.json", help="Fisier output JSON")
    parser.add_argument("--visualize", action="store_true", help="Genereaza video annotat + timeline PNG")
    parser.add_argument("--video-out", type=str, default=None,
                        help="Path video output (default: <output>.mp4)")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dataset-name", type=str, default=None)
    parser.add_argument("--fps", type=int, default=15, help="FPS pentru video output")
    return parser.parse_args()


# get_label_names: vezi utils/viz_opencv.py


def main():
    args = parse_args()
    device = torch.device(args.device)

    # Validare args
    if args.checkpoint is None and args.checkpoint_pattern is None:
        raise ValueError("Specifica --checkpoint sau --checkpoint-pattern.")
    if args.checkpoint is not None and args.checkpoint_pattern is not None:
        print("[WARN] Ambele --checkpoint si --checkpoint-pattern specificate. Folosesc --checkpoint-pattern.")

    # Director output: outputs/<nume_folder_data_root>/<video_id>.ext
    subfolder = os.path.basename(os.path.normpath(args.data_root))
    out_dir = os.path.join("outputs", subfolder)
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.dirname(args.output):
        args.output = os.path.join(out_dir, f"{args.video_id}.json")
    if args.video_out and not os.path.dirname(args.video_out):
        args.video_out = os.path.join(out_dir, os.path.basename(args.video_out))

    cfg = OmegaConf.merge(
        OmegaConf.load("configs/base.yaml"),
        OmegaConf.load(args.config),
    )

    dataset_name = args.dataset_name or cfg.dataset.name
    label_names = get_label_names(dataset_name)

    print(f"Dataset: {dataset_name} | Clase: {label_names}")
    print(f"Device: {device}")
    print(f"Output dir: {out_dir}")

    # -----------------------------------------------------------------------
    # Incarcare features pre-extrase
    # -----------------------------------------------------------------------
    boxes_all = None  # pentru vizualizare
    frames = None     # pentru vizualizare

    print(f"\n[1/3] Incarcare features pre-extrase pentru: {args.video_id}")
    data = load_preextracted_features(
        args.video_id, args.data_root, clip_dim=cfg.model.clip_dim,
    )
    roi_features = data["roi_features"].to(device)
    geo_features = data["geo_features"].to(device)
    entity_types = data["entity_types"].to(device)
    clip_features = data["clip_features"].to(device)
    gt_labels = data["gt_labels"]
    boxes_all = data.get("bboxes")
    input_label = args.video_id

    if args.visualize:
        video_path = args.input
        if not video_path:
            candidates = [
                os.path.join(args.data_root, f"{args.video_id}.mp4"),
                os.path.join(args.data_root, f"{args.video_id}.avi"),
                os.path.join(args.data_root, args.video_id, f"{args.video_id}.mp4"),
                os.path.join(args.data_root, "..", f"{args.video_id}.mp4"),
            ]
            for cand in candidates:
                if os.path.exists(cand):
                    video_path = cand
                    break
        if video_path and os.path.exists(video_path):
            print(f"  Incarca frame-uri pentru vizualizare din: {video_path}")
            reader = VideoReader(video_path, max_frames=args.max_frames)
            frames = reader.read_frames()
            if frames:
                print(f"  {len(frames)} frame-uri incarcate.")
            else:
                print(f"  [WARN] Nu am gasit frame-uri in: {video_path}")
        else:
            print(f"  [WARN] Nu am gasit video pentru vizualizare. Specifica --input <path>.")

    print(f"  ROI: {roi_features.shape}")
    print(f"  Geo: {geo_features.shape}")
    print(f"  Entity types: {entity_types.tolist()}")

    # -----------------------------------------------------------------------
    # Incarcare model
    # -----------------------------------------------------------------------
    print("\n[2/3] Incarcare model...")
    model = VHOIP(cfg, label_names, device=str(device)).to(device)

    # -----------------------------------------------------------------------
    # Inferenta — single sau ensemble
    # -----------------------------------------------------------------------
    print("\n[3/3] Inferenta...")
    if args.checkpoint_pattern:
        segment_logits = ensemble_inference(
            model, roi_features, geo_features, entity_types,
            args.checkpoint_pattern, args.num_folds, str(device),
        )
    else:
        load_checkpoint(args.checkpoint, model, device=str(device))
        model.set_inference_mode(True)
        with torch.no_grad():
            out = model(
                roi_features=roi_features,
                geo_features=geo_features,
                entity_types=entity_types,
            )
            segment_logits = out["segment_logits"]

    # -----------------------------------------------------------------------
    # Decodare
    # -----------------------------------------------------------------------
    pred_classes = segment_logits.argmax(dim=-1).squeeze(0).cpu().numpy()  # (N,)
    B, N, C = segment_logits.shape
    M = entity_types.shape[1]
    S = N // M
    if pred_classes.ndim == 1:
        pred_classes = pred_classes.reshape(S, M)

    predictions_json = decode_predictions(segment_logits, label_names, entity_types)

    # -----------------------------------------------------------------------
    # Rezultate
    # -----------------------------------------------------------------------
    result = {
        "input": input_label,
        "dataset": dataset_name,
        "num_frames": int(S),
        "num_entities": int(M),
        "predictions": predictions_json,
    }

    # Comparare cu ground truth daca e disponibil
    if gt_labels is not None:
        pred_flat = pred_classes.flatten()[:len(gt_labels)]
        gt_flat = gt_labels[:len(pred_flat)]
        acc = np.mean(pred_flat == gt_flat)
        result["ground_truth_accuracy"] = float(acc)
        print(f"\n  Ground truth accuracy: {acc:.2%}")

        # Per-class accuracy
        print("\n  Per-class comparison:")
        for cls_idx, cls_name in enumerate(label_names):
            mask = gt_flat == cls_idx
            if mask.sum() > 0:
                cls_acc = np.mean(pred_flat[mask] == cls_idx)
                print(f"    {cls_name:<20}: {cls_acc:.2%} ({mask.sum()} samples)")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\nRezultate salvate in: {args.output}")

    # Sumar
    print("\nSumar predictii:")
    for frame_info in [predictions_json[0], predictions_json[-1]]:
        print(f"  Frame {frame_info['frame']}:")
        for ent in frame_info["entities"]:
            print(f"    - Entitate {ent['id']} ({ent['type']}): {ent['predicted']} ({ent['confidence']:.2%})")

    # Distributie
    print("\nDistributie predictii:")
    for cls_idx, cls_name in enumerate(label_names):
        count = np.sum(pred_classes == cls_idx)
        if count > 0:
            pct = count / pred_classes.size * 100
            print(f"  {cls_name:<20} {'#' * int(pct/2):<25} {pct:.1f}%")

    # Vizualizare
    if args.visualize and frames is not None and boxes_all is not None:
        print("\n[4/4] Generare vizualizare...")
        if args.video_out:
            video_out = args.video_out
        else:
            out_dir = os.path.dirname(args.output) or "."
            video_out = os.path.join(out_dir, f"{args.video_id}.mp4")
        raw_path = video_out.replace(".mp4", "_raw.npz")

        save_visualization_video(frames, boxes_all, pred_classes, label_names, video_out, fps=args.fps)
        save_raw_data_npz(raw_path, frames, boxes_all, pred_classes, entity_types.cpu().numpy())


if __name__ == "__main__":
    main()
