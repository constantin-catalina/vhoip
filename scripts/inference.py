"""
inference.py
Script pentru inferenta VHOIP pe imagini sau video.

Moduri de functionare:

1) --video-id (recomandat pentru dataset-ul MPHOI-72):
   Incarca features pre-extrase din fisierele .npy (aceleasi folosite la training).
   Elimina mismatch-urile intre pipeline-ul de training si cel de inferenta.

    python inference.py --config configs/mphoi72.yaml \
                        --video-id Subject12-task_1_cheering-take_0 \
                        --data-root data/mphoi72/ \
                        --checkpoint_pattern "checkpoints/mphoi72_fold{fold}/w/o_L_Ant/best_model.pth" \
                        --num_folds 28 --output results.json

2) --input (inferenta pe video noi, cu extragere live):
   Extrage automat ROI features (Faster R-CNN), keypoints (MediaPipe)
   si entity types din detectii. ATENTIE: rezultatele pot diferi fata de
   training din cauza mismatch-urilor de distributie a features.

    python inference.py --config configs/mphoi72.yaml \
                        --checkpoint checkpoints/best_model.pth \
                        --input path/to/video.mp4 \
                        --output results.json --visualize

Dependinte suplimentare (doar pentru modul --input):
    pip install mediapipe
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

import json
import argparse
from typing import List, Tuple, Optional, Dict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import cv2
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.ops import roi_pool
from omegaconf import OmegaConf

# Proiect
from data.preprocess import VideoReader, IMAGENET_MEAN, IMAGENET_STD
from models.vhoip import VHOIP
from utils.checkpoint import load_checkpoint
from utils.video_annotation import (
    COLORS, get_label_names, draw_predictions_on_frame,
    draw_timeline, save_visualization_video,
)

# Vizualizare: vezi utils/viz_opencv.py


# ---------------------------------------------------------------------------
# MediaPipe Pose Extractor
# ---------------------------------------------------------------------------

class MediaPipeExtractor:
    """
    Extrage keypoints umane (33 landmarks) cu MediaPipe PoseLandmarker per entitate.
    Se ruleaza pe crop-urile individuale detectate de Faster R-CNN.
    Foloseste API-ul MediaPipe Tasks (nou, inlocuieste mp.solutions).
    """

    NUM_LANDMARKS = 33

    def __init__(self, static_image_mode: bool = False):
        try:
            from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions
            from mediapipe.tasks.python.core.base_options import BaseOptions
            from mediapipe.tasks.python.vision import RunningMode
        except ImportError:
            raise ImportError(
                "MediaPipe nu este instalat sau versiunea nu suporta Tasks API. Ruleaza:\n"
                "  pip install mediapipe>=0.10.0"
            )

        model_url = (
            "https://storage.googleapis.com/mediapipe-models/"
            "pose_landmarker/pose_landmarker_heavy/float16/latest/"
            "pose_landmarker_heavy.task"
        )
        model_path = os.path.join(
            os.path.dirname(__file__), "..", "pose_landmarker_heavy.task"
        )
        model_path = os.path.abspath(model_path)

        if not os.path.exists(model_path):
            print(f"Descarc model MediaPipe Pose: {model_path} ...")
            import urllib.request
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            urllib.request.urlretrieve(model_url, model_path)
            print("Model descarcata.")

        base_options = BaseOptions(model_asset_path=model_path)
        options = PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
        )
        self.landmarker = PoseLandmarker.create_from_options(options)

    def extract(self, frame_bgr: np.ndarray, box: np.ndarray) -> Optional[np.ndarray]:
        """
        Args:
            frame_bgr: (H, W, 3) imagine BGR completa
            box: (4,) bounding box [x1, y1, x2, y2] in pixeli

        Returns:
            landmarks: (33, 2) array [x, y] in coordonatele frame-ului original,
                       sau None daca MediaPipe nu gaseste pose.
        """
        H, W = frame_bgr.shape[:2]
        x1, y1, x2, y2 = map(int, box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)

        if x2 - x1 < 5 or y2 - y1 < 5:
            return None

        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        from mediapipe import Image, ImageFormat
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)

        result = self.landmarker.detect(mp_image)

        if not result.pose_landmarks:
            return None

        landmarks = []
        for lm in result.pose_landmarks[0]:
            px = lm.x * (x2 - x1) + x1
            py = lm.y * (y2 - y1) + y1
            landmarks.append([px, py])

        return np.array(landmarks, dtype=np.float32)  # (33, 2)

    def close(self):
        self.landmarker.close()


# ---------------------------------------------------------------------------
# Faster R-CNN cu ROI + labels
# ---------------------------------------------------------------------------

class InferenceRCNN(torch.nn.Module):
    """
    Wrapper peste Faster R-CNN care returneaza ROI features, boxes si labels.
    """

    def __init__(self, device: str = "cuda", score_threshold: float = 0.3):
        super().__init__()
        self.device = device
        self.score_threshold = score_threshold

        self.model = fasterrcnn_resnet50_fpn(pretrained=True)
        self.model.eval().to(device)
        self.backbone = self.model.backbone

        self.fc_layers = torch.nn.Sequential(
            torch.nn.Linear(256 * 7 * 7, 2048),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, 2048),
        ).to(device)

        for p in self.parameters():
            p.requires_grad = False

        # COCO class 1 = person
        self.PERSON_LABEL = 1

    @torch.no_grad()
    def forward(self, frame_tensor: torch.Tensor, max_entities: int = 5):
        """
        Args:
            frame_tensor: (3, H, W) normalizat ImageNet
        Returns:
            roi_feats:  (max_entities, 2048)
            boxes:      (max_entities, 4)
            labels:     (max_entities,) int, 1=person(human), restul=object
            scores:     (max_entities,)
        """
        batch = frame_tensor.unsqueeze(0).to(self.device)
        detections = self.model(batch)[0]

        keep = detections["scores"] >= self.score_threshold
        boxes_all = detections["boxes"][keep]
        labels_all = detections["labels"][keep]
        scores_all = detections["scores"][keep]

        if len(boxes_all) > max_entities:
            topk = scores_all.argsort(descending=True)[:max_entities]
            boxes_all = boxes_all[topk]
            labels_all = labels_all[topk]
            scores_all = scores_all[topk]

        num_detected = len(boxes_all)
        H_img, W_img = frame_tensor.shape[1], frame_tensor.shape[2]

        if num_detected == 0:
            boxes_all = torch.tensor([[0., 0., W_img, H_img]], device=self.device)
            labels_all = torch.tensor([1], device=self.device, dtype=torch.int64)
            scores_all = torch.tensor([1.0], device=self.device)
            num_detected = 1

        # ROI Pooling
        feat_dict = self.backbone(batch)
        feature_map = feat_dict["0"]
        spatial_scale = feature_map.shape[2] / H_img

        batch_boxes = torch.cat([
            torch.zeros(len(boxes_all), 1, device=self.device),
            boxes_all
        ], dim=1)

        roi_feats = roi_pool(
            feature_map,
            batch_boxes,
            output_size=(7, 7),
            spatial_scale=spatial_scale,
        )
        roi_feats = roi_feats.flatten(1)
        roi_feats = self.fc_layers(roi_feats)

        # Padding la max_entities
        M = max_entities
        padded_feats = torch.zeros(M, 2048, device=self.device)
        padded_boxes = torch.zeros(M, 4, device=self.device)
        padded_labels = torch.ones(M, device=self.device, dtype=torch.int64)
        padded_scores = torch.zeros(M, device=self.device)

        n = min(num_detected, M)
        padded_feats[:n] = roi_feats[:n]
        padded_boxes[:n] = boxes_all[:n]
        padded_labels[:n] = labels_all[:n]
        padded_scores[:n] = scores_all[:n]

        # Mapare: person (1) -> 0 (human), restul -> 1 (object)
        entity_types = torch.where(padded_labels == self.PERSON_LABEL, 0, 1)

        return padded_feats.cpu(), padded_boxes.cpu(), entity_types.cpu(), padded_scores.cpu()


# ---------------------------------------------------------------------------
# Utilitare geometry
# ---------------------------------------------------------------------------

def bbox_to_keypoints(box: np.ndarray) -> np.ndarray:
    """
    Converteste bbox [x1, y1, x2, y2] in 4 colturi -> (4, 2).
    """
    x1, y1, x2, y2 = box
    return np.array([
        [x1, y1],
        [x2, y1],
        [x2, y2],
        [x1, y2],
    ], dtype=np.float32)


def build_geometry_sequence(
    frames: List[np.ndarray],
    boxes: np.ndarray,           # (S, M, 4)
    entity_types: np.ndarray,    # (S, M)
    mp_extractor: MediaPipeExtractor,
) -> np.ndarray:
    """
    Construieste geo_features (S, J, 4) pentru toate frame-urile.
    """
    S, M = boxes.shape[:2]

    keypoints_list: List[List[Optional[np.ndarray]]] = []

    for s in range(S):
        frame_kp = []
        for m in range(M):
            et = entity_types[s, m]
            box = boxes[s, m]

            if box[2] - box[0] < 1 or box[3] - box[1] < 1:
                frame_kp.append(None)
                continue

            if et == 0:  # human
                kp = mp_extractor.extract(frames[s], box)
                if kp is None:
                    kp = bbox_to_keypoints(box)
                frame_kp.append(kp)
            else:  # object
                kp = bbox_to_keypoints(box)
                frame_kp.append(kp)
        keypoints_list.append(frame_kp)

    K_per_entity = []
    for m in range(M):
        km = 4
        for s in range(S):
            if keypoints_list[s][m] is not None:
                km = keypoints_list[s][m].shape[0]
                break
        K_per_entity.append(km)

    J = sum(K_per_entity)
    all_positions = np.zeros((S, J, 2), dtype=np.float32)

    for s in range(S):
        idx = 0
        for m in range(M):
            km = K_per_entity[m]
            kp = keypoints_list[s][m]
            if kp is not None and kp.shape[0] == km:
                all_positions[s, idx:idx+km] = kp
            idx += km

    velocities = np.zeros_like(all_positions)
    velocities[1:] = all_positions[1:] - all_positions[:-1]

    geo = np.concatenate([all_positions, velocities], axis=-1).astype(np.float32)
    return geo


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def extract_features(
    input_path: str,
    device: str,
    max_entities: int = 5,
    max_frames: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray, List[np.ndarray]]:
    """
    Proceseaza un video sau director de imagini si returneaza tensori gata pentru model.

    Returns:
        roi_features:  (1, S, M, 2048)
        geo_features:  (1, S, J, 4)
        entity_types:  (1, M)
        boxes:         (S, M, 4) — pentru vizualizare
        frames:        lista frame-uri BGR
    """
    reader = VideoReader(input_path, max_frames=max_frames)
    frames = reader.read_frames()
    if not frames:
        raise ValueError(f"Nu am gasit frame-uri in: {input_path}")

    S = len(frames)
    print(f"Procesez {S} frame-uri...")

    imagenet_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    rcnn = InferenceRCNN(device=device).to(device)
    mp_extractor = MediaPipeExtractor(static_image_mode=(S == 1))

    roi_all = np.zeros((S, max_entities, 2048), dtype=np.float32)
    boxes_all = np.zeros((S, max_entities, 4), dtype=np.float32)
    etypes_all = np.ones((S, max_entities), dtype=np.int64)

    for s, frame_bgr in enumerate(frames):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_tensor = imagenet_transform(frame_rgb)

        roi_feats, boxes, etypes, _ = rcnn(frame_tensor, max_entities=max_entities)

        roi_all[s] = roi_feats.numpy()
        boxes_all[s] = boxes.numpy()
        etypes_all[s] = etypes.numpy()

    geo_all = build_geometry_sequence(frames, boxes_all, etypes_all, mp_extractor)
    mp_extractor.close()

    roi_t = torch.FloatTensor(roi_all).unsqueeze(0)
    geo_t = torch.FloatTensor(geo_all).unsqueeze(0)
    etypes_t = torch.LongTensor(etypes_all[0]).unsqueeze(0)

    return roi_t, geo_t, etypes_t, boxes_all, frames


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
    # Input: unul din --video-id sau --input este obligatoriu
    parser.add_argument("--video-id", type=str, default=None,
                        help="Video ID din dataset (ex: Subject12-task_1_cheering-take_0). "
                             "Incarca features pre-extrase din data-root, identic cu training.")
    parser.add_argument("--data-root", type=str, default="data/mphoi72/",
                        help="Directorul radacina al dataset-ului (folosit cu --video-id)")
    parser.add_argument("--input", type=str, default=None,
                        help="Path video sau director imagini (extragere live, nu recomandat)")
    parser.add_argument("--output", type=str, default="inference_results.json", help="Fisier output JSON")
    parser.add_argument("--visualize", action="store_true", help="Genereaza video annotat + timeline PNG")
    parser.add_argument("--video-out", type=str, default=None,
                        help="Path video output (default: <output>.mp4)")
    parser.add_argument("--max-entities", type=int, default=5)
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
    if args.video_id is None and args.input is None:
        raise ValueError("Specifica --video-id (recomandat) sau --input.")
    if args.video_id and args.input:
        raise ValueError("Specifica doar unul din --video-id sau --input, nu ambele.")
    if args.checkpoint is None and args.checkpoint_pattern is None:
        raise ValueError("Specifica --checkpoint sau --checkpoint-pattern.")
    if args.checkpoint is not None and args.checkpoint_pattern is not None:
        print("[WARN] Ambele --checkpoint si --checkpoint-pattern specificate. Folosesc --checkpoint-pattern.")

    cfg = OmegaConf.merge(
        OmegaConf.load("configs/base.yaml"),
        OmegaConf.load(args.config),
    )

    dataset_name = args.dataset_name or cfg.dataset.name
    label_names = get_label_names(dataset_name)

    print(f"Dataset: {dataset_name} | Clase: {label_names}")
    print(f"Device: {device}")

    # -----------------------------------------------------------------------
    # Incarcare features
    # -----------------------------------------------------------------------
    boxes_all = None  # pentru vizualizare (doar modul --input)
    frames = None     # pentru vizualizare (doar modul --input)

    if args.video_id:
        # Mod RECOMANDAT: features pre-extrase din dataset (identic cu training)
        print(f"\n[1/3] Incarcare features pre-extrase pentru: {args.video_id}")
        data = load_preextracted_features(
            args.video_id, args.data_root, clip_dim=cfg.model.clip_dim,
        )
        roi_features  = data["roi_features"].to(device)
        geo_features  = data["geo_features"].to(device)
        entity_types   = data["entity_types"].to(device)
        clip_features  = data["clip_features"].to(device)
        gt_labels      = data["gt_labels"]
        input_label    = args.video_id
    else:
        # Mod live: extragere cu Faster R-CNN + MediaPipe (poate avea mismatch-uri)
        print(f"\n[1/3] Extragere features ROI + geometrie (live)...")
        print("[WARN] Modul live poate produce rezultate diferite fata de training din cauza mismatch-urilor.")
        roi, geo, etypes, boxes_all, frames = extract_features(
            args.input,
            device=str(device),
            max_entities=args.max_entities,
            max_frames=args.max_frames,
        )
        roi_features = roi.to(device)
        geo_features = geo.to(device)
        entity_types  = etypes.to(device)
        clip_features = None
        gt_labels      = None
        input_label    = args.input

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

    # Vizualizare (doar modul --input)
    if args.visualize and frames is not None and boxes_all is not None:
        print("\n[4/4] Generare vizualizare...")
        video_out = args.video_out or args.output.replace(".json", ".mp4")
        if video_out == args.output:
            video_out = "inference_visualization.mp4"

        save_visualization_video(frames, boxes_all, pred_classes, label_names, video_out, fps=args.fps)

        raw_path = args.output.replace(".json", "_raw.npz")
        save_raw_data_npz(raw_path, frames, boxes_all, pred_classes, entity_types.cpu().numpy())


if __name__ == "__main__":
    main()
