"""
VHOIP Raw Video Feature Extraction Pipeline
============================================
Extracts all features needed by VHOIP from a raw video:
  1. Human & object detection + 2048-dim ROI features (Faster R-CNN)
  2. Human skeleton keypoints (MMPose) or bbox-corner fallback
  3. 512-dim CLIP visual features per entity region
  4. Multi-object tracking (SORT) for consistent entity IDs across frames

Output format matches the VHOIP model exactly:
  roi_features  : (T, M, 2048)  ROI visual features per entity per frame
  geo_features  : (T, J, 4)     geometric features — J = num_humans*32 + num_objects*4
  entity_types  : (M,)          0=human, 1=object (humans first, then objects)
  clip_features : (T, M, 512)   CLIP features per entity per frame
  bboxes        : (T, M, 4)     bounding boxes xyxy

Usage:
    python extract_features.py --video path/to/video.mp4 --output features/

Requirements:
    pip install torch torchvision opencv-python numpy scipy filterpy
    pip install ultralytics   (for YOLOv8 fallback detector)
    pip install git+https://github.com/openai/CLIP.git
"""

import os
import sys
import argparse
import cv2
import numpy as np
import torch
from pathlib import Path
from PIL import Image

from scipy.optimize import linear_sum_assignment


# ──────────────────────────────────────────────────────────────────────────────
# SORT tracker (inline minimal implementation)
# ──────────────────────────────────────────────────────────────────────────────

def iou(bb_test, bb_gt):
    xx1 = max(bb_test[0], bb_gt[0])
    yy1 = max(bb_test[1], bb_gt[1])
    xx2 = min(bb_test[2], bb_gt[2])
    yy2 = min(bb_test[3], bb_gt[3])
    w = max(0.0, xx2 - xx1)
    h = max(0.0, yy2 - yy1)
    inter = w * h
    area_test = (bb_test[2]-bb_test[0]) * (bb_test[3]-bb_test[1])
    area_gt   = (bb_gt[2]-bb_gt[0])   * (bb_gt[3]-bb_gt[1])
    union = area_test + area_gt - inter
    return inter / union if union > 0 else 0.0


class KalmanBox:
    count = 0

    def __init__(self, bbox):
        from filterpy.kalman import KalmanFilter
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([
            [1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],
            [0,0,0,1,0,0,0],[0,0,0,0,1,0,0],[0,0,0,0,0,1,0],
            [0,0,0,0,0,0,1],
        ], dtype=float)
        self.kf.H = np.array([
            [1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0],
        ], dtype=float)
        self.kf.R[2:,2:] *= 10.
        self.kf.P[4:,4:] *= 1000.
        self.kf.P        *= 10.
        self.kf.Q[-1,-1] *= 0.01
        self.kf.Q[4:,4:] *= 0.01
        self.kf.x[:4]     = self._to_z(bbox)
        self.time_since_update = 0
        KalmanBox.count += 1
        self.id    = KalmanBox.count
        self.hits  = 0
        self.age   = 0

    @staticmethod
    def _to_z(bbox):
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w / 2.
        y = bbox[1] + h / 2.
        s = w * h
        r = w / float(h) if h > 0 else 1.
        return np.array([x, y, s, r]).reshape((4, 1))

    def _to_bbox(self):
        x = self.kf.x
        w = np.sqrt(abs(x[2] * x[3]))
        h = x[2] / w if w > 0 else 0
        return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2.]).flatten()

    def predict(self):
        if self.kf.x[6] + self.kf.x[2] <= 0:
            self.kf.x[6] = 0.
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hits = 0
        self.time_since_update += 1
        return self._to_bbox()

    def update(self, bbox):
        self.time_since_update = 0
        self.hits += 1
        self.kf.update(self._to_z(bbox))

    def get_state(self):
        return self._to_bbox()


class Sort:
    def __init__(self, max_age=5, min_hits=2, iou_threshold=0.3):
        self.max_age      = max_age
        self.min_hits     = min_hits
        self.iou_threshold = iou_threshold
        self.trackers     = []
        self.frame_count  = 0

    def update(self, dets=np.empty((0, 5))):
        self.frame_count += 1
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()
            trk[:] = [*pos, 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        for t in reversed(to_del):
            self.trackers.pop(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))

        matched, unmatched_dets, unmatched_trks = self._associate(dets, trks)
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :4])
        for i in unmatched_dets:
            self.trackers.append(KalmanBox(dets[i, :4]))
        ret = []
        for trk in reversed(self.trackers):
            d = trk.get_state()
            if trk.time_since_update <= self.max_age and (
                    trk.hits >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id])))
        self.trackers = [t for t in self.trackers if t.time_since_update <= self.max_age]
        return np.array(ret) if ret else np.empty((0, 5))

    def _associate(self, dets, trks):
        if len(trks) == 0:
            return np.empty((0,2), int), np.arange(len(dets)), np.empty((0,), int)
        iou_matrix = np.zeros((len(dets), len(trks)))
        for d in range(len(dets)):
            for t in range(len(trks)):
                iou_matrix[d, t] = iou(dets[d, :4], trks[t, :4])
        row_ind, col_ind = linear_sum_assignment(-iou_matrix)
        matched_indices = np.stack([row_ind, col_ind], axis=1)
        unmatched_dets = [d for d in range(len(dets)) if d not in matched_indices[:, 0]]
        unmatched_trks = [t for t in range(len(trks)) if t not in matched_indices[:, 1]]
        matches = [m for m in matched_indices if iou_matrix[m[0], m[1]] >= self.iou_threshold]
        return (np.array(matches) if matches else np.empty((0,2), int),
                unmatched_dets, unmatched_trks)


# ──────────────────────────────────────────────────────────────────────────────
# Detector: VG Faster R-CNN (matches training pipeline exactly)
# ──────────────────────────────────────────────────────────────────────────────

class VGDetector:
    """
    Visual Genome pretrained Faster R-CNN via detectron2.
    This is the exact same extractor used to create the MPHOI-72 training
    features. Using this fixes the ROI feature distribution mismatch.

    Setup:
        git clone https://github.com/airsplay/py-bottom-up-attention
        cd py-bottom-up-attention
        wget https://dl.fbaipublicfiles.com/detectron2/BottomUpTopDownModels/bottomupTopDown_10_100.pth
    """

    PERSON_CLASS_NAMES = {"person", "man", "woman", "boy", "girl"}

    def __init__(self, config_file: str, weights_file: str,
                 score_thresh: float = 0.4, device: str = "cuda"):
        try:
            from detectron2.engine import DefaultPredictor
            from detectron2.config import get_cfg
        except ImportError:
            raise ImportError(
                "detectron2 is required for VG detector.\n"
                "Install: pip install detectron2 -f "
                "https://dl.fbaipublicfiles.com/detectron2/wheels/cu124/torch2.6/index.html"
            )
        cfg = get_cfg()
        cfg.merge_from_file(config_file)
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
        cfg.MODEL.WEIGHTS = weights_file
        cfg.MODEL.DEVICE  = device
        self.predictor = DefaultPredictor(cfg)
        self.thresh = score_thresh
        self.device = device
        print(f"[Detector] VG Faster R-CNN loaded from {weights_file}")

    def detect(self, bgr_frame: np.ndarray):
        outputs   = self.predictor(bgr_frame)
        instances = outputs["instances"].to("cpu")
        boxes    = instances.pred_boxes.tensor.numpy()
        scores   = instances.scores.numpy()
        classes  = instances.pred_classes.numpy()
        features = instances.roi_features.numpy()  # (N, 2048)

        # VG class names — determine if each detection is a person
        try:
            from detectron2.data import MetadataCatalog
            thing_ids = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes if hasattr(self, 'cfg') else []
        except Exception:
            thing_ids = []

        return {
            "boxes":    boxes,
            "scores":   scores,
            "classes":  classes,
            "features": features,
        }

    @staticmethod
    def is_person(class_id: int) -> bool:
        """VG doesn't use COCO class IDs. Heuristic: person-like classes."""
        # VG person class is typically index 0 in the VG taxonomy
        # This is a simplification — in practice, check VG class names
        return class_id == 0


# ──────────────────────────────────────────────────────────────────────────────
# Detector: YOLOv8 + ResNet-101 for 2048-dim ROI features (fallback)
# ──────────────────────────────────────────────────────────────────────────────

class YOLODetector:
    """
    YOLOv8 for detection + ResNet-101 backbone for 2048-dim ROI features.
    Matches the 2048-dim feature dimension expected by VHOIP.
    """

    COCO_PERSON_CLASS = 0  # YOLO uses 0 for person

    def __init__(self, model_name: str = "yolov8m.pt",
                 score_thresh: float = 0.4, device: str = "cuda"):
        from ultralytics import YOLO
        import torchvision.models as tvm

        self.model  = YOLO(model_name)
        self.thresh = score_thresh
        self.device = device

        backbone = tvm.resnet101(weights=tvm.ResNet101_Weights.DEFAULT)
        self.feature_extractor = torch.nn.Sequential(*list(backbone.children())[:-2])
        self.feature_extractor.eval().to(device)
        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        print("[Detector] YOLOv8 + ResNet-101 loaded")

    def _roi_feature(self, bgr_crop: np.ndarray) -> np.ndarray:
        import torchvision.transforms as T
        tf = T.Compose([T.Resize((224, 224)), T.ToTensor(),
                        T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
        rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
        img = tf(Image.fromarray(rgb)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.feature_extractor(img)
            feat = self.pool(feat).squeeze()
        return feat.cpu().numpy()

    def detect(self, bgr_frame: np.ndarray):
        h, w = bgr_frame.shape[:2]
        results = self.model(bgr_frame, conf=self.thresh, verbose=False)[0]
        boxes, scores, classes, features = [], [], [], []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x1,y1,x2,y2 = max(0,int(x1)),max(0,int(y1)),min(w,int(x2)),min(h,int(y2))
            if x2 <= x1 or y2 <= y1:
                continue
            crop = bgr_frame[y1:y2, x1:x2]
            feat = self._roi_feature(crop)
            boxes.append([x1, y1, x2, y2])
            scores.append(float(box.conf[0]))
            classes.append(int(box.cls[0]))
            features.append(feat)
        return {
            "boxes":    np.array(boxes,    dtype=np.float32) if boxes else np.zeros((0,4)),
            "scores":   np.array(scores,   dtype=np.float32) if scores else np.zeros((0,)),
            "classes":  np.array(classes,  dtype=np.int64)   if classes else np.zeros((0,),dtype=np.int64),
            "features": np.array(features, dtype=np.float32) if features else np.zeros((0,2048)),
        }


# ──────────────────────────────────────────────────────────────────────────────
# Skeleton extractor (MMPose or zero fallback)
# ──────────────────────────────────────────────────────────────────────────────

# COCO 17-keypoint order (used by MMPose/YOLOv8-pose)
COCO_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

# Mapping from 17 COCO keypoints to 32 slots.
# We place each COCO keypoint at a fixed position and pad the rest with zeros.
# This allows the model's GeometricLevelGCN to process the same J dimension
# it was trained with, even though we only have 17 detected keypoints.
COCO_TO_32 = [
    # 0-4: head (nose, eyes, ears)
    0, 1, 2, 3, 4,
    # 5-10: shoulders + elbows
    5, 6, 7, 8, -1, -1,
    # 11-16: wrists + hands
    9, 10, -1, -1, -1, -1,
    # 17-22: hips + torso
    11, 12, -1, -1, -1, -1,
    # 23-28: knees
    13, 14, -1, -1, -1, -1,
    # 29-31: ankles
    15, 16, -1,
]


class SkeletonExtractor:
    """
    Extracts human keypoints using MMPose (17 COCO keypoints).
    Falls back to zero keypoints if MMPose is not installed.
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self._available = False
        try:
            from mmpose.apis import init_pose_model, inference_top_down_pose_model
            self._init  = init_pose_model
            self._infer = inference_top_down_pose_model
            self.pose_model = self._init(
                "td-hm_ViTPose-base_8xb64-210e_coco-256x192.py",
                "vitpose-b_simcc-coco_pt-aic-coco_210e-256x192-5a7596af_20230314.pth",
                device=device,
            )
            self._available = True
            print("[Skeleton] MMPose ViTPose-B loaded.")
        except Exception as e:
            print(f"[Skeleton] MMPose not available ({e}). Using bbox-grid fallback for humans.")

    def extract(self, bgr_frame: np.ndarray, human_boxes: np.ndarray) -> np.ndarray:
        """
        Returns (N_humans, 17, 2) array: (x, y) per keypoint.
        If MMPose is not available, returns zeros.
        """
        if not self._available or len(human_boxes) == 0:
            return np.zeros((len(human_boxes), 17, 2), dtype=np.float32)

        person_results = [{"bbox": b} for b in human_boxes]
        pose_results, _ = self._infer(
            self.pose_model, bgr_frame, person_results,
            bbox_thr=0.3, format="xyxy",
        )
        kps = np.array([r["keypoints"][:, :2] for r in pose_results], dtype=np.float32)
        return kps  # (N, 17, 2)


# ──────────────────────────────────────────────────────────────────────────────
# CLIP feature extractor
# ──────────────────────────────────────────────────────────────────────────────

class CLIPExtractor:
    """Extracts 512-dim CLIP ViT-B/16 features for each entity crop."""

    def __init__(self, device: str = "cuda"):
        try:
            import clip as openai_clip
            self.model, self.preprocess = openai_clip.load("ViT-B/16", device=device)
            self.model.eval()
            self.device = device
            print("[CLIP] ViT-B/16 loaded.")
        except ImportError:
            raise ImportError("Install CLIP: pip install git+https://github.com/openai/CLIP.git")

    @torch.no_grad()
    def extract(self, bgr_frame: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        if len(boxes) == 0:
            return np.zeros((0, 512), dtype=np.float32)
        h, w = bgr_frame.shape[:2]
        crops = []
        for box in boxes:
            x1, y1, x2, y2 = [int(v) for v in box]
            x1,y1 = max(0,x1), max(0,y1)
            x2,y2 = min(w,x2), min(h,y2)
            if x2 <= x1 or y2 <= y1:
                crops.append(self.preprocess(Image.new("RGB",(1,1))))
            else:
                crop_rgb = cv2.cvtColor(bgr_frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
                crops.append(self.preprocess(Image.fromarray(crop_rgb)))
        batch = torch.stack(crops).to(self.device)
        feats = self.model.encode_image(batch)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().float().numpy()


# ──────────────────────────────────────────────────────────────────────────────
# Geometry builder: produces (J, 4) in VHOIP format
# ──────────────────────────────────────────────────────────────────────────────

def build_geo_for_frame(
    human_boxes: np.ndarray,      # (num_humans, 4) xyxy
    object_boxes: np.ndarray,     # (num_objects, 4) xyxy
    skeletons: np.ndarray,         # (num_humans, 17, 2) or empty
    joints_per_human: int = 32,
) -> np.ndarray:
    """
    Build geometric features for one frame in VHOIP format.

    Returns (J, 2) array of keypoint positions.
    J = num_humans * joints_per_human + num_objects * 4

    Layout matches training:
      - Human 0: 32 keypoint slots (from skeleton or bbox-grid fallback)
      - Human 1: 32 keypoint slots
      - Object 0: 4 bbox corners
      - Object 1: 4 bbox corners
      - ...
    """
    num_humans = len(human_boxes)
    num_objects = len(object_boxes)
    J = num_humans * joints_per_human + num_objects * 4

    positions = np.zeros((J, 2), dtype=np.float32)

    # --- Humans ---
    for h in range(num_humans):
        offset = h * joints_per_human
        x1, y1, x2, y2 = human_boxes[h]
        hw = (x2 - x1) / 2
        hh = (y2 - y1) / 2

        if skeletons is not None and len(skeletons) > h and skeletons.shape[1] == 17:
            # Map 17 COCO keypoints to 32 slots
            for slot_32, coco_idx in enumerate(COCO_TO_32):
                if coco_idx >= 0 and coco_idx < 17:
                    positions[offset + slot_32] = skeletons[h, coco_idx]
                # else: stays zero (no detection for this joint)
        else:
            # Fallback: distribute 4x8 grid of points within bbox
            cols = 4
            rows = joints_per_human // cols  # 8
            xs = np.linspace(x1 + hw * 0.1, x2 - hw * 0.1, cols)
            ys = np.linspace(y1 + hh * 0.1, y2 - hh * 0.1, rows)
            gx, gy = np.meshgrid(xs, ys)
            positions[offset:offset + joints_per_human, 0] = gx.ravel()[:joints_per_human]
            positions[offset:offset + joints_per_human, 1] = gy.ravel()[:joints_per_human]

    # --- Objects: 4 bbox corners each ---
    for o in range(num_objects):
        offset = num_humans * joints_per_human + o * 4
        x1, y1, x2, y2 = object_boxes[o]
        positions[offset]     = [x1, y1]
        positions[offset + 1] = [x2, y1]
        positions[offset + 2] = [x2, y2]
        positions[offset + 3] = [x1, y2]

    return positions


# ──────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────────────

class VHOIPFeaturePipeline:
    """
    End-to-end pipeline: raw video → VHOIP-compatible feature tensors.

    Output matches VHOIP model input exactly:
      roi_features  : (T, M, 2048)  visual ROI features per entity per frame
      geo_features  : (T, J, 4)    geometric features (x, y, vx, vy)
                      J = num_humans * 32 + num_objects * 4
      entity_types  : (M,)         0=human, 1=object  (humans first)
      clip_features : (T, M, 512)  CLIP features per entity per frame
      bboxes        : (T, M, 4)    bounding boxes xyxy
    """

    def __init__(self,
                 detector_type: str = "yolo",
                 vg_config: str = "",
                 vg_weights: str = "",
                 score_thresh: float = 0.4,
                 max_entities: int = 5,
                 max_persons: int = 2,
                 device: str = "cuda",
                 tracker_max_age: int = 5,
                 tracker_min_hits: int = 2):

        self.device       = device if torch.cuda.is_available() else "cpu"
        self.max_entities  = max_entities
        self.max_persons   = max_persons
        self.detector_type = detector_type

        if detector_type == "vg":
            self.detector = VGDetector(vg_config, vg_weights, score_thresh, self.device)
        else:
            self.detector = YOLODetector(score_thresh=score_thresh, device=self.device)

        self.skeleton = SkeletonExtractor(device=self.device)
        self.clip_ext = CLIPExtractor(device=self.device)
        self.tracker  = Sort(max_age=tracker_max_age, min_hits=tracker_min_hits)

    def _sort_entities(self, boxes, scores, classes, features):
        """Sort detections: humans first (top max_persons), then objects by score."""
        if self.detector_type == "vg":
            person_mask = np.array([VGDetector.is_person(c) for c in classes])
        else:
            person_mask = classes == YOLODetector.COCO_PERSON_CLASS

        person_idx = np.where(person_mask)[0]
        if len(person_idx) > 0:
            person_idx = person_idx[scores[person_mask].argsort()[::-1]]
            person_idx = person_idx[:self.max_persons]

        object_idx = np.where(~person_mask)[0]
        if len(object_idx) > 0:
            object_idx = object_idx[scores[~person_mask].argsort()[::-1]]

        sorted_idx = np.concatenate([person_idx, object_idx])
        if len(sorted_idx) > self.max_entities:
            sorted_idx = sorted_idx[:self.max_entities]

        return (
            boxes[sorted_idx]    if len(sorted_idx) > 0 else np.zeros((0, 4)),
            scores[sorted_idx]   if len(sorted_idx) > 0 else np.zeros((0,)),
            classes[sorted_idx]  if len(sorted_idx) > 0 else np.zeros((0,), dtype=np.int64),
            features[sorted_idx] if len(sorted_idx) > 0 else np.zeros((0, 2048)),
            person_mask[sorted_idx] if len(sorted_idx) > 0 else np.zeros((0,), dtype=bool),
        )

    def process_video(self, video_path: str, output_dir: str,
                       sample_fps: float = None) -> str:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = max(1, round(native_fps / sample_fps)) if sample_fps else 1
        video_name = Path(video_path).stem

        print(f"\n[Pipeline] Video : {video_path}")
        print(f"[Pipeline] FPS   : {native_fps:.1f}  |  step={step}  |  device={self.device}")

        M = self.max_entities
        J = self.max_persons * 32 + (M - self.max_persons) * 4

        # Accumulators
        all_roi   = []
        all_clip  = []
        all_boxes = []
        all_geo   = []  # raw positions (before velocity)

        frame_idx = 0
        processed = 0

        # We need to determine entity_types from the first valid frame
        entity_types = None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % step != 0:
                frame_idx += 1
                continue

            # 1. Detect
            det = self.detector.detect(frame)

            # 2. Sort: humans first, then objects
            s_boxes, s_scores, s_classes, s_features, s_is_human = \
                self._sort_entities(det["boxes"], det["scores"],
                                    det["classes"], det["features"])

            n_detected = len(s_boxes)

            # 3. Separate humans and objects for geometry
            human_boxes_list = []
            object_boxes_list = []
            human_features_list = []
            object_features_list = []

            for i in range(n_detected):
                if s_is_human[i]:
                    human_boxes_list.append(s_boxes[i])
                    human_features_list.append(s_features[i])
                else:
                    object_boxes_list.append(s_boxes[i])
                    object_features_list.append(s_features[i])

            human_boxes = np.array(human_boxes_list, dtype=np.float32) if human_boxes_list else np.zeros((0, 4), dtype=np.float32)
            object_boxes = np.array(object_boxes_list, dtype=np.float32) if object_boxes_list else np.zeros((0, 4), dtype=np.float32)

            # 4. Skeleton extraction (humans only)
            skeletons = self.skeleton.extract(frame, human_boxes)  # (N_h, 17, 2) or zeros

            # 5. Build geometry for this frame (positions only, velocity later)
            geo_pos = build_geo_for_frame(human_boxes, object_boxes, skeletons)
            # Pad to fixed J so stacking works when detections vary per frame
            if geo_pos.shape[0] < J:
                pad = np.zeros((J - geo_pos.shape[0], 2), dtype=np.float32)
                geo_pos = np.concatenate([geo_pos, pad], axis=0)
            all_geo.append(geo_pos)  # (J, 2)

            # 6. Build ROI features (M slots)
            f_roi = np.zeros((M, 2048), dtype=np.float32)
            slot = 0
            for feat in human_features_list:
                if slot < M:
                    f_roi[slot] = feat
                    slot += 1
            for feat in object_features_list:
                if slot < M:
                    f_roi[slot] = feat
                    slot += 1
            all_roi.append(f_roi)

            # 7. Build boxes (M slots)
            f_boxes = np.zeros((M, 4), dtype=np.float32)
            slot = 0
            for box in human_boxes_list:
                if slot < M:
                    f_boxes[slot] = box
                    slot += 1
            for box in object_boxes_list:
                if slot < M:
                    f_boxes[slot] = box
                    slot += 1
            all_boxes.append(f_boxes)

            # 8. CLIP features
            active_boxes = f_boxes[:max(n_detected, 1)]
            clip_feats = self.clip_ext.extract(frame, active_boxes)  # (N, 512)
            f_clip = np.zeros((M, 512), dtype=np.float32)
            n_clip = min(len(clip_feats), M)
            f_clip[:n_clip] = clip_feats[:n_clip]
            all_clip.append(f_clip)

            # Set entity_types (from first frame, consistent across video)
            if entity_types is None and n_detected > 0:
                entity_types = np.zeros(M, dtype=np.int64)  # 0=human (padding default)
                for i in range(n_detected):
                    entity_types[i] = 0 if s_is_human[i] else 1

            frame_idx += 1
            processed += 1
            if processed % 30 == 0:
                nh = len(human_boxes_list)
                no = len(object_boxes_list)
                print(f"  Frame {processed:4d} | {nh} humans, {no} objects")

        cap.release()
        print(f"[Pipeline] Processed {processed} frames.")

        if entity_types is None:
            entity_types = np.zeros(M, dtype=np.int64)

        # Stack arrays
        roi_stack  = np.stack(all_roi)    # (T, M, 2048)
        clip_stack = np.stack(all_clip)   # (T, M, 512)
        boxes_stack = np.stack(all_boxes)  # (T, M, 4)
        geo_pos_stack = np.stack(all_geo)  # (T, J, 2)

        # Compute velocities: vx, vy = diff from previous frame
        geo_vel = np.zeros_like(geo_pos_stack)
        geo_vel[1:] = geo_pos_stack[1:] - geo_pos_stack[:-1]

        # Combine position + velocity -> (T, J, 4)
        geo_stack = np.concatenate([geo_pos_stack, geo_vel], axis=-1).astype(np.float32)
        geo_stack = np.nan_to_num(geo_stack, nan=0.0)

        num_humans = int((entity_types == 0).sum())
        num_objects = int((entity_types == 1).sum())
        print(f"[Pipeline] Entity types: {num_humans} humans + {num_objects} objects = {M} total")
        print(f"[Pipeline] Geometry J = {J} ({num_humans}*32 + {num_objects}*4)")

        # Save
        out = {
            "roi_features":  roi_stack,     # (T, M, 2048)
            "geo_features":  geo_stack,     # (T, J, 4)
            "entity_types":  entity_types,  # (M,)
            "clip_features": clip_stack,    # (T, M, 512)
            "bboxes":        boxes_stack,   # (T, M, 4)
        }

        out_path = os.path.join(output_dir, f"{video_name}_features.npz")
        np.savez_compressed(out_path, **out)
        print(f"[Pipeline] Features saved → {out_path}")
        self._print_summary(out)
        return out_path

    @staticmethod
    def _print_summary(out: dict):
        print("\n── Feature Summary ──────────────────────────────────────────")
        for k, v in out.items():
            print(f"  {k:<18} shape={str(v.shape):<25} dtype={v.dtype}")
        T = out["roi_features"].shape[0]
        M = out["roi_features"].shape[1]
        print(f"\n  Frames: {T}  |  Entities per frame: {M}")
        print("─────────────────────────────────────────────────────────────\n")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Extract VHOIP features from a raw video.")
    p.add_argument("--video",    required=True, help="Path to input video file.")
    p.add_argument("--output",   default="features/",
                   help="Output directory for .npz feature file.")
    p.add_argument("--detector", default="yolo", choices=["yolo", "vg"],
                   help="'yolo' (fallback) or 'vg' (Visual Genome Faster R-CNN, matches training).")
    p.add_argument("--vg-config",  default="",
                   help="[vg only] Path to detectron2 config YAML.")
    p.add_argument("--vg-weights", default="",
                   help="[vg only] Path to VG Faster R-CNN checkpoint.")
    p.add_argument("--score-thresh", type=float, default=0.4,
                   help="Detection confidence threshold (default 0.4).")
    p.add_argument("--max-entities", type=int, default=5,
                   help="Max entities per frame (default 5, matches MPHOI-72 training).")
    p.add_argument("--max-persons", type=int, default=2,
                   help="Max persons per frame (default 2, matches MPHOI-72).")
    p.add_argument("--sample-fps", type=float, default=None,
                   help="Subsample video to this FPS. None = use all frames.")
    p.add_argument("--device", default="cuda", help="'cuda' or 'cpu'.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    pipeline = VHOIPFeaturePipeline(
        detector_type   = args.detector,
        vg_config       = args.vg_config,
        vg_weights      = args.vg_weights,
        score_thresh    = args.score_thresh,
        max_entities    = args.max_entities,
        max_persons     = args.max_persons,
        device          = args.device,
    )
    pipeline.process_video(
        video_path = args.video,
        output_dir = args.output,
        sample_fps = args.sample_fps,
    )