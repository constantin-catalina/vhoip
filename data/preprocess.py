"""
data/preprocess.py
Extragere offline a features necesare pentru antrenare:
  1. Detectie entitati cu Faster R-CNN (pre-antrenat pe Visual Genome)
  2. Extragere ROI pooling features (2048-dim) per entitate per frame
  3. Extragere features CLIP vizuale (512-dim) per entitate per frame
  4. Salvare ca fisiere .npy in data/<dataset>/features/

Acest script se ruleaza O SINGURA DATA inainte de antrenare.
Rezultatele se salveaza pe disc si se reutilizeaza la fiecare epoch.

Utilizare:
    python data/preprocess.py --dataset cad120 --data_root data/cad120/
    python data/preprocess.py --dataset mphoi72 --data_root data/mphoi72/
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.ops import roi_pool
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm
import cv2
import json
import clip


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def get_imagenet_transform():
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

# ---------------------------------------------------------------------------
# Faster R-CNN cu ROI Pooling
# ---------------------------------------------------------------------------

class FasterRCNNExtractor(nn.Module):
    """
    Extrage features ROI (2048-dim) folosind Faster R-CNN
    pre-antrenat pe Visual Genome (conform paper-ului).

    Nota: paper-ul foloseste un model antrenat pe Visual Genome.
    Aici folosim ResNet50-FPN pre-antrenat pe COCO ca aproximatie
    (disponibil in torchvision fara download suplimentar).
    Pentru reproducere exacta, descarca modelul Visual Genome de la:
    https://github.com/airsplay/py-bottom-up-attention
    """

    def __init__(self, device: str = "cuda", score_threshold: float = 0.3):
        super().__init__()
        self.device = device
        self.score_threshold = score_threshold

        # Incarca Faster R-CNN pre-antrenat
        print("  Incarcare Faster R-CNN (ResNet50-FPN)...")
        self.model = fasterrcnn_resnet50_fpn(pretrained=True)
        self.model.eval()
        self.model.to(device)

        # Extragem backbone-ul pentru ROI features
        self.backbone = self.model.backbone

        # ROI Align/Pool la dimensiunea necesara
        self.roi_output_size = (7, 7)    # standard pentru ResNet50
        self.roi_pool_dim = 2048         # dupa fc layers

        # FC layers pentru a obtine 2048-dim features din ROI
        self.fc_layers = nn.Sequential(
            nn.Linear(256 * 7 * 7, 2048),   # 256 = FPN output channels
            nn.ReLU(),
            nn.Linear(2048, 2048),
        ).to(device)

        # Freeze tot - nu antrenam Faster R-CNN
        for param in self.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def detect_and_extract(
        self,
        frame: torch.Tensor,
        max_entities: int = 5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Detecteaza entitati si extrage ROI features dintr-un frame.

        Args:
            frame: (3, H, W) tensor normalizat ImageNet
            max_entities: numarul maxim de entitati de returnat

        Returns:
            roi_features: (max_entities, 2048) - padded cu zeros daca < max_entities
            boxes:        (max_entities, 4)    - bounding boxes [x1, y1, x2, y2]
        """
        frame_batch = frame.unsqueeze(0).to(self.device)   # (1, 3, H, W)

        # Detectie
        detections = self.model(frame_batch)[0]

        # Filtreaza dupa scor
        keep = detections["scores"] >= self.score_threshold
        boxes = detections["boxes"][keep]       # (K, 4)
        scores = detections["scores"][keep]     # (K,)

        # Sorteaza dupa scor si pastreaza primele max_entities
        if len(boxes) > max_entities:
            topk = scores.argsort(descending=True)[:max_entities]
            boxes = boxes[topk]

        num_detected = len(boxes)

        # Extrage features din backbone (FPN)
        features_dict = self.backbone(frame_batch)
        # Folosim feature map-ul de la scala 0 (cea mai detaliata)
        feature_map = features_dict["0"]   # (1, 256, H/4, W/4)

        H_img, W_img = frame.shape[1], frame.shape[2]

        if num_detected == 0:
            # Fallback: foloseste intregul frame ca o singura entitate
            boxes = torch.tensor([[0., 0., W_img, H_img]], device=self.device)
            num_detected = 1

        # ROI Pooling
        # boxes trebuie sa fie in format [batch_idx, x1, y1, x2, y2]
        batch_boxes = torch.cat([
            torch.zeros(len(boxes), 1, device=self.device),
            boxes
        ], dim=1)   # (K, 5)

        spatial_scale = feature_map.shape[2] / H_img
        roi_feats = roi_pool(
            feature_map,
            batch_boxes,
            output_size=self.roi_output_size,
            spatial_scale=spatial_scale,
        )   # (K, 256, 7, 7)

        # Flatten si proiecteaza la 2048
        roi_feats = roi_feats.flatten(1)          # (K, 256*7*7)
        roi_feats = self.fc_layers(roi_feats)     # (K, 2048)

        # Padding la max_entities
        padded_feats = torch.zeros(max_entities, 2048, device=self.device)
        padded_boxes = torch.zeros(max_entities, 4, device=self.device)

        n = min(num_detected, max_entities)
        padded_feats[:n] = roi_feats[:n]
        padded_boxes[:n] = boxes[:n]

        return padded_feats.cpu(), padded_boxes.cpu()


# ---------------------------------------------------------------------------
# CLIP Visual Extractor
# ---------------------------------------------------------------------------

class CLIPExtractor:
    """
    Extrage features CLIP vizuale (512-dim) din crop-uri de entitati.
    """

    def __init__(self, model_name: str = "ViT-B/16", device: str = "cuda"):
        self.device = device
        print(f"  Incarcare CLIP ({model_name})...")
        self.model, self.preprocess = clip.load(model_name, device=device)
        self.model.eval()

        for param in self.model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def extract_from_boxes(
        self,
        frame_bgr: np.ndarray,
        boxes: torch.Tensor,
        max_entities: int = 5,
    ) -> torch.Tensor:
        """
        Extrage features CLIP din crop-urile corespunzatoare box-urilor.

        Args:
            frame_bgr: (H, W, 3) numpy array BGR (OpenCV format)
            boxes:     (max_entities, 4) bounding boxes [x1, y1, x2, y2]
            max_entities: numarul de entitati

        Returns:
            clip_features: (max_entities, 512)
        """
        from PIL import Image

        H, W = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        crops = []
        for i in range(max_entities):
            box = boxes[i]
            x1, y1, x2, y2 = box.tolist()

            # Verifica daca box-ul e valid (non-zero)
            if x2 - x1 < 1 or y2 - y1 < 1:
                # Box invalid - foloseste frame intreg
                crop = frame_rgb
            else:
                x1, y1 = max(0, int(x1)), max(0, int(y1))
                x2, y2 = min(W, int(x2)), min(H, int(y2))
                crop = frame_rgb[y1:y2, x1:x2]
                if crop.size == 0:
                    crop = frame_rgb

            pil_crop = Image.fromarray(crop)
            crops.append(self.preprocess(pil_crop))

        crops_tensor = torch.stack(crops).to(self.device)   # (M, 3, 224, 224)
        features = self.model.encode_image(crops_tensor)    # (M, 512)
        features = torch.nn.functional.normalize(features.float(), dim=-1)
        return features.cpu()


# ---------------------------------------------------------------------------
# Video Reader
# ---------------------------------------------------------------------------

class VideoReader:
    """Citeste frame-urile dintr-un fisier video sau director de imagini."""

    def __init__(self, video_path: str, max_frames: Optional[int] = None):
        self.video_path = video_path
        self.max_frames = max_frames

    def read_frames(self) -> List[np.ndarray]:
        """
        Returneaza lista de frame-uri BGR numpy arrays.
        Suporta:
          - fisiere video (.mp4, .avi, .mov)
          - directoare cu imagini (.jpg, .png)
        """
        if os.path.isdir(self.video_path):
            return self._read_from_dir()
        else:
            return self._read_from_video()

    def _read_from_dir(self) -> List[np.ndarray]:
        exts = (".jpg", ".jpeg", ".png", ".bmp")
        files = sorted([
            f for f in os.listdir(self.video_path)
            if f.lower().endswith(exts)
        ])
        if self.max_frames:
            files = files[:self.max_frames]

        frames = []
        for f in files:
            img = cv2.imread(os.path.join(self.video_path, f))
            if img is not None:
                frames.append(img)
        return frames

    def _read_from_video(self) -> List[np.ndarray]:
        cap = cv2.VideoCapture(self.video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
            if self.max_frames and len(frames) >= self.max_frames:
                break
        cap.release()
        return frames


# ---------------------------------------------------------------------------
# Preprocessor principal
# ---------------------------------------------------------------------------

class DatasetPreprocessor:
    """
    Preprocesseaza un intreg dataset si salveaza features pe disc.

    Structura output:
        data/<dataset>/
            features/
                <video_id>_roi.npy    (S, M, 2048)
                <video_id>_clip.npy   (S, M, 512)
                <video_id>_boxes.npy  (S, M, 4)   - bounding boxes
            labels/
                <video_id>_seg.npy    (N,)  - etichete segment
                <video_id>_frame.npy  (N,)  - etichete frame
            splits/
                train_<fold>.txt
                test_<fold>.txt
    """

    def __init__(
        self,
        data_root: str,
        device: str = "cuda",
        max_entities: int = 5,
        max_frames: Optional[int] = None,
    ):
        self.data_root = data_root
        self.device = device
        self.max_entities = max_entities
        self.max_frames = max_frames

        self.imagenet_transform = get_imagenet_transform()

        # Initializeaza extractoare
        self.rcnn = FasterRCNNExtractor(device=device)
        self.clip_extractor = CLIPExtractor(device=device)

        # Creeaza directoare output
        os.makedirs(os.path.join(data_root, "features"), exist_ok=True)
        os.makedirs(os.path.join(data_root, "labels"), exist_ok=True)
        os.makedirs(os.path.join(data_root, "splits"), exist_ok=True)

    def process_video(
        self,
        video_id: str,
        video_path: str,
        labels_seg: np.ndarray,    # (N,) etichete segment
        labels_frame: np.ndarray,  # (N,) etichete frame
    ) -> bool:
        """
        Proceseaza un singur video si salveaza features.

        Returns:
            True daca s-a procesat cu succes, False altfel.
        """
        # Verifica daca exista deja (skip daca da)
        roi_path = os.path.join(self.data_root, "features", f"{video_id}_roi.npy")
        if os.path.exists(roi_path):
            return True

        # Citeste frame-uri
        reader = VideoReader(video_path, self.max_frames)
        frames = reader.read_frames()

        if not frames:
            print(f"  WARN: niciun frame gasit pentru {video_id}")
            return False

        S = len(frames)
        M = self.max_entities

        roi_all   = np.zeros((S, M, 2048), dtype=np.float32)
        clip_all  = np.zeros((S, M, 512),  dtype=np.float32)
        boxes_all = np.zeros((S, M, 4),    dtype=np.float32)

        for s, frame_bgr in enumerate(frames):
            # Conversie BGR -> RGB -> tensor ImageNet
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_tensor = self.imagenet_transform(frame_rgb)  # (3, H, W)

            # Faster R-CNN -> ROI features + boxes
            roi_feats, boxes = self.rcnn.detect_and_extract(frame_tensor, M)
            roi_all[s]   = roi_feats.numpy()
            boxes_all[s] = boxes.numpy()

            # CLIP -> features per crop
            clip_feats = self.clip_extractor.extract_from_boxes(frame_bgr, boxes, M)
            clip_all[s] = clip_feats.numpy()

        # Salveaza features
        np.save(roi_path, roi_all)
        np.save(os.path.join(self.data_root, "features", f"{video_id}_clip.npy"), clip_all)
        np.save(os.path.join(self.data_root, "features", f"{video_id}_boxes.npy"), boxes_all)

        # Salveaza etichete
        np.save(os.path.join(self.data_root, "labels", f"{video_id}_seg.npy"), labels_seg)
        np.save(os.path.join(self.data_root, "labels", f"{video_id}_frame.npy"), labels_frame)

        return True

    def process_dataset(self, video_list: List[Dict]) -> None:
        """
        Proceseaza toate video-urile din dataset.

        Args:
            video_list: lista de dicts cu cheile:
                - video_id: str
                - video_path: str
                - labels_seg: np.ndarray (N,)
                - labels_frame: np.ndarray (N,)
        """
        print(f"\nPreprocessing {len(video_list)} video-uri...")
        success, fail = 0, 0

        for item in tqdm(video_list, desc="Procesare video-uri"):
            ok = self.process_video(
                item["video_id"],
                item["video_path"],
                item["labels_seg"],
                item["labels_frame"],
            )
            if ok:
                success += 1
            else:
                fail += 1

        print(f"Gata: {success} OK, {fail} erori.")

    def generate_splits(
        self,
        video_ids: List[str],
        subject_ids: List[int],
        strategy: str = "leave_one_subject_out",
        num_folds: Optional[int] = None,
    ) -> None:
        """
        Genereaza fisierele de split pentru cross-validare.

        Args:
            video_ids:   lista de video IDs
            subject_ids: subject ID corespunzator fiecarui video
            strategy:    'leave_one_subject_out' sau 'two_subject_out'
            num_folds:   numarul de fold-uri (None = dedus din subject_ids)
        """
        unique_subjects = sorted(set(subject_ids))
        splits_dir = os.path.join(self.data_root, "splits")

        if strategy == "leave_one_subject_out":
            for fold, test_subject in enumerate(unique_subjects):
                train_ids = [v for v, s in zip(video_ids, subject_ids) if s != test_subject]
                test_ids  = [v for v, s in zip(video_ids, subject_ids) if s == test_subject]

                with open(os.path.join(splits_dir, f"train_{fold}.txt"), "w") as f:
                    f.write("\n".join(train_ids))
                with open(os.path.join(splits_dir, f"test_{fold}.txt"), "w") as f:
                    f.write("\n".join(test_ids))

                print(f"  Fold {fold}: train={len(train_ids)}, test={len(test_ids)}")

        elif strategy == "two_subject_out":
            # MPHOI-72: 2 subiecti in test la fiecare fold
            from itertools import combinations
            pairs = list(combinations(unique_subjects, 2))
            for fold, (s1, s2) in enumerate(pairs):
                train_ids = [v for v, s in zip(video_ids, subject_ids) if s not in (s1, s2)]
                test_ids  = [v for v, s in zip(video_ids, subject_ids) if s in (s1, s2)]

                with open(os.path.join(splits_dir, f"train_{fold}.txt"), "w") as f:
                    f.write("\n".join(train_ids))
                with open(os.path.join(splits_dir, f"test_{fold}.txt"), "w") as f:
                    f.write("\n".join(test_ids))

        print(f"Splits salvate in {splits_dir}/")


# ---------------------------------------------------------------------------
# Script-uri specifice per dataset
# ---------------------------------------------------------------------------

def preprocess_cad120(data_root: str, device: str) -> None:
    """
    Preprocesseaza CAD-120.

    Structura asteptata in data_root:
        raw/
            Subject1/
                <activity_name>/
                    0510175426/        (video ID)
                        rgb/           (frame-uri .png)
                    ...
            Subject2/ ...
        annotations/
            Subject1_annotations.txt
            ...
    """
    preprocessor = DatasetPreprocessor(data_root, device=device, max_entities=5)

    video_list = []
    subject_ids = []
    video_ids_all = []

    raw_dir = os.path.join(data_root, "raw")
    ann_dir = os.path.join(data_root, "annotations")

    # Mapeaza label-uri la indici
    from data.dataset import CAD120Dataset
    label_to_idx = {l: i for i, l in enumerate(CAD120Dataset.ACTIVITY_LABELS)}

    if not os.path.exists(raw_dir):
        raise FileNotFoundError(
            f"{raw_dir} nu exista. Creeaza structura de date mai intai."
        )

    for subject_name in sorted(os.listdir(raw_dir)):
        subject_path = os.path.join(raw_dir, subject_name)
        if not os.path.isdir(subject_path):
            continue

        subject_id = int(subject_name.replace("Subject", ""))

        for activity in sorted(os.listdir(subject_path)):
            activity_path = os.path.join(subject_path, activity)
            label_idx = label_to_idx.get(activity.lower(), 0)

            for video_id in sorted(os.listdir(activity_path)):
                video_path = os.path.join(activity_path, video_id, "rgb")
                if not os.path.isdir(video_path):
                    continue

                num_frames = len([f for f in os.listdir(video_path) if f.endswith(".png")])
                M = 5   # max entities

                # Etichete simple: toata secventa = aceeasi activitate
                labels_seg   = np.full(num_frames * M, label_idx, dtype=np.int64)
                labels_frame = np.full(num_frames * M, label_idx, dtype=np.int64)

                video_list.append({
                    "video_id": f"{subject_name}_{activity}_{video_id}",
                    "video_path": video_path,
                    "labels_seg": labels_seg,
                    "labels_frame": labels_frame,
                })
                subject_ids.append(subject_id)
                video_ids_all.append(f"{subject_name}_{activity}_{video_id}")

    preprocessor.process_dataset(video_list)
    preprocessor.generate_splits(video_ids_all, subject_ids, "leave_one_subject_out")




# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Preproceseaza dataset pentru VHOIP")
    parser.add_argument(
        "--dataset", type=str, required=True,
        choices=["cad120", "mphoi72", "bimanual"],
    )
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Genereaza date sintetice (pentru testare fara date reale)",
    )
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--num_videos", type=int, default=40)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.synthetic:
        print(f"Generare date sintetice pentru {args.dataset}...")
        _generate_synthetic_data(
            args.data_root,
            num_videos=args.num_videos,
            num_classes=args.num_classes,
        )
    elif args.dataset == "cad120":
        preprocess_cad120(args.data_root, args.device)
    else:
        raise NotImplementedError(
            f"Preprocessare pentru {args.dataset} nu este implementata aici. "
            f"Foloseste setup_mphoi72.py pentru MPHOI-72."
        )