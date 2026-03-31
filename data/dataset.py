"""
dataset.py
Clase PyTorch Dataset pentru cele trei seturi de date VHOIP.

Fiecare sample contine:
- features ROI (2048-dim) extrase cu Faster R-CNN
- features CLIP vizuale (512-dim) extrase offline
- etichete segment si frame
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional


class HOIVideoDataset(Dataset):
    """
    Dataset de baza pentru video HOI recognition.
    Subclasele (CAD120Dataset, MPHOI72Dataset, etc.) extind aceasta clasa.

    Structura asteptata in data/<dataset_name>/:
        features/
            <video_id>_roi.npy      # (S, M, 2048) - S frames, M entities
            <video_id>_clip.npy     # (S, M, 512)  - features CLIP vizuale
        labels/
            <video_id>_seg.npy      # (N,) - etichete segment per entitate
            <video_id>_frame.npy    # (N,) - etichete frame per entitate
        splits/
            train_<fold>.txt
            test_<fold>.txt
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        fold: int = 0,
        roi_dim: int = 2048,
        clip_dim: int = 512,
    ):
        self.root = root
        self.split = split
        self.fold = fold
        self.roi_dim = roi_dim
        self.clip_dim = clip_dim

        self.video_ids = self._load_split()

    def _load_split(self) -> List[str]:
        split_file = os.path.join(
            self.root, "splits", f"{self.split}_{self.fold}.txt"
        )
        if not os.path.exists(split_file):
            raise FileNotFoundError(
                f"Nu gasesc split file: {split_file}\n"
                f"Ruleaza mai intai data/preprocess.py pentru a genera splits."
            )
        with open(split_file, "r") as f:
            return [line.strip() for line in f if line.strip()]

    def __len__(self) -> int:
        return len(self.video_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        video_id = self.video_ids[idx]

        # Incarcare features ROI (extrase cu Faster R-CNN)
        roi_path = os.path.join(self.root, "features", f"{video_id}_roi.npy")
        roi_features = np.load(roi_path)  # (S, M, 2048)

        # Incarcare features CLIP vizuale (extrase offline)
        clip_path = os.path.join(self.root, "features", f"{video_id}_clip.npy")
        clip_features = np.load(clip_path)  # (S, M, 512)

        # Incarcare etichete
        seg_labels = np.load(
            os.path.join(self.root, "labels", f"{video_id}_seg.npy")
        )  # (N,)
        frame_labels = np.load(
            os.path.join(self.root, "labels", f"{video_id}_frame.npy")
        )  # (N,)

        return {
            "video_id": video_id,
            "roi_features": torch.FloatTensor(roi_features),
            "clip_features": torch.FloatTensor(clip_features),
            "seg_labels": torch.LongTensor(seg_labels),
            "frame_labels": torch.LongTensor(frame_labels),
        }


class CAD120Dataset(HOIVideoDataset):
    """Dataset CAD-120 - single person, 10 sub-activitati."""

    NUM_CLASSES_ACTIVITY = 10
    NUM_CLASSES_AFFORDANCE = 12
    NUM_SUBJECTS = 4

    # Vocabular HOI pentru template-uri CLIP
    ACTIVITY_LABELS = [
        "eating", "drinking", "pouring", "cooking",
        "microwaving", "sweeping", "reading", "arranging",
        "taking medicine", "relaxing",
    ]

    def __init__(self, root: str, split: str = "train", fold: int = 0, **kwargs):
        super().__init__(root, split, fold, **kwargs)
        self.num_classes = self.NUM_CLASSES_ACTIVITY


class MPHOI72Dataset(HOIVideoDataset):
    """Dataset MPHOI-72 - multi-person, 13 sub-activitati."""

    NUM_CLASSES = 13
    NUM_SUBJECTS = 12

    ACTIVITY_LABELS = [
        "approaching", "lifting", "pouring", "placing", "drinking",
        "cheering", "retreating", "working", "asking", "solving",
        "sitting", "cutting", "drying",
    ]

    def __init__(self, root: str, split: str = "train", fold: int = 0, **kwargs):
        super().__init__(root, split, fold, **kwargs)
        self.num_classes = self.NUM_CLASSES


class BimanualDataset(HOIVideoDataset):
    """Dataset Bimanual Actions - two-hand HOI, 14 action labels."""

    NUM_CLASSES = 14
    NUM_SUBJECTS = 6

    ACTIVITY_LABELS = [
        "idle", "approach", "lift", "place", "retreat",
        "hold", "saw", "screw", "stir", "pour",
        "open", "close", "cut", "mix",
    ]

    def __init__(self, root: str, split: str = "train", fold: int = 0, **kwargs):
        super().__init__(root, split, fold, **kwargs)
        self.num_classes = self.NUM_CLASSES


def get_dataset(name: str, **kwargs) -> HOIVideoDataset:
    """Factory function - returneaza dataset-ul corect dupa nume."""
    datasets = {
        "cad120": CAD120Dataset,
        "mphoi72": MPHOI72Dataset,
        "bimanual": BimanualDataset,
    }
    if name not in datasets:
        raise ValueError(f"Dataset necunoscut: {name}. Alege din {list(datasets.keys())}")
    return datasets[name](**kwargs)