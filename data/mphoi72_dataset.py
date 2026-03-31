"""
data/mphoi72_dataset.py
Dataset specific pentru MPHOI-72 care citeste direct din fisierele .zarr
furnizate de autorii 2G-GCN (https://github.com/tanqiu98/2G-GCN).

Structura asteptata in data_root/:
    mphoi_derived_features/
        faster_rcnn.zarr/           <- ROI features (2048-dim)
        human_bounding_boxes.zarr/  <- bounding boxes umane
        object_bounding_boxes.zarr/ <- bounding boxes obiecte
        human_pose.zarr/            <- keypoints skeleton (optional)
    mphoi_action_id_to_action_name.json
    mphoi_ground_truth_labels.json

Instalare dependenta zarr:
    pip install zarr

Utilizare:
    from data.mphoi72_dataset import MPHOI72ZarrDataset, prepare_mphoi72_splits
    prepare_mphoi72_splits('data/mphoi72/')
    ds = MPHOI72ZarrDataset('data/mphoi72/', split='train', fold=0)
"""

import os
import json
import re
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple
import zarr


# ---------------------------------------------------------------------------
# Utilitare pentru citirea structurii MPHOI-72
# ---------------------------------------------------------------------------

def load_action_mapping(data_root: str) -> Dict[int, str]:
    """
    Incarca maparea action_id -> action_name.
    Ex: {0: 'approach', 1: 'lift', ...}
    """
    path = os.path.join(data_root, "mphoi_action_id_to_action_name.json")
    with open(path, "r") as f:
        raw = json.load(f)
    # cheile pot fi string-uri in JSON
    return {int(k): v for k, v in raw.items()}


def load_ground_truth(data_root: str) -> dict:
    """
    Incarca etichetele ground truth.

    Structura JSON asteptata (tipica pentru 2G-GCN):
    {
      "video_id_1": {
        "subject_id": 1,
        "entities": [
          {"entity_id": 0, "type": "human", "labels": [0, 0, 1, 1, 2, ...]},
          {"entity_id": 1, "type": "object", "labels": [3, 3, 3, 4, 4, ...]},
          ...
        ]
      },
      ...
    }

    Nota: structura exacta depinde de versiunea dataset-ului.
    Inspectam JSON-ul si adaptam daca e diferita.
    """
    path = os.path.join(data_root, "mphoi_ground_truth_labels.json")
    with open(path, "r") as f:
        return json.load(f)


def inspect_zarr(data_root: str) -> None:
    """
    Afiseaza structura zarr-urilor pentru a intelege formatul exact.
    Ruleaza o data inainte de a folosi dataset-ul.
    """
    features_dir = os.path.join(data_root, "mphoi_derived_features")

    for zarr_name in ["faster_rcnn.zarr", "human_bounding_boxes.zarr",
                      "object_bounding_boxes.zarr", "human_pose.zarr"]:
        zarr_path = os.path.join(features_dir, zarr_name)
        if not os.path.exists(zarr_path):
            print(f"  {zarr_name}: nu exista")
            continue

        try:
            z = zarr.open(zarr_path, mode="r")
            print(f"\n{zarr_name}:")
            print(f"  Tip: {type(z)}")

            if hasattr(z, "keys"):
                print(f"  Chei: {list(z.keys())[:10]}")
                # Arata primul element
                first_key = list(z.keys())[0]
                first_val = z[first_key]
                if hasattr(first_val, "shape"):
                    print(f"  Primul element '{first_key}': shape={first_val.shape}, dtype={first_val.dtype}")
                elif hasattr(first_val, "keys"):
                    child_keys = list(first_val.keys())
                    print(f"  Primul element '{first_key}' este group cu chei: {child_keys}")
                    if child_keys:
                        child = first_val[child_keys[0]]
                        print(
                            f"    Child '{child_keys[0]}': "
                            f"shape={getattr(child, 'shape', None)}, "
                            f"dtype={getattr(child, 'dtype', None)}"
                        )
            elif hasattr(z, "shape"):
                print(f"  Shape: {z.shape}, dtype: {z.dtype}")
        except Exception as e:
            print(f"  {zarr_name}: eroare - {e}")


# ---------------------------------------------------------------------------
# Convertor zarr -> numpy (pas intermediar)
# ---------------------------------------------------------------------------

def convert_zarr_to_npy(data_root: str, output_dir: Optional[str] = None) -> None:
    """
    Converteste features zarr in fisiere .npy compatibile cu pipeline-ul VHOIP.
    Se ruleaza O SINGURA DATA.

    Citeste:
        faster_rcnn.zarr      -> (S, M, 2048) ROI features per video
        ground_truth_labels   -> etichete per entitate

    Salveaza in output_dir (default: data_root/features/ si data_root/labels/):
        <video_id>_roi.npy    (S, M, 2048)
        <video_id>_clip.npy   (S, M, 512)  <- placeholder zeros (CLIP se extrage separat)
        <video_id>_seg.npy    (N,)
        <video_id>_frame.npy  (N,)
    """
    if output_dir is None:
        output_dir = data_root

    features_out = os.path.join(output_dir, "features")
    labels_out = os.path.join(output_dir, "labels")
    os.makedirs(features_out, exist_ok=True)
    os.makedirs(labels_out, exist_ok=True)

    features_dir = os.path.join(data_root, "mphoi_derived_features")
    rcnn_zarr_path = os.path.join(features_dir, "faster_rcnn.zarr")

    print("Deschid faster_rcnn.zarr...")
    rcnn_store = zarr.open(rcnn_zarr_path, mode="r")

    print("Incarc ground truth labels...")
    gt_data = load_ground_truth(data_root)
    action_map = load_action_mapping(data_root)

    print(f"  {len(gt_data)} video-uri gasite in ground truth")
    print(f"  {len(action_map)} clase: {list(action_map.values())}")

    converted = 0
    skipped = 0

    for video_id, video_info in gt_data.items():
        roi_out_path = os.path.join(features_out, f"{video_id}_roi.npy")

        # Skip daca exista deja
        if os.path.exists(roi_out_path):
            skipped += 1
            continue

        # --- ROI features din zarr ---
        try:
            roi_features = _extract_roi_features(rcnn_store, video_id, video_info)
        except KeyError as e:
            print(f"  WARN: video {video_id} nu are features ROI: {e}")
            continue

        S, M, D = roi_features.shape

        # --- Etichete ---
        seg_labels, frame_labels = _extract_labels(video_info, S, M)

        # --- Salveaza ---
        np.save(roi_out_path, roi_features.astype(np.float32))
        np.save(
            os.path.join(features_out, f"{video_id}_clip.npy"),
            np.zeros((S, M, 512), dtype=np.float32),
        )  # placeholder CLIP
        np.save(
            os.path.join(labels_out, f"{video_id}_seg.npy"),
            seg_labels.astype(np.int64),
        )
        np.save(
            os.path.join(labels_out, f"{video_id}_frame.npy"),
            frame_labels.astype(np.int64),
        )

        converted += 1

    print(f"\nConversie completa: {converted} convertite, {skipped} deja existente.")
    return list(gt_data.keys())


def extract_clip_features(
    data_root: str,
    model_name: str = "ViT-B/16",
    device: str = "cuda",
    batch_size: int = 64,
    output_dir: Optional[str] = None,
) -> None:
    """
    Extrage features CLIP vizuale pentru fiecare ROI si le salveaza ca _clip.npy,
    inlocuind placeholder-urile de zerouri.

    Trebuie apelata DUPA convert_zarr_to_npy() si INAINTE de antrenare.
    Furnizeaza prior-ul CLIP real (G_init) modelului VHOIP.

    Deoarece nu avem imagini brute (doar ROI pooling 2048-dim), proiectam
    features-urile ROI in spatiul intern al ViT-B/16 (768-dim) si rulam
    transformer-ul CLIP pentru a obtine reprezentari in spatiul CLIP (512-dim).

    IMPORTANT pentru dtype: CLIP se incarca in float16 pe CUDA si float32 pe CPU.
    Toti tensorii nostri trebuie castati la dtype-ul CLIP inainte de forward pass.
    """
    try:
        import clip as clip_lib
    except ImportError:
        raise ImportError("Instaleaza CLIP: pip install git+https://github.com/openai/CLIP.git")

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    if output_dir is None:
        output_dir = os.path.join(data_root, "features")

    print(f"\nIncarc CLIP {model_name} pe {device}...")
    clip_model, _ = clip_lib.load(model_name, device=device)
    clip_model.eval()
    clip_dim = 512  # ViT-B/16 output dim

    # Detectam dtype-ul real al modelului: float16 pe CUDA, float32 pe CPU.
    # Toti tensorii nostri trebuie sa fie in acelasi dtype — altfel PyTorch
    # arunca "mat1 and mat2 must have same dtype" in multi_head_attention_forward.
    clip_dtype = clip_model.visual.transformer.resblocks[0].attn.in_proj_weight.dtype
    print(f"  CLIP dtype detectat: {clip_dtype}")

    # Proiectie ROI (2048) -> spatiu intern ViT-B/16 (768-dim patch embedding).
    # Initializare ortogonala in float32, apoi recastata la clip_dtype.
    roi_dim = 2048
    vit_patch_dim = 768
    proj = nn.Linear(roi_dim, vit_patch_dim, bias=False)
    nn.init.orthogonal_(proj.weight)                        # orthogonal init cere float32
    proj = proj.to(device=device, dtype=clip_dtype)         # cast la dtype-ul CLIP

    features_dir = os.path.join(data_root, "features")
    roi_files = sorted(f for f in os.listdir(features_dir) if f.endswith("_roi.npy"))
    print(f"Procesez {len(roi_files)} video-uri...")

    for roi_file in roi_files:
        video_id = roi_file.replace("_roi.npy", "")
        clip_out_path = os.path.join(output_dir, f"{video_id}_clip.npy")

        roi = np.load(os.path.join(features_dir, roi_file))   # (S, M, 2048) float32
        S, M, D = roi.shape

        # Castam la clip_dtype (float16 pe GPU) — obligatoriu pentru compatibilitate
        roi_tensor = torch.tensor(roi, device=device, dtype=clip_dtype)
        roi_flat = roi_tensor.reshape(S * M, D)                # (S*M, 2048)

        clip_feats_list = []
        visual = clip_model.visual

        with torch.no_grad():
            for i in range(0, S * M, batch_size):
                batch_roi = roi_flat[i: i + batch_size]        # (b, 2048) in clip_dtype

                # Proiectare ROI -> patch embedding (768-dim), tot in clip_dtype
                patch_tokens = proj(batch_roi)                  # (b, 768)
                # Normalizam in float32 pentru precizie, recastam inapoi
                patch_tokens = F.normalize(patch_tokens.float(), dim=-1).to(clip_dtype)

                b = patch_tokens.shape[0]
                num_patches = 196   # 224/16 * 224/16 = 196 patch-uri pentru ViT-B/16

                # Replicam token-ul proiectat pe toate cele 196 pozitii de patch
                patch_seq = patch_tokens.unsqueeze(1).expand(b, num_patches, -1)   # (b, 196, 768)

                # cls token + positional embedding — deja in clip_dtype
                cls_tokens = visual.class_embedding.unsqueeze(0).expand(b, -1, -1) # (b, 1, 768)
                x = torch.cat([cls_tokens, patch_seq], dim=1)                       # (b, 197, 768)
                x = x + visual.positional_embedding                                 # (b, 197, 768)
                x = visual.ln_pre(x)
                x = x.permute(1, 0, 2).to(clip_dtype)                              # (197, b, 768)
                x = visual.transformer(x)                                           # (197, b, 768)
                x = x.permute(1, 0, 2)                                              # (b, 197, 768)
                x = visual.ln_post(x[:, 0, :])                                      # (b, 768) cls token
                if visual.proj is not None:
                    x = x @ visual.proj                                              # (b, 512)

                # Convertim la float32 pentru salvare — numpy nu suporta float16 bine
                feats = F.normalize(x.float(), dim=-1)          # (b, 512) float32
                clip_feats_list.append(feats.cpu().numpy())

        clip_feats = np.concatenate(clip_feats_list, axis=0)   # (S*M, 512)
        clip_feats = clip_feats.reshape(S, M, clip_dim)         # (S, M, 512)
        np.save(clip_out_path, clip_feats.astype(np.float32))
        print(f"  {video_id}: shape={clip_feats.shape}, norm_mean={np.linalg.norm(clip_feats, axis=-1).mean():.4f}")

    print(f"\nCLIP features salvate in {output_dir}/ ({len(roi_files)} video-uri).")


def _extract_roi_features(
    rcnn_store: zarr.Group,
    video_id: str,
    video_info: dict,
) -> np.ndarray:
    """
    Extrage ROI features pentru un video din zarr.

    Zarr-ul 2G-GCN e organizat de obicei ca:
        rcnn_store[video_id]       -> array (S, M, 2048)
    sau
        rcnn_store[video_id]["features"] -> array (S, M, 2048)

    Adaptam dupa structura reala inspectata cu inspect_zarr().
    """
    # Incearca accesul direct
    if video_id in rcnn_store:
        data = rcnn_store[video_id]
        if hasattr(data, "shape"):
            arr = np.array(data)
        elif hasattr(data, "keys") and "Human1" in data and "Human2" in data:
            # Format MPHOI-72 observat: group cu Human1, Human2, objects
            h1 = np.array(data["Human1"])  # (S, 2048)
            h2 = np.array(data["Human2"])  # (S, 2048)

            if h1.ndim != 2 or h2.ndim != 2:
                raise ValueError(f"Human features au shape neasteptat pentru {video_id}: {h1.shape}, {h2.shape}")

            S = min(h1.shape[0], h2.shape[0])
            human_feats = [h1[:S], h2[:S]]

            object_feats = None
            if "objects" in data:
                obj = np.array(data["objects"])  # (S, O, 2048) sau (S, 2048)
                if obj.ndim == 3:
                    S = min(S, obj.shape[0])
                    human_feats = [h[:S] for h in human_feats]
                    object_feats = obj[:S]
                elif obj.ndim == 2:
                    S = min(S, obj.shape[0])
                    human_feats = [h[:S] for h in human_feats]
                    object_feats = obj[:S, None, :]

            # Construieste (S, M, 2048)
            humans = np.stack(human_feats, axis=1)  # (S, 2, 2048)
            if object_feats is not None:
                arr = np.concatenate([humans, object_feats], axis=1)
            else:
                arr = humans
        elif "features" in data:
            arr = np.array(data["features"])
        else:
            # Ia primul array disponibil
            first_key = list(data.keys())[0]
            arr = np.array(data[first_key])
    else:
        raise KeyError(f"video_id '{video_id}' nu e in zarr")

    # Asigura forma (S, M, 2048)
    if arr.ndim == 2:
        # (S*M, 2048) -> trebuie sa stim M
        M = _infer_num_entities(video_info)
        S = arr.shape[0] // M
        arr = arr.reshape(S, M, -1)
    elif arr.ndim == 3:
        pass   # deja (S, M, D)
    else:
        raise ValueError(f"Shape neasteptat pentru {video_id}: {arr.shape}")

    return arr


def _infer_num_entities(video_info: dict) -> int:
    """Deduce numarul de entitati din informatiile video."""
    if "entities" in video_info:
        return len(video_info["entities"])
    # MPHOI-72 are 2 persoane + 2-4 obiecte = 4-6 entitati
    # Default conservativ:
    return 5


def _extract_labels(
    video_info: dict,
    S: int,
    M: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extrage etichetele segment si frame din informatiile video.

    Returneaza:
        seg_labels:   (N,) = (S*M,) - eticheta dominanta per entitate per frame
        frame_labels: (N,) = (S*M,) - identic cu seg_labels pentru simplitate
    """
    N = S * M
    seg_labels = np.zeros(N, dtype=np.int64)
    frame_labels = np.zeros(N, dtype=np.int64)

    if "entities" in video_info:
        entities = video_info["entities"]
        for entity_idx, entity in enumerate(entities[:M]):
            raw_labels = entity.get("labels", [0] * S)
            # Asigura lungimea S
            labels_arr = np.array(raw_labels[:S], dtype=np.int64)
            if len(labels_arr) < S:
                labels_arr = np.pad(
                    labels_arr,
                    (0, S - len(labels_arr)),
                    constant_values=labels_arr[-1] if len(labels_arr) > 0 else 0,
                )
            # Scrie in N = S*M flatten (ordonare: frame-major)
            for s in range(S):
                flat_idx = s * M + entity_idx
                seg_labels[flat_idx] = labels_arr[s]
                frame_labels[flat_idx] = labels_arr[s]

    elif "Human1" in video_info or "Human2" in video_info:
        # Format MPHOI-72 observat: chei Human1/Human2 cu etichete per frame.
        human1 = np.array(video_info.get("Human1", []), dtype=np.int64)
        human2 = np.array(video_info.get("Human2", []), dtype=np.int64)

        if human1.size == 0 and human2.size == 0:
            return seg_labels, frame_labels

        if human1.size == 0:
            human1 = human2
        if human2.size == 0:
            human2 = human1

        l1 = np.pad(human1[:S], (0, max(0, S - len(human1))), mode="edge") if len(human1) > 0 else np.zeros(S, dtype=np.int64)
        l2 = np.pad(human2[:S], (0, max(0, S - len(human2))), mode="edge") if len(human2) > 0 else l1

        # Entitati 0/1 = Human1/Human2, restul obiecte (folosim eticheta Human1 ca proxy video-level)
        for s in range(S):
            if M >= 1:
                seg_labels[s * M + 0] = l1[s]
                frame_labels[s * M + 0] = l1[s]
            if M >= 2:
                seg_labels[s * M + 1] = l2[s]
                frame_labels[s * M + 1] = l2[s]
            for m in range(2, M):
                seg_labels[s * M + m] = l1[s]
                frame_labels[s * M + m] = l1[s]

    elif "labels" in video_info:
        # Format alternativ: etichete directe per frame
        raw = np.array(video_info["labels"][:S], dtype=np.int64)
        for m in range(M):
            for s in range(S):
                seg_labels[s * M + m] = raw[s] if s < len(raw) else 0
                frame_labels[s * M + m] = raw[s] if s < len(raw) else 0

    return seg_labels, frame_labels


# ---------------------------------------------------------------------------
# Generare splits cross-validare
# ---------------------------------------------------------------------------

def prepare_mphoi72_splits(data_root: str) -> None:
    """
    Genereaza splits de cross-validare pentru MPHOI-72.
    Strategia din paper: 'two_subject_out' (2 subiecti in test).

    Se apeleaza dupa convert_zarr_to_npy().
    """
    splits_dir = os.path.join(data_root, "splits")
    os.makedirs(splits_dir, exist_ok=True)

    gt_data = load_ground_truth(data_root)

    # Extrage subject_id per video
    video_ids = []
    subject_ids = []

    for video_id, info in gt_data.items():
        video_ids.append(video_id)
        # subject_id poate fi in info direct sau dedus din video_id
        subj = info.get("subject_id", None)
        if subj is None:
            # Incearca sa-l extraga din numele video-ului
            # Ex: "S1_activity_001" -> subject 1
            # Ex: "Subject12-task_..." -> subject 12
            m = re.search(r"Subject(\d+)", str(video_id), flags=re.IGNORECASE)
            if m:
                subj = int(m.group(1))
            else:
                parts = str(video_id).split("_")
                for p in parts:
                    if p.startswith("S") and p[1:].isdigit():
                        subj = int(p[1:])
                        break
            if subj is None:
                subj = 1  # fallback
        subject_ids.append(int(subj))

    unique_subjects = sorted(set(subject_ids))
    print(f"Subiecti gasiti: {unique_subjects}")

    from itertools import combinations
    pairs = list(combinations(unique_subjects, 2))

    for fold, (s1, s2) in enumerate(pairs):
        train_ids = [v for v, s in zip(video_ids, subject_ids) if s not in (s1, s2)]
        test_ids = [v for v, s in zip(video_ids, subject_ids) if s in (s1, s2)]

        with open(os.path.join(splits_dir, f"train_{fold}.txt"), "w") as f:
            f.write("\n".join(str(v) for v in train_ids))
        with open(os.path.join(splits_dir, f"test_{fold}.txt"), "w") as f:
            f.write("\n".join(str(v) for v in test_ids))

        print(f"  Fold {fold} (S{s1}+S{s2} in test): train={len(train_ids)}, test={len(test_ids)}")

    print(f"\n{len(pairs)} fold-uri salvate in {splits_dir}/")


# ---------------------------------------------------------------------------
# Dataset PyTorch pentru MPHOI-72
# ---------------------------------------------------------------------------

class MPHOI72ZarrDataset(Dataset):
    """
    Dataset PyTorch pentru MPHOI-72 care citeste fisierele .npy
    generate de convert_zarr_to_npy().

    Compatibil cu pipeline-ul de antrenare VHOIP existent.
    """

    NUM_CLASSES = 13
    NUM_SUBJECTS = 12

    ACTIVITY_LABELS = [
        "approaching", "lifting", "pouring", "placing", "drinking",
        "cheering", "retreating", "working", "asking", "solving",
        "sitting", "cutting", "drying",
    ]

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        fold: int = 0,
        roi_dim: int = 2048,
        clip_dim: int = 512,
    ):
        self.data_root = data_root
        self.split = split
        self.fold = fold
        self.roi_dim = roi_dim
        self.clip_dim = clip_dim
        self.num_classes = self.NUM_CLASSES

        split_file = os.path.join(data_root, "splits", f"{split}_{fold}.txt")
        if not os.path.exists(split_file):
            raise FileNotFoundError(
                f"Nu gasesc {split_file}.\n"
                f"Ruleaza mai intai:\n"
                f"  from data.mphoi72_dataset import convert_zarr_to_npy, prepare_mphoi72_splits\n"
                f"  convert_zarr_to_npy('data/mphoi72/')\n"
                f"  prepare_mphoi72_splits('data/mphoi72/')"
            )

        with open(split_file, "r") as f:
            self.video_ids = [l.strip() for l in f if l.strip()]

        print(f"MPHOI72 [{split}/fold{fold}]: {len(self.video_ids)} video-uri")

    def __len__(self) -> int:
        return len(self.video_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        video_id = self.video_ids[idx]
        feat_dir = os.path.join(self.data_root, "features")
        label_dir = os.path.join(self.data_root, "labels")

        roi = np.load(os.path.join(feat_dir, f"{video_id}_roi.npy"))
        clip = np.load(os.path.join(feat_dir, f"{video_id}_clip.npy"))
        seg = np.load(os.path.join(label_dir, f"{video_id}_seg.npy"))
        frame = np.load(os.path.join(label_dir, f"{video_id}_frame.npy"))

        return {
            "video_id": video_id,
            "roi_features": torch.FloatTensor(roi),
            "clip_features": torch.FloatTensor(clip),
            "seg_labels": torch.LongTensor(seg),
            "frame_labels": torch.LongTensor(frame),
        }