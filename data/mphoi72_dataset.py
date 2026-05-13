"""
data/mphoi72_dataset.py
Dataset specific pentru MPHOI-72 care citeste direct din fisierele .zarr
furnizate de autorii 2G-GCN (https://github.com/tanqiu98/2G-GCN).

Structura asteptata in data_root/:
    mphoi_derived_features/
        faster_rcnn.zarr/           <- ROI features (2048-dim)
        human_bounding_boxes.zarr/  <- bounding boxes umane  [x1,y1,x2,y2]
        object_bounding_boxes.zarr/ <- bounding boxes obiecte [x1,y1,x2,y2]
        human_pose.zarr/            <- keypoints skeleton uman (J_h joints)
    mphoi_action_id_to_action_name.json
    mphoi_ground_truth_labels.json

Fisiere generate de convert_zarr_to_npy() in data_root/features/:
    <video_id>_roi.npy          (S, M, 2048)  - ROI pooling features
    <video_id>_geo.npy          (S, J, 4)     - geometric keypoints (pozitie+viteza)
    <video_id>_entity_types.npy (M,)          - 0=human, 1=object per entitate
    <video_id>_bbox.npy         (S, M, 4)     - bounding boxes [x1,y1,x2,y2]

CLIP visual features (_clip.npy) NU sunt generate de convert_zarr_to_npy().
Se extrag separat din video-uri brute cu extract_clip_features_from_videos().
Daca nu exista, modelul foloseste fallback text-based pentru G_init.

Geometric features (geo) — format conform 2G-GCN §4.1:
    J = J_h * N_humans + 4 * N_objects
        unde J_h = nr joints per human (ex. 32 pentru Azure Kinect)
              4 = cele 4 colturi ale bounding box-ului obiectului
    Fiecare keypoint: [x, y, vx, vy]  (pozitie + viteza pe frame)
    Viteza = diferenta pozitie fata de frame-ul precedent (0 pentru primul frame)

Entity types — indexare:
    [Human1, Human2, ..., Object1, Object2, ...]
    Corespunde cu ordinea entitatilor in roi_features (axa M).
    0 = human, 1 = object

Utilizare:
    from data.mphoi72_dataset import (
        MPHOI72ZarrDataset, convert_zarr_to_npy,
        prepare_mphoi72_splits, extract_clip_features_from_videos,
    )
    convert_zarr_to_npy('data/mphoi72/')
    # Optional: extrage CLIP real din video-uri brute (necesita raw videos)
    extract_clip_features_from_videos('data/mphoi72/', 'data/mphoi72/videos/')
    prepare_mphoi72_splits('data/mphoi72/')
    ds = MPHOI72ZarrDataset('data/mphoi72/', split='train', fold=0)
    sample = ds[0]
    # sample['geo_features']:   (S, J, 4)  FloatTensor
    # sample['entity_types']:   (M,)       LongTensor  0=human, 1=object
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
# Constante dataset
# ---------------------------------------------------------------------------

MPHOI72_ACTIVITY_LABELS = [
    "approaching", "lifting", "pouring", "placing", "drinking",
    "cheering", "retreating", "working", "asking", "solving",
    "sitting", "cutting", "drying",
]

# Numarul de joints per human din Azure Kinect Body Tracking SDK
# (folosit de MPHOI-72 conform paperului)
# Kinect v2 = 25 joints, Azure Kinect = 32 joints
# Daca valoarea din zarr difera, _extract_geo_features() o detecteaza automat.
KINECT_JOINTS_DEFAULT = 32

# Dimensiunea featurei geometrice per keypoint: [x, y, vx, vy]
GEO_FEAT_DIM = 4


# ---------------------------------------------------------------------------
# Utilitare pentru citirea structurii MPHOI-72
# ---------------------------------------------------------------------------

def load_action_mapping(data_root: str) -> Dict[int, str]:
    """Incarca maparea action_id -> action_name. Ex: {0: 'approach', 1: 'lift', ...}"""
    path = os.path.join(data_root, "mphoi_action_id_to_action_name.json")
    with open(path, "r") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def load_ground_truth(data_root: str) -> dict:
    """Incarca etichetele ground truth din JSON."""
    path = os.path.join(data_root, "mphoi_ground_truth_labels.json")
    with open(path, "r") as f:
        return json.load(f)


def inspect_zarr(data_root: str) -> None:
    """
    Afiseaza structura zarr-urilor. Ruleaza o data inainte de conversie
    pentru a intelege formatul exact al datelor.
    """
    features_dir = os.path.join(data_root, "mphoi_derived_features")

    for zarr_name in [
        "faster_rcnn.zarr",
        "human_bounding_boxes.zarr",
        "object_bounding_boxes.zarr",
        "human_pose.zarr",
    ]:
        zarr_path = os.path.join(features_dir, zarr_name)
        if not os.path.exists(zarr_path):
            print(f"  {zarr_name}: nu exista")
            continue

        try:
            z = zarr.open(zarr_path, mode="r")
            print(f"\n{zarr_name}:")
            print(f"  Tip: {type(z)}")

            if hasattr(z, "keys"):
                keys = list(z.keys())
                print(f"  Chei ({len(keys)}): {keys[:5]}{'...' if len(keys) > 5 else ''}")
                first_key = keys[0]
                first_val = z[first_key]
                if hasattr(first_val, "shape"):
                    print(f"  Primul element '{first_key}': shape={first_val.shape}, dtype={first_val.dtype}")
                elif hasattr(first_val, "keys"):
                    child_keys = list(first_val.keys())
                    print(f"  Primul element '{first_key}' group cu chei: {child_keys}")
                    for ck in child_keys[:3]:
                        child = first_val[ck]
                        print(f"    '{ck}': shape={getattr(child, 'shape', None)}, dtype={getattr(child, 'dtype', None)}")
            elif hasattr(z, "shape"):
                print(f"  Shape: {z.shape}, dtype: {z.dtype}")
        except Exception as e:
            print(f"  {zarr_name}: eroare - {e}")


# ---------------------------------------------------------------------------
# Extragere features geometrice din zarr
# ---------------------------------------------------------------------------

def _extract_geo_features(
    pose_store: Optional[zarr.Group],
    hbox_store: Optional[zarr.Group],
    obox_store: Optional[zarr.Group],
    video_id: str,
    S: int,
    num_humans: int,
    num_objects: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extrage si construieste geo_features si entity_types pentru un video.

    Geometric features conform 2G-GCN §4.1:
        - Skeleton uman: pozitia (x, y) a fiecarui joint per frame
        - Bbox obiect: cele 4 colturi (x1,y1,x2,y2) -> tratate ca 4 "keypoints"
        - Viteza: diferenta pozitie fata de frame-ul anterior

    Args:
        pose_store:  zarr cu skeleton joints umani (None daca nu exista)
        hbox_store:  zarr cu bounding boxes umane  (None daca nu exista)
        obox_store:  zarr cu bounding boxes obiecte (None daca nu exista)
        video_id:    ID-ul videoclipului
        S:           numarul de frame-uri
        num_humans:  numarul de persoane (de obicei 2 pentru MPHOI-72)
        num_objects: numarul de obiecte

    Returns:
        geo:          (S, J, GEO_FEAT_DIM=4)  float32
                      J = num_humans * joints_per_human + num_objects * 4
        entity_types: (M,)  int64  (0=human, 1=object)
                      M = num_humans + num_objects, ordinea: [humans..., objects...]
    """
    keypoint_sequences = []   # fiecare element: (S, K, 2) — K keypoints, (x,y)

    # -----------------------------------------------------------------------
    # 1. Skeleton joints umani
    # -----------------------------------------------------------------------
    joints_per_human = KINECT_JOINTS_DEFAULT

    if pose_store is not None and video_id in pose_store:
        pose_data = pose_store[video_id]

        # Posibile formate in zarr:
        #   (a) group cu 'Human1', 'Human2' -> fiecare (S, J_h, 2) sau (S, J_h*2)
        #   (b) array direct (S, N_humans, J_h, 2)
        #   (c) group cu 'Human1', 'Human2' -> fiecare (S, J_h) cu (x,y) interleaved

        if hasattr(pose_data, "keys"):
            human_keys = sorted([k for k in pose_data.keys()
                                  if k.lower().startswith("human")])
            for hk in human_keys[:num_humans]:
                arr = np.array(pose_data[hk], dtype=np.float32)  # diverse forme
                arr = _normalize_keypoint_array(arr, S)           # -> (S, J_h, 2)
                joints_per_human = arr.shape[1]
                keypoint_sequences.append(arr)

        elif hasattr(pose_data, "shape"):
            arr = np.array(pose_data, dtype=np.float32)
            # Presupunem (S, N_humans, J_h, 2) sau (S, N_humans*J_h, 2)
            if arr.ndim == 4:
                # (S, N_h, J_h, 2)
                joints_per_human = arr.shape[2]
                for h in range(min(arr.shape[1], num_humans)):
                    keypoint_sequences.append(arr[:, h, :, :])  # (S, J_h, 2)
            elif arr.ndim == 3:
                # (S, N_h*J_h, 2) — trebuie impartit in N_h bucati
                total_joints = arr.shape[1]
                joints_per_human = total_joints // max(num_humans, 1)
                for h in range(num_humans):
                    s = h * joints_per_human
                    e = s + joints_per_human
                    keypoint_sequences.append(arr[:, s:e, :])

    # Daca skeleton lipseste, folosim bbox-urile umane ca proxy
    if len(keypoint_sequences) < num_humans and hbox_store is not None and video_id in hbox_store:
        hbox_data = hbox_store[video_id]
        existing = len(keypoint_sequences)

        if hasattr(hbox_data, "keys"):
            human_keys = sorted([k for k in hbox_data.keys()
                                  if k.lower().startswith("human")])
            for hk in human_keys[existing:num_humans]:
                arr = np.array(hbox_data[hk], dtype=np.float32)  # (S, 4) [x1,y1,x2,y2]
                arr = _bbox_to_keypoints(arr, S)                  # (S, 4, 2) corners
                keypoint_sequences.append(arr)
        elif hasattr(hbox_data, "shape"):
            arr = np.array(hbox_data, dtype=np.float32)
            if arr.ndim == 3:
                for h in range(existing, min(arr.shape[1], num_humans)):
                    kp = _bbox_to_keypoints(arr[:, h, :], S)
                    keypoint_sequences.append(kp)

    # Fallback: completeaza cu zerouri daca tot nu avem suficienti umani
    while len(keypoint_sequences) < num_humans:
        J_h = joints_per_human if keypoint_sequences else 4
        keypoint_sequences.append(np.zeros((S, J_h, 2), dtype=np.float32))

    # -----------------------------------------------------------------------
    # 2. Bbox corners obiecte (4 keypoints per obiect)
    # -----------------------------------------------------------------------
    for _ in range(num_objects):
        keypoint_sequences.append(np.zeros((S, 4, 2), dtype=np.float32))

    if obox_store is not None and video_id in obox_store:
        obox_data = obox_store[video_id]
        obj_idx = 0

        if hasattr(obox_data, "keys"):
            obj_keys = sorted([k for k in obox_data.keys()
                                if not k.lower().startswith("human")])
            for ok in obj_keys[:num_objects]:
                arr = np.array(obox_data[ok], dtype=np.float32)
                kp = _bbox_to_keypoints(arr, S)   # (S, 4, 2)
                keypoint_sequences[num_humans + obj_idx] = kp
                obj_idx += 1

        elif hasattr(obox_data, "shape"):
            arr = np.array(obox_data, dtype=np.float32)
            if arr.ndim == 3:
                for o in range(min(arr.shape[1], num_objects)):
                    kp = _bbox_to_keypoints(arr[:, o, :], S)
                    keypoint_sequences[num_humans + o] = kp

    # -----------------------------------------------------------------------
    # 3. Construieste geo_features: (S, J_total, 4) — pozitie + viteza
    # -----------------------------------------------------------------------
    # Concateneaza toti keypoints: (S, J_total, 2)
    all_positions = np.concatenate(keypoint_sequences, axis=1)  # (S, J, 2)

    # Viteza = diferenta pozitie fata de frame-ul anterior
    velocities = np.zeros_like(all_positions)                    # (S, J, 2)
    velocities[1:] = all_positions[1:] - all_positions[:-1]     # frame[t] - frame[t-1]

    # Concateneaza pozitie + viteza: (S, J, 4)
    geo = np.concatenate([all_positions, velocities], axis=-1).astype(np.float32)

    # -----------------------------------------------------------------------
    # 4. Entity types: (M,) — 0=human, 1=object
    # -----------------------------------------------------------------------
    entity_types = np.array(
        [0] * num_humans + [1] * num_objects,
        dtype=np.int64,
    )

    return geo, entity_types


def _normalize_keypoint_array(arr: np.ndarray, S: int) -> np.ndarray:
    """
    Normalizeaza un array de keypoints la forma (S, J, 2).
    Gestioneaza diverse formate intalnite in zarr-urile 2G-GCN.
    """
    arr = arr.astype(np.float32)

    if arr.ndim == 2:
        # (S, J*2) — x si y interleaved
        if arr.shape[0] == S:
            J = arr.shape[1] // 2
            arr = arr.reshape(S, J, 2)
        else:
            # (J, 2) — un singur frame, repeta pe S
            arr = np.tile(arr[np.newaxis], (S, 1, 1))

    elif arr.ndim == 3:
        # (S, J, 2) — direct
        if arr.shape[0] != S:
            # Poate fi (J, S, 2) — transpunem
            arr = arr.transpose(1, 0, 2)
        arr = arr[:S]

    # Asigura exact S frame-uri (trunchiem sau completam cu zerouri)
    if arr.shape[0] < S:
        pad = np.zeros((S - arr.shape[0], arr.shape[1], 2), dtype=np.float32)
        arr = np.concatenate([arr, pad], axis=0)
    elif arr.shape[0] > S:
        arr = arr[:S]

    return arr  # (S, J, 2)


def _bbox_to_keypoints(bbox: np.ndarray, S: int) -> np.ndarray:
    """
    Converteste un array de bounding boxes [x1, y1, x2, y2]
    in 4 keypoints (colturile bbox-ului): shape (S, 4, 2).

    Colturile: top-left, top-right, bottom-right, bottom-left.
    """
    bbox = bbox.astype(np.float32)

    if bbox.ndim == 1:
        # Un singur bbox (4,) — repeta pe S frame-uri
        bbox = np.tile(bbox[np.newaxis], (S, 1))

    # Asigura (S, 4)
    if bbox.shape[0] < S:
        pad = np.zeros((S - bbox.shape[0], 4), dtype=np.float32)
        bbox = np.concatenate([bbox, pad], axis=0)
    bbox = bbox[:S]

    x1, y1, x2, y2 = bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3]

    # Cele 4 colturi: (S, 4, 2)
    corners = np.stack([
        np.stack([x1, y1], axis=-1),   # top-left
        np.stack([x2, y1], axis=-1),   # top-right
        np.stack([x2, y2], axis=-1),   # bottom-right
        np.stack([x1, y2], axis=-1),   # bottom-left
    ], axis=1)  # (S, 4, 2)

    return corners


def _extract_bboxes(
    hbox_store: Optional[zarr.Group],
    obox_store: Optional[zarr.Group],
    video_id: str,
    S: int,
    num_humans: int,
    num_objects: int,
) -> np.ndarray:
    """
    Extrage bounding boxes brute (x1, y1, x2, y2) per entitate per frame.
    Ordinea: [Human1, Human2, ..., Object1, Object2, ...].

    Returns:
        bboxes: (S, M, 4) float32 — padding cu zerouri daca S difera
    """
    M = num_humans + num_objects
    bboxes = np.zeros((S, M, 4), dtype=np.float32)

    if hbox_store is not None and video_id in hbox_store:
        hbox_data = hbox_store[video_id]
        if hasattr(hbox_data, "keys"):
            human_keys = sorted([k for k in hbox_data.keys()
                                 if k.lower().startswith("human")])
            for i, hk in enumerate(human_keys[:num_humans]):
                arr = np.array(hbox_data[hk], dtype=np.float32)  # (S, 4)
                if arr.ndim == 2:
                    n = min(arr.shape[0], S)
                    bboxes[:n, i] = arr[:n]
        elif hasattr(hbox_data, "shape"):
            arr = np.array(hbox_data, dtype=np.float32)
            if arr.ndim == 3:
                n = min(arr.shape[0], S)
                for i in range(min(arr.shape[1], num_humans)):
                    bboxes[:n, i] = arr[:n, i, :]

    if obox_store is not None and video_id in obox_store:
        obox_data = obox_store[video_id]
        if hasattr(obox_data, "keys"):
            obj_keys = sorted([k for k in obox_data.keys()
                               if not k.lower().startswith("human")])
            for i, ok in enumerate(obj_keys[:num_objects]):
                arr = np.array(obox_data[ok], dtype=np.float32)  # (S, 4)
                if arr.ndim == 2:
                    n = min(arr.shape[0], S)
                    bboxes[:n, num_humans + i] = arr[:n]
        elif hasattr(obox_data, "shape"):
            arr = np.array(obox_data, dtype=np.float32)
            if arr.ndim == 3:
                n = min(arr.shape[0], S)
                for i in range(min(arr.shape[1], num_objects)):
                    bboxes[:n, num_humans + i] = arr[:n, i, :]

    return bboxes


# ---------------------------------------------------------------------------
# Convertor zarr -> numpy (pas intermediar, rulat O SINGURA DATA)
# ---------------------------------------------------------------------------

def convert_zarr_to_npy(data_root: str, output_dir: Optional[str] = None) -> List[str]:
    """
    Converteste features zarr in fisiere .npy compatibile cu pipeline-ul VHOIP.
    Genereaza: _roi.npy, _geo.npy, _entity_types.npy, _seg.npy, _frame.npy.
    Bounding boxes sunt salvate ca _bbox.npy (necesare pentru extragerea
    reala a features CLIP din crop-uri de imagini).

    Fisierele _clip.npy NU sunt create aici — se extrag separat din video-uri
    brute cu extract_clip_features_from_videos().

    Se ruleaza O SINGURA DATA (skip daca fisierele exista deja).

    Args:
        data_root:  directorul radacina al MPHOI-72
        output_dir: directorul de output (default: data_root)

    Returns:
        Lista video_id-urilor convertite.
    """
    if output_dir is None:
        output_dir = data_root

    features_out = os.path.join(output_dir, "features")
    labels_out   = os.path.join(output_dir, "labels")
    os.makedirs(features_out, exist_ok=True)
    os.makedirs(labels_out, exist_ok=True)

    features_dir = os.path.join(data_root, "mphoi_derived_features")

    # Deschide zarr-urile
    print("Deschid faster_rcnn.zarr...")
    rcnn_store = zarr.open(os.path.join(features_dir, "faster_rcnn.zarr"), mode="r")

    # Zarr-urile geometrice sunt optionale (pose poate lipsi pe unele versiuni)
    pose_store = _open_zarr_optional(os.path.join(features_dir, "human_pose.zarr"), "human_pose.zarr")
    hbox_store = _open_zarr_optional(os.path.join(features_dir, "human_bounding_boxes.zarr"), "human_bounding_boxes.zarr")
    obox_store = _open_zarr_optional(os.path.join(features_dir, "object_bounding_boxes.zarr"), "object_bounding_boxes.zarr")

    print("Incarc ground truth labels...")
    gt_data    = load_ground_truth(data_root)
    action_map = load_action_mapping(data_root)
    print(f"  {len(gt_data)} video-uri, {len(action_map)} clase: {list(action_map.values())}")

    converted, skipped, errors = 0, 0, 0

    for video_id, video_info in gt_data.items():
        roi_out_path = os.path.join(features_out, f"{video_id}_roi.npy")

        if os.path.exists(roi_out_path):
            skipped += 1
            continue

        # --- ROI features ---
        try:
            roi_features = _extract_roi_features(rcnn_store, video_id, video_info)
        except KeyError as e:
            print(f"  WARN: video {video_id} nu are ROI features: {e}")
            errors += 1
            continue

        S, M, D = roi_features.shape

        # --- Etichete ---
        seg_labels, frame_labels = _extract_labels(video_info, S, M)

        # --- Numarul de umani si obiecte (pentru geo) ---
        num_humans, num_objects = _count_entity_types(video_info, M)

        # --- Geometric features ---
        try:
            geo, entity_types = _extract_geo_features(
                pose_store=pose_store,
                hbox_store=hbox_store,
                obox_store=obox_store,
                video_id=video_id,
                S=S,
                num_humans=num_humans,
                num_objects=num_objects,
            )
        except Exception as e:
            print(f"  WARN: geo features pentru {video_id} au esuat ({e}). Folosesc zerouri.")
            J = num_humans * KINECT_JOINTS_DEFAULT + num_objects * 4
            geo = np.zeros((S, J, GEO_FEAT_DIM), dtype=np.float32)
            entity_types = np.array([0] * num_humans + [1] * num_objects, dtype=np.int64)

        # --- Bounding boxes (pentru extragerea CLIP din video-uri brute) ---
        bboxes = _extract_bboxes(hbox_store, obox_store, video_id, S, num_humans, num_objects)

        # --- Salveaza ---
        np.save(roi_out_path,                                          roi_features.astype(np.float32))
        np.save(os.path.join(features_out, f"{video_id}_geo.npy"),    geo)
        np.save(os.path.join(features_out, f"{video_id}_entity_types.npy"), entity_types)
        np.save(os.path.join(features_out, f"{video_id}_bbox.npy"),   bboxes)
        np.save(os.path.join(labels_out,   f"{video_id}_seg.npy"),    seg_labels.astype(np.int64))
        np.save(os.path.join(labels_out,   f"{video_id}_frame.npy"),  frame_labels.astype(np.int64))

        converted += 1
        if converted % 10 == 0:
            print(f"  Convertite: {converted}")

    print(f"\nConversie completa: {converted} convertite, {skipped} deja existente, {errors} erori.")
    return list(gt_data.keys())


def _open_zarr_optional(path: str, name: str) -> Optional[zarr.Group]:
    """Deschide un zarr daca exista, altfel returneaza None cu avertisment."""
    if not os.path.exists(path):
        print(f"  WARN: {name} nu exista — features geometrice vor fi zerouri pentru aceasta sursa.")
        return None
    try:
        return zarr.open(path, mode="r")
    except Exception as e:
        print(f"  WARN: nu pot deschide {name}: {e}")
        return None


def _count_entity_types(video_info: dict, M: int) -> Tuple[int, int]:
    """
    Determina numarul de umani si obiecte dintr-un video.

    Incearca sa extraga din video_info (structura JSON) sau
    foloseste valorile default MPHOI-72: 2 umani + M-2 obiecte.
    """
    if isinstance(video_info, dict) and "entities" in video_info:
        entities = video_info["entities"]
        num_humans  = sum(1 for e in entities if e.get("type", "") == "human")
        num_objects = sum(1 for e in entities if e.get("type", "") == "object")
        if num_humans + num_objects == M:
            return num_humans, num_objects

    # Default MPHOI-72: 2 umani + restul obiecte (2-4 obiecte)
    num_humans = min(2, M)
    num_objects = M - num_humans
    return num_humans, num_objects


def _extract_roi_features(
    rcnn_store: zarr.Group,
    video_id: str,
    video_info: dict,
) -> np.ndarray:
    """Extrage ROI features pentru un video din zarr. Output: (S, M, 2048)."""
    if video_id not in rcnn_store:
        raise KeyError(f"video_id '{video_id}' nu e in zarr")

    data = rcnn_store[video_id]

    if hasattr(data, "shape"):
        arr = np.array(data)

    elif hasattr(data, "keys"):
        keys = list(data.keys())

        if "Human1" in keys and "Human2" in keys:
            # Format observat in MPHOI-72: group cu Human1, Human2, objects
            h1 = np.array(data["Human1"], dtype=np.float32)  # (S, 2048)
            h2 = np.array(data["Human2"], dtype=np.float32)
            S  = min(h1.shape[0], h2.shape[0])
            parts = [h1[:S, np.newaxis, :], h2[:S, np.newaxis, :]]  # lista (S, 1, 2048)

            if "objects" in data:
                obj = np.array(data["objects"], dtype=np.float32)
                if obj.ndim == 2:
                    obj = obj[:S, np.newaxis, :]   # (S, 1, 2048)
                elif obj.ndim == 3:
                    obj = obj[:S]                   # (S, O, 2048)
                parts.append(obj)

            arr = np.concatenate(parts, axis=1)    # (S, M, 2048)

        elif "features" in keys:
            arr = np.array(data["features"], dtype=np.float32)
        else:
            arr = np.array(data[keys[0]], dtype=np.float32)
    else:
        raise ValueError(f"Format necunoscut pentru {video_id}")

    # Asigura forma (S, M, 2048)
    if arr.ndim == 2:
        M = _infer_num_entities(video_info)
        S = arr.shape[0] // M
        arr = arr.reshape(S, M, -1)
    elif arr.ndim != 3:
        raise ValueError(f"Shape neasteptat pentru {video_id}: {arr.shape}")

    return arr.astype(np.float32)


def _infer_num_entities(video_info: dict) -> int:
    """Deduce numarul de entitati M dintr-un video."""
    if isinstance(video_info, dict):
        if "entities" in video_info:
            return len(video_info["entities"])
        if "num_entities" in video_info:
            return int(video_info["num_entities"])
    return 4  # default MPHOI-72: 2 umani + 2 obiecte


def _extract_labels(
    video_info: dict,
    S: int,
    M: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extrage etichetele segment si frame pentru un video.

    Returns:
        seg_labels:   (N,) = (S*M,)  etichete per entitate per frame (flatten)
        frame_labels: (N,) = (S*M,)  identic cu seg_labels pentru compatibilitate
    """
    if isinstance(video_info, dict) and "entities" in video_info:
        labels_per_entity = []
        for entity in video_info["entities"]:
            lbl = entity.get("labels", entity.get("actions", []))
            lbl_arr = np.array(lbl, dtype=np.int64)
            # Ajusteaza la S frame-uri
            if len(lbl_arr) >= S:
                lbl_arr = lbl_arr[:S]
            else:
                lbl_arr = np.pad(lbl_arr, (0, S - len(lbl_arr)), constant_values=lbl_arr[-1] if len(lbl_arr) > 0 else 0)
            labels_per_entity.append(lbl_arr)

        if labels_per_entity:
            # Construim (S, M) si aplatizam la (S*M,) in ordine frame-major (C-order):
            # [s0m0, s0m1, ..., s0mM-1, s1m0, ...] — identic cu bms_to_bn din backbone.
            label_matrix = np.stack(labels_per_entity, axis=1)  # (S, M)
            flat_labels = label_matrix.flatten()                  # (S*M,)
            return flat_labels, flat_labels.copy()

    # Fallback: zerouri
    N = S * M
    return np.zeros(N, dtype=np.int64), np.zeros(N, dtype=np.int64)


# ---------------------------------------------------------------------------
# Generare splits cross-validare (two-subject-out pentru MPHOI-72)
# ---------------------------------------------------------------------------

def prepare_mphoi72_splits(data_root: str) -> None:
    """
    Genereaza split-urile cross-validare two-subject-out conform paperului.

    MPHOI-72 are 5 subiecti (8 grupuri de 2). Strategia din paper:
    la fiecare fold, 2 subiecti (un grup) sunt in test, restul in train.

    Se apeleaza dupa convert_zarr_to_npy().
    """
    splits_dir = os.path.join(data_root, "splits")
    os.makedirs(splits_dir, exist_ok=True)

    gt_data = load_ground_truth(data_root)

    video_ids   = []
    subject_ids = []

    for video_id, info in gt_data.items():
        video_ids.append(video_id)
        subj = None
        if isinstance(info, dict):
            subj = info.get("subject_id", None)
        if subj is None:
            m = re.search(r"Subject(\d+)", str(video_id), flags=re.IGNORECASE)
            if m:
                subj = int(m.group(1))
            else:
                for p in str(video_id).split("_"):
                    if p.startswith("S") and p[1:].isdigit():
                        subj = int(p[1:])
                        break
        subject_ids.append(int(subj) if subj is not None else 1)

    unique_subjects = sorted(set(subject_ids))
    print(f"Subiecti gasiti: {unique_subjects}")

    from itertools import combinations
    pairs = list(combinations(unique_subjects, 2))

    for fold, (s1, s2) in enumerate(pairs):
        train_ids = [v for v, s in zip(video_ids, subject_ids) if s not in (s1, s2)]
        test_ids  = [v for v, s in zip(video_ids, subject_ids) if s in (s1, s2)]

        with open(os.path.join(splits_dir, f"train_{fold}.txt"), "w") as f:
            f.write("\n".join(str(v) for v in train_ids))
        with open(os.path.join(splits_dir, f"test_{fold}.txt"), "w") as f:
            f.write("\n".join(str(v) for v in test_ids))

        print(f"  Fold {fold} (S{s1}+S{s2} in test): train={len(train_ids)}, test={len(test_ids)}")

    print(f"\n{len(pairs)} fold-uri salvate in {splits_dir}/")


# ---------------------------------------------------------------------------
# Dataset PyTorch
# ---------------------------------------------------------------------------

class MPHOI72ZarrDataset(Dataset):
    """
    Dataset PyTorch pentru MPHOI-72 care citeste fisierele .npy
    generate de convert_zarr_to_npy().

    Returneaza per sample:
        roi_features:   (S, M, 2048)  FloatTensor — input vizual Faster R-CNN
        geo_features:   (S, J, 4)     FloatTensor — keypoints geometrice (pozitie+viteza)
        entity_types:   (M,)          LongTensor  — 0=human, 1=object
        seg_labels:     (N,)          LongTensor  — etichete segment-level (N=S*M)
        frame_labels:   (N,)          LongTensor  — etichete frame-level
        clip_features:  (S, M, 512)   FloatTensor — features CLIP (placeholder sau reale)
        video_id:       str
    """

    NUM_CLASSES = 13
    ACTIVITY_LABELS = MPHOI72_ACTIVITY_LABELS

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        fold: int = 0,
        roi_dim: int = 2048,
        clip_dim: int = 512,
    ):
        self.data_root  = data_root
        self.split      = split
        self.fold       = fold
        self.roi_dim    = roi_dim
        self.clip_dim   = clip_dim
        self.num_classes = self.NUM_CLASSES

        split_file = os.path.join(data_root, "splits", f"{split}_{fold}.txt")
        if not os.path.exists(split_file):
            raise FileNotFoundError(
                f"Nu gasesc {split_file}.\n"
                "Ruleaza mai intai:\n"
                "  from data.mphoi72_dataset import convert_zarr_to_npy, prepare_mphoi72_splits\n"
                "  convert_zarr_to_npy('data/mphoi72/')\n"
                "  prepare_mphoi72_splits('data/mphoi72/')"
            )

        with open(split_file, "r") as f:
            self.video_ids = [l.strip() for l in f if l.strip()]

        print(f"MPHOI72 [{split}/fold{fold}]: {len(self.video_ids)} video-uri")

    def __len__(self) -> int:
        return len(self.video_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        video_id  = self.video_ids[idx]
        feat_dir  = os.path.join(self.data_root, "features")
        label_dir = os.path.join(self.data_root, "labels")

        # --- ROI features ---
        roi = np.load(os.path.join(feat_dir, f"{video_id}_roi.npy"))

        # --- CLIP features (placeholder zeros sau reale dupa extract_clip_features) ---
        clip_path = os.path.join(feat_dir, f"{video_id}_clip.npy")
        clip = np.load(clip_path) if os.path.exists(clip_path) else np.zeros(
            (roi.shape[0], roi.shape[1], self.clip_dim), dtype=np.float32
        )

        # --- Geometric features ---
        geo_path = os.path.join(feat_dir, f"{video_id}_geo.npy")
        if os.path.exists(geo_path):
            geo = np.load(geo_path)
        else:
            # Fallback: zerouri daca fisierul nu exista (versiune veche de date)
            S, M = roi.shape[:2]
            J    = 2 * KINECT_JOINTS_DEFAULT + (M - 2) * 4
            geo  = np.zeros((S, J, GEO_FEAT_DIM), dtype=np.float32)

        # --- Entity types ---
        etypes_path = os.path.join(feat_dir, f"{video_id}_entity_types.npy")
        if os.path.exists(etypes_path):
            entity_types = np.load(etypes_path)
        else:
            # Fallback: primii 2 sunt umani, restul obiecte
            M            = roi.shape[1]
            entity_types = np.array([0, 0] + [1] * (M - 2), dtype=np.int64)

        # --- Etichete ---
        seg   = np.load(os.path.join(label_dir, f"{video_id}_seg.npy"))
        frame = np.load(os.path.join(label_dir, f"{video_id}_frame.npy"))

        return {
            "video_id":      video_id,
            "roi_features":  torch.FloatTensor(roi),           # (S, M, 2048)
            "geo_features":  torch.FloatTensor(geo),           # (S, J, 4)
            "entity_types":  torch.LongTensor(entity_types),   # (M,)
            "clip_features": torch.FloatTensor(clip),          # (S, M, 512)
            "seg_labels":    torch.LongTensor(seg),            # (N,)
            "frame_labels":  torch.LongTensor(frame),          # (N,)
        }


def collate_fn(batch: List[Dict]) -> Dict:
    """
    Collate function pentru DataLoader.

    Gestioneaza video-uri cu dimensiuni diferite (S si J variabile)
    prin padding la maximul din batch.

    Args:
        batch: lista de sample-uri din __getitem__

    Returns:
        dict cu tensori batch-uiti, padded la (B, max_S, *, *)
    """
    video_ids = [s["video_id"] for s in batch]

    max_S = max(s["roi_features"].shape[0] for s in batch)
    max_J = max(s["geo_features"].shape[1] for s in batch)
    max_N = max(s["seg_labels"].shape[0]   for s in batch)

    B  = len(batch)
    M  = batch[0]["roi_features"].shape[1]
    D  = batch[0]["roi_features"].shape[2]
    C  = batch[0]["clip_features"].shape[2]

    roi_batch    = torch.zeros(B, max_S, M, D)
    geo_batch    = torch.zeros(B, max_S, max_J, GEO_FEAT_DIM)
    clip_batch   = torch.zeros(B, max_S, M, C)
    etypes_batch = torch.zeros(B, M, dtype=torch.long)
    seg_batch    = torch.full((B, max_N), -1, dtype=torch.long)   # -1 = ignored by CE
    frame_batch  = torch.full((B, max_N), -1, dtype=torch.long)

    for i, s in enumerate(batch):
        Si  = s["roi_features"].shape[0]
        Ji  = s["geo_features"].shape[1]
        Ni  = s["seg_labels"].shape[0]

        roi_batch[i, :Si]         = s["roi_features"]
        geo_batch[i, :Si, :Ji]    = s["geo_features"]
        clip_batch[i, :Si]        = s["clip_features"]
        etypes_batch[i]           = s["entity_types"]
        seg_batch[i, :Ni]         = s["seg_labels"]
        frame_batch[i, :Ni]       = s["frame_labels"]

    return {
        "video_ids":     video_ids,
        "roi_features":  roi_batch,    # (B, S, M, 2048)
        "geo_features":  geo_batch,    # (B, S, J, 4)
        "entity_types":  etypes_batch, # (B, M)
        "clip_features": clip_batch,   # (B, S, M, 512)
        "seg_labels":    seg_batch,    # (B, N)
        "frame_labels":  frame_batch,  # (B, N)
    }


# ---------------------------------------------------------------------------
# CLIP Visual Feature Extraction din video-uri brute (conform paper §3.2)
# ---------------------------------------------------------------------------

def extract_clip_features_from_videos(
    data_root: str,
    videos_dir: str,
    model_name: str = "ViT-B/16",
    device: str = "cuda",
    output_dir: Optional[str] = None,
) -> None:
    """
    Extrage features CLIP vizuale reale din crop-uri de imagini brute.

    Pentru fiecare video, citeste frame-urile din fisierul video,
    decupeaza regiunile de interes folosind bounding boxes salvate,
    si ruleaza crop-urile prin encoder-ul CLIP vizual real.

    Aceasta functie aliniaza codul cu paper-ul VHOIP §3.2, Fig. 2,
    unde features CLIP vizuale sunt extrase din crop-uri reale,
    nu prin proiectie artificiala a ROI pooling.

    Args:
        data_root:  directorul radacina al MPHOI-72 (contine features/ si labels/)
        videos_dir: directorul cu fisiere video brute (.mp4, .avi, etc.)
        model_name: modelul CLIP (default ViT-B/16, conform paper)
        device:     'cuda' sau 'cpu'
        output_dir: directorul de output (default: data_root/features/)
    """
    try:
        import clip as clip_lib
        import cv2
        from PIL import Image
    except ImportError as e:
        raise ImportError(
            f"Dependente lipsa pentru extragerea CLIP: {e}. "
            "Ruleaza: pip install git+https://github.com/openai/CLIP.git opencv-python Pillow"
        )

    if output_dir is None:
        output_dir = os.path.join(data_root, "features")
    os.makedirs(output_dir, exist_ok=True)

    features_dir = os.path.join(data_root, "features")
    bbox_files = sorted(f for f in os.listdir(features_dir) if f.endswith("_bbox.npy"))

    if not bbox_files:
        raise RuntimeError(
            "Nu am gasit fisiere _bbox.npy in features/. "
            "Ruleaza mai intai convert_zarr_to_npy() pentru a genera bounding boxes."
        )

    print(f"\nIncarc CLIP {model_name} pe {device}...")
    clip_model, clip_preprocess = clip_lib.load(model_name, device=device)
    clip_model.eval()
    clip_dtype = clip_model.visual.transformer.resblocks[0].attn.in_proj_weight.dtype
    print(f"  CLIP incarcat (dtype: {clip_dtype}).")

    # Mapare video_id -> path video
    video_exts = (".mp4", ".avi", ".mov", ".mkv")
    video_files = {}
    for root, dirs, files in os.walk(videos_dir):
        for f in files:
            if f.lower().endswith(video_exts):
                vid_id = os.path.splitext(f)[0]
                video_files[vid_id] = os.path.join(root, f)

    processed, skipped, missing_video = 0, 0, 0

    for bbox_file in bbox_files:
        video_id = bbox_file.replace("_bbox.npy", "")
        clip_out_path = os.path.join(output_dir, f"{video_id}_clip.npy")

        if os.path.exists(clip_out_path):
            skipped += 1
            continue

        # Load bounding boxes
        bboxes = np.load(os.path.join(features_dir, bbox_file))  # (S, M, 4)
        S, M, _ = bboxes.shape

        # Cauta fisier video (case-insensitive)
        video_path = video_files.get(video_id)
        if video_path is None:
            for vid_id, path in video_files.items():
                if vid_id.lower() == video_id.lower():
                    video_path = path
                    break

        if video_path is None or not os.path.exists(video_path):
            print(f"  [SKIP] {video_id}: nu am gasit video in {videos_dir}")
            missing_video += 1
            continue

        # Citeste frame-uri
        from data.preprocess import VideoReader
        reader = VideoReader(video_path)
        frames = reader.read_frames()

        if not frames:
            print(f"  [SKIP] {video_id}: nu am putut citi frame-uri")
            missing_video += 1
            continue

        n_frames = len(frames)
        if n_frames != S:
            print(f"  [WARN] {video_id}: video are {n_frames} frame-uri, bbox are {S}. Folosesc min.")
            n = min(n_frames, S)
            frames = frames[:n]
            bboxes = bboxes[:n]
            S = n

        # Extrage CLIP din crop-uri reale
        clip_all = np.zeros((S, M, 512), dtype=np.float32)

        with torch.no_grad():
            for s in range(S):
                frame_bgr = frames[s]
                H, W = frame_bgr.shape[:2]
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

                crops = []
                for m in range(M):
                    x1, y1, x2, y2 = bboxes[s, m]

                    # Verifica daca box-ul e valid
                    if x2 - x1 < 1 or y2 - y1 < 1:
                        crop = frame_rgb
                    else:
                        x1, y1 = max(0, int(x1)), max(0, int(y1))
                        x2, y2 = min(W, int(x2)), min(H, int(y2))
                        crop = frame_rgb[y1:y2, x1:x2]
                        if crop.size == 0:
                            crop = frame_rgb

                    pil_crop = Image.fromarray(crop)
                    crops.append(clip_preprocess(pil_crop))

                if crops:
                    crops_tensor = torch.stack(crops).to(device=device, dtype=clip_dtype)
                    feats = clip_model.encode_image(crops_tensor)  # (M, 512)
                    feats = torch.nn.functional.normalize(feats.float(), dim=-1)
                    clip_all[s] = feats.cpu().numpy()

        np.save(clip_out_path, clip_all.astype(np.float32))
        print(
            f"  {video_id}: shape={clip_all.shape}, "
            f"norm_mean={np.linalg.norm(clip_all, axis=-1).mean():.4f}"
        )
        processed += 1

    print(f"\nCLIP features extrase: {processed} video-uri, {skipped} skip, {missing_video} fara video.")