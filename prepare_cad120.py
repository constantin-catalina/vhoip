"""
prepare_cad120.py
Converteste datele CAD-120 din format zarr + pickle in fisiere .npy
compatibile cu pipeline-ul de antrenare VHOIP.

Genereaza:
    data/cad120/features/<video_id>_roi.npy   (S, 4, 2048) - skeleton + 3 obiecte
    data/cad120/features/<video_id>_clip.npy  (S, 4, 512)  - zeros placeholder
    data/cad120/labels/<video_id>_seg.npy     (S*4,)
    data/cad120/labels/<video_id>_frame.npy   (S*4,)
    data/cad120/splits/train_<fold>.txt
    data/cad120/splits/test_<fold>.txt        (leave-one-subject-out, 4 fold-uri)

Utilizare:
    python prepare_cad120.py
    python prepare_cad120.py --data_root data/cad120/
"""

import os
import sys
import types
import argparse
import numpy as np
import zarr


# ---------------------------------------------------------------------------
# Monkey-patch vhoi.cad120classes (pachet de cercetare, nedistribuit public)
# ---------------------------------------------------------------------------

def _patch_vhoi():
    vhoi_mod = types.ModuleType("vhoi")
    cad120_mod = types.ModuleType("vhoi.cad120classes")

    class CAD120Video:
        def __setstate__(self, d):
            self.__dict__.update(d)

    class CAD120VideoSegment:
        def __setstate__(self, d):
            self.__dict__.update(d)

    cad120_mod.CAD120Video = CAD120Video
    cad120_mod.CAD120VideoSegment = CAD120VideoSegment
    vhoi_mod.cad120classes = cad120_mod
    sys.modules["vhoi"] = vhoi_mod
    sys.modules["vhoi.cad120classes"] = cad120_mod


# ---------------------------------------------------------------------------
# Incarcare date sursa
# ---------------------------------------------------------------------------

def load_pickle(data_root: str) -> dict:
    import pickle
    path = os.path.join(data_root, "features", "preprocessed", "cad120data.pickle")
    with open(path, "rb") as f:
        return pickle.load(f)


def load_zarr(data_root: str) -> zarr.Group:
    path = os.path.join(data_root, "features", "faster_rcnn", "features.zarr")
    return zarr.open(path, mode="r")


def load_subject_map(data_root: str) -> dict:
    """Returneaza {video_id: subject_name} ex: {'0510141923': 'Subject3'}"""
    path = os.path.join(data_root, "dictionaries", "video-id_to_subject.txt")
    mapping = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split()
                if len(parts) >= 2:
                    mapping[parts[0]] = parts[1]
    return mapping


# ---------------------------------------------------------------------------
# Extragere etichete per-frame din pickle
# ---------------------------------------------------------------------------

def build_frame_labels(video_obj, num_frames: int) -> np.ndarray:
    """
    Reconstruieste eticheta subactivity per frame din segmentele video.
    Etichetele in pickle sunt 1-indexed (1..10); le convertim la 0-indexed (0..9).
    Fallback: eticheta 0 (reaching) pentru frame-urile neacoperite.
    """
    labels = np.zeros(num_frames, dtype=np.int64)

    for seg in video_obj._video_segments:
        if seg.start_frame is None or seg.subactivity is None:
            continue
        start = max(0, seg.start_frame - 1)   # 1-indexed -> 0-indexed
        end   = min(num_frames, seg.end_frame) # end inclusiv in pickle
        label = max(0, seg.subactivity - 1)    # 1-indexed -> 0-indexed
        labels[start:end] = label

    return labels


# ---------------------------------------------------------------------------
# Extragere ROI features din zarr
# ---------------------------------------------------------------------------

def extract_roi(zarr_group: zarr.Group, video_id: str, max_entities: int = 6) -> np.ndarray:
    """
    Combina skeleton (S, 2048) si objects (S, K, 2048) intr-un tensor (S, max_entities, 2048).
    Entitate 0 = skeleton, entitati 1..K = obiectele detectate, restul zero-padded.
    max_entities=6 asigura forma consistenta cu MPHOI-72 pentru batching in DataLoader.
    """
    grp = zarr_group[video_id]

    skeleton = np.array(grp["skeleton"])   # (S, 2048)
    objects  = np.array(grp["objects"])    # (S, K, 2048) unde K variaza 2-5

    S = min(skeleton.shape[0], objects.shape[0])
    skeleton = skeleton[:S]                # (S, 2048)
    objects  = objects[:S]                 # (S, K, 2048)

    K = objects.shape[1]                   # numar real de obiecte

    roi = np.zeros((S, max_entities, 2048), dtype=np.float32)
    roi[:, 0, :]    = skeleton             # slot 0: skeleton
    n_obj = min(K, max_entities - 1)
    roi[:, 1:1+n_obj, :] = objects[:, :n_obj, :]   # slot-uri 1..n_obj: obiecte

    return roi


# ---------------------------------------------------------------------------
# Generare split-uri leave-one-subject-out
# ---------------------------------------------------------------------------

def generate_splits(video_ids: list, subject_map: dict, splits_dir: str) -> None:
    subjects = sorted(set(subject_map[v] for v in video_ids if v in subject_map))
    print(f"  Subiecti: {subjects} -> {len(subjects)} fold-uri")

    os.makedirs(splits_dir, exist_ok=True)
    for fold, test_subject in enumerate(subjects):
        train = [v for v in video_ids if subject_map.get(v) != test_subject]
        test  = [v for v in video_ids if subject_map.get(v) == test_subject]

        with open(os.path.join(splits_dir, f"train_{fold}.txt"), "w") as f:
            f.write("\n".join(train))
        with open(os.path.join(splits_dir, f"test_{fold}.txt"), "w") as f:
            f.write("\n".join(test))

        print(f"  Fold {fold} (test={test_subject}): train={len(train)}, test={len(test)}")


# ---------------------------------------------------------------------------
# Script principal
# ---------------------------------------------------------------------------

def prepare(data_root: str) -> None:
    _patch_vhoi()

    features_out = os.path.join(data_root, "features")
    labels_out   = os.path.join(data_root, "labels")
    splits_out   = os.path.join(data_root, "splits")
    os.makedirs(features_out, exist_ok=True)
    os.makedirs(labels_out,   exist_ok=True)
    os.makedirs(splits_out,   exist_ok=True)

    print("Incarc pickle...")
    pickle_data = load_pickle(data_root)

    print("Incarc zarr...")
    zarr_store  = load_zarr(data_root)
    zarr_ids    = set(zarr_store.keys())

    print("Incarc subject map...")
    subject_map = load_subject_map(data_root)

    # Videoclipuri care au atat pickle cat si zarr
    valid_ids = [vid for vid in pickle_data.keys() if vid in zarr_ids]
    print(f"\n{len(valid_ids)} video-uri valide (din {len(pickle_data)} in pickle, {len(zarr_ids)} in zarr)")

    converted, skipped, errors = 0, 0, 0

    for video_id in valid_ids:
        roi_path = os.path.join(features_out, f"{video_id}_roi.npy")
        if os.path.exists(roi_path):
            skipped += 1
            continue

        try:
            roi = extract_roi(zarr_store, video_id)       # (S, 4, 2048)
        except Exception as e:
            print(f"  WARN [{video_id}] ROI: {e}")
            errors += 1
            continue

        S, M, _ = roi.shape   # M=4

        # Etichete per frame
        frame_labels = build_frame_labels(pickle_data[video_id], S)  # (S,)

        # Flatten la (S*M,): fiecare entitate intr-un frame primeste aceeasi eticheta
        seg_labels_flat   = np.repeat(frame_labels, M)   # (S*M,)
        frame_labels_flat = seg_labels_flat.copy()

        # CLIP placeholder (zeros) — poate fi inlocuit cu extract_clip_features() din mphoi72_dataset.py
        clip = np.zeros((S, M, 512), dtype=np.float32)

        np.save(roi_path, roi)
        np.save(os.path.join(features_out, f"{video_id}_clip.npy"),   clip)
        np.save(os.path.join(labels_out,   f"{video_id}_seg.npy"),    seg_labels_flat)
        np.save(os.path.join(labels_out,   f"{video_id}_frame.npy"),  frame_labels_flat)

        converted += 1

    print(f"\nFeatures: {converted} convertite, {skipped} deja existente, {errors} erori.")

    print("\nGenerez split-uri...")
    processed_ids = [v for v in valid_ids if os.path.exists(
        os.path.join(features_out, f"{v}_roi.npy")
    )]
    generate_splits(processed_ids, subject_map, splits_out)

    print(f"\nGata! Date salvate in {data_root}")
    print(f"  features/: {converted + skipped} video-uri x (S, 4, 2048) ROI + (S, 4, 512) CLIP")
    print(f"  labels/:   {converted + skipped} video-uri x (S*4,) etichete subactivity (0-9)")
    print(f"  splits/:   leave-one-subject-out, {len(set(subject_map.values()))} fold-uri")


def parse_args():
    parser = argparse.ArgumentParser(description="Pregatire date CAD-120 pentru VHOIP")
    parser.add_argument("--data_root", type=str, default="data/cad120/")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    prepare(args.data_root)
