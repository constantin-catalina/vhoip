"""
setup_mphoi72.py
Script de setup complet pentru MPHOI-72.
Ruleaza o singura data dupa ce ai dezarhivat MPHOI-72.zip.

Pasi:
  1. Inspecteaza structura zarr (afiseaza formatul exact)
  2. Converteste zarr -> .npy compatibil cu VHOIP
  3. Genereaza splits cross-validare
  4. Verifica dataset-ul

Utilizare:
    # Pasul 0: dezarhiveaza
    # MPHOI-72.zip -> data/mphoi72/

    # Pasul 1: inspecteaza structura (important - ruleaza prima data)
    python setup_mphoi72.py --data_root data/mphoi72/ --inspect

    # Pasul 2: conversie completa
    python setup_mphoi72.py --data_root data/mphoi72/

    # Daca ai erori de format, editeaza functiile _extract_roi_features()
    # si _extract_labels() din data/mphoi72_dataset.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

import argparse
import os
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data/mphoi72/",
                        help="Directorul cu datele MPHOI-72 dezarhivate")
    parser.add_argument("--inspect", action="store_true",
                        help="Doar inspecteaza structura, nu converti")
    parser.add_argument("--verify", action="store_true",
                        help="Verifica fisierele generate")
    parser.add_argument("--extract_clip", action="store_true",
                        help="Extrage features CLIP vizuale reale (ruleaza dupa conversie)")
    parser.add_argument("--clip_device", type=str, default="cuda",
                        help="Device pentru extractia CLIP (cuda/cpu)")
    parser.add_argument("--clip_batch_size", type=int, default=64,
                        help="Batch size pentru inferenta CLIP")
    return parser.parse_args()


def inspect_json_structure(data_root: str) -> None:
    """Afiseaza structura JSON-urilor pentru a intelege formatul."""

    print("\n" + "=" * 60)
    print("INSPECTARE JSON-uri")
    print("=" * 60)

    # Action mapping
    action_path = os.path.join(data_root, "mphoi_action_id_to_action_name.json")
    if os.path.exists(action_path):
        with open(action_path) as f:
            actions = json.load(f)
        print(f"\nmphoi_action_id_to_action_name.json:")
        print(f"  {len(actions)} clase:")
        for k, v in sorted(actions.items(), key=lambda x: int(x[0])):
            print(f"    {k}: {v}")
    else:
        print(f"WARN: nu gasesc {action_path}")

    # Ground truth
    gt_path = os.path.join(data_root, "mphoi_ground_truth_labels.json")
    if os.path.exists(gt_path):
        with open(gt_path) as f:
            gt = json.load(f)

        print(f"\nmphoi_ground_truth_labels.json:")
        print(f"  {len(gt)} video-uri")

        # Arata primele 2 intrari pentru a intelege structura
        for i, (vid_id, info) in enumerate(list(gt.items())[:2]):
            print(f"\n  Video '{vid_id}':")
            if isinstance(info, dict):
                for k, v in info.items():
                    if isinstance(v, list) and len(v) > 5:
                        print(f"    {k}: [{v[0]}, {v[1]}, ... ({len(v)} elemente)]")
                    else:
                        print(f"    {k}: {v}")
            elif isinstance(info, list):
                print(f"    lista de {len(info)} elemente")
                if info:
                    print(f"    primul element: {info[0]}")
    else:
        print(f"WARN: nu gasesc {gt_path}")


def inspect_zarr_structure(data_root: str) -> None:
    """Afiseaza structura zarr-urilor."""
    try:
        import zarr
    except ImportError:
        print("WARN: zarr nu e instalat. Ruleaza: pip install zarr")
        return

    print("\n" + "=" * 60)
    print("INSPECTARE ZARR-uri")
    print("=" * 60)

    from data.mphoi72_dataset import inspect_zarr
    inspect_zarr(data_root)


def run_conversion(data_root: str) -> None:
    """Ruleaza conversia completa zarr -> npy + generare splits."""
    try:
        import zarr
    except ImportError:
        print("ERROR: zarr nu e instalat.")
        print("Instaleaza cu: pip install zarr")
        sys.exit(1)

    from data.mphoi72_dataset import convert_zarr_to_npy, prepare_mphoi72_splits

    print("\n" + "=" * 60)
    print("CONVERSIE ZARR -> NPY")
    print("=" * 60)
    convert_zarr_to_npy(data_root)

    print("\n" + "=" * 60)
    print("GENERARE SPLITS")
    print("=" * 60)
    prepare_mphoi72_splits(data_root)


def verify_dataset(data_root: str) -> None:
    """Verifica fisierele generate si afiseaza statistici."""
    print("\n" + "=" * 60)
    print("VERIFICARE DATASET")
    print("=" * 60)

    features_dir = os.path.join(data_root, "features")
    labels_dir   = os.path.join(data_root, "labels")
    splits_dir   = os.path.join(data_root, "splits")

    # Numara fisiere
    if os.path.exists(features_dir):
        roi_files  = [f for f in os.listdir(features_dir) if f.endswith("_roi.npy")]
        clip_files = [f for f in os.listdir(features_dir) if f.endswith("_clip.npy")]
        print(f"\nFeatures:")
        print(f"  ROI files:  {len(roi_files)}")
        print(f"  CLIP files: {len(clip_files)}")

        if roi_files:
            # Arata dimensiunile primului fisier
            sample = np.load(os.path.join(features_dir, roi_files[0]))
            print(f"  Shape exemplu ROI: {sample.shape}")
            print(f"    S={sample.shape[0]} frames, M={sample.shape[1]} entitati, D={sample.shape[2]}")
    else:
        print("ERROR: directorul features/ nu exista. Ruleaza conversia mai intai.")
        return

    # Splits
    if os.path.exists(splits_dir):
        split_files = os.listdir(splits_dir)
        print(f"\nSplits ({len(split_files)} fisiere):")
        for sf in sorted(split_files):
            with open(os.path.join(splits_dir, sf)) as f:
                count = len([l for l in f if l.strip()])
            print(f"  {sf}: {count} video-uri")
    else:
        print("ERROR: splits/ nu exista.")
        return

    # Test incarcare dataset
    print("\nTest incarcare dataset...")
    try:
        from data.mphoi72_dataset import MPHOI72ZarrDataset
        ds = MPHOI72ZarrDataset(data_root, split="train", fold=0)
        sample = ds[0]
        print(f"  OK! Un sample:")
        for k, v in sample.items():
            if hasattr(v, "shape"):
                print(f"    {k}: {v.shape} {v.dtype}")
            else:
                print(f"    {k}: {v}")
    except Exception as e:
        print(f"  ERROR: {e}")
        return

    print("\nDataset gata de folosit!")
    print("\nPentru antrenare:")
    print("  python train.py --config configs/mphoi72.yaml --fold 0")


def main():
    args = parse_args()

    # Verifica ca data_root exista
    if not os.path.exists(args.data_root):
        print(f"ERROR: directorul '{args.data_root}' nu exista.")
        print(f"Asigura-te ca ai dezarhivat MPHOI-72.zip in {args.data_root}")
        sys.exit(1)

    print(f"Data root: {os.path.abspath(args.data_root)}")

    if args.inspect:
        # Doar inspectare - nu converti
        inspect_json_structure(args.data_root)
        inspect_zarr_structure(args.data_root)
        print("\nDupa ce ai inteles structura, ruleaza fara --inspect pentru conversie.")

    elif args.extract_clip:
        # Extrage features CLIP vizuale reale (pas separat, dupa conversie)
        from data.mphoi72_dataset import extract_clip_features
        print("\nExtrag features CLIP vizuale...")
        extract_clip_features(
            args.data_root,
            device=args.clip_device,
            batch_size=args.clip_batch_size,
        )
        print("\nDupa extragere, poti rula antrenarea:")
        print("  python train.py --config configs/mphoi72.yaml --fold 0")

    elif args.verify:
        verify_dataset(args.data_root)

    else:
        # Flux complet
        inspect_json_structure(args.data_root)
        inspect_zarr_structure(args.data_root)

        print("\nIncep conversia...")
        run_conversion(args.data_root)

        print("\nVerificare finala...")
        verify_dataset(args.data_root)

        print("\n" + "=" * 60)
        print("PAS URMATOR (OBLIGATORIU pentru VHOIP):")
        print("  Extrage features CLIP vizuale reale:")
        print("  python setup_mphoi72.py --data_root data/mphoi72/ --extract_clip")
        print("Fara acest pas, G_init va fi zero si prior-ul CLIP nu va fi folosit.")
        print("=" * 60)


if __name__ == "__main__":
    main()