# inspect_cad120.py
# Inspecteaza structura datelor CAD-120.
# Utilizare:
#   python inspect_cad120.py --data_root "C:/Users/Catalina/Downloads/CAD-120"

import argparse
import os
import pickle
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str,
                        default="C:/Users/Catalina/Downloads/CAD-120")
    return parser.parse_args()


def inspect_pickle(pickle_path):
    print("=" * 60)
    print("INSPECTARE PICKLE:", pickle_path)
    print("=" * 60)

    data = None
    for enc in [None, "latin1", "bytes"]:
        try:
            with open(pickle_path, "rb") as f:
                if enc is None:
                    data = pickle.load(f)
                else:
                    data = pickle.load(f, encoding=enc)
            break
        except Exception as e:
            last_err = e
    if data is None:
        raise RuntimeError(f"Nu pot deschide pickle: {last_err}")

    print(f"Tip date: {type(data)}")

    if isinstance(data, dict):
        keys = list(data.keys())
        print(f"Nr. chei (video-uri): {len(keys)}")
        print(f"Primele 5 chei: {keys[:5]}")
        print()

        # Inspecteaza primele 2 intrari
        for key in keys[:2]:
            val = data[key]
            print(f"--- Cheie: '{key}' ---")
            print(f"  Tip valoare: {type(val)}")
            if isinstance(val, dict):
                print(f"  Sub-chei: {list(val.keys())}")
                for k, v in val.items():
                    if hasattr(v, "shape"):
                        print(f"    {k}: shape={v.shape}, dtype={v.dtype}")
                    elif isinstance(v, list):
                        print(f"    {k}: list cu {len(v)} elemente, primul={v[0] if v else 'gol'}")
                    elif isinstance(v, np.ndarray):
                        print(f"    {k}: ndarray shape={v.shape}")
                    else:
                        print(f"    {k}: {type(v).__name__} = {str(v)[:100]}")
            elif hasattr(val, "shape"):
                print(f"  Shape: {val.shape}, dtype: {val.dtype}")
            elif isinstance(val, list):
                print(f"  Lista de {len(val)} elemente")
                if val:
                    print(f"  Primul element: {type(val[0])} = {str(val[0])[:100]}")
            print()

    elif isinstance(data, list):
        print(f"Lista de {len(data)} elemente")
        print(f"Primul element tip: {type(data[0])}")
        print(f"Primul element: {str(data[0])[:200]}")
    else:
        print(f"Date: {str(data)[:300]}")


def inspect_zarr(zarr_path):
    print("=" * 60)
    print("INSPECTARE ZARR:", zarr_path)
    print("=" * 60)

    try:
        import zarr
    except ImportError:
        print("zarr nu e instalat. Ruleaza: pip install zarr")
        return

    z = zarr.open(zarr_path, mode="r")
    print(f"Tip zarr: {type(z)}")

    if hasattr(z, "keys"):
        keys = list(z.keys())
        print(f"Nr. video-uri: {len(keys)}")
        print(f"Primele 5 chei: {keys[:5]}")
        print()

        for key in keys[:2]:
            val = z[key]
            print(f"--- Cheie: '{key}' ---")
            if hasattr(val, "shape"):
                print(f"  Shape: {val.shape}, dtype: {val.dtype}")
                arr = np.array(val)
                print(f"  Min={arr.min():.4f}, Max={arr.max():.4f}, Mean={arr.mean():.4f}")
            elif hasattr(val, "keys"):
                sub_keys = list(val.keys())
                print(f"  Sub-group cu chei: {sub_keys}")
                for sk in sub_keys[:4]:
                    sv = val[sk]
                    if hasattr(sv, "shape"):
                        print(f"    {sk}: shape={sv.shape}, dtype={sv.dtype}")
                    else:
                        print(f"    {sk}: {type(sv)}")
            print()
    elif hasattr(z, "shape"):
        print(f"Array direct: shape={z.shape}, dtype={z.dtype}")


def inspect_dictionaries(dict_dir):
    print("=" * 60)
    print("INSPECTARE DICTIONARIES")
    print("=" * 60)

    for fname in os.listdir(dict_dir):
        fpath = os.path.join(dict_dir, fname)
        print(f"\n--- {fname} ---")
        with open(fpath, "r") as f:
            lines = f.readlines()
        for line in lines[:15]:
            print(f"  {line.rstrip()}")
        if len(lines) > 15:
            print(f"  ... ({len(lines)} linii total)")


def main():
    args = parse_args()
    root = args.data_root

    dict_dir = os.path.join(root, "dictionaries")
    pickle_path = os.path.join(root, "features", "preprocessed", "cad120data.pickle")
    zarr_path = os.path.join(root, "features", "faster_rcnn", "features.zarr")

    if os.path.exists(dict_dir):
        inspect_dictionaries(dict_dir)
    else:
        print(f"WARN: nu gasesc {dict_dir}")

    print()
    if os.path.exists(pickle_path):
        inspect_pickle(pickle_path)
    else:
        print(f"WARN: nu gasesc {pickle_path}")

    print()
    if os.path.exists(zarr_path):
        inspect_zarr(zarr_path)
    else:
        print(f"WARN: nu gasesc {zarr_path}")


if __name__ == "__main__":
    main()