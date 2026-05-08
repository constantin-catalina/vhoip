"""
diag_t_diversity.py
Diagnostic: verify T (text prototype) diversity after training fold 0.

Usage:
    python diag_t_diversity.py
    python diag_t_diversity.py --config configs/mphoi72_c6b.yaml --fold 0
    python diag_t_diversity.py --ckpt checkpoints/mphoi72_c6b_fold0/best_model.pth
"""

import argparse
import torch
from omegaconf import OmegaConf

from data.dataset import MPHOI72Dataset
from models.vhoip import VHOIP


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/mphoi72_c6b.yaml")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Override checkpoint path (default: derived from config+fold)")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def _resolve_defaults(config_path: str):
    """
    OmegaConf doesn't process Hydra-style `defaults` lists.
    Walk the list manually and merge in order, so that mphoi72_c6b.yaml
    correctly inherits dataset/training keys from mphoi72.yaml.
    """
    raw = OmegaConf.load(config_path)
    defaults = OmegaConf.select(raw, "defaults", default=[])
    layers = [OmegaConf.load("configs/base.yaml")]
    for entry in defaults:
        name = entry if isinstance(entry, str) else list(entry.values())[0]
        if name == "base":
            continue  # already loaded first
        layers.append(OmegaConf.load(f"configs/{name}.yaml"))
    layers.append(raw)
    cfg = layers[0]
    for layer in layers[1:]:
        cfg = OmegaConf.merge(cfg, layer)
    return cfg


def main():
    args = parse_args()
    device = torch.device(args.device)

    cfg = _resolve_defaults(args.config)

    dataset_name = OmegaConf.select(cfg, "dataset.name", default="mphoi72")
    experiment_name = f"{dataset_name}_c6b_fold{args.fold}"
    ckpt_path = args.ckpt or f"checkpoints/{experiment_name}/best_model.pth"

    label_names = MPHOI72Dataset.ACTIVITY_LABELS

    print(f"Loading model from: {ckpt_path}")
    model = VHOIP(cfg, label_names, device=str(device)).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    with torch.no_grad():
        T = model.text_encoder.encode_labels(label_names, subject="person")

    print(f"\nT shape: {T.shape}")

    norms = T.norm(dim=-1)
    print(f"T norms (should be ~1.0):\n  {norms.tolist()}")

    sim = T @ T.T
    print(f"\nT pairwise cosine similarities ({T.shape[0]}x{T.shape[0]}):")
    print(sim)

    # Off-diagonal max similarity per row (nearest neighbour cosine similarity).
    # Lower mean = more diverse T. Values close to 1.0 indicate collapse.
    sim_no_diag = sim.clone().fill_diagonal_(0)
    max_off_diag = sim_no_diag.abs().max(dim=1).values
    mean_max_off_diag = max_off_diag.mean().item()
    print(f"\nMax off-diagonal |cosine| per class: {max_off_diag.tolist()}")
    print(f"Mean max off-diagonal similarity: {mean_max_off_diag:.4f}")
    print("  (<0.90 = good diversity | >0.95 = risk of collapse)")

    min_off_diag = sim_no_diag[sim_no_diag != 0].min().item()
    print(f"Min non-zero off-diagonal similarity: {min_off_diag:.4f}")


if __name__ == "__main__":
    main()
