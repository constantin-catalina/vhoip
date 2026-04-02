"""
checkpoint.py
Salvare si incarcare modele.
"""

import os
import torch
from typing import Optional


def build_checkpoint_state(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict,
) -> dict:
    return {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict,
    checkpoint_dir: str,
    is_best: bool = False,
    filename: Optional[str] = None,
    save_local: bool = True,
) -> dict:
    state = build_checkpoint_state(model, optimizer, epoch, metrics)

    if not save_local:
        return {"checkpoint": None, "best_checkpoint": None, "state": state}

    os.makedirs(checkpoint_dir, exist_ok=True)

    fname = filename or f"epoch_{epoch:03d}.pth"
    path = os.path.join(checkpoint_dir, fname)
    torch.save(state, path)

    saved = {"checkpoint": path, "best_checkpoint": None, "state": state}

    if is_best:
        best_path = os.path.join(checkpoint_dir, "best_model.pth")
        torch.save(state, best_path)
        saved["best_checkpoint"] = best_path
        print(f"  --> Salvat best model (FSUM={metrics.get('fsum', 0):.1f})")

    print(f"  --> Checkpoint salvat: {path}")
    return saved


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cuda",
) -> dict:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    print(f"  --> Incarcat checkpoint din epoch {checkpoint['epoch']}")
    return checkpoint