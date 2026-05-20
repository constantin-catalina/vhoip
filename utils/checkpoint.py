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
    criterion: torch.nn.Module = None,
    save_frozen: bool = False,
) -> dict:
    if save_frozen:
        model_state = model.state_dict()
    else:
        # Only save trainable params + buffers (excludes frozen CLIP, saves ~500 MB)
        trainable_names = {name for name, p in model.named_parameters() if p.requires_grad}
        model_state = {
            k: v for k, v in model.state_dict().items()
            if k in trainable_names or not k.endswith(('.weight', '.bias'))
        }

    state = {
        "epoch": epoch,
        "model_state_dict": model_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }

    # Save criterion state (e.g. learnable logit_scale for CosineSimilarityLoss)
    if criterion is not None:
        trainable_crit = {name for name, p in criterion.named_parameters() if p.requires_grad}
        if trainable_crit:
            state["criterion_state_dict"] = {
                k: v for k, v in criterion.state_dict().items()
                if k in trainable_crit
            }

    return state


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict,
    checkpoint_dir: str,
    criterion: torch.nn.Module = None,
    is_best: bool = False,
    save_last: bool = False,
    save_local: bool = True,
) -> dict:
    state = build_checkpoint_state(model, optimizer, epoch, metrics, criterion=criterion)

    if not save_local:
        return {"checkpoint": None, "best_checkpoint": None, "last_checkpoint": None, "state": state}

    os.makedirs(checkpoint_dir, exist_ok=True)

    saved = {"checkpoint": None, "best_checkpoint": None, "last_checkpoint": None, "state": state}

    if save_last:
        last_path = os.path.join(checkpoint_dir, "last_model.pth")
        torch.save(state, last_path)
        saved["last_checkpoint"] = last_path

    if is_best:
        best_path = os.path.join(checkpoint_dir, "best_model.pth")
        torch.save(state, best_path)
        saved["best_checkpoint"] = best_path
        print(f"  --> Salvat best model (FSUM={metrics.get('fsum', 0):.1f})")

    return saved


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cuda",
    criterion: torch.nn.Module = None,
) -> dict:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    # strict=False allows loading partial state dicts (e.g. without frozen CLIP weights)
    missing, unexpected = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    if missing:
        print(f"  [INFO] Chei lipsa in checkpoint (parametri frozen, initializati la valori default): {len(missing)}")
    if unexpected:
        print(f"  [WARN] Chei neasteptate in checkpoint: {unexpected}")

    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Restore criterion state (e.g. learnable logit_scale)
    if criterion is not None and "criterion_state_dict" in checkpoint:
        criterion.load_state_dict(checkpoint["criterion_state_dict"], strict=False)

    print(f"  --> Incarcat checkpoint din epoch {checkpoint['epoch']}")
    return checkpoint