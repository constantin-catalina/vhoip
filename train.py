"""
train.py
Entry point pentru antrenarea VHOIP.

Utilizare:
    python train.py --config configs/cad120.yaml
    python train.py --config configs/cad120.yaml --fold 0
    python train.py --config configs/mphoi72.yaml --fold 0 --resume checkpoints/epoch_010.pth
"""

import argparse
import os
import tempfile
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from data.dataset import get_dataset, CAD120Dataset, MPHOI72Dataset, BimanualDataset
from models.vhoip import VHOIP
from models.losses import VHOIPLoss
from utils.logger import Logger
from utils.metrics import compute_metrics_epoch
from utils.checkpoint import save_checkpoint, load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Antrenare VHOIP")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--wandb", action="store_true", help="Activeaza logging in Weights & Biases")
    parser.add_argument("--wandb_project", type=str, default=None, help="Numele proiectului W&B")
    parser.add_argument("--wandb_entity", type=str, default=None, help="Entity/username W&B")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="Nume custom pentru run-ul W&B")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def get_label_names(dataset_name: str):
    mapping = {
        "cad120": CAD120Dataset.ACTIVITY_LABELS,
        "mphoi72": MPHOI72Dataset.ACTIVITY_LABELS,
        "bimanual": BimanualDataset.ACTIVITY_LABELS,
    }
    return mapping[dataset_name]


def train_one_epoch(model, dataloader, optimizer, criterion, scaler, device, logger, epoch, cfg):
    model.train()
    model.set_inference_mode(False)
    total_losses = {}

    for batch_idx, batch in enumerate(dataloader):
        roi = batch["roi_features"].to(device)
        seg_labels = batch["seg_labels"].to(device)
        frame_labels = batch["frame_labels"].to(device)

        optimizer.zero_grad()

        with torch.amp.autocast(
            device_type=device.type,
            enabled=cfg.training.use_amp and device.type == "cuda",
        ):
            outputs = model(roi, labels=seg_labels)
            B, N, C = outputs["segment_logits"].shape
            losses = criterion(
                outputs["segment_logits"].reshape(B * N, C),
                outputs["frame_logits"].reshape(B * N, C),
                outputs["mi_scores"].reshape(B * N, C),
                outputs["cos_similarities"].reshape(B * N, C),
                seg_labels.reshape(B * N),
                frame_labels.reshape(B * N),
            )

        scaler.scale(losses["total"]).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.training.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        for k, v in losses.items():
            total_losses[k] = total_losses.get(k, 0) + v.item()

        if batch_idx % cfg.logging.log_every == 0:
            logger.info(
                f"  [{batch_idx}/{len(dataloader)}] "
                f"Loss={losses['total'].item():.4f} "
                f"MI={losses['l_mi'].item():.4f} "
                f"Cos={losses['l_cos'].item():.4f}"
            )

    model.end_of_epoch(epoch)
    return {k: v / len(dataloader) for k, v in total_losses.items()}


@torch.no_grad()
def initialize_global_representation(model, dataloader, device, logger):
    """
    Initializeaza G din prototipurile CLIP vizuale pe setul de train.

    IMPORTANT: clip_features trebuie sa fie extrase real cu CLIPVisualEncoder
    inainte de antrenare (via setup_mphoi72.py sau un script separat).
    Daca fisierele _clip.npy contin zerouri (placeholder), G_init va fi
    zero si beneficiul prior-ului CLIP va fi pierdut.
    """
    clip_features_all = []
    labels_all = []

    for batch in dataloader:
        clip = batch["clip_features"]             # (B, S, M, clip_dim)
        seg_labels = batch["seg_labels"]          # (B, N)

        clip_flat = clip.reshape(-1, clip.shape[-1])
        labels_flat = seg_labels.reshape(-1)

        valid_mask = labels_flat >= 0
        if valid_mask.any():
            clip_features_all.append(clip_flat[valid_mask])
            labels_all.append(labels_flat[valid_mask])

    if not clip_features_all:
        raise RuntimeError("Nu exista etichete valide pentru initializarea lui G.")

    clip_features = torch.cat(clip_features_all, dim=0).to(device)
    labels = torch.cat(labels_all, dim=0).long().to(device)

    # Verifica daca features CLIP sunt placeholder (zerouri).
    # Daca da, ridica exceptie — antrenamentul cu G_init=0 produce rezultate
    # invalide (L_Cos compara cu zero, MI loss nu are prior real).
    feature_norm = clip_features.norm(dim=-1).mean().item()
    if feature_norm < 0.5:
        # L2-normalized CLIP features should have norm ~1.0.
        # If norm << 1.0, the _clip.npy files still contain zero placeholders
        # or were extracted with the random-projection workaround.
        # Fall back to text-based G initialization (T is always valid).
        logger.info(
            f"  clip_features norm medie: {feature_norm:.4f} (placeholder/invalid).\n"
            f"  Fallback: G initializat din text features T (CLIP text encoder).\n"
            f"  Pentru CLIP vizual real, ruleaza extract_clip_features() din mphoi72_dataset.py."
        )
        model.initialize_G_from_text()
        return

    logger.info(f"  clip_features norm medie: {feature_norm:.4f} (OK)")
    model.initialize_G(clip_features, labels)
    logger.info(f"G initializat din {clip_features.shape[0]} entitati de train")


def _frames_to_segments(frame_labels):
    """Group consecutive frames with the same class label into (start, end, class) segments."""
    if not frame_labels:
        return []
    segments = []
    start = 0
    for i in range(1, len(frame_labels)):
        if frame_labels[i] != frame_labels[start]:
            segments.append((start, i, frame_labels[start]))
            start = i
    segments.append((start, len(frame_labels), frame_labels[start]))
    return segments


@torch.no_grad()
def evaluate(model, dataloader, device, iou_thresholds):
    model.set_inference_mode(True)
    all_preds, all_gts = [], []

    for batch in dataloader:
        roi = batch["roi_features"].to(device)   # (B, S, M, D)
        frame_labels = batch["frame_labels"]      # (B, S*M) frame-major
        B, S, M, _ = roi.shape

        outputs = model(roi)
        frame_pred = outputs["frame_logits"].argmax(dim=-1).cpu()  # (B, S*M)

        # frame_pred e ordonat frame-major: index [s*M + m] = frame s, entitate m.
        # Procesam fiecare entitate ca o secventa temporala independenta de lungime S,
        # astfel incat segmentele sa aiba durate reale si IoU sa varieze cu pragul.
        frame_pred_3d = frame_pred.reshape(B, S, M)        # (B, S, M)
        frame_labels_3d = frame_labels.reshape(B, S, M)    # (B, S, M)

        for b in range(B):
            for m in range(M):
                entity_pred = frame_pred_3d[b, :, m].tolist()    # secventa temporala entitate m
                entity_gt   = frame_labels_3d[b, :, m].tolist()
                all_preds.append(_frames_to_segments(entity_pred))
                all_gts.append(_frames_to_segments(entity_gt))

    return compute_metrics_epoch(all_preds, all_gts, iou_thresholds)


def main():
    args = parse_args()
    device = torch.device(args.device)

    cfg = OmegaConf.merge(
        OmegaConf.load("configs/base.yaml"),
        OmegaConf.load(args.config),
    )

    experiment_name = f"{cfg.dataset.name}_fold{args.fold}"

    # Fiecare fold primeste propriul subdirector pentru checkpoints si logs,
    # altfel fold-urile se suprascriu reciproc (best_model.pth, TensorBoard etc.)
    # Rezultat: checkpoints/mphoi72_fold0/, checkpoints/mphoi72_fold1/, ...
    base_checkpoint_dir = OmegaConf.select(cfg, "logging.checkpoint_dir", default="checkpoints/")
    base_log_dir = OmegaConf.select(cfg, "logging.log_dir", default="logs/")
    OmegaConf.update(cfg, "logging.checkpoint_dir", os.path.join(base_checkpoint_dir, experiment_name))
    OmegaConf.update(cfg, "logging.log_dir", os.path.join(base_log_dir, experiment_name))

    wandb_enabled_cfg = bool(OmegaConf.select(cfg, "logging.wandb_enabled", default=False))
    wandb_project_cfg = OmegaConf.select(cfg, "logging.wandb_project", default="vhoip")
    wandb_entity_cfg = OmegaConf.select(cfg, "logging.wandb_entity", default=None)
    wandb_run_name_cfg = OmegaConf.select(cfg, "logging.wandb_run_name", default=None)
    local_logging_enabled = bool(OmegaConf.select(cfg, "logging.local_logging_enabled", default=True))
    local_checkpoints_enabled = bool(OmegaConf.select(cfg, "logging.local_checkpoints_enabled", default=True))

    use_wandb = args.wandb or wandb_enabled_cfg
    wandb_project = args.wandb_project or wandb_project_cfg
    wandb_entity = args.wandb_entity or wandb_entity_cfg
    wandb_run_name = args.wandb_run_name or wandb_run_name_cfg
    if wandb_run_name:
        wandb_run_name = str(wandb_run_name).format(
            dataset=cfg.dataset.name,
            fold=args.fold,
            experiment=experiment_name,
        )

    logger = Logger(
        cfg.logging.log_dir,
        experiment_name,
        use_wandb=use_wandb,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        wandb_run_name=wandb_run_name,
        wandb_config=OmegaConf.to_container(cfg, resolve=True),
        wandb_group=cfg.dataset.name,
        wandb_job_type="fold",
        enable_local_logging=local_logging_enabled,
    )
    logger.info(f"Config: {args.config} | Device: {device} | Fold: {args.fold}")

    train_ds = get_dataset(cfg.dataset.name, root=cfg.dataset.root, split="train", fold=args.fold)
    val_ds = get_dataset(cfg.dataset.name, root=cfg.dataset.root, split="test", fold=args.fold)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.training.batch_size,
        shuffle=True, num_workers=cfg.data.num_workers, pin_memory=cfg.data.pin_memory,
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
    logger.info(f"Train: {len(train_ds)} video-uri | Val: {len(val_ds)} video-uri")

    label_names = get_label_names(cfg.dataset.name)
    model = VHOIP(cfg, label_names, device=str(device)).to(device)

    stats = model.count_parameters()
    logger.info(
        f"Parametri: total={stats['total']:,} | "
        f"antrenabili={stats['trainable']:,} | "
        f"frozen (CLIP)={stats['frozen']:,}"
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.training.epochs)
    criterion = VHOIPLoss(cfg.training.lambda1, cfg.training.lambda2, cfg.training.lambda3)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.training.use_amp and device.type == "cuda")

    start_epoch, best_fsum = 0, 0.0
    if args.resume:
        ckpt = load_checkpoint(args.resume, model, optimizer, str(device))
        start_epoch = ckpt["epoch"] + 1
        best_fsum = ckpt["metrics"].get("fsum", 0.0)
        logger.info(f"Resuming din epoch {start_epoch}, best FSUM={best_fsum:.1f}")
    else:
        logger.info("Initializez G din prototipuri CLIP vizuale (train set)...")
        initialize_global_representation(model, train_loader, device, logger)

    logger.info("Incep antrenarea...")
    for epoch in range(start_epoch, cfg.training.epochs):
        logger.info(f"\nEpoch {epoch + 1}/{cfg.training.epochs}")

        train_losses = train_one_epoch(
            model, train_loader, optimizer, criterion,
            scaler, device, logger, epoch, cfg,
        )
        logger.log_losses(train_losses, epoch)
        logger.info(f"  Train loss: {train_losses['total']:.4f}")
        scheduler.step()

        if (epoch + 1) % 5 == 0 or epoch == cfg.training.epochs - 1:
            metrics = evaluate(model, val_loader, device, cfg.evaluation.iou_thresholds)
            logger.log_metrics(metrics, epoch)

            is_best = metrics["fsum"] > best_fsum
            if is_best:
                best_fsum = metrics["fsum"]

            if (epoch + 1) % cfg.logging.save_every == 0 or is_best:
                saved = save_checkpoint(
                    model,
                    optimizer,
                    epoch,
                    metrics,
                    cfg.logging.checkpoint_dir,
                    is_best,
                    save_local=local_checkpoints_enabled,
                )

                # W&B-only mode: dump a temporary checkpoint file for artifact upload.
                temp_ckpt_path = None
                temp_best_path = None
                if not local_checkpoints_enabled and use_wandb:
                    fd, temp_ckpt_path = tempfile.mkstemp(prefix=f"{experiment_name}_ep{epoch:03d}_", suffix=".pth")
                    os.close(fd)
                    torch.save(saved["state"], temp_ckpt_path)

                    if is_best:
                        fd2, temp_best_path = tempfile.mkstemp(prefix=f"{experiment_name}_best_ep{epoch:03d}_", suffix=".pth")
                        os.close(fd2)
                        torch.save(saved["state"], temp_best_path)

                logger.log_checkpoint_artifact(
                    checkpoint_path=temp_ckpt_path or saved["checkpoint"],
                    best_checkpoint_path=temp_best_path or saved.get("best_checkpoint"),
                    epoch=epoch,
                    metrics=metrics,
                    is_best=is_best,
                )

                if temp_ckpt_path and os.path.exists(temp_ckpt_path):
                    os.remove(temp_ckpt_path)
                if temp_best_path and os.path.exists(temp_best_path):
                    os.remove(temp_best_path)

    logger.info(f"\nAntrenare finalizata. Best FSUM: {best_fsum:.1f}")
    logger.log_summary({"best_fsum": best_fsum, "fold": args.fold})
    logger.close()


if __name__ == "__main__":
    main()