"""
train.py
Entry point pentru antrenarea VHOIP.

Utilizare:
    python train.py --config configs/cad120.yaml
    python train.py --config configs/cad120.yaml --fold 0
    python train.py --config configs/mphoi72.yaml --fold 0 --resume checkpoints/epoch_010.pth
    python train.py --config configs/mphoi72.yaml --fold 0 --override "training.seg_sigma=4.0" --override "training.lambda2=0.5"
"""

import argparse
import os
import random
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from data.dataset import get_dataset, CAD120Dataset, MPHOI72Dataset, BimanualDataset
from data.mphoi72_dataset import MPHOI72ZarrDataset, collate_fn
from models.vhoip import VHOIP
from models.losses import VHOIPLoss
from utils.logger import Logger
from utils.metrics import compute_metrics_epoch
from utils.checkpoint import save_checkpoint, load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Antrenare VHOIP")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42, help="Random seed pentru reproductibilitate")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--wandb", action="store_true", help="Activeaza logging in Weights & Biases")
    parser.add_argument("--wandb_project", type=str, default=None, help="Numele proiectului W&B")
    parser.add_argument("--wandb_entity", type=str, default=None, help="Entity/username W&B")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="Nume custom pentru run-ul W&B")
    parser.add_argument("--experiment_name", type=str, default=None, help="Nume experiment (subdirector checkpoints si W&B run name)")
    parser.add_argument(
        "--override",
        action="append",
        default=None,
        help="Override config value (format: key=value, e.g., training.seg_sigma=4.0)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def set_seed(seed: int):
    """Seteaza seed-ul pentru toate librariile de randomizare."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def worker_init_fn(worker_id: int):
    """Initializer pentru workerii DataLoader cu seed propriu."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_label_names(dataset_name: str):
    mapping = {
        "cad120": CAD120Dataset.ACTIVITY_LABELS,
        "mphoi72": MPHOI72Dataset.ACTIVITY_LABELS,
        "bimanual": BimanualDataset.ACTIVITY_LABELS,
    }
    return mapping[dataset_name]


def generate_run_id() -> str:
    """Genereaza un ID unic pentru rulare (folosit pentru izolarea checkpoints/logs)."""
    try:
        import wandb
        return wandb.util.generate_id()
    except Exception:
        import uuid
        return uuid.uuid4().hex[:8]


def train_one_epoch(model, dataloader, optimizer, criterion, scaler, device, logger, epoch, cfg):
    model.train()
    model.set_inference_mode(False)
    total_losses = {}
    stage1_epochs = cfg.training.get("stage1_epochs", 10)
    current_stage = 1 if epoch < stage1_epochs else 2

    # Ramp L_Seg weight linearly from lambda1_start to lambda1_final
    # over stage2 epochs, to avoid the sudden loss spike at transition
    if current_stage == 1:
        lambda1 = 0.0
    else:
        lambda1_start = cfg.training.get("lambda1_start", 0.1)
        lambda1_final = cfg.training.get("lambda1_final", cfg.training.lambda1)
        ramp_epochs   = cfg.training.get("stage2_epochs", 40)
        stage2_epoch  = epoch - stage1_epochs          # 0-indexed within stage 2
        t             = min(stage2_epoch / ramp_epochs, 1.0)
        lambda1       = lambda1_start + t * (lambda1_final - lambda1_start)

    # Update criterion's lambda1 dynamically
    criterion.lambda1 = lambda1

    for batch_idx, batch in enumerate(dataloader):
        roi = batch["roi_features"].to(device)
        geo = batch["geo_features"].to(device)
        entity_types = batch["entity_types"].to(device)
        seg_labels = batch["seg_labels"].to(device)
        frame_labels = batch["frame_labels"].to(device)

        optimizer.zero_grad()

        # Build anticipation labels: segment_labels shifted one step forward in time.
        # ant_labels[b, t] = seg_labels[b, t+1]; last position is set to -1 (ignored).
        ant_labels = seg_labels.roll(-1, dims=1)
        ant_labels[:, -1] = -1

        with torch.amp.autocast(
            device_type=device.type,
            enabled=cfg.training.use_amp and device.type == "cuda",
        ):
            outputs = model(
                roi_features=roi,
                geo_features=geo,
                entity_types=entity_types,
                labels=seg_labels,
                training_stage=current_stage,
            )
            losses = criterion(
                segment_logits=outputs["segment_logits"],
                frame_logits=outputs["frame_logits"],
                u_soft=outputs["u_soft"],
                mi_scores=outputs["mi_scores"],
                cos_similarities=outputs["cos_similarities"],
                segment_labels=seg_labels,
                frame_labels=frame_labels,
                anticipation_labels=ant_labels,
                training_stage=current_stage,
                prompt_reg_loss=outputs.get("prompt_reg_loss"),
            )

        if torch.isnan(losses["total"]) or torch.isinf(losses["total"]):
            logger.info(
                f"  [WARN] NaN/Inf loss at batch {batch_idx} — skipping step."
            )
            optimizer.zero_grad()
            continue

        scaler.scale(losses["total"]).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=cfg.training.grad_clip
        )
        scaler.step(optimizer)
        scaler.update()

        for k, v in losses.items():
            total_losses[k] = total_losses.get(k, 0) + v.item()

        if batch_idx % cfg.logging.log_every == 0:
            logger.info(
                f"  [{batch_idx}/{len(dataloader)}] "
                f"Loss={losses['total'].item():.4f} "
                f"Ant={losses['l_ant'].item():.4f} "
                f"MI={losses['l_mi'].item():.4f} "
                f"Cos={losses['l_cos'].item():.4f} "
                f"GradNorm={grad_norm:.2f}"
            )

    model.end_of_epoch(epoch)
    return {k: v / len(dataloader) for k, v in total_losses.items()}


@torch.no_grad()
def initialize_global_representation(model, dataloader, device, logger):
    """
    Initializeaza G din prototipurile CLIP vizuale pe setul de train.

    IMPORTANT: clip_features trebuie sa fie extrase real cu CLIP din crop-uri
    de imagini brute inainte de antrenare (via extract_clip_features_from_videos()
    din data/mphoi72_dataset.py). Daca fisierele _clip.npy lipsesc sau contin
    zerouri, fallback-ul la text-based G_init este automat.
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
            f"  Pentru CLIP vizual real, ruleaza extract_clip_features_from_videos() din data/mphoi72_dataset.py."
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
        geo = batch["geo_features"].to(device)
        entity_types = batch["entity_types"].to(device)
        seg_labels = batch["seg_labels"].to(device)
        frame_labels = batch["frame_labels"]      # (B, S*M) frame-major
        B, S, M, _ = roi.shape

        outputs = model(
            roi_features=roi,
            geo_features=geo,
            entity_types=entity_types,
            labels=seg_labels,
        )
        frame_pred = outputs["segment_logits"].argmax(dim=-1).cpu()  # (B, S*M)

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


def _parse_override_value(v: str):
    """Parseaaza valoarea string in int, float, bool, None sau string."""
    try:
        return int(v)
    except ValueError:
        try:
            return float(v)
        except ValueError:
            low = v.lower()
            if low == "true":
                return True
            elif low == "false":
                return False
            elif low in ("none", "null"):
                return None
            return v


def apply_overrides(cfg, overrides: list):
    """Aplica override-uri CLI pe config OmegaConf."""
    if not overrides:
        return
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Override invalid (lipseste '='): {override}")
        key, value_str = override.split("=", 1)
        value = _parse_override_value(value_str)
        OmegaConf.update(cfg, key, value)
        print(f"  [override] {key} = {value}")


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    cfg = OmegaConf.merge(
        OmegaConf.load("configs/base.yaml"),
        OmegaConf.load(args.config),
    )

    if args.override:
        print("Aplic override-uri CLI:")
        apply_overrides(cfg, args.override)

    experiment_name = f"{cfg.dataset.name}_fold{args.fold}"

    run_id = generate_run_id()
    exp_subdir = args.experiment_name or run_id

    base_checkpoint_dir = OmegaConf.select(cfg, "logging.checkpoint_dir", default="checkpoints/")
    base_log_dir = OmegaConf.select(cfg, "logging.log_dir", default="logs/")
    OmegaConf.update(cfg, "logging.checkpoint_dir", os.path.join(base_checkpoint_dir, experiment_name, exp_subdir))
    OmegaConf.update(cfg, "logging.log_dir", os.path.join(base_log_dir, experiment_name, exp_subdir))
    print(f"Run ID: {run_id}")
    print(f"  Experiment:  {exp_subdir}")
    print(f"  Checkpoints: {cfg.logging.checkpoint_dir}")
    print(f"  Logs:        {cfg.logging.log_dir}")

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
            seed=args.seed,
            experiment_name=args.experiment_name or "default",
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
        wandb_id=run_id,
        enable_local_logging=local_logging_enabled,
    )
    logger.info(f"Config: {args.config} | Device: {device} | Fold: {args.fold}")

    if cfg.dataset.name == "mphoi72":
        train_ds = MPHOI72ZarrDataset(cfg.dataset.root, split="train", fold=args.fold)
        val_ds   = MPHOI72ZarrDataset(cfg.dataset.root, split="test",  fold=args.fold)
    else:
        train_ds = get_dataset(cfg.dataset.name, root=cfg.dataset.root, split="train", fold=args.fold)
        val_ds   = get_dataset(cfg.dataset.name, root=cfg.dataset.root, split="test",  fold=args.fold)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.training.batch_size,
        shuffle=True, num_workers=cfg.data.num_workers, pin_memory=cfg.data.pin_memory,
        collate_fn=collate_fn,
        worker_init_fn=worker_init_fn,
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate_fn)
    logger.info(f"Train: {len(train_ds)} video-uri | Val: {len(val_ds)} video-uri")

    label_names = get_label_names(cfg.dataset.name)
    model = VHOIP(cfg, label_names, device=str(device)).to(device)

    stats = model.count_parameters()
    logger.info(
        f"Parametri: total={stats['total']:,} | "
        f"antrenabili={stats['trainable']:,} | "
        f"frozen (CLIP)={stats['frozen']:,}"
    )

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.training.learning_rate,
    )
    if cfg.training.get("scheduler", "cosine") == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", patience=cfg.training.get("scheduler_patience", 5),
            factor=cfg.training.get("scheduler_factor", 0.5), verbose=True,
        )
        use_plateau = True
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.training.epochs)
        use_plateau = False
    criterion = VHOIPLoss(
        cfg.training.lambda1,
        cfg.training.lambda2,
        cfg.training.lambda3,
        lambda_ant=cfg.training.get("lambda_ant", 1.0),
        seg_sigma=cfg.training.get("seg_sigma", 2.0),
        seg_pos_weight=cfg.training.get("seg_pos_weight", 5.0),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.training.use_amp and device.type == "cuda")

    start_epoch, best_fsum = 0, 0.0
    if args.resume:
        ckpt = load_checkpoint(args.resume, model, optimizer, str(device))
        start_epoch = ckpt["epoch"] + 1
        best_fsum = ckpt["metrics"].get("fsum", 0.0)
        logger.info(f"Resuming din epoch {start_epoch}, best FSUM={best_fsum:.1f}")
        # G si _initialized_flag sunt salvate/restaurate automat via state_dict().
        # _check_initialized() in update() sincronizeaza flag-ul cu G pentru
        # backward compatibility cu checkpoint-uri vechi.
        logger.info("G si flag-ul de initializare restaurate din checkpoint.")

        # Reset scheduler so Stage 2 gets its own full cycle
        stage2_epochs = cfg.training.epochs - start_epoch
        if use_plateau:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", patience=cfg.training.get("scheduler_patience", 5),
                factor=cfg.training.get("scheduler_factor", 0.5), verbose=True,
            )
            logger.info("Scheduler resetat: ReduceLROnPlateau.")
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=stage2_epochs
            )
            logger.info(f"Scheduler resetat: CosineAnnealingLR cu T_max={stage2_epochs} epoci.")
    else:
        logger.info("Initializez G din prototipuri CLIP vizuale (train set)...")
        initialize_global_representation(model, train_loader, device, logger)

    logger.info("Incep antrenarea...")
    stage1_epochs = cfg.training.get("stage1_epochs", 5)
    for epoch in range(start_epoch, cfg.training.epochs):
        # GSM temperature annealing: 1.0 -> 0.7 over stage 2
        if epoch >= stage1_epochs:
            t = min((epoch - stage1_epochs) / max(cfg.training.epochs - stage1_epochs, 1), 1.0)
            temp = 1.0 - 0.3 * t
            model.backbone.set_gsm_temperature(temp)

        logger.info(
            f"\nEpoch {epoch + 1}/{cfg.training.epochs}  "
            f"(GSM temp={model.backbone.boundary_detector.temperature:.3f})"
        )

        train_losses = train_one_epoch(
            model, train_loader, optimizer, criterion,
            scaler, device, logger, epoch, cfg,
        )
        logger.log_losses(train_losses, epoch + 1)
        logger.info(f"  Train loss: {train_losses['total']:.4f}")
        if not use_plateau:
            scheduler.step()

        metrics = evaluate(model, val_loader, device, cfg.evaluation.iou_thresholds)
        logger.log_metrics(metrics, epoch + 1)

        if use_plateau:
            scheduler.step(metrics["fsum"])

        # Save last checkpoint every epoch
        save_checkpoint(
            model,
            optimizer,
            epoch,
            metrics,
            cfg.logging.checkpoint_dir,
            is_best=False,
            save_last=True,
            save_local=local_checkpoints_enabled,
        )

        is_best = metrics["fsum"] > best_fsum
        if is_best:
            best_fsum = metrics["fsum"]
            saved = save_checkpoint(
                model,
                optimizer,
                epoch,
                metrics,
                cfg.logging.checkpoint_dir,
                is_best=True,
                save_local=local_checkpoints_enabled,
            )
            # Log checkpoint ca artifact W&B (izolat per run_id)
            if saved.get("best_checkpoint"):
                logger.log_checkpoint_artifact(
                    checkpoint_path=None,
                    epoch=epoch,
                    metrics=metrics,
                    is_best=True,
                    best_checkpoint_path=saved["best_checkpoint"],
                )


    logger.info(f"\nAntrenare finalizata. Best FSUM: {best_fsum:.1f}")
    logger.log_summary({"best_fsum": best_fsum, "fold": args.fold})
    logger.close()

    # Elibereaza memoria GPU intre fold-uri (util cand rulezi toate fold-urile secvential)
    if args.device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
        import gc
        gc.collect()


if __name__ == "__main__":
    main()