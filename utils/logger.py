"""
logger.py
Logging in terminal si TensorBoard.
"""

import os
import logging
from typing import Optional
from torch.utils.tensorboard import SummaryWriter

try:
    import wandb
except ImportError:
    wandb = None


class Logger:
    def __init__(
        self,
        log_dir: str,
        experiment_name: str,
        use_wandb: bool = False,
        wandb_project: str = "vhoip",
        wandb_entity: str = None,
        wandb_run_name: str = None,
        wandb_config: dict = None,
        wandb_group: str = None,
        wandb_job_type: str = "fold",
        wandb_id: str = None,
        enable_local_logging: bool = True,
    ):
        self.enable_local_logging = enable_local_logging
        self.experiment_name = experiment_name

        self.writer = None
        handlers = [logging.StreamHandler()]

        if self.enable_local_logging:
            os.makedirs(log_dir, exist_ok=True)
            self.writer = SummaryWriter(os.path.join(log_dir, experiment_name))
            handlers.append(logging.FileHandler(os.path.join(log_dir, f"{experiment_name}.log")))

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(message)s",
            datefmt="%H:%M:%S",
            handlers=handlers,
        )
        self.log = logging.getLogger(__name__)

        self.wandb_run = None
        if use_wandb:
            if wandb is None:
                self.log.warning("W&B activat, dar pachetul wandb nu este instalat. Continui fara W&B.")
            else:
                init_kwargs = dict(
                    project=wandb_project,
                    entity=wandb_entity,
                    name=wandb_run_name or experiment_name,
                    config=wandb_config,
                    group=wandb_group,
                    job_type=wandb_job_type,
                )
                if wandb_id is not None:
                    init_kwargs["id"] = wandb_id
                    init_kwargs["resume"] = "allow"
                self.wandb_run = wandb.init(**init_kwargs)
                self.log.info("W&B initializat cu succes")

    def log_losses(self, losses: dict, step: int) -> None:
        if self.writer is not None:
            for name, val in losses.items():
                self.writer.add_scalar(f"Loss/{name}", val, step)

        if self.wandb_run is not None:
            log_dict = {f"loss/{name}": float(val) for name, val in losses.items()}
            self.wandb_run.log(log_dict, step=step)

    def log_metrics(self, metrics: dict, epoch: int, split: str = "val") -> None:
        if self.writer is not None:
            for name, val in metrics.items():
                if not name.endswith("_std"):
                    self.writer.add_scalar(f"Metrics_{split}/{name}", val, epoch)

        if self.wandb_run is not None:
            log_dict = {
                f"metrics/{split}/{name}": float(val)
                for name, val in metrics.items()
                if not name.endswith("_std")
            }
            self.wandb_run.log(log_dict, step=epoch)

        self.log.info(
            f"Epoch {epoch} [{split}] | "
            f"F1@10={metrics.get('f1_10', 0):.1f} "
            f"F1@25={metrics.get('f1_25', 0):.1f} "
            f"F1@50={metrics.get('f1_50', 0):.1f} "
            f"FSUM={metrics.get('fsum', 0):.1f}"
        )

    def info(self, msg: str) -> None:
        self.log.info(msg)

    def log_checkpoint_artifact(
        self,
        checkpoint_path: Optional[str],
        epoch: int,
        metrics: dict,
        is_best: bool = False,
        best_checkpoint_path: Optional[str] = None,
    ) -> None:
        if self.wandb_run is None or wandb is None:
            return

        # Cel putin unul dintre checkpoint-uri trebuie sa existe.
        has_checkpoint = checkpoint_path is not None and os.path.exists(checkpoint_path)
        has_best = best_checkpoint_path is not None and os.path.exists(best_checkpoint_path)
        if not has_checkpoint and not has_best:
            return

        # Include wandb run ID in artifact name so different runs on the same fold
        # do not overwrite each other's checkpoints.
        run_id = self.wandb_run.id if self.wandb_run else "local"
        artifact = wandb.Artifact(
            name=f"{self.experiment_name}-{run_id}-epoch-{epoch:03d}",
            type="model",
            metadata={
                "epoch": epoch,
                "fsum": metrics.get("fsum", 0.0),
                "f1_10": metrics.get("f1_10", 0.0),
                "f1_25": metrics.get("f1_25", 0.0),
                "f1_50": metrics.get("f1_50", 0.0),
            },
        )
        if has_checkpoint:
            artifact.add_file(checkpoint_path, name=os.path.basename(checkpoint_path))

        if has_best:
            artifact.add_file(best_checkpoint_path, name=os.path.basename(best_checkpoint_path))

        aliases = ["latest", f"epoch-{epoch:03d}"]
        if is_best:
            aliases.append("best")

        self.wandb_run.log_artifact(artifact, aliases=aliases)

    def log_summary(self, metrics: dict) -> None:
        """Salveaza metricile finale in wandb run summary (vizibile in runs table)."""
        if self.wandb_run is not None:
            for k, v in metrics.items():
                self.wandb_run.summary[k] = v

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
        if self.wandb_run is not None:
            self.wandb_run.finish()