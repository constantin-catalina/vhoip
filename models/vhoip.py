"""
vhoip.py
Modelul principal VHOIP - asambleaza toate modulele.

Pipeline training (Fig. 2 din paper):
  1. Backbone extrage Z si logits din ROI features
  2. MLP proiecteaza Z -> Z' (normalizat)
  3. Discriminatorul calculeaza scorurile MI cu G
  4. Similaritatea cosinus intre Z' si T (text features)
  5. Loss-ul total combina toate cele 4 componente

Pipeline inference:
  - Identic cu 2G-GCN (fara costuri suplimentare)
  - Se folosesc doar segment_logits din backbone
"""

import torch
import torch.nn as nn

from omegaconf import DictConfig
from typing import Dict, Optional, List

from models.backbone import Backbone2GGCN
from models.clip_modules import (
    CLIPTextEncoder,
    MLPProjection,
    IntegratedGlobalRepresentation,
    FeaturesCollector,
)
from models.discriminator import Discriminator


class VHOIP(nn.Module):
    """
    Video-based Human-Object Interaction recognition with CLIP Prior knowledge.

    Args:
        cfg: configuratia completa (din YAML)
        label_names: lista de verbe/activitati pentru template-urile CLIP
        device: torch device
    """

    def __init__(
        self,
        cfg: DictConfig,
        label_names: List[str],
        device: str = "cuda",
    ):
        super().__init__()

        self.device = device
        self.num_classes = cfg.model.num_classes
        self.hidden_dim = cfg.model.hidden_dim
        self.clip_dim = cfg.model.clip_dim

        # --- Backbone 2G-GCN ---
        self.backbone = Backbone2GGCN(
            input_dim=cfg.data.roi_dim,
            hidden_dim=cfg.model.hidden_dim,
            num_classes=cfg.model.num_classes,
            num_layers=cfg.model.num_layers,
            dropout=cfg.model.dropout,
        )

        # --- MLP Projection Z -> Z' ---
        self.mlp_proj = MLPProjection(
            input_dim=cfg.model.hidden_dim,
            output_dim=cfg.model.clip_dim,
            dropout=cfg.model.dropout,
        )

        # --- Discriminator MI ---
        self.discriminator = Discriminator(
            feature_dim=cfg.model.hidden_dim,
            global_dim=cfg.model.clip_dim,
        )

        # --- CLIP Text Encoder (frozen) ---
        self.text_encoder = CLIPTextEncoder(
            model_name=cfg.model.clip_model,
            device=device,
        )

        # --- Reprezentarea globala integrata G ---
        self.global_rep = IntegratedGlobalRepresentation(
            num_classes=cfg.model.num_classes,
            feature_dim=cfg.model.clip_dim,
            rho=cfg.training.rho,
            warmup_epochs=cfg.training.warmup_epochs,
        )

        # --- Colector features pentru EMA update ---
        self.collector = FeaturesCollector()

        # Precomputa reprezentarile textuale T (o singura data)
        self.register_buffer(
            "T",
            self._compute_text_features(
                label_names,
                subject=cfg.dataset.clip_subject,
                template=cfg.dataset.clip_template,
            ),
        )

        # Flag pentru modul inference (nu mai calculeaza MI si Cos)
        self._inference_mode = False

    def _compute_text_features(
        self,
        label_names: List[str],
        subject: str,
        template: str,
    ) -> torch.Tensor:
        """Precomputa T la initializare (frozen, nu se recalculeaza)."""
        with torch.no_grad():
            T = self.text_encoder.encode_labels(
                label_names,
                subject=subject,
                template=template,
            )
        print(f"  Text features T precomputed: {T.shape}")
        return T

    def initialize_G_from_text(self) -> None:
        """
        Initializeaza G din text features T (CLIP text encoder).
        Fallback cand features CLIP vizuale nu sunt disponibile sau sunt invalide.
        T este deja in spatiul CLIP 512-dim, corect normalizat L2.
        """
        self.global_rep.initialize(self.T.clone())
        print(f"  G initializat din text features T (shape: {self.T.shape})")

    def initialize_G(self, clip_visual_features: torch.Tensor, labels: torch.Tensor) -> None:
        """
        Initializeaza G cu prototipurile CLIP (G_init).
        Se apeleaza o singura data inainte de antrenare,
        dupa ce am extras features CLIP pentru tot training set-ul.

        Args:
            clip_visual_features: (N_total, clip_dim) - features CLIP vizuale
            labels: (N_total,) - etichete corespunzatoare
        """
        from models.clip_modules import Prototyping
        proto = Prototyping(self.num_classes, self.clip_dim)
        g_init = proto(clip_visual_features.to(self.device), labels.to(self.device))
        self.global_rep.initialize(g_init)

    def forward(
        self,
        roi_features: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass VHOIP.

        Args:
            roi_features:  (B, S, M, 2048) - ROI pooling features din Faster R-CNN
            clip_features: (B, S, M, 512)  - features CLIP vizuale (optional, pentru G_init)
            labels:        (B, N,)          - etichete (optional, pentru collector)
            adj:           (B, M, M)        - matrice adiacenta (optional)

        Returns:
            dict cu toate output-urile necesare pentru loss si evaluare
        """
        # --- Pas 1: Backbone ---
        backbone_out = self.backbone(roi_features)
        z = backbone_out["z"]                          # (B, N, hidden_dim)
        frame_logits = backbone_out["frame_logits"]    # (B, N, C)
        segment_logits = backbone_out["segment_logits"]  # (B, N, C)

        if self._inference_mode:
            # La inferenta returnam doar ce e necesar
            return {
                "segment_logits": segment_logits,
                "frame_logits": frame_logits,
            }

        # --- Pas 2: Proiectie Z -> Z' ---
        z_prime = self.mlp_proj(z)   # (B, N, clip_dim) - normalizat L2

        # --- Pas 3: Discriminator MI cu G curent ---
        G = self.global_rep.get_G()  # (C, clip_dim)
        mi_scores = self.discriminator(z, G)  # (B, N, C)

        # --- Pas 4: Similaritate cosinus Z' vs T ---
        # T: (C, clip_dim), z_prime: (B, N, clip_dim)
        # -> cos_sim: (B, N, C)
        cos_similarities = torch.einsum("bnd,cd->bnc", z_prime, self.T)

        # --- Colecteaza Z' pentru EMA update la sfarsitul epoch-ului ---
        if labels is not None:
            self.collector.add(z_prime, labels)

        return {
            "segment_logits": segment_logits,   # (B, N, C)
            "frame_logits": frame_logits,        # (B, N, C)
            "mi_scores": mi_scores,              # (B, N, C)
            "cos_similarities": cos_similarities,  # (B, N, C)
            "z_prime": z_prime,                  # (B, N, clip_dim)
        }

    def end_of_epoch(self, epoch: int) -> None:
        """
        Apelat la sfarsitul fiecarui epoch de antrenare.
        Actualizeaza G cu EMA si reseteaza colectorul.
        """
        features, labels = self.collector.get_all()
        if features is not None:
            self.global_rep.update(
                features.to(self.device),
                labels.to(self.device),
                epoch,
            )
        self.collector.reset()

    def set_inference_mode(self, inference: bool = True) -> None:
        """Comuta intre training si inference mode."""
        self._inference_mode = inference
        if inference:
            self.eval()
        else:
            self.train()

    def get_trainable_params(self):
        """
        Returneaza doar parametrii antrenabili (exclude CLIP frozen).
        Util pentru a verifica ce se antreneaza si pentru optimizer.
        """
        trainable = []
        frozen = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                trainable.append(name)
            else:
                frozen.append(name)
        return trainable, frozen

    def count_parameters(self) -> Dict[str, int]:
        """Numara parametrii modelului."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total": total,
            "trainable": trainable,
            "frozen": total - trainable,
        }