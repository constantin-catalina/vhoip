"""
vhoip.py
Modelul principal VHOIP - asambleaza toate modulele.

Pipeline training (Fig. 2 din VHOIP paper):
  1. Backbone (2G-GCN) extrage Z (primul nivel BiRNN) si logits din ROI features
     + features geometrice (skeleton + bbox keypoints)
  2. MLP proiecteaza Z -> Z' (L2 normalizat, dim=512)
  3. Discriminatorul calculeaza scorurile MI intre Z si G (Eq. 1)
  4. Similaritatea cosinus intre Z' si T (reprezentarile textuale CLIP) (Eq. 3)
  5. Loss-ul total: L = L_Label + λ1*L_Seg + λ2*L_MI + λ3*L_Cos (Eq. 4)

Pipeline inference:
  - Identic cu 2G-GCN (fara costuri suplimentare)
  - Se folosesc doar segment_logits din backbone
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

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

    Combina backbone-ul 2G-GCN cu cunostintele prior ale CLIP pentru a
    imbunatati recunoasterea HOI fine-grained in video.

    Args:
        cfg:         configuratia completa (din YAML, merge base + dataset)
        label_names: lista de verbe/activitati pentru template-urile CLIP
                     (ex. ["approach", "lift", "pour", ...] pentru MPHOI-72)
        device:      torch device ("cuda" sau "cpu")
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

        # -----------------------------------------------------------------------
        # Backbone 2G-GCN (geometric GCN + fusion graph + BiRNN)
        # Parametrii geo_input_dim, C1, C2 pot fi configurati in YAML.
        # Valorile default C1=64, C2=128 sunt din paper.
        # -----------------------------------------------------------------------
        self.backbone = Backbone2GGCN(
            input_dim=cfg.data.roi_dim,
            hidden_dim=cfg.model.hidden_dim,
            num_classes=cfg.model.num_classes,
            num_layers=cfg.model.num_layers,
            dropout=cfg.model.dropout,
            geo_input_dim=getattr(cfg.data, "geo_input_dim", 4),
            C1=getattr(cfg.model, "geo_C1", 64),
            C2=getattr(cfg.model, "geo_C2", 128),
        )

        # -----------------------------------------------------------------------
        # MLP Projection Z -> Z'  (VHOIP §3.2)
        # Two-layer MLP cu GELU, output L2-normalizat
        # -----------------------------------------------------------------------
        self.mlp_proj = MLPProjection(
            input_dim=cfg.model.hidden_dim,
            output_dim=cfg.model.clip_dim,
            dropout=cfg.model.dropout,
        )

        # -----------------------------------------------------------------------
        # Discriminator MI  (Eq. 1 din paper)
        # Primeste Z (hidden_dim) si G (clip_dim) si produce scoruri binare
        # -----------------------------------------------------------------------
        self.discriminator = Discriminator(
            feature_dim=cfg.model.hidden_dim,
            global_dim=cfg.model.clip_dim,
        )

        # -----------------------------------------------------------------------
        # CLIP Text Encoder (frozen)  (VHOIP §3.4)
        # -----------------------------------------------------------------------
        self.text_encoder = CLIPTextEncoder(
            model_name=cfg.model.clip_model,
            device=device,
        )

        # -----------------------------------------------------------------------
        # Reprezentarea globala integrata G  (VHOIP §3.2, Alg. 1)
        # G = rho*G + (1-rho)*V  dupa warm-up
        # -----------------------------------------------------------------------
        self.global_rep = IntegratedGlobalRepresentation(
            num_classes=cfg.model.num_classes,
            feature_dim=cfg.model.clip_dim,
            rho=cfg.training.rho,
            warmup_epochs=cfg.training.warmup_epochs,
        )

        # Colector features Z' pentru EMA update la sfarsitul epoch-ului
        self.collector = FeaturesCollector()

        # Precomputa reprezentarile textuale T (o singura data, frozen)
        self.register_buffer(
            "T",
            self._compute_text_features(
                label_names,
                subject=cfg.dataset.clip_subject,
                template=cfg.dataset.clip_template,
            ),
        )

        # Flag pentru modul inference
        self._inference_mode = False

    # ---------------------------------------------------------------------------
    # Initializare
    # ---------------------------------------------------------------------------

    def _compute_text_features(
        self,
        label_names: List[str],
        subject: str,
        template: str,
    ) -> torch.Tensor:
        """Precomputa T la initializare (frozen, nu se recalculeaza in training)."""
        with torch.no_grad():
            T = self.text_encoder.encode_labels(
                label_names,
                subject=subject,
                template=template,
            )
        print(f"  Text features T precomputed: {T.shape}")
        return T

    def initialize_G(self, clip_visual_features: torch.Tensor, labels: torch.Tensor) -> None:
        """
        Initializeaza G cu prototipurile CLIP vizuale (G_init, stanga Fig. 2).

        Se apeleaza O SINGURA DATA inainte de antrenare, dupa ce features
        CLIP vizuale au fost extrase pentru intregul training set.

        Args:
            clip_visual_features: (N_total, clip_dim) — features CLIP vizuale
            labels:               (N_total,) — etichete corespunzatoare
        """
        from models.clip_modules import Prototyping
        proto = Prototyping(self.num_classes, self.clip_dim)
        g_init = proto(
            clip_visual_features.to(self.device),
            labels.to(self.device),
        )
        self.global_rep.initialize(g_init)
        print(f"  G initializat cu prototipuri CLIP vizuale (shape: {g_init.shape})")

    def initialize_G_from_text(self) -> None:
        """
        Fallback: initializeaza G din text features T.

        Folosit cand features CLIP vizuale nu sunt disponibile.
        T este deja in spatiul CLIP 512-dim, L2-normalizat.
        """
        self.global_rep.initialize(self.T.clone())
        print(f"  G initializat din text features T (shape: {self.T.shape})")

    # ---------------------------------------------------------------------------
    # Forward pass
    # ---------------------------------------------------------------------------

    def forward(
        self,
        roi_features: torch.Tensor,
        geo_features: Optional[torch.Tensor] = None,
        entity_types: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        training_stage: int = 2,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass VHOIP.

        Args:
            roi_features:   (B, S, M, 2048) — ROI pooling features din Faster R-CNN
            geo_features:   (B, S, J, 4)    — keypoints geometrice (optional)
            entity_types:   (B, M)          — 0=human, 1=object (optional)
            labels:         (B, N)          — etichete ground-truth (optional)
            training_stage: 1 = ASSIGN Stage 1 (dense, L_Seg off)
                            2 = ASSIGN Stage 2 (full Gumbel-Softmax, default)
                            Ignorat la inferenta.

        Returns:
            La training:
                dict cu: segment_logits, frame_logits, u_soft,
                         mi_scores, cos_similarities, z_prime
            La inferenta (inference_mode=True):
                dict cu: segment_logits, frame_logits
        """
        # -----------------------------------------------------------------------
        # Pas 1: Backbone 2G-GCN + ASSIGN
        # Extrage Z (primul nivel BiRNN), frame/segment logits, u_soft
        # -----------------------------------------------------------------------
        backbone_out = self.backbone(
            roi_features=roi_features,
            geo_features=geo_features,
            entity_types=entity_types,
            training_stage=training_stage,
        )
        z              = backbone_out["z"]              # (B, N, hidden_dim)
        frame_logits   = backbone_out["frame_logits"]   # (B, N, C)
        segment_logits = backbone_out["segment_logits"] # (B, N, C)
        u_soft         = backbone_out["u_soft"]         # (B, N)

        # La inferenta, returnam doar output-ul backbone-ului
        if self._inference_mode:
            return {
                "segment_logits": segment_logits,
                "frame_logits":   frame_logits,
            }

        # -----------------------------------------------------------------------
        # Pas 2: MLP Projection Z -> Z'  (VHOIP §3.2)
        # Z vine din PRIMUL nivel BiRNN (frame-level), L2-normalizat
        # -----------------------------------------------------------------------
        z_prime = self.mlp_proj(z)   # (B, N, clip_dim), L2-normalizat

        # -----------------------------------------------------------------------
        # Pas 3: Discriminator MI  (Eq. 1)
        # z: (B, N, hidden_dim) — L2-normalizat in interiorul Discriminator.forward()
        # G: (C, clip_dim)
        # mi_scores: (B, N, C) — logit-uri brute (sigmoid in BCEWithLogitsLoss)
        # -----------------------------------------------------------------------
        G = self.global_rep.get_G()              # (C, clip_dim)
        mi_scores = self.discriminator(z, G)     # (B, N, C)

        # -----------------------------------------------------------------------
        # Pas 4: Similaritate cosinus Z' vs T  (Eq. 3)
        # Ambii sunt L2-normalizati => dot product = cosine similarity
        # T: (C, clip_dim), z_prime: (B, N, clip_dim)
        # cos_similarities: (B, N, C)
        # -----------------------------------------------------------------------
        cos_similarities = torch.einsum("bnd,cd->bnc", z_prime, self.T)

        # -----------------------------------------------------------------------
        # Colecteaza Z' pentru EMA update la sfarsitul epoch-ului (Alg. 1)
        # Filtram NaN/Inf inainte de a adauga in colector pentru a evita
        # poluarea prototipurilor V cu features corupte de la gradient explosion.
        # -----------------------------------------------------------------------
        if labels is not None:
            if not (torch.isnan(z_prime).any() or torch.isinf(z_prime).any()):
                self.collector.add(z_prime, labels)

        return {
            "segment_logits":   segment_logits,    # (B, N, C) — pentru L_Label
            "frame_logits":     frame_logits,      # (B, N, C) — pentru L_Seg
            "u_soft":           u_soft,            # (B, N)    — pentru L_Seg (ASSIGN Eq. 11)
            "mi_scores":        mi_scores,         # (B, N, C) — pentru L_MI
            "cos_similarities": cos_similarities,  # (B, N, C) — pentru L_Cos
            "z_prime":          z_prime,           # (B, N, clip_dim)
        }

    # ---------------------------------------------------------------------------
    # Utilitare training
    # ---------------------------------------------------------------------------

    def end_of_epoch(self, epoch: int) -> None:
        """
        Apelat la sfarsitul fiecarui epoch de antrenare.
        Calculeaza prototipurile V si actualizeaza G cu EMA (Alg. 1).
        Reseteaza colectorul pentru epoch-ul urmator.
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
        """Comuta intre modul training si modul inferenta."""
        self._inference_mode = inference
        if inference:
            self.eval()
        else:
            self.train()

    def get_trainable_params(self) -> tuple:
        """
        Returneaza (lista_antrenabili, lista_frozen).
        Util pentru debug si pentru configurarea optimizer-ului.
        CLIP (text_encoder) trebuie sa fie mereu frozen.
        """
        trainable, frozen = [], []
        for name, param in self.named_parameters():
            if param.requires_grad:
                trainable.append(name)
            else:
                frozen.append(name)
        return trainable, frozen

    def count_parameters(self) -> Dict[str, int]:
        """Numara parametrii modelului (total, antrenabili, frozen)."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total": total,
            "trainable": trainable,
            "frozen": total - trainable,
        }