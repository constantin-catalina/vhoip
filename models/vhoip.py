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

C6b improvements in this file:
  1. FIX: T is correctly populated before switching to inference mode via
     set_inference_mode(), so evaluation uses the real learned T, not zeros.
  2. Temperature scaling: cosine similarities are multiplied by
     text_encoder.temperature (learnable scalar, exp(log_temp)) before
     being passed to L_Cos — matches how CLIP trains its own contrastive head.
  3. Prompt regularization loss (L_PromptReg): cosine similarity between
     learned T and frozen T, averaged over classes and returned in the output
     dict so losses.py can include it in the total.
     lambda4 controls its weight (default 0.1 — light anchor, not dominant).
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
        self.use_learnable_prompts = getattr(cfg.model, "use_learnable_prompts", False)
        self.disable_geo_branch = getattr(cfg.model, "disable_geo_branch", False)

        # -----------------------------------------------------------------------
        # Backbone 2G-GCN (geometric GCN + fusion graph + BiRNN)
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
        # -----------------------------------------------------------------------
        self.mlp_proj = MLPProjection(
            input_dim=cfg.model.hidden_dim,
            output_dim=cfg.model.clip_dim,
            dropout=cfg.model.dropout,
        )

        # -----------------------------------------------------------------------
        # Discriminator MI  (Eq. 1 din paper)
        # -----------------------------------------------------------------------
        self.discriminator = Discriminator(
            feature_dim=cfg.model.hidden_dim,
            global_dim=cfg.model.clip_dim,
        )

        # -----------------------------------------------------------------------
        # CLIP Text Encoder (frozen CLIP, optional learnable prompts for C6b)
        # num_classes passed so LearnablePromptEncoder can allocate class offsets.
        # -----------------------------------------------------------------------
        self.text_encoder = CLIPTextEncoder(
            model_name=cfg.model.clip_model,
            device=device,
            use_learnable_prompts=self.use_learnable_prompts,
            n_ctx=getattr(cfg.model, "prompt_n_ctx", 16),
            ctx_init=getattr(cfg.model, "prompt_ctx_init", "a photo of a person"),
            num_classes=cfg.model.num_classes,
        )

        # -----------------------------------------------------------------------
        # Rappresentazione globale integrata G  (VHOIP §3.2, Alg. 1)
        # -----------------------------------------------------------------------
        self.global_rep = IntegratedGlobalRepresentation(
            num_classes=cfg.model.num_classes,
            feature_dim=cfg.model.clip_dim,
            rho=cfg.training.rho,
            warmup_epochs=cfg.training.warmup_epochs,
        )

        # Colector features Z' pentru EMA update la sfarsitul epoch-ului
        self.collector = FeaturesCollector()

        # Store label info
        self._label_names = label_names
        self._clip_subject = cfg.dataset.clip_subject
        self._clip_template = cfg.dataset.clip_template

        # -----------------------------------------------------------------------
        # Precompute frozen T — used for both baseline T and prompt reg loss.
        # This is correct for ALL modes (C6, C6b, baseline).
        # -----------------------------------------------------------------------
        frozen_T = self.text_encoder.precompute_frozen_T(
            label_names,
            subject=cfg.dataset.clip_subject,
            template=cfg.dataset.clip_template,
        )
        num_templates = (
            len(cfg.dataset.clip_template)
            if isinstance(cfg.dataset.clip_template, (list,))
            else 1
        )
        print(f"  Text features T precomputed: {frozen_T.shape} ({num_templates} template(s) ensembled)")

        if self.use_learnable_prompts:
            # T will be recomputed each forward pass — register a buffer as
            # placeholder so state_dict() includes it and inference works correctly
            # after set_inference_mode() is called (see FIX below).
            self.register_buffer("T", frozen_T.clone())
            print("  C6b: T buffer initialized from frozen templates; "
                  "will be updated dynamically during training and before inference.")
        else:
            self.register_buffer("T", frozen_T)

        # C6b prompt regularization weight
        self._lambda4 = getattr(cfg.training, "lambda4", 0.1)

        # Flag para modo de inferencia
        self._inference_mode = False

    # ---------------------------------------------------------------------------
    # Initializare
    # ---------------------------------------------------------------------------

    def initialize_G(self, clip_visual_features: torch.Tensor, labels: torch.Tensor) -> None:
        from models.clip_modules import Prototyping
        proto = Prototyping(self.num_classes, self.clip_dim)
        g_init = proto(
            clip_visual_features.to(self.device),
            labels.to(self.device),
        )
        self.global_rep.initialize(g_init)
        print(f"  G initializat cu prototipuri CLIP vizuale (shape: {g_init.shape})")

    def initialize_G_from_text(self) -> None:
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

        Returns at training:
            segment_logits, frame_logits, u_soft,
            mi_scores, cos_similarities, z_prime,
            prompt_reg_loss (scalar, 0.0 if not C6b)

        Returns at inference:
            segment_logits, frame_logits
        """
        # -----------------------------------------------------------------------
        # Pas 1: Backbone 2G-GCN + ASSIGN
        # -----------------------------------------------------------------------
        if self.disable_geo_branch:
            geo_features = None
        backbone_out = self.backbone(
            roi_features=roi_features,
            geo_features=geo_features,
            entity_types=entity_types,
            training_stage=training_stage,
        )
        z              = backbone_out["z"]
        frame_logits   = backbone_out["frame_logits"]
        segment_logits = backbone_out["segment_logits"]
        u_soft         = backbone_out["u_soft"]

        if self._inference_mode:
            return {
                "segment_logits": segment_logits,
                "frame_logits":   frame_logits,
            }

        # -----------------------------------------------------------------------
        # Pas 2: MLP Projection Z -> Z'
        # -----------------------------------------------------------------------
        z_prime = self.mlp_proj(z)   # (B, N, clip_dim), L2-normalizat

        # -----------------------------------------------------------------------
        # Pas 3: Discriminator MI  (Eq. 1)
        # -----------------------------------------------------------------------
        G = self.global_rep.get_G()
        mi_scores = self.discriminator(z, G)

        # -----------------------------------------------------------------------
        # Pas 4: Text features T + cosine similarity with temperature scaling
        # -----------------------------------------------------------------------
        if self.use_learnable_prompts:
            # C6b: recompute T — gradients flow through ctx and class offsets
            T = self.text_encoder.encode_labels(
                self._label_names,
                subject=self._clip_subject,
                # template intentionally NOT passed: learnable prompts replace templates
            )
            # Update the T buffer so it's available at inference time (FIX #4)
            with torch.no_grad():
                self.T.copy_(T.detach())
        else:
            T = self.T

        # Apply learnable temperature to cosine similarities.
        # Both z_prime and T are L2-normalised, so einsum = cosine similarity.
        # Multiplying by temperature sharpens the distribution (same as CLIP).
        cos_similarities = torch.einsum("bnd,cd->bnc", z_prime, T) * self.text_encoder.temperature

        # -----------------------------------------------------------------------
        # Pas 5: Prompt regularization loss (C6b only)
        # L_PromptReg = 1 - mean cosine similarity between learned T and frozen T.
        # This anchors the context vectors to the CLIP space they started from,
        # preventing catastrophic drift while still allowing adaptation.
        # -----------------------------------------------------------------------
        if self.use_learnable_prompts:
            frozen_T = self.text_encoder.get_frozen_T()
            if frozen_T is not None:
                # Both T and frozen_T are L2-normalised — dot product = cosine sim
                cos_sim_prompt = (T * frozen_T.to(T.device)).sum(dim=-1)  # (C,)
                prompt_reg_loss = (1.0 - cos_sim_prompt).mean()
            else:
                prompt_reg_loss = torch.tensor(0.0, device=z.device)
        else:
            prompt_reg_loss = torch.tensor(0.0, device=z.device)

        # -----------------------------------------------------------------------
        # Collect Z' for EMA update
        # -----------------------------------------------------------------------
        if labels is not None:
            self.collector.add(z_prime, labels)

        return {
            "segment_logits":   segment_logits,
            "frame_logits":     frame_logits,
            "u_soft":           u_soft,
            "mi_scores":        mi_scores,
            "cos_similarities": cos_similarities,
            "z_prime":          z_prime,
            "prompt_reg_loss":  prompt_reg_loss,   # new: used by VHOIPLoss
        }

    # ---------------------------------------------------------------------------
    # Utilitare training
    # ---------------------------------------------------------------------------

    def end_of_epoch(self, epoch: int) -> None:
        features, labels = self.collector.get_all()
        if features is not None:
            self.global_rep.update(
                features.to(self.device),
                labels.to(self.device),
                epoch,
            )
        self.collector.reset()

    def set_inference_mode(self, inference: bool = True) -> None:
        """
        Switch between training and inference mode.

        FIX: In C6b mode, we ensure self.T is populated with the current
        learned text features BEFORE switching to eval mode, so that any
        subsequent inference run uses the real learned T instead of zeros
        or stale values from a previous epoch.
        """
        if inference and self.use_learnable_prompts:
            # Recompute T one final time and freeze it into the buffer
            with torch.no_grad():
                T_final = self.text_encoder.encode_labels(
                    self._label_names,
                    subject=self._clip_subject,
                )
                self.T.copy_(T_final)

        self._inference_mode = inference
        if inference:
            self.eval()
        else:
            self.train()

    def get_trainable_params(self) -> tuple:
        trainable, frozen = [], []
        for name, param in self.named_parameters():
            if param.requires_grad:
                trainable.append(name)
            else:
                frozen.append(name)
        return trainable, frozen

    def count_parameters(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total": total,
            "trainable": trainable,
            "frozen": total - trainable,
        }