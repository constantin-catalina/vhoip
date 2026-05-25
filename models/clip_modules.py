"""
clip_modules.py
Module CLIP din VHOIP:
  - CLIPVisualEncoder   : extrage features vizuale din regiunile de interes
  - CLIPTextEncoder     : transforma template-uri HOI in reprezentari textuale T
  - Prototyping         : calculeaza prototipuri per clasa (G_init si V)
  - IntegratedGlobalRep : EMA update G = rho*G + (1-rho)*V  (Alg. 1 din paper)
  - MLPProjection       : proiecteaza Z -> Z' (two-layer MLP cu GELU)

Nota importanta pentru GPU 4-8GB:
  CLIP e folosit FROZEN - nu se antreneaza, deci nu consuma memorie
  in backward pass. Features CLIP se pot extrage offline (recomandat).

C6b improvements over baseline:
  1. Fixed token slicing — context tokens are inserted correctly regardless
     of class name length, anchored to the real EOS position.
  2. Class-specific context offsets — each class gets a learnable residual
     on top of the shared context, allowing fine-grained HOI discrimination.
  3. Learnable temperature scalar on cosine similarities (in CLIPTextEncoder)
     — sharpens the L_Cos cross-entropy signal the same way CLIP itself does.
  4. Prompt regularization loss (prompt_reg_loss) — cosine alignment between
     learned T and frozen T, preventing context vectors from drifting out of
     the CLIP embedding space.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Dict
import clip   # pip install git+https://github.com/openai/CLIP.git


# ---------------------------------------------------------------------------
# MLP Projection Z -> Z'
# ---------------------------------------------------------------------------

class MLPProjection(nn.Module):
    """
    Two-layer MLP cu GELU (din paper, sectiunea 3.2).
    Proiecteaza features intermediare Z in spatiul de comparatie cu T.

    Z (hidden_dim) -> Z' (clip_dim=512) pentru comparatie cosinus cu textul.
    """

    def __init__(self, input_dim: int = 256, output_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        mid_dim = (input_dim + output_dim) // 2

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, mid_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mid_dim, output_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, N, input_dim)
        Returns:
            z_prime: (B, N, output_dim) - L2 normalizat
        """
        z_prime = self.mlp(z)
        return F.normalize(z_prime, dim=-1)   # L2 norm (din paper)


class SimpleMLPProjection(nn.Module):
    """
    Original VHOIP MLP: two linear layers with GELU, no dropout, no L2 norm.
    Used for ablation: w/o improved projection head.
    """

    def __init__(self, input_dim: int = 256, output_dim: int = 512):
        super().__init__()
        mid_dim = (input_dim + output_dim) // 2
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, mid_dim),
            nn.GELU(),
            nn.Linear(mid_dim, output_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.mlp(z)


# ---------------------------------------------------------------------------
# CLIP Visual Encoder (frozen)
# ---------------------------------------------------------------------------

class CLIPVisualEncoder(nn.Module):
    """
    Wrapper peste CLIP visual encoder (ViT-B/16).
    Folosit FROZEN - nu se antreneaza niciodata.

    Extrage features pentru regiunile de interes (crops) din video.
    In practica, aceste features se extrag OFFLINE o singura data
    si se salveaza ca fisiere .npy pentru a economisi timp si VRAM.
    """

    def __init__(self, model_name: str = "ViT-B/16", device: str = "cuda"):
        super().__init__()
        self.device = device

        # Incarca CLIP (doar encoder-ul vizual)
        clip_model, self.preprocess = clip.load(model_name, device=device)
        self.visual_encoder = clip_model.visual
        self.feature_dim = 512   # output dim CLIP ViT-B/16

        # Freeze complet - nu antrenam CLIP
        for param in self.visual_encoder.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (N, 3, H, W) - crops normalizate CLIP
        Returns:
            features: (N, 512) - L2 normalizate
        """
        features = self.visual_encoder(images.to(self.device))
        return F.normalize(features.float(), dim=-1)

    @torch.no_grad()
    def extract_offline(self, dataloader, save_path: str) -> None:
        """
        Extrage features CLIP pentru intregul dataset si le salveaza pe disc.
        Apeleaza o singura data inainte de antrenare.
        """
        import numpy as np
        import os

        all_features = {}
        for batch in dataloader:
            video_ids = batch["video_id"]
            crops = batch["crops"]    # (B, S, M, 3, H, W)
            B, S, M, C, H, W = crops.shape

            crops_flat = crops.view(B * S * M, C, H, W)
            feats = self.forward(crops_flat)
            feats = feats.view(B, S, M, -1).cpu().numpy()

            for i, vid_id in enumerate(video_ids):
                all_features[vid_id] = feats[i]

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        import numpy as np
        np.save(save_path, all_features)
        print(f"Features CLIP salvate in {save_path}")


# ---------------------------------------------------------------------------
# Learnable Prompt Encoder (CoOp-style, C6b) — IMPROVED
# ---------------------------------------------------------------------------

class LearnablePromptEncoder(nn.Module):
    """
    C6b — Learnable Prompt Context Tokens (CoOp-style), improved version.

    Changes vs. original:
      1. Fixed token slicing: context insertion is now anchored to the real
         EOS position (tokens.argmax(dim=-1)), not derived from seq_len - n_ctx.
         This prevents class tokens from being silently truncated for longer
         class names or large n_ctx values.

      2. Class-specific context offsets: each of the C classes gets a small
         learnable residual vector (class_ctx_offsets, shape C x n_ctx x D)
         added on top of the shared context. This lets the model learn that
         "approach" and "pour" need different prompt shapes while still
         sharing a common base — important for fine-grained HOI.
         Initialized to zero so training starts from the shared-context baseline.

    Architecture per class k:
        [SOS] [ctx_shared + offset_k] x n_ctx [class_tokens_k] [EOS] [PAD...]

    Reference: Zhou et al., "Learning to Prompt for Vision-Language Models"
               (CoOp), IJCV 2022. Adapted for V-HOI recognition.
    """

    def __init__(
        self,
        clip_model,
        num_classes: int,
        n_ctx: int = 16,
        ctx_init: str = "a photo of a person",
        clip_dim: int = 512,
        device: str = "cuda",
    ):
        super().__init__()
        self.device = device
        self.n_ctx = n_ctx
        self.clip_model = clip_model  # frozen
        self.dtype = clip_model.dtype
        self.num_classes = num_classes

        # --- Shared context vectors (same as original CoOp) ---
        if ctx_init:
            ctx_init_tokens = clip.tokenize(ctx_init).to(device)
            with torch.no_grad():
                init_embeddings = clip_model.token_embedding(ctx_init_tokens)
                # Take exactly n_ctx tokens starting after SOS
                init_ctx = init_embeddings[0, 1:n_ctx + 1, :]  # (n_ctx, clip_dim)
            self.ctx = nn.Parameter(init_ctx.to(self.dtype))
        else:
            ctx_vectors = torch.empty(n_ctx, clip_dim, dtype=self.dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            self.ctx = nn.Parameter(ctx_vectors)

        # --- Class-specific context offsets (NEW) ---
        # Shape: (num_classes, n_ctx, clip_dim), initialized to zero
        # so the model starts at the shared-context baseline and can diverge
        # gradually as training provides class-specific gradients.
        self.class_ctx_offsets = nn.Parameter(
            torch.zeros(num_classes, n_ctx, clip_dim, dtype=self.dtype)
        )

        for param in self.clip_model.parameters():
            param.requires_grad = False

    def forward(self, label_names: List[str], subject: str = "person") -> torch.Tensor:
        """
        Encode class names with learnable context prepended.

        FIX: Token insertion is now anchored to the real EOS position of each
        class string (tokens.argmax(dim=-1)), so class tokens are never
        silently truncated regardless of class name length or n_ctx value.

        Returns:
            T: (C, clip_dim) L2-normalized text features
        """
        class_texts = [f"{subject} {label}" for label in label_names]
        C = len(class_texts)

        with torch.no_grad():
            tokens = clip.tokenize(class_texts).to(self.device)
            token_embeddings = self.clip_model.token_embedding(tokens)
            # token_embeddings: (C, seq_len, clip_dim)

        seq_len = token_embeddings.shape[1]

        # --- FIX: anchor to real EOS positions ---
        # EOS token is the highest token ID in each row.
        eos_positions = tokens.argmax(dim=-1)   # (C,) — real EOS index per class

        # Build composed sequences class-by-class so each uses its own EOS anchor.
        # This is slightly slower than a fully batched op but correct and clear.
        ctx_shared = self.ctx.unsqueeze(0).expand(C, -1, -1)          # (C, n_ctx, D)
        ctx_shared = ctx_shared.to(token_embeddings.dtype)
        offsets = self.class_ctx_offsets.to(token_embeddings.dtype)    # (C, n_ctx, D)
        ctx_per_class = ctx_shared + offsets                            # (C, n_ctx, D)

        composed_list = []
        for i in range(C):
            eos_pos = eos_positions[i].item()   # scalar int

            # SOS: position 0
            sos = token_embeddings[i, :1, :]                            # (1, D)

            # Class content tokens: from position 1 up to (but not including) EOS
            # These are the actual word-piece tokens for "{subject} {label}"
            class_tokens = token_embeddings[i, 1:eos_pos, :]            # (eos_pos-1, D)

            # EOS + any padding: from eos_pos onwards
            suffix = token_embeddings[i, eos_pos:, :]                   # (seq_len-eos_pos, D)

            # Compose: [SOS] [ctx x n_ctx] [class_tokens] [EOS] [PAD...]
            composed = torch.cat([
                sos,                    # (1, D)
                ctx_per_class[i],       # (n_ctx, D)
                class_tokens,           # (variable, D)
                suffix,                 # (EOS + PAD, D)
            ], dim=0)                   # (1 + n_ctx + (eos_pos-1) + (seq_len-eos_pos), D)

            # Truncate or pad to seq_len
            if composed.shape[0] >= seq_len:
                composed = composed[:seq_len]
            else:
                pad_len = seq_len - composed.shape[0]
                pad = torch.zeros(pad_len, composed.shape[1],
                                  dtype=composed.dtype, device=composed.device)
                composed = torch.cat([composed, pad], dim=0)

            composed_list.append(composed)

        composed_batch = torch.stack(composed_list, dim=0).type(self.dtype)  # (C, seq_len, D)

        # Run through CLIP transformer (same as original)
        x = composed_batch + self.clip_model.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)                                         # (seq_len, C, D)
        x = self.clip_model.transformer(x)
        x = x.permute(1, 0, 2)                                         # (C, seq_len, D)
        x = self.clip_model.ln_final(x).type(self.dtype)

        T = x[torch.arange(C), tokens.argmax(dim=-1)]                  # (C, D)
        T = T @ self.clip_model.text_projection                         # (C, clip_dim)

        return F.normalize(T.float(), dim=-1)                           # (C, clip_dim)


# ---------------------------------------------------------------------------
# CLIP Text Encoder (frozen + optional learnable prompts)
# ---------------------------------------------------------------------------

class CLIPTextEncoder(nn.Module):
    """
    Wrapper peste CLIP text encoder.
    Transforma template-uri HOI in reprezentari textuale T.

    C6b additions:
      - learnable_temperature: a scalar log-temperature initialized to log(1/0.07)
        (same as CLIP's default), applied to cosine similarities in vhoip.py.
        Exposed as self.log_temp so vhoip.py can use it directly.
      - get_frozen_T(): returns the frozen text features computed at init,
        used by the prompt regularization loss.
    """

    def __init__(
        self,
        model_name: str = "ViT-B/16",
        device: str = "cuda",
        use_learnable_prompts: bool = False,
        n_ctx: int = 16,
        ctx_init: str = "a photo of a person",
        num_classes: int = 0,   # required when use_learnable_prompts=True
        learnable_temp: bool = True,
    ):
        super().__init__()
        self.device = device
        self.use_learnable_prompts = use_learnable_prompts

        clip_model, _ = clip.load(model_name, device=device)
        # C6b: force float32 for the text encoder.  CLIP loads in fp16 on CUDA
        # by default, but back-prop through a frozen fp16 transformer with
        # learnable inputs is numerically unstable and produces NaN after a
        # handful of steps.
        clip_model = clip_model.float()
        self.text_encoder = clip_model
        self.feature_dim = 512

        for param in self.text_encoder.parameters():
            param.requires_grad = False

        # Temperature applied to cosine similarities.
        # If learnable: log-space parameter starting at log(0.07) (CLIP default).
        # If fixed: constant 1.0 (log_temp=0, no grad).
        if learnable_temp:
            self.log_temp = nn.Parameter(torch.tensor(np.log(0.07), dtype=torch.float32))  # learnable
        else:
            self.log_temp = nn.Parameter(torch.tensor(0.0, dtype=torch.float32), requires_grad=False)

        if use_learnable_prompts:
            assert num_classes > 0, "num_classes must be set when use_learnable_prompts=True"
            self.prompt_encoder = LearnablePromptEncoder(
                clip_model=self.text_encoder,
                num_classes=num_classes,
                n_ctx=n_ctx,
                ctx_init=ctx_init,
                clip_dim=self.feature_dim,
                device=device,
            )
            print(f"  C6b: LearnablePromptEncoder initialized "
                  f"(n_ctx={n_ctx}, num_classes={num_classes}, ctx_init='{ctx_init}')")
        else:
            self.prompt_encoder = None

        # Frozen T — stored at init, used for prompt regularization loss in C6b.
        # Populated by CLIPTextEncoder.precompute_frozen_T() called from vhoip.py.
        self._frozen_T: Optional[torch.Tensor] = None

    @property
    def temperature(self) -> torch.Tensor:
        """Positive temperature scalar: exp(log_temp). Clamped to [0.01, 1.0]."""
        return self.log_temp.clamp(min=-4.6, max=0.0).exp()  
    
    def precompute_frozen_T(
        self,
        label_names: List[str],
        subject: str,
        template,
    ) -> torch.Tensor:
        """
        Compute and store frozen text features T (no grad, fixed templates).
        Called once at model init. Returns T and also stores it internally
        so get_frozen_T() works at any point later.
        """
        from omegaconf import ListConfig
        if isinstance(template, ListConfig):
            template = list(template)

        with torch.no_grad():
            if isinstance(template, str):
                templates = [template]
            else:
                templates = list(template)

            all_features = []
            for tmpl in templates:
                texts = [tmpl.format(subject=subject, verb=label) for label in label_names]
                tokens = clip.tokenize(texts).to(self.device)
                feats = self.text_encoder.encode_text(tokens)
                feats = F.normalize(feats.float(), dim=-1)
                all_features.append(feats)

            if len(all_features) == 1:
                T = all_features[0]
            else:
                T = F.normalize(torch.stack(all_features, dim=0).mean(dim=0), dim=-1)

        self._frozen_T = T
        return T

    def get_frozen_T(self) -> Optional[torch.Tensor]:
        """Returns frozen T (computed at init). None if not yet precomputed."""
        return self._frozen_T

    def encode_labels(
        self,
        label_names: List[str],
        subject: str = "person",
        template = "A photo of a {subject} {verb}",
    ) -> torch.Tensor:
        """
        C6 — Multi-Template Prompt Ensembling / C6b — Learnable Prompts.

        In C6b mode: uses LearnablePromptEncoder (gradients flow through ctx).
        In C6 mode:  static multi-template ensemble (no grad needed).
        """
        if self.prompt_encoder is not None:
            # C6b path — gradient flows through learnable ctx + class offsets
            return self.prompt_encoder(label_names, subject=subject)

        # Original C6 / baseline path (no grad needed)
        with torch.no_grad():
            from omegaconf import ListConfig
            if isinstance(template, ListConfig):
                template = list(template)
            if isinstance(template, str):
                templates = [template]
            else:
                templates = list(template)

            all_features = []
            for tmpl in templates:
                texts = [
                    tmpl.format(subject=subject, verb=label)
                    for label in label_names
                ]
                tokens = clip.tokenize(texts).to(self.device)
                feats = self.text_encoder.encode_text(tokens)
                feats = F.normalize(feats.float(), dim=-1)
                all_features.append(feats)

            if len(all_features) == 1:
                return all_features[0]

            stacked = torch.stack(all_features, dim=0)
            T = stacked.mean(dim=0)
            return F.normalize(T, dim=-1)


# ---------------------------------------------------------------------------
# Prototyping
# ---------------------------------------------------------------------------

class Prototyping(nn.Module):
    """
    Calculeaza prototipuri per clasa prin medierea si L2 normalizarea
    feature-urilor din aceeasi clasa.

    Folosit pentru:
      - G_init: prototipuri CLIP vizuale (calculat o data la inceput)
      - V: prototipuri V-HOI (calculat dupa fiecare epoch)
    """

    def __init__(self, num_classes: int, feature_dim: int):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            features: (N, feature_dim) - features colectate
            labels:   (N,) - eticheta clasei pentru fiecare feature (long)
        Returns:
            prototypes: (C, feature_dim) - prototip per clasa, L2 normalizat
        """
        device = features.device
        prototypes = torch.zeros(self.num_classes, self.feature_dim, device=device)
        counts = torch.zeros(self.num_classes, device=device)

        for c in range(self.num_classes):
            mask = labels == c
            if mask.sum() > 0:
                prototypes[c] = features[mask].mean(dim=0)
                counts[c] = mask.sum()

        # L2 normalizare (din paper)
        prototypes = F.normalize(prototypes, dim=-1)
        return prototypes


# ---------------------------------------------------------------------------
# Integrated Global Representation (EMA)
# ---------------------------------------------------------------------------

class IntegratedGlobalRepresentation(nn.Module):
    """
    Implementeaza Algorithm 1 din paper:
    G = rho * G + (1 - rho) * V

    G este initializat cu G_init (prototipuri CLIP).
    Dupa warm-up, G se actualizeaza cu V (prototipuri V-HOI)
    la sfarsitul fiecarui epoch.

    G este reprezentarea globala integrata folosita de discriminator.
    """

    def __init__(
        self,
        num_classes: int,
        feature_dim: int,
        rho: float = 0.9,
        warmup_epochs: int = 5,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.rho = rho
        self.warmup_epochs = warmup_epochs

        # G - reprezentarea globala integrata (buffer, nu parametru antrenabil)
        self.register_buffer(
            "G",
            torch.zeros(num_classes, feature_dim)
        )
        self.prototyping = Prototyping(num_classes, feature_dim)
        # Flag explicit ca buffer pentru robustete la save/load checkpoint.
        self.register_buffer("_initialized_flag", torch.tensor(False))

    def _check_initialized(self) -> None:
        """
        Sincronizeaza flag-ul cu buffer-ul G (backward-compatible).
        Daca un checkpoint vechi a incarcat G dar nu avea _initialized_flag,
        detecteaza automat si seteaza flag-ul corect.
        """
        if not self._initialized_flag.item() and self.G.norm().item() > 0:
            self._initialized_flag = torch.tensor(True)

    def initialize(self, g_init: torch.Tensor) -> None:
        self.G.copy_(g_init)
        self._initialized_flag = torch.tensor(True)
        print(f"  G initializat cu prototipuri CLIP (shape: {g_init.shape})")

    @torch.no_grad()
    def update(
        self,
        collected_features: torch.Tensor,
        labels: torch.Tensor,
        epoch: int,
    ) -> None:
        self._check_initialized()
        if not self._initialized_flag.item():
            raise RuntimeError("Apeleaza initialize() cu G_init inainte de update().")

        valid_mask = ~torch.isnan(collected_features).any(dim=-1)
        if valid_mask.sum() == 0:
            print("  [WARN] EMA update sarit: toate features colectate sunt NaN.")
            return
        collected_features = collected_features[valid_mask]
        labels = labels[valid_mask]

        valid_labels = labels != -1
        if valid_labels.sum() == 0:
            return
        collected_features = collected_features[valid_labels]
        labels = labels[valid_labels]

        V = self.prototyping(collected_features, labels)

        if epoch >= self.warmup_epochs - 1:
            has_samples = (V.norm(dim=-1) > 1e-6)
            new_G = self.rho * self.G + (1 - self.rho) * V
            self.G = torch.where(has_samples.unsqueeze(-1), new_G, self.G)
            norms = self.G.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            self.G = self.G / norms

    def get_G(self) -> torch.Tensor:
        return self.G


# ---------------------------------------------------------------------------
# CLIP Features Collector (pentru prototyping la sfarsitul epoch-ului)
# ---------------------------------------------------------------------------

class FeaturesCollector:
    """
    Colecteaza features si etichete de-a lungul unui epoch
    pentru a calcula prototipurile V dupa fiecare epoch.
    """

    def __init__(self):
        self.features_list = []
        self.labels_list = []

    def add(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        f = features.detach().cpu()
        l = labels.detach().cpu()
        if f.dim() == 3:
            f = f.reshape(-1, f.shape[-1])
            l = l.reshape(-1)
        self.features_list.append(f)
        self.labels_list.append(l)

    def get_all(self):
        if not self.features_list:
            return None, None
        return (
            torch.cat(self.features_list, dim=0),
            torch.cat(self.labels_list, dim=0),
        )

    def reset(self) -> None:
        self.features_list = []
        self.labels_list = []