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
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
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
        np.save(save_path, all_features)
        print(f"Features CLIP salvate in {save_path}")


# ---------------------------------------------------------------------------
# Learnable Prompt Encoder (CoOp-style, C6b)
# ---------------------------------------------------------------------------

class LearnablePromptEncoder(nn.Module):
    """
    C6b — Learnable Prompt Context Tokens (CoOp-style).

    Adds n_ctx learnable context vectors prepended to the class token
    embeddings inside CLIP's text encoder. CLIP stays completely frozen.
    Only the context vectors are trained.

    Architecture:
        Input per class: [SOS] [ctx_1] ... [ctx_n] [class_token] [EOS]
        vs original:     [SOS] [template_tokens...] [class_token] [EOS]

    Reference: Zhou et al., "Learning to Prompt for Vision-Language Models"
               (CoOp), IJCV 2022. Adapted for V-HOI recognition.
    """

    def __init__(
        self,
        clip_model,
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

        if ctx_init:
            ctx_init_tokens = clip.tokenize(ctx_init).to(device)
            with torch.no_grad():
                init_embeddings = clip_model.token_embedding(ctx_init_tokens)
                init_ctx = init_embeddings[0, 1:n_ctx+1, :]  # (n_ctx, clip_dim)
            self.ctx = nn.Parameter(init_ctx.float())
        else:
            ctx_vectors = torch.empty(n_ctx, clip_dim)
            nn.init.normal_(ctx_vectors, std=0.02)
            self.ctx = nn.Parameter(ctx_vectors)

        for param in self.clip_model.parameters():
            param.requires_grad = False

    def forward(self, label_names: List[str], subject: str = "person") -> torch.Tensor:
        """
        Encode class names with learnable context prepended.

        Returns:
            T: (C, clip_dim) L2-normalized text features
        """
        class_texts = [f"{subject} {label}" for label in label_names]

        with torch.no_grad():
            tokens = clip.tokenize(class_texts).to(self.device)
            token_embeddings = self.clip_model.token_embedding(tokens)
            # token_embeddings: (C, seq_len, clip_dim)

        C, seq_len, _ = token_embeddings.shape

        sos = token_embeddings[:, :1, :]                        # (C, 1, D)
        ctx = self.ctx.unsqueeze(0).expand(C, -1, -1)          # (C, n_ctx, D)
        ctx = ctx.to(token_embeddings.dtype)

        class_token_start = 1
        class_token_end = seq_len - self.n_ctx - 1
        class_token_end = max(class_token_end, class_token_start + 1)

        class_tokens = token_embeddings[:, class_token_start:class_token_end, :]
        suffix = token_embeddings[:, class_token_end:, :]

        composed = torch.cat([sos, ctx, class_tokens, suffix], dim=1)
        composed = composed[:, :seq_len, :]                     # (C, seq_len, D)

        x = composed + self.clip_model.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)                                 # (seq_len, C, D)
        x = self.clip_model.transformer(x)
        x = x.permute(1, 0, 2)                                 # (C, seq_len, D)
        x = self.clip_model.ln_final(x).type(self.dtype)

        T = x[torch.arange(C), tokens.argmax(dim=-1)]          # (C, D)
        T = T @ self.clip_model.text_projection                 # (C, clip_dim)

        return F.normalize(T.float(), dim=-1)                   # (C, clip_dim)


# ---------------------------------------------------------------------------
# CLIP Text Encoder (frozen)
# ---------------------------------------------------------------------------

class CLIPTextEncoder(nn.Module):
    """
    Wrapper peste CLIP text encoder.
    Transforma template-uri HOI in reprezentari textuale T.

    Template din paper: "A photo of a/an [subject] [verb-ing/able]"
    Ex: "A photo of a person eating"
        "A photo of a hand sawing"
    """

    def __init__(
        self,
        model_name: str = "ViT-B/16",
        device: str = "cuda",
        use_learnable_prompts: bool = False,
        n_ctx: int = 16,
        ctx_init: str = "a photo of a person",
    ):
        super().__init__()
        self.device = device
        self.use_learnable_prompts = use_learnable_prompts

        clip_model, _ = clip.load(model_name, device=device)
        self.text_encoder = clip_model
        self.feature_dim = 512

        for param in self.text_encoder.parameters():
            param.requires_grad = False

        if use_learnable_prompts:
            self.prompt_encoder = LearnablePromptEncoder(
                clip_model=self.text_encoder,
                n_ctx=n_ctx,
                ctx_init=ctx_init,
                clip_dim=self.feature_dim,
                device=device,
            )
            print(f"  C6b: LearnablePromptEncoder initialized "
                  f"(n_ctx={n_ctx}, ctx_init='{ctx_init}')")
        else:
            self.prompt_encoder = None

    def encode_labels(
        self,
        label_names: List[str],
        subject: str = "person",
        template: str = "A photo of a {subject} {verb}",
    ) -> torch.Tensor:
        """
        C6 — Multi-Template Prompt Ensembling.

        Accepts either a single template string or a list of templates.
        When multiple templates are given, encodes all of them and averages
        the resulting embeddings before L2-normalizing — this is the static
        ensemble that forms the T representation used in L_Cos.

        Single template (baseline behaviour):
            template = "A photo of a {subject} {verb}"
        Multi-template (C6):
            template = [
                "A photo of a {subject} {verb}",
                "A {subject} is {verb}",
                "The {subject} performs the action of {verb}",
            ]
        """
        if self.prompt_encoder is not None:
            # C6b path — gradient flows through learnable ctx only
            return self.prompt_encoder(label_names, subject=subject)

        # Original C6 / baseline path (no grad needed)
        with torch.no_grad():
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
                feats = self.text_encoder.encode_text(tokens)   # (C, 512)
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
        # NOTA: _initialized NU mai este stocat ca atribut Python simplu.
        # Este derivat din buffer-ul G (vezi property de mai jos), astfel incat
        # starea este intotdeauna corecta dupa save/load checkpoint.

    @property
    def _initialized(self) -> bool:
        """
        Returneaza True daca G a fost initializat cu valori non-zero.
        Derivat din buffer-ul G, deci corect si dupa resume din checkpoint.
        """
        return bool(self.G.norm().item() > 0)

    @_initialized.setter
    def _initialized(self, value: bool) -> None:
        # Setter no-op: permite cod extern (ex. train.py) sa seteze flag-ul
        # fara a ridica eroare, dar starea reala ramane derivata din G.
        # Daca value=True si G e zero, inseamna o eroare logica upstream.
        if value and not bool(self.G.norm().item() > 0):
            import warnings
            warnings.warn(
                "S-a setat _initialized=True dar G este inca zero. "
                "Asigura-te ca initialize() sau load_checkpoint() a fost apelat corect.",
                RuntimeWarning,
                stacklevel=2,
            )

    def initialize(self, g_init: torch.Tensor) -> None:
        """
        Initializeaza G cu G_init (prototipurile CLIP).
        Se apeleaza o singura data la inceputul antrenarii.

        Args:
            g_init: (C, feature_dim) - prototipuri CLIP
        """
        self.G.copy_(g_init)
        # _initialized este acum derivat din G — nu mai trebuie setat manual.
        print(f"  G initializat cu prototipuri CLIP (shape: {g_init.shape})")

    @torch.no_grad()
    def update(
        self,
        collected_features: torch.Tensor,
        labels: torch.Tensor,
        epoch: int,
    ) -> None:
        """
        Actualizeaza G cu EMA dupa fiecare epoch (dupa warm-up).
        Se apeleaza la sfarsitul fiecarui epoch de antrenare.

        Args:
            collected_features: (N_total, feature_dim) - toate Z' din epoch
            labels:             (N_total,) - etichetele corespunzatoare
            epoch:              epoch curent
        """
        if not self._initialized:
            raise RuntimeError("Apeleaza initialize() cu G_init inainte de update().")

        # Filtreaza features NaN inainte de prototyping.
        # NaN-urile apar cand gradientii explodeaza si polueaza Z' colectat.
        valid_mask = ~torch.isnan(collected_features).any(dim=-1)  # (N,)
        if valid_mask.sum() == 0:
            # Toate features sunt NaN — skip update complet, pastreaza G curent
            print("  [WARN] EMA update sarit: toate features colectate sunt NaN.")
            return
        collected_features = collected_features[valid_mask]
        labels = labels[valid_mask]

        # Filtreaza si etichetele invalide (-1 = padding)
        valid_labels = labels != -1
        if valid_labels.sum() == 0:
            return
        collected_features = collected_features[valid_labels]
        labels = labels[valid_labels]

        # Calculeaza V (prototipurile V-HOI din acest epoch)
        V = self.prototyping(collected_features, labels)   # (C, feature_dim)

        if epoch >= self.warmup_epochs - 1:
            # EMA update: G = rho*G + (1-rho)*V
            # Actualizeaza DOAR clasele care au avut sample-uri in acest epoch.
            # Clasele fara sample-uri au prototip zero dupa F.normalize -> NaN.
            # Le pastram pe cele din G-ul precedent.
            has_samples = (V.norm(dim=-1) > 1e-6)  # (C,) — clase cu date reale

            new_G = self.rho * self.G + (1 - self.rho) * V
            self.G = torch.where(has_samples.unsqueeze(-1), new_G, self.G)

            # Re-normalizeaza doar randurile actualizate (cele cu has_samples)
            # pentru a nu disturba randurile neactualizate
            norms = self.G.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            self.G = self.G / norms

    def get_G(self) -> torch.Tensor:
        """Returneaza G curent (C, feature_dim)."""
        return self.G


# ---------------------------------------------------------------------------
# CLIP Features Collector (pentru prototyping la sfarsitul epoch-ului)
# ---------------------------------------------------------------------------

class FeaturesCollector:
    """
    Colecteaza features si etichete de-a lungul unui epoch
    pentru a calcula prototipurile V dupa fiecare epoch.

    Folosit in bucla de antrenare:
        collector = FeaturesCollector()
        for batch in dataloader:
            ...
            collector.add(z_prime.detach(), labels)
        V = collector.get_all_features()
    """

    def __init__(self):
        self.features_list = []
        self.labels_list = []

    def add(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        """
        Args:
            features: (B, N, D) sau (N, D) - features Z' cu stop_gradient
            labels:   (B, N) sau (N,) - etichete
        """
        f = features.detach().cpu()
        l = labels.detach().cpu()

        if f.dim() == 3:
            f = f.reshape(-1, f.shape[-1])
            l = l.reshape(-1)

        self.features_list.append(f)
        self.labels_list.append(l)

    def get_all(self):
        """Returns: (N_total, D), (N_total,)"""
        if not self.features_list:
            return None, None
        return (
            torch.cat(self.features_list, dim=0),
            torch.cat(self.labels_list, dim=0),
        )

    def reset(self) -> None:
        self.features_list = []
        self.labels_list = []