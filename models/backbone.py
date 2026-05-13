"""
backbone.py
Implementarea backbone-ului 2G-GCN + ASSIGN conform paperurilor:
  - Morais et al., "Learning Asynchronous and Sparse Human-Object Interaction
    in Videos", CVPR 2021  (ASSIGN)
  - Qiao et al., "Multi-person Human-object Interaction Recognition" (2G-GCN)

Arhitectura (Fig. 2 ASSIGN + Fig. 3 2G-GCN):

  Frame-level layer (ASSIGN §3.3 + 2G-GCN §4):
    1. GeometricLevelGCN      — GCN pe skeleton + bbox keypoints  (2G-GCN Eq. 1–3)
    2. FusionLevelGraph       — fuziune raw ROI + geometric -> D  (2G-GCN Eq. 4)
    3. FrameLevelBiRNN        — BiRNN pe features imbogatite, h^e_{t,f}  (ASSIGN Eq. 1)
    4. SpatialMessagePassing  — mesaje intra/inter-class  (ASSIGN Eq. 2–4)
    5. SegmentBoundaryDetector — MLP + Gumbel-Softmax, u^e_t  (ASSIGN Eq. 5)

Training in doua stagii (ASSIGN §3.5):
  Stage 1: u^e_t := 1 everywhere (dense), L_Seg oprit
  Stage 2: Gumbel-Softmax activ, L_Seg pornit
  Tranzitia se face prin parametrul `training_stage` in forward().

Notatii:
  B  = batch size
  S  = frame-uri
  M  = entitati per frame
  N  = S * M
  D  = hidden_dim
  C  = num_classes
  J  = keypoints geometrice
  C1=64, C2=128 (2G-GCN)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict


# ---------------------------------------------------------------------------
# Scaled dot-product attention  (ASSIGN Eq. 3 / 2G-GCN Eq. 4)
# Att(q, {z_i}) = sum_i softmax(q^T z_i / sqrt(d)) * z_i
# ---------------------------------------------------------------------------

def scaled_dot_attention(
    query: torch.Tensor,
    keys_values: torch.Tensor,
) -> torch.Tensor:
    """
    Args:
        query:       (..., 1, D)
        keys_values: (..., K, D)
    Returns:         (..., 1, D)
    """
    d = query.shape[-1]
    scores  = torch.matmul(query, keys_values.transpose(-2, -1)) / (d ** 0.5)
    weights = torch.softmax(scores, dim=-1)
    return torch.matmul(weights, keys_values)


# ---------------------------------------------------------------------------
# Geometric-level GCN  (2G-GCN §4.1–4.2, Eq. 1–3)
# ---------------------------------------------------------------------------

class GeometricLevelGCN(nn.Module):
    """
    Graful geometric din 2G-GCN.
    Intrare: pozitii + viteze keypoints  (B, J, geo_input_dim)
    Iesire:  reprezentari geometrice  (B, J, C2)
    """

    def __init__(self, geo_input_dim: int = 4, C1: int = 64, C2: int = 128):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(geo_input_dim, C1), nn.ReLU(),
            nn.Linear(C1, C1),           nn.ReLU(),
        )
        self.theta = nn.Linear(C1, C2, bias=False)
        self.phi   = nn.Linear(C1, C2, bias=False)
        self.W_g   = nn.Linear(C1, C2, bias=False)

    def forward(self, geo: torch.Tensor) -> torch.Tensor:
        g  = self.embed(geo)
        At = torch.softmax(
            torch.bmm(self.theta(g), self.phi(g).transpose(1, 2)), dim=-1
        )
        return torch.bmm(At, self.W_g(g))   # (B, J, C2)


# ---------------------------------------------------------------------------
# Fusion-level Graph  (2G-GCN §4.3, Eq. 4)
# ---------------------------------------------------------------------------

class FusionLevelGraph(nn.Module):
    """
    Fuziune features vizuale + geometrice.
    Geometry->object: inclus. Geometry->human: inclus (conform paper 2G-GCN).
    """

    def __init__(self, visual_dim: int = 256, geo_dim: int = 128,
                 hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.visual_proj = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.geo_proj = nn.Sequential(nn.Linear(geo_dim, hidden_dim), nn.ReLU())
        self.scale    = hidden_dim ** 0.5
        self.norm     = nn.LayerNorm(hidden_dim)
        self.dropout  = nn.Dropout(dropout)

    def forward(self, visual: torch.Tensor,
                geo_out: Optional[torch.Tensor] = None,
                entity_types: Optional[torch.Tensor] = None) -> torch.Tensor:
        v = self.visual_proj(visual)
        if geo_out is not None:
            all_keys = torch.cat([v, self.geo_proj(geo_out)], dim=1)
        else:
            all_keys = v
        scores = torch.bmm(v, all_keys.transpose(1, 2)) / self.scale
        weights = self.dropout(torch.softmax(scores, dim=-1))
        return self.norm(torch.bmm(weights, all_keys) + v)


# ---------------------------------------------------------------------------
# Frame-level BiRNN  (ASSIGN §3.3, Eq. 1)
# ---------------------------------------------------------------------------

class FrameLevelBiRNN(nn.Module):
    """Primul nivel BiRNN, produce h^e_{t,f} + frame-level logits."""

    def __init__(self, input_dim: int = 2048, hidden_dim: int = 256,
                 num_classes: int = 10, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        # Dupa FusionLevelGraph, inputul este deja hidden_dim.
        # Proiectia este necesara doar cand inputul brut (ex. 2048-dim ROI)
        # ajunge direct in BiRNN.
        if input_dim != hidden_dim:
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
        else:
            self.input_proj = nn.Identity()
        self.birnn = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.frame_classifier = nn.Linear(hidden_dim, num_classes)
        self.dropout    = nn.Dropout(dropout)
        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """x: (B*M, S, input_dim) -> z: (B*M, S, D), frame_logits: (B*M, S, C)"""
        h = self.input_proj(x)
        z, _ = self.birnn(h)
        z = self.dropout(z)
        return z, self.frame_classifier(z)


# ---------------------------------------------------------------------------
# Spatial Message Passing  (ASSIGN §3.3, Eq. 2–4)
# ---------------------------------------------------------------------------

class SpatialMessagePassing(nn.Module):
    """
    Mesaje spatiale intra/inter-class per frame.

    m^{inter->e}_{t,f} = Att([x^e;h^e], {[x^k;h^k]}_{c^k != c^e})   (Eq. 2)
    m^{intra->e}_{t,f} = Att([x^e;h^e], {[x^k;h^k]}_{k!=e,c^k=c^e}) (Eq. 4)

    Daca nu exista vecini, mesajul este zero.
    """

    def forward(
        self,
        combined: torch.Tensor,       # (B, M, 2D)
        entity_types: torch.Tensor,   # (B, M) int64
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, M, FD = combined.shape
        scale = FD ** 0.5

        scores = torch.bmm(combined, combined.transpose(1, 2)) / scale  # (B, M, M)

        types_row = entity_types.unsqueeze(2).expand(B, M, M)  # (B, M, M) query types
        types_col = entity_types.unsqueeze(1).expand(B, M, M)  # (B, M, M) key types
        diag      = torch.eye(M, dtype=torch.bool, device=combined.device).unsqueeze(0)

        inter_mask = types_col != types_row           # (B, M, M) — different class
        intra_mask = (types_col == types_row) & ~diag  # (B, M, M) — same class, not self

        inter_w = torch.nan_to_num(
            torch.softmax(scores.masked_fill(~inter_mask, float('-inf')), dim=-1), nan=0.0
        )
        intra_w = torch.nan_to_num(
            torch.softmax(scores.masked_fill(~intra_mask, float('-inf')), dim=-1), nan=0.0
        )

        return torch.bmm(inter_w, combined), torch.bmm(intra_w, combined)


# ---------------------------------------------------------------------------
# Segment Boundary Detector  (ASSIGN §3.3, Eq. 5)
# ---------------------------------------------------------------------------

class SegmentBoundaryDetector(nn.Module):
    """
    u^e_t = GSM( gamma([x^e_t; h^e_{t,f}; m^{intra}; m^{inter}]) )

    - Training Stage 2: Gumbel-Softmax (differentiable), Straight-Through dla backward
    - Training Stage 1: u := 1 everywhere (handled in Backbone2GGCN.forward)
    - Inference: argmax

    Input dim = 6*D: [x(D); h_f(D); m_intra(2D); m_inter(2D)]
    """

    def __init__(self, input_dim: int, temperature: float = 1.0,
                 threshold: float = 0.5):
        super().__init__()
        mid = input_dim // 2
        self.gamma = nn.Sequential(
            nn.Linear(input_dim, mid), nn.ReLU(),
            nn.Linear(mid, 2),
        )
        self.temperature = temperature
        self.threshold   = threshold

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:  x: (B_flat, input_dim)
        Returns:
            u_soft: (B_flat,)  — real-valued (for L_Seg loss)
            u_hard: (B_flat,)  — binary float, ST gradient at training
        """
        logits = self.gamma(x)   # (B_flat, 2)

        if self.training:
            gsm    = F.gumbel_softmax(logits, tau=self.temperature, hard=False)
            u_soft = gsm[:, 1]                                  # prob "change"
            u_hard = (u_soft > self.threshold).float()
            u_hard = u_hard - u_soft.detach() + u_soft          # Straight-Through
        else:
            probs  = torch.softmax(logits / self.temperature, dim=-1)
            u_soft = probs[:, 1]
            u_hard = (u_soft > self.threshold).float()

        return u_soft, u_hard


# ---------------------------------------------------------------------------
# Segment-level Message Passing  (ASSIGN §3.4, Eq. 6–7)
# ---------------------------------------------------------------------------

def segment_message_passing(
    h_s: torch.Tensor,           # (B, M, D)
    entity_types: torch.Tensor,  # (B, M)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculeaza mesajele inter/intra-class la nivel de segment (Eq. 6–7).
    Identic cu SpatialMessagePassing dar opereaza pe h_s (D) in loc de [x;h](2D).
    """
    B, M, D = h_s.shape
    scale = D ** 0.5

    scores = torch.bmm(h_s, h_s.transpose(1, 2)) / scale  # (B, M, M)

    types_row = entity_types.unsqueeze(2).expand(B, M, M)
    types_col = entity_types.unsqueeze(1).expand(B, M, M)
    diag      = torch.eye(M, dtype=torch.bool, device=h_s.device).unsqueeze(0)

    inter_mask = types_col != types_row
    intra_mask = (types_col == types_row) & ~diag

    inter_w = torch.nan_to_num(
        torch.softmax(scores.masked_fill(~inter_mask, float('-inf')), dim=-1), nan=0.0
    )
    intra_w = torch.nan_to_num(
        torch.softmax(scores.masked_fill(~intra_mask, float('-inf')), dim=-1), nan=0.0
    )

    return torch.bmm(inter_w, h_s), torch.bmm(intra_w, h_s)


# ---------------------------------------------------------------------------
# Segment-level Layer  (ASSIGN §3.4, Eq. 6–10)
# ---------------------------------------------------------------------------

class SegmentLevelLayer(nn.Module):
    """
    Al doilea nivel ASSIGN — sparse si asincron.

    z^e_t = [h^e_{t,f}; m^{inter}_{t,f}; m^{intra}_{t,f};
              m^{inter}_{t,s}; m^{intra}_{t,s}]   (Eq. 8)

    h^e_{t,s} = BiRNNs(z^e_t, ...)    (Eq. 9) — actualizat doar cand u^e_t=1
    y^e_t     = Softmax(sigma(h^e_{t,s}))         (Eq. 10)

    Implementare offline-compatible:
      BiRNN-ul bidirectional necesita toate frame-urile in avans.
      Aplicam zero-masking la frame-urile cu u_hard=0 (skip),
      ceea ce previne update-urile la acele pozitii si pastreaza
      contextul din segmentul anterior — echivalent functional cu
      versiunea online a ASSIGN pentru antrenare pe video complete.

    Mesajele segment-level m^{inter/intra}_{t,s} (Eq. 6-7) se calculeaza
    iterativ, frame cu frame, folosind h_s acumulat pana la t.
    Deoarece BiRNN este bidirectional (necesita toate frame-urile),
    folosim o aproximatie: calculam mesajele pe h_f_fused (forward pass)
    ca initializare, then refinam cu h_s in sens forward-only.
    Aceasta este o aproximatie valida pentru antrenare offline.

    Input per frame: 5D = D(h_f) + D(m_inter_f_proj) + D(m_intra_f_proj)
                         + D(m_inter_s) + D(m_intra_s)
    (mesajele frame-level 2D sunt proiectate la D inainte de concatenare)
    """

    def __init__(self, hidden_dim: int = 256, num_classes: int = 10,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Proiectie mesaje frame (2D -> D) pentru a le aduce la aceeasi dim
        self.frame_msg_proj = nn.Linear(2 * hidden_dim, hidden_dim)

        # BiRNNs: input = 5*D  (Eq. 8 simplificat)
        self.birnn = nn.GRU(
            input_size=5 * hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # sigma + classifier (Eq. 10)
        self.sigma      = nn.Linear(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.dropout    = nn.Dropout(dropout)

    def forward(
        self,
        h_f: torch.Tensor,          # (B, M, S, D)  — frame-level hiddens
        m_inter_f: torch.Tensor,    # (B, M, S, 2D) — frame inter-class msgs
        m_intra_f: torch.Tensor,    # (B, M, S, 2D) — frame intra-class msgs
        u_hard: torch.Tensor,       # (B, M, S)     — boundary signal
        entity_types: torch.Tensor, # (B, M)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            h_s:            (B, M, S, D)
            segment_logits: (B, M, S, C)
        """
        B, M, S, D = h_f.shape
        device = h_f.device

        # Proiecteaza mesajele frame (2D -> D)
        m_inter_f_p = self.frame_msg_proj(m_inter_f)   # (B, M, S, D)
        m_intra_f_p = self.frame_msg_proj(m_intra_f)   # (B, M, S, D)

        # Calculeaza mesajele segment-level iterativ (aproximare forward-only)
        # Initializam h_s_prev cu h_f[:,:,0,:]
        m_inter_s_list = []
        m_intra_s_list = []
        h_s_prev = h_f[:, :, 0, :].clone()   # (B, M, D)

        for t in range(S):
            mi_s, ma_s = segment_message_passing(h_s_prev, entity_types)
            m_inter_s_list.append(mi_s)      # (B, M, D)
            m_intra_s_list.append(ma_s)
            # Actualizeaza h_s_prev doar unde u_hard=1
            update_mask = u_hard[:, :, t].unsqueeze(-1)   # (B, M, 1)
            h_s_prev = h_s_prev * (1 - update_mask) + h_f[:, :, t, :] * update_mask

        m_inter_s = torch.stack(m_inter_s_list, dim=2)   # (B, M, S, D)
        m_intra_s = torch.stack(m_intra_s_list, dim=2)   # (B, M, S, D)

        # z^e_t = [h_f; m_inter_f_proj; m_intra_f_proj; m_inter_s; m_intra_s]  (Eq. 8)
        z = torch.cat([
            h_f,          # (B, M, S, D)
            m_inter_f_p,  # (B, M, S, D)
            m_intra_f_p,  # (B, M, S, D)
            m_inter_s,    # (B, M, S, D)
            m_intra_s,    # (B, M, S, D)
        ], dim=-1)        # (B, M, S, 5D)

        # BiRNNs per entitate  (Eq. 9)
        # IMPORTANT: BiRNN vede TOATE frame-urile (fara zero-masking).
        # u_hard gateaza doar clasificatorul per frame (segment logits),
        # nu inputul BiRNN — conform ASSIGN §3.4 care updateaza selectiv
        # starea h_s, dar lasa BiRNN sa proceseze contextul complet.
        z_flat = z.reshape(B * M, S, -1)
        h_s_flat, _ = self.birnn(z_flat)            # (B*M, S, D)
        h_s_flat    = self.dropout(h_s_flat)
        h_s         = h_s_flat.reshape(B, M, S, D)

        # Eq. 10: y^e_t = Softmax(sigma(h^e_{t,s}))
        # Fiecare frame primeste propria clasificare din h_s calculat de BiRNN.
        # u_hard NU suprima logit-urile la frame-urile de skip — BiRNN-ul
        # a vazut deja contextul complet si produce h_s valide la fiecare t.
        segment_logits = self.classifier(F.relu(self.sigma(h_s)))   # (B, M, S, C)

        return h_s, segment_logits


# ---------------------------------------------------------------------------
# Backbone complet  2G-GCN + ASSIGN
# ---------------------------------------------------------------------------

class Backbone2GGCN(nn.Module):
    """
    Backbone-ul complet integrand ASSIGN cu 2G-GCN.

    Tranzitia intre stagii de training se face prin parametrul
    `training_stage` in forward():
      training_stage=1  -> u^e_t := 1 (dense), Stage 1 din ASSIGN §3.5
      training_stage=2  -> Gumbel-Softmax activ, Stage 2 complet

    La inferenta (eval mode), modelul foloseste argmax pe u^e_t.

    Args:
        input_dim:           dimensiunea ROI pooling (2048)
        hidden_dim:          dimensiunea ascunsa BiRNN (256)
        num_classes:         numar de clase
        num_layers:          straturi GRU
        dropout:             rata dropout
        geo_input_dim:       dimensiunea featurei geometrice (4)
        C1, C2:              dimensiuni GCN geometric (64, 128)
        gsm_temp:            temperatura Gumbel-Softmax initiala (1.0)
        boundary_threshold:  prag binarizare u^e_t (0.5)
    """

    def __init__(
        self,
        input_dim: int = 2048,
        hidden_dim: int = 256,
        num_classes: int = 10,
        num_layers: int = 2,
        dropout: float = 0.3,
        geo_input_dim: int = 4,
        C1: int = 64,
        C2: int = 128,
        gsm_temp: float = 1.0,
        boundary_threshold: float = 0.5,
    ):
        super().__init__()
        self.hidden_dim  = hidden_dim
        self.num_classes = num_classes

        self.geo_gcn   = GeometricLevelGCN(geo_input_dim, C1, C2)
        # Dupa FusionLevelGraph, inputul BiRNN este hidden_dim (nu 2048 brut).
        self.frame_birnn = FrameLevelBiRNN(hidden_dim, hidden_dim, num_classes, num_layers, dropout)
        # FusionLevelGraph primeste ROI brut (input_dim=2048) si produce hidden_dim.
        # Conform 2G-GCN Fig. 3: fusion se aplica INAINTE de BiRNN.
        self.fusion_graph = FusionLevelGraph(input_dim, C2, hidden_dim, dropout)
        self.msg_passing  = SpatialMessagePassing()

        # Detector: input = [x_fused(D); h_f(D); m_intra(2D); m_inter(2D)] = 6D
        self.boundary_detector = SegmentBoundaryDetector(
            input_dim=6 * hidden_dim,
            temperature=gsm_temp,
            threshold=boundary_threshold,
        )

        self.segment_layer = SegmentLevelLayer(hidden_dim, num_classes, num_layers, dropout)

    def set_gsm_temperature(self, temp: float) -> None:
        """Ajusteaza temperatura Gumbel-Softmax (scade pe parcursul training-ului)."""
        self.boundary_detector.temperature = temp

    def forward(
        self,
        roi_features: torch.Tensor,
        geo_features: Optional[torch.Tensor] = None,
        entity_types: Optional[torch.Tensor] = None,
        training_stage: int = 2,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            roi_features:   (B, S, M, 2048)
            geo_features:   (B, S, J, 4)    optional
            entity_types:   (B, M) int64    optional, 0=human 1=object
            training_stage: 1=dense (ASSIGN Stage 1), 2=full (ASSIGN Stage 2)
        Returns:
            z:              (B, N, D)   — Z pt VHOIP MI loss (primul nivel BiRNN)
            frame_logits:   (B, N, C)   — pt L_Seg
            segment_logits: (B, N, C)   — pt L_Label
            u_soft:         (B, N)      — semnal binar relaxat pt L_Seg
        """
        B, S, M, D_in = roi_features.shape
        device = roi_features.device

        if entity_types is None:
            entity_types = torch.tensor(
                [[0, 0] + [1] * (M - 2)] * B,
                dtype=torch.long, device=device,
            )

        # ---- 1. Geometric GCN + Fusion Graph pe raw ROI features  (2G-GCN Fig. 3) ----
        # Ordinea corecta din paper: Geometric -> Fusion -> BiRNN.
        # FusionLevelGraph proiecteaza visual_dim (2048) -> hidden_dim intern.
        roi_bms = roi_features.permute(0, 2, 1, 3)   # (B, M, S, 2048)
        x_fused_list = []
        for s in range(S):
            geo_out_s = self.geo_gcn(geo_features[:, s]) if geo_features is not None else None
            x_fused_list.append(
                self.fusion_graph(roi_bms[:, :, s, :], geo_out_s, entity_types)
            )
        x_fused = torch.stack(x_fused_list, dim=2)   # (B, M, S, D)

        # ---- 2. Frame-level BiRNN pe features imbogatite  (ASSIGN §3.3, Eq. 1) ----
        x_bm = x_fused.reshape(B * M, S, self.hidden_dim)
        z_bm, fl_bm = self.frame_birnn(x_bm)
        h_f      = z_bm.reshape(B, M, S, self.hidden_dim)    # (B, M, S, D)
        fl_4d    = fl_bm.reshape(B, M, S, self.num_classes)  # frame logits 4D

        # Z pentru VHOIP (primul nivel BiRNN, dupa fusion geometric)
        z_intermediate = h_f   # (B, M, S, D)

        # h_f_fused = h_f (fusion deja aplicat la input BiRNN)
        h_f_fused = h_f

        # ---- 3. x^e_t = enriched frame representation (dupa fusion) ----
        # Conform ASSIGN Eq. 5, detectorul primeste x^e_t imbogatit (fused),
        # nu ROI brut.
        x_proj_bms = x_fused  # (B, M, S, D) — deja proiectat de FusionLevelGraph

        # ---- 4. Spatial Message Passing per frame  (Eq. 2–4) ----
        m_inter_f_list, m_intra_f_list = [], []
        for s in range(S):
            combined_s = torch.cat([x_proj_bms[:, :, s, :],
                                     h_f_fused[:, :, s, :]], dim=-1)  # (B, M, 2D)
            mi, ma = self.msg_passing(combined_s, entity_types)
            m_inter_f_list.append(mi)
            m_intra_f_list.append(ma)
        m_inter_f = torch.stack(m_inter_f_list, dim=2)   # (B, M, S, 2D)
        m_intra_f = torch.stack(m_intra_f_list, dim=2)   # (B, M, S, 2D)

        # ---- 5. Segment Boundary Detector  (Eq. 5) ----
        if training_stage == 1:
            # Stage 1: toate frame-urile sunt granite (dense)
            u_soft_bms = torch.ones(B, M, S, device=device)
            u_hard_bms = torch.ones(B, M, S, device=device)
        else:
            # Stage 2: Gumbel-Softmax
            # Input: [x_fused(D); h_f(D); m_intra(2D); m_inter(2D)] = 6D
            det_in = torch.cat([
                x_proj_bms,   # (B, M, S, D) — fused features
                h_f_fused,    # (B, M, S, D)
                m_intra_f,    # (B, M, S, 2D)
                m_inter_f,    # (B, M, S, 2D)
            ], dim=-1)        # (B, M, S, 6D)

            det_flat = det_in.reshape(B * M * S, -1)   # (B*M*S, 6D)
            u_soft_flat, u_hard_flat = self.boundary_detector(det_flat)
            u_soft_bms = u_soft_flat.reshape(B, M, S)
            u_hard_bms = u_hard_flat.reshape(B, M, S)

        # ---- 6. Segment-level Layer  (Eq. 6–10) ----
        _, seg_logits_4d = self.segment_layer(
            h_f=h_f_fused,
            m_inter_f=m_inter_f,
            m_intra_f=m_intra_f,
            u_hard=u_hard_bms,
            entity_types=entity_types,
        )   # (B, M, S, C)

        # ---- Flatten (B, M, S, *) -> (B, N, *)  cu N=S*M ----
        def bms_to_bn(t):
            # (B, M, S, K) -> (B, S, M, K) -> (B, S*M, K)
            if t.dim() == 4:
                return t.permute(0, 2, 1, 3).reshape(B, S * M, -1)
            # (B, M, S) -> (B, S*M)
            return t.permute(0, 2, 1).reshape(B, S * M)

        return {
            "z":              bms_to_bn(z_intermediate),  # (B, N, D)
            "frame_logits":   bms_to_bn(fl_4d),           # (B, N, C)
            "segment_logits": bms_to_bn(seg_logits_4d),   # (B, N, C)
            "u_soft":         bms_to_bn(u_soft_bms),      # (B, N)
        }