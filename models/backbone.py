"""
backbone.py
Implementarea backbone-ului 2G-GCN din paper.

Arhitectura (bazata pe ASSIGN + 2G-GCN):
  1. Frame-level BiRNN  - modeleaza dinamica temporala per entitate
  2. Graph layer        - modeleaza relatiile spatiale intre entitati
  3. Segment-level BiRNN - clasificare segment dupa segmentare temporala
  4. Fusion graph       - combina features vizuale + geometrice

Notatie din paper:
  S = numar de frame-uri
  M = numar de entitati per frame (persoane + obiecte)
  N = S * M = numar total de entitati in video
  C = numar de clase
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Temporal Segment Pooling
# ---------------------------------------------------------------------------

def temporal_segment_pooling(
    z: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """
    Grupeaza frame-urile consecutive cu aceeasi eticheta/predictie in segmente
    si inlocuieste feature-urile fiecarui frame cu media segmentului sau.

    Echivalentul simplificat al segmentarii temporale din ASSIGN (Fig. 2):
    in loc de un discriminator de segmente antrenat separat, folosim
    predictiile frame-level ca granita de segment — self-consistent
    intre antrenare si inferenta si se imbunatateste odata cu modelul.

    Args:
        z:      (B, M, S, hidden_dim) - frame features per entitate
        labels: (B, M, S) - predictii sau etichete per frame per entitate (long)
    Returns:
        z_seg: (B, M, S, hidden_dim) - aceeasi forma, features mediate per segment
    """
    B, M, S, _ = z.shape
    z_seg = z.clone()

    for b in range(B):
        for m in range(M):
            feat = z[b, m]      # (S, D)
            lbl  = labels[b, m] # (S,)

            # Detecteaza granitele (unde eticheta se schimba)
            boundaries = [0]
            for t in range(1, S):
                if lbl[t] != lbl[t - 1]:
                    boundaries.append(t)
            boundaries.append(S)

            # Mean-pooling per segment
            for i in range(len(boundaries) - 1):
                s0, s1 = boundaries[i], boundaries[i + 1]
                z_seg[b, m, s0:s1] = feat[s0:s1].mean(dim=0)

    return z_seg


# ---------------------------------------------------------------------------
# Graph Convolution Layer
# ---------------------------------------------------------------------------

class GraphConvolution(nn.Module):
    """
    Strat simplu de convolutie pe graf (GCN).
    H' = sigma(D^{-1/2} A D^{-1/2} H W)
    """

    def __init__(self, in_dim: int, out_dim: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:   (B, N, in_dim)  - features noduri
            adj: (B, N, N)       - matrice de adiacenta normalizata
        Returns:
            (B, N, out_dim)
        """
        support = self.linear(x)           # (B, N, out_dim)
        output = torch.bmm(adj, support)   # (B, N, out_dim)
        return F.relu(output)


def build_adjacency(num_entities: int, device: torch.device) -> torch.Tensor:
    """
    Construieste o matrice de adiacenta fully-connected normalizata.
    In lipsa keypoints/skeleton, folosim un graf complet simplu.
    Pentru 2G-GCN complet ai nevoie de keypoints din dataset.

    Args:
        num_entities: numarul de noduri M
        device: torch device
    Returns:
        (M, M) matrice normalizata
    """
    adj = torch.ones(num_entities, num_entities, device=device)
    # Normalizare D^{-1/2} A D^{-1/2}
    degree = adj.sum(dim=1, keepdim=True).clamp(min=1)
    adj = adj / degree
    return adj


# ---------------------------------------------------------------------------
# Frame-level BiRNN
# ---------------------------------------------------------------------------

class FrameLevelBiRNN(nn.Module):
    """
    Primul nivel BiRNN din ASSIGN/2G-GCN.
    Modeleaza dinamica temporala si produce features intermediare Z
    (folosite de VHOIP pentru MI si prototyping).

    Input:  features ROI per frame per entitate
    Output: reprezentari temporale + frame-level logits
    """

    def __init__(
        self,
        input_dim: int = 2048,
        hidden_dim: int = 256,
        num_classes: int = 10,
        num_layers: int = 1,
        dropout: float = 0.3,
    ):
        super().__init__()

        # Proiectie initiala (reduce 2048 -> hidden_dim)
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # BiRNN temporal
        self.birnn = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,   # bidiectional => output = hidden_dim
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Classifier frame-level (pentru L_Seg)
        self.frame_classifier = nn.Linear(hidden_dim, num_classes)

        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, S, input_dim) - features ROI per frame
                B = batch, S = frames per entitate
        Returns:
            z:            (B, S, hidden_dim) - features intermediare (Z din paper)
            frame_logits: (B, S, num_classes) - predictii frame-level
        """
        # Proiectie initiala
        h = self.input_proj(x)          # (B, S, hidden_dim)

        # BiRNN
        z, _ = self.birnn(h)            # (B, S, hidden_dim)
        z = self.dropout(z)

        # Frame-level predictions (pentru L_Seg)
        frame_logits = self.frame_classifier(z)   # (B, S, num_classes)

        return z, frame_logits


# ---------------------------------------------------------------------------
# Segment-level BiRNN
# ---------------------------------------------------------------------------

class SegmentLevelBiRNN(nn.Module):
    """
    Al doilea nivel BiRNN - opereaza pe segmente (nu frame-uri).
    Produce clasificarea finala a interactiunilor.

    In ASSIGN, segmentarea e facuta cu un discriminator de segmente.
    Aici simplificam: lucram pe reprezentarile mediate per segment.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_classes: int = 10,
        num_layers: int = 1,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.birnn = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.segment_classifier = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            z: (B, N, hidden_dim) - features agregate per segment
        Returns:
            h:              (B, N, hidden_dim) - features segment
            segment_logits: (B, N, num_classes)
        """
        h, _ = self.birnn(z)            # (B, N, hidden_dim)
        h = self.dropout(h)
        segment_logits = self.segment_classifier(h)   # (B, N, num_classes)
        return h, segment_logits


# ---------------------------------------------------------------------------
# Fusion Graph (2G-GCN specific) — scaled dot-product attention
# ---------------------------------------------------------------------------

class FusionGraphLayer(nn.Module):
    """
    Stratul de fuziune din 2G-GCN cu scaled dot-product attention (Fig. 2 din paper).
    Inlocuieste GCN-ul simplu cu un mecanism de atentie care invata adaptiv
    relatiile intre entitati (human-object, human-human) per frame.

    A_t = softmax(Q * K^T / sqrt(d_k))
    out = A_t * V + x  (residual)
    """

    def __init__(self, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()

        self.d_k = hidden_dim

        self.W_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_v = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, M, hidden_dim) - entity features per frame
        Returns:
            (B, M, hidden_dim)
        """
        Q = self.W_q(x)   # (B, M, d_k)
        K = self.W_k(x)   # (B, M, d_k)
        V = self.W_v(x)   # (B, M, d_k)

        # Scaled dot-product attention
        scale = self.d_k ** 0.5
        attn = torch.bmm(Q, K.transpose(1, 2)) / scale   # (B, M, M)
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.bmm(attn, V)          # (B, M, d_k)
        out = self.out_proj(out)          # (B, M, hidden_dim)

        # Residual + LayerNorm
        out = self.norm(out + x)
        return out


# ---------------------------------------------------------------------------
# Backbone complet 2G-GCN
# ---------------------------------------------------------------------------

class Backbone2GGCN(nn.Module):
    """
    Backbone-ul complet 2G-GCN / ASSIGN adaptat pentru VHOIP.

    Pipeline per video:
      1. frame_birnn:   (B, S, 2048) -> Z (B, S, hidden_dim) + frame_logits
      2. fusion_graph:  modeleaza relatii spatiale intre entitati per frame
      3. segment_birnn: (B, N, hidden_dim) -> segment_logits

    Nota: In implementarea completa, Z de la frame_birnn e cel folosit de
    CLIP modules pentru prototyping si MI loss.
    """

    def __init__(
        self,
        input_dim: int = 2048,
        hidden_dim: int = 256,
        num_classes: int = 10,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.frame_birnn = FrameLevelBiRNN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            num_layers=num_layers,
            dropout=dropout,
        )

        self.fusion_graph = FusionGraphLayer(
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        self.segment_birnn = SegmentLevelBiRNN(
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            num_layers=num_layers,
            dropout=dropout,
        )

        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

    def forward(self, roi_features: torch.Tensor) -> dict:
        """
        Args:
            roi_features: (B, S, M, input_dim)
                B = batch size
                S = number of frames
                M = number of entities per frame
                input_dim = 2048 (ROI pooling dim)
            adj: (B, M, M) optional adjacency matrix

        Returns:
            dict cu:
                z:              (B, N, hidden_dim) - features intermediare (primul nivel)
                frame_logits:   (B, N, num_classes) - predictii frame-level
                segment_logits: (B, N, num_classes) - predictii segment-level
            unde N = S * M
        """
        B, S, M, D = roi_features.shape

        # --- Pas 1: Frame-level BiRNN per entitate ---
        # Reshape: trateaza fiecare entitate independent prin timp
        # (B, S, M, D) -> (B*M, S, D)
        x = roi_features.permute(0, 2, 1, 3)          # (B, M, S, D)
        x = x.reshape(B * M, S, D)                    # (B*M, S, D)

        z, frame_logits = self.frame_birnn(x)          # (B*M, S, hidden_dim)

        # Reshape inapoi
        z = z.reshape(B, M, S, -1)                    # (B, M, S, hidden)
        frame_logits = frame_logits.reshape(B, M, S, -1)

        # --- Pas 2: Graph per frame (relatii spatiale) ---
        # Aplica GCN pe fiecare frame in parte
        z_graph_list = []
        for s in range(S):
            z_s = z[:, :, s, :]                       # (B, M, hidden_dim)
            z_s = self.fusion_graph(z_s)               # (B, M, hidden_dim)
            z_graph_list.append(z_s)

        z_graph = torch.stack(z_graph_list, dim=2)    # (B, M, S, hidden_dim)

        # --- Pas 2.5: Temporal segment pooling ---
        # Foloseste predictiile frame-level ca granita de segment.
        # Self-consistent intre antrenare si inferenta; se imbunatateste odata cu modelul.
        frame_preds = frame_logits.argmax(dim=-1)      # (B, M, S)
        z_graph = temporal_segment_pooling(z_graph, frame_preds)

        # --- Pas 3: Segment-level BiRNN ---
        # Flatten M si S -> N = M*S entitati
        # (B, M, S, hidden) -> (B, N, hidden)
        z_flat = z_graph.permute(0, 2, 1, 3)          # (B, S, M, hidden)
        N = S * M
        z_flat = z_flat.reshape(B, N, -1)             # (B, N, hidden_dim)

        _, segment_logits = self.segment_birnn(z_flat) # (B, N, num_classes)

        # Features intermediare Z (primul nivel, inainte de graph)
        # Acestea sunt folosite de CLIP modules (Z din paper)
        z_intermediate = z.permute(0, 2, 1, 3)        # (B, S, M, hidden)
        z_intermediate = z_intermediate.reshape(B, N, -1)  # (B, N, hidden_dim)

        # Frame logits flatten
        frame_logits = frame_logits.permute(0, 2, 1, 3)   # (B, S, M, C)
        frame_logits = frame_logits.reshape(B, N, -1)      # (B, N, num_classes)

        return {
            "z": z_intermediate,           # (B, N, hidden_dim) - pentru CLIP modules
            "frame_logits": frame_logits,  # (B, N, num_classes) - pentru L_Seg
            "segment_logits": segment_logits,  # (B, N, num_classes) - pentru L_Label
        }