"""
discriminator.py
Discriminatorul pentru Mutual Information loss (Eq. 1 din VHOIP paper).

Formula din paper (Eq. 1):
    y_hat_{i,k} = sigma1( MLP(sigma1(g_k))  *  MLP(sigma2(||z_i||_2)) )

unde:
    sigma1 = sigmoid
    sigma2 = PReLU
    ||z_i||_2 = z_i L2-normalizat   <-- IMPORTANT: normare explicita
    g_k   = componenta k din G (reprezentarea globala integrata, dim=clip_dim)
    z_i   = feature intermediar al entitatii i (din primul nivel BiRNN, dim=hidden_dim)

Discriminatorul primeste toate N*C perechi (z_i, g_k) si decide:
    - pozitiv (1): z_i si g_k sunt din aceeasi clasa
    - negativ (0): z_i si g_k sunt din clase diferite

NOTA despre implementare:
    Sigmoid-ul final din Eq. 1 NU este aplicat explicit in forward().
    In schimb, BCEWithLogitsLoss din losses.py aplica sigmoid intern
    pentru stabilitate numerica superioara (echivalent matematic).
    Atentie: nu aplicati sigmoid si in forward() si in loss — ar fi dublu.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class Discriminator(nn.Module):
    """
    Discriminator binar din DGI adaptat pentru VHOIP (Eq. 1 din paper).

    Arhitectura conforma cu Eq. 1:
        Bratul global:  g_k  (clip_dim)
                         -> sigma1 = Sigmoid
                         -> MLP: Linear(clip_dim -> feature_dim)
                         -> vector h_g  (feature_dim)

        Bratul local:   z_i  (hidden_dim)
                         -> L2 normalizare: ||z_i||_2  (conform Eq. 1)
                         -> sigma2 = PReLU
                         -> MLP: Linear(hidden_dim -> feature_dim)
                         -> vector h_z  (feature_dim)

        Output:         dot product h_g * h_z  ->  scor binar (raw logit)
                        (sigmoid aplicat implicit de BCEWithLogitsLoss)

    Args:
        feature_dim: dimensiunea Z (hidden_dim din backbone, ex. 256)
        global_dim:  dimensiunea G (CLIP feature dim, ex. 512)
        mlp_dim:     dimensiunea intermediara a MLP-urilor (default: feature_dim)
    """

    def __init__(
        self,
        feature_dim: int = 256,
        global_dim: int = 512,
        mlp_dim: Optional[int] = None,
    ):
        super().__init__()

        if mlp_dim is None:
            mlp_dim = feature_dim

        # -----------------------------------------------------------------------
        # Bratul global: sigma1(g_k) -> MLP
        # Ordinea din Eq. 1: INTAI sigmoid, APOI MLP
        # -----------------------------------------------------------------------
        self.global_branch = nn.Sequential(
            nn.Sigmoid(),                          # sigma1 din Eq. 1
            nn.Linear(global_dim, mlp_dim),        # MLP
        )

        # -----------------------------------------------------------------------
        # Bratul local: ||z_i||_2 -> sigma2 -> MLP
        # L2-normalizarea este aplicata in forward() inainte de acest branch.
        # -----------------------------------------------------------------------
        self.prelu = nn.PReLU()                    # sigma2 din Eq. 1
        self.local_mlp = nn.Linear(feature_dim, mlp_dim)

    def forward(
        self,
        z: torch.Tensor,
        G: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculeaza scorurile pentru toate perechile (z_i, g_k).

        Args:
            z: (B, N, feature_dim) — features intermediare Z din backbone
               (primul nivel BiRNN, INAINTE de proiectia MLP spre Z')
            G: (C, global_dim)     — reprezentarile globale integrate
               (prototipuri CLIP sau EMA G)
        Returns:
            scores: (B, N, C) — logit-uri brute per pereche (z_i, g_k)
                    (fara sigmoid — BCEWithLogitsLoss aplica sigmoid intern)
        """
        B, N, _ = z.shape
        C = G.shape[0]

        # Sanitizeaza NaN-urile inainte de orice calcul.
        # Daca gradientii au explodat upstream, z sau G pot contine NaN,
        # ceea ce ar face ca toate scorurile si loss-ul MI sa devina NaN.
        # nan_to_num(0.0) trateaza NaN ca vector zero — discriminatorul
        # va produce scoruri neutrale (aproape de 0) pentru aceste intrari,
        # ceea ce este mai bine decat propagarea NaN prin tot modelul.
        z = torch.nan_to_num(z, nan=0.0)
        G = torch.nan_to_num(G, nan=0.0)

        # -----------------------------------------------------------------------
        # Bratul local (Eq. 1: sigma2(||z_i||_2) -> MLP)
        # Pas 1: L2-normalizare explicita conform ||z_i||_2 din Eq. 1
        # -----------------------------------------------------------------------
        z_norm = F.normalize(z, p=2, dim=-1)       # (B, N, feature_dim)

        # Pas 2: PReLU (sigma2)
        z_activated = self.prelu(z_norm)            # (B, N, feature_dim)

        # Pas 3: MLP
        h_z = self.local_mlp(z_activated)           # (B, N, mlp_dim)

        # -----------------------------------------------------------------------
        # Bratul global (Eq. 1: sigma1(g_k) -> MLP)
        # G: (C, global_dim) — trebuie extins la (1, C, mlp_dim) pentru broadcasting
        # -----------------------------------------------------------------------
        # self.global_branch: Sigmoid -> Linear
        # Input: (C, global_dim), Output: (C, mlp_dim)
        h_g = self.global_branch(G)                 # (C, mlp_dim)

        # -----------------------------------------------------------------------
        # Dot product (Eq. 1: h_g_k * h_z_i) pentru toate perechile (i, k)
        # h_z: (B, N, mlp_dim)
        # h_g: (C, mlp_dim)
        # scores[b, n, c] = h_z[b, n, :] . h_g[c, :]
        # -----------------------------------------------------------------------
        # h_g: (C, mlp_dim) -> (1, mlp_dim, C) pentru einsum cu h_z
        h_g_T = h_g.t().unsqueeze(0)               # (1, mlp_dim, C)
        scores = torch.bmm(
            h_z,                                    # (B, N, mlp_dim)
            h_g_T.expand(B, -1, -1),               # (B, mlp_dim, C)
        )                                           # (B, N, C)

        # Returnam logit-uri brute (sigmoid aplicat de BCEWithLogitsLoss)
        return scores