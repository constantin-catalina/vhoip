"""
discriminator.py
Discriminatorul pentru Mutual Information loss (Eq. 1 din paper).

y_hat_{i,k} = sigma1(MLP(sigma1(g_k)) * MLP(sigma2(||z_i||_2)))

unde:
  sigma1 = sigmoid
  sigma2 = PReLU
  g_k    = componenta k din G (reprezentarea globala integrata)
  z_i    = feature intermediar al entitatii i (din primul nivel BiRNN)

Discriminatorul primeste toate N*C perechi (z_i, g_k) si decide:
  - pozitiv (1): z_i si g_k sunt din aceeasi clasa
  - negativ (0): z_i si g_k sunt din clase diferite
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):
    """
    Discriminator binar din DGI adaptat pentru VHOIP (Eq. 1 din paper).

    Arhitectura:
      Bratul global:  g_k  -> sigmoid -> MLP -> vector
      Bratul local:   z_i  -> L2 norm -> PReLU -> MLP -> vector
      Output:         dot product -> scor binar per pereche (z_i, g_k)
    """

    def __init__(self, feature_dim: int = 256, global_dim: int = 512):
        """
        Args:
            feature_dim: dimensiunea Z (output backbone, hidden_dim)
            global_dim:  dimensiunea G (CLIP feature dim, 512)
        """
        super().__init__()

        # Bratul pentru reprezentarile globale g_k
        # g_k are dimensiunea CLIP (512), il proiectam la feature_dim
        self.global_mlp = nn.Sequential(
            nn.Linear(global_dim, feature_dim),
            nn.Sigmoid(),   # sigma1 din paper
        )

        # Bratul pentru features locale z_i
        self.prelu = nn.PReLU()   # sigma2 din paper
        self.local_mlp = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
        )

        # Sigmoid final pentru scorul binar
        self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        z: torch.Tensor,
        G: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculeaza scorurile pentru toate perechile (z_i, g_k).

        Args:
            z: (B, N, feature_dim) - features intermediare din backbone (Z)
            G: (C, global_dim)     - reprezentarile globale integrate

        Returns:
            scores: (B, N, C) - scor binar pentru fiecare pereche (z_i, g_k)
        """
        B, N, D = z.shape
        C = G.shape[0]

        # --- Bratul global: procesa fiecare g_k ---
        # (C, global_dim) -> (C, feature_dim)
        g_proc = self.global_mlp(G)   # (C, feature_dim)

        # --- Bratul local: procesa fiecare z_i ---
        # L2 normalizare (||z_i||_2 din paper)
        z_norm = F.normalize(z, dim=-1)           # (B, N, feature_dim)
        z_proc = self.prelu(z_norm)               # sigma2
        z_proc = self.local_mlp(z_proc)           # (B, N, feature_dim)

        # --- Dot product pentru toate perechile ---
        # z_proc: (B, N, D) x g_proc: (C, D).T -> (B, N, C)
        scores = torch.einsum("bnd,cd->bnc", z_proc, g_proc)
        scores = self.sigmoid(scores)             # (B, N, C)

        return scores