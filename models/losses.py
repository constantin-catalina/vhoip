"""
losses.py
Loss-urile VHOIP conform paperurilor ASSIGN + VHOIP:

  L_total = L_Label + lambda1*L_Seg + lambda2*L_MI + lambda3*L_Cos  (VHOIP Eq. 4)

  L_Label — NLL pe predictii segment-level, per frame  (ASSIGN Eq. 12)
  L_Seg   — BCE intre u_soft si ground-truth de granita smoothed  (ASSIGN Eq. 11)
  L_MI    — Mutual Information loss cu discriminator DGI  (VHOIP Eq. 2)
  L_Cos   — Cosine similarity loss intre Z' si T  (VHOIP Eq. 3)

L_Seg (ASSIGN §3.5, Eq. 11):
  Supervizeaza segmentarea cu un semnal binar smoothed.
  Ground-truth de granita: u^e_t = 1 la ultimul frame al unui segment, 0 altfel.
  Smoothed cu filtru Gaussian (sigma=4) pentru a softiza tranzitia.
  Compara cu u_soft (iesirea reala a Gumbel-Softmax din SegmentBoundaryDetector).
  BCE(u_hat_smooth, u_smooth_gt) medie pe toate entitatile si frame-urile.

  La Stage 1 (training_stage=1): L_Seg este oprit (lambda1=0 sau nu se calculeaza).
  La Stage 2 (training_stage=2): L_Seg este activ.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional


# ---------------------------------------------------------------------------
# Utilitare pentru L_Seg
# ---------------------------------------------------------------------------

def compute_boundary_gt(
    frame_labels: torch.Tensor,
    sigma: float = 4.0,
) -> torch.Tensor:
    """
    Calculeaza ground-truth-ul de granita de segment (ASSIGN Eq. 11).

    Un frame t este granita de segment daca eticheta se schimba dupa el:
        u^e_t = 1  daca frame_labels[t] != frame_labels[t+1]  (sau t = T-1)
        u^e_t = 0  altfel

    Acest semnal binar este smoothed cu un filtru Gaussian (sigma=4)
    pentru a produce u_smooth_gt — tinta pentru BCE cu u_soft.

    Args:
        frame_labels: (B, N) long — etichete per frame-entitate flatten
                      N = S * M, -1 = ignorat (padding)
        sigma:        deviatie standard filtru Gaussian (din ASSIGN)
    Returns:
        boundary_gt_smooth: (B, N) float — semnal smoothed in [0, 1]
    """
    B, N = frame_labels.shape
    device = frame_labels.device

    # Granita binara bruta: 1 la ultimul frame al unui segment
    boundary_gt = torch.zeros(B, N, dtype=torch.float32, device=device)

    for b in range(B):
        labels_b = frame_labels[b]   # (N,)
        for t in range(N - 1):
            if labels_b[t] == -1 or labels_b[t + 1] == -1:
                continue
            if labels_b[t] != labels_b[t + 1]:
                boundary_gt[b, t] = 1.0
        # Ultimul frame valid este intotdeauna granita
        last_valid = (labels_b != -1).nonzero(as_tuple=True)[0]
        if last_valid.numel() > 0:
            boundary_gt[b, last_valid[-1]] = 1.0

    # Smoothing Gaussian pe axa temporala (axa N)
    # Cream kernel Gaussian 1D si aplicam convolutie padding='same'
    boundary_gt_smooth = _gaussian_smooth_1d(boundary_gt, sigma)

    return boundary_gt_smooth


def _gaussian_smooth_1d(x: torch.Tensor, sigma: float = 4.0) -> torch.Tensor:
    """
    Aplica smoothing Gaussian 1D pe ultima dimensiune.
    Args:
        x:     (B, N) float
        sigma: deviatie standard (din ASSIGN: sigma=4)
    Returns: (B, N) float, valori in [0, 1] (clampate)
    """
    # Dimensiunea kernel-ului: 6*sigma + 1 (captureaza 99.7% din distributie)
    kernel_size = int(6 * sigma) + 1
    if kernel_size % 2 == 0:
        kernel_size += 1

    # Kernel Gaussian
    half = kernel_size // 2
    coords = torch.arange(kernel_size, dtype=torch.float32, device=x.device) - half
    kernel = torch.exp(-0.5 * (coords / sigma) ** 2)
    kernel = kernel / kernel.sum()                          # normalizeaza

    # Convolutie 1D cu padding 'same'
    # x: (B, N) -> (B, 1, N) pentru F.conv1d
    x_3d = x.unsqueeze(1)                                   # (B, 1, N)
    kernel_3d = kernel.unsqueeze(0).unsqueeze(0)             # (1, 1, K)
    smoothed = F.conv1d(x_3d, kernel_3d, padding=half)      # (B, 1, N)
    smoothed = smoothed.squeeze(1)                           # (B, N)

    return smoothed.clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# Componente de loss
# ---------------------------------------------------------------------------

class SegmentationLoss(nn.Module):
    """
    L_Seg — Segmentation loss conform ASSIGN Eq. 11.

    BCE intre u_soft (iesirea Gumbel-Softmax) si ground-truth-ul de granita
    smoothed cu filtru Gaussian (sigma=4).

    L_Seg = (1/T) * sum_t [ (1/N) * sum_e BCE(u_hat^e_t, u_smooth^e_t) ]
          = mean_over_all_valid_positions BCE(u_soft, boundary_gt_smooth)

    La Stage 1 (training_stage=1): L_Seg = 0 (nu se calculeaza).
    La Stage 2 (training_stage=2): L_Seg activ.

    Args:
        sigma: deviatie standard filtru Gaussian pt smoothing (default=4, din ASSIGN)
    """

    def __init__(self, sigma: float = 4.0):
        super().__init__()
        self.sigma = sigma

    def forward(
        self,
        u_soft: torch.Tensor,          # (B, N) — iesirea detector (real-valued)
        frame_labels: torch.Tensor,    # (B, N) — etichete frame (-1=ignorat)
        training_stage: int = 2,
    ) -> torch.Tensor:
        """
        Args:
            u_soft:         (B, N) — semnal binar relaxat din SegmentBoundaryDetector
            frame_labels:   (B, N) — etichete frame-level cu -1 pentru padding
            training_stage: 1 = loss oprit (Stage 1 ASSIGN), 2 = loss activ
        Returns:
            scalar loss
        """
        if training_stage == 1:
            return torch.tensor(0.0, device=u_soft.device, requires_grad=False)

        # Calculeaza ground-truth smoothed
        with torch.no_grad():
            boundary_smooth = compute_boundary_gt(frame_labels, self.sigma)

        # Masca pozitiile valide (nu padding)
        valid_mask = (frame_labels != -1).float()   # (B, N)

        # BCEWithLogits este sigur sub autocast; convertim probabilitatile deja
        # produse de detector in logits echivalenti pentru a pastra aceeasi tinta.
        u_logits = torch.logit(u_soft.float().clamp(1e-6, 1 - 1e-6))
        bce_per_pos = F.binary_cross_entropy_with_logits(
            u_logits,
            boundary_smooth.float(),
            reduction='none',
        )

        # Media doar pe pozitiile valide
        loss = (bce_per_pos * valid_mask).sum() / valid_mask.sum().clamp(min=1)
        return loss


class MutualInformationLoss(nn.Module):
    """
    L_MI — Mutual Information loss (VHOIP Eq. 2).

    BCE intre scorurile discriminatorului si target-urile binare one-hot.
    Maximizeaza MI intre Z si G (reprezentarile globale integrate).
    """

    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            scores: (B, N, C) sau (N, C) — logit-uri brute din discriminator
            labels: (B, N) sau (N,) — etichete reale (long)
        Returns:
            scalar loss
        """
        # Flatten la (N_total, C)
        if scores.dim() == 3:
            B, N, C = scores.shape
            scores = scores.reshape(B * N, C)
            labels = labels.reshape(B * N)
        else:
            C = scores.shape[1]

        # Masca pozitii valide (-1 = ignorat)
        valid = labels != -1
        if not valid.any():
            return torch.tensor(0.0, device=scores.device)
        scores = scores[valid]
        labels = labels[valid]

        N_v, C = scores.shape
        safe_labels = labels.clamp(0, C - 1)
        targets = torch.zeros(N_v, C, device=scores.device)
        targets.scatter_(1, safe_labels.unsqueeze(1), 1.0)

        return self.bce(scores, targets)


class CosineSimilarityLoss(nn.Module):
    """
    L_Cos — Cosine similarity loss (VHOIP Eq. 3).

    CE intre similaritatile cosinus Z' vs T si etichetele reale.
    """

    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, similarities: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            similarities: (B, N, C) sau (N, C)
            labels:       (B, N) sau (N,) — long
        Returns:
            scalar loss
        """
        if similarities.dim() == 3:
            B, N, C = similarities.shape
            similarities = similarities.reshape(B * N, C)
            labels       = labels.reshape(B * N)
        return self.ce(similarities, labels)


# ---------------------------------------------------------------------------
# Loss total VHOIP
# ---------------------------------------------------------------------------

class VHOIPLoss(nn.Module):
    """
    Loss total VHOIP (Eq. 4 din paper):

        L = L_Label + lambda1*L_Seg + lambda2*L_MI + lambda3*L_Cos

    L_Label: NLL pe segment logits per frame (ASSIGN Eq. 12)
    L_Seg:   BCE pe boundary signal (ASSIGN Eq. 11) — activ doar la Stage 2
    L_MI:    MI loss (VHOIP Eq. 2)
    L_Cos:   Cosine loss (VHOIP Eq. 3)

    Args:
        lambda1: coeficient L_Seg  (default=1.0)
        lambda2: coeficient L_MI   (default=0.5)
        lambda3: coeficient L_Cos  (default=0.5)
        seg_sigma: sigma filtru Gaussian pt L_Seg (default=4.0, din ASSIGN)
    """

    def __init__(
        self,
        lambda1: float = 1.0,
        lambda2: float = 0.5,
        lambda3: float = 0.5,
        seg_sigma: float = 4.0,
    ):
        super().__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3

        self.l_label = nn.CrossEntropyLoss(ignore_index=-1)
        self.l_seg   = SegmentationLoss(sigma=seg_sigma)
        self.l_mi    = MutualInformationLoss()
        self.l_cos   = CosineSimilarityLoss()

    def forward(
        self,
        segment_logits:   torch.Tensor,   # (B, N, C)
        frame_logits:     torch.Tensor,   # (B, N, C)  — neutilizat direct, mentinut pt compat.
        u_soft:           torch.Tensor,   # (B, N)     — boundary signal
        mi_scores:        torch.Tensor,   # (B, N, C)
        cos_similarities: torch.Tensor,   # (B, N, C)
        segment_labels:   torch.Tensor,   # (B, N)     — etichete segment (-1=ignorat)
        frame_labels:     torch.Tensor,   # (B, N)     — etichete frame (-1=ignorat)
        training_stage:   int = 2,
    ) -> dict:
        """
        Returns:
            dict cu loss-ul total si componentele (pentru logging).
        """
        B, N, C = segment_logits.shape

        # L_Label: NLL per frame pe segment logits (ASSIGN Eq. 12)
        # Flatten (B, N, C) -> (B*N, C) si (B, N) -> (B*N,)
        l_label = self.l_label(
            segment_logits.reshape(B * N, C),
            segment_labels.reshape(B * N),
        )

        # L_Seg: BCE pe boundary signal (ASSIGN Eq. 11)
        # Activ doar la Stage 2; Stage 1 returneaza 0
        l_seg = self.l_seg(u_soft, frame_labels, training_stage)

        # L_MI: MI loss (VHOIP Eq. 2)
        l_mi = self.l_mi(mi_scores, segment_labels)

        # L_Cos: Cosine loss (VHOIP Eq. 3)
        l_cos = self.l_cos(cos_similarities, segment_labels)

        total = (
            l_label
            + self.lambda1 * l_seg
            + self.lambda2 * l_mi
            + self.lambda3 * l_cos
        )

        return {
            "total":   total,
            "l_label": l_label.detach(),
            "l_seg":   l_seg.detach() if isinstance(l_seg, torch.Tensor) else l_seg,
            "l_mi":    l_mi.detach(),
            "l_cos":   l_cos.detach(),
        }