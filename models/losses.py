"""
losses.py
Loss-urile VHOIP conform paperurilor ASSIGN + VHOIP:

  L_total = L_Label + L_Ant + lambda1*L_Seg + lambda2*L_MI + lambda3*L_Cos  (VHOIP Eq. 4)

  C6b adds:
  L_total += lambda4 * L_PromptReg

  L_PromptReg = 1 - mean cosine similarity(T_learned, T_frozen)
  This anchors the learnable context vectors to the original CLIP embedding
  space. Weight lambda4=0.1 is light enough that the prompts can still
  adapt but cannot drift so far that CLIP generalization is lost.
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
    B, N = frame_labels.shape
    device = frame_labels.device

    boundary_gt = torch.zeros(B, N, dtype=torch.float32, device=device)

    for b in range(B):
        labels_b = frame_labels[b]
        for t in range(N - 1):
            if labels_b[t] == -1 or labels_b[t + 1] == -1:
                continue
            if labels_b[t] != labels_b[t + 1]:
                boundary_gt[b, t] = 1.0
        last_valid = (labels_b != -1).nonzero(as_tuple=True)[0]
        if last_valid.numel() > 0:
            boundary_gt[b, last_valid[-1]] = 1.0

    boundary_gt_smooth = _gaussian_smooth_1d(boundary_gt, sigma)
    return boundary_gt_smooth


def _gaussian_smooth_1d(x: torch.Tensor, sigma: float = 4.0) -> torch.Tensor:
    kernel_size = int(6 * sigma) + 1
    if kernel_size % 2 == 0:
        kernel_size += 1

    half = kernel_size // 2
    coords = torch.arange(kernel_size, dtype=torch.float32, device=x.device) - half
    kernel = torch.exp(-0.5 * (coords / sigma) ** 2)
    kernel = kernel / kernel.sum()

    x_3d = x.unsqueeze(1)
    kernel_3d = kernel.unsqueeze(0).unsqueeze(0)
    smoothed = F.conv1d(x_3d, kernel_3d, padding=half)
    smoothed = smoothed.squeeze(1)

    return smoothed.clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# Componente de loss
# ---------------------------------------------------------------------------

class SegmentationLoss(nn.Module):
    def __init__(self, sigma: float = 4.0):
        super().__init__()
        self.sigma = sigma

    def forward(
        self,
        u_soft: torch.Tensor,
        frame_labels: torch.Tensor,
        training_stage: int = 2,
    ) -> torch.Tensor:
        if training_stage == 1:
            return torch.tensor(0.0, device=u_soft.device, requires_grad=False)

        with torch.no_grad():
            boundary_smooth = compute_boundary_gt(frame_labels, self.sigma)

        valid_mask = (frame_labels != -1).float()

        u_logits = torch.logit(u_soft.float().clamp(1e-6, 1 - 1e-6))
        bce_per_pos = F.binary_cross_entropy_with_logits(
            u_logits,
            boundary_smooth.float(),
            reduction='none',
        )

        loss = (bce_per_pos * valid_mask).sum() / valid_mask.sum().clamp(min=1)
        return loss


class MutualInformationLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if scores.dim() == 3:
            B, N, C = scores.shape
            scores = scores.reshape(B * N, C)
            labels = labels.reshape(B * N)
        else:
            C = scores.shape[1]

        valid = labels != -1
        if not valid.any():
            return torch.tensor(0.0, device=scores.device)
        scores = scores[valid]
        labels = labels[valid]

        N_v, C = scores.shape
        safe_labels = labels.clamp(0, C - 1)
        targets = torch.zeros(N_v, C, device=scores.device)
        targets.scatter_(1, safe_labels.unsqueeze(1), 1.0)

        if torch.isnan(scores).any():
            return torch.tensor(0.0, device=scores.device, requires_grad=True)

        return self.bce(scores, targets)


class CosineSimilarityLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=-1, label_smoothing=0.1)

    def forward(self, similarities: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
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
    Loss total VHOIP (Eq. 4 din paper) + C6b prompt regularization:

        L = L_Label + lambda_ant*L_Ant + lambda1*L_Seg
              + lambda2*L_MI + lambda3*L_Cos
              + lambda4*L_PromptReg   (C6b only; 0 otherwise)

    Args:
        lambda1:     coeficient L_Seg  (default=1.0)
        lambda2:     coeficient L_MI   (default=0.5)
        lambda3:     coeficient L_Cos  (default=0.5)
        lambda_ant:  coeficient L_Ant  (default=1.0)
        lambda4:     coeficient L_PromptReg (default=0.1, C6b anchor loss)
        seg_sigma:   sigma filtru Gaussian pt L_Seg (default=4.0)
    """

    def __init__(
        self,
        lambda1: float = 1.0,
        lambda2: float = 0.5,
        lambda3: float = 0.5,
        lambda_ant: float = 1.0,
        lambda4: float = 0.1,
        seg_sigma: float = 4.0,
    ):
        super().__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.lambda_ant = lambda_ant
        self.lambda4 = lambda4

        self.l_label = nn.CrossEntropyLoss(ignore_index=-1, label_smoothing=0.1)
        self.l_seg   = SegmentationLoss(sigma=seg_sigma)
        self.l_mi    = MutualInformationLoss()
        self.l_cos   = CosineSimilarityLoss()

    def forward(
        self,
        segment_logits:      torch.Tensor,
        frame_logits:        torch.Tensor,
        u_soft:              torch.Tensor,
        mi_scores:           torch.Tensor,
        cos_similarities:    torch.Tensor,
        segment_labels:      torch.Tensor,
        frame_labels:        torch.Tensor,
        anticipation_labels: Optional[torch.Tensor] = None,
        training_stage:      int = 2,
        prompt_reg_loss:     Optional[torch.Tensor] = None,  # C6b
    ) -> dict:
        """
        Returns:
            dict cu loss-ul total si componentele (pentru logging).
        """
        B, N, C = segment_logits.shape

        l_label = self.l_label(
            segment_logits.reshape(B * N, C),
            segment_labels.reshape(B * N),
        )

        if anticipation_labels is not None:
            l_ant = self.l_label(
                segment_logits.reshape(B * N, C),
                anticipation_labels.reshape(B * N),
            )
        else:
            l_ant = torch.tensor(0.0, device=segment_logits.device)

        l_seg = self.l_seg(u_soft, frame_labels, training_stage)
        l_mi  = self.l_mi(mi_scores, segment_labels)
        l_cos = self.l_cos(cos_similarities, segment_labels)

        # C6b prompt regularization
        l_prompt_reg = (
            prompt_reg_loss
            if prompt_reg_loss is not None
            else torch.tensor(0.0, device=segment_logits.device)
        )

        total = (
            l_label
            + self.lambda_ant * l_ant
            + self.lambda1 * l_seg
            + self.lambda2 * l_mi
            + self.lambda3 * l_cos
            + self.lambda4 * l_prompt_reg
        )

        return {
            "total":          total,
            "l_label":        l_label.detach(),
            "l_ant":          l_ant.detach(),
            "l_seg":          l_seg.detach() if isinstance(l_seg, torch.Tensor) else l_seg,
            "l_mi":           l_mi.detach(),
            "l_cos":          l_cos.detach(),
            "l_prompt_reg":   l_prompt_reg.detach(),
        }