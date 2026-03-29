"""
losses.py
Defineste clasele de loss pentru VHOIP, conform formulatiilor din paper:
L_total = L_Label + lambda1*L_Seg + lambda2*L_MI + lambda3*L_Cos
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MutualInformationLoss(nn.Module):
    """
    L_MI - Mutual Information loss (Eq. 2 din paper).
    Maximizeaza MI intre features locale Z si reprezentarile
    globale integrate G, folosind un discriminator binar.
    """

    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            scores: (N, C) - scorurile discriminatorului pentru toate perechile (z_i, g_k)
            labels: (N,) - clasa adevarata pentru fiecare entitate (long)
        Returns:
            scalar loss
        """
        N, C = scores.shape

        # Construim target binar: 1 daca (z_i, g_k) sunt din aceeasi clasa
        targets = torch.zeros(N, C, device=scores.device)
        targets.scatter_(1, labels.unsqueeze(1), 1.0)

        return self.bce(scores, targets)


class CosineSimilarityLoss(nn.Module):
    """
    L_Cos - aliniere text-visual (Eq. 3 din paper).
    Aliniaza proiectiile Z' cu reprezentarile textuale T din CLIP.
    """

    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, similarities: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            similarities: (N, C) - similaritati cosinus intre Z' si T
            labels: (N,) - clase adevarate (long)
        Returns:
            scalar loss
        """
        return self.ce(similarities, labels)


class SegmentationLoss(nn.Module):
    """
    L_Seg - loss de segmentare temporala (frame-level).
    Mostenita din 2G-GCN/ASSIGN.
    """

    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, frame_logits: torch.Tensor, frame_labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frame_logits: (N, C) - predictii per frame
            frame_labels: (N,) - etichete per frame (-1 = ignorat)
        Returns:
            scalar loss
        """
        return self.ce(frame_logits, frame_labels)


class VHOIPLoss(nn.Module):
    """
    Loss total VHOIP (Eq. 4 din paper):
    L = L_Label + lambda1*L_Seg + lambda2*L_MI + lambda3*L_Cos
    """

    def __init__(self, lambda1: float = 1.0, lambda2: float = 0.5, lambda3: float = 0.5):
        super().__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3

        self.l_label = nn.CrossEntropyLoss()
        self.l_seg = SegmentationLoss()
        self.l_mi = MutualInformationLoss()
        self.l_cos = CosineSimilarityLoss()

    def forward(
        self,
        segment_logits: torch.Tensor,    # (N, C) - predictii segment-level
        frame_logits: torch.Tensor,      # (N, C) - predictii frame-level
        mi_scores: torch.Tensor,         # (N, C) - scoruri discriminator
        cos_similarities: torch.Tensor,  # (N, C) - similaritati cosinus
        segment_labels: torch.Tensor,    # (N,) - etichete segment
        frame_labels: torch.Tensor,      # (N,) - etichete frame
    ) -> dict:
        """
        Returns:
            dict cu loss-ul total si componentele individuale (pentru logging)
        """
        l_label = self.l_label(segment_logits, segment_labels)
        l_seg = self.l_seg(frame_logits, frame_labels)
        l_mi = self.l_mi(mi_scores, segment_labels)
        l_cos = self.l_cos(cos_similarities, segment_labels)

        total = l_label + self.lambda1 * l_seg + self.lambda2 * l_mi + self.lambda3 * l_cos

        return {
            "total": total,
            "l_label": l_label.detach(),
            "l_seg": l_seg.detach(),
            "l_mi": l_mi.detach(),
            "l_cos": l_cos.detach(),
        }