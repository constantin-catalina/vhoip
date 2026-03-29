"""
utils/visualize.py
Vizualizare rezultate segmentare temporala (similar cu Fig. 3 din paper).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import List, Optional


# Paleta de culori pentru clase (similara cu paper-ul)
PALETTE = [
    "#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F",
    "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F", "#BAB0AC",
    "#D37295", "#FABFD2", "#8CD17D", "#B6992D",
]


def plot_segmentation(
    ground_truth: List[int],
    prediction: List[int],
    label_names: List[str],
    title: str = "",
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Vizualizeaza o bara de segmentare temporala (GT vs Predictie).
    Similar cu Fig. 3 din paper.

    Args:
        ground_truth: lista de etichete per frame (GT)
        prediction:   lista de etichete per frame (predictie)
        label_names:  numele claselor
        title:        titlu grafic
        save_path:    calea unde se salveaza (None = nu salveaza)
        show:         afiseaza graficul interactiv
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 3), sharex=True)
    T = len(ground_truth)
    x = np.arange(T)

    for ax, labels, row_title in zip(
        axes, [ground_truth, prediction], ["Ground Truth", "VHOIP (ours)"]
    ):
        for t in range(T):
            cls = labels[t]
            color = PALETTE[cls % len(PALETTE)]
            ax.barh(0, 1, left=t, height=0.6, color=color, edgecolor="none")

        ax.set_xlim(0, T)
        ax.set_ylim(-0.4, 0.4)
        ax.set_yticks([0])
        ax.set_yticklabels([row_title], fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.tick_params(left=False)

    # Legenda
    unique_classes = sorted(set(ground_truth + prediction))
    patches = [
        mpatches.Patch(color=PALETTE[c % len(PALETTE)], label=label_names[c])
        for c in unique_classes
        if c < len(label_names)
    ]
    fig.legend(handles=patches, loc="lower center", ncol=min(len(patches), 7),
               fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.15))

    if title:
        fig.suptitle(title, fontsize=10, y=1.02)

    axes[-1].set_xlabel("Frame", fontsize=9)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"  Salvat: {save_path}")

    if show:
        plt.show()

    plt.close()


def plot_multi_segmentation(
    results: List[dict],
    label_names: List[str],
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Vizualizeaza mai multe video-uri simultan (ca Fig. 3 din paper).

    Args:
        results: lista de dicts cu cheile 'title', 'gt', 'pred'
        label_names: numele claselor
    """
    n = len(results)
    fig, axes = plt.subplots(n * 2, 1, figsize=(14, 2.5 * n))

    if n == 1:
        axes = [axes] if not hasattr(axes, "__len__") else axes

    row = 0
    for result in results:
        title   = result.get("title", "")
        gt      = result["gt"]
        pred    = result["pred"]
        T       = len(gt)

        for labels, row_title in [(gt, "GT"), (pred, "VHOIP")]:
            ax = axes[row]
            for t in range(T):
                cls = labels[t]
                color = PALETTE[cls % len(PALETTE)]
                ax.barh(0, 1, left=t, height=0.7, color=color, edgecolor="none")

            ax.set_xlim(0, T)
            ax.set_ylim(-0.4, 0.4)
            ax.set_yticks([0])
            ax.set_yticklabels([row_title], fontsize=8)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_visible(False)
            ax.tick_params(left=False, bottom=(row_title == "VHOIP"))

            if row_title == "GT" and title:
                ax.set_title(title, fontsize=9, loc="left", pad=2)

            row += 1

    # Legenda globala
    all_classes = set()
    for r in results:
        all_classes.update(r["gt"] + r["pred"])

    patches = [
        mpatches.Patch(color=PALETTE[c % len(PALETTE)], label=label_names[c])
        for c in sorted(all_classes)
        if c < len(label_names)
    ]
    fig.legend(handles=patches, loc="lower center", ncol=min(len(patches), 8),
               fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.1)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"  Salvat: {save_path}")

    if show:
        plt.show()

    plt.close()


def plot_loss_curves(
    train_losses: dict,
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Vizualizeaza curbele de loss in timpul antrenarii.

    Args:
        train_losses: dict cu cheile 'total', 'l_label', 'l_seg', 'l_mi', 'l_cos'
                      fiecare avand o lista de valori per epoch
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Stanga: Loss total
    axes[0].plot(train_losses["total"], color="#4E79A7", linewidth=2, label="Total")
    axes[0].set_title("Loss total", fontsize=10)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Dreapta: Componente individuale
    colors = {"l_label": "#F28E2B", "l_seg": "#E15759", "l_mi": "#76B7B2", "l_cos": "#59A14F"}
    labels = {"l_label": "L_Label", "l_seg": "L_Seg", "l_mi": "L_MI", "l_cos": "L_Cos"}

    for key, color in colors.items():
        if key in train_losses:
            axes[1].plot(train_losses[key], color=color, linewidth=1.5, label=labels[key])

    axes[1].set_title("Componente loss", fontsize=10)
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)

    if show:
        plt.show()

    plt.close()