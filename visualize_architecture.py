"""
Diagrama de blocuri pentru arhitectura VHOIP.

Utilizare:
    python visualize_architecture.py --config configs/mphoi72.yaml --output vhoip_architecture
"""

import argparse
import os
import re
from pathlib import Path
from datetime import datetime

from graphviz import Digraph
from omegaconf import OmegaConf

from data.dataset import get_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Diagrama blocuri VHOIP")
    parser.add_argument("--config", type=str, default="configs/mphoi72.yaml")
    parser.add_argument("--output", type=str, default="vhoip_architecture")
    parser.add_argument("--format", type=str, default="svg", choices=["svg", "png", "pdf"])
    parser.add_argument("--rankdir", type=str, default="LR", choices=["LR", "TB"])
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"])
    parser.add_argument(
        "--batch-size-value",
        type=int,
        default=None,
        help="Valoarea explicita pentru B in etichete (implicit: training.batch_size din config).",
    )
    parser.add_argument(
        "--num-segments-value",
        type=int,
        default=None,
        help="Valoarea explicita pentru N in etichete (daca lipseste, ramane simbolic N).",
    )
    parser.add_argument(
        "--num-frames-value",
        type=int,
        default=None,
        help="Valoarea explicita pentru S in etichete (frames).",
    )
    parser.add_argument(
        "--num-entities-value",
        type=int,
        default=None,
        help="Valoarea explicita pentru M in etichete (entities/ROIs).",
    )
    return parser.parse_args()


def ensure_graphviz_in_path() -> None:
    """Adauga path-ul Graphviz in procesul curent daca nu este deja disponibil."""
    graphviz_candidates = [
        r"C:\Program Files\Graphviz\bin",
        r"C:\Program Files (x86)\Graphviz\bin",
    ]
    for candidate in graphviz_candidates:
        dot_exe = os.path.join(candidate, "dot.exe")
        if os.path.exists(dot_exe) and candidate not in os.environ.get("PATH", ""):
            os.environ["PATH"] = f"{candidate};{os.environ.get('PATH', '')}"
            break


def extract_vhoip_modules(vhoip_file: str = "models/vhoip.py"):
    """
    Extrage numele modulelor asignate in __init__ prin pattern-ul self.<name> = ...
    pentru a mentine diagrama sincronizata cu implementarea modelului.
    """
    path = Path(vhoip_file)
    if not path.exists():
        return []

    text = path.read_text(encoding="utf-8", errors="ignore")
    assignment_matches = re.findall(r"self\.(\w+)\s*=\s*(.+)", text)

    # Pastram doar asignarile care par module reale (constructor call),
    # evitand atribute scalar precum device/num_classes/etc.
    matches = []
    for name, expr in assignment_matches:
        expr = expr.strip()
        if "(" in expr and not expr.startswith(("cfg", "self.", "torch.")):
            matches.append(name)

    ordered_unique = []
    seen = set()
    for name in matches:
        if name not in seen:
            ordered_unique.append(name)
            seen.add(name)
    return ordered_unique


def infer_s_m_from_dataset(cfg, split: str, fold: int):
    """
    Incearca sa deduca S (frames) si M (entitati) din primul sample real al dataset-ului.
    Returneaza (S, M) sau (None, None) daca nu reuseste.
    """
    try:
        ds = get_dataset(cfg.dataset.name, root=cfg.dataset.root, split=split, fold=fold)
        if len(ds) == 0:
            return None, None
        sample = ds[0]
        roi = sample["roi_features"]
        if roi.ndim >= 3:
            return int(roi.shape[0]), int(roi.shape[1])
    except Exception:
        return None, None
    return None, None


def build_diagram(
    cfg,
    batch_size_value=None,
    num_segments_value=None,
    num_frames_value=None,
    num_entities_value=None,
) -> Digraph:
    g = Digraph("VHOIP", format="svg")
    g.attr(
        rankdir="LR",
        splines="spline",
        pad="0.22",
        nodesep="0.5",
        ranksep="0.7",
        bgcolor="white",
        fontname="Segoe UI",
        fontsize="14",
        labelloc="t",
        label="VHOIP Architecture Overview",
    )

    g.attr(
        "node",
        shape="box",
        style="rounded,filled",
        color="#334155",
        fillcolor="#f8fafc",
        fontname="Segoe UI",
        fontsize="10",
        penwidth="1.2",
        margin="0.1,0.08",
    )
    g.attr("edge", color="#475569", arrowsize="0.7", penwidth="1.1", fontname="Segoe UI", fontsize="9")

    hidden_dim = cfg.model.hidden_dim
    clip_dim = cfg.model.clip_dim
    roi_dim = cfg.data.roi_dim
    num_classes = cfg.model.num_classes
    batch_size = batch_size_value if batch_size_value is not None else cfg.training.batch_size
    s_dim = num_frames_value if num_frames_value is not None else "S"
    m_dim = num_entities_value if num_entities_value is not None else "M"
    if num_segments_value is not None:
        n_dim = num_segments_value
    elif isinstance(s_dim, int) and isinstance(m_dim, int):
        n_dim = s_dim * m_dim
    else:
        n_dim = "N"

    discovered_modules = extract_vhoip_modules()
    module_set = set(discovered_modules)

    with g.subgraph(name="cluster_inputs") as c:
        c.attr(label="Inputs", color="#bfdbfe", style="rounded")
        c.node("roi", f"ROI Features\n({batch_size},{s_dim},{m_dim},{roi_dim})", fillcolor="#e0f2fe")
        c.node("adj", f"Adjacency A (optional)\n({batch_size},{m_dim},{m_dim})", fillcolor="#e0f2fe")

    with g.subgraph(name="cluster_core") as c:
        c.attr(label="Core VHOIP", color="#bbf7d0", style="rounded")
        if "backbone" in module_set:
            c.node(
                "backbone",
                (
                    "Backbone 2G-GCN\n"
                    f"z: ({batch_size},{n_dim},{hidden_dim})\n"
                    f"frame_logits: ({batch_size},{n_dim},{num_classes})\n"
                    f"segment_logits: ({batch_size},{n_dim},{num_classes})"
                ),
                fillcolor="#dcfce7",
            )
        if "mlp_proj" in module_set:
            c.node(
                "zprime",
                f"MLP Projection\nz to z_prime ({batch_size},{n_dim},{clip_dim})\nL2 normalize",
                fillcolor="#fef3c7",
            )
        if "discriminator" in module_set:
            c.node("disc", f"MI Discriminator\nmi_scores: ({batch_size},{n_dim},{num_classes})", fillcolor="#e9d5ff")
        # Cosine este operatie functionala in forward (nu nn.Module), dar face parte din pipeline.
        c.node("cos", f"Cosine Similarity\ncos_similarities: ({batch_size},{n_dim},{num_classes})", fillcolor="#e9d5ff")

    with g.subgraph(name="cluster_priors") as c:
        c.attr(label="CLIP Priors", color="#fecdd3", style="rounded")
        if "text_encoder" in module_set:
            c.node("text", f"CLIP Text Encoder (frozen)\nT: ({num_classes},{clip_dim})", fillcolor="#fce7f3")
        if "global_rep" in module_set:
            c.node("global", f"Integrated Global Rep G\nG: ({num_classes},{clip_dim})\nEMA per epoch", fillcolor="#fee2e2")

    with g.subgraph(name="cluster_outputs") as c:
        c.attr(label="Outputs", color="#c7d2fe", style="rounded")
        c.node("seg_out", f"segment_logits\n({batch_size},{n_dim},{num_classes})", fillcolor="#dbeafe")
        c.node("frame_out", f"frame_logits\n({batch_size},{n_dim},{num_classes})", fillcolor="#dbeafe")
        c.node("mi_out", f"mi_scores\n({batch_size},{n_dim},{num_classes})", fillcolor="#ede9fe")
        c.node("cos_out", f"cos_similarities\n({batch_size},{n_dim},{num_classes})", fillcolor="#ede9fe")

    g.node(
        "note",
        "Inference uses only backbone outputs (segment/frame logits).",
        shape="note",
        style="filled",
        fillcolor="#fffbeb",
        color="#a16207",
        fontsize="9",
    )

    if "backbone" in module_set:
        g.edge("roi", "backbone")
        g.edge("adj", "backbone", style="dashed")

    if "backbone" in module_set and "mlp_proj" in module_set:
        g.edge("backbone", "zprime", label="z")
    if "backbone" in module_set and "discriminator" in module_set:
        g.edge("backbone", "disc", label="z")

    if "text_encoder" in module_set:
        g.edge("text", "cos", label="T")
    if "mlp_proj" in module_set:
        g.edge("zprime", "cos", label="z_prime")
    if "global_rep" in module_set and "discriminator" in module_set:
        g.edge("global", "disc", label="G")

    if "backbone" in module_set:
        g.edge("backbone", "seg_out", label="segment_logits")
        g.edge("backbone", "frame_out", label="frame_logits")
    if "discriminator" in module_set:
        g.edge("disc", "mi_out")
    g.edge("cos", "cos_out")

    g.edge("seg_out", "note", style="dotted", arrowhead="none")

    return g


def render_with_fallback(graph: Digraph, output_base: str):
    """
    Randeaza graful; daca fisierul tinta e blocat (permission denied),
    scrie automat in alt nume cu timestamp.
    """
    try:
        return graph.render(output_base, cleanup=True)
    except Exception as exc:  # graphviz ridica CalledProcessError prin wrapper intern
        msg = str(exc)
        if "Permission denied" not in msg:
            raise

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        alt_output = f"{output_base}_{ts}"
        print(
            f"Atentie: fisierul {output_base}.{graph.format} este probabil deschis. "
            f"Salvez in: {alt_output}.{graph.format}"
        )
        return graph.render(alt_output, cleanup=True)


def main():
    args = parse_args()
    ensure_graphviz_in_path()

    cfg = OmegaConf.merge(
        OmegaConf.load("configs/base.yaml"),
        OmegaConf.load(args.config),
    )

    inferred_s, inferred_m = infer_s_m_from_dataset(cfg, split=args.split, fold=args.fold)
    s_value = args.num_frames_value if args.num_frames_value is not None else inferred_s
    m_value = args.num_entities_value if args.num_entities_value is not None else inferred_m

    graph = build_diagram(
        cfg,
        batch_size_value=args.batch_size_value,
        num_segments_value=args.num_segments_value,
        num_frames_value=s_value,
        num_entities_value=m_value,
    )
    graph.format = args.format
    graph.graph_attr.update(rankdir=args.rankdir)

    discovered_modules = extract_vhoip_modules()
    if discovered_modules:
        print("Module detectate din models/vhoip.py:", ", ".join(discovered_modules))
    else:
        print("Atentie: nu am putut detecta modulele din models/vhoip.py; diagrama poate fi partiala.")

    if inferred_s is not None and inferred_m is not None:
        print(f"Shape inferat din dataset ({args.split}, fold {args.fold}): S={inferred_s}, M={inferred_m}")
    else:
        print("Nu am putut inferea automat S si M din dataset; folosesc override sau valori simbolice.")

    if args.format == "png":
        graph.graph_attr.update(dpi=str(args.dpi))

    out_path = render_with_fallback(graph, args.output)
    print(f"Diagrama salvata: {out_path}")


if __name__ == "__main__":
    main()
