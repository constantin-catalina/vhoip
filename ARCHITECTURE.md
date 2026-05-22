# VHOIP Architecture

This document describes the neural architecture of **VHOIP (Video-based Human-Object Interaction with CLIP Prior)**. VHOIP is a deep learning model for fine-grained temporal Human-Object Interaction (HOI) recognition in video. It combines a **2G-GCN + ASSIGN** spatiotemporal backbone with **CLIP** text-visual priors to improve recognition accuracy.

---

## Table of Contents

- [Overview](#overview)
- [Input Representation](#input-representation)
- [Stage 1: Backbone (2G-GCN + ASSIGN)](#stage-1-backbone-2g-gcn--assign)
- [Stage 2: CLIP Alignment](#stage-2-clip-alignment)
- [Stage 3: Mutual Information Discriminator](#stage-3-mutual-information-discriminator)
- [Global Representation (EMA)](#global-representation-ema)
- [Loss Functions](#loss-functions)
- [Two-Stage Training](#two-stage-training)
- [Inference Pipeline](#inference-pipeline)
- [Model Files](#model-files)
- [Hyperparameters](#hyperparameters)

---

## Overview

The VHOIP architecture is composed of three main parts:

1. **Backbone (`Backbone2GGCN`)**: Extracts spatiotemporal features from video frames using a **2-Graph GCN** and **ASSIGN** (Asynchronous Sparse Segmentation).
2. **CLIP Alignment (`CLIPTextEncoder`, `MLPProjection`)**: Projects learned video features into CLIP's shared text-visual embedding space and aligns them with class-specific text prototypes.
3. **Mutual Information (`Discriminator`, `IntegratedGlobalRepresentation`)**: A discriminator learns to match local video features with a global class prototype, encouraging coherent representations.

The final loss is a weighted sum of segmentation, mutual information, cosine similarity, and prompt regularization terms.

---

## Input Representation

Each input sample (video) is represented by pre-extracted tensors:

| Tensor | Shape | Description |
|--------|-------|-------------|
| `roi_features` | `(B, S, M, 2048)` | Faster R-CNN ROI-pooled features per entity |
| `geo_features` | `(B, S, J, 4)` | Skeleton keypoints (x, y, vx, vy) + object bbox corners |
| `entity_types` | `(B, M)` | Entity type labels: `0 = human`, `1 = object` |
| `clip_features` | `(B, S, M, 512)` | Offline CLIP ViT-B/16 visual features (optional) |
| `segment_labels` | `(B, N)` | Per-segment class labels (`N = S * M`) |
| `frame_labels` | `(B, N)` | Per-frame class labels |

Where:
- `B` = batch size
- `S` = number of frames
- `M` = entities per frame
- `J` = geometric keypoints
- `N = S * M`

---

## Stage 1: Backbone (2G-GCN + ASSIGN)

**File**: `models/backbone.py`

The backbone integrates a **2-Graph GCN** (geometric + fusion graph) for spatial feature extraction with **ASSIGN** (bidirectional RNN + sparse boundary detection) for temporal segmentation.

### 1.1 GeometricLevelGCN

**Input**: `geo_features` `(B, J, 4)` — positions and velocities of skeleton joints.

**Architecture**:
- Linear embedding: `4 -> C1=64 -> C1=64` (with ReLU)
- Self-attention adjacency matrix: `At = softmax(Theta(G) * Phi(G)^T)`
- Graph convolution: output = `At @ W_g(G)`

**Output**: `(B, J, C2=128)` geometric embeddings.

### 1.2 FusionLevelGraph

**Input**: `roi_features` `(B, M, 2048)` + `geo_out` `(B, J, 128)`

**Architecture**:
- Project visual features: `2048 -> hidden_dim=256` (2-layer MLP + ReLU)
- Project geometric features: `128 -> hidden_dim=256`
- Scaled dot-product attention between all visual and geometric tokens
- Residual connection + LayerNorm + Dropout

**Output**: `(B, M, hidden_dim=256)` fused per-entity frame representations.

### 1.3 FrameLevelBiRNN

**Input**: `(B*M, S, 256)` — fused features reshaped per entity.

**Architecture**:
- Optional input projection (if input_dim != hidden_dim)
- Bidirectional GRU: `input_size=256`, `hidden_size=128`, `num_layers=2`, `batch_first=True`
- Dropout + frame classifier: `Linear(256 -> num_classes)`

**Output**:
- `z`: `(B, N, 256)` — hidden states (flattened `B, M, S` -> `B, N`)
- `frame_logits`: `(B, N, C)` — per-frame classification logits

### 1.4 SpatialMessagePassing

**Input**: `combined = [x_fused; h_f]` `(B, M, 512)` + `entity_types` `(B, M)`

**Architecture**:
- Scaled dot-product attention over entity pairs
- `inter_mask`: entities of **different** types (human -> object, object -> human)
- `intra_mask`: entities of **same** type (human -> human)
- Masked softmax with NaN-to-zero fallback

**Output**: `m_inter` `(B, M, 512)`, `m_intra` `(B, M, 512)`

### 1.5 SegmentBoundaryDetector

**Input**: `concat[x_fused(D); h_f(D); m_intra(2D); m_inter(2D)]` = `(B*M*S, 6*D)`

**Architecture**:
- MLP: `6D -> 3D -> 2` (ReLU hidden)
- **Training Stage 2**: Gumbel-Softmax with straight-through estimator
  - `u_soft` = probability of boundary change
  - `u_hard` = binary mask (1 = new segment, 0 = continue)
- **Training Stage 1**: `u := 1` everywhere (dense boundaries)
- **Inference**: argmax over softmax

**Output**:
- `u_soft`: `(B, N)` — soft boundary probabilities
- `u_hard`: `(B, N)` — binary boundary mask

### 1.6 SegmentLevelLayer

**Input**:
- `h_f`: `(B, M, S, D)` — frame-level hidden states
- `m_inter_f`, `m_intra_f`: `(B, M, S, 2D)` — frame-level messages
- `u_hard`: `(B, M, S)` — boundary mask

**Architecture**:
1. Project messages: `2D -> D`
2. Iterative segment-level message passing:
   - Initialize `h_s_prev = h_f[:, :, 0, :]`
   - For each frame `t`: compute `m_inter_s`, `m_intra_s` using current `h_s_prev`
   - Update `h_s_prev` only where `u_hard=1`
3. Concatenate `z = [h_f; m_inter_f_p; m_intra_f_p; m_inter_s; m_intra_s]` -> `(B, M, S, 5D)`
4. Second BiRNN: `input=5D`, `hidden_size=D/2`, `bidirectional=True`
5. Classifier: `sigma(h_s) -> num_classes`

**Output**:
- `h_s`: `(B, M, S, D)` — segment-level hidden states
- `segment_logits`: `(B, M, S, C)` — per-segment classification logits

### Backbone Output Summary

The `Backbone2GGCN.forward()` returns:

| Key | Shape | Purpose |
|-----|-------|---------|
| `z` | `(B, N, 256)` | Hidden states for MI discriminator |
| `frame_logits` | `(B, N, C)` | Frame-level predictions |
| `segment_logits` | `(B, N, C)` | Segment-level predictions (final output) |
| `u_soft` | `(B, N)` | Boundary soft probabilities for `L_Seg` |

---

## Stage 2: CLIP Alignment

**Files**: `models/clip_modules.py`, `models/vhoip.py`

After the backbone extracts `z`, an MLP projects it into CLIP's 512-dimensional space.

### 2.1 MLPProjection

**Input**: `z` `(B, N, 256)`

**Architecture**:
- Linear: `256 -> 384`
- GELU + Dropout
- Linear: `384 -> 512`
- L2 normalization

**Output**: `z_prime` `(B, N, 512)` — normalized features for cosine comparison.

### 2.2 CLIPTextEncoder

Encodes class names into text prototypes `T` using CLIP's transformer.

**Modes**:
- **Baseline / C6**: Static multi-template ensemble. Example templates:
  - `"A photo of a person {verb}"`
  - `"A photo of a {subject} {verb}"`
- **C6b (Learnable Prompts)**: CoOp-style learnable context vectors.

#### C6b LearnablePromptEncoder

**Architecture**:
- **Shared context**: `ctx` `(n_ctx=16, 512)` initialized from `"a photo of a person"`
- **Class-specific offsets**: `class_ctx_offsets` `(C, 16, 512)` — small residual vectors per class, initialized to zero
- **Fixed token slicing**: context tokens are inserted anchored to the real EOS position, preventing truncation bugs

**Text construction per class**:
```
[SOS] [ctx_shared + offset_k] x 16 [class_tokens] [EOS] [PAD...]
```

**Output**: `T` `(C, 512)` — L2-normalized class text embeddings.

### 2.3 Cosine Similarity with Learnable Temperature

The CLIP text encoder exposes a learnable temperature scalar:

```
temperature = exp(clamp(log_temp, max=3.5))
```

Cosine similarities are scaled by this temperature before cross-entropy:

```
cos_similarities = (z_prime @ T^T) * temperature   # (B, N, C)
```

This sharpens the distribution similarly to how CLIP trains its own contrastive head.

---

## Stage 3: Mutual Information Discriminator

**File**: `models/discriminator.py`

Implements Equation 1 from the VHOIP paper:

```
y_hat_{i,k} = sigmoid( MLP(sigmoid(g_k))  *  MLP(PReLU(||z_i||_2)) )
```

### Architecture

**Global branch** (class prototype):
- Input: `G` `(C, 512)`
- Sigmoid (`sigma1`)
- MLP: `512 -> 256 -> 256` (ReLU hidden)
- Output: `h_g` `(C, 256)`

**Local branch** (per-entity feature):
- Input: `z` `(B, N, 256)`
- L2 normalization (`||z_i||_2`)
- PReLU (`sigma2`)
- MLP: `256 -> 256 -> 256` (ReLU hidden)
- Output: `h_z` `(B, N, 256)`

**Output**:
- Dot product: `scores = h_z @ h_g^T` -> `(B, N, C)`
- **No explicit sigmoid** — `BCEWithLogitsLoss` applies it internally for numerical stability.
- NaN sanitization: replaces NaN inputs with zeros to prevent gradient explosion propagation.

---

## Global Representation (EMA)

**File**: `models/clip_modules.py` — `IntegratedGlobalRepresentation`

Implements **Algorithm 1** from the paper:

```
G = rho * G + (1 - rho) * V
```

Where:
- `G` `(C, 512)`: integrated global representation (buffer, not learnable)
- `V` `(C, 512)`: class prototypes computed from collected `z_prime` features after each epoch
- `rho = 0.9`: EMA decay coefficient

### Initialization

- **Primary**: `initialize_G(clip_visual_features, labels)` — average and L2-normalize offline CLIP visual features per class.
- **Fallback**: `initialize_G_from_text()` — use frozen text prototypes `T`.

### Update Schedule

- During **warmup epochs** (default 5): `G` is frozen at its initialized value.
- After warmup: `G` is updated with EMA at the end of each epoch using collected `z_prime` features.

---

## Loss Functions

**File**: `models/losses.py`

The total loss is:

```
L_total = L_Label + lambda_ant * L_Ant + lambda1 * L_Seg + lambda2 * L_MI + lambda3 * L_Cos + lambda4 * L_PromptReg
```

| Loss | Description | Implementation |
|------|-------------|----------------|
| `L_Label` | Segment classification | `CrossEntropyLoss` with label smoothing (0.1) |
| `L_Ant` | Anticipation (shifted labels) | Same CE loss on future-shifted labels; disabled by default (`lambda_ant=0`) |
| `L_Seg` | Boundary detection | BCE with Gaussian-smoothed GT boundaries (`sigma=2.0`), positive weight `5.0` |
| `L_MI` | Mutual information | `BCEWithLogitsLoss` on discriminator scores vs. one-hot class targets |
| `L_Cos` | CLIP cosine alignment | `CrossEntropyLoss` on temperature-scaled cosine similarities |
| `L_PromptReg` | Prompt drift anchor | `1 - mean(cos_sim(T_learned, T_frozen))`; weight `lambda4=0.1` |

### Boundary Ground Truth (`L_Seg`)

A boundary is marked at frame `t` if `frame_labels[t] != frame_labels[t+1]`. The binary signal is then smoothed with a 1D Gaussian kernel (`sigma=2.0`) to provide soft targets.

---

## Two-Stage Training

**File**: `models/backbone.py` — `Backbone2GGCN.forward()`

ASSIGN uses a two-stage training strategy:

### Stage 1 (Dense Training)
- `training_stage=1`
- All frames are treated as segment boundaries (`u := 1` everywhere)
- `L_Seg` is disabled
- The model learns frame-level and segment-level classification without boundary prediction

### Stage 2 (Sparse Training)
- `training_stage=2`
- Gumbel-Softmax boundary detector is active
- `L_Seg` is enabled
- The model learns to predict sparse, asynchronous segments

Transition from Stage 1 to Stage 2 typically happens after a small number of warmup epochs (e.g., 3 epochs for MPHOI-72).

---

## Inference Pipeline

**File**: `models/vhoip.py`

At inference time, the model switches to evaluation mode:

1. `set_inference_mode(True)`:
   - Recomputes final learned text features `T` (if using learnable prompts) and freezes them into the buffer
   - Sets `eval()` mode
   - Sets `_inference_mode = True`

2. Forward pass returns only:
   - `segment_logits`: `(B, N, C)` — final predictions
   - `frame_logits`: `(B, N, C)` — auxiliary frame-level predictions

3. No CLIP text encoding, discriminator scoring, or EMA updates happen during inference.

---

## Model Files

| File | Purpose |
|------|---------|
| `models/vhoip.py` | **Main orchestrator**. Assembles all submodules, implements forward pass, inference mode, EMA end-of-epoch update |
| `models/backbone.py` | **2G-GCN + ASSIGN backbone**. Geometric GCN, fusion graph, BiRNNs, boundary detector, segment-level layer |
| `models/clip_modules.py` | **CLIP integration**. Text encoder, visual encoder, learnable prompts, MLP projection, prototyping, EMA global representation |
| `models/discriminator.py` | **MI discriminator**. Binary classifier over `(z, G)` pairs implementing Eq. 1 |
| `models/losses.py` | **Loss definitions**. Segmentation, MI, cosine similarity, and combined VHOIP loss |

---

## Hyperparameters

**File**: `configs/base.yaml`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model.clip_model` | `ViT-B/16` | Frozen CLIP encoder |
| `model.clip_dim` | `512` | CLIP feature dimension |
| `model.hidden_dim` | `256` | Backbone hidden dimension |
| `model.num_layers` | `2` | GRU layers |
| `model.dropout` | `0.3` | Dropout rate |
| `model.use_learnable_prompts` | `true` | Enable C6b learnable prompts |
| `model.prompt_n_ctx` | `16` | Number of learnable context tokens |
| `training.epochs` | `50` | Training epochs |
| `training.batch_size` | `4` | Batch size |
| `training.learning_rate` | `1e-3` | Learning rate |
| `training.lambda1` | `1.0` | `L_Seg` weight |
| `training.lambda2` | `0.5` | `L_MI` weight |
| `training.lambda3` | `0.5` | `L_Cos` weight |
| `training.lambda4` | `0.1` | `L_PromptReg` weight (C6b) |
| `training.lambda_ant` | `0.0` | Anticipation loss weight (disabled) |
| `training.rho` | `0.9` | EMA coefficient for `G` |
| `training.warmup_epochs` | `5` | Epochs before EMA updates begin |
| `training.grad_clip` | `0.5` | Gradient clipping norm |
| `training.seg_sigma` | `2.0` | Gaussian kernel width for boundary smoothing |
| `training.seg_pos_weight` | `5.0` | BCE positive weight for boundaries |
| `data.roi_dim` | `2048` | Faster R-CNN ROI feature dimension |
| `evaluation.iou_thresholds` | `[0.10, 0.25, 0.50]` | IoU thresholds for F1@k metrics |

---

## Complete Data Flow

```
Input:
  roi_features  (B, S, M, 2048)
  geo_features  (B, S, J, 4)
  entity_types  (B, M)

Stage 1: Backbone (2G-GCN + ASSIGN)
  ├─ GeometricLevelGCN  ->  geo_out  (B, J, 128)
  ├─ FusionLevelGraph   ->  x_fused  (B, M, S, 256)
  ├─ FrameLevelBiRNN    ->  z  (B, N, 256), frame_logits  (B, N, C)
  ├─ SpatialMessagePassing  ->  m_inter, m_intra  (B, M, S, 512)
  ├─ SegmentBoundaryDetector  ->  u_soft, u_hard  (B, N)
  └─ SegmentLevelLayer  ->  segment_logits  (B, N, C)

Stage 2: CLIP Alignment
  ├─ MLPProjection(z)   ->  z_prime  (B, N, 512)  [L2 normalized]
  └─ CLIPTextEncoder    ->  T  (C, 512)  [L2 normalized]
      ├─ Baseline: multi-template ensemble
      └─ C6b: learnable context vectors + class offsets
  cos_similarities = (z_prime @ T^T) * temperature  (B, N, C)

Stage 3: Mutual Information
  ├─ IntegratedGlobalRepresentation  ->  G  (C, 512)
  └─ Discriminator(z, G)  ->  mi_scores  (B, N, C)

Losses:
  L_total = CE(segment_logits) + lambda1 * BCE(u_soft, boundary_gt)
            + lambda2 * BCE(mi_scores, one_hot)
            + lambda3 * CE(cos_similarities)
            + lambda4 * (1 - mean(cos_sim(T_learned, T_frozen)))

End of Epoch:
  G = rho * G + (1 - rho) * Prototyping(z_prime, labels)
```
