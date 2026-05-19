# VHOIP Raw Video Pipeline

End-to-end feature extraction and inference pipeline for VHOIP
(Video-based Human-Object Interaction recognition with CLIP Prior knowledge).

## What This Does

```
Raw Video
    │
    ▼
┌─────────────────────────────────────────────────┐
│  extract_features.py                            │
│                                                 │
│  1. Detect humans & objects (YOLOv8 or VG RCNN) │
│  2. Track entities across frames (SORT)         │
│  3. Extract 2048-dim ROI visual features        │
│  4. Extract 512-dim CLIP features per region    │
│  5. Extract skeleton keypoints (MMPose)         │
└─────────────────────────────────────────────────┘
    │
    ▼  features/video_features.npz
    │
    ▼
┌─────────────────────────────────────────────────┐
│  run_inference.py                               │
│                                                 │
│  6. Load .npz features                          │
│  7. Run VHOIP / 2G-GCN model                   │
│  8. Output per-frame HOI labels                 │
└─────────────────────────────────────────────────┘
    │
    ▼  predictions.csv  (frame, entity, sub-activity, affordance)
```

---

## Installation

### Step 1 — Base dependencies

```bash
pip install torch torchvision
pip install opencv-python numpy scipy filterpy Pillow
pip install ultralytics                        # YOLOv8 (easy fallback detector)
pip install git+https://github.com/openai/CLIP.git   # CLIP
```

### Step 2 — Skeleton extraction (optional but recommended for 2G-GCN)

```bash
pip install -U openmim
mim install mmengine mmcv mmdet mmpose
```

### Step 3 — Visual Genome Faster R-CNN (exact match to the paper)

This matches what VHOIP uses. Requires detectron2.

```bash
# Install detectron2
pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu118/torch2.0/index.html

# Clone the bottom-up attention repo (VG Faster R-CNN weights)
git clone https://github.com/airsplay/py-bottom-up-attention.git
cd py-bottom-up-attention

# Download the VG pretrained weights
wget https://dl.fbaipublicfiles.com/detectron2/BottomUpTopDownModels/bottomupTopDown_10_100.pth
```

Use `--detector vg` with the config and weights paths when running extraction.

---

## Usage

### Feature Extraction

**Quick start (YOLOv8 detector, no extra setup):**
```bash
python extract_features.py \
    --video my_video.mp4 \
    --output features/ \
    --detector yolo \
    --score-thresh 0.4 \
    --max-entities 10
```

**Paper-accurate (Visual Genome Faster R-CNN):**
```bash
python extract_features.py \
    --video my_video.mp4 \
    --output features/ \
    --detector vg \
    --vg-config py-bottom-up-attention/configs/VG-Detection/faster_rcnn_R_101_C4_caffe.yaml \
    --vg-weights bottomupTopDown_10_100.pth \
    --score-thresh 0.4
```

**Subsample to 5 FPS (faster processing):**
```bash
python extract_features.py \
    --video my_video.mp4 \
    --output features/ \
    --sample-fps 5
```

---

### Run Inference

```bash
python run_inference.py \
    --features features/my_video_features.npz \
    --checkpoint checkpoints/vhoip_mphoi72.pth \
    --dataset mphoi72 \
    --output-csv predictions.csv
```

**Supported datasets:** `mphoi72`, `cad120`, `bimanual`

---

### Connecting to Your VHOIP Model

In `run_inference.py`, replace the `_load_model()` method with your actual model:

```python
def _load_model(self, checkpoint_path, dataset, device):
    import sys
    sys.path.insert(0, "/path/to/your/vhoip_repo")
    from models.vhoip import VHOIP        # adjust to your module name

    n_classes = len(LABEL_MAPS[dataset]["sub_activities"])
    model = VHOIP(n_classes=n_classes, visual_dim=2048, clip_dim=512)
    ckpt  = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])   # adjust key if needed
    return model.to(device)
```

---

## Output Format

### .npz feature file

| Array | Shape | Description |
|---|---|---|
| `roi_features` | (T, N, 2048) | ROI-pooled visual features per entity |
| `clip_features` | (T, N, 512) | CLIP ViT-B/16 features per entity |
| `geo_features` | (T, N, 34) | Skeleton keypoints (17×xy, normalized) |
| `boxes` | (T, N, 4) | Bounding boxes xyxy |
| `entity_ids` | (T, N) | Consistent tracker IDs across frames |
| `is_human` | (T, N) | 1=human, 0=object |
| `valid_mask` | (T, N) | 1=slot occupied, 0=padding |

T = number of frames, N = max entities per frame (set with `--max-entities`)

### predictions.csv

```
frame, entity_id, type,   sub_activity, sub_conf, affordance, aff_conf
0,     1,         human,  approach,     0.821,    —,          0.000
0,     2,         object, null,         0.634,    reachable,  0.712
1,     1,         human,  lift,         0.779,    —,          0.000
...
```

---

## Pipeline Architecture

```
Video frames
    │
    ├──► YOLOv8 / VG Faster R-CNN
    │         │
    │         ├── boxes (N, 4)
    │         ├── class IDs (N,)
    │         └── ROI features (N, 2048)   ──────────────────► roi_features
    │
    ├──► SORT Tracker
    │         └── entity_ids (N,)          ──────────────────► entity_ids
    │
    ├──► MMPose ViTPose-B
    │         └── keypoints (N_h, 17, 3)
    │                  └── normalize by box ──────────────────► geo_features
    │
    └──► CLIP ViT-B/16
              └── encode cropped regions   ──────────────────► clip_features
```

---

## Potential Diploma Contributions

This pipeline itself exposes several research gaps you can address:

1. **Detection quality vs HOI accuracy**: Compare YOLOv8 vs VG Faster R-CNN
   as the detector and measure the impact on final F1@k scores.

2. **Open-vocabulary detection**: Replace Faster R-CNN with GroundingDINO
   or OWL-ViT (both CLIP-based like VHOIP) for better zero-shot entity detection.

3. **Tracking robustness**: Study how tracking errors (ID switches, missed
   detections) propagate into HOI recognition performance.

4. **End-to-end fine-tuning**: Currently detection and recognition are decoupled.
   Joint training could improve performance.
