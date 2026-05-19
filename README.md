# VHOIP — Video-based Human-Object Interaction with CLIP Prior

Fine-grained temporal HOI recognition using a 2G-GCN + ASSIGN backbone with CLIP text-visual priors. Supports MPHOI-72, CAD-120, and Bimanual Actions datasets.

## Setup

```bash
pip install -r requirements.txt
```

## Commands

### Training

```bash
python train.py --config configs/mphoi72.yaml
```

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | required | Dataset config YAML (`configs/mphoi72.yaml`, `configs/cad120.yaml`, `configs/bimanual.yaml`) |
| `--fold` | 0 | Cross-validation fold index |
| `--seed` | 42 | Random seed |
| `--resume` | None | Path to checkpoint to resume training from |
| `--experiment_name` | None | Experiment name (subdirectory for checkpoints) |
| `--wandb` | False | Enable Weights & Biases logging |
| `--wandb_project` | None | W&B project name |
| `--wandb_entity` | None | W&B entity/username |
| `--wandb_run_name` | None | W&B run name (supports `{dataset}`, `{fold}`, `{seed}`, `{experiment_name}` placeholders) |
| `--override` | None | Override config values. Repeatable. Example: `--override training.lambda2=0.3 --override training.epochs=100` |
| `--device` | auto | Torch device (`cuda`, `cpu`) |

Examples:

```bash
# Train fold 3 with W&B logging
python train.py --config configs/mphoi72.yaml --fold 3 --wandb --wandb_project my-project

# Resume from a checkpoint
python train.py --config configs/mphoi72.yaml --fold 0 --resume checkpoints/fold0/last_model.pth

# Override learning rate and epochs
python train.py --config configs/mphoi72.yaml --fold 0 --override training.learning_rate=5e-4 --override training.epochs=100
```

### Inference on a Video

```bash
python scripts/inference.py --config configs/mphoi72.yaml --video-id SUBJECT1_TASK1 --checkpoint checkpoints/fold0/best_model.pth
```

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | required | Config YAML |
| `--checkpoint` | None | Path to a single checkpoint |
| `--checkpoint-pattern` | None | Glob pattern with `{fold}` for ensembling across folds (e.g. `checkpoints/fold{fold}/best_model.pth`) |
| `--num-folds` | 28 | Number of folds when using `--checkpoint-pattern` |
| `--video-id` | None | Dataset video ID (loads pre-extracted features) |
| `--data-root` | `data/mphoi72/` | Dataset root (used with `--video-id`) |
| `--input` | None | Path to a raw video file or image directory (live extraction, slower) |
| `--output` | `inference_results.json` | Output JSON file |
| `--visualize` | False | Generate annotated video + timeline PNG |
| `--video-out` | None | Output video path (when `--visualize` is set) |
| `--max-entities` | 5 | Max entities per frame |
| `--max-frames` | None | Cap on number of frames to process |
| `--device` | auto | Torch device |
| `--dataset-name` | None | Override dataset name from config |
| `--fps` | 15 | FPS for output video |

Examples:

```bash
# Run on a dataset video (recommended — uses cached features)
python scripts/inference.py --config configs/mphoi72.yaml --video-id SUBJECT1_TASK1 --checkpoint checkpoints/fold0/best_model.pth

# Ensemble across all 28 folds
python scripts/inference.py --config configs/mphoi72.yaml --video-id SUBJECT1_TASK1 --checkpoint-pattern "checkpoints/fold{fold}/best_model.pth" --num-folds 28

# Run on a raw video file with visualization
python scripts/inference.py --config configs/mphoi72.yaml --input my_video.mp4 --checkpoint checkpoints/fold0/best_model.pth --visualize --video-out output.mp4

# Save raw data for later visualization
python scripts/inference.py --config configs/mphoi72.yaml --video-id SUBJECT1_TASK1 --checkpoint checkpoints/fold0/best_model.pth --output results.json
```

### Visualization

#### Test Set Segmentation Bars

Generates GT-vs-prediction temporal segmentation bar plots for test videos:

```bash
python scripts/plot_test_videos.py --checkpoint checkpoints/fold0/best_model.pth --fold 0
```

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoint` | required | Path to `best_model.pth` |
| `--fold` | 0 | Cross-validation fold |
| `--num_videos` | 5 | Number of test videos to visualize |
| `--entity` | 0 | Entity index to visualize (0 = first person) |
| `--output_dir` | `outputs/viz` | Output directory |
| `--device` | `cuda` | Device |

Outputs:
- Per-video PNG: `outputs/viz/fold0_{video_id}_e0.png`
- Combined PNG: `outputs/viz/fold0_combined.png`

#### Demo Video from Inference Results

Produces an annotated MP4 video and timeline PNG from raw inference output:

```bash
# From raw .npz data (no re-extraction needed)
python scripts/generate_demo_video.py --raw results_wo_L_Ant_raw.npz --dataset mphoi72 --output demo.mp4

# From JSON results + original video (re-extracts bounding boxes)
python scripts/generate_demo_video.py --results results_wo_L_Ant.json --input original_video.mp4 --dataset mphoi72 --output demo.mp4
```

| Flag | Default | Description |
|------|---------|-------------|
| `--raw` | None | Path to `.npz` file from inference.py |
| `--results` | None | Path to JSON results file |
| `--input` | None | Original video path (required with `--results`) |
| `--dataset` | required | One of: `cad120`, `mphoi72`, `bimanual` |
| `--output` | `visualization.mp4` | Output video path |
| `--timeline-png` | None | Optional timeline PNG path |
| `--fps` | 15 | FPS for output video |

### Ablation Studies

#### Full Ablation (6 variants)

Runs across all fold combinations: `w/o_L_Ant`, `w/o_LearnPrompts`, `w/o_L_PromptReg`, `w/o_GeoBranch`, `w/o_L_MI`, `w/o_L_Cos`.

```bash
python scripts/run_ablations.py --config configs/mphoi72.yaml --folds 5 --seed 42
```

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | `configs/mphoi72.yaml` | Config YAML |
| `--folds` | 5 | Number of folds per variant |
| `--seed` | 42 | Random seed |
| `--device` | `cuda` | Device |
| `--csv` | `ablation_results.csv` | CSV output path |
| `--resume` | False | Skip folds where `best_model.pth` already exists |

#### Anticipation Ablation (4 variants, all with L_Ant removed)

Runs: `w/o_L_Ant+w/o_L_PromptReg`, `w/o_L_Ant+w/o_GeoBranch`, `w/o_L_Ant+w/o_L_MI`, `w/o_L_Ant+w/o_L_Cos`.

```bash
python scripts/run_ablations_ant.py --config configs/mphoi72.yaml --folds 5 --seed 42
```

Same flags as `run_ablations.py`, default CSV is `ablation_ant_results.csv`.

### Dataset Setup

#### MPHOI-72 Preprocessing

Converts zarr features to npy, generates cross-validation splits, and optionally extracts CLIP visual features:

```bash
# Full setup (convert + verify)
python scripts/setup_mphoi72.py --data_root data/mphoi72/ --verify

# Inspect zarr structure only
python scripts/setup_mphoi72.py --inspect

# Extract CLIP visual features after conversion
python scripts/setup_mphoi72.py --data_root data/mphoi72/ --extract_clip --clip_device cuda
```

| Flag | Default | Description |
|------|---------|-------------|
| `--data_root` | `data/mphoi72/` | Dataset root directory |
| `--inspect` | False | Inspect zarr structure only |
| `--verify` | False | Verify generated files |
| `--extract_clip` | False | Extract CLIP visual features |
| `--clip_device` | `cuda` | Device for CLIP extraction |
| `--clip_batch_size` | 64 | Batch size for CLIP inference |

#### Generic Preprocessing

For CAD-120 and Bimanual datasets:

```bash
python data/preprocess.py --dataset cad120 --data_root data/cad120/
python data/preprocess.py --dataset bimanual --data_root data/bimanual/
```

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset` | required | One of: `cad120`, `mphoi72`, `bimanual` |
| `--data_root` | required | Dataset root directory |
| `--device` | auto | Device (`cuda` / `cpu`) |

### Metrics Extraction

Extracts best FSUM from saved checkpoints:

```bash
python scripts/extract_best_fsum.py --checkpoint_dir checkpoints/
```

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoint_dir` | `checkpoints` | Root checkpoint directory |
| `--dataset` | None | Filter by dataset name |
| `--experiment` | None | Filter by experiment name |
| `--csv` | None | Optional CSV output path |

## Configuration

Configs are merged in order: `base.yaml` → dataset YAML → CLI `--override` flags.

Key parameters in `configs/base.yaml`:

| Section | Key | Default | Description |
|---------|-----|---------|-------------|
| `model` | `clip_model` | `ViT-B/16` | CLIP encoder |
| `model` | `use_learnable_prompts` | true | Enable C6b learnable prompts |
| `model` | `prompt_n_ctx` | 16 | Number of context tokens |
| `training` | `epochs` | 50 | Training epochs |
| `training` | `batch_size` | 4 | Batch size |
| `training` | `learning_rate` | 1e-3 | Learning rate |
| `training` | `lambda1` | 1.0 | L_Seg weight |
| `training` | `lambda2` | 0.5 | L_MI weight |
| `training` | `lambda3` | 0.5 | L_Cos weight |
| `training` | `lambda_ant` | 0.0 | L_Anticipation weight (disabled by default) |
| `training` | `rho` | 0.9 | EMA coefficient for G update |
| `training` | `seg_sigma` | 2.0 | Gaussian kernel width for boundary smoothing |
| `training` | `seg_pos_weight` | 5.0 | BCE pos_weight for boundary imbalance |
| `evaluation` | `iou_thresholds` | [0.10, 0.25, 0.50] | IoU thresholds for F1@k |

## Loss Function

```
L_total = L_Label + lambda_ant * L_Ant + lambda1 * L_Seg + lambda2 * L_MI + lambda3 * L_Cos + lambda4 * L_PromptReg
```

| Component | Description |
|-----------|-------------|
| `L_Label` | Cross-entropy with label smoothing (0.1) |
| `L_Ant` | Anticipation loss (shifted segment labels; disabled by default) |
| `L_Seg` | Boundary detection BCE with Gaussian-smoothed ground truth |
| `L_MI` | Mutual information loss (discriminator) |
| `L_Cos` | Cosine similarity cross-entropy with learnable temperature |
| `L_PromptReg` | Prompt regularization (1 − mean cosine similarity between learned and frozen prompts) |

## Evaluation Metrics

- **F1@10**, **F1@25**, **F1@50** — per IoU thresholds 0.10, 0.25, 0.50
- **FSUM** = F1@10 + F1@25 + F1@50 (primary metric)