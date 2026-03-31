# VHOIP - Video-based Human-Object Interaction with CLIP Prior Knowledge

Implementare pentru lucrarea de licenta bazata pe:

> Baek & Choe, "VHOIP: Video-based Human-Object Interaction recognition with CLIP Prior knowledge", Pattern Recognition Letters, 2024.

## Setup

```bash
# 1. Creaza mediu virtual
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # Linux/Mac

# 2. Instaleaza dependentele (alege varianta CUDA corecta din requirements.txt)
pip install -r requirements.txt

# 3. Verifica instalarea
python -c "import torch; print(torch.cuda.is_available())"
```

## Structura proiect

```
vhoip/
├── configs/        # hiperparametri per dataset
├── data/           # dataset classes + preprocessing
├── models/         # arhitectura VHOIP
├── utils/          # metrici, logging, checkpoints
├── train.py        # antrenare
├── evaluate.py     # evaluare cross-validare
└── demo.py         # inferenta pe video nou
```

## Antrenare

```bash
# Incepe cu CAD-120 (cel mai mic dataset)
python train.py --config configs/cad120.yaml --fold 0

# MPHOI-72
python train.py --config configs/mphoi72.yaml --fold 0

# Monitorizeaza antrenarea
tensorboard --logdir logs/
```

## Evaluare

```bash
python evaluate.py --config configs/cad120.yaml --checkpoint checkpoints/best_model.pth
```

## Dataset-uri

| Dataset          | Persoane    | Clase | Cross-val             |
| ---------------- | ----------- | ----- | --------------------- |
| CAD-120          | 1           | 10    | leave-one-subject-out |
| MPHOI-72         | 2           | 13    | two-subject-out       |
| Bimanual Actions | 1 (2 maini) | 14    | leave-one-subject-out |

## Rezultate reproduse (tinta)

| Dataset  | F1@10 | F1@25 | F1@50 | FSUM  |
| -------- | ----- | ----- | ----- | ----- |
| MPHOI-72 | 70.3  | 65.7  | 52.6  | 188.6 |
| CAD-120  | 90.1  | 86.6  | 76.4  | 518.0 |
| Bimanual | 84.5  | 81.6  | 68.9  | 235.0 |
