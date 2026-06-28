# PPE Compliance Detection System

A real-time computer vision system that detects personal protective equipment (PPE) compliance — hardhats and safety vests — using a fine-tuned YOLOv8 model. Built to flag missing safety equipment on a construction/industrial site from a live camera feed, a captured photo, or an uploaded image.

## What it does

The model detects 5 classes:
- `Hardhat` / `NO-Hardhat`
- `Safety Vest` / `NO-Safety Vest`
- `Person`

Detecting the *absence* of PPE (not just its presence) is what makes this a compliance system rather than a simple object detector — the goal is catching violations, not just confirming gear exists somewhere in frame.

## Results

Fine-tuned YOLOv8s on 2,535 training images, evaluated on a held-out test set of 60 images:

| Metric | Score |
|---|---|
| mAP@0.5 | 83.5% |
| mAP@0.5:0.95 | 54.0% |
| Inference speed (CPU only) | 9.6 FPS |

**Per-class breakdown (test set):**

| Class | mAP@0.5 |
|---|---|
| Hardhat | 96.1% |
| NO-Hardhat | 58.6% |
| Safety Vest | 85.4% |
| NO-Safety Vest | 88.4% |
| Person | 89.1% |

`NO-Hardhat` is the weakest class, almost certainly due to having the fewest training instances of the five. This was identified by comparing per-class precision-recall curves and the confusion matrix (both included in `/results`) — a natural next step would be augmenting this class with more examples.

## Dataset

[Construction Site Safety Image Dataset](https://www.kaggle.com/datasets/snehilsanyal/construction-site-safety-image-dataset-roboflow) by Snehil Sanyal, originally sourced from Roboflow Universe, licensed CC BY 4.0. The original dataset has 10 classes across 2,801 images; this project filters it down to the 5 classes above (Hardhat, NO-Hardhat, Safety Vest, NO-Safety Vest, Person), keeping 2,535/84/60 images across train/valid/test.

## How it was built

1. Filtered the source dataset's YOLO-format labels down to the 5 target classes, re-indexing class IDs and dropping any image left with zero relevant annotations
2. Fine-tuned `yolov8s.pt` for 50 epochs (early stopping patience=10) at 640px resolution on a T4 GPU via Kaggle
3. Validated on the held-out test split to get an unbiased mAP reading
4. Benchmarked CPU-only inference speed locally to confirm real-time feasibility without GPU acceleration

Training config (full details in `args.yaml`): default YOLOv8 augmentation pipeline (mosaic, HSV color jitter, horizontal flip), batch size 16, image size 640.

## Running it

**Requirements:**
```bash
pip install ultralytics opencv-python pillow
```

**GUI app** (image upload / webcam capture / live feed detection):
```bash
python3 app5.py
```

**Just the model, no GUI:**
```python
from ultralytics import YOLO
model = YOLO("best.pt")
results = model.predict("path/to/image.jpg", conf=0.25)
```

## Repo contents

- `app5.py` — Tkinter GUI application (image upload, webcam capture, live video feed modes)
- `best.pt` — fine-tuned YOLOv8s weights
- `results/` — confusion matrix, precision-recall curve, class distribution plot, training args
- `benchmark.py` — CPU inference speed benchmark script

## What this doesn't do (yet)

- No training/fine-tuning script included in this repo (training was done in a Kaggle notebook — the dataset filtering logic is straightforward YOLO label re-indexing, happy to share on request)
- Not deployed as a hosted API/web service — runs locally only
- `NO-Hardhat` detection is noticeably weaker than other classes due to class imbalance in the source dataset
