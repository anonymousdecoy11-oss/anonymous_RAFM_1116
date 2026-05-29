# RAFM: Routing-Aware Feature Module for PCB Defect Detection

RAFM is a lightweight backbone plugin for YOLO-family detectors that encodes PCB Manhattan routing geometry through four Sobel-initialised depthwise filter banks, raising mAP50-95 from 69.20 to 76.50 on PKU-Market-PCB with only 0.09M additional parameters.

---

## Repository Structure

```
ultralytics/nn/modules/block.py      ← RAFM class implementation
ultralytics/nn/tasks.py              ← RAFM registered
train_pcb.py                         ← training script
```

---

## Where RAFM is Implemented

**`ultralytics/nn/modules/block.py`**
Search for `class RAFM` — the full module including directional filter banks, anomaly scorers, cross-direction attention, and gated residual.

**`ultralytics/nn/tasks.py`**
Search for `RAFM` — imported and registered so the YAML parser can resolve it by name.

---

## Adding RAFM to a YAML Config

Insert RAFM after the C3k2 blocks at P2 and P3 in the backbone. Use `[]` for args so channel width is inferred automatically:

```yaml
backbone:
  - [-1, 1, Conv,  [64, 3, 2]]
  - [-1, 1, Conv,  [128, 3, 2]]
  - [-1, 2, C3k2,  [256, False, 0.25]]
  - [-1, 1, RAFM,  []]               # ← insert here (P2, stride 4)
  - [-1, 1, Conv,  [256, 3, 2]]
  - [-1, 2, C3k2,  [512, False, 0.25]]
  - [-1, 1, RAFM,  []]               # ← insert here (P3, stride 8)
  ...
```

Update all downstream layer indices in the head accordingly (+1 per RAFM insertion).

---

## Training

```bash
python train_pcb.py
```

Key hyperparameters used:

| Setting | Value |
|---|---|
| Epochs | 400 |
| Batch size | 32 |
| Input size | 640×640 |
| Optimiser | MuSGD (lr=0.01, cosine annealing) |
| Seed | 42 |

---

## Results on PKU-Market-PCB

| Model | Params (M) | mAP50 (%) | mAP50-95 (%) |
|---|---|---|---|
| YOLO26n | 2.51 | 99.10 | 69.20 |
| **YOLO26n + RAFM** | **2.60** | **99.29** | **76.50** |

## Results on DsPCBSD

| Model | Params (M) | mAP50 (%) | mAP50-95 (%) |
|---|---|---|---|
| YOLO26n | 2.51 | 99.15 | 68.20 |
| **YOLO26n + RAFM** | **2.60** | **99.48** | **77.20** |
