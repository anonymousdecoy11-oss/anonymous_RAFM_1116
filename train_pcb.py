import os
import random
import time
import numpy as np
import torch
from pathlib import Path

# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
from ultralytics import YOLO
# model = YOLO(f'ultralytics/cfg/models/v12/yolov12{MODEL_SIZE}.yaml')
# ── Change only these between experiments ──────────────────────────────────
MODEL_YAML  = 'ultralytics/cfg/models/12/yolo12.yaml'
EXPERIMENT  = 'NAME'

# ── Fix these forever ──────────────────────────────────────────────────────
SEED        = 42
DATA_YAML   = '/path/data.yaml'
EPOCHS      = 400
BATCH       = 32
IMGSZ       = 200
DEVICE      = '2'

 # ── Seed everything ────────────────────────────────────────────────────────
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

# ── Train ──────────────────────────────────────────────────────────────────
model   = YOLO(MODEL_YAML)
results = model.train(
    data    = DATA_YAML,
    epochs  = EPOCHS,
    batch   = BATCH,
    imgsz   = IMGSZ,
    device  = DEVICE,
    seed    = SEED,
    name    = EXPERIMENT,

#     # ── Optimizer (fixed) ──────────────────────────────────────────────────
#     # optimizer    = 'SGD',
#     # lr0          = 0.01,
#     # lrf          = 0.01,
#     # momentum     = 0.937,
#     # weight_decay = 0.0005,
#     # warmup_epochs    = 3.0,
#     # warmup_momentum  = 0.8,
#     # warmup_bias_lr   = 0.1,
#     # cos_lr           = True,

 # ── Loss weights (fixed) ───────────────────────────────────────────────
     # box = 7.5,
     # cls = 0.5,
     # dfl = 1.5,

#     # # ── Augmentation (fixed) ───────────────────────────────────────────────
#     # degrees    = 90.0,
#     # fliplr     = 0.5,
#     # flipud     = 0.5,
#     # mosaic     = 0.5,
#     # copy_paste = 0.3,
#     # hsv_h      = 0.0,
#     # hsv_s      = 0.3,
#     # hsv_v      = 0.4,
#     # translate  = 0.1,
#     # scale      = 0.5,
#     # shear      = 0.0,
#     # mixup      = 0.0,
)
print(f"\nTraining complete!")

# ── Test ───────────────────────────────────────────────────────────────────
best = Path(f'runs/path/weights/best.pt')
if not best.exists():
    print(f"\nWeights not found at {best}")
else:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    print(f"\nRunning test set evaluation...")
    test_model   = YOLO(str(best))
    test_results = test_model.val(
        data   = DATA_YAML,
        split  = 'test',
        imgsz  = IMGSZ,
        device = DEVICE,
        name   = f'{EXPERIMENT}_test',
    )

    # ── Accuracy ───────────────────────────────────────────────────────────
    map50   = test_results.results_dict.get('metrics/mAP50(B)',    0)
    map5095 = test_results.results_dict.get('metrics/mAP50-95(B)', 0)
    print(f"\n{'─'*40}")
    print(f"mAP50      : {map50:.4f}")
    print(f"mAP50-95   : {map5095:.4f}")

    # ── Model size ─────────────────────────────────────────────────────────
    size_mb = best.stat().st_size / 1e6
    params  = sum(p.numel() for p in test_model.model.parameters())
    print(f"\nModel size : {size_mb:.2f} MB")
    print(f"Parameters : {params/1e6:.2f} M")

    # ── Latency and FPS ────────────────────────────────────────────────────
    print(f"\nMeasuring latency...")
    dummy = torch.zeros(1, 3, IMGSZ, IMGSZ).cuda()
    net   = test_model.model.eval().cuda()

    with torch.no_grad():
        for _ in range(50):
            _ = net(dummy)

    torch.cuda.synchronize()
    times = []
    with torch.no_grad():
        for _ in range(200):
            t0 = time.perf_counter()
            _  = net(dummy)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

    times   = sorted(times)[10:-10]
    mean_ms = sum(times) / len(times) * 1000
    p95_ms  = times[int(len(times) * 0.95)] * 1000
    fps     = 1000 / mean_ms
    print(f"Latency (mean) : {mean_ms:.2f} ms")
    print(f"Latency (p95)  : {p95_ms:.2f} ms")
    print(f"FPS            : {fps:.1f}")
    print(f"{'─'*40}")
