# Speed Estimation Pipeline

A complete guide to how the pipeline works end-to-end, with a deep dive into the
**pre-processing & caching** stage (M2-a) that turns raw comma2k19 dashcam data
into fast-to-load training samples.

Companion docs: [`CONFIGURATION.md`](CONFIGURATION.md) (every config key),
[`DECISIONS.md`](DECISIONS.md) (recorded decisions), [`PLAN.md`](PLAN.md) (the
original orchestrated plan).

---

## 1. High-level overview

```text
                        ┌────────────────────────────────────────────────────┐
                        │  configs/  (Hydra: default.yaml + groups)          │
                        └───────────────────────┬────────────────────────────┘
                                                │ typed Config
                ┌───────────────────────────────┴───────────────────────────────┐
                │                                                                │
     ┌──────────▼───────────┐                                      ┌─────────────▼────────────┐
     │  PRE-PROCESSING      │   (comma2k19 only, offline, once)     │  TRAINING / EVAL /      │
     │  scripts/preprocess  │                                      │  INFERENCE              │
     └──────────┬───────────┘                                      └─────────────┬────────────┘
                │                                                              │
   raw HEVC + global_pose       decode → RGB → downscale → cache                │
                │                                                              │
     ┌──────────▼───────────┐     load cache (mmap) → SpeedSequence →           │
     │  CACHE               │──▶  validate → split → windows → SpeedDataset  ───┤
     │  data/interim/...    │                                                  │
     └──────────────────────┘                                                  │
                                                                               │
   Model: EncoderRecurrent (encoder → temporal → head)  →  raw speed (mph)     │
                                                                               │
   Post-process: clip → smoothing → stationary detector  →  final speed         │
                                                                               │
   Evaluate: regression (MAE/RMSE/R²) + buckets + stationary (P/R/F1)          │
   Benchmark: model-only + end-to-end latency                                  │
   Track:    MLflow (per-epoch metrics, params, artifacts)                      │
```

Two independent halves:

1. **Pre-processing (offline, run once)** — decode the heavy H.265 video into a
   compact, memory-mapped cache. `scripts/preprocess.py`.
2. **Training/eval/inference (online)** — read that cache (lazily), slide windows
   over it, train/evaluate a model. `scripts/train.py` + friends.

The cache is the contract between them: once built, you can sweep **any** model,
window, or hyperparameter without touching the raw video again.

---

## 2. Raw data layout (comma2k19)

```
data/raw/Chunk_1/
└── <route>/                     e.g. b0c9d2329ad1606b_2018-08-02--08-34-47
    └── <segment>/               one ~1-minute clip (index 0..N)
        ├── video.hevc           H.265, ~1200 frames, 20 fps, 874×1164
        ├── preview.png
        ├── raw_log.bz2          (unused — CAN bus, future work)
        ├── processed_log/       CAN/GNSS/IMU (unused)
        └── global_pose/         ← the labels
            ├── frame_times         (1200,) float64   boot-time seconds, dt = 0.05 s
            ├── frame_velocities    (1200, 3) float64  ECEF velocity (m/s)
            ├── frame_positions     (1200, 3) float64
            ├── frame_orientations  (1200, 4) float64
            └── frame_gps_times     (1200, 2) float64
```

**Gotchas that matter:**

- **`global_pose` files have NO `.npy` extension** — they're NumPy `.npy`-format
  binary, just named `frame_times`, `frame_velocities`, etc. (`np.load` sniffs the
  format from magic bytes, so the missing extension is fine).
- **Never trust OpenCV's frame count / fps metadata** (it's garbage for HEVC) —
  the frame count always comes from `len(frame_times)`.
- **Label source** is `global_pose/frame_velocities` — comma.ai's tightly-coupled
  INS/GNSS/vision optimizer (Laika). Speed = magnitude of the velocity vector.
  (Caveat: because it fuses *vision*, the label isn't 100% independent of the
  frames — recorded in `DECISIONS.md`.)

Only `video.hevc` + `frame_times` + `frame_velocities` are consumed. Everything
else is kept for future phases (CAN baselines, etc.).

---

## 3. Pre-processing & caching (the focus)

### 3.1 Why cache at all?

The raw data is ~37 MB of H.265 **per segment** (≈ 9 GB for Chunk_1), and
decoding H.265 is slow (~5 s per segment). If you re-decoded on every training
run — or worse, every epoch — training would be dominated by video decoding.
Worse, holding hundreds of decoded frames in RAM is infeasible.

The cache solves both problems:

1. **Decode once, reuse forever** — expensive decode happens exactly once.
2. **Memory-map instead of load** — the cache is read lazily, so only the frames
   actually touched by a batch are paged into RAM.

### 3.2 The two-stage flow

```text
Stage 1 (preprocess.py, run once):
  video.hevc ──cv2 decode──▶ frames (H×W×3 BGR)
                             │  BGR→RGB, resize to cache_image_size (128×128)
                             ▼
              frames.npy  (uint8 [N,128,128,3])   +   meta.npz (speeds, timestamps)

Stage 2 (build_datasets, every run):
  frames.npy ──np.load(mmap_mode='r')──▶ lazy memmap, no RAM cost
```

### 3.3 Cache format (per segment)

| File | Contents |
| --- | --- |
| `<route>__<seg>.frames.npy` | `uint8 [N, H', W', 3]` RGB, downscaled to `cache_image_size` (default 128²) |
| `<route>__<seg>.meta.npz` | `speeds_mph` `float64 [N]` + `timestamps` `float64 [N]` (tiny) |
| `manifest.json` | maps each `<route>/<seg>` → `{frames_file, meta_file, n_frames, group_id}` |

Why two files? **Frames are the big array (~59 MB each)** and need to be
memory-mapped; speeds/timestamps are tiny (~10 KB) and are just loaded.

### 3.4 Why `.npy` and not `.npz` (a subtle, important detail)

NumPy **cannot memory-map arrays stored inside a `.npz`** (a zip archive) —
`np.load("x.npz", mmap_mode="r")` silently loads them fully into RAM. With ~190
segments that's ~11 GB → out-of-memory.

A plain **`.npy`** file *does* support `mmap_mode="r"`, returning a true lazy
`np.memmap`. That's why frames are stored as `.npy`:

```python
frames = np.load(path, mmap_mode="r")   # np.memmap, lazy — near-zero RAM
frames[[0, 7, 42]]                       # only these frames are paged in
```

This was a real bug found and fixed (`be5fb1e`): the original `.npz` cache
exhausted memory on the full dataset.

### 3.5 `CachedFrameSource` (the lazy reader)

```python
class CachedFrameSource:
    def __init__(self, path):
        self._f = np.load(path, mmap_mode="r")   # lazy memmap
        self._n = self._f.shape[0]

    def __len__(self): return self._n
    def get(self, indices):                       # uint8 [k, H, W, C]
        return np.asarray(self._f[list(indices)])
```

It also implements `__getstate__`/`__setstate__` so it's **picklable** (serialize
only the path, reopen the mmap in the worker) — required for
`training.num_workers > 0` on Windows, where DataLoader workers are spawned
processes that receive the dataset by pickle.

### 3.6 What happens in `cache_segment` (per segment)

```python
frame_times   = np.load(global_pose / "frame_times")       # no .npy extension
velocities    = np.load(global_pose / "frame_velocities")
speeds_mph    = np.linalg.norm(velocities, axis=1) * MPH_PER_MPS   # m/s → mph
frames        = decode_segment_frames(segment_dir, image_size)     # BGR→RGB, resize

assert frames.shape[0] == frame_times.shape[0]   # fail loudly on mismatch

np.save(frames_path, frames)                       # .npy → mmap-able
np.savez(meta_path, speeds_mph=speeds_mph, timestamps=frame_times)
```

- `MPH_PER_MPS = 2.2369362921` (exact m/s → mph conversion).
- `decode_segment_frames` uses OpenCV (imported **lazily** — it's in the
  `video` extra, not required for offline tests).

### 3.7 Disk footprint & knobs

| `cache_image_size` | per segment | full Chunk_1 (~252 segments) |
| --- | --- | --- |
| 128×128 (default) | ~59 MB | ~15 GB |
| 64×64 | ~15 MB | ~3.7 GB |
| 32×32 | ~3.7 MB | ~0.9 GB |

- `data.comma2k19.cache_image_size` — re-cache at a different resolution.
- `data.comma2k19.max_segments` — only process the first N segments (⚠️ takes the
  first N **contiguously**, all from route #1 — not a splittable pilot; see below).

Run it:

```bash
python scripts/preprocess.py dataset=comma2k19                          # full
python scripts/preprocess.py dataset=comma2k19 data.comma2k19.max_segments=10  # pilot
python scripts/inspect_data.py dataset=comma2k19                         # verify cache
```

---

## 4. Data pipeline (online, every run)

### 4.1 `SpeedSequence` (the unit of data)

```python
SpeedSequence(
    sequence_id = "<route>/<seg>",     # unique
    group_id    = "<route>",           # SPLITS ARE MADE ON THIS
    frames      = CachedFrameSource,   # lazy mmap
    speeds_mph  = float64 [N],
    timestamps  = float64 [N],         # seconds, strictly increasing
)
```

`group_id` = route is the crucial bit: **splits are made on routes**, not frames
or windows, so a whole unseen drive ends up in exactly one split (no leakage).

### 4.2 Validation (`validate_sequence`)

Applies the edge-case policies: time gaps (`dt > max_gap_factor/frame_rate`) split
the sequence into `<id>#<k>` children; duplicate timestamps raise or drop;
negative/over-max speeds raise; irregular rate warns. Fails loudly — never
silently alters data.

### 4.3 Splits

`make_group_split` deterministically shuffles sorted unique `group_id`s and
partitions by `split_ratios` (default 0.6/0.2/0.2). `assert_no_leakage` guarantees
no route appears in two splits. The manifest is cached to disk and reused for
reproducibility; a copy is written into each run dir.

### 4.4 Sliding windows (the core temporal idea)

A **window** is `sequence_length` (=`T`) frames ending at frame `t`; the **label is
the speed at `t`** (causal — no future frames). Two more knobs:

| Knob | Meaning |
| --- | --- |
| `frame_step` (=k) | take every k-th frame *inside* the window (temporal subsampling) |
| `window_stride` | how far `t` advances between consecutive windows |

Window ending at `t` contains frames `[t-(T-1)k, …, t-k, t]` (ascending). At 20 Hz
with `T=10, k=5`, each window is 0.5 s of history → predict current speed.

**Per-split stride (important):** `build_datasets` uses `window_stride` for
**train only**, and **locks val/test to stride 1**:

- *Training* at stride 8 → ~8× fewer (redundant) samples → faster epochs. Fine,
  because the model is stride-agnostic (see below).
- *Val/test* at stride 1 → a dense prediction for **every frame**, required for the
  frame-aligned metrics and the frame-by-frame stationary detector.

```python
for split in ("train", "val", "test"):
    stride = data.window_stride if split == "train" else 1
    windows = _build_split_windows(sequences, data_cfg, stride=stride)
```

### 4.5 Per-frame transform

```text
uint8 [H,W,C]  → Resize(image_size) → ToTensor (→ [C,H,W] float in [0,1]) → Normalize(mean, std)
```

`mean=[0.485,0.456,0.406]`, `std=[0.229,0.224,0.225]` are the **ImageNet RGB
constants** (hardcoded, not fitted) — required so the pretrained ResNet/ViT weights
are used correctly. `image_size` must match the encoder (64 for `tiny_cnn`, 128 for
the ResNet/MobileNet/EfficientNet backbones, 224 for `vit_b_16`).

### 4.6 `SpeedDataset` / DataLoader

`SpeedDataset.__getitem__` returns `{frames: [T,C,H,W], target_mph, timestamp,
frame_index, sequence_id}`. Train loader is shuffled with `drop_last=True`; val/test
are unshuffled. `persistent_workers=True` when `num_workers > 0` (avoids
re-spawning workers every epoch on Windows).

---

## 5. Models

One generic class composes everything:

```text
EncoderRecurrent:  encoder → (per-frame features) → temporal → head → [B]
```

- **Encoders** (`encoder=`, → a per-frame *vector* or *spatial map*):
  - `tiny_cnn` — small from-scratch CNN → 64-d vector.
  - `resnet18` / `resnet50` / `mobilenet_v3_small` / `efficientnet_b0` — frozen
    ImageNet backbones (head removed), global-pooled to their native dim.
  - `vit_b_16` — frozen ImageNet ViT-B/16, CLS token → 768-d.
- **Temporal** (`temporal=`, over the T per-frame features):
  - `gru`, `lstm` (vector path), `convlstm` (spatial path).
- **Head** (`head=`, regression):
  - `linear` — single `Linear(hidden → 1)`.
  - `mlp` — configurable `hidden_dims`/`activation`/`dropout`.
- `spatial` is inferred from `temporal=convlstm`; `forward` dispatches on whether
  the encoder output is 2-D (vector → LSTM/GRU) or 4-D (spatial → ConvLSTM).

Compose on the CLI, e.g. `encoder=resnet50 temporal=gru head=mlp`. See
`src/models/architectures.py` and `configs/{encoder,temporal,head}/*.yaml`.

---

## 6. Training

`run_experiment` (`src/experiment.py`) orchestrates, in order:

1. validate config → create run dir → seed → metadata → start MLflow run + log params.
2. build datasets/loaders → build model → log param count.
3. `trainer.fit(...)` (train/val only; **test untouched**).
   - Loss: `mse` (mph²) or `l1` (mph). Optimizer: AdamW.
   - `ModelCheckpoint` on `val/mae_mph` (best + last), `EarlyStopping`.
   - `TrackerMetricsCallback` forwards `train/loss_epoch`, `val/loss`, `val/mae_mph`
     to MLflow **per epoch**.
4. load best checkpoint → predict on `splits_to_evaluate` (val, test) →
   post-process → evaluate.
5. benchmark (model-only + end-to-end latency) → write artifacts → MLflow metrics/artifacts.

Evaluation/inference device is `evaluation.device` (default `auto` → GPU when
available).

---

## 7. Post-processing & evaluation

**Post-processing (causal, per frame):**

```text
raw mph → clip (min 0) → smoothing (none|ema|median) → stationary detector → final
```

The **stationary detector** is a hysteresis state machine: `≤ enter_threshold`
(2.0 mph) for `enter_persistence_frames` consecutive frames → STOPPED; `>
exit_threshold` (2.5) for N frames → MOVING. It emits an independent
`stationary_pred` flag and no longer zeroes `final_speed_mph`. It's causal and
`reset()` at each sequence boundary.

**Metrics** (on the final predictions): regression MAE/RMSE/R²/Median-AE (overall +
per-bucket), plus stationary precision/recall/F1 + confusion counts. Full results
in `metrics.json`.

**Streaming inference** (`StreamingSpeedEstimator`) runs the *same* code path —
transform → model → post-process — one frame at a time, returning `None` until the
buffer holds a full window.

---

## 8. Outputs of a run

```text
output/runs/<name>/
├── config.yaml          metadata.json          metrics.json
├── predictions.csv      split.json
├── checkpoints/best.ckpt  last.ckpt
├── model.pt             # standalone export (no Lightning wrapper)
├── training_logs/       # CSVLogger curves
└── reports/latency_samples.csv

output/mlflow/mlflow.db  # MLflow SQLite store (view: mlflow ui --backend-store-uri sqlite:///output/mlflow/mlflow.db)
```

---

## 9. End-to-end commands

```bash
# 1. Cache the data once
python scripts/preprocess.py dataset=comma2k19
python scripts/inspect_data.py dataset=comma2k19

# 2. Train (encoder + image_size must match)
python scripts/train.py dataset=comma2k19 encoder=tiny_cnn  temporal=lstm     preprocessing.image_size=[64,64]
python scripts/train.py dataset=comma2k19 encoder=resnet18  temporal=lstm     preprocessing.image_size=[128,128]
python scripts/train.py dataset=comma2k19 encoder=resnet18  temporal=convlstm preprocessing.image_size=[128,128]

# 3. Browse experiments
mlflow ui --backend-store-uri sqlite:///output/mlflow/mlflow.db
```
