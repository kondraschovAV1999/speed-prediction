# Configuration Guide

How to configure the speed-estimation pipeline: the config system, every
section, the comma2k19 (real-data) workflow, and common overrides.

---

## 1. How configuration works

Configuration is **Hydra**-based. `configs/default.yaml` is the root config and
selects a **dataset**, a **model**, and an **experiment** from config *groups*:

```text
configs/
├── default.yaml              # root: paths, preprocessing, training, postprocessing,
│                             #       evaluation, benchmark, tracking, logging
├── dataset/
│   ├── synthetic.yaml        # synthetic striped-video data (M1)
│   └── comma2k19.yaml        # real comma2k19 dashcam data (M2)
├── model/
│   ├── tiny_cnn_gru.yaml     # M1 skeleton
│   ├── cnn_lstm.yaml
│   ├── resnet_lstm.yaml
│   └── resnet_convlstm.yaml
└── experiment/
    ├── baseline.yaml         # normal run
    └── smoke.yaml            # fast CI-style run
```

You override anything on the command line with dot paths:

```bash
python scripts/train.py dataset=comma2k19 model=resnet_lstm preprocessing.image_size=[128,128] training.epochs=10
```

Downstream code always receives a **typed config** (`src.core.config.Config`) —
never raw dicts — so a typo or wrong type raises a clear `ConfigError`.

---

## 2. End-to-end workflows

### 2.1 Synthetic (M1, no data required)

```bash
python scripts/train.py                              # baseline (tiny_cnn_gru)
python scripts/train.py experiment=smoke             # fast CI-style run
python scripts/train.py training.learning_rate=0.001 # override anything
```

### 2.2 comma2k19 (real data)

```bash
# 1. Decode + cache the raw dataset once (HEVC -> RGB .npz + manifest).
#    ~13 GiB for all of Chunk_1; ~5.5 s per segment (~20-25 min for 252 segments).
python scripts/preprocess.py dataset=comma2k19
python scripts/preprocess.py dataset=comma2k19 data.comma2k19.max_segments=10   # pilot

# 2. Inspect the cache (counts, 20 Hz check, speed stats, split sizes).
python scripts/inspect_data.py dataset=comma2k19

# 3. Train a baseline. The cache is NOT touched by window/model/hyperparameter
#    changes, so you can sweep these freely without re-running preprocess.
python scripts/train.py dataset=comma2k19 model=cnn_lstm        preprocessing.image_size=[64,64]
python scripts/train.py dataset=comma2k19 model=resnet_lstm     preprocessing.image_size=[128,128]
python scripts/train.py dataset=comma2k19 model=resnet_convlstm preprocessing.image_size=[128,128]
```

---

## 3. Reference — every config section

### 3.1 `paths`

```yaml
paths:
  output_root: output      # ALL generated artifacts live under here
  data_root: data
```

### 3.2 `experiment`

```yaml
experiment:
  name: baseline           # used in run-dir names
  seed: 42                 # data splits + RNG
  deterministic: false     # true => torch.use_deterministic_algorithms
```

### 3.3 `preprocessing` (per-frame transform)

```yaml
preprocessing:
  image_size: [32, 32]              # [H, W] input to the model (per-model, see §4)
  normalize: true
  mean: [0.485, 0.456, 0.406]       # ImageNet RGB mean  (FIXED — not fitted on data)
  std:  [0.229, 0.224, 0.225]       # ImageNet RGB std   (FIXED — not fitted on data)
```

- The pipeline runs `Resize -> ToTensor -> Normalize` per frame.
- `mean`/`std` are the **ImageNet normalization constants**, hardcoded (never
  computed from your data). They are required so that the pretrained ResNet18
  weights are used correctly; frames are RGB, so the RGB constants apply.
- `image_size` is the only preprocessing knob that must **match the model** (§4).

### 3.4 `data` (provided by the dataset group)

Shared keys (both datasets):

| Key | Meaning |
| --- | --- |
| `source` | `synthetic` or `comma2k19` |
| `root` | data directory |
| `frame_rate_hz` | nominal frame rate (10 synthetic / 20 comma2k19) |
| `split_manifest` | where the split manifest is written/cached |
| `split_ratios` | `{train, val, test}` fractions (must sum to 1) |
| `sequence_length` | `T` = frames per sliding window (model's history) |
| `window_stride` | training-window overlap dial (**train split only**) |
| `frame_step` | take every k-th frame inside a window |
| `max_gap_factor` | time gaps > factor/frame_rate split the sequence |
| `on_duplicate_timestamp` | `raise` or `drop_later` |
| `allow_negative_speed` | `false` |
| `max_speed_mph` | validation ceiling |

`synthetic` sub-block: `num_sequences`, `frames_per_sequence`, `image_size`,
`noise_std`.

`comma2k19` sub-block:

```yaml
comma2k19:
  speed_source: global_pose   # label source (frame_velocities magnitude)
  use_cache: true             # false => decode video on the fly (slow)
  cache_dir: data/interim/comma2k19
  cache_image_size: [128, 128]   # resolution stored on disk during preprocessing
  max_segments: null          # int => only cache/use the first N segments
```

**Sliding window semantics:** a window is `sequence_length` frames ending at
frame `t`; the label is the speed at `t` (causal — no future frames).
`window_stride` controls overlap:

- applies to **training only**; val/test are always stride 1 (required by the
  frame-aligned metrics and the frame-by-frame stationary detector),
- larger stride = fewer training windows = faster epochs (less redundant data),
- does **not** affect inference cost, and does **not** require re-preprocessing.

### 3.5 `model` (provided by the model group)

```yaml
name: cnn_lstm                # registry key in src/models/architectures.py
encoder: {name: tiny_cnn, pretrained: false, feature_dim: 64}
temporal: {name: lstm, hidden_size: 32, num_layers: 1}
checkpoint: null              # optional path to resume from
```

The generic model composes `encoder -> per-frame features -> temporal -> head`.
Encoders/temporal modules are registered by name (see §4).

### 3.6 `training`

```yaml
training:
  batch_size: 32
  epochs: 5
  learning_rate: 0.0001
  weight_decay: 0.00001
  optimizer: adamw
  loss: mse                  # mse | l1
  monitor: val/mae_mph
  early_stopping_patience: 5
  gradient_clip_val: null
  accelerator: auto          # auto -> GPU when available
  devices: 1
  precision: "32-true"
  num_workers: 0
```

### 3.7 `postprocessing` (causal, applied after the model)

```yaml
postprocessing:
  clip:      {enabled: true, min_mph: 0.0, max_mph: null}
  smoothing: {method: none, ema_alpha: 0.5, median_window: 5}   # none|ema|median
  stationary:                # hysteresis detector: speed <= enter for N frames => STOPPED
    enabled: true
    enter_threshold_mph: 2.0
    exit_threshold_mph: 2.5
    enter_persistence_frames: 3
    exit_persistence_frames: 3
    initial_state: moving
```

Pipeline per frame: `raw -> clip -> smoothing -> stationary -> final`. When
stationary, `final_speed_mph` is exactly `0.0`.

### 3.8 `evaluation`

```yaml
evaluation:
  stationary_label_threshold_mph: 2.0   # ground-truth "stopped" cutoff
  speed_buckets:
    - {name: low,    min_mph: 0,  max_mph: 15}
    - {name: medium, min_mph: 15, max_mph: 45}
    - {name: high,   min_mph: 45, max_mph: null}
  splits_to_evaluate: [val, test]
  device: auto                          # eval/inference device (auto -> GPU)
```

### 3.9 `benchmark`

```yaml
benchmark:
  enabled: true
  device: auto
  num_warmup: 10
  num_iters: 100
```

### 3.10 `tracking`

```yaml
tracking:
  enabled: true
  backend: mlflow
  mlflow:
    tracking_uri: null          # null => local store under output/mlflow
    experiment_name: speed-estimation
```

### 3.11 `logging`

```yaml
logging:
  level: INFO                   # DEBUG | INFO | WARNING | ERROR
```

---

## 4. Models and required `preprocessing.image_size`

| Model | Encoder | Temporal | `preprocessing.image_size` |
| --- | --- | --- | --- |
| `tiny_cnn_gru` | tiny from-scratch CNN (→ 64-d vector) | GRU | `[32, 32]` |
| `cnn_lstm` | tiny from-scratch CNN (→ 64-d vector) | LSTM | `[64, 64]` |
| `resnet_lstm` | **frozen** pretrained ResNet18 (global pool → 512-d) | LSTM | `[128, 128]` |
| `resnet_convlstm` | **frozen** pretrained ResNet18 (spatial map → 32 ch) | ConvLSTM | `[128, 128]` |

ResNet18 `encoder` keys: `pretrained: true` (downloads ImageNet weights once,
cached by torchvision), `frozen: true` (backbone `requires_grad=False` + `eval()`),
`spatial: true|false`, `weights_path: null` (point at a local `.pth` to run offline).

**Note on resolution:** the cache is stored at `comma2k19.cache_image_size`
(default 128×128). `preprocessing.image_size` resizes at load time, so the model
input can be ≤ the cache size without re-preprocessing. If you want native 224×224
ResNet, re-cache with `data.comma2k19.cache_image_size=[224,224]` first.

---

## 5. Common overrides (examples)

```bash
# Data / windows (no re-preprocessing needed)
data.sequence_length=12
data.window_stride=8
data.frame_step=2
data.split_ratios='{train:0.7,val:0.15,test:0.15}'

# Subset the dataset
data.comma2k19.max_segments=50

# Training
training.epochs=20
training.learning_rate=0.001
training.batch_size=16
training.loss=l1

# Eval / device
evaluation.device=cuda
benchmark.enabled=false
tracking.enabled=false
```

---

## 6. Outputs

Each run writes under `output/runs/<name>/`:

```text
config.yaml  metadata.json  metrics.json  predictions.csv  split.json
checkpoints/best.ckpt + last.ckpt
model.pt                    # standalone export (no Lightning wrapper)
training_logs/  reports/latency_samples.csv
```

plus `output/logs/<name>.log` and figures under `output/figures/<name>/`.

---

## 7. Troubleshooting

- **"Manifest not found … run scripts/preprocess.py first"** — you're training
  with `dataset=comma2k19` before caching the data. Run `preprocess.py` first.
- **No log output on screen** — fixed: Hydra disabled import-time loggers;
  `setup_logging` re-enables them. If you ever see silent scripts again, check
  `logging.level`.
- **`OpenCV is required …`** — install the optional extra: `pip install -e ".[video]"`.
- **ResNet download fails / offline** — set `encoder.weights_path` to a local
  `.pth` (or run once online so torchvision caches it).
- **Slow preprocessing** — it's decoding H.265; reduce work with
  `data.comma2k19.max_segments=N` or `data.comma2k19.cache_image_size=[64,64]`.
