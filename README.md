# Speed Estimation Pipeline

Vehicle speed estimation from video frames: a complete, tested, CPU-runnable
vertical slice on **synthetic data** (M1). A tiny CNN+GRU regresses per-frame
speed (mph) from a sliding window of frames; a causal post-processing stage
(clip → smoothing → hysteresis stationary detector) turns raw predictions into
a deployable streaming estimator.

## Install

The canonical environment manager is Conda (the development harness manages a
per-project environment automatically; `environment.yml` documents the
reproducible spec).

```bash
conda env create -f environment.yml
conda activate speed-estimation
python -m pip install -e ".[dev]"
```

## Quick start

```bash
python scripts/train.py                     # baseline synthetic run
python scripts/train.py experiment=smoke    # fast CI-style run
python scripts/train.py training.learning_rate=0.001   # override any key
```

## Commands

| Command | Purpose |
| --- | --- |
| `python scripts/train.py` | Full train + evaluate + benchmark + figures for one run |
| `python scripts/evaluate.py run=<name>` | Re-run evaluation from a saved checkpoint |
| `python scripts/predict.py model.checkpoint=<path> input=<dir>` | Stream predictions over a frame folder |
| `python scripts/visualize.py run=<name>` | Regenerate figures from a run's `predictions.csv` |

## comma2k19 baselines (real data)

Models are composed from three Hydra groups — `encoder`, `temporal`, `head` —
selected on the CLI (no file per combination).

- `encoder=` `tiny_cnn`, `resnet18`, `resnet50`, `mobilenet_v3_small`, `efficientnet_b0`, `vit_b_16`, `deeplabv3_resnet50`
- `temporal=` `gru`, `lstm`, `convlstm`
- `head=` `linear`, `mlp`

| Baseline | `preprocessing.image_size` |
| --- | --- |
| `encoder=tiny_cnn temporal=lstm` | `[64, 64]` |
| `encoder=resnet18 temporal=lstm` | `[128, 128]` |
| `encoder=resnet18 temporal=convlstm` | `[128, 128]` |
| `encoder=vit_b_16 temporal=lstm` | `[224, 224]` |

Workflow:

```bash
# 1. Decode + cache raw data (HEVC -> 128x128 RGB .npz + manifest). ~13 GiB for Chunk_1.
python scripts/preprocess.py dataset=comma2k19
# pilot: python scripts/preprocess.py dataset=comma2k19 data.comma2k19.max_segments=10

# 2. Inspect the cache (counts, 20 Hz, speed stats, split sizes).
python scripts/inspect_data.py dataset=comma2k19

# 3. Train a baseline (add data.window_stride=1 for full overlap; default is 5).
python scripts/train.py dataset=comma2k19 encoder=tiny_cnn temporal=lstm preprocessing.image_size=[64,64]
python scripts/train.py dataset=comma2k19 encoder=resnet18 temporal=lstm preprocessing.image_size=[128,128]
python scripts/train.py dataset=comma2k19 encoder=resnet18 temporal=convlstm preprocessing.image_size=[128,128]
python scripts/train.py dataset=comma2k19 encoder=vit_b_16 temporal=lstm preprocessing.image_size=[224,224]
```

Notes:

- Cache lives in `data/interim/comma2k19/` (uncompressed `.npz`, memory-mapped at
  load). Change `data.comma2k19.cache_image_size` to re-cache at another resolution.
- `data.window_stride` applies to training only; validation/test are always
  stride 1 (required by the frame-aligned metrics and stationary detector).
- ResNet/ViT weights download once (ImageNet) and are cached by torchvision; point
  `model.encoder.weights_path` at a local `.pth` to run offline.
- `vit_b_16` uses a frozen ViT-B/16 at 224×224 (CLS token); it is ~5–10× heavier
  per frame than ResNet18. `encoder=vit_b_32` is the cheaper fallback.
- The regression head is `head=linear` (single layer) or `head=mlp`; tune it via
  `head.hidden_dims=[...]`, `head.activation=relu|gelu|tanh|...`, `head.dropout=0.1`.
- `convlstm` needs a spatial-capable encoder (`resnet18`); `feature_dim` defaults to
  512. Use `model.encoder.feature_dim=32` for a lighter ConvLSTM.
- Each run exports a standalone `model.pt` (no Lightning wrapper) alongside the
  Lightning checkpoints.
- Evaluation/inference device is `evaluation.device` (default `auto` -> GPU when
  available).

## Configuration

Configuration is Hydra-based. `configs/default.yaml` is the root; `dataset`,
`encoder`, `temporal`, `head`, and `experiment` are config groups. Downstream code
receives typed dataclasses (`src.config.Config`) — never raw dicts.

- `dataset=synthetic` — synthetic data parameters.
- `encoder=resnet18|resnet50|vit_b_16|...` — spatial encoder.
- `temporal=lstm|gru|convlstm` — temporal aggregator.
- `head=linear|mlp` — regression head.
- `experiment=baseline|smoke` — run presets.

Overrides use dot paths, e.g. `postprocessing.stationary.enter_threshold_mph=1.5`,
or `model.encoder.feature_dim=32`.

See [`docs/CONFIGURATION.md`](docs/CONFIGURATION.md) for the full configuration guide.

## Output contract

Every run writes under `output/runs/<run_name>/` (plus `output/logs/` and
`output/figures/`):

```text
output/runs/<run_name>/
├── config.yaml          # resolved config
├── metadata.json        # env + git + seed + status + duration + best ckpt
├── metrics.json         # canonical metrics (schema_version: 1)
├── predictions.csv      # canonical predictions table
├── split.json           # split manifest actually used
├── checkpoints/best.ckpt (+ last.ckpt)
├── training_logs/       # Lightning CSVLogger curves
└── reports/latency_samples.csv
```

`metrics.json` reports overall + per-bucket regression metrics (MAE/RMSE/R²/Median
AE), stationary precision/recall/F1 with confusion counts, and `model_only` /
`end_to_end` latency summaries.

## Assumptions (open decisions D1–D8)

See `docs/DECISIONS.md` for the full record. Highlights:

- **D1** Labels in mph; synthetic only for M1.
- **D2** 10 Hz constant nominal frame rate.
- **D4** Inference is strictly causal (real-time requirement).
- **D6** "Creeping ≤ 2 mph" ground truth = label `<= 2.0`.
- **D7** Detector initial state is MOVING.

## Testing

```bash
ruff check src tests scripts
pytest -q                    # unit suite (< 60 s)
pytest -q -m integration     # integration suite (< 120 s)
```

All tests run offline on CPU; nothing downloads model weights (`pretrained: false`).
