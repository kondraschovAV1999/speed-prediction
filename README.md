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

## Configuration

Configuration is Hydra-based. `configs/default.yaml` is the root; `dataset`,
`model`, and `experiment` are config groups. Downstream code receives typed
dataclasses (`src.config.Config`) — never raw dicts.

- `dataset=synthetic` — synthetic data parameters.
- `model=tiny_cnn_gru` — model architecture.
- `experiment=baseline|smoke` — run presets.

Overrides use dot paths, e.g. `postprocessing.stationary.enter_threshold_mph=1.5`.

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
