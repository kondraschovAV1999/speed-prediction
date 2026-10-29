# Speed Estimation Pipeline — Orchestrated Implementation Plan (v2)

> Save this file in the repo as `docs/PLAN.md`. It supersedes `speed_estimation_pipeline_plan.md`.
> Audience: (a) the **orchestrator** (a strong model or a human) who owns contracts and reviews gates, and (b) **implementer agents** who may be less capable and each get ONE work package (WP).

---

## 0. How to use this document

**Implementer agent:** read §3 (conventions), §5 (contracts), §6 (behaviour specs) and **only your own WP in §8**. Do not read ahead or implement other WPs.

**Orchestrator:** read everything. Your jobs: freeze contracts (WP0), dispatch WPs (parallel where §8.1 allows), run the gate checks in §9, resolve questions, and keep `docs/PROGRESS.md` and `docs/DECISIONS.md` current.

Why it is structured this way: weaker agents fail by (1) inventing interfaces, (2) silently changing shared code, (3) writing tests that assert nothing, (4) leaving stubs that "look done". The plan therefore pins every shared interface up front, gives each agent a narrow file-ownership list, and defines verifiable acceptance commands per WP.

---

## 1. Review findings (what changed vs. v1, and why)

| #  | Problem in v1                                                                                                                                                                                                | Resolution in v2                                                                                                                                                                                                                 |
| -- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1  | **Stationary thresholds contradict the requirement.** `threshold_mph: 2.0` coexists with `enter: 1.5` / `exit: 2.5`. With enter=1.5, a vehicle moving at 1.8 mph is never zeroed, violating "≤ 2 mph → 0.0". | One label threshold (`evaluation.stationary_label_threshold_mph = 2.0`, used only for ground truth). Detector `enter_threshold_mph = 2.0` (matches spec), `exit_threshold_mph = 2.5`. Separate enter/exit persistence. See §6.1. |
| 2  | **Post-processing was not specified as causal/stateful.** A stateful detector fed shuffled or batched predictions gives garbage; state also leaks across sequence boundaries.                                | Detector is causal, must be `reset()` at every sequence start, and offline evaluation runs the *same* `step()` code path as streaming inference. Parity test required. §6.1, §6.5.                                               |
| 3  | **Window/label semantics undefined** (which frame's speed is the label? padding?).                                                                                                                           | Window of `T` frames ending at `t`; label = speed at `t`; no padding, no lookahead. §6.2.                                                                                                                                        |
| 4  | **Split design conflict:** config has `train_path/val_path/test_path` (pre-split dirs) but plan requires sequence-level splitting at runtime.                                                                | Single dataset root + a **split manifest** keyed on `group_id` (source drive/video), saved to `data/splits/` and copied into the run dir. §6.3.                                                                                  |
| 5  | **Leakage risk beyond adjacent frames:** segments from the same drive can land in different splits.                                                                                                          | Split on `group_id`, not `sequence_id`; leakage assertion helper.                                                                                                                                                                |
| 6  | **Missing modules:** `src/pipeline.py`, benchmark and orchestration code are referenced but absent from the tree; tests listed for modules with no test file.                                                | Added `schemas.py`, `buckets.py`, `pipeline.py`, `benchmark.py`, `experiment.py`; full test inventory in §10.                                                                                                                    |
| 7  | **No shared contracts** → parallel agents would diverge.                                                                                                                                                     | Concrete dataclasses, tensor shapes, DataFrame columns and function signatures in §5.                                                                                                                                            |
| 8  | **Evaluation/visualization/reporting coupling** unclear.                                                                                                                                                     | One canonical **predictions table** (DataFrame ⇄ `predictions.csv`) is the only input to evaluation, reporting, visualization. `visualize.py run=<name>` then works from disk.                                                   |
| 9  | **Output contract inconsistent** (`metrics.json` vs `reports/evaluation.json` vs optional `report.html`).                                                                                                    | `metrics.json` is canonical. `reports/` is reserved (benchmark samples now, HTML later). No HTML report in M1.                                                                                                                   |
| 10 | **Hydra/Lightning/MLflow write outside `output/`** by default (`outputs/`, `lightning_logs/`, `mlruns/`), violating "all artifacts under `output/`".                                                         | Explicit redirects in config and trainer (§4, WP1, WP4, WP8).                                                                                                                                                                    |
| 11 | **Hydra layout:** `--config-name baseline` would fail (baseline lives in the `experiment/` group); `run=<name>` collides conceptually with Hydra's run dir.                                                  | Exact `default.yaml` with defaults list and `# @package` headers (§4). Experiment selected via `experiment=baseline`.                                                                                                            |
| 12 | **Pretrained ResNet downloads weights** → tests need network.                                                                                                                                                | `pretrained: false` for skeleton/tests. No network in tests.                                                                                                                                                                     |
| 13 | **Synthetic data unspecified** → bucket/stationary metrics would be NaN or trivial.                                                                                                                          | Synthetic generator spec guaranteeing stop → accelerate through all buckets → stop in every sequence. §6.4.                                                                                                                      |
| 14 | **Edge cases were a list of names, not decisions.**                                                                                                                                                          | Policy table with a decision and a test for each. §6.7.                                                                                                                                                                          |
| 15 | **Runtime metric ambiguity:** per-frame `inference_ms` in predictions vs a benchmark; GPU timing needs sync.                                                                                                 | Benchmark is separate (`benchmark.py`), batch=1 streaming, `model_only` and `end_to_end` reported separately; `inference_ms` is NaN in batch evaluation. §6.6.                                                                   |
| 16 | **Metric edge cases:** R² undefined on tiny/constant sets; empty buckets; zero predicted positives.                                                                                                          | Metrics return `None` (JSON null) plus counts; empty eval set raises. §6.5.                                                                                                                                                      |
| 17 | **"Previous speed" baseline is not deployable** (needs ground-truth speed at inference).                                                                                                                     | Relabelled as an oracle/persistence *reference*; deployable baselines listed in §11.                                                                                                                                             |
| 18 | **Dependency bloat / fragility:** scipy, tqdm unused; opencv only needed for real video; hand-pinned versions from memory are error-prone.                                                                   | Trimmed; opencv is an optional extra; lock file is generated *after* tests pass.                                                                                                                                                 |
| 19 | **Initial deliverable too large for one agent.**                                                                                                                                                             | 12 small WPs with a parallelization map and gates.                                                                                                                                                                               |
| 20 | **Lightning import ambiguity** (`pl.` snippet).                                                                                                                                                              | Standardize on `import lightning as L`.                                                                                                                                                                                          |
| 21 | **Frame-level ResNet features + LSTM may lose fine motion cues** (speed is a motion quantity).                                                                                                               | Not a skeleton concern; recorded as Phase-4 hypotheses (§11).                                                                                                                                                                    |
| 22 | **BatchNorm with batch size 1** errors in train mode; mentioned only as an edge case.                                                                                                                        | Skeleton model avoids BatchNorm; train loader uses `drop_last=True`.                                                                                                                                                             |

Kept from v1: the overall architecture, flat `src/` modules, Hydra + Lightning + MLflow-behind-an-abstraction, sequence-level splitting, hysteresis-based stationary detector, bucket/regression/stationary/runtime metrics, run-per-execution directories, "skeleton first, real model later".

---

## 2. Scope and milestones

- **M1 (this plan's focus):** a complete, tested, CPU-runnable vertical slice on **synthetic data** with a tiny model. `python scripts/train.py` produces the full artifact set (§7.6).
- **M2+ (§11):** real dataset, baselines, CNN+LSTM, stationary-algorithm comparison, runtime benchmark on target hardware, experiments. **Blocked** on the open decisions in §2.1. Do not start until M1 is accepted.

**M1 non-goals:** optimizing accuracy, real video decoding, HTML reports, multi-GPU, hyperparameter search, complex architectures.

### 2.1 Open decisions (assumed defaults let M1 proceed; confirm before M2)

| ID | Question                                                                                                               | Assumed default for M1              | Blocks                     |
| -- | ---------------------------------------------------------------------------------------------------------------------- | ----------------------------------- | -------------------------- |
| D1 | Dataset format (video files + label CSV? pre-extracted frames?), label source (CAN/GPS/OBD), units (mph / m/s / km/h)? | Synthetic only; labels in mph       | M2                         |
| D2 | Native frame rate and whether it is constant?                                                                          | 10 Hz, constant nominal             | M2                         |
| D3 | Is the camera/vehicle fixed across data (monocular scale ambiguity, intrinsics)?                                       | Ignored                             | M2 model choice            |
| D4 | Must inference be strictly causal (no future frames)?                                                                  | **Yes** (real-time requirement)     | design already assumes yes |
| D5 | Target runtime (FPS, device: CPU/GPU/embedded)?                                                                        | Report only; no pass/fail threshold | M2 benchmark               |
| D6 | Is "creeping ≤ 2 mph" ground truth derived from the label (`<= 2.0`)?                                                  | Yes                                 | M2                         |
| D7 | Initial detector state at sequence start (MOVING vs STOPPED)?                                                          | MOVING                              | M2 tuning                  |
| D8 | "Structured logging" = stdlib logging with a fixed format and `key=value` event messages (not JSON lines)?             | Yes; JSON lines can be added later  | none                       |

Record answers in `docs/DECISIONS.md` (template in §9.5). Agents must **never invent answers**; use the default and note the assumption.

---

## 3. Global conventions (apply to every WP)

**Non-negotiable rules**

1. Application code lives only in `src/`. `scripts/` are thin (parse config → call `src`).
2. Import style: package is imported as `src` (e.g. `from src.schemas import SpeedPrediction`). Project is installed editable: `python -m pip install -e ".[dev]"`.
3. Python 3.11+, `from __future__ import annotations` at top of every module, type hints on public functions.
4. The canonical environment is Conda, named `speed-estimation`; all commands assume `conda activate speed-estimation`. Do not create/use a project venv.
5. Lightning: `import lightning as L` (package `lightning`, **not** `pytorch_lightning`).
6. No `print()` anywhere (enforced by ruff rule `T20`). Use `logging.getLogger(__name__)`.
7. No hard-coded paths or hyperparameters in code; read from the typed config. Defaults live in `configs/`.
8. Never overwrite a previous run. Run dirs are created with `exist_ok=False`.
9. Tests must run offline on CPU and must not download anything (`pretrained: false`).
10. No silent data alteration: clipping, smoothing, dropping or splitting data must be explicit, configured and logged.
11. Never leak test information into training, preprocessing statistics, early stopping, checkpoint selection or threshold tuning (tune on **val** only).
12. Fail loudly: raise the project exceptions in `schemas.py`; never `except: pass`; never return fake values to "make tests pass".
13. Do not add dependencies beyond `pyproject.toml` without orchestrator approval.
14. Do not edit files you do not own (see WP "Owns"). If a contract must change, write the request to `docs/CONTRACT_REQUESTS.md` and stop.
15. Separation: model ≠ Lightning wrapper; post-processing ≠ model; stationary detection ≠ regression; evaluation ≠ training; visualization ≠ evaluation logic.
16. Interfaces only where ≥2 implementations are expected (model, tracker, smoothing, frame source). No other abstractions.
17. Document assumptions/unresolved dataset decisions instead of inventing them.
18. `final_speed_mph` is the clipped/smoothed prediction and is never zeroed by the stationary detector; `stationary_pred` is an independent boolean output (normalize `-0.0` to `0.0`, never emit `-0.0`).

**Units & types:** speeds in **mph** (`float`); timestamps in **seconds** (`float`); images `uint8` HWC at I/O boundaries, `float32` CHW inside tensors.

**Logging format:** `%(asctime)s | %(levelname)-7s | %(name)s | %(message)s`. Event messages use `key=value` pairs, e.g. `event=dataset_sizes train=812 val=140 test=140`. Handlers attach to the **root** logger so Lightning/MLflow logs also reach the log file.

**Ruff:** line length 100, rules `E,F,I,B,UP,T20`.

---

## 4. Repository layout and canonical configuration

```text
speed-estimation/
├── README.md
├── pyproject.toml
├── environment.yml             # canonical Conda environment specification
├── requirements.txt            # generated from the validated Conda environment AFTER M1 tests pass
├── .gitignore
├── docs/
│   ├── PLAN.md                 # this file
│   ├── PROGRESS.md             # orchestrator-owned tracker
│   ├── DECISIONS.md            # open decisions and answers
│   └── CONTRACT_REQUESTS.md    # agents append requests here
├── configs/
│   ├── default.yaml
│   ├── experiment/{baseline.yaml, smoke.yaml}
│   ├── model/{tiny_cnn_gru.yaml, cnn_lstm.yaml}   # cnn_lstm added in M2
│   └── dataset/{synthetic.yaml}                   # dashcam.yaml added in M2
├── data/{raw,interim,processed,splits}/           # each with .gitkeep; contents git-ignored
├── output/                                        # git-ignored; ALL generated artifacts
│   ├── runs/<run_name>/…   logs/<run_name>.log   figures/<run_name>/…
│   ├── hydra/              mlflow/
├── scripts/{train.py, evaluate.py, predict.py, visualize.py}
├── src/
│   ├── __init__.py
│   ├── schemas.py          # dataclasses + exceptions (contracts)
│   ├── config.py           # typed config dataclasses + compose/validate
│   ├── logging_utils.py
│   ├── run_manager.py      # run dirs, metadata, seeding
│   ├── tracking.py
│   ├── buckets.py
│   ├── data.py             # sequences, validation, splits, synthetic data, Dataset
│   ├── preprocessing.py    # frame transforms + window indexing
│   ├── models.py
│   ├── training.py         # LightningModule + Trainer builder
│   ├── postprocessing.py
│   ├── pipeline.py         # raw prediction collection + StreamingSpeedEstimator
│   ├── evaluation.py
│   ├── benchmark.py
│   ├── reporting.py
│   ├── visualization.py
│   └── experiment.py       # run_experiment(cfg): the train flow
└── tests/
    ├── conftest.py
    ├── unit/…              # see §10
    └── integration/…

```

### 4.1 Conda environment and dependency policy

Conda is the **canonical environment manager** for this project. Create the environment from the committed `environment.yml`; do not instruct agents or users to create a Python `venv`.

The repository should contain an `environment.yml` with:

- environment name: `speed-estimation`
- Python 3.11
- core scientific/ML dependencies available through Conda where practical
- PyTorch and its compatible runtime dependencies
- development tools required by the project

Use:

```bash
conda env create -f environment.yml
conda activate speed-estimation
python -m pip install -e ".[dev]"
```

`pyproject.toml` remains the Python project/package dependency declaration, while `environment.yml` defines the reproducible Conda development environment. Do not duplicate exact package pins unnecessarily across both files.

For CI or another machine, the environment must be created through Conda first. All verification commands in this plan assume the `speed-estimation` Conda environment is active.

For a platform-specific exact environment snapshot after M1 passes, optionally create:

```bash
conda list --explicit > conda-spec.txt
```

`requirements.txt` is retained for pip interoperability, but it is **not** the canonical environment definition.

### 4.2 `pyproject.toml` essentials

- `requires-python = ">=3.11"`.
- Runtime deps (lower bounds only): `torch>=2.2`, `torchvision>=0.17`, `lightning>=2.2`, `torchmetrics>=1.3`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `hydra-core>=1.3`, `omegaconf>=2.3`, `mlflow>=2.10`, `pyyaml`.
- Extras: `dev = [pytest, pytest-cov, ruff, mypy]`, `video = [opencv-python]` (M2 only).
- `[tool.setuptools.packages.find] include = ["src*"]`.
- `[tool.pytest.ini_options]`: `testpaths=["tests"]`, markers `slow`, `integration`, `gpu`, `mlflow`.
- `[tool.ruff]` as in §3.
- Dropped vs v1: `scipy`, `tqdm` (not needed; smoothing is implemented in NumPy; Lightning has its own progress bar).

### 4.3 Canonical `configs/default.yaml`

```yaml
defaults:
  - _self_
  - dataset: synthetic
  - model: tiny_cnn_gru
  - experiment: baseline
  - override hydra/job_logging: disabled   # we configure logging ourselves
  - override hydra/hydra_logging: disabled

hydra:
  run:
    dir: output/hydra/${now:%Y%m%d_%H%M%S}   # keep Hydra's own files under output/
  job:
    chdir: false                              # never change the working directory
  output_subdir: null

paths:
  output_root: output
  data_root: data

experiment:
  name: baseline
  seed: 42
  deterministic: false      # true => torch.use_deterministic_algorithms(True, warn_only=True)

# `data` is provided by the dataset group (see dataset/synthetic.yaml)

preprocessing:
  image_size: [32, 32]      # [H, W]
  normalize: true
  mean: [0.485, 0.456, 0.406]   # fixed constants, NOT fitted on data
  std:  [0.229, 0.224, 0.225]

# `model` is provided by the model group

training:
  batch_size: 32
  epochs: 5
  learning_rate: 0.0001
  weight_decay: 0.00001
  optimizer: adamw
  loss: mse                 # mse | l1   (huber etc. are future work)
  monitor: val/mae_mph
  early_stopping_patience: 5
  gradient_clip_val: null
  accelerator: auto
  devices: 1
  precision: "32-true"
  num_workers: 0

postprocessing:
  clip:      {enabled: true, min_mph: 0.0, max_mph: null}
  smoothing: {method: none, ema_alpha: 0.5, median_window: 5}   # none | ema | median (causal)
  stationary:
    enabled: true
    enter_threshold_mph: 2.0      # speed <= enter for N consecutive frames -> STOPPED
    exit_threshold_mph: 2.5       # speed >  exit  for M consecutive frames -> MOVING
    enter_persistence_frames: 3
    exit_persistence_frames: 3
    initial_state: moving

evaluation:
  stationary_label_threshold_mph: 2.0   # ground truth only: true_stationary = true_speed <= this
  speed_buckets:                        # lower bound inclusive, upper exclusive
    - {name: low,    min_mph: 0,  max_mph: 15}
    - {name: medium, min_mph: 15, max_mph: 45}
    - {name: high,   min_mph: 45, max_mph: null}
  splits_to_evaluate: [val, test]

benchmark:
  enabled: true
  device: auto
  num_warmup: 10
  num_iters: 100

tracking:
  enabled: true
  backend: mlflow
  mlflow:
    tracking_uri: null        # null => local store under output/mlflow (see WP8)
    experiment_name: speed-estimation

logging:
  level: INFO

```

`configs/dataset/synthetic.yaml` (note the `@package` header — group name differs from the target key):

```yaml
# @package data
source: synthetic
root: data/processed
split_manifest: data/splits/synthetic_seed${experiment.seed}.json
split_ratios: {train: 0.7, val: 0.15, test: 0.15}
sequence_length: 10        # T frames per window
window_stride: 1           # stride between window end indices (eval requires 1)
frame_step: 1              # take every k-th frame inside a window
frame_rate_hz: 10
max_gap_factor: 1.5        # dt > factor/frame_rate_hz splits the sequence
on_duplicate_timestamp: raise   # raise | drop_later
allow_negative_speed: false
max_speed_mph: 200
synthetic: {num_sequences: 12, frames_per_sequence: 120, image_size: [32, 32], noise_std: 0.02}

```

`configs/model/tiny_cnn_gru.yaml` (default package = `model`, no header needed):

```yaml
name: tiny_cnn_gru
encoder: {name: tiny_cnn, pretrained: false, feature_dim: 64}
temporal: {name: gru, hidden_size: 32, num_layers: 1}
checkpoint: null

```

`configs/experiment/baseline.yaml`:

```yaml
# @package _global_
experiment:
  name: baseline

```

`configs/experiment/smoke.yaml` (fast CI run):

```yaml
# @package _global_
experiment:
  name: smoke
data:
  synthetic: {num_sequences: 6, frames_per_sequence: 60}
training: {epochs: 2, batch_size: 16}
benchmark: {num_warmup: 2, num_iters: 10}
tracking: {enabled: false}

```

Usage: `python scripts/train.py` (baseline), `python scripts/train.py experiment=smoke`, `python scripts/train.py training.learning_rate=0.001`.

---

## 5. Contracts (frozen by WP0 — do not change without orchestrator approval)

### 5.1 `src/schemas.py`

```python
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, Sequence
import numpy as np

class PipelineError(Exception): ...
class ConfigError(PipelineError): ...
class DataValidationError(PipelineError): ...
class EmptyEvaluationError(PipelineError): ...

class FrameSource(Protocol):
    """Lazy-friendly frame access. M1: InMemoryFrames. M2: video-backed."""
    def __len__(self) -> int: ...
    def get(self, indices: Sequence[int]) -> np.ndarray: ...   # uint8 [k, H, W, C]

@dataclass(frozen=True)
class SpeedSequence:
    sequence_id: str          # unique; gap-split children are "<orig>#<k>"
    group_id: str             # source drive/video; SPLITS ARE MADE ON THIS
    frames: FrameSource
    speeds_mph: np.ndarray    # float64 [N]; NaN = missing label
    timestamps: np.ndarray    # float64 [N] seconds, strictly increasing after validation

@dataclass(frozen=True)
class SpeedPrediction:        # streaming-inference output (one per new frame once buffer is full)
    sequence_id: str
    frame_index: int          # index of the window's LAST frame within the sequence
    timestamp: float
    raw_speed_mph: float      # model output, before any post-processing
    final_speed_mph: float    # clipped + smoothed prediction; independent of `stationary`
    stationary: bool
    inference_ms: float       # NaN if not measured

@dataclass(frozen=True)
class PostProcessResult:
    final_speed_mph: float
    stationary: bool

@dataclass
class EvaluationResult:
    split: str
    n_samples: int
    overall: dict             # final speed vs truth: mae, rmse, r2, median_ae   (None if undefined)
    overall_raw: dict         # same metrics on raw_speed_mph
    buckets: dict             # name -> {n, mae, rmse, r2, median_ae}
    stationary: dict          # precision, recall, f1, tp, fp, fn, tn, n_true_stationary
    latency: dict | None = None   # filled by benchmark, not by evaluate_predictions
    def to_dict(self) -> dict: ...

@dataclass(frozen=True)
class RunPaths:
    run_name: str
    run_dir: Path
    checkpoints_dir: Path
    reports_dir: Path
    figures_dir: Path
    log_file: Path
    config_path: Path
    metadata_path: Path
    metrics_path: Path
    predictions_path: Path
    split_path: Path

```

### 5.2 Config dataclasses (`src/config.py`, schema part owned by WP0)

Typed dataclasses mirroring §4.2 **1:1** (`Config` → `ExperimentCfg`, `PathsCfg`, `DataCfg`, `PreprocessingCfg`, `ModelCfg`, `TrainingCfg`, `PostprocessingCfg` (`ClipCfg`, `SmoothingCfg`, `StationaryCfg`), `EvaluationCfg`, `BucketCfg`, `BenchmarkCfg`, `TrackingCfg`, `LoggingCfg`). `ModelCfg` has `name: str`, `encoder: dict[str, Any]`, `temporal: dict[str, Any]`, `checkpoint: str | None`. Typos in YAML must raise `ConfigError` (use `OmegaConf.structured` + merge).

Downstream code receives these typed objects (e.g. `cfg.postprocessing.stationary.enter_threshold_mph`), never raw dicts.

### 5.3 Tensor and batch contracts

| Item                         | Shape / type                                                                                                                       |
| ---------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| Window frames (model input)  | `float32 [B, T, C, H, W]`                                                                                                          |
| Model output                 | `float32 [B]` (mph, **unclipped**)                                                                                                 |
| Batch dict from `DataLoader` | `{"frames": [B,T,C,H,W], "target_mph": [B] float32, "timestamp": [B] float64, "frame_index": [B] int64, "sequence_id": list[str]}` |
| `target_mph`                 | speed at the window's last frame; NaN-target windows are never produced                                                            |

### 5.4 Canonical predictions table (DataFrame ⇄ `predictions.csv`)

Columns, in this order:

`split, sequence_id, frame_index, timestamp, true_speed_mph, raw_speed_mph, final_speed_mph, stationary_pred, inference_ms`

- Sorted by `(split, sequence_id, frame_index)`; within a sequence `frame_index` strictly increasing.
- `inference_ms` is NaN in batch evaluation.
- `final_speed_mph` and `stationary_pred` are absent from the *raw* table returned by `collect_raw_predictions` and added by `postprocess_dataframe`.
- Evaluation, reporting and visualization consume **only** this table (plus config for bucket definitions).

### 5.5 Public API (signatures are the contract)

| Module              | Public API                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| ------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `config.py`         | `compose_config(overrides: Sequence[str] = (), config_dir: Path \| None = None) -> DictConfig` (Hydra compose API, for tests/scripts); `validate_config(cfg: DictConfig) -> Config`                                                                                                                                                                                                                                                                                                                                                                                                      |
| `logging_utils.py`  | `setup_logging(log_file: Path \| None, level: str = "INFO") -> None` (idempotent: removes handlers it previously added); `get_logger(name: str) -> logging.Logger`                                                                                                                                                                                                                                                                                                                                                                                                                       |
| `run_manager.py`    | `create_run(cfg: Config, now: datetime \| None = None) -> RunPaths`; `set_seed(seed: int, deterministic: bool = False) -> None`; `collect_environment() -> dict`; `write_metadata(paths, meta: dict) -> None`; `update_metadata(paths, **fields) -> None`; `load_run(run_name: str, output_root: Path) -> tuple[RunPaths, Config]`                                                                                                                                                                                                                                                       |
| `tracking.py`       | `class ExperimentTracker(ABC)`: `start_run(run_name, tags)`, `log_params(dict)`, `log_metrics(dict, step=None)`, `log_artifact(path)`, `finish(status="FINISHED")`; `MLflowTracker`, `NoOpTracker`; `create_tracker(cfg: Config, paths: RunPaths) -> ExperimentTracker`                                                                                                                                                                                                                                                                                                                  |
| `buckets.py`        | `@dataclass BucketSpec(name, min_mph, max_mph \| None)`; `buckets_from_cfg(cfg: EvaluationCfg) -> list[BucketSpec]`; `assign_buckets(speeds: np.ndarray, buckets) -> np.ndarray` (object array of names; values outside all buckets → `ConfigError`/`ValueError`)                                                                                                                                                                                                                                                                                                                        |
| `data.py`           | `InMemoryFrames`; `generate_synthetic_sequences(cfg: DataCfg, seed: int) -> list[SpeedSequence]`; `validate_sequence(seq, cfg) -> list[SpeedSequence]` (returns gap-split children); `make_group_split(group_ids, ratios, seed) -> dict[str, list[str]]`; `load_or_create_split(sequences, cfg, seed, run_split_path) -> dict[str, list[str]]`; `assert_no_leakage(split) -> None`; `SpeedDataset(sequences, window_index, transform, sequence_length, frame_step)`; `build_datasets(cfg, seed) -> dict[str, SpeedDataset]`; `build_dataloaders(datasets, cfg) -> dict[str, DataLoader]` |
| `preprocessing.py`  | `Resize(size)`, `ToTensor()`, `Normalize(mean, std)`, `Compose([...])` — each takes `uint8 [H,W,C]`/tensor per frame; `build_transform(cfg: PreprocessingCfg)`; `build_window_index(seq_lengths, sequence_length, stride, frame_step) -> list[tuple[int,int]]` as `(seq_idx, end_idx)`; `window_frame_indices(end_idx, sequence_length, frame_step) -> list[int]`                                                                                                                                                                                                                        |
| `models.py`         | `class BaseSpeedModel(nn.Module)`: `forward(frames[B,T,C,H,W]) -> Tensor[B]`; `MODEL_REGISTRY: dict[str, type]`; `build_model(cfg: ModelCfg) -> BaseSpeedModel`                                                                                                                                                                                                                                                                                                                                                                                                                          |
| `training.py`       | `class SpeedLightningModule(L.LightningModule)`; `build_trainer(cfg: Config, paths: RunPaths) -> L.Trainer`; `load_model_from_checkpoint(path, cfg) -> BaseSpeedModel`                                                                                                                                                                                                                                                                                                                                                                                                                   |
| `postprocessing.py` | `StationaryDetector(cfg)`: `reset()`, `update(speed_mph) -> bool`, property `stationary`; `SpeedPostProcessor(cfg: PostprocessingCfg)`: `reset()`, `step(raw_mph) -> PostProcessResult`, `process_sequence(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]`; `postprocess_dataframe(df, cfg) -> DataFrame`                                                                                                                                                                                                                                                                             |
| `pipeline.py`       | `collect_raw_predictions(model, dataloader, device, split) -> DataFrame`; `class StreamingSpeedEstimator(model, transform, postprocessor, sequence_length, frame_step, max_gap_s, device)`: `reset()`, `update(frame_hwc_uint8, timestamp) -> SpeedPrediction \| None`                                                                                                                                                                                                                                                                                                                   |
| `evaluation.py`     | `regression_metrics(y_true, y_pred) -> dict`; `stationary_metrics(true_speed, pred_stationary, label_threshold) -> dict`; `latency_summary(samples_ms) -> dict`; `evaluate_predictions(df, cfg: EvaluationCfg, split: str) -> EvaluationResult`                                                                                                                                                                                                                                                                                                                                          |
| `benchmark.py`      | `benchmark_model(model, example_input, device, num_warmup, num_iters) -> tuple[np.ndarray, dict]`; `benchmark_end_to_end(estimator, frames, timestamps, num_warmup, num_iters) -> tuple[np.ndarray, dict]`                                                                                                                                                                                                                                                                                                                                                                               |
| `reporting.py`      | `write_metrics_json(paths, results: dict[str, EvaluationResult], latency: dict \| None, extra: dict) -> None`; `write_predictions_csv(paths, df) -> None`; `read_predictions_csv(path) -> DataFrame`                                                                                                                                                                                                                                                                                                                                                                                     |
| `visualization.py`  | `plot_predicted_vs_actual(df, out) `, `plot_speed_timeseries(df, out, sequence_id=None)`, `plot_error_distribution(df, out)`, `plot_error_by_bucket(result, out)`, `plot_stationary_confusion(result, out)`, `plot_latency(samples_ms, summary, out)`, `generate_all_figures(df, result, latency_samples, paths) -> list[Path]` (all use the `Agg` backend, never `plt.show()`)                                                                                                                                                                                                          |
| `experiment.py`     | `run_experiment(cfg: DictConfig) -> RunPaths`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |

### 5.6 Output contract (every completed run)

```text
output/runs/<run_name>/
├── config.yaml              # resolved config
├── metadata.json            # env + git + seed + status(running|completed|failed) + duration + best ckpt
├── metrics.json             # canonical metrics (schema below)
├── predictions.csv          # canonical predictions table (test split; val rows also included)
├── split.json               # copy of the split manifest actually used
├── checkpoints/best.ckpt    # (+ last.ckpt)
├── training_logs/           # Lightning CSVLogger output (curves)
└── reports/latency_samples.csv
output/logs/<run_name>.log
output/figures/<run_name>/{speed_timeseries,predicted_vs_actual,error_distribution,
                           error_by_bucket,stationary_confusion_matrix,latency}.png

```

Run name: `YYYYMMDD_HHMMSS_<experiment-name>_<4 hex>`; experiment name sanitized to `[A-Za-z0-9_-]`; regenerate the suffix on collision.

`metrics.json` schema (`schema_version: 1`):

```json
{
  "schema_version": 1,
  "run_name": "...",
  "splits": {
    "test": {
      "n_samples": 0,
      "overall":     {"mae": 0.0, "rmse": 0.0, "r2": 0.0, "median_ae": 0.0},
      "overall_raw": {"mae": 0.0, "rmse": 0.0, "r2": 0.0, "median_ae": 0.0},
      "buckets": {"low": {"n": 0, "mae": null, "rmse": null, "r2": null, "median_ae": null}},
      "stationary": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0, "fp": 0, "fn": 0, "tn": 0, "n_true_stationary": 0}
    }
  },
  "latency": {
    "model_only": {"mean_ms": 0.0, "median_ms": 0.0, "p95_ms": 0.0, "fps": 0.0, "n": 0, "device": "cpu"},
    "end_to_end": {"...": "same keys"}
  },
  "extra": {"best_checkpoint": "...", "best_val_metric": 0.0}
}

```

---

## 6. Behaviour specifications

### 6.1 Stationary detection and post-processing (causal)

Pipeline per frame: `raw → clip (if enabled) → smoothing (if any) → stationary detector → final`.

- `raw_speed_mph` is stored **before** clipping. Clipping is an explicit config operation (`postprocessing.clip`).
- Smoothing is **causal**: `ema` (`s_t = α·x_t + (1-α)·s_{t-1}`, `s_0 = x_0`) or `median` over the last `k` values (fewer at the start). No future values.
- Detector state machine, evaluated **after** including the current frame: 
  - `MOVING → STOPPED` when `speed <= enter_threshold_mph` for `enter_persistence_frames` consecutive frames.
  - `STOPPED → MOVING` when `speed > exit_threshold_mph` for `exit_persistence_frames` consecutive frames.
  - Counters reset whenever the condition breaks; both counters reset on a state change.
  - Boundary rule: `== enter_threshold` counts as low (`<=`); `== exit_threshold` does **not** count as high (strict `>`).
  - Constraint (validated): `enter_threshold_mph <= exit_threshold_mph`, persistence frames `>= 1`.
- `final_speed_mph` is the smoothed/clipped value; it is never zeroed by the detector. `stationary_pred` is an independent boolean. Normalize `-0.0` to `0.0`.
- If `stationary.enabled` is false: `stationary_pred = False` everywhere and a WARNING says stationary metrics are not meaningful.
- NaN/inf raw input → raise `ValueError` (fail loudly; do not guess).
- Known trade-off (document in README): persistence delays the `stationary_pred` flag by `N-1` frames when stopping, which lowers stationary recall at transitions. Phase 5 compares configurations on **val**.

Reference test vector (enter=2.0, N=3, exit=2.5, M=3, smoothing none, clip off):

| speeds                          | stationary flags                      |
| ------------------------------- | ------------------------------------- |
| `[5, 1.9, 1.9, 1.9, 1.9]`       | `[F, F, F, T, T]`                     |
| `[5, 1.9, 1.9, 2.0, 1.9]`       | `[F, F, F, T, T]` (2.0 counts as low) |
| `[5, 1.9, 5, 1.9, 1.9, 1.9]`    | `[F, F, F, F, F, T]` (counter reset)  |
| stopped, then `[2.5, 2.5, 2.5]` | stays stopped (2.5 is not `> 2.5`)    |
| stopped, then `[2.6, 2.6, 2.6]` | `[T, T, F]`                           |

### 6.2 Windows and labels

- A window is `T = sequence_length` frames; with `frame_step = k` its frame indices are `end, end-k, …, end-(T-1)k` (ascending order inside the tensor).
- Valid `end` satisfies `end >= (T-1)·k`. **No padding.** Sequences too short yield zero windows (warn); a split with zero windows total raises `DataValidationError`.
- Label = `speeds_mph[end]`. Windows whose label is NaN are dropped (count logged).
- Training may use `window_stride > 1`. **Evaluation/test must use stride 1** (persistence is defined in frames); validate and raise otherwise.
- `frame_index` in outputs = `end`.

### 6.3 Splits and leakage

- `make_group_split(group_ids, ratios, seed)`: deterministic shuffle of **sorted unique** `group_id`s, then partition by ratios; every split must be non-empty else `DataValidationError` (with an explanatory message about too few groups).
- Manifest JSON: `{"seed": int, "ratios": {...}, "splits": {"train": [group_id,...], ...}}` saved to `data.split_manifest`; reused if present (reproducibility) and verified to cover **exactly** the available groups (unknown/missing groups → error; never assign silently). A copy is written to `RunPaths.split_path`.
- `assert_no_leakage(split)`: no group appears in two splits.
- Train/val/test datasets are built from sequences filtered by `group_id`; gap-split children inherit `group_id`.

### 6.4 Synthetic data generator (must make every metric non-trivial)

Every sequence (any length ≥ 40 frames) follows a speed profile defined by **fractions of its length**: stopped (0 mph, with small 0–1.5 mph jitter) ≈ 15% → ramp up ≈ 30% to a peak drawn from U(50, 70) → hold ≈ 15% → ramp down ≈ 25% → stopped ≈ 15%. Therefore every sequence, and so every split, contains stationary frames and all three buckets.

Frames: `uint8 [H,W,3]` vertical-stripe pattern whose horizontal offset advances each frame by `k · speed_mph` pixels (motion ∝ speed), plus Gaussian noise (`noise_std`). Fully deterministic given `seed`. Each sequence gets its own `group_id` (`synth_00`, …). Timestamps `i / frame_rate_hz`.

### 6.5 Evaluation

- Input: predictions table with `final_speed_mph` and `stationary_pred`. Sort check: raise if not ordered per sequence.
- Regression metrics on `final_speed_mph` (**primary**) and on `raw_speed_mph` (`overall_raw`): MAE, RMSE, R², Median AE. Implement MAE/RMSE/MedAE with NumPy; R² with `sklearn.metrics.r2_score`. Returns `None` for R² when `n < 2` or truth variance is 0.
- Buckets: assign by **ground-truth** speed using `buckets.assign_buckets` (`min <= v < max`; `max=None` → +inf). Empty bucket → all metrics `None`, `n = 0`.
- Stationary: `true_stationary = true_speed_mph <= label_threshold` (spec: `<= 2.0`); positive class = stationary; precision/recall/F1 with zero-division → `0.0`, plus raw `tp/fp/fn/tn` counts so a zero is interpretable.
- Empty table → `EmptyEvaluationError`.
- Latency summary keys: `mean_ms, median_ms, p95_ms, fps (=1000/mean_ms), n`.

### 6.6 Benchmark

- `model.eval()`, `torch.inference_mode()`, batch size 1, a **fixed** input tensor prepared before timing (no disk I/O or data loading inside the timed region).
- Warm-up iterations discarded. Use `time.perf_counter()`. On CUDA: `torch.cuda.synchronize()` before starting and after finishing each timed call.
- Two measurements: `model_only` (one forward on a prepared window) and `end_to_end` (streaming estimator `update()` on in-memory frames: transform + forward + post-processing).
- Samples saved to `reports/latency_samples.csv`; summaries into `metrics.json["latency"]`.
- Note for M2: recomputing CNN features for the full window each frame is wasteful; feature caching is a later optimization, not M1.

### 6.7 Edge-case policy table

| Case                                                          | Decision                                                                                                         | Where                    | Test                                    |
| ------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- | ------------------------ | --------------------------------------- |
| Missing frames / time gap `dt > max_gap_factor/frame_rate_hz` | Split sequence into children `<id>#k`; no interpolation                                                          | `validate_sequence`      | `test_data`                             |
| Duplicate timestamps                                          | `raise` (default) or `drop_later` per config; log count                                                          | `validate_sequence`      | `test_data`                             |
| Non-monotonic timestamps                                      | raise `DataValidationError`                                                                                      | same                     | `test_data`                             |
| Irregular frame rate                                          | Warn if median dt deviates >10% from nominal; **no resampling** (dataset-specific decision, D2)                  | same                     | `test_data`                             |
| Missing/NaN speed label                                       | Drop windows with NaN target; drop sequence if none valid; log counts                                            | dataset build            | `test_data`                             |
| Negative speed label                                          | raise unless `allow_negative_speed`                                                                              | validate                 | `test_data`                             |
| Speed label `> max_speed_mph`                                 | raise                                                                                                            | validate                 | `test_data`                             |
| Sequence shorter than window                                  | Skip with warning; error if split ends up empty                                                                  | window index             | `test_preprocessing`                    |
| Speeds exactly 0 / 2 / 15 / 45                                | stationary: 0 and 2 → true; buckets: 0,2 → low; 15 → medium; 45 → high                                           | evaluation               | `test_metrics`, `test_buckets`          |
| Prediction < 0                                                | Clipped only if `clip.enabled` (default true, min 0); raw preserved                                              | postprocessing           | `test_postprocessing`                   |
| Extremely high prediction                                     | Not altered unless `clip.max_mph` set; reported as-is                                                            | postprocessing           | `test_postprocessing`                   |
| NaN/inf prediction                                            | raise `ValueError`                                                                                               | postprocessing           | `test_postprocessing`                   |
| Stationary transitions                                        | Per §6.1 vectors                                                                                                 | detector                 | `test_postprocessing`                   |
| First frames after startup                                    | Streaming returns `None` until buffer holds a full window; detector starts in `initial_state`                    | pipeline                 | `test_pipeline`                         |
| End of sequence                                               | No padding or lookahead; detector `reset()` between sequences                                                    | postprocessing/pipeline  | `test_postprocessing`                   |
| Stream timestamp gap                                          | Streaming estimator clears buffer and resets post-processor (mirrors offline gap split)                          | pipeline                 | `test_pipeline`                         |
| Batch size 1                                                  | Model and Lightning module must work with `B=1`; skeleton model uses no BatchNorm; train loader `drop_last=True` | models                   | `test_models`                           |
| CPU / CUDA                                                    | `auto` device; CUDA tests marked `gpu` and skipped when unavailable                                              | all                      | `test_models`                           |
| Empty evaluation set                                          | `EmptyEvaluationError`                                                                                           | evaluation               | `test_metrics`                          |
| Empty bucket                                                  | metrics `None`, `n=0`                                                                                            | evaluation               | `test_metrics`                          |
| `-0.0` output                                                 | normalized to `0.0`; CSV never contains `-0.0`                                                                   | postprocessing/reporting | `test_postprocessing`, `test_reporting` |

---

## 7. Orchestration model

### 7.1 Dependency map and parallel groups

```text
WP0 (contracts, scaffold)   ◄── orchestrator or strongest agent; everything depends on it
 │
 ├── Group A (parallel, independent):
 │     WP1 config+logging+run manager
 │     WP2 data core (sequences, validation, splits, synthetic)
 │     WP5 postprocessing
 │     WP6 evaluation + buckets tests
 │     WP8 tracking
 │     WP4 models + LightningModule (tests use random tensors)
 │
 ├── Group B (after the listed deps):
 │     WP3 preprocessing + Dataset        ← WP2
 │     WP7 reporting + visualization      ← WP6
 │     WP9 pipeline + benchmark           ← WP3, WP4, WP5
 │
 ├── Group C:
 │     WP10 experiment + scripts/train.py + integration tests  ← WP1,2,3,4,5,6,7,8,9
 │     WP11 evaluate/predict/visualize scripts                  ← WP10
 │
 └── WP12 M1 acceptance review (orchestrator)

```

### 7.2 Work package index

| WP   | Title                                                 | Depends       | Owner level  |
| ---- | ----------------------------------------------------- | ------------- | ------------ |
| WP0  | Scaffold, schemas, config schema, buckets, conftest   | —             | strong       |
| WP1  | Config compose/validate, logging, run manager, seed   | WP0           | any          |
| WP2  | Data core                                             | WP0           | any          |
| WP3  | Preprocessing + SpeedDataset + dataloaders            | WP2           | any          |
| WP4  | Models + LightningModule + trainer builder            | WP0           | medium       |
| WP5  | Post-processing + stationary detector                 | WP0           | any          |
| WP6  | Evaluation metrics                                    | WP0           | any          |
| WP7  | Reporting + visualization                             | WP6           | any          |
| WP8  | Tracking (MLflow behind abstraction)                  | WP0           | medium       |
| WP9  | Streaming pipeline + benchmark                        | WP3, WP4, WP5 | medium       |
| WP10 | Experiment orchestration + train script + integration | all above     | strong       |
| WP11 | Secondary scripts                                     | WP10          | any          |
| WP12 | Acceptance                                            | WP11          | orchestrator |

---

## 8. Work packages

Every WP ends with: run its **Verify** commands and submit the handoff report (§9.3). "Owns" = the only files you may create/edit.

### WP0 — Scaffold, contracts, config schema

**Owns:** `environment.yml`, `pyproject.toml`, `.gitignore`, `README.md`, all `.gitkeep`, `docs/*`, `src/__init__.py`, `src/schemas.py`, `src/config.py` (**dataclasses only**), `src/buckets.py`, `tests/conftest.py`, `tests/unit/test_buckets.py`, `tests/unit/test_schemas.py`, `configs/**` (all YAML from §4.2).
**Do:**

1. Create the tree (§4) with empty placeholder modules (a module docstring only; no fake functions).
2. Write `pyproject.toml` per §4.2; `.gitignore` ignoring `output/`, `data/**` (keep `.gitkeep`), `__pycache__`, `.venv`, `.conda/`, `mlruns/`, `outputs/`, `lightning_logs/`.
3. Implement `schemas.py` exactly as §5.1 (including `EvaluationResult.to_dict`).
4. Implement config dataclasses (§5.2) and `buckets.py` (§5.5).
5. Write `conftest.py` fixtures: `tmp_output_root`, `tiny_config` (a `Config` using synthetic data with 6 sequences × 60 frames, 2 epochs, tracking off, outputs under `tmp_path`), `rng`.
6. Create all YAML in `configs/` exactly as §4.2. **Tests:** `test_buckets` (boundaries 0/14.99/15/44.99/45/200; `max=None`; value below first bucket raises; names unique), `test_schemas` (dataclass construction, `to_dict` is JSON-serializable). **Verify:** `python -m pip install -e ".[dev]"` · `python -c "import src.schemas, src.config, src.buckets"` · `ruff check src tests` · `pytest tests/unit/test_buckets.py tests/unit/test_schemas.py -q` **Must not:** implement any other module's logic.

### WP1 — Config, logging, run manager

**Owns:** `src/config.py` (compose/validate functions only; do not alter dataclasses), `src/logging_utils.py`, `src/run_manager.py`, `tests/unit/test_config.py`, `tests/unit/test_logging.py`, `tests/unit/test_run_manager.py`.
**Do:**

- `compose_config` via `hydra.initialize_config_dir` + `hydra.compose` (config dir defaults to repo `configs/`); `validate_config` merges into the structured schema and raises `ConfigError` on unknown keys, wrong types, `enter > exit` thresholds, persistence `< 1`, `window_stride != 1` is allowed in config but flagged for eval elsewhere.
- `setup_logging` per §3 (console + file on the **root** logger; idempotent; creates parent dir).
- `create_run`: name format §5.6, creates `runs/<name>/{checkpoints,reports}`, `logs/`, `figures/<name>/`, writes `config.yaml` (resolved), returns `RunPaths`; `exist_ok=False`; retry new suffix on collision.
- `set_seed`: `lightning.seed_everything(seed, workers=True)`; if `deterministic` call `torch.use_deterministic_algorithms(True, warn_only=True)`; log that determinism is not guaranteed across hardware/versions.
- `collect_environment`: python/torch/lightning versions, CUDA version, GPU name (None if absent), platform, cpu count, git commit + dirty flag (via `subprocess`, `None` if unavailable), argv.
- `write_metadata` / `update_metadata` (read-modify-write JSON). Metadata fields in §5.6. **Tests:** compose default and `experiment=smoke`; override `training.learning_rate`; invalid key → `ConfigError`; run name regex, uniqueness for 50 rapid calls, dirs exist, `config.yaml` round-trips; seeding gives identical `torch.rand` / `np.random` after re-seed; metadata keys present; logging writes to file, calling twice does not duplicate lines; git unavailable path (monkeypatch) returns `None`. **Verify:** `pytest tests/unit/test_config.py tests/unit/test_logging.py tests/unit/test_run_manager.py -q` · `ruff check src tests` · `python -c "from src.config import compose_config; print_ = compose_config(); assert print_.experiment.seed == 42"` (use `assert`, not `print`). **Must not:** change `schemas.py`; write outside `tmp_path` in tests.

### WP2 — Data core

**Owns:** `src/data.py` (functions listed here only), `tests/unit/test_data.py`.
**Do:** `InMemoryFrames` (implements `FrameSource`), `generate_synthetic_sequences` (§6.4), `validate_sequence` (§6.7 rows: gaps → children `<id>#k`, duplicates, non-monotonic, negatives, max speed, irregular-rate warning), `make_group_split`, `load_or_create_split`, `assert_no_leakage` (§6.3).
**Tests:** determinism (same seed ⇒ identical arrays; different seed ⇒ different); every synthetic sequence has speeds in all three buckets and ≥1 frame `<= 2.0`; frames shape/dtype; gap split produces correct child lengths and ids; each policy row; split determinism, disjointness, non-empty, manifest reuse, mismatched-group manifest raises; leakage helper raises on overlap.
**Verify:** `pytest tests/unit/test_data.py -q` · `ruff check src tests`
**Must not:** implement `SpeedDataset` (WP3), touch preprocessing.

### WP3 — Preprocessing and Dataset

**Owns:** `src/preprocessing.py`, **append** to `src/data.py` (`SpeedDataset`, `build_datasets`, `build_dataloaders` only — do not modify WP2 functions), `tests/unit/test_preprocessing.py`, additions to `tests/unit/test_data.py` in a new class `TestDataset`.
**Do:** transforms (`Resize` via `torch.nn.functional.interpolate`; **no OpenCV**), `ToTensor` (uint8 HWC→float32 CHW in [0,1]), `Normalize` (constants from config), `Compose`, `build_transform`; `build_window_index`, `window_frame_indices` (§6.2); `SpeedDataset.__getitem__` returns the dict in §5.3; `build_datasets` validates sequences, applies split (§6.3), builds window indexes, logs `event=dataset_sizes …`; dataloaders: train shuffle + `drop_last=True`, val/test no shuffle, `num_workers` from config; enforce `window_stride == 1` for val/test.
**Tests:** window count formula; first/last window indices; `frame_step=2`; short sequence → zero windows; NaN-target windows dropped; label equals speed at `end`; batch dict keys/shapes/dtypes; transform output range/shape; **no train/val/test group overlap in built datasets**; eval stride ≠ 1 raises.
**Verify:** `pytest tests/unit/test_preprocessing.py tests/unit/test_data.py -q`
**Must not:** fit any statistic on data.

### WP4 — Models and Lightning module

**Owns:** `src/models.py`, `src/training.py`, `tests/unit/test_models.py`, `tests/unit/test_training.py`.
**Do:**

- `BaseSpeedModel`, registry, `build_model` (unknown name → `ConfigError` listing valid names).
- `tiny_cnn_gru`: per-frame conv encoder (2–3 conv layers, **GroupNorm or no norm, no BatchNorm**, global pool → `feature_dim`), GRU over time, linear head → `[B]`. Supports `B=1`. Encoder/temporal parts are separate submodules selected by config so LSTM/other swaps need no pipeline change (only add a registry entry; do not implement them now).
- `SpeedLightningModule(model, training_cfg)`: loss `mse|l1`; logs `train/loss`, `val/loss`, `val/mae_mph` (`on_epoch=True`); optimizer `adamw`; `predict_step` returns raw predictions plus batch metadata (`frame_index`, `sequence_id`, `timestamp`, `target_mph`).
- `build_trainer(cfg, paths)`: `default_root_dir=paths.run_dir`, `CSVLogger(save_dir=paths.run_dir, name="training_logs")`, `ModelCheckpoint(dirpath=paths.checkpoints_dir, monitor=cfg.training.monitor, mode="min", save_top_k=1, save_last=True, filename="best")`, `EarlyStopping`, `enable_progress_bar=False` when not a TTY, `deterministic` per config. Nothing may write to repo-root `lightning_logs/`.
- `load_model_from_checkpoint(path, cfg)`. **Tests:** output shape `[B]` for B=1 and B=4, finite; unknown model name; **overfit-one-batch test** (≤ 60 optimizer steps, loss decreases ≥ 30%); LightningModule runs `training_step`/`validation_step` on random tensors; trainer builder paths point inside `tmp_path`; checkpoint save/load round trip gives identical outputs; param count > 0. **Verify:** `pytest tests/unit/test_models.py tests/unit/test_training.py -q` **Must not:** use pretrained weights or network; import from `data.py`.

### WP5 — Post-processing

**Owns:** `src/postprocessing.py`, `tests/unit/test_postprocessing.py`.
**Do:** exactly §6.1. `process_sequence` **must be implemented as `reset()` + loop of `step()`** (same code path as streaming). `postprocess_dataframe(df, cfg)` groups by `sequence_id`, orders by `frame_index`, resets between sequences, adds `final_speed_mph`, `stationary_pred`; asserts stride-1 consecutive frames within each sequence (else raises `ValueError`).
**Tests:** all §6.1 vectors; boundary values (2.0, 2.5); clip on/off with negative and huge values; EMA and median numerical examples (hand-computed); NaN raises; `stationary_pred` independent of `final_speed_mph`; `-0.0` normalized to `0.0`; state reset between sequences (second sequence starts in `initial_state`); `process_sequence` equals manual `step` loop; disabled detector ⇒ all False; parameter validation (`enter > exit` raises).
**Verify:** `pytest tests/unit/test_postprocessing.py -q`
**Must not:** import torch.

### WP6 — Evaluation

**Owns:** `src/evaluation.py`, `tests/unit/test_metrics.py`.
**Do:** §6.5. Pure NumPy/pandas/sklearn; no torch.
**Tests:** hand-computed MAE/RMSE/MedAE/R² on a 5-point example; perfect prediction; constant truth ⇒ `r2 is None`; `n=1` ⇒ `r2 is None`; bucket edge assignment (14.99/15/44.99/45); empty bucket ⇒ `None` and `n=0`; stationary confusion counts on a hand example incl. truth exactly 2.0 ⇒ stationary; zero predicted positives ⇒ precision 0.0 with counts; empty frame ⇒ `EmptyEvaluationError`; unsorted table ⇒ error; `latency_summary` on `[1,2,3,4,100]` (mean, median, p95, fps); `to_dict` is JSON-serializable (no NaN: use `None`).
**Verify:** `pytest tests/unit/test_metrics.py -q`

### WP7 — Reporting and visualization

**Owns:** `src/reporting.py`, `src/visualization.py`, `tests/unit/test_reporting.py`, `tests/unit/test_visualization.py`.
**Do:** §5.5/§5.6. `write_metrics_json` emits `schema_version: 1` JSON (NaN → null; `allow_nan=False`). `write_predictions_csv` keeps the §5.4 column order and normalizes `-0.0` to `0.0`. Figures: `Agg` backend; one function per figure; filenames exactly as §5.6; figure closed after saving (`plt.close`); handles empty buckets and single-class stationary data without crashing.
**Tests:** JSON round-trip and schema keys; CSV round-trip equality (dtypes, column order); no `-0.0` in CSV text; each plot creates a non-empty PNG in `tmp_path`; `generate_all_figures` returns 6 paths; empty-bucket result does not raise.
**Verify:** `pytest tests/unit/test_reporting.py tests/unit/test_visualization.py -q`
**Must not:** compute metrics (use `EvaluationResult` values).

### WP8 — Tracking

**Owns:** `src/tracking.py`, `tests/unit/test_tracking.py`.
**Do:** `ExperimentTracker` ABC, `NoOpTracker`, `MLflowTracker`, `create_tracker`. **Every MLflow call is wrapped: failures log a WARNING and never raise** (MLflow stays optional). Tracking URI: `cfg.tracking.mlflow.tracking_uri` or, if null, a local store under `<output_root>/mlflow` — **check the installed MLflow version's documentation for the currently recommended local backend (file stores are being phased out in recent versions; a SQLite URI is the usual alternative) and record the choice in `docs/DECISIONS.md`**. Tag every MLflow run with `local_run_name`. `log_params` flattens nested dicts with dotted keys and truncates values to MLflow limits. `tracking.enabled=false` ⇒ `NoOpTracker`.
**Tests:** `NoOpTracker` accepts all calls; `create_tracker` returns the right class; param flattening; MLflow failure path (monkeypatch `mlflow.start_run` to raise ⇒ no exception propagates); one `@pytest.mark.mlflow` test with a `tmp_path` tracking URI that checks a run with the `local_run_name` tag exists (skipped if MLflow import fails).
**Verify:** `pytest tests/unit/test_tracking.py -q`
**Must not:** write MLflow data outside `output/` (tests: `tmp_path`).

### WP9 — Streaming pipeline and benchmark

**Owns:** `src/pipeline.py`, `src/benchmark.py`, `tests/unit/test_pipeline.py`, `tests/unit/test_benchmark.py`.
**Do:**

- `collect_raw_predictions`: eval mode, `torch.inference_mode()`, iterate a non-shuffled loader, build the raw table (§5.4 without final columns; `inference_ms=NaN`), sorted per §5.4.
- `StreamingSpeedEstimator.update`: appends transformed frame to a ring buffer; returns `None` until `(T-1)·frame_step + 1` frames are buffered; on timestamp gap `> max_gap_s` or non-increasing timestamp: clear buffer, `postprocessor.reset()`, log WARNING; otherwise predicts raw speed, runs `postprocessor.step`, returns `SpeedPrediction` with measured `inference_ms`.
- `benchmark_model`, `benchmark_end_to_end` per §6.6; return `(samples_ms, latency_summary(...))`. **Tests:** returns `None` for first `T-1` frames then a prediction each frame; gap resets buffer; **parity test:** on one synthetic sequence, streaming `raw_speed_mph` equals offline `collect_raw_predictions` within `1e-5`, and streaming `final_speed_mph`/`stationary` equal offline `postprocess_dataframe` output exactly; benchmark returns `num_iters` samples, positive values, p95 ≥ median; model is in eval mode and no gradients are tracked. **Verify:** `pytest tests/unit/test_pipeline.py tests/unit/test_benchmark.py -q`

### WP10 — Experiment orchestration, train script, integration

**Owns:** `src/experiment.py`, `scripts/train.py`, `tests/integration/test_pipeline.py`, `README.md` (usage sections only).
**Do:** `run_experiment(cfg)` in this order, logging each step as `event=…`:

1. validate config → `create_run` → `setup_logging` → `set_seed` → metadata (`status=running`) → tracker start + `log_params`.
2. build datasets/loaders (save split to `split_path`) → build model → log parameter count.
3. `build_trainer` → `fit` (train/val only; **test untouched**).
4. load best checkpoint → `collect_raw_predictions` for each split in `evaluation.splits_to_evaluate` → `postprocess_dataframe` → `evaluate_predictions`.
5. benchmark (if enabled) → `write_metrics_json`, `write_predictions_csv` → `generate_all_figures`.
6. log metrics + artifacts to tracker → `update_metadata(status="completed", duration_s, best checkpoint, best val metric)` → tracker `finish`. On any exception: log traceback, `update_metadata(status="failed", error=…)`, `tracker.finish("FAILED")`, re-raise. `scripts/train.py`: `@hydra.main(config_path="../configs", config_name="default", version_base="1.3")` → `run_experiment(cfg)`. Nothing else. **Integration tests** (marked `integration`, CPU, `< 120 s` total, config = smoke overrides + `paths.output_root=<tmp_path>`): (1) full run creates **every** artifact in §5.6; (2) `metrics.json` has all keys, all three buckets with `n > 0`, stationary `n_true_stationary > 0`; (3) all final values where `stationary_pred` is true equal `0.0`; (4) two consecutive runs create two distinct run dirs and neither is modified by the other; (5) a forced failure (monkeypatch dataset builder to raise) leaves `metadata.json` with `status=failed`; (6) test split's groups are disjoint from train's; (7) one subprocess smoke test: `python scripts/train.py experiment=smoke paths.output_root=<tmp> tracking.enabled=false` exits 0 and leaves no `outputs/` or `lightning_logs/` in the repo root. **Verify:** `pytest -q -m integration` · `python scripts/train.py experiment=smoke` · `git status --short` shows nothing outside `output/` (which is ignored). **Must not:** put logic in `scripts/train.py`.

### WP11 — Secondary scripts

**Owns:** `scripts/evaluate.py`, `scripts/predict.py`, `scripts/visualize.py`, `tests/integration/test_scripts.py`; may add thin helpers to `src/experiment.py` (new functions only).
**Do:** `evaluate.py run=<name>`: `load_run`, re-run evaluation from the checkpoint, write results to `reports/eval_<timestamp>/` (**never overwrite** `metrics.json`). `visualize.py run=<name>`: regenerate figures from `predictions.csv` into `figures/<name>/` under a timestamped subfolder if files exist. `predict.py model.checkpoint=<path> input=<sequence dir>`: run `StreamingSpeedEstimator` over a synthetic/in-memory or frame-folder input and write a predictions CSV into a new run dir.
**Verify:** `pytest tests/integration/test_scripts.py -q`

### WP12 — Acceptance (orchestrator)

Run the checklist in §12. Only then generate `requirements.txt` from the validated Conda environment and tag M1. Prefer `conda list --explicit > conda-spec.txt` for an exact platform-specific lock if desired; `requirements.txt` is retained for pip interoperability.

---

## 9. Orchestration protocol

### 9.1 Dispatch prompt template (paste per agent)

```text
You are implementing ONE work package of a larger project. Read docs/PLAN.md sections 3, 5, 6 and
ONLY your work package (WP<N>) in section 8.

Rules:
- Edit only the files listed under "Owns". Never edit schemas.py or other agents' files.
- Do not change any function signature, dataclass or config key in section 5. If you believe a
  change is needed, append it to docs/CONTRACT_REQUESTS.md and STOP.
- Do not invent behaviour: if the spec is silent, choose the simplest option, note it in your
  handoff report under "Assumptions".
- No print(), no network access in tests, no new dependencies, no hard-coded paths.
- No placeholder implementations: every public function must be fully implemented and tested.
- Tests must assert specific values (hand-computed where specified), not just "does not crash".
- Run every command under "Verify" and paste the real output into your report.
- If anything fails and you cannot fix it within your owned files, STOP and report; do not work around it.

Deliver the handoff report from docs/PLAN.md section 9.3.

```

### 9.2 Gate checks the orchestrator runs per WP (before marking it accepted)

1. `git diff --stat` — only the WP's owned files changed.
2. `ruff check src tests scripts` clean; `pytest <WP tests> -q` passes (re-run yourself; do not trust pasted output).
3. Contract compliance: public signatures match §5.5 (`python -c "import inspect, src.<module> as m; print(inspect.signature(m.<fn>))"`).
4. `grep -rn "print(" src scripts` → empty.
5. `grep -rnE "['\"](output|data)/" src` → only config-default strings, no hard-coded paths in logic.
6. Spot-check **test quality**: open 2–3 tests; each must assert concrete values. Mutation check: temporarily break one core line (e.g. change `<=` to `<` in the detector) and confirm a test fails.
7. No stubs: `grep -rnE "NotImplementedError|TODO|pass$" src` reviewed (only allowed in ABC methods).
8. Update `docs/PROGRESS.md`.

### 9.3 Handoff report template (agent output)

```text
WP: <id>   Status: done | blocked
Files created/changed: <list>
Public API implemented: <signatures>
Verify output: <pasted real output>
Assumptions made: <list or "none">
Deviations from spec: <list or "none">
Known gaps / risks: <list or "none">
Contract change requests: <list or "none">

```

### 9.4 `docs/PROGRESS.md` template

```text
| WP | Agent | Status (todo/in-progress/review/accepted) | Gate notes | Date |
|----|-------|-------------------------------------------|------------|------|

```

### 9.5 `docs/DECISIONS.md` template

```text
| ID | Decision / answer | Source (user/orchestrator/assumed) | Date | Impact |

```

Pre-fill with D1–D8 from §2.1 (marked "assumed").

### 9.6 When agents get stuck

- **Version/API uncertainty (Hydra, Lightning, MLflow):** inspect the installed package (`pip show`, `help()`, source) rather than recalling from memory.
- **Failing test in someone else's module:** report, do not patch.
- **Spec ambiguity:** simplest option + document; do not ask the user directly (the orchestrator batches questions).

---

## 10. Test inventory (all CPU, offline)

| File                                                            | Covers                                                       |
| --------------------------------------------------------------- | ------------------------------------------------------------ |
| `unit/test_buckets.py`, `test_schemas.py`                       | bucket edges, dataclasses                                    |
| `unit/test_config.py`, `test_logging.py`, `test_run_manager.py` | compose/validate, handlers, run dirs, seed, metadata         |
| `unit/test_data.py`                                             | synthetic data, validation policies, splits, dataset         |
| `unit/test_preprocessing.py`                                    | transforms, window index                                     |
| `unit/test_models.py`, `test_training.py`                       | shapes, B=1, overfit-one-batch, checkpoint round trip        |
| `unit/test_postprocessing.py`                                   | detector vectors, smoothing, clip, exact `0.0`               |
| `unit/test_metrics.py`                                          | regression/bucket/stationary/latency metrics                 |
| `unit/test_reporting.py`, `test_visualization.py`               | JSON/CSV contracts, PNG creation                             |
| `unit/test_tracking.py`                                         | NoOp, MLflow failure isolation                               |
| `unit/test_pipeline.py`, `test_benchmark.py`                    | streaming parity, gap reset, timing sanity                   |
| `integration/test_pipeline.py`                                  | full run, artifacts, no leakage, failure metadata, CLI smoke |
| `integration/test_scripts.py`                                   | secondary scripts                                            |

Budgets: unit suite < 60 s, integration < 120 s on a laptop CPU.

---

## 11. After M1 (do not start until M1 accepted and D1–D3, D5 answered)

| Phase                 | Work                                                                                                                                                                                                                                                       | Notes                                                                                                                                                                   |
| --------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| M2-a Real data        | Implement a video/frame-backed `FrameSource` and a `dashcam` dataset config; data-inspection script; verify frames, timestamps, labels, sequence/group IDs, windows, splits.                                                                               | Add `video` extra (OpenCV). Split by drive/video (`group_id`).                                                                                                          |
| M2-b Baselines        | (1) constant train-median speed; (2) a **deployable** classical baseline, e.g. dense optical-flow magnitude features + linear/ridge regression; (3) "previous ground-truth speed" only as an **oracle/persistence reference**, labelled as non-deployable. | Establish the lower bound on val/test with the same evaluation code.                                                                                                    |
| M2-c CNN + temporal   | `cnn_lstm` (pretrained ResNet18 + LSTM) as a registry entry; then GRU/Transformer swaps via config only.                                                                                                                                                   | Hypotheses if frame-level features underperform: frame-difference/optical-flow input channels, 3D/late-fusion temporal models, per-frame feature caching for streaming. |
| M2-d Stationary study | Compare simple threshold vs hysteresis vs persistence, and optionally a multi-task stationary head; select on **val**; report precision/recall/F1 and transition latency.                                                                                  | Config-driven selection.                                                                                                                                                |
| M2-e Runtime          | Benchmark on target hardware (D5), model-only vs end-to-end.                                                                                                                                                                                               | Report mean/median/P95/FPS.                                                                                                                                             |
| M2-f Experiments      | Architecture/hyperparameter comparisons, only on a stable backbone.                                                                                                                                                                                        | One variable at a time; all runs tracked.                                                                                                                               |

---

## 12. M1 Definition of Done (orchestrator checklist)

Environment: **Conda**, using the repository-level `environment.yml`.

```bash
conda env create -f environment.yml
conda activate speed-estimation
python -m pip install -e ".[dev]"
```

Do not use `venv` for the project setup. Conda is the canonical environment manager. `pip` is used only inside the activated Conda environment for the editable project install and packages that are not provided through the Conda specification.

- [ ] `ruff check src tests scripts` clean.
- [ ] `pytest -q` (unit) passes in < 60 s; `pytest -q -m integration` passes in < 120 s.
- [ ] `python scripts/train.py experiment=smoke` exits 0; `python scripts/train.py` (baseline, synthetic) exits 0 on CPU.
- [ ] `output/runs/<run>/` contains every file in §5.6; `output/logs/<run>.log` and the six figures exist and are non-empty.
- [ ] `metadata.json` has seed, versions, git commit, status `completed`, duration, best checkpoint.
- [ ] `metrics.json` has MAE/RMSE/R²/MedAE (overall and raw), all three buckets with `n > 0`, stationary precision/recall/F1 with counts, `model_only` and `end_to_end` latency (mean/median/P95/FPS).
- [ ] `final_speed_mph` is never zeroed by the stationary detector (stationary rows keep their predicted speed); no `-0.0` in `predictions.csv`.
- [ ] Streaming-vs-offline parity test passes.
- [ ] No leakage: split manifest present, groups disjoint, test not used during `fit`.
- [ ] Two runs never share a directory; a failed run is marked `failed` in metadata.
- [ ] With `tracking.enabled=true`, an MLflow run exists tagged with the local run name; with MLflow unreachable the run still completes.
- [ ] Nothing generated outside `output/` (`git status` clean; no `outputs/`, `lightning_logs/`, `mlruns/` in repo root).
- [ ] README documents install, commands, config usage, output contract, assumptions D1–D8.
- [ ] `requirements.txt` generated from the validated Conda environment; optional `conda-spec.txt` may be generated for an exact platform-specific snapshot.