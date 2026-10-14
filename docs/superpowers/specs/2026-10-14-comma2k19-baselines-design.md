# Design: comma2k19 baselines — ingestion + CNN/LSTM, ResNet/LSTM, ResNet/ConvLSTM

**Date:** 2026-10-14
**Status:** approved for implementation
**Scope:** implement the comma2k19 ingestion layer (M2-a, per the approved
`2026-10-13-comma2k19-ingestion-design.md`) and three baseline models
(`cnn_lstm`, `resnet_lstm`, `resnet_convlstm`), plus configs and offline tests.
**Explicitly out of scope:** running decode/training on real data (the user runs
it themselves), the ConvLSTM *experiment* evaluation, feature caching, runtime
benchmark tuning, the stationary study, and any CNN+ConvLSTM combination.

---

## 1. Context

The repo is a complete, tested M1 vertical slice (CNN+GRU on synthetic data).
The data layer already exposes the seams we reuse unchanged: `FrameSource`,
`SpeedSequence`, `validate_sequence`, `make_group_split`/`assert_no_leakage`,
`SpeedDataset`, `build_datasets`, `build_dataloaders`, and the window/preprocessing
machinery in `src/data/preprocessing.py`.

Two things are added:

1. **Ingestion (M2-a):** turn `data/raw/Chunk_1/<route>/<seg>/` (HEVC video +
   `global_pose/*.npy`) into `SpeedSequence`s, via a two-stage decode→cache pass.
2. **Models (M2-c):** three baselines added through the existing
   `ENCODER_REGISTRY` / `TEMPORAL_REGISTRY` / `MODEL_REGISTRY`, reusing the
   already-generic encoder+recurrent model.

## 2. Goals and non-goals

**Goals**

- Implement the comma2k19 ingestion layer faithfully per the approved M2-a design.
- Add three models as registry entries: `cnn_lstm`, `resnet_lstm`, `resnet_convlstm`.
- Support the sliding-window compute dials: per-split `window_stride` (train-only
  stride, val/test locked to 1), `frame_step`, `max_segments`.
- Cache pretrained ResNet weights locally (`weights_path`) and export a standalone
  trained model (`model.pt`) at the end of each run.
- Offline, CPU, no-download unit tests; one `@slow` real-data integration test.

**Non-goals**

- Executing the decode/train on real data (the user runs it).
- ConvLSTM stacking (`num_layers > 1`), CNN+ConvLSTM, GRU/Transformer swaps.
- Per-frame feature caching (M2-e), runtime benchmark targets, stationary study.
- Changing M1 evaluation/splits/post-processing behavior beyond the stride fix.

## 3. Architecture overview

```
data/raw/Chunk_1/<route>/<seg>/{video.hevc, global_pose/frame_times|frame_velocities.npy}
        │  scripts/preprocess.py   (decode HEVC → BGR→RGB → downscale 128² → .npz)
        ▼
data/interim/comma2k19/<route>__<seg>.npz   +  manifest.json
        │  load_comma2k19_sequences   (memory-mapped CachedFrameSource)
        ▼
list[SpeedSequence]  →  validate_sequence → make_group_split → SpeedDataset   (existing)
        │  build_datasets dispatch on cfg.data.source
        ▼
windows [B,T,C,H,W]  →  EncoderRecurrent (encoder → temporal → head)  →  [B]
```

## 4. Components (files)

| File | Action | What |
|---|---|---|
| `src/data/comma2k19.py` | **new** | `decode_segment_frames`, `cache_segment`, `build_manifest`, `load_comma2k19_sequences`, `CachedFrameSource`, `VideoFrameSource` |
| `src/data/dataset.py` | edit | `build_datasets` dispatch on `cfg.data.source`; per-split stride |
| `src/core/config.py` | edit | add `Comma2k19Cfg` + `DataCfg.comma2k19` field |
| `src/models/architectures.py` | edit | rename `TinyCNNGRU`→`EncoderRecurrent`; add `LSTMTemporal`, `ConvLSTM`(+cell), `ResNet18Encoder`; generalized construction + dual-path forward; registry entries |
| `src/models/training.py` | edit | add `save_model_export` / `load_model_export` |
| `src/experiment.py` | edit | call `save_model_export` after loading best checkpoint |
| `configs/dataset/comma2k19.yaml` | **new** | comma2k19 dataset group |
| `configs/model/{cnn_lstm,resnet_lstm,resnet_convlstm}.yaml` | **new** | model configs |
| `scripts/preprocess.py`, `scripts/inspect_data.py` | **new** | decode→cache CLI; data sanity check |
| `tests/unit/test_comma2k19.py` | **new** | offline fixture tests |
| `tests/unit/test_models.py` | edit | add `cnn_lstm`/`resnet_lstm`/`resnet_convlstm` shape+freeze tests; fix unknown-encoder test name |
| `tests/integration/test_comma2k19.py` | **new** | `@slow` real-data test (skips if data absent) |
| `README.md`, `docs/DECISIONS.md` | edit / new | usage + decisions |

## 5. Ingestion layer

### 5.1 Cache format

Per segment `data/interim/comma2k19/<route>__<seg>.npz`:

- `frames` — `uint8 [N, 128, 128, 3]` RGB, downscaled to `cache_image_size`.
- `speeds_mph` — `float64 [N]`.
- `timestamps` — `float64 [N]`.

`manifest.json` (generated deterministically during preprocessing):

```json
{
  "cache_image_size": [128, 128],
  "segments": {
    "<route>/<seg>": {"file": "<route>__<seg>.npz", "n_frames": 1200, "group_id": "<route>"}
  }
}
```

### 5.2 Label provenance

`global_pose/frame_velocities` magnitude `‖v‖` × `2.23694` → mph (D-M2-1). Frame
count always from `len(frame_times)`, never OpenCV metadata. `dt = 0.05 s` →
`frame_rate_hz: 20`. RGB everywhere (OpenCV BGR → RGB at decode/cache).

### 5.3 Frame sources

- `CachedFrameSource` (default): `np.load(path, mmap_mode="r")` — lazy/memory-mapped;
  `get(indices)` indexes only the requested frames. Essential: `build_datasets`
  holds ~250 sequences at once, and eager loading would use ~14 GB RAM.
- `VideoFrameSource` (fallback when `use_cache: false`): lazy cv2 decode, BGR→RGB,
  `__len__` from `frame_times`.

### 5.4 `build_datasets` dispatch

```python
if data_cfg.source == "synthetic":
    sequences = generate_synthetic_sequences(data_cfg, seed)
elif data_cfg.source == "comma2k19":
    sequences = load_comma2k19_sequences(data_cfg, seed)
else:
    raise ConfigError(f"Unknown data source: {data_cfg.source!r}")
```

Everything after "produce `list[SpeedSequence]`" is shared.

### 5.5 IDs and splits

`sequence_id = "<route>/<seg>"`, `group_id = "<route>"` (opaque route dir name).
Splits are on `group_id` via `make_group_split` (deterministic) — no leakage across
routes. Gap-split children are `<id>#<k>` (rare on continuous highway data).

## 6. Config additions

### 6.1 Dataclasses (`src/core/config.py`)

```python
@dataclass
class Comma2k19Cfg:
    speed_source: str = "global_pose"
    use_cache: bool = True
    cache_dir: str = "data/interim/comma2k19"
    cache_image_size: list[int] = field(default_factory=lambda: [128, 128])
    max_segments: int | None = None   # null = all; int = limit for smoke/pilot
```

`DataCfg` gains `comma2k19: Comma2k19Cfg = field(default_factory=Comma2k19Cfg)`
(mirroring the existing `synthetic: SyntheticCfg`).

### 6.2 `configs/dataset/comma2k19.yaml`

```yaml
# @package data
source: comma2k19
root: data/raw/Chunk_1
frame_rate_hz: 20
split_manifest: data/splits/comma2k19_seed${experiment.seed}.json
split_ratios: {train: 0.7, val: 0.15, test: 0.15}
sequence_length: 10
window_stride: 5            # train-only stride (see §7)
frame_step: 1
max_gap_factor: 1.5
on_duplicate_timestamp: raise
allow_negative_speed: false
max_speed_mph: 200
comma2k19:
  speed_source: global_pose
  use_cache: true
  cache_dir: data/interim/comma2k19
  cache_image_size: [128, 128]
  max_segments: null
```

**Deviation from the M2-a doc:** comma2k19-specific keys are nested under a
`comma2k19:` block (matching the `synthetic:` structure), not flat under `data`.

## 7. Sliding-window data layer

### 7.1 Semantics (unchanged, documented)

A window is `T = sequence_length` frames ending at `end`; the label is
`speeds_mph[end]`; causal (no future frames). `frame_step` = temporal subsampling;
`window_stride` = overlap dial. At 20 Hz with `T=10`, each sample is 0.5 s of
history → predict current speed.

### 7.2 Per-split stride (fix)

`build_datasets` currently hardcodes stride 1 for all splits and raises on
`window_stride != 1`. Change to:

```python
for split_name in ("train", "val", "test"):
    split_sequences = [s for s in validated if s.group_id in split[split_name]]
    stride = data_cfg.window_stride if split_name == "train" else 1
    windows = _build_split_windows(split_sequences, data_cfg, stride=stride)
```

Remove the blanket `!= 1` raise. `_validate_semantics` already enforces
`window_stride >= 1`. Val/test stay stride 1 (required for frame-aligned metrics
and the frame-by-frame stationary detector). Default `window_stride: 5` on
comma2k19 (50% overlap at `T=10`; ≈5× faster training). Overridable via
`data.window_stride=1`.

## 8. Models

### 8.1 Generic model `EncoderRecurrent` (renamed from `TinyCNNGRU`)

Construction is generalized so encoder and temporal modules each receive their
full config dict (minus `name`); `input_dim` is injected for the temporal module:

```python
class EncoderRecurrent(BaseSpeedModel):
    def __init__(self, encoder: dict, temporal: dict):
        super().__init__()
        feature_dim = int(encoder.get("feature_dim", 64))
        hidden_size = int(temporal.get("hidden_size", 32))
        # validate names against registries (raise ConfigError) ...
        self.encoder = ENCODER_REGISTRY[encoder["name"]](
            **{k: v for k, v in encoder.items() if k != "name"})
        tmp = {k: v for k, v in temporal.items() if k != "name"}
        tmp["input_dim"] = feature_dim
        self.temporal = TEMPORAL_REGISTRY[temporal["name"]](**tmp)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, frames: Tensor) -> Tensor:   # [B, T, C, H, W]
        b, t, c, h, w = frames.shape
        feats = self.encoder(frames.reshape(b * t, c, h, w))
        if feats.ndim == 2:                         # vector path (LSTM/GRU)
            last = self.temporal(feats.reshape(b, t, -1))      # [B, hidden]
        else:                                       # spatial path (ConvLSTM)
            _, ch, fh, fw = feats.shape
            last = self.temporal(feats.reshape(b, t, ch, fh, fw))  # [B, hidden, fh, fw]
            last = last.mean(dim=(2, 3))            # global pool -> [B, hidden]
        return self.head(last).squeeze(-1)          # [B]
```

`tiny_cnn_gru` remains registered (name unchanged); `TinyCNNEncoder` gains a
`**kwargs` catch-all to tolerate unused keys (e.g. `pretrained`).

### 8.2 `LSTMTemporal`

```python
class LSTMTemporal(nn.Module):
    def __init__(self, input_dim, hidden_size=32, num_layers=1):
        self.lstm = nn.LSTM(input_dim, hidden_size, num_layers, batch_first=True)
    def forward(self, x):            # [B, T, input_dim]
        out, _ = self.lstm(x)
        return out[:, -1, :]         # [B, hidden_size]
```

### 8.3 `ResNet18Encoder` (frozen, vector or spatial)

```python
class ResNet18Encoder(nn.Module):
    def __init__(self, feature_dim=512, pretrained=True, frozen=True,
                 spatial=False, weights_path=None):
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = torchvision.models.resnet18(weights=weights)
        if weights_path is not None:            # local cached weights (offline reuse)
            backbone.load_state_dict(torch.load(weights_path, map_location="cpu"))
        if spatial:
            self.backbone = nn.Sequential(*list(backbone.children())[:-2])  # -> [B,512,H/32,W/32]
            self.proj = nn.Conv2d(512, feature_dim, 1)                       # always project
        else:
            backbone.fc = nn.Identity()         # avgpool+flatten -> [B, 512]
            self.backbone = backbone
            self.proj = nn.Linear(512, feature_dim) if feature_dim != 512 else nn.Identity()
        if frozen:
            for p in self.backbone.parameters():
                p.requires_grad_(False)
            self.backbone.eval()                # BN uses running stats; avoids B=1 train-mode BN errors

    def forward(self, x):                        # [B, C, H, W]
        return self.proj(self.backbone(x))       # [B, feature_dim] or [B, feature_dim, H', W']
```

Notes:

- `spatial=false` → `[B, 512]` vector (global-pooled).
- `spatial=true` → conv stem only, always 1×1-projected to `feature_dim` channels
  (default 32). At 128² the map is `[feature_dim, 4, 4]`.
- `frozen=true` → backbone `requires_grad=False` + `eval()`; only projection,
  temporal, and head train. `count_parameters()` (sums `requires_grad`) reports
  just those.

### 8.4 `ConvLSTM` (spatially recurrent, single layer, "very small")

Adapted from the KL4805 reference (`ConvLSTM-Pytorch`, Shi et al. 2015),
`batch_first`, single layer:

```python
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        pad = kernel_size // 2
        self.conv = nn.Conv2d(input_dim + hidden_dim, 4 * hidden_dim, kernel_size, padding=pad)
    def forward(self, x, state):
        h, c = state
        i, f, g, o = self.conv(torch.cat([x, h], dim=1)).chunk(4, dim=1)
        i, f, g, o = torch.sigmoid(i), torch.sigmoid(f), torch.tanh(g), torch.sigmoid(o)
        c2 = f * c + i * g
        return o * torch.tanh(c2), c2

class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_size, kernel_size=3, num_layers=1):
        if num_layers != 1:
            raise ConfigError("ConvLSTM num_layers must be 1 (stacking is future work)")
        self.cell = ConvLSTMCell(input_dim, hidden_size, kernel_size)
        self.hidden_size = hidden_size
    def forward(self, x):                         # [B, T, C, H, W]
        b, t, _, h, w = x.shape
        h = torch.zeros(b, self.hidden_size, h, w, device=x.device)
        c = torch.zeros_like(h)
        for step in range(t):
            h, c = self.cell(x[:, step], (h, c))
        return h                                  # [B, hidden_size, h, w]
```

### 8.5 Registries (final)

```python
ENCODER_REGISTRY = {"tiny_cnn": TinyCNNEncoder, "resnet18": ResNet18Encoder}
TEMPORAL_REGISTRY = {"gru": GRUTemporal, "lstm": LSTMTemporal, "convlstm": ConvLSTM}
MODEL_REGISTRY = {
    "tiny_cnn_gru": EncoderRecurrent,
    "cnn_lstm": EncoderRecurrent,
    "resnet_lstm": EncoderRecurrent,
    "resnet_convlstm": EncoderRecurrent,
}
```

### 8.6 Model configs

```yaml
# configs/model/cnn_lstm.yaml
name: cnn_lstm
encoder: {name: tiny_cnn, pretrained: false, feature_dim: 64}
temporal: {name: lstm, hidden_size: 32, num_layers: 1}
checkpoint: null
```

```yaml
# configs/model/resnet_lstm.yaml
name: resnet_lstm
encoder: {name: resnet18, pretrained: true, feature_dim: 512, frozen: true,
          spatial: false, weights_path: null}
temporal: {name: lstm, hidden_size: 64, num_layers: 1}
checkpoint: null
```

```yaml
# configs/model/resnet_convlstm.yaml
name: resnet_convlstm
encoder: {name: resnet18, pretrained: true, feature_dim: 32, frozen: true,
          spatial: true, weights_path: null}
temporal: {name: convlstm, hidden_size: 32, kernel_size: 3, num_layers: 1}
checkpoint: null
```

`preprocessing.image_size` (per run, documented in README):

- `cnn_lstm` → `[64, 64]`
- `resnet_lstm` → `[128, 128]`
- `resnet_convlstm` → `[128, 128]`

### 8.7 Freeze semantics (summary)

`frozen=true` → backbone weights `requires_grad=False`, `backbone.eval()`. Only
projection + LSTM/ConvLSTM + head train. `pretrained=true` downloads ImageNet
weights at build time unless `weights_path` points to a local file. Offline tests
always use `pretrained: false`.

## 9. ResNet caching and model export

- **Pretrained weight caching:** torchvision caches ImageNet weights to
  `~/.cache/torch/hub/checkpoints/` after first download. `weights_path` lets the
  user download once and reuse offline.
- **Trained-model export:** `experiment.run_experiment` calls
  `save_model_export(loaded_model, paths.run_dir / "model.pt", cfg.model)` after
  loading the best checkpoint. The file is self-describing:

  ```python
  {"state_dict": model.state_dict(), "model_name": cfg.name,
   "encoder": cfg.encoder, "temporal": cfg.temporal}
  ```

  `load_model_export(path) -> BaseSpeedModel` (in `src/models/training.py`)
  rebuilds via `build_model` and loads the `state_dict` — no Lightning needed,
  for `scripts/predict.py` and deployment.

## 10. Error handling

- Missing `root` or zero segments → `DataValidationError`.
- Missing `global_pose/frame_times` or `frame_velocities` → `DataValidationError`.
- Decoded frame count ≠ `len(frame_times)` during caching → raise (fail loudly).
- OpenCV missing (preprocess, or `use_cache: false`) → clear error pointing to
  `pip install -e ".[video]"`.
- Unknown model/encoder/temporal name → `ConfigError` (existing).
- Reuse existing policies for gaps/duplicates/negatives/speed-out-of-range.

## 11. Testing

- **`tests/unit/test_comma2k19.py`** (offline, fixtures — no OpenCV, no network):
  mph conversion, RGB order, `sequence_id`/`group_id` construction, speed
  magnitude, 20 Hz dt, gap-split reuse, split-by-route leakage, `max_segments`
  limit, `CachedFrameSource` mmap reads, `build_datasets` dispatch, per-split
  stride (train stride 5 / val+test stride 1).
- **`tests/unit/test_models.py`** (offline):
  - `cnn_lstm` output `[B]` for B=1 and B=4, finite.
  - `resnet_lstm` (`pretrained=false, frozen=true`) output `[B]`, finite; assert
    backbone params have `requires_grad=False` while LSTM/head train.
  - `resnet_convlstm` (`pretrained=false, frozen=true`, `spatial=true`) output
    `[B]`, finite; same frozen-param assertion; assert the spatial path is taken
    (encoder output is 4-D).
  - Update `test_unknown_encoder_raises` to use a genuinely-fake name (it
    currently uses `"resnet"`, now adjacent to `resnet18`).
  - Existing `tiny_cnn_gru` tests unchanged.
- **`tests/integration/test_comma2k19.py`** (`@pytest.mark.slow`, skips if
  `data/raw/Chunk_1` absent): one real segment decodes 1200 frames, speeds finite,
  `dt = 0.05`, mph conversion.
- Existing M1 unit + integration suites keep passing unchanged.

## 12. Documentation

- `README.md`: comma2k19 setup (preprocess → train), cache explanation/disk
  footprint (128² ≈ 13 GiB for Chunk_1), model selection, overrides, example
  commands:

  ```bash
  python scripts/preprocess.py dataset=comma2k19
  python scripts/preprocess.py dataset=comma2k19 data.comma2k19.max_segments=10
  python scripts/train.py dataset=comma2k19 model=cnn_lstm preprocessing.image_size=[64,64]
  python scripts/train.py dataset=comma2k19 model=resnet_lstm preprocessing.image_size=[128,128]
  python scripts/train.py dataset=comma2k19 model=resnet_convlstm preprocessing.image_size=[128,128]
  ```

- `docs/DECISIONS.md` (create): record D1–D8 (PLAN §2.1), D-M2-1…9 (ingestion),
  plus: nested `comma2k19:` config; cache 128 / CNN 64 / ResNet 128; per-split
  stride; frozen-ResNet semantics; `window_stride=5` default; three-model baseline
  set; `weights_path` + `model.pt` export.

## 13. Decisions summary

| ID | Decision |
| --- | --- |
| D-B-1 | Three baselines: `cnn_lstm`, `resnet_lstm`, `resnet_convlstm`. |
| D-B-2 | Cache at 128×128; CNN consumes 64, ResNet consumes 128. |
| D-B-3 | ResNet backbone frozen (`requires_grad=False` + `eval()`); train projection + temporal + head only. |
| D-B-4 | ResNet vector mode global-pools to 512-d; spatial mode 1×1-projects to 32-ch maps (4×4 at 128²). |
| D-B-5 | ConvLSTM single-layer, "very small" (hidden 16–32 channels); stacking deferred. |
| D-B-6 | Per-split stride: train uses `data.window_stride` (default 5), val/test locked to 1. |
| D-B-7 | `weights_path` for offline ResNet reuse; `model.pt` standalone export per run. |
| D-B-8 | `EncoderRecurrent` generalizes the old `TinyCNNGRU`; `tiny_cnn_gru` name stays registered. |

## 14. Out of scope / notes

- **Deferred:** per-frame feature caching (M2-e), ConvLSTM stacking, CNN+ConvLSTM,
  GRU/Transformer swaps, stationary study, runtime targets.
- **Known existing limitation (not changed):** `run_experiment` evaluates on CPU
  (`collect_raw_predictions(..., "cpu", ...)`); fine for M1, may want device
  config later for real-data runs — flagged, out of scope here.
- **ConvLSTM map size:** at 128² the spatial map is 4×4; if ConvLSTM underperforms,
  bump `cache_image_size` to 224 (7×7 maps) with a config override, no code change.
- **Label caveat (from M2-a):** `frame_velocities` comes from comma.ai's
  vision-fused INS/GNSS optimizer, so the label is not strictly independent of the
  frames; recorded in `docs/DECISIONS.md`.
