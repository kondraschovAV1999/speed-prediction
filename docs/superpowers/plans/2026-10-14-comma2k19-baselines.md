# comma2k19 Baselines Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the comma2k19 ingestion layer and three baseline models (`cnn_lstm`, `resnet_lstm`, `resnet_convlstm`) to the existing speed-estimation pipeline, with configs and offline tests.

**Architecture:** Reuse the existing `SpeedSequence -> validate -> split -> window -> SpeedDataset` pipeline. Add a two-stage decode->cache ingestion path (`scripts/preprocess.py` -> `data/interim/comma2k19/*.npz` -> `load_comma2k19_sequences`). Add models as registry entries on the existing generic encoder+recurrent model (`EncoderRecurrent`), which dispatches on vector (LSTM/GRU) vs spatial (ConvLSTM) encoder output.

**Tech Stack:** Python 3.11+, PyTorch 2.x + torchvision, Lightning 2.x, Hydra/OmegaConf, NumPy, pandas, pytest, ruff.

**Spec:** `docs/superpowers/specs/2026-10-14-comma2k19-baselines-design.md` (authoritative).

## Global Constraints

- Conda env name: `speed-pred` (the user's active environment). All commands run inside it via the harness (do not wrap in `conda activate`).
- Python 3.11+; `from __future__ import annotations` at the top of every module; type hints on public functions.
- Import style: package imported as `src` (e.g. `from src.core.config import DataCfg`).
- `import lightning as L` (never `pytorch_lightning`).
- **No `print()`** anywhere (ruff rule `T20`). Use `logging.getLogger(__name__)` or `from src.core.logging_utils import get_logger`.
- No hard-coded paths or hyperparameters in code; read from typed config. Defaults live in `configs/`.
- Tests run **offline on CPU and never download weights** (`pretrained: false` everywhere in tests).
- Units: speeds in mph (`float`), timestamps in seconds (`float`); images `uint8 HWC` at I/O boundary, `float32 CHW` in tensors.
- Model contract: `forward(frames[B,T,C,H,W]) -> Tensor[B]` (mph, unclipped).
- ruff: line-length 100, rules `E,F,I,B,UP,T20`. Verify with `ruff check src tests scripts`.
- Fail loudly: raise `ConfigError` / `DataValidationError`; never `except: pass`; never return fake values.
- OpenCV imported **lazily** inside `src/data/comma2k19.py` (it stays in the `video` extra).
- Commit style: `feat:` / `test:` / `docs:` prefix. One commit per task.

---

## Task 1: Config schema — `Comma2k19Cfg`, `DataCfg.comma2k19`, `EvaluationCfg.device`

**Files:**
- Modify: `src/core/config.py`
- Create: `configs/dataset/comma2k19.yaml`
- Modify: `configs/default.yaml`
- Test: `tests/unit/test_config.py`

**Interfaces:**
- Produces: `Comma2k19Cfg(speed_source: str, use_cache: bool, cache_dir: str, cache_image_size: list[int], max_segments: int | None)`; `DataCfg.comma2k19: Comma2k19Cfg`; `EvaluationCfg.device: str = "auto"`.

- [ ] **Step 1: Add the `Comma2k19Cfg` dataclass** (after `SyntheticCfg` in `src/core/config.py`, around line 49):

```python
@dataclass
class Comma2k19Cfg:
    speed_source: str = "global_pose"
    use_cache: bool = True
    cache_dir: str = "data/interim/comma2k19"
    cache_image_size: list[int] = field(default_factory=lambda: [128, 128])
    max_segments: int | None = None  # null = all; int = limit for smoke/pilot
```

- [ ] **Step 2: Add the field to `DataCfg`** (after `synthetic: SyntheticCfg = field(default_factory=SyntheticCfg)`):

```python
    comma2k19: Comma2k19Cfg = field(default_factory=Comma2k19Cfg)
```

- [ ] **Step 3: Add `device` to `EvaluationCfg`** (append the field to the existing dataclass):

```python
@dataclass
class EvaluationCfg:
    stationary_label_threshold_mph: float = 2.0
    speed_buckets: list[BucketCfg] = field(default_factory=list)
    splits_to_evaluate: list[str] = field(default_factory=lambda: ["val", "test"])
    device: str = "auto"  # "auto" -> cuda if available, else cpu (see resolve_device)
```

- [ ] **Step 4: Create `configs/dataset/comma2k19.yaml`**:

```yaml
# @package data
source: comma2k19
root: data/raw/Chunk_1
frame_rate_hz: 20
split_manifest: data/splits/comma2k19_seed${experiment.seed}.json
split_ratios: {train: 0.7, val: 0.15, test: 0.15}
sequence_length: 10
window_stride: 5            # train-only stride (see Task 4)
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

- [ ] **Step 5: Add `device: auto` to `configs/default.yaml`** (inside the `evaluation:` block, after `splits_to_evaluate: [val, test]`):

```yaml
evaluation:
  stationary_label_threshold_mph: 2.0
  speed_buckets:
    - {name: low,    min_mph: 0,  max_mph: 15}
    - {name: medium, min_mph: 15, max_mph: 45}
    - {name: high,   min_mph: 45, max_mph: null}
  splits_to_evaluate: [val, test]
  device: auto
```

- [ ] **Step 6: Write the tests** (append to `tests/unit/test_config.py`):

```python
def test_compose_comma2k19():
    cfg = validate_config(compose_config(["dataset=comma2k19"]))
    assert cfg.data.source == "comma2k19"
    assert cfg.data.frame_rate_hz == 20
    assert cfg.data.window_stride == 5
    assert cfg.data.comma2k19.cache_image_size == [128, 128]
    assert cfg.data.comma2k19.max_segments is None


def test_evaluation_device_defaults_auto():
    cfg = validate_config(compose_config())
    assert cfg.evaluation.device == "auto"
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `pytest tests/unit/test_config.py -q`
Expected: all pass (existing 11 + 2 new).

- [ ] **Step 8: Verify lint**

Run: `ruff check src/core/config.py`
Expected: clean.

- [ ] **Step 9: Commit**

```bash
git add src/core/config.py configs/dataset/comma2k19.yaml configs/default.yaml tests/unit/test_config.py
git commit -m "feat: add comma2k19 config schema and evaluation device"
```

---

## Task 2: comma2k19 frame sources + segment discovery

**Files:**
- Create: `src/data/comma2k19.py`
- Test: `tests/unit/test_comma2k19.py`

**Interfaces:**
- Produces: `CachedFrameSource(path) -> FrameSource`; `VideoFrameSource(segment_dir, n_frames, image_size) -> FrameSource`; `discover_segments(root: Path) -> list[tuple[str, str, Path]]`; `MPH_PER_MPS: float`.

- [ ] **Step 1: Create `src/data/comma2k19.py`** (full initial content):

```python
"""comma2k19 ingestion (M2-a): frame sources, discovery, decode, cache, load.

OpenCV is imported lazily inside the functions that need it, so this module can
be imported in offline tests (it only imports numpy/pathlib/stdlib at top level).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from src.core.logging_utils import get_logger
from src.core.schemas import DataValidationError

logger = get_logger(__name__)

# 1 m/s = 3600 / 1609.344 mph (exact).
MPH_PER_MPS = 2.2369362921


def _require_cv2():
    """Import cv2 or raise a clear, actionable error."""
    try:
        import cv2  # noqa: PLC0415
    except ImportError as exc:
        raise DataValidationError(
            "OpenCV is required for comma2k19 video decoding. Install with "
            "`pip install -e '.[video]'`."
        ) from exc
    return cv2


class CachedFrameSource:
    """Memory-mapped frame source over an (uncompressed) cached .npz."""

    def __init__(self, path: str | Path):
        self._f = np.load(str(path), mmap_mode="r")
        self._n = int(self._f["frames"].shape[0])

    def __len__(self) -> int:
        return self._n

    def get(self, indices: Sequence[int]) -> np.ndarray:  # uint8 [k, H, W, C]
        return np.asarray(self._f["frames"][list(indices)])


class VideoFrameSource:
    """Lazy cv2 decode of a segment's video.hevc (fallback when use_cache=false)."""

    def __init__(self, segment_dir: Path, n_frames: int, image_size: tuple[int, int]):
        self.segment_dir = segment_dir
        self.n_frames = n_frames
        self.image_size = image_size

    def __len__(self) -> int:
        return self.n_frames

    def get(self, indices: Sequence[int]) -> np.ndarray:
        cv2 = _require_cv2()
        h, w = self.image_size
        cap = cv2.VideoCapture(str(self.segment_dir / "video.hevc"))
        if not cap.isOpened():
            raise DataValidationError(f"Could not open video: {self.segment_dir / 'video.hevc'}")
        out = []
        for idx in sorted(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                raise DataValidationError(f"Failed to decode frame {idx} in {self.segment_dir}")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            out.append(cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA))
        cap.release()
        return np.stack(out)


def discover_segments(root: Path) -> list[tuple[str, str, Path]]:
    """Return sorted ``(route, seg, segment_dir)`` tuples under ``root``."""
    segments: list[tuple[str, str, Path]] = []
    for route_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for seg_dir in sorted(p for p in route_dir.iterdir() if p.is_dir()):
            if (seg_dir / "global_pose").is_dir() and (seg_dir / "video.hevc").exists():
                segments.append((route_dir.name, seg_dir.name, seg_dir))
    return segments
```

- [ ] **Step 2: Write the tests** in `tests/unit/test_comma2k19.py`:

```python
"""Tests for src.data.comma2k19 (offline: no OpenCV, no network)."""

from __future__ import annotations

import builtins
from pathlib import Path

import numpy as np
import pytest

from src.core.schemas import DataValidationError
from src.data.comma2k19 import CachedFrameSource, discover_segments


@pytest.fixture
def cached_npz(tmp_path: Path) -> Path:
    frames = np.arange(5 * 8 * 8 * 3, dtype=np.uint8).reshape(5, 8, 8, 3)
    path = tmp_path / "seg.npz"
    np.savez(path, frames=frames, speeds_mph=np.ones(5), timestamps=np.arange(5))
    return path


def test_cached_frame_source_len_and_get(cached_npz):
    src = CachedFrameSource(cached_npz)
    assert len(src) == 5
    got = src.get([0, 2])
    assert got.shape == (2, 8, 8, 3)
    assert got.dtype == np.uint8


def test_discover_segments_route_seg(tmp_path: Path):
    seg = tmp_path / "routeA" / "0"
    seg.mkdir(parents=True)
    (seg / "global_pose").mkdir()
    (seg / "video.hevc").write_bytes(b"")
    (tmp_path / "routeA" / "1").mkdir(parents=True)  # no global_pose -> skipped
    assert discover_segments(tmp_path) == [("routeA", "0", seg)]


def test_video_frame_source_missing_cv2_raises(tmp_path: Path, monkeypatch):
    from src.data.comma2k19 import VideoFrameSource

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "cv2":
            raise ImportError("No module named 'cv2'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    src = VideoFrameSource(tmp_path, n_frames=1, image_size=(8, 8))
    with pytest.raises(DataValidationError, match="OpenCV"):
        src.get([0])
```

- [ ] **Step 3: Run tests to verify they pass**

Run: `pytest tests/unit/test_comma2k19.py -q`
Expected: 3 passed.

- [ ] **Step 4: Verify lint**

Run: `ruff check src/data/comma2k19.py tests/unit/test_comma2k19.py`
Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add src/data/comma2k19.py tests/unit/test_comma2k19.py
git commit -m "feat: add comma2k19 frame sources and discovery"
```

## Task 3: comma2k19 decode, cache, manifest, load

**Files:**
- Modify: `src/data/comma2k19.py` (append the functions below)
- Test: `tests/unit/test_comma2k19.py` (append tests)

**Interfaces:**
- Consumes: `_require_cv2`, `CachedFrameSource`, `MPH_PER_MPS` (Task 2).
- Produces: `decode_segment_frames(segment_dir, image_size) -> np.ndarray`; `cache_segment(segment_dir, cache_path, image_size) -> dict`; `build_manifest(cache_dir, segments) -> None`; `load_comma2k19_sequences(cfg: DataCfg, seed: int) -> list[SpeedSequence]`.

- [ ] **Step 1: Append the ingest functions to `src/data/comma2k19.py`**, and add the `json`, `DataCfg`, `SpeedSequence` imports at the top:

```python
import json  # add to the existing imports at top

from src.core.config import DataCfg   # add
from src.core.schemas import SpeedSequence  # add (alongside DataValidationError)


def decode_segment_frames(segment_dir: Path, image_size: tuple[int, int]) -> np.ndarray:
    """Decode video.hevc to ``uint8 [N, H, W, 3]`` RGB, downscaled to ``image_size``."""
    cv2 = _require_cv2()
    h, w = image_size
    cap = cv2.VideoCapture(str(segment_dir / "video.hevc"))
    if not cap.isOpened():
        raise DataValidationError(f"Could not open video: {segment_dir / 'video.hevc'}")
    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA))
    cap.release()
    if not frames:
        return np.empty((0, h, w, 3), dtype=np.uint8)
    return np.stack(frames).astype(np.uint8)


def cache_segment(segment_dir: Path, cache_path: Path, image_size: tuple[int, int]) -> dict:
    """Decode + cache one segment to an *uncompressed* .npz; return ``{"n_frames": int}``."""
    global_pose = segment_dir / "global_pose"
    frame_times = np.load(global_pose / "frame_times.npy")
    velocities = np.load(global_pose / "frame_velocities.npy")
    speeds_mph = np.linalg.norm(velocities, axis=1) * MPH_PER_MPS

    frames = decode_segment_frames(segment_dir, image_size)
    if frames.shape[0] != frame_times.shape[0]:
        raise DataValidationError(
            f"{segment_dir}: decoded {frames.shape[0]} frames but "
            f"frame_times has {frame_times.shape[0]}"
        )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    # np.savez (NOT savez_compressed) so the cache is memory-mappable.
    np.savez(cache_path, frames=frames, speeds_mph=speeds_mph, timestamps=frame_times)
    return {"n_frames": int(frame_times.shape[0])}


def build_manifest(cache_dir: Path, segments: dict[str, dict]) -> None:
    """Write manifest.json mapping ``<route>/<seg>`` -> {file, n_frames, group_id}."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "manifest.json").write_text(
        json.dumps({"segments": segments}, indent=2, sort_keys=True), encoding="utf-8"
    )


def load_comma2k19_sequences(cfg: DataCfg, seed: int) -> list[SpeedSequence]:
    """Load cached comma2k19 segments into SpeedSequence objects (sorted, optional cap)."""
    cache_dir = Path(cfg.comma2k19.cache_dir)
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.exists():
        raise DataValidationError(
            f"Manifest not found at {manifest_path}; run scripts/preprocess.py first"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = sorted(manifest["segments"].items())
    if cfg.comma2k19.max_segments is not None:
        entries = entries[: cfg.comma2k19.max_segments]

    sequences: list[SpeedSequence] = []
    for seg_id, info in entries:
        npz_path = cache_dir / info["file"]
        data = np.load(npz_path, mmap_mode="r")
        sequences.append(
            SpeedSequence(
                sequence_id=seg_id,
                group_id=info["group_id"],
                frames=CachedFrameSource(npz_path),
                speeds_mph=np.asarray(data["speeds_mph"], dtype=float),
                timestamps=np.asarray(data["timestamps"], dtype=float),
            )
        )
    return sequences
```

- [ ] **Step 2: Append the tests** to `tests/unit/test_comma2k19.py`:

```python
import json  # add at top of test file

import src.data.comma2k19 as c2k
from src.core.config import Comma2k19Cfg, DataCfg


def test_cache_segment_mph_conversion(tmp_path: Path, monkeypatch):
    seg = tmp_path / "seg"
    gp = seg / "global_pose"
    gp.mkdir(parents=True)
    velocities = np.array([[3.0, 0.0, 0.0], [0.0, 4.0, 0.0]])  # 3 and 4 m/s
    np.save(gp / "frame_times.npy", np.array([0.0, 0.05]))
    np.save(gp / "frame_velocities.npy", velocities)
    monkeypatch.setattr(
        c2k, "decode_segment_frames", lambda d, s: np.zeros((2, 4, 4, 3), dtype=np.uint8)
    )
    info = c2k.cache_segment(seg, tmp_path / "out.npz", (4, 4))
    assert info["n_frames"] == 2
    data = np.load(tmp_path / "out.npz")
    assert np.allclose(data["speeds_mph"], [3.0 * c2k.MPH_PER_MPS, 4.0 * c2k.MPH_PER_MPS])
    assert data["frames"].shape == (2, 4, 4, 3)


def test_cache_segment_frame_count_mismatch_raises(tmp_path: Path, monkeypatch):
    seg = tmp_path / "seg"
    gp = seg / "global_pose"
    gp.mkdir(parents=True)
    np.save(gp / "frame_times.npy", np.array([0.0, 0.05, 0.10]))  # 3 frames
    np.save(gp / "frame_velocities.npy", np.zeros((3, 3)))
    monkeypatch.setattr(
        c2k, "decode_segment_frames", lambda d, s: np.zeros((2, 4, 4, 3), dtype=np.uint8)
    )
    with pytest.raises(DataValidationError):
        c2k.cache_segment(seg, tmp_path / "out.npz", (4, 4))


def test_load_sequences_ids_and_max_segments(tmp_path: Path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    for i in range(3):
        fname = f"route{i}__0.npz"
        np.savez(
            cache_dir / fname,
            frames=np.zeros((5, 4, 4, 3), dtype=np.uint8),
            speeds_mph=np.ones(5),
            timestamps=np.arange(5),
        )
    manifest = {
        "segments": {
            f"route{i}/0": {"file": f"route{i}__0.npz", "n_frames": 5, "group_id": f"route{i}"}
            for i in range(3)
        }
    }
    (cache_dir / "manifest.json").write_text(json.dumps(manifest))

    cfg = DataCfg(comma2k19=Comma2k19Cfg(cache_dir=str(cache_dir), max_segments=2))
    seqs = c2k.load_comma2k19_sequences(cfg, seed=0)
    assert len(seqs) == 2
    assert seqs[0].sequence_id == "route0/0"
    assert seqs[0].group_id == "route0"
    assert seqs[0].speeds_mph.shape == (5,)
    assert seqs[0].timestamps.shape == (5,)


def test_load_sequences_missing_manifest_raises(tmp_path: Path):
    cfg = DataCfg(comma2k19=Comma2k19Cfg(cache_dir=str(tmp_path / "nope")))
    with pytest.raises(DataValidationError):
        c2k.load_comma2k19_sequences(cfg, seed=0)


def test_preprocess_flow_glue(tmp_path: Path, monkeypatch):
    """discover -> cache -> manifest -> load, end to end (decode mocked)."""
    seg = tmp_path / "raw" / "routeA" / "0"
    gp = seg / "global_pose"
    gp.mkdir(parents=True)
    (seg / "video.hevc").write_bytes(b"")
    np.save(gp / "frame_times.npy", np.array([0.0, 0.05]))
    np.save(gp / "frame_velocities.npy", np.array([[5.0, 0, 0], [0, 6.0, 0]]))
    monkeypatch.setattr(
        c2k, "decode_segment_frames", lambda d, s: np.zeros((2, 4, 4, 3), dtype=np.uint8)
    )

    cache_dir = tmp_path / "cache"
    manifest: dict = {}
    for route, sid, seg_dir in c2k.discover_segments(tmp_path / "raw"):
        fname = f"{route}__{sid}.npz"
        info = c2k.cache_segment(seg_dir, cache_dir / fname, (4, 4))
        manifest[f"{route}/{sid}"] = {"file": fname, "n_frames": info["n_frames"], "group_id": route}
    c2k.build_manifest(cache_dir, manifest)

    cfg = DataCfg(comma2k19=Comma2k19Cfg(cache_dir=str(cache_dir)))
    seqs = c2k.load_comma2k19_sequences(cfg, seed=0)
    assert len(seqs) == 1
    assert seqs[0].sequence_id == "routeA/0"
    assert seqs[0].speeds_mph.shape == (2,)
```

- [ ] **Step 3: Run tests**

Run: `pytest tests/unit/test_comma2k19.py -q`
Expected: 8 passed.

- [ ] **Step 4: Verify lint**

Run: `ruff check src/data/comma2k19.py tests/unit/test_comma2k19.py`
Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add src/data/comma2k19.py tests/unit/test_comma2k19.py
git commit -m "feat: add comma2k19 decode/cache/load"
```

---

## Task 4: `build_datasets` dispatch + per-split stride

**Files:**
- Modify: `src/data/dataset.py`
- Test: `tests/unit/test_data.py` (replace one test), `tests/unit/test_comma2k19.py` (add one test)

**Interfaces:**
- Consumes: `load_comma2k19_sequences(cfg: DataCfg, seed: int)` (Task 3).
- Produces: `build_datasets` dispatches on `cfg.data.source`; train uses `window_stride`, val/test use stride 1.

- [ ] **Step 1: Update imports in `src/data/dataset.py`**

Change:
```python
from src.core.schemas import DataValidationError, SpeedSequence
```
to:
```python
from src.core.schemas import ConfigError, DataValidationError, SpeedSequence
```
And add after the existing `from src.data.preprocessing import ...` line:
```python
from src.data.comma2k19 import load_comma2k19_sequences
```

- [ ] **Step 2: Replace the hardcoded source + blanket stride guard in `build_datasets`**

Replace:
```python
    if data_cfg.window_stride != 1:
        raise DataValidationError(
            f"window_stride={data_cfg.window_stride} is not supported: "
            "evaluation splits (val/test) require stride 1"
        )

    sequences = generate_synthetic_sequences(data_cfg, seed)
```
with:
```python
    if data_cfg.source == "synthetic":
        sequences = generate_synthetic_sequences(data_cfg, seed)
    elif data_cfg.source == "comma2k19":
        sequences = load_comma2k19_sequences(data_cfg, seed)
    else:
        raise ConfigError(f"Unknown data source: {data_cfg.source!r}")
```

- [ ] **Step 3: Per-split stride in the split loop**

Replace:
```python
        split_sequences = [s for s in validated if s.group_id in split[split_name]]
        windows = _build_split_windows(split_sequences, data_cfg, stride=1)
```
with:
```python
        split_sequences = [s for s in validated if s.group_id in split[split_name]]
        stride = data_cfg.window_stride if split_name == "train" else 1
        windows = _build_split_windows(split_sequences, data_cfg, stride=stride)
```

- [ ] **Step 4: Replace the obsolete test** in `tests/unit/test_data.py`

Delete this method inside `class TestDataset`:
```python
    def test_eval_stride_not_one_raises(self, tiny_config):
        from src.data.dataset import build_datasets

        tiny_config.data.window_stride = 2
        with pytest.raises(DataValidationError):
            build_datasets(tiny_config, seed=42)
```
and add a module-level helper (near the top of the file, after imports) plus a replacement method:

```python
def _ends_by_seq(ds):
    ends: dict[int, list[int]] = {}
    for seq_idx, end in ds.window_index:
        ends.setdefault(seq_idx, []).append(end)
    return ends


# (inside class TestDataset)
    def test_train_stride_two_val_test_stride_one(self, tiny_config):
        from src.data.dataset import build_datasets

        tiny_config.data.window_stride = 2
        datasets = build_datasets(tiny_config, seed=42)
        for seq_idx, ends in _ends_by_seq(datasets["train"]).items():
            assert all(b - a == 2 for a, b in zip(ends, ends[1:]))
        for split_name in ("val", "test"):
            for seq_idx, ends in _ends_by_seq(datasets[split_name]).items():
                assert all(b - a == 1 for a, b in zip(ends, ends[1:]))
```

- [ ] **Step 5: Add the dispatch test** to `tests/unit/test_comma2k19.py`:

```python
def test_build_datasets_dispatches_comma2k19(tmp_path: Path):
    from src.core.config import compose_config, validate_config
    from src.data.dataset import build_datasets

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    manifest_segments = {}
    for g in range(6):
        fname = f"route{g}__0.npz"
        np.savez(
            cache_dir / fname,
            frames=np.zeros((20, 8, 8, 3), dtype=np.uint8),
            speeds_mph=np.ones(20) * 20.0,
            timestamps=np.arange(20) * 0.05,
        )
        manifest_segments[f"route{g}/0"] = {"file": fname, "n_frames": 20, "group_id": f"route{g}"}
    (cache_dir / "manifest.json").write_text(json.dumps({"segments": manifest_segments}))

    cfg = validate_config(compose_config(["dataset=comma2k19", "preprocessing.image_size=[8,8]"]))
    cfg.data.comma2k19.cache_dir = str(cache_dir)
    cfg.data.split_manifest = str(tmp_path / "splits" / "m.json")

    datasets = build_datasets(cfg, seed=42)
    assert set(datasets) == {"train", "val", "test"}
    assert all(len(ds) > 0 for ds in datasets.values())
```

- [ ] **Step 6: Run the data + comma2k19 test suites**

Run: `pytest tests/unit/test_data.py tests/unit/test_comma2k19.py -q`
Expected: all pass.

- [ ] **Step 7: Verify the whole unit suite still passes**

Run: `pytest tests/unit -q`
Expected: all pass.

- [ ] **Step 8: Commit**

```bash
git add src/data/dataset.py tests/unit/test_data.py tests/unit/test_comma2k19.py
git commit -m "feat: dispatch comma2k19 source and per-split stride"
```

## Task 5: Generalize model + add `cnn_lstm`

**Files:**
- Modify: `src/models/architectures.py`
- Create: `configs/model/cnn_lstm.yaml`
- Test: `tests/unit/test_models.py`

**Interfaces:**
- Produces: `EncoderRecurrent` (renamed from `TinyCNNGRU`), `LSTMTemporal`, `TEMPORAL_REGISTRY["lstm"]`, `MODEL_REGISTRY["cnn_lstm"]`.

- [ ] **Step 1: Add `**kwargs` to `TinyCNNEncoder.__init__`** (so the stray `pretrained` key is tolerated). Change the signature line only:

```python
    def __init__(self, feature_dim: int = 64, **kwargs):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Linear(64 * 4 * 4, feature_dim)
```

- [ ] **Step 2: Add `LSTMTemporal`** (after `GRUTemporal`):

```python
class LSTMTemporal(nn.Module):
    """LSTM over the per-frame feature sequence; returns the last timestep."""

    def __init__(self, input_dim: int, hidden_size: int = 32, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim, hidden_size=hidden_size, num_layers=num_layers, batch_first=True
        )

    def forward(self, x: Tensor) -> Tensor:  # x: [B, T, input_dim]
        out, _ = self.lstm(x)
        return out[:, -1, :]  # [B, hidden_size]
```

- [ ] **Step 3: Rename `TinyCNNGRU` -> `EncoderRecurrent` and generalize it.** Replace the entire `TinyCNNGRU` class with:

```python
class EncoderRecurrent(BaseSpeedModel):
    """Generic encoder -> per-frame features -> recurrent temporal -> head.

    The encoder may return a per-frame vector ``[B*T, feature_dim]`` (fed to an
    LSTM/GRU) or a per-frame spatial map ``[B*T, C, H, W]`` (fed to a ConvLSTM).
    ``forward`` dispatches on the number of dimensions of the encoder output.
    """

    def __init__(self, encoder: dict, temporal: dict):
        super().__init__()
        feature_dim = int(encoder.get("feature_dim", 64))
        hidden_size = int(temporal.get("hidden_size", 32))

        encoder_name = encoder.get("name", "tiny_cnn")
        if encoder_name not in ENCODER_REGISTRY:
            raise ConfigError(f"Unknown encoder {encoder_name!r}")
        temporal_name = temporal.get("name", "gru")
        if temporal_name not in TEMPORAL_REGISTRY:
            raise ConfigError(f"Unknown temporal module {temporal_name!r}")

        self.encoder = ENCODER_REGISTRY[encoder_name](
            **{k: v for k, v in encoder.items() if k != "name"}
        )
        tmp_kwargs = {k: v for k, v in temporal.items() if k != "name"}
        tmp_kwargs["input_dim"] = feature_dim
        self.temporal = TEMPORAL_REGISTRY[temporal_name](**tmp_kwargs)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, frames: Tensor) -> Tensor:  # frames: [B, T, C, H, W]
        b, t, c, h, w = frames.shape
        feats = self.encoder(frames.reshape(b * t, c, h, w))
        if feats.ndim == 2:  # vector path (LSTM/GRU)
            last = self.temporal(feats.reshape(b, t, -1))  # [B, hidden]
        else:  # spatial path (ConvLSTM)
            _, ch, fh, fw = feats.shape
            last = self.temporal(feats.reshape(b, t, ch, fh, fw))  # [B, hidden, fh, fw]
            last = last.mean(dim=(2, 3))  # [B, hidden]
        return self.head(last).squeeze(-1)  # [B]
```

- [ ] **Step 4: Update the registries** (bottom of the file):

```python
ENCODER_REGISTRY: dict[str, type[nn.Module]] = {"tiny_cnn": TinyCNNEncoder}
TEMPORAL_REGISTRY: dict[str, type[nn.Module]] = {"gru": GRUTemporal, "lstm": LSTMTemporal}
MODEL_REGISTRY: dict[str, type[BaseSpeedModel]] = {
    "tiny_cnn_gru": EncoderRecurrent,
    "cnn_lstm": EncoderRecurrent,
}
```

- [ ] **Step 5: Create `configs/model/cnn_lstm.yaml`**:

```yaml
name: cnn_lstm
encoder: {name: tiny_cnn, pretrained: false, feature_dim: 64}
temporal: {name: lstm, hidden_size: 32, num_layers: 1}
checkpoint: null
```

- [ ] **Step 6: Add the test** to `tests/unit/test_models.py`:

```python
def test_cnn_lstm_shape():
    cfg = ModelCfg(
        name="cnn_lstm",
        encoder={"name": "tiny_cnn", "pretrained": False, "feature_dim": 64},
        temporal={"name": "lstm", "hidden_size": 32, "num_layers": 1},
        checkpoint=None,
    )
    model = build_model(cfg)
    out = model(torch.randn(2, 5, 3, 32, 32))
    assert out.shape == (2,)
    assert torch.isfinite(out).all()
```

- [ ] **Step 7: Run the model tests (regression + new)**

Run: `pytest tests/unit/test_models.py tests/unit/test_training.py -q`
Expected: all pass (existing `tiny_cnn_gru` tests unchanged, `cnn_lstm` new).

- [ ] **Step 8: Verify lint + full unit suite**

Run: `ruff check src/models/architectures.py && pytest tests/unit -q`
Expected: clean; all pass.

- [ ] **Step 9: Commit**

```bash
git add src/models/architectures.py configs/model/cnn_lstm.yaml tests/unit/test_models.py
git commit -m "feat: generalize model and add cnn_lstm"
```

---

## Task 6: `ResNet18Encoder` + `resnet_lstm`

**Files:**
- Modify: `src/models/architectures.py`
- Create: `configs/model/resnet_lstm.yaml`
- Test: `tests/unit/test_models.py`

**Interfaces:**
- Produces: `ResNet18Encoder(feature_dim=512, pretrained=True, frozen=True, spatial=False, weights_path=None)`; `ENCODER_REGISTRY["resnet18"]`; `MODEL_REGISTRY["resnet_lstm"]`.

- [ ] **Step 1: Add torchvision imports** at the top of `src/models/architectures.py`:

```python
import torchvision
from torchvision.models import ResNet18_Weights
```

- [ ] **Step 2: Add `ResNet18Encoder`** (after `TinyCNNEncoder`):

```python
class ResNet18Encoder(nn.Module):
    """Frozen (or trainable) ResNet18 backbone with the classification head removed.

    ``spatial=False`` -> global-average pool to a vector ``[B, feature_dim]``.
    ``spatial=True``  -> conv stem only, 1x1-projected to ``feature_dim`` channels
                         ``[B, feature_dim, H/32, W/32]``.
    """

    def __init__(
        self,
        feature_dim: int = 512,
        pretrained: bool = True,
        frozen: bool = True,
        spatial: bool = False,
        weights_path: str | None = None,
    ):
        super().__init__()
        weights = None if weights_path is not None else (
            ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        )
        backbone = torchvision.models.resnet18(weights=weights)
        if weights_path is not None:
            backbone.load_state_dict(torch.load(weights_path, map_location="cpu"))

        if spatial:
            self.backbone = nn.Sequential(*list(backbone.children())[:-2])  # -> [B,512,H/32,W/32]
            self.proj = nn.Conv2d(512, feature_dim, 1)
        else:
            backbone.fc = nn.Identity()  # avgpool + flatten -> [B, 512]
            self.backbone = backbone
            self.proj = nn.Linear(512, feature_dim) if feature_dim != 512 else nn.Identity()

        if frozen:
            for p in self.backbone.parameters():
                p.requires_grad_(False)
            self.backbone.eval()  # BN running stats; avoids B=1 train-mode BN errors

    def forward(self, x: Tensor) -> Tensor:  # x: [B, C, H, W]
        return self.proj(self.backbone(x))
```

- [ ] **Step 3: Update the registries**:

```python
ENCODER_REGISTRY: dict[str, type[nn.Module]] = {"tiny_cnn": TinyCNNEncoder, "resnet18": ResNet18Encoder}
MODEL_REGISTRY: dict[str, type[BaseSpeedModel]] = {
    "tiny_cnn_gru": EncoderRecurrent,
    "cnn_lstm": EncoderRecurrent,
    "resnet_lstm": EncoderRecurrent,
}
```

- [ ] **Step 4: Create `configs/model/resnet_lstm.yaml`**:

```yaml
name: resnet_lstm
encoder: {name: resnet18, pretrained: true, feature_dim: 512, frozen: true, spatial: false, weights_path: null}
temporal: {name: lstm, hidden_size: 64, num_layers: 1}
checkpoint: null
```

- [ ] **Step 5: Update `test_unknown_encoder_raises`** in `tests/unit/test_models.py` (it currently uses `"resnet"`, now adjacent to `"resnet18"`):

```python
def test_unknown_encoder_raises():
    cfg = make_model_cfg()
    cfg.encoder = {"name": "definitely_not_an_encoder", "pretrained": True, "feature_dim": 64}
    with pytest.raises(ConfigError):
        build_model(cfg)
```

- [ ] **Step 6: Add the test** to `tests/unit/test_models.py`:

```python
def test_resnet_lstm_frozen_shape_and_trainable():
    cfg = ModelCfg(
        name="resnet_lstm",
        encoder={"name": "resnet18", "pretrained": False, "feature_dim": 512,
                 "frozen": True, "spatial": False},
        temporal={"name": "lstm", "hidden_size": 64, "num_layers": 1},
        checkpoint=None,
    )
    model = build_model(cfg)
    out = model(torch.randn(2, 3, 3, 128, 128))
    assert out.shape == (2,)
    assert torch.isfinite(out).all()
    # frozen backbone: zero trainable params inside the backbone
    backbone_trainable = [p for p in model.encoder.backbone.parameters() if p.requires_grad]
    assert len(backbone_trainable) == 0
    # temporal + head still trainable, but fewer trainable than total params
    assert count_parameters(model) > 0
    assert count_parameters(model) < sum(p.numel() for p in model.parameters())
```

- [ ] **Step 7: Run the model tests**

Run: `pytest tests/unit/test_models.py -q`
Expected: all pass (note: `resnet18(weights=None)` is random-init - **no network, no download**).

- [ ] **Step 8: Verify lint + full unit suite**

Run: `ruff check src/models/architectures.py && pytest tests/unit -q`
Expected: clean; all pass.

- [ ] **Step 9: Commit**

```bash
git add src/models/architectures.py configs/model/resnet_lstm.yaml tests/unit/test_models.py
git commit -m "feat: add resnet_lstm with frozen backbone"
```

---

## Task 7: `ConvLSTM` + `resnet_convlstm`

**Files:**
- Modify: `src/models/architectures.py`
- Create: `configs/model/resnet_convlstm.yaml`
- Test: `tests/unit/test_models.py`

**Interfaces:**
- Produces: `ConvLSTMCell`, `ConvLSTM(input_dim, hidden_size, kernel_size=3, num_layers=1)`; `TEMPORAL_REGISTRY["convlstm"]`; `MODEL_REGISTRY["resnet_convlstm"]`.

- [ ] **Step 1: Add `ConvLSTMCell` and `ConvLSTM`** (after `LSTMTemporal`):

```python
class ConvLSTMCell(nn.Module):
    """Single ConvLSTM cell (Shi et al. 2015): one gate convolution over [x, h]."""

    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv2d(input_dim + hidden_dim, 4 * hidden_dim, kernel_size, padding=pad)

    def forward(self, x: Tensor, state: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
        h, c = state
        i, f, g, o = self.conv(torch.cat([x, h], dim=1)).chunk(4, dim=1)
        i, f, g, o = torch.sigmoid(i), torch.sigmoid(f), torch.tanh(g), torch.sigmoid(o)
        c_next = f * c + i * g
        return o * torch.tanh(c_next), c_next


class ConvLSTM(nn.Module):
    """Spatially recurrent ConvLSTM (single layer), batch-first [B, T, C, H, W]."""

    def __init__(self, input_dim: int, hidden_size: int, kernel_size: int = 3, num_layers: int = 1):
        super().__init__()
        if num_layers != 1:
            raise ConfigError("ConvLSTM num_layers must be 1 (stacking is future work)")
        self.cell = ConvLSTMCell(input_dim, hidden_size, kernel_size)
        self.hidden_size = hidden_size

    def forward(self, x: Tensor) -> Tensor:  # [B, T, C, H, W] -> [B, hidden_size, H, W]
        b, t, _, h, w = x.shape
        h = torch.zeros(b, self.hidden_size, h, w, device=x.device)
        c = torch.zeros_like(h)
        for step in range(t):
            h, c = self.cell(x[:, step], (h, c))
        return h
```

- [ ] **Step 2: Update the registries**:

```python
TEMPORAL_REGISTRY: dict[str, type[nn.Module]] = {
    "gru": GRUTemporal,
    "lstm": LSTMTemporal,
    "convlstm": ConvLSTM,
}
MODEL_REGISTRY: dict[str, type[BaseSpeedModel]] = {
    "tiny_cnn_gru": EncoderRecurrent,
    "cnn_lstm": EncoderRecurrent,
    "resnet_lstm": EncoderRecurrent,
    "resnet_convlstm": EncoderRecurrent,
}
```

- [ ] **Step 3: Create `configs/model/resnet_convlstm.yaml`**:

```yaml
name: resnet_convlstm
encoder: {name: resnet18, pretrained: true, feature_dim: 32, frozen: true, spatial: true, weights_path: null}
temporal: {name: convlstm, hidden_size: 32, kernel_size: 3, num_layers: 1}
checkpoint: null
```

- [ ] **Step 4: Add the tests** to `tests/unit/test_models.py`:

```python
def test_convlstm_output_shape():
    from src.models.architectures import ConvLSTM

    m = ConvLSTM(input_dim=8, hidden_size=16, kernel_size=3)
    out = m(torch.randn(2, 5, 8, 4, 4))
    assert out.shape == (2, 16, 4, 4)


def test_resnet_convlstm_spatial_shape_and_frozen():
    cfg = ModelCfg(
        name="resnet_convlstm",
        encoder={"name": "resnet18", "pretrained": False, "feature_dim": 32,
                 "frozen": True, "spatial": True},
        temporal={"name": "convlstm", "hidden_size": 32, "kernel_size": 3, "num_layers": 1},
        checkpoint=None,
    )
    model = build_model(cfg)
    out = model(torch.randn(2, 3, 3, 128, 128))
    assert out.shape == (2,)
    assert torch.isfinite(out).all()
    # encoder returns a 4-D spatial map with the projected channel count
    feats = model.encoder(torch.randn(4, 3, 128, 128))
    assert feats.ndim == 4
    assert feats.shape[1] == 32
    assert len([p for p in model.encoder.backbone.parameters() if p.requires_grad]) == 0
```

- [ ] **Step 5: Run the model tests**

Run: `pytest tests/unit/test_models.py -q`
Expected: all pass.

- [ ] **Step 6: Verify lint + full unit suite**

Run: `ruff check src/models/architectures.py && pytest tests/unit -q`
Expected: clean; all pass.

- [ ] **Step 7: Commit**

```bash
git add src/models/architectures.py configs/model/resnet_convlstm.yaml tests/unit/test_models.py
git commit -m "feat: add resnet_convlstm with spatial ConvLSTM"
```

## Task 8: Model export + GPU evaluation device

**Files:**
- Modify: `src/models/training.py`
- Modify: `src/experiment.py`
- Test: `tests/unit/test_training.py`

**Interfaces:**
- Produces: `save_model_export(model, path, cfg) -> None`; `load_model_export(path) -> BaseSpeedModel`.

- [ ] **Step 1: Add `save_model_export` / `load_model_export` to `src/models/training.py`** (at the end of the file):

```python
def save_model_export(model: BaseSpeedModel, path: Path | str, cfg: ModelCfg) -> None:
    """Save a standalone, self-describing model export (no Lightning wrapper)."""
    payload = {
        "state_dict": model.state_dict(),
        "model_name": cfg.name,
        "encoder": cfg.encoder,
        "temporal": cfg.temporal,
    }
    torch.save(payload, str(path))


def load_model_export(path: Path | str) -> BaseSpeedModel:
    """Load a standalone model export produced by :func:`save_model_export`."""
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    cfg = ModelCfg(
        name=payload["model_name"], encoder=payload["encoder"], temporal=payload["temporal"]
    )
    model = build_model(cfg)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model
```

- [ ] **Step 2: Wire the export + device into `src/experiment.py`**

Add `save_model_export` to the existing `from src.models.training import (...)` import block.

After `loaded_model = load_model_from_checkpoint(best_checkpoint, cfg.model)`, add:
```python
        save_model_export(loaded_model, paths.run_dir / "model.pt", cfg.model)
```

Replace the `"cpu"` literal in the evaluation loop with `cfg.evaluation.device`:
```python
            raw_df = collect_raw_predictions(loaded_model, loaders[split_name], cfg.evaluation.device, split_name)
```

Replace the `"cpu"` literal in `evaluate_run` with `cfg.evaluation.device`:
```python
        raw = collect_raw_predictions(model, loaders[split_name], cfg.evaluation.device, split_name)
```

- [ ] **Step 3: Add the round-trip test** to `tests/unit/test_training.py`:

```python
def test_model_export_roundtrip(tmp_path: Path):
    from src.models.training import load_model_export, save_model_export

    model = build_model(make_model_cfg())
    model.eval()
    path = tmp_path / "model.pt"
    save_model_export(model, path, make_model_cfg())
    loaded = load_model_export(path)
    x = torch.randn(2, 5, 3, 32, 32)
    with torch.no_grad():
        assert torch.allclose(model(x), loaded(x))
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/unit/test_training.py -q`
Expected: all pass.

- [ ] **Step 5: Verify lint + full unit suite**

Run: `ruff check src/models/training.py src/experiment.py && pytest tests/unit -q`
Expected: clean; all pass.

- [ ] **Step 6: Commit**

```bash
git add src/models/training.py src/experiment.py tests/unit/test_training.py
git commit -m "feat: add model export and GPU evaluation device"
```

---

## Task 9: `scripts/preprocess.py` and `scripts/inspect_data.py`

**Files:**
- Create: `scripts/preprocess.py`
- Create: `scripts/inspect_data.py`

**Interfaces:**
- Consumes: `discover_segments`, `cache_segment`, `build_manifest`, `load_comma2k19_sequences` (Tasks 2-3); `make_group_split` (existing).

- [ ] **Step 1: Create `scripts/preprocess.py`**:

```python
"""Stage-1 CLI: decode comma2k19 segments into the interim cache + manifest."""

from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig

from src.core.config import validate_config
from src.core.logging_utils import get_logger, setup_logging
from src.core.schemas import DataValidationError
from src.data.comma2k19 import build_manifest, cache_segment, discover_segments

logger = get_logger(__name__)


@hydra.main(config_path="../configs", config_name="default", version_base="1.3")
def main(cfg: DictConfig) -> None:
    cfg = validate_config(cfg)
    setup_logging(None, level=cfg.logging.level)
    data = cfg.data
    cache_dir = Path(data.comma2k19.cache_dir)
    image_size = tuple(data.comma2k19.cache_image_size)

    segments = discover_segments(Path(data.root))
    if not segments:
        raise DataValidationError(f"No segments found under {Path(data.root)!r}")

    manifest: dict[str, dict] = {}
    max_segments = data.comma2k19.max_segments
    for route, seg, seg_dir in segments:
        if max_segments is not None and len(manifest) >= max_segments:
            break
        seg_id = f"{route}/{seg}"
        fname = f"{route}__{seg}.npz"
        info = cache_segment(seg_dir, cache_dir / fname, image_size)
        manifest[seg_id] = {"file": fname, "n_frames": info["n_frames"], "group_id": route}
        logger.info("event=cache_segment segment=%s frames=%d", seg_id, info["n_frames"])

    build_manifest(cache_dir, manifest)
    logger.info("event=preprocess_complete segments=%d cache_dir=%s", len(manifest), cache_dir)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Create `scripts/inspect_data.py`**:

```python
"""Inspect the comma2k19 cache: counts, speed stats, 20 Hz, split sizes."""

from __future__ import annotations

import numpy as np
import hydra
from omegaconf import DictConfig

from src.core.config import validate_config
from src.core.logging_utils import get_logger, setup_logging
from src.data.comma2k19 import load_comma2k19_sequences
from src.data.dataset import make_group_split

logger = get_logger(__name__)


@hydra.main(config_path="../configs", config_name="default", version_base="1.3")
def main(cfg: DictConfig) -> None:
    cfg = validate_config(cfg)
    setup_logging(None, level=cfg.logging.level)
    sequences = load_comma2k19_sequences(cfg.data, cfg.experiment.seed)
    logger.info("event=inspect_sequences count=%d", len(sequences))
    for seq in sequences:
        logger.info(
            "event=inspect_sequence id=%s group=%s frames=%d",
            seq.sequence_id,
            seq.group_id,
            len(seq.frames),
        )
    seq = sequences[0]
    dt = float(np.median(np.diff(seq.timestamps)))
    speeds = seq.speeds_mph
    logger.info("event=inspect_median_dt dt=%.4f expected=0.05", dt)
    logger.info(
        "event=inspect_speed min=%.2f max=%.2f mean=%.2f",
        float(speeds.min()),
        float(speeds.max()),
        float(speeds.mean()),
    )
    split = make_group_split(
        sorted({s.group_id for s in sequences}), cfg.data.split_ratios, cfg.experiment.seed
    )
    for name, groups in split.items():
        logger.info("event=inspect_split split=%s groups=%d", name, len(groups))


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Verify syntax + lint**

Run: `ruff check scripts/preprocess.py scripts/inspect_data.py`
Expected: clean.

- [ ] **Step 4: Commit**

```bash
git add scripts/preprocess.py scripts/inspect_data.py
git commit -m "feat: add preprocess and inspect scripts"
```

---

## Task 10: Real-data integration test (slow)

**Files:**
- Create: `tests/integration/test_comma2k19.py`

**Interfaces:**
- Consumes: `discover_segments`, `decode_segment_frames` (Tasks 2-3).

- [ ] **Step 1: Create `tests/integration/test_comma2k19.py`**:

```python
"""Real-data comma2k19 integration (skipped unless data/raw/Chunk_1 present)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.data.comma2k19 import MPH_PER_MPS, decode_segment_frames, discover_segments

pytestmark = pytest.mark.slow

RAW_ROOT = Path("data/raw/Chunk_1")


@pytest.mark.skipif(not RAW_ROOT.exists(), reason="comma2k19 data not present")
def test_decode_one_real_segment():
    segments = discover_segments(RAW_ROOT)
    assert segments, "no segments found"
    seg_dir = segments[0][2]
    gp = seg_dir / "global_pose"
    frame_times = np.load(gp / "frame_times.npy")
    velocities = np.load(gp / "frame_velocities.npy")

    frames = decode_segment_frames(seg_dir, (64, 64))
    assert frames.shape[0] == frame_times.shape[0]
    assert frames.shape[1:] == (64, 64, 3)
    assert frames.dtype == np.uint8
    # 20 Hz, dt = 0.05 s
    assert np.allclose(np.diff(frame_times), 0.05, atol=1e-9)
    # speeds finite, positive after m/s -> mph
    speeds_mph = np.linalg.norm(velocities, axis=1) * MPH_PER_MPS
    assert np.isfinite(speeds_mph).all()
    assert speeds_mph.min() > 0
```

- [ ] **Step 2: Run it against the real data (only if present)**

Run: `pytest tests/integration/test_comma2k19.py -q -m slow`
Expected: 1 passed if `data/raw/Chunk_1` exists, else skipped.

- [ ] **Step 3: Verify lint**

Run: `ruff check tests/integration/test_comma2k19.py`
Expected: clean.

- [ ] **Step 4: Commit**

```bash
git add tests/integration/test_comma2k19.py
git commit -m "test: add comma2k19 real-data integration test"
```

---

## Task 11: Documentation

**Files:**
- Modify: `README.md`
- Create: `docs/DECISIONS.md`

- [ ] **Step 1: Add a "comma2k19 baselines" section to `README.md`** (after the existing "Commands" section):

```markdown
## comma2k19 baselines (real data)

Three models are available over the comma2k19 dataset:

| Model | Encoder | Temporal | `preprocessing.image_size` |
| --- | --- | --- | --- |
| `cnn_lstm` | tiny from-scratch CNN | LSTM | `[64, 64]` |
| `resnet_lstm` | frozen pretrained ResNet18 (global pool) | LSTM | `[128, 128]` |
| `resnet_convlstm` | frozen pretrained ResNet18 (spatial map) | ConvLSTM | `[128, 128]` |

Workflow:

```bash
# 1. Decode + cache raw data (HEVC -> 128x128 RGB .npz + manifest). ~13 GiB for Chunk_1.
python scripts/preprocess.py dataset=comma2k19
# pilot: python scripts/preprocess.py dataset=comma2k19 data.comma2k19.max_segments=10

# 2. Inspect the cache (counts, 20 Hz, speed stats, split sizes).
python scripts/inspect_data.py dataset=comma2k19

# 3. Train a baseline (add data.window_stride=1 for full overlap; default is 5).
python scripts/train.py dataset=comma2k19 model=cnn_lstm preprocessing.image_size=[64,64]
python scripts/train.py dataset=comma2k19 model=resnet_lstm preprocessing.image_size=[128,128]
python scripts/train.py dataset=comma2k19 model=resnet_convlstm preprocessing.image_size=[128,128]
```

Notes:

- Cache lives in `data/interim/comma2k19/` (uncompressed `.npz`, memory-mapped at
  load). Change `data.comma2k19.cache_image_size` to re-cache at another resolution.
- `data.window_stride` applies to training only; validation/test are always
  stride 1 (required by the frame-aligned metrics and stationary detector).
- ResNet weights download once (ImageNet) and are cached by torchvision; point
  `encoder.weights_path` at a local `.pth` to run offline.
- Each run exports a standalone `model.pt` (no Lightning wrapper) alongside the
  Lightning checkpoints.
- Evaluation/inference device is `evaluation.device` (default `auto` -> GPU when
  available).
```

- [ ] **Step 2: Create `docs/DECISIONS.md`** with the recorded decisions:

```markdown
# Decisions log

## Open decisions (PLAN.md §2.1, assumed defaults for M1)

| ID | Decision | Source | Date |
| --- | --- | --- | --- |
| D1 | Labels in mph; comma2k19 `global_pose/frame_velocities` | assumed | 2026-10-14 |
| D2 | 20 Hz constant nominal frame rate | measured | 2026-10-14 |
| D3 | Monocular scale ambiguity ignored | assumed | 2026-10-14 |
| D4 | Inference is strictly causal (no future frames) | required | 2026-10-14 |
| D5 | Runtime reported, no pass/fail threshold | assumed | 2026-10-14 |
| D6 | "Creeping <= 2 mph" = label `<= 2.0` | assumed | 2026-10-14 |
| D7 | Detector initial state MOVING | assumed | 2026-10-14 |
| D8 | Structured logging = stdlib key=value | assumed | 2026-10-14 |

## Ingestion (M2-a, from 2026-10-13 design)

| ID | Decision |
| --- | --- |
| D-M2-1 | m/s -> mph at ingestion (x 2.2369362921) |
| D-M2-2 | Label = `frame_velocities` magnitude (frame-aligned) |
| D-M2-3 | RGB everywhere; OpenCV BGR -> RGB at decode/cache |
| D-M2-4 | `frame_rate_hz: 20`; count from `len(frame_times)` |
| D-M2-5 | `sequence_id = <route>/<seg>`; `group_id = <route>` |
| D-M2-6 | Split on `group_id` (route), no leakage |
| D-M2-7 | OpenCV lazy-imported; `video` extra |
| D-M2-8 | Two-stage cache: decode -> `.npz` + manifest |
| D-M2-9 | Cache resolution default 128x128 (configurable) |

## Baselines (2026-10-14 design)

| ID | Decision |
| --- | --- |
| D-B-1 | Three baselines: `cnn_lstm`, `resnet_lstm`, `resnet_convlstm` |
| D-B-2 | Cache 128x128; CNN consumes 64, ResNet consumes 128 |
| D-B-3 | ResNet backbone frozen (`requires_grad=False` + `eval()`) |
| D-B-4 | ResNet vector mode global-pools to 512-d; spatial mode 1x1-projects to 32 ch |
| D-B-5 | ConvLSTM single-layer, "very small" (hidden 16-32 ch) |
| D-B-6 | Per-split stride: train `window_stride` (default 5), val/test 1 |
| D-B-7 | `weights_path` offline reuse; `model.pt` standalone export per run |
| D-B-8 | `EncoderRecurrent` generalizes `TinyCNNGRU`; `tiny_cnn_gru` stays registered |
| D-B-9 | `evaluation.device` (default `auto` -> GPU) |
```

- [ ] **Step 3: Verify the docs render (no lint on markdown; just eyeball)**

Run: `git diff --stat`
Expected: only `README.md` + `docs/DECISIONS.md` changed.

- [ ] **Step 4: Commit**

```bash
git add README.md docs/DECISIONS.md
git commit -m "docs: document comma2k19 baselines and decisions"
```

---

## Final verification (after all tasks)

```bash
ruff check src tests scripts
pytest tests/unit -q                 # full offline unit suite
pytest tests/integration -q -m slow  # real-data test (skips if data absent)
```

Expected: ruff clean; unit suite green; slow test green or skipped.



