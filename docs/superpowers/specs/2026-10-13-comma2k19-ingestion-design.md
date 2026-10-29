# Design: comma2k19 data-ingestion layer (M2-a)

**Date:** 2026-10-13
**Status:** approved for implementation
**Scope:** add a real-data ingestion layer for the comma2k19 dataset to the
existing speed-estimation pipeline. This is "M2-a" from `docs/PLAN.md`.
**Explicitly out of scope:** model changes, baselines, training on real data,
stationary-algorithm study, runtime benchmarking (all deferred to M2-b…M2-f).

---

## 1. Context

The repo is a complete, tested M1 vertical slice: a CNN+GRU regresses per-frame
speed (mph) from a sliding window of frames, with a causal post-processing stage.
It currently reads **synthetic** data only, via `generate_synthetic_sequences`.

The data layer already has clean seams we reuse unchanged:

- `FrameSource` protocol (`__len__` + `get(indices) -> uint8 [k,H,W,C]`),
- `SpeedSequence` dataclass (`sequence_id`, `group_id`, `frames`, `speeds_mph`,
  `timestamps`),
- `validate_sequence` (gap-split, duplicates, negatives, rate warnings),
- `make_group_split` / `assert_no_leakage` (deterministic **group** split),
- `SpeedDataset` / `build_datasets` / `build_dataloaders` (windows, transforms).

This design adds **only** the real-data source and a caching/preprocessing pass.
Nothing downstream (splits, windows, model, post-processing, evaluation) changes.

## 2. Verified facts about the data

Inspected live against `data/raw/Chunk_1` (RAV4, dongle `b0c9d2329ad1606b`):

| Item | Value |
| --- | --- |
| Layout | `Chunk_1/<route>/<segment>/` with `video.hevc`, `preview.png`, `raw_log.bz2`, `processed_log/`, `global_pose/` |
| Route dir naming | `b0c9d2329ad1606b_2018-08-02--08-34-47` — `\|` became `_` (NTFS workaround) |
| Segment count / route | ~12 segments (1 minute each) |
| `global_pose/` files | `frame_times`, `frame_velocities`, `frame_positions`, `frame_orientations`, `frame_gps_times` |
| `frame_times` | `(1200,)` float64, boot-time seconds, **dt = 0.05 s exactly (20 Hz)** |
| `frame_velocities` | `(1200, 3)` float64, ECEF velocity m/s; speed = ‖v‖ |
| Speed range (chunk 1) | ~7.9–20.0 m/s ≈ **17.8–44.8 mph** (highway) |
| `video.hevc` | decodes **all 1200 frames** via OpenCV 5.0; `(874,1164,3)` uint8 **BGR**; 1.45 s per segment |
| OpenCV metadata | `fps=25`, `frame_count` garbage → **never trust it** |
| `processed_log/CAN/` | `radar`, `raw_can`, `speed`, `steering_angle`, `wheel_speed` |

### 2.1 Label provenance (`frame_velocities`)

`global_pose` is the camera pose in ECEF computed by comma.ai's **tightly coupled
INS/GNSS/Vision optimizer** using **Laika** (post-processed GNSS). `frame_velocities`
is the velocity component, aligned 1:1 to video frames. **Caveat:** because the
optimizer fuses **Vision** (the camera itself), the label is not strictly
independent of the input frames — a mild circularity for speed-from-video. On
open highway GNSS/INS dominate; record this in `docs/DECISIONS.md`.

### 2.2 File inventory

Full tree (verified locally on `Chunk_1`, RAV4):

```text
data/raw/Chunk_1/
└── b0c9d2329ad1606b_<date>--<time>/        # route: "|" became "_"
    └── <seg>/                               # 1-minute segment (0–N)
        ├── video.hevc        # H.265, 20 fps, 1200 frames, ~37 MB, BGR
        ├── preview.png       # full-res first frame, ~497 KB
        ├── raw_log.bz2       # raw capnp CAN log, ~6.9 MB (not used here)
        ├── processed_log/
        │   ├── CAN/    {radar, raw_can, speed, steering_angle, wheel_speed}
        │   ├── GNSS/   {live_gnss_qcom, live_gnss_ublox, raw_gnss_qcom, raw_gnss_ublox}
        │   └── IMU/    {acceleration, gyro, gyro_bias, gyro_uncalibrated, magnetic, magnetic_uncalibrated}
        └── global_pose/
            ├── frame_times         (1200,)   float64  boot seconds, dt = 0.05 s
            ├── frame_velocities    (1200, 3) float64  ECEF velocity (m/s)
            ├── frame_positions     (1200, 3) float64  ECEF position (m)
            ├── frame_orientations  (1200, 4) float64  quaternion (Hamilton)
            └── frame_gps_times     (1200, 2) float64  [gps week, time-of-week]
```

Notes:

- Every `global_pose` entry is a `.npy` file (extensions omitted above for
  brevity). Shapes/dtypes above were verified directly; `video.hevc` decodes all
  1200 frames via OpenCV 5.0 as `(874,1164,3)` uint8 BGR.
- **Verified locally:** `frame_times/velocities/positions/orientations/gps_times`
  shapes and dtypes; the `processed_log/CAN/` listing; full video decode.
  **From the dataset README (not read by this work):** the GNSS/IMU contents.
- **Naming discrepancies vs the README:** the README calls CAN speed `car_speed`
  and wheel speeds `wheel_speeds`, but the actual files are `speed` and
  `wheel_speed`; the README's `global_pos/` is actually `global_pose/`.
- Only `video.hevc` + `frame_times` + `frame_velocities` are consumed in M2-a.
  Everything else is unused and kept for future phases (CAN baselines, etc.).

## 3. Decisions

| ID | Decision |
| --- | --- |
| D-M2-1 | **Units:** convert m/s → mph at ingestion (`× 2.23694`); the pipeline stays mph end-to-end. |
| D-M2-2 | **Label:** `global_pose/frame_velocities` magnitude only (already frame-aligned). No CAN path in M2-a. |
| D-M2-3 | **Channel order:** RGB everywhere. OpenCV decodes BGR, so decode/cache converts `BGR→RGB` (matches the reference `pix_fmt='rgb24'` and the ImageNet `mean/std` in preprocessing). |
| D-M2-4 | **Frame rate:** `frame_rate_hz: 20` (measured dt = 0.05 s). Frame count comes from `len(frame_times)`, never OpenCV metadata. |
| D-M2-5 | **Sequence/group:** one `SpeedSequence` = one segment; `group_id` = route dir name (opaque unique key); `sequence_id` = `<route>/<segment>`. |
| D-M2-6 | **Split policy:** split on `group_id` (route), never on frames/windows. Reuse `make_group_split` + `assert_no_leakage`. |
| D-M2-7 | **OpenCV:** imported lazily inside `src/data/comma2k19.py`; remains in the `video` extra; clear error if missing. |
| D-M2-8 | **Caching:** two-stage. Preprocess decodes once into a downscaled per-segment `.npz` + manifest; dataset build reads the cache. |
| D-M2-9 | **Cache resolution:** default `128×128`, configurable (`cache_image_size`). |

## 4. Architecture

Two-stage flow:

```
raw (video.hevc + global_pose/*.npy)
      │  scripts/preprocess.py  (decode once, BGR→RGB, downscale)
      ▼
interim cache (data/interim/comma2k19/<route>__<seg>.npz + manifest.json)
      │  load_comma2k19_sequences  (read cache → SpeedSequence)
      ▼
validate_sequence → make_group_split → windows → SpeedDataset  (existing, unchanged)
```

### 4.1 Cache format

Per segment: `data/interim/comma2k19/<route>__<seg>.npz` with keys

- `frames` — `uint8 [N, H', W', 3]` RGB, downscaled to `cache_image_size`
- `speeds_mph` — `float64 [N]`
- `timestamps` — `float64 [N]`

`manifest.json` maps `segment_id -> {file, n_frames, group_id}`; generated
deterministically during preprocessing so dataset build never re-walks the raw
tree and reuses the cache across runs.

### 4.2 Frame sources

- `CachedFrameSource` (default): opens the cached `.npz` with
  `np.load(path, mmap_mode="r")` — **lazy, memory-mapped** — and serves `get()`
  by indexing the requested frames. This is essential: `build_datasets` holds
  ~250 `SpeedSequence`s at once, and eagerly loading every segment's frames
  would use ~14 GB of RAM. Memory-mapping pages in only the windows actually
  read. Implements the existing `FrameSource` protocol.
- `VideoFrameSource` (fallback when `use_cache: false`): lazy cv2 decode of
  `video.hevc` with `BGR→RGB`; `__len__` from `frame_times`.

Both return `uint8 [k,H,W,C]` RGB, so downstream transforms are unchanged.

### 4.3 `build_datasets` dispatch

`build_datasets` branches on `cfg.data.source`:

- `synthetic` → `generate_synthetic_sequences` (unchanged)
- `comma2k19` → `load_comma2k19_sequences`

Everything after "produce `list[SpeedSequence]`" is shared.

## 5. Components

| File | Purpose |
| --- | --- |
| `configs/dataset/comma2k19.yaml` | dataset group (`@package data`) — see §6 |
| `src/data/comma2k19.py` | `decode_segment_frames`, `cache_segment`, `load_comma2k19_sequences`, `CachedFrameSource`, `VideoFrameSource` |
| `src/data/dataset.py` | add `build_datasets` dispatch (edit only the sequence-source line) |
| `src/core/config.py` | add `Comma2k19Cfg` dataclass + `DataCfg.comma2k19` field (mirror `SyntheticCfg`) |
| `scripts/preprocess.py` | Stage-1 CLI: decode → cache → manifest |
| `scripts/inspect_data.py` | verify counts/shapes/speed stats, 20 Hz, gap-splits, group disjointness |
| `tests/unit/test_comma2k19.py` | fixture tests (offline) |
| `tests/integration/test_comma2k19.py` | `@pytest.mark.slow` real-data test (skips if data absent) |
| `README.md` | config + caching documentation (§8) |
| `docs/DECISIONS.md` | create; record D1–D8 (from PLAN §2.1) + D-M2-1…D-M2-9 + caveats |

## 6. Config schema (`configs/dataset/comma2k19.yaml`)

```yaml
# @package data
source: comma2k19
root: data/raw/Chunk_1
frame_rate_hz: 20
speed_source: global_pose          # only global_pose in M2-a
use_cache: true
cache_dir: data/interim/comma2k19
cache_image_size: [128, 128]       # [H, W] of cached frames
max_segments: null                 # null = all; int = limit for smoke runs
split_manifest: data/splits/comma2k19_seed${experiment.seed}.json
split_ratios: {train: 0.7, val: 0.15, test: 0.15}
sequence_length: 10
window_stride: 1
frame_step: 1
max_gap_factor: 1.5
on_duplicate_timestamp: raise
allow_negative_speed: false
max_speed_mph: 200
```

`Comma2k19Cfg` holds the new keys: `speed_source`, `use_cache`, `cache_dir`,
`cache_image_size`, `max_segments`.

## 7. Splitting policy (explicit)

- Split on **`group_id` = route** via `make_group_split` (deterministic shuffle of
  sorted unique groups, partitioned by `split_ratios`).
- `assert_no_leakage` guarantees no route appears in two splits → test = **entire
  unseen drives**, no window shares frames with a neighbor across a split.
- **Rationale:** random/adjacent-frame splitting is optimistic because neighboring
  frames are highly correlated; splitting by recording session prevents leakage.
- **Documented limitation:** comma2k19 is one vehicle per chunk on the same
  stretch of I-280, so route-level splits still share camera/scene/driver. This
  is the strongest split available and is correct for this dataset; record it in
  `docs/DECISIONS.md` as a generalization caveat.

## 8. Documentation (required)

1. **Configuration** (`README.md` + this doc): how to select `dataset=comma2k19`,
   every new key, and Hydra override examples:

   ```bash
   python scripts/preprocess.py dataset=comma2k19 data.max_segments=10
   python scripts/preprocess.py dataset=comma2k19 data.cache_image_size=[64,64]
   python scripts/train.py     dataset=comma2k19 data.use_cache=false
   ```

2. **Caching** (`README.md` + this doc): what it is, where
   (`data/interim/comma2k19/`), how to regenerate (`scripts/preprocess.py`),
   disk footprint (`128² ≈ 13 GiB` for all of chunk 1; `64² ≈ 3.5 GiB`;
   `32² ≈ 0.9 GiB`), and how to change resolution / limit segments / disable
   cache.

## 9. Error handling

- Missing `root` or zero segments → `DataValidationError`.
- Missing `global_pose/frame_times` or `frame_velocities` → `DataValidationError`.
- Decoded frame count ≠ `len(frame_times)` in the cache step → raise (fail loudly).
- OpenCV missing and needed (preprocess, or `use_cache: false`) → clear error
  pointing to `pip install -e ".[video]"`.
- Reuse existing policies for gaps/duplicates/negatives/speed-out-of-range
  unchanged.

## 10. Testing

- **Offline fixture tests** (`tests/unit/test_comma2k19.py`): tiny fake segment
  tree (small `.npy` + a cached `.npz`); assert mph conversion, RGB order,
  group/sequence id construction, speed magnitude, 20 Hz, gap-split reuse,
  split-by-route leakage, `max_segments` limit.
- **Real-data test** (`tests/integration/test_comma2k19.py`, `@pytest.mark.slow`,
  skipped when `data/raw/Chunk_1` absent): one segment decodes 1200 frames,
  speeds finite, mph conversion, dt = 0.05 s.

## 11. Outputs at each stage (worked examples)

### Stage 0 — raw (input, not produced by us)

The segment tree shown in §2.2.

### Stage 1 — cache (`scripts/preprocess.py`)

Per segment: `data/interim/comma2k19/b0c9d2329ad1606b_2018-08-02--08-34-47__40.npz`

```python
npz.files == ["frames", "speeds_mph", "timestamps"]
npz["frames"].shape    # (1200, 128, 128, 3) uint8, RGB
npz["speeds_mph"]      # float64 (1200,)  e.g. [17.76, 17.93, 18.09, ...]
npz["timestamps"]      # float64 (1200,)  e.g. [46408.547, 46408.597, ...]
```

Manifest: `data/interim/comma2k19/manifest.json`

```json
{
  "cache_image_size": [128, 128],
  "segments": {
    "b0c9d2329ad1606b_2018-08-02--08-34-47/40": {
      "file": "b0c9d2329ad1606b_2018-08-02--08-34-47__40.npz",
      "n_frames": 1200,
      "group_id": "b0c9d2329ad1606b_2018-08-02--08-34-47"
    }
  }
}
```

### Stage 2 — `SpeedSequence` (`load_comma2k19_sequences`)

```python
SpeedSequence(
    sequence_id="b0c9d2329ad1606b_2018-08-02--08-34-47/40",
    group_id="b0c9d2329ad1606b_2018-08-02--08-34-47",
    frames=CachedFrameSource(path=..., n_frames=1200),   # FrameSource, lazy
    speeds_mph=array([17.76, 17.93, 18.09, ...]),         # (1200,) float64
    timestamps=array([46408.547, 46408.597, ...]),        # (1200,) float64
)
```

### Stage 3 — validation + split (existing code)

Split manifest `data/splits/comma2k19_seed42.json`, keyed by **route/group**:

```json
{
  "seed": 42,
  "ratios": {"train": 0.7, "val": 0.15, "test": 0.15},
  "splits": {
    "train": ["b0c9d2329ad1606b_2018-07-27--06-03-57", "..."],
    "val":   ["b0c9d2329ad1606b_2018-07-29--11-17-20", "..."],
    "test":  ["b0c9d2329ad1606b_2018-08-14--10-32-01", "..."]
  }
}
```

(Gap-splitting would emit `<id>#<k>` children; rare on continuous highway data.)

### Stage 4 — dataset window / batch (existing `SpeedDataset`)

One `__getitem__` sample — a window of `T=10` frames ending at `frame_index`:

```python
{
    "frames":      torch.float32 [10, 3, 32, 32],   # Resize→ToTensor→Normalize
    "target_mph":  torch.float32 scalar,            # speed at the window's LAST frame
    "timestamp":   torch.float64 scalar,
    "frame_index": torch.int64 scalar,              # e.g. 9
    "sequence_id": "b0c9d2329ad1606b_2018-08-02--08-34-47/40",
}
```

After `build_dataloaders` (batching): `frames` → `[B, 10, 3, 32, 32]`, the rest → `[B]`.

### Stage 5 — downstream (existing, **not** produced by this work)

`predictions.csv` columns are unchanged; shown for context only:

```csv
split,sequence_id,frame_index,timestamp,true_speed_mph,raw_speed_mph,final_speed_mph,stationary_pred,inference_ms
test,b0c9d2329ad1606b_2018-08-02--08-34-47/40,9,46408.547,17.93,17.6,17.6,False,
```

## 12. Non-goals (deferred to M2-b…M2-f)

Real-data training, deployable baselines, CNN+LSTM, stationary study, runtime
benchmark, and the accuracy implication of the "high" bucket being nearly empty
on highway-only chunk 1 (document, do not solve).
