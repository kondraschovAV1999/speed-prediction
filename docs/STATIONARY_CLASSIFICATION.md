# Stationary vs. Moving Classification

This document describes how the codebase decides whether the ego vehicle is
**stationary** or **moving**. It is derived from the implementation, not from
intent, and cites exact source locations so it stays verifiable.

## TL;DR

There are **two distinct notions** of "stationary" in this codebase, and they are
deliberately not the same function:

| Notion | Where it lives | Purpose | Rule |
|--------|----------------|---------|------|
| **Predicted stationary** (`stationary_pred`) | `src/inference/postprocessing.py` → `StationaryDetector` | Turn the model's noisy per-frame speed into a stable STOPPED/MOVING state (an independent boolean output) | Hysteresis state machine (thresholds + consecutive-frame persistence) |
| **True stationary** (ground-truth label) | `src/evaluation/metrics.py` → `stationary_metrics` | Score how well `stationary_pred` matches reality | `true_speed_mph <= stationary_label_threshold_mph` |

The first is a **post-processing algorithm** applied to predictions; the second
is a **labeling rule** applied only to ground-truth speed for evaluation.

---

## 1. The stationary detector (prediction side)

File: `src/inference/postprocessing.py`, class `StationaryDetector` (lines 40–85).

### 1.1 States

A two-state finite state machine with hysteresis (a Schmitt-trigger-style
detector):

- `MOVING` (`"moving"`)
- `STOPPED` (`"stopped"`)

`stationary == True` exactly when the state is `STOPPED`
(`stationary` property, lines 63–65).

### 1.2 Transition rules (EARS)

| State | Condition | Effect |
|-------|-----------|--------|
| MOVING → STOPPED | `speed_mph <= enter_threshold_mph` for **`enter_persistence_frames` consecutive frames** | state becomes `STOPPED`, exit counter reset |
| STOPPED → MOVING | `speed_mph > exit_threshold_mph` for **`exit_persistence_frames` consecutive frames** | state becomes `MOVING`, enter counter reset |

Implementation in `update()` (lines 67–85):

```python
if self._state == self.MOVING:
    if speed_mph <= self._cfg.enter_threshold_mph:
        self._enter_count += 1
    else:
        self._enter_count = 0
    if self._enter_count >= self._cfg.enter_persistence_frames:
        self._state = self.STOPPED
        self._exit_count = 0
else:  # STOPPED
    if speed_mph > self._cfg.exit_threshold_mph:
        self._exit_count += 1
    else:
        self._exit_count = 0
    if self._exit_count >= self._cfg.exit_persistence_frames:
        self._state = self.MOVING
        self._enter_count = 0
```

### 1.3 Boundary semantics (inclusive vs. strict)

- **Enter is inclusive:** `<= enter_threshold_mph` counts toward stopping.
  A speed exactly equal to the enter threshold advances the STOPPED counter.
- **Exit is strict:** `> exit_threshold_mph` counts toward moving.
  A speed exactly equal to the exit threshold **does not** advance the MOVING
  counter — the vehicle stays STOPPED.

This asymmetry is what creates hysteresis: with `enter < exit`, the vehicle
must fall below `enter` to stop and rise *above* `exit` to move again, so the
detector does not flap when speed hovers near a single threshold.

Verified by `tests/unit/test_postprocessing.py`:

- `test_vector_enter_boundary_counts_equal_as_low` — `2.0` is treated as "low" (enters).
- `test_vector_exit_boundary_strict` — `2.5` is **not** `> 2.5`, so it stays stopped;
  `2.6` exits after `M` frames.

### 1.4 Counter reset behavior

A single non-qualifying frame resets the corresponding counter to zero
(not decrements it). Example from `test_vector_counter_reset`:

```
speed:  [5, 1.9, 5, 1.9, 1.9, 1.9]   (enter=2.0, N=3)
state:  [M,  M,   M,  M,   M,   S]
```

The `5` at index 2 resets the enter counter, so three consecutive `1.9`s are
required *again* before stopping.

### 1.5 Initial state and reset

- The detector starts in `initial_state` (default `"moving"`).
- `reset()` (lines 57–61) restores `initial_state` and clears both counters.
- `reset()` is called at the start of every sequence, so state never leaks
  between sequences. `test_postprocess_dataframe_resets_between_sequences`
  verifies a sequence that ends STOPPED is followed by a sequence that starts MOVING.

---

## 2. Where the detector sits in the pipeline

File: `src/inference/postprocessing.py`, `SpeedPostProcessor` (lines 88–170).

Per frame, the post-processing order is:

```text
raw_speed_mph → clip (min/max) → smoothing (none | ema | median) → stationary detector → final_speed_mph
```

```python
value = self._clip(raw)       # clip enabled: min 0.0, optional max
value = self._smooth(value)   # causal EMA or trailing median
stationary = self._detector.update(value)   # detector sees the SMOOTHED value
final = value  # never zeroed by the stationary detector
```

Two consequences worth noting:

1. **The detector operates on the post-smoothing value, not the raw model
   output.** If `smoothing.method` is `ema` or `median`, the speed compared
   against `enter_threshold_mph`/`exit_threshold_mph` is the smoothed speed.
   (The default config uses `median` smoothing.)
2. **`final_speed_mph` is the clipped/smoothed value and is never zeroed by the
   detector.** The `stationary` flag is an independent boolean output. Negative
   zero is still normalized (`if final == 0.0: final = 0.0`, lines 122–123),
   and the CSV writer re-normalizes as a backstop
   (`src/evaluation/reporting.py`, lines 56–58).

### 2.1 Validation and error handling

- `SpeedPostProcessor.step` raises `ValueError` on NaN or infinite input
  (lines 110–112).
- `StationaryDetector.__init__` raises `ValueError` if
  `enter_threshold_mph > exit_threshold_mph` or either persistence frame count
  is `< 1` (lines 46–53). The same checks exist at config-validation time in
  `src/core/config.py` `_validate_semantics` (lines 251–260).

### 2.2 Detector disabled

If `postprocessing.stationary.enabled` is `false`, no detector is created, a
warning is logged (`event=stationary_disabled note=stationary_metrics_not_meaningful`),
and every frame is reported as moving (`stationary_pred = False`, `final_speed_mph`
kept as the clipped/smoothed value). See `SpeedPostProcessor.__init__`
(lines 91–97) and `step` (lines 117–121).

---

## 3. Batch and streaming entry points

Both paths funnel through the same `SpeedPostProcessor`:

- **Batch / offline:** `postprocess_dataframe(df, cfg)`
  (`src/inference/postprocessing.py`, lines 173–205). Groups by `sequence_id`,
  sorts by `frame_index`, resets the processor between sequences, and requires
  stride-1 consecutive `frame_index` within each sequence (raises `ValueError`
  otherwise). Adds `final_speed_mph` and `stationary_pred` columns.
- **Streaming:** `StreamingSpeedEstimator.update` calls `self.postprocessor.step(raw)`
  (`src/inference/pipeline.py`, line 142). `reset()` / gap handling mirror the
  offline behavior.

---

## 4. Ground-truth labeling (evaluation side)

File: `src/evaluation/metrics.py`, `stationary_metrics` (lines 36–62).

The positive class is **stationary**, defined as:

```python
true_stationary = true_speed_mph <= label_threshold
```

`label_threshold` = `evaluation.stationary_label_threshold_mph` (default `2.0`).
Note this is a separate knob from the detector's `enter_threshold_mph` — it is
used **only** to label ground truth, never to alter predictions.

The function computes a confusion matrix over the pair
(`true_stationary`, `pred_stationary`) and reports:

- `precision`, `recall`, `f1` (zero-division → `0.0`)
- raw counts `tp`, `fp`, `fn`, `tn`
- `n_true_stationary` (how many frames are actually stationary)

Boundary: a true speed exactly equal to the threshold is **stationary**
(`<=`), verified by `test_stationary_truth_exactly_threshold_is_stationary`.

---

## 5. Configuration

There are three sources that define the numeric behavior. **They do not all
agree** (see §6).

### 5.1 Dataclass defaults

`src/core/config.py`, `StationaryCfg` (lines 131–138):

| Field | Default |
|-------|---------|
| `enabled` | `True` |
| `enter_threshold_mph` | `2.0` |
| `exit_threshold_mph` | `2.5` |
| `enter_persistence_frames` | `3` |
| `exit_persistence_frames` | `3` |
| `initial_state` | `"moving"` |

### 5.2 `configs/default.yaml` (what actually runs)

Lines 58–70:

| Field | Value |
|-------|-------|
| `enter_threshold_mph` | `2` |
| `exit_threshold_mph` | `3` |
| `enter_persistence_frames` | `2` |
| `exit_persistence_frames` | `4` |
| `initial_state` | `moving` |
| `evaluation.stationary_label_threshold_mph` | `2.0` |

The YAML is authoritative for real runs (Hydra composes `default.yaml`, and
`validate_config` merges it over the dataclass schema).

---

## 6. Design trade-offs and flagged discrepancies

### 6.1 Persistence delays the stationary flag by `N-1` frames

Because entering STOPPED requires `N` consecutive qualifying frames, the
`stationary_pred` flag lags the true stop by up to `N-1` frames. This lowers
stationary **recall** around transitions. The same lag applies symmetrically on
exit. This is a known, accepted trade-off (noted in `docs/PLAN.md` §6.1), and is
exactly why the detector study milestone (M2-d) exists.

### 6.2 Default YAML disagrees with the dataclass defaults

`configs/default.yaml` uses `exit_threshold_mph: 3` and persistence `2/4`,
whereas `StationaryCfg` dataclass defaults and the reference test vectors use
`exit_threshold_mph: 2.5` and persistence `3/3`. If someone constructs
`StationaryCfg()` / `PostprocessingCfg()` directly (as the unit tests do), they
get **2.5 / 3 / 3**, not the YAML's **3 / 2 / 4**. Anything reading config only
through Hydra + `validate_config` is unaffected.

### 6.3 Hysteresis creates a "labeled moving but predicted stationary" band

With the default YAML (`enter=2`, `exit=3`) and label threshold `2.0`, a speed
in the open interval `(2.0, 3.0]` mph is:

- **ground-truth "moving"** (`true_speed > 2.0`), but
- **predicted "stationary"** once the detector is already STOPPED (exit requires
  `> 3.0`), so `stationary_pred = True`.

This band is an intentional consequence of hysteresis (it prevents flapping) but
is a structural source of stationary false positives against the `<= 2.0` label.
Any report of stationary precision should be read with this band in mind.

### 6.4 Smoothing interacts with thresholds

The detector sees the **smoothed** speed. With `median` smoothing (the default),
a brief spike that would otherwise trip `exit_threshold_mph` may be filtered out
before the detector sees it, effectively widening hysteresis. The thresholds are
interpreted as "of the smoothed signal", not the raw model output.

---

## 7. Artifacts produced

- **Predictions:** `final_speed_mph` and `stationary_pred` columns are added to
  `predictions.csv` (`postprocess_dataframe`; canonical column order in
  `src/inference/postprocessing.py` lines 27–37 and `src/evaluation/reporting.py`
  lines 14–24).
- **Metrics:** `metrics.json` → `splits.<split>.stationary`
  (`{precision, recall, f1, tp, fp, fn, tn, n_true_stationary}`), via
  `src/evaluation/reporting.py` → `write_metrics_json`.
- **Figure:** `stationary_confusion_matrix.png`, drawn by
  `plot_stationary_confusion` in `src/evaluation/visualization.py` (lines 133–147).
  Axes are `true` (rows) vs `predicted` (cols) over labels `{moving, stationary}`,
  with `[[tn, fp], [fn, tp]]`.

---

## 8. Quick reference: how to read a frame's state

1. Take the model output `raw_speed_mph`.
2. Clip (default min `0.0`) and smooth (default median window 5).
3. Feed the smoothed value to the hysteresis machine:
   - MOVING → STOPPED after `enter_persistence_frames` frames `<= enter_threshold_mph`.
   - STOPPED → MOVING after `exit_persistence_frames` frames `> exit_threshold_mph`.
4. Emit `stationary_pred = (state == STOPPED)` and
   `final_speed_mph = smoothed_value` (independent of the flag).
5. At evaluation, compare `stationary_pred` against
   `true_stationary = (true_speed_mph <= stationary_label_threshold_mph)`.
