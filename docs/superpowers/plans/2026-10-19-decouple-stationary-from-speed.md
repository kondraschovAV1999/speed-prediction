# Decouple Stationary from Speed Output — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `final_speed_mph` the pure post-processed speed prediction (clip → smoothing, never zeroed), and keep the hysteresis detector as an independent `stationary_pred` boolean output.

**Architecture:** One-line behavioral change in `SpeedPostProcessor.step` (`src/inference/postprocessing.py`): the `stationary` flag is still computed and returned, but it no longer overwrites the speed value. Everything downstream (evaluation, reporting, figures) already treats the two as separate signals, so no structural change is needed — only the zeroing is removed and the now-stale docs/tests are updated.

**Tech Stack:** Python 3.10+, PyTorch + Lightning (model/training only), pandas/numpy (post-processing + eval), pytest, ruff. `postprocessing.py` must stay torch-free.

## Global Constraints

- `src/inference/postprocessing.py` **must not** import torch (WP5 rule).
- Tests run offline on CPU; no pretrained weights and no network access.
- `ruff check src tests scripts` must stay clean.
- `pytest -q` (unit) < 60 s; `pytest -q -m integration` < 120 s.
- Predictions-table column order is frozen (§5.4): `split, sequence_id, frame_index, timestamp, true_speed_mph, raw_speed_mph, final_speed_mph, stationary_pred, inference_ms`. Do not reorder.
- Commit style matches repo history: lowercase `prefix-short-description` (e.g. `feat-`, `test-`, `docs-`), no scope parens.

---

## Task 1: Stop zeroing `final_speed_mph` (code + unit tests)

**Files:**
- Modify: `src/inference/postprocessing.py` (module docstring + `step()`)
- Modify: `src/core/schemas.py` (comment on `SpeedPrediction.final_speed_mph`)
- Test: `tests/unit/test_postprocessing.py`

**Interfaces:**
- Consumes: `StationaryCfg`, `PostprocessingCfg`, `PostProcessResult` (existing).
- Produces: `SpeedPostProcessor.step(raw_mph: float) -> PostProcessResult` where `final_speed_mph` is the clipped/smoothed value (never zeroed) and `stationary` is the hysteresis state. Signatures unchanged — only the value of `final_speed_mph` changes.

- [ ] **Step 1: Update the unit tests to assert the new behavior (TDD — they fail against current code)**

In `tests/unit/test_postprocessing.py`:

1. `test_vector_basic_enter` (currently asserts `finals == [5.0, 1.9, 1.9, 0.0, 0.0]`). Change the final assertion:

```python
def test_vector_basic_enter():
    finals, flags = run([5, 1.9, 1.9, 1.9, 1.9], make_cfg())
    assert list(flags) == [False, False, False, True, True]
    assert np.allclose(finals, [5.0, 1.9, 1.9, 1.9, 1.9])
```

2. Replace the section header comment + `test_final_is_exactly_zero_when_stationary` with a test asserting independence:

```python
# --------------------------------------------------------------------------- #
# Stationary flag is independent of the final speed value
# --------------------------------------------------------------------------- #
def test_stationary_flag_independent_of_final():
    finals, flags = run([5, 1.9, 1.9, 1.9, 1.9], make_cfg())
    # flags still follow the hysteresis state machine
    assert list(flags) == [False, False, False, True, True]
    # but final_speed_mph is the (unclipped, unsmoothed) prediction, never zeroed
    assert np.allclose(finals, [5.0, 1.9, 1.9, 1.9, 1.9])
    # a stationary frame may carry a non-zero final speed
    assert finals[3] == 1.9
    assert flags[3] is True
```

(Keep `test_negative_zero_normalized` unchanged — it verifies the `-0.0 → 0.0` normalization, which still applies.)

3. `test_postprocess_dataframe_adds_columns_and_order` — change the last two assertions:

```python
    # 1.9 for 4 consecutive frames -> stops at index 2 (N=3)
    assert list(out["stationary_pred"]) == [False, False, True, True]
    # final speed keeps the predicted value, even while stationary
    assert out["final_speed_mph"].iloc[2] == 1.9
    assert out["final_speed_mph"].iloc[3] == 1.9
```

- [ ] **Step 2: Run the tests and confirm they FAIL**

Run: `pytest tests/unit/test_postprocessing.py -q`
Expected: FAIL — `test_vector_basic_enter`, `test_stationary_flag_independent_of_final`, and `test_postprocess_dataframe_adds_columns_and_order` fail because `final` is still zeroed.

- [ ] **Step 3: Change `SpeedPostProcessor.step` to stop zeroing**

In `src/inference/postprocessing.py`, replace the body of `step()` (lines 117–124):

```python
        stationary = False
        if self._detector is not None:
            stationary = self._detector.update(value)

        # Speed and stationary are independent outputs: the detector's state no
        # longer zeroes the speed value.
        final = value
        if final == 0.0:  # normalize -0.0 -> 0.0 (harmless safety net)
            final = 0.0
        return PostProcessResult(final_speed_mph=final, stationary=stationary)
```

- [ ] **Step 4: Update the module docstring**

In `src/inference/postprocessing.py`, change line 9 of the docstring:

```python
* ``final_speed_mph`` is the clipped/smoothed value; ``stationary`` is an
  independent boolean output and never zeroes the speed.
```

- [ ] **Step 5: Update the schema comment**

In `src/core/schemas.py`, line 72:

```python
    final_speed_mph: float  # clipped + smoothed predicted speed; independent of `stationary`
```

- [ ] **Step 6: Run the unit tests and confirm they PASS**

Run: `pytest tests/unit/test_postprocessing.py -q`
Expected: PASS (all tests, including `test_negative_zero_normalized`).

- [ ] **Step 7: Run ruff on the changed files**

Run: `ruff check src/inference/postprocessing.py src/core/schemas.py tests/unit/test_postprocessing.py`
Expected: clean (no output).

- [ ] **Step 8: Commit**

```bash
git add src/inference/postprocessing.py src/core/schemas.py tests/unit/test_postprocessing.py
git commit -m feat-decouple-stationary-from-speed-output
```

---

## Task 2: Update the integration test

**Files:**
- Test: `tests/integration/test_full_run.py`

**Interfaces:**
- Consumes: `run_experiment` (existing, unchanged).
- Produces: a passing integration test asserting the new invariant (stationary rows exist and `final_speed_mph >= 0`). The "never zeroed" behavior is robustly covered by the Task 1 unit test.

- [ ] **Step 1: Replace `test_stationary_rows_have_zero_final`**

In `tests/integration/test_full_run.py`, replace lines 70–78:

```python
@pytest.mark.integration
def test_stationary_flag_independent_of_final(tmp_path):
    paths = run_experiment(_smoke_cfg(tmp_path))
    import pandas as pd

    df = pd.read_csv(paths.run_dir / "predictions.csv")
    stationary_rows = df[df["stationary_pred"] == True]  # noqa: E712
    assert len(stationary_rows) > 0
    # final_speed_mph is the clipped/smoothed prediction (min 0.0); the
    # stationary detector must not force it to 0.
    assert (stationary_rows["final_speed_mph"] >= 0.0).all()
```

- [ ] **Step 2: Run the integration test**

Run: `pytest tests/integration/test_full_run.py::test_stationary_flag_independent_of_final -q -m integration`
Expected: PASS.

> Note: this test runs a full smoke training run (synthetic, 2 epochs) and is slow (~30–90 s). If the smoke run cannot execute in your environment (e.g. the local `configs/default.yaml` currently selects `dataset=comma2k19`, which needs the cached data), that is a pre-existing environment/config issue outside this task's scope — the assertion change itself is correct.

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_full_run.py
git commit -m test-update-stationary-integration-assertions
```

---

## Task 3: Update documentation

**Files (all modify):**
- `README.md`
- `docs/PIPELINE.md`
- `docs/CONFIGURATION.md`
- `docs/PLAN.md`
- `docs/STATIONARY_CLASSIFICATION.md`

**Interfaces:** none (docs only).

### README.md

- [ ] **Step 1:** In the intro (lines 5–7), change the post-processing description to reflect two independent outputs:

```markdown
speed (mph) from a sliding window of frames; a causal post-processing stage
(clip → smoothing) produces the speed estimate, while a hysteresis detector
emits an independent moving/stationary flag (`stationary_pred`).
```

### docs/PIPELINE.md

- [ ] **Step 2:** In §7 (lines 336–339), change the detector paragraph:

```markdown
The **stationary detector** is a hysteresis state machine: `≤ enter_threshold`
(2.0 mph) for `enter_persistence_frames` consecutive frames → STOPPED; `>
exit_threshold` (2.5) for N frames → MOVING. It emits an independent
`stationary_pred` flag and no longer zeroes `final_speed_mph`. It's causal and
`reset()` at each sequence boundary.
```

### docs/CONFIGURATION.md

- [ ] **Step 3:** In §3.7 (lines 214–215), change:

```markdown
Pipeline per frame: `raw -> clip -> smoothing -> stationary -> final`. The
`stationary` step emits an independent `stationary_pred` flag and does not
modify the speed value.
```

### docs/PLAN.md

- [ ] **Step 4:** Line 94 (requirement #18):

```markdown
18. `final_speed_mph` is the clipped/smoothed prediction and is never zeroed by the stationary detector; `stationary_pred` is an independent boolean output (normalize `-0.0` to `0.0`, never emit `-0.0`).
```

- [ ] **Step 5:** Line 377 (schema comment in §5.1):

```markdown
    final_speed_mph: float    # clipped + smoothed prediction; independent of `stationary`
```

- [ ] **Step 6:** Line 520 (§6.1):

```markdown
- `final_speed_mph` is the smoothed/clipped value; it is never zeroed by the detector. `stationary_pred` is an independent boolean. Normalize `-0.0` to `0.0`.
```

- [ ] **Step 7:** Line 523 (§6.1 trade-off):

```markdown
- Known trade-off (document in README): persistence delays the `stationary_pred` flag by `N-1` frames when stopping, which lowers stationary recall at transitions. Phase 5 compares configurations on **val**.
```

- [ ] **Step 8:** Line 708 (WP5 tests list) — replace the `final == 0.0 …` clause:

```markdown
**Tests:** all §6.1 vectors; boundary values (2.0, 2.5); clip on/off with negative and huge values; EMA and median numerical examples (hand-computed); NaN raises; `stationary_pred` independent of `final_speed_mph`; `-0.0` normalized to `0.0`; state reset between sequences (second sequence starts in `initial_state`); `process_sequence` equals manual `step` loop; disabled detector ⇒ all False; parameter validation (`enter > exit` raises).
```

- [ ] **Step 9:** Line 722 (WP7 `write_predictions_csv`) — replace "writes stationary finals as `0.0`":

```markdown
**Do:** §5.5/§5.6. `write_metrics_json` emits `schema_version: 1` JSON (NaN → null; `allow_nan=False`). `write_predictions_csv` keeps the §5.4 column order and normalizes `-0.0` to `0.0`. Figures: `Agg` backend; one function per figure; filenames exactly as §5.6; figure closed after saving (`plt.close`); handles empty buckets and single-class stationary data without crashing.
```

- [ ] **Step 10:** Line 894 (final checklist):

```markdown
- [ ] `final_speed_mph` is never zeroed by the stationary detector (stationary rows keep their predicted speed); no `-0.0` in `predictions.csv`.
```

### docs/STATIONARY_CLASSIFICATION.md

- [ ] **Step 11:** Line 14 (TL;DR table, "zero out `final_speed_mph`" cell):

```markdown
| **Predicted stationary** (`stationary_pred`) | `src/inference/postprocessing.py` → `StationaryDetector` | Turn the model's noisy per-frame speed into a stable STOPPED/MOVING state (an independent boolean output) | Hysteresis state machine (thresholds + consecutive-frame persistence) |
```

- [ ] **Step 12:** Line 120 (code snippet in §2):

```python
final = value  # never zeroed by the stationary detector
```

- [ ] **Step 13:** Lines 129–132 (§2 consequence 2):

```markdown
2. **`final_speed_mph` is the clipped/smoothed value and is never zeroed by the
   detector.** The `stationary` flag is an independent boolean output. Negative
   zero is still normalized (`if final == 0.0: final = 0.0`, lines 122–123),
   and the CSV writer re-normalizes as a backstop
   (`src/evaluation/reporting.py`, lines 56–58).
```

- [ ] **Step 14:** Line 232 (§6.1 header) and lines 234–235:

```markdown
### 6.1 Persistence delays the stationary flag by `N-1` frames

Because entering STOPPED requires `N` consecutive qualifying frames, the
`stationary_pred` flag lags the true stop by up to `N-1` frames. This lowers
stationary **recall** around transitions. The same lag applies symmetrically on
exit. This is a known, accepted trade-off (noted in `docs/PLAN.md` §6.1), and is
exactly why the detector study milestone (M2-d) exists.
```

- [ ] **Step 15:** Lines 255–256 (§6.3):

```markdown
- **predicted "stationary"** once the detector is already STOPPED (exit requires
  `> 3.0`), so `stationary_pred = True`.
```

- [ ] **Step 16:** Lines 294–295 (§8 step 4):

```markdown
4. Emit `stationary_pred = (state == STOPPED)` and
   `final_speed_mph = smoothed_value` (independent of the flag).
```

- [ ] **Step 17: Verify no stale zeroing claims remain**

Run: `rg -n "final_speed_mph.*0\.0|0\.0 when stationary|zeroing|writes stationary finals" docs/ src/ README.md` (use ripgrep via the `bash` tool: `rg "..."`)
Expected: no remaining matches that describe `final_speed_mph` being zeroed when stationary (the `-0.0 → 0.0` normalization notes are expected and fine).

- [ ] **Step 18: Commit**

```bash
git add README.md docs/PIPELINE.md docs/CONFIGURATION.md docs/PLAN.md docs/STATIONARY_CLASSIFICATION.md
git commit -m docs-update-stationary-speed-decoupling
```

---

## Task 4: Full verification

**Files:** none (verification only).

- [ ] **Step 1: Lint the whole repo**

Run: `ruff check src tests scripts`
Expected: clean.

- [ ] **Step 2: Run the full unit suite**

Run: `pytest -q`
Expected: PASS in < 60 s.

- [ ] **Step 3: Run the full integration suite**

Run: `pytest -q -m integration`
Expected: PASS in < 120 s (if your environment can run the smoke training; see the Task 2 note about the local `configs/default.yaml` dataset selection).

- [ ] **Step 4: Smoke CLI run**

Run: `python scripts/train.py experiment=smoke`
Expected: exits 0, artifacts under `output/`, nothing outside `output/`.

- [ ] **Step 5: Confirm clean git status (no untracked build artifacts)**

Run: `git status --short`
Expected: no new files outside `output/` (which is git-ignored).

---

## Self-Review Notes (checked)

- **Spec coverage:** the spec's three behaviors are covered — zeroing removed (Task 1 Step 3), detector preserved as independent flag (Task 1 Steps 3–6), metrics unchanged (no task needed — `evaluate_predictions` already separates `overall`/`stationary`). Docs update maps 1:1 to the spec's file list.
- **Placeholder scan:** every step has exact code/text and exact commands.
- **Type consistency:** `PostProcessResult(final_speed_mph=…, stationary=…)` signature is unchanged across all tasks; `stationary_pred` column name is unchanged everywhere.
