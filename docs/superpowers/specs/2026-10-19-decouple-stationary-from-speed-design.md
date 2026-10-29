# Design: decouple stationary classification from speed output

**Date:** 2026-10-19
**Status:** approved for implementation
**Scope:** stop zeroing `final_speed_mph` when the vehicle is classified stationary,
so that speed prediction and moving/stationary classification are evaluated as two
independent outputs. The existing hysteresis detector is kept unchanged, but now
serves only as a rule-based stationary classifier that no longer mutates the speed
value.

---

## 1. Context

The post-processing pipeline currently couples the two tasks at the output:

```text
raw → clip → smoothing → stationary detector → final
```

When the hysteresis detector (`StationaryDetector`) is in the STOPPED state,
`SpeedPostProcessor.step` overwrites the smoothed speed with exactly `0.0`
(`src/inference/postprocessing.py`, lines 121–123):

```python
final = 0.0 if stationary else value
```

This means the headline regression metrics (`overall`, i.e. RMSE/MAE/R²/median-AE
in `src/evaluation/metrics.py` → `evaluate_predictions`) are computed on a signal
that has been artificially zeroed. A vehicle truly moving at ~2.5 mph but classified
stationary contributes a full ~2.5 mph error to RMSE — a zeroing artifact, not a
speed-prediction error.

We want speed prediction and stationary classification to be **two independent
outputs**, each scored on its own merit:

- **Speed** — the model's predicted speed, post-processed only by clip + smoothing.
  Evaluated with regression metrics (overall + per-bucket).
- **Stationary** — a separate boolean classification, evaluated with its own
  precision/recall/F1 + confusion counts.

The stationary detection is not removed — it becomes a standalone rule-based
classifier (the hysteresis state machine) that is reported alongside, but no longer
touches, the speed value.

## 2. Goals and non-goals

**Goals**

- `final_speed_mph` is never zeroed by the stationary detector; it is the
  clip → smoothing output, always.
- `stationary_pred` remains a valid, independently evaluated boolean classification.
- Headline speed metrics reflect speed error only (no zeroing artifact).
- Preserve all existing thresholds, persistence, reset, clip, and smoothing behavior.

**Non-goals**

- No new model architecture (the model stays a single regression head).
- No learned classification head and no classic-ML classifier head (the hysteresis
  detector is the rule-based stationary classifier for now; a learned/ML comparison
  is future work, tracked separately).
- No change to `stationary_metrics`, the `stationary_pred` column, or the
  confusion-matrix figure.
- No change to clip or smoothing semantics.

## 3. Behavioral change

### 3.1 Post-processing

In `SpeedPostProcessor.step` (`src/inference/postprocessing.py`):

```python
# before
final = 0.0 if stationary else value
if final == 0.0:          # also normalizes -0.0 -> 0.0
    final = 0.0

# after
final = value
if final == 0.0:          # keep -0.0 -> 0.0 normalization (harmless safety net)
    final = 0.0
```

`value` is already the clip → smoothing output. The `stationary` flag is still
computed and returned, but is no longer used to alter `final`.

### 3.2 Resulting per-frame outputs

| Column | Value after change |
|--------|--------------------|
| `raw_speed_mph` | model output, unprocessed |
| `final_speed_mph` | `clip(raw) → smoothing` (no zeroing) |
| `stationary_pred` | hysteresis state on the smoothed value (`True` iff STOPPED) |

A stopped vehicle now reports its (noisy) predicted speed — e.g. ~1–2 mph — together
with `stationary_pred = True`. Consumers combine both signals; the speed number no
longer collapses to 0 by itself.

### 3.3 Evaluation

`evaluate_predictions` (`src/evaluation/metrics.py`) needs no structural change:
it already computes `overall` (on `final_speed_mph`), `overall_raw` (on
`raw_speed_mph`), and `stationary` (on `stationary_pred`) separately. After the
change:

- `overall` = smoothed speed vs truth (regression error only).
- `overall_raw` = raw model vs truth.
- `stationary` = unchanged P/R/F1 + confusion counts.

`overall` and `overall_raw` now differ only by clip + smoothing.

## 4. Files to change

| File | Change |
|------|--------|
| `src/inference/postprocessing.py` | `step()`: remove zeroing; update module docstring (drop "`final_speed_mph` is exactly `0.0` when stationary"). |
| `src/core/schemas.py` | Update `SpeedPrediction.final_speed_mph` comment ("exactly 0.0 when stationary" → "post-processed predicted speed (not zeroed)"). |
| `tests/unit/test_postprocessing.py` | Update zeroing assertions; add coverage that `stationary_pred` is independent of `final_speed_mph`. |
| `tests/integration/test_full_run.py` | Replace `test_stationary_rows_have_zero_final` with the new expectation (stationary rows keep their predicted speed). |
| `README.md` | Update pipeline description (zeroing removal). |
| `docs/PIPELINE.md` §7 | Update post-processing + metrics text. |
| `docs/CONFIGURATION.md` | Update the stationary section (detector no longer zeroes). |
| `docs/PLAN.md` | Update §5.1 (`final_speed_mph` comment, line 377), §6.1 (the "`0.0` if stationary" rule, line 520), and requirement #18 (line 94). |
| `docs/STATIONARY_CLASSIFICATION.md` | Update §1/§2 zeroing claims and the "final = 0.0" statements. |

## 5. Test plan

Update and extend existing tests:

1. `test_vector_basic_enter` — flags unchanged; `finals` now equal the input speeds
   (e.g. `[5.0, 1.9, 1.9, 1.9, 1.9]`), not `[5.0, 1.9, 1.9, 0.0, 0.0]`.
2. `test_final_is_exactly_zero_when_stationary` — replace with a test that the
   final value equals the clipped/smoothed value regardless of the flag.
3. `test_negative_zero_normalized` — still relevant only insofar as `-0.0` inputs
   normalize to `0.0`; keep, adjusted for no zeroing.
4. `test_postprocess_dataframe_adds_columns_and_order` — `final_speed_mph` stays
   `1.9` for the stationary rows while `stationary_pred` is `True`.
5. `test_postprocess_dataframe_resets_between_sequences` — flags unchanged; finals
   reflect raw values.
6. New: assert `stationary_pred` and `final_speed_mph` are independent (a row can
   be `stationary_pred == True` with `final_speed_mph > 0`).
7. Integration `test_full_run.py` — no row is required to have `final == 0.0`;
   assert `stationary` confusion metrics still `n_true_stationary > 0` and that the
   stationary rows are flagged regardless of their speed value.

## 6. Risks / notes

- **Frozen-contract change.** `docs/PLAN.md` §5.1 (schemas) and §6.1 (stationary
  detection) are marked "frozen by WP0 — do not change without orchestrator
  approval." This design changes the *behavioral* contract ("`final_speed_mph` is
  exactly `0.0` when stationary") without changing any API *signature*. It is
  explicitly authorized by the project owner via this design.
- **Downstream consumers of `final_speed_mph`** that assume it is `0.0` when stopped
  must now read `stationary_pred` for the stopped decision. This is an intended
  contract change; the two signals must be combined.
- The zeroing artifact disappears from RMSE, so `overall` may change value for the
  same model — this is expected and is the point of the change, not a regression.
- The hysteresis detector still runs on the **smoothed** speed; its input is
  unchanged by this design.

## 7. Out of scope (future work)

- Learned binary classification head (multi-task) and/or a classic-ML classifier
  (logistic regression / random forest / GBDT on the frozen encoder+temporal
  embeddings) as comparison baselines. These were discussed and deferred; they
  depend on this decoupling first.
