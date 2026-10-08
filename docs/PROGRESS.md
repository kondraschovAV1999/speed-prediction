# Progress

| WP | Agent | Status | Gate notes | Date |
| --- | --- | --- | --- | --- |
| WP0 | primary | accepted | scaffold + contracts; 12 tests | 2026-10-08 |
| WP1 | primary | accepted | config/logging/run-manager; 24 tests | 2026-10-08 |
| WP2 | primary | accepted | data core; 20 tests | 2026-10-08 |
| WP3 | primary | accepted | preprocessing + dataset; 34 tests | 2026-10-08 |
| WP4 | primary | accepted | models + training; 11 tests | 2026-10-08 |
| WP5 | primary | accepted | postprocessing; 19 tests | 2026-10-08 |
| WP6 | primary | accepted | evaluation; 14 tests | 2026-10-08 |
| WP7 | primary | accepted | reporting + visualization; 8 tests | 2026-10-08 |
| WP8 | primary | accepted | tracking; 8 tests | 2026-10-08 |
| WP9 | primary | accepted | pipeline + benchmark; 7 tests | 2026-10-08 |
| WP10 | primary | accepted | experiment + train script; 7 integration tests | 2026-10-08 |
| WP11 | primary | accepted | secondary scripts; 3 integration tests | 2026-10-08 |
| WP12 | primary | accepted | full suite 147 tests; smoke + baseline trains exit 0 | 2026-10-08 |

Deviations from plan (recorded in DECISIONS.md and handoff notes):

- Dev environment is the harness-managed Conda env (Python 3.10), not
  `speed-estimation`/3.11; `requires-python = ">=3.10"`.
- MLflow local backend uses a SQLite URI (file stores phased out in MLflow 3.x).
- `smoke` experiment uses its own split manifest (`synthetic_smoke_seed*.json`)
  to avoid colliding with the baseline manifest at the same seed.
- `integration/test_pipeline.py` renamed to `test_full_run.py` to avoid the
  module-name clash with `unit/test_pipeline.py`.
- The tiny-CNN encoder uses adaptive 4x4 pooling (not 1x1 global pooling) so the
  temporal module can recover motion cues.
