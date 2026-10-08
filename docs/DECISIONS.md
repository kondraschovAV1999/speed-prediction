# Decisions

| ID  | Decision / answer | Source | Date | Impact |
| --- | --- | --- | --- | --- |
| D1  | Dataset format / label source / units: synthetic only for M1; labels in mph | assumed | 2026-10-08 | blocks M2 |
| D2  | Native frame rate: 10 Hz, constant nominal | assumed | 2026-10-08 | blocks M2 |
| D3  | Camera/vehicle fixed (monocular scale, intrinsics): ignored | assumed | 2026-10-08 | blocks M2 model choice |
| D4  | Inference strictly causal: **yes** | assumed | 2026-10-08 | design already assumes yes |
| D5  | Target runtime (FPS/device): report only, no pass/fail threshold | assumed | 2026-10-08 | blocks M2 benchmark |
| D6  | "Creeping ≤ 2 mph" ground truth derived from label (`<= 2.0`): yes | assumed | 2026-10-08 | blocks M2 |
| D7  | Detector initial state at sequence start: MOVING | assumed | 2026-10-08 | blocks M2 tuning |
| D8  | "Structured logging" = stdlib logging, `key=value` events (not JSON lines): yes | assumed | 2026-10-08 | none |
| D-env | Dev environment: harness-managed Conda env (Python 3.10), not `speed-estimation`/3.11; `requires-python = ">=3.10"` | orchestrator | 2026-10-08 | pyproject only |
| D-mlflow | Local MLflow backend: SQLite URI (`sqlite:///…/mlflow.db`) — file stores are phased out in MLflow 3.x | orchestrator | 2026-10-08 | WP8 |

> All assumed defaults: confirm before M2 (see PLAN.md §2.1).
