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

## ViT baseline (2026-10-15 design)

| ID | Decision |
| --- | --- |
| D-V-1 | `vit_lstm` = frozen ImageNet ViT + LSTM + regression head (`EncoderRecurrent`) |
| D-V-2 | CLS-token feature (768-d); optional `feature_dim` projection |
| D-V-3 | Input 224x224 (native positional embeddings; no interpolation) |
| D-V-4 | Frozen backbone (consistent with `resnet_lstm`) |
| D-V-5 | Variants b16/b32/l16/l32; default `vit_b_16` |
| D-V-6 | No new dependencies (torchvision ViT) |

## Composable model config (2026-10-16 design)

| ID | Decision |
| --- | --- |
| D-F-1 | `encoder`/`temporal`/`head` composable config groups; `model` namespace kept |
| D-F-2 | `spatial` inferred from `temporal.name == "convlstm"` |
| D-F-3 | Head MLP via `hidden_dims` + global `activation`/`dropout`; `[]` = single linear |
| D-F-4 | `MODEL_REGISTRY` removed; `build_model` always builds `EncoderRecurrent` |
| D-F-5 | `model.name` is a label (default `encoder_recurrent`) |
| D-F-6 | Encoder files use native `feature_dim`; override for lighter heads |
