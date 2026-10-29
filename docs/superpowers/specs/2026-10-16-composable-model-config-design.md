# Design: composable encoder / temporal / head model config

**Date:** 2026-10-16
**Status:** approved for implementation
**Scope:** replace the per-combination `model/*.yaml` configs and the redundant
`MODEL_REGISTRY` with three composable Hydra groups — `encoder`, `temporal`,
`head` — plus a configurable MLP regression head. Any encoder × temporal × head
combination is selected on the CLI without a new file or registry entry.

---

## 1. Context

Every model registered in `MODEL_REGISTRY` mapped to the *same* class
(`EncoderRecurrent`); the registry and one config file per `encoder × temporal`
combination existed only for naming/validation. The regression head was a fixed
`nn.Linear(hidden_size, 1)`. This design (a) makes `encoder`/`temporal`/`head`
first-class config groups and (b) makes the head a configurable MLP.

## 2. Goals and non-goals

**Goals**

- `encoder` / `temporal` / `head` composable config groups; `build_model` is generic.
- Configurable regression head (`hidden_dims`, `activation`, `dropout`).
- `spatial` inferred from the temporal module (ConvLSTM ⇒ spatial encoder).
- Migrate existing `model/*.yaml` into the new groups, preserving current values.

**Non-goals**

- Per-layer activation/dropout (a global `activation`/`dropout` is enough for now).
- New model classes beyond `EncoderRecurrent` (no registry; add one later if needed).
- Bidirectional / dropout on the LSTM/GRU temporal modules (future work).

## 3. Config layout

```
configs/
  default.yaml          # defaults: - encoder: resnet18   - temporal: convlstm   - head: linear
  encoder/  {tiny_cnn, resnet18, resnet50, mobilenet_v3_small, efficientnet_b0, vit_b_16}.yaml
  temporal/ {gru, lstm, convlstm}.yaml
  head/     {linear, mlp}.yaml
```

Each group writes into the `model` namespace via an `@package` header:

```yaml
# configs/encoder/resnet50.yaml
# @package model.encoder
name: backbone
backbone: resnet50
pretrained: true
feature_dim: 2048
frozen: true
weights_path: null
```

```yaml
# configs/temporal/convlstm.yaml
# @package model.temporal
name: convlstm
hidden_size: 32
kernel_size: 3
num_layers: 1
```

```yaml
# configs/head/linear.yaml
# @package model.head
hidden_dims: []
activation: none
dropout: 0.0
```

Usage:

```bash
python scripts/train.py encoder=resnet50 temporal=gru
python scripts/train.py encoder=vit_b_16 temporal=lstm head=mlp preprocessing.image_size=[224,224]
```

## 4. `ModelCfg`

```python
@dataclass
class ModelCfg:
    name: str = "encoder_recurrent"          # label only (export/tracking)
    encoder: dict[str, Any] = field(default_factory=dict)
    temporal: dict[str, Any] = field(default_factory=dict)
    head: dict[str, Any] = field(default_factory=dict)
    checkpoint: str | None = None
```

`name` is no longer used for dispatch; it labels the standalone export.

## 5. `EncoderRecurrent` changes

- Signature `(encoder, temporal, head=None)`.
- `spatial = (temporal.name == "convlstm")`; injected into spatial-capable
  encoders (`backbone`, `resnet18`). Vector-only encoders (`tiny_cnn`, `vit`)
  raise a `ConfigError` when paired with `convlstm`.
- `feature_dim` stays an explicit encoder key (native value by default; override
  e.g. `encoder.feature_dim=32` for a lighter ConvLSTM).
- Head built from `head` dict:

```python
dims = [hidden_size, *head.hidden_dims, 1]
Linear(d0→d1) → Dropout → Act → … → Linear(d_last→1)
```

`hidden_dims: []` ⇒ single `Linear(hidden_size, 1)` (current behavior).
Activations: `none|relu|gelu|leaky_relu|tanh|sigmoid`.

## 6. `build_model`

`MODEL_REGISTRY` removed. `build_model(cfg)` always returns
`EncoderRecurrent(encoder=cfg.encoder, temporal=cfg.temporal, head=cfg.head)`.

## 7. Export

`save_model_export` / `load_model_export` now carry `head` (missing key tolerated
for old exports).

## 8. Migrated feature_dim values

| encoder | feature_dim |
| --- | --- |
| tiny_cnn | 64 |
| resnet18 | 512 |
| resnet50 | 2048 |
| mobilenet_v3_small | 576 |
| efficientnet_b0 | 1280 |
| vit_b_16 | 768 |

## 9. Decisions

| ID | Decision |
| --- | --- |
| D-F-1 | `encoder`/`temporal`/`head` composable groups; `model` namespace kept |
| D-F-2 | `spatial` inferred from `temporal.name == "convlstm"` |
| D-F-3 | Head MLP via `hidden_dims` + global `activation`/`dropout`; `[]` = single linear |
| D-F-4 | `MODEL_REGISTRY` removed; `build_model` generic |
| D-F-5 | `model.name` is a label (default `encoder_recurrent`) |
| D-F-6 | Encoder files use native `feature_dim`; override for lighter heads |
