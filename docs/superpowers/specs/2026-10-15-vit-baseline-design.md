# Design: ViT + LSTM speed-estimation baseline (`vit_lstm`)

**Date:** 2026-10-15
**Status:** approved for implementation
**Scope:** add a Vision-Transformer encoder to the existing encoder→temporal→head
framework so that `vit_lstm` joins `resnet_lstm`, `mobilenet_lstm`, and
`efficientnet_lstm` as an attention-based encoder baseline. Trainable on
comma2k19 (not a downloadable pretrained speed model — the literature has no
such model; see the 2026-10-15 discussion).

---

## 1. Context

The repo already has a generic `EncoderRecurrent` model (in
`src/models/architectures.py`) that composes any `ENCODER_REGISTRY` entry with a
temporal module (`lstm`/`gru`/`convlstm`) and a regression head. Three frozen
torchvision backbones (ResNet18, MobileNetV3, EfficientNetB0) are available
through `BackboneEncoder`. A Vision Transformer cannot reuse `BackboneEncoder`
unchanged: ViT's feature is a **token sequence** `[B, N+1, hidden]`, not a
pooled vector `[B, dim]` or a conv map. This design adds a small dedicated
`ViTEncoder` and registers `vit_lstm`.

## 2. Goals and non-goals

**Goals**

- Add `ViTEncoder` (frozen ImageNet ViT, CLS-token feature) + `"vit"` registry key.
- Register `vit_lstm` (`EncoderRecurrent` with ViT + LSTM + regression head).
- Offline (no-download) unit tests for shape and freeze semantics.
- `configs/model/vit_lstm.yaml`, README + DECISIONS updates.

**Non-goals**

- Fine-tuning the ViT backbone (frozen only, matching `resnet_lstm`).
- ViT spatial/ConvLSTM mode (ViT tokens are not a regular conv grid).
- Video/spatiotemporal transformers (TimeSformer/ViViT).
- A plain ViT→mean-pool→MLP head (LSTM kept so the *encoder* is the only variable).

## 3. Architecture

```
frames [B, T, 3, 224, 224]
  → ViTEncoder (per-frame, frozen)
       heads → Identity()          →  [B·T, 197, 768]
       CLS token [:, 0]            →  [B·T, 768]
       Linear proj (Identity if 768 == feature_dim)
  → LSTM(input_dim=feature_dim, hidden=64)   → last timestep [B, 64]
  → Linear(64, 1) → speed mph [B]
```

`EncoderRecurrent.forward` dispatches on `feats.ndim == 2` (vector path) — ViT
returns a 2-D vector, so no change to the wrapper is needed.

## 4. Components (files)

| File | Action | What |
|---|---|---|
| `src/models/architectures.py` | edit | add `ViTEncoder` + `_VIT_BUILDERS`; register `"vit"` in `ENCODER_REGISTRY`; add `"vit_lstm"` to `MODEL_REGISTRY` |
| `configs/model/vit_lstm.yaml` | new | model config |
| `tests/unit/test_models.py` | edit | `test_vit_lstm_shape_and_frozen`, `test_unknown_vit_variant_raises` |
| `README.md` | edit | baselines table + `image_size: [224, 224]` + compute note |
| `docs/DECISIONS.md` | edit | record the ViT baseline decisions |

## 5. Encoder spec

```python
class ViTEncoder(nn.Module):
    def __init__(self, feature_dim=768, pretrained=True, frozen=True,
                 weights_path=None, variant="vit_b_16", image_size=224): ...
    def forward(self, x: Tensor) -> Tensor:        # [B, C, H, W]
        tokens = self.backbone(x)                  # [B, N+1, hidden]
        return self.proj(tokens[:, 0])             # [B, feature_dim]
```

- Variants: `vit_b_16`, `vit_b_32`, `vit_l_16`, `vit_l_32` (all expose
  `IMAGENET1K_V1`); `vit_h_14` excluded (SWAG-only weights).
- `weights = None` when `pretrained=false` (tests) or `weights_path` is set.
- `weights_path` loads a local state_dict (offline reuse), same as `BackboneEncoder`.
- `frozen=true` → `requires_grad_(False)` + `eval()` (LayerNorm/GELU only; no BN).
- Native feature dim read from `model.hidden_dim` (768 for B, 1024 for L).
- Input fixed at 224×224 (native positional embeddings; no interpolation).

## 6. Config (`configs/model/vit_lstm.yaml`)

```yaml
name: vit_lstm
encoder: {name: vit, variant: vit_b_16, pretrained: true, feature_dim: 768, frozen: true, weights_path: null, image_size: 224}
temporal: {name: lstm, hidden_size: 64, num_layers: 1}
checkpoint: null
```

`preprocessing.image_size: [224, 224]` must accompany it (README documents this).

## 7. Tests (offline, `pretrained: false`, no network)

- `vit_lstm` output `[B]` and finite for `B=2, T=3` at `224×224`.
- Frozen: zero trainable params inside `model.encoder.backbone`; temporal + head
  trainable; total trainable < total params.
- Unknown `variant` raises `ConfigError`.

## 8. Compute note

ViT-B/16 (86M params, 197 tokens/frame) is roughly 5–10× heavier per frame than
ResNet18. `vit_b_32` is the cheaper fallback (coarser patches). No new
dependencies — torchvision (already a dep) provides the ViT models and weights.

## 9. Decisions

| ID | Decision |
| --- | --- |
| D-V-1 | `vit_lstm` = frozen ImageNet ViT + LSTM + regression head (`EncoderRecurrent`) |
| D-V-2 | CLS-token feature (768-d), optional `feature_dim` projection |
| D-V-3 | Input 224×224 (native pos-embeddings; no interpolation) |
| D-V-4 | Frozen backbone (consistent with `resnet_lstm`) |
| D-V-5 | Variants b16/b32/l16/l32; default `vit_b_16` |
| D-V-6 | No new dependencies (torchvision ViT) |
