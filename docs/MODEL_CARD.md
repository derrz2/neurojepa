# NeuroJEPA pretrained encoders

## Purpose

Research representations for preprocessed ROI-by-time fMRI series. Encoders
return embeddings; they do not directly return clinical diagnoses or
validated clinical predictions. Downstream heads must be trained using
appropriate training/validation/test splits.

## Artifact status

| Variant | Encoder parameters | Width / layers / heads | Weight status |
| --- | ---: | --- | --- |
| `2m` | 2,654,336 | 128 / 12 / 2 | Epoch 63, encoder-only safetensors |
| `10m` | 10,417,408 | 256 / 12 / 4 | Epoch 299, encoder-only safetensors |

Both configurations use temporal patches of 40 and ROI patches of 1.
Both released candidates use four register tokens. Exact normalization,
rotary-position, attention-gating, and layer-scale settings are stored in
the JSON configuration shipped with each architecture.

Both available checkpoints were referenced by preserved downstream experiment
records. Parameter counts and optimizer-step metadata have been checked.
The paper-table identity remains provisional: parameter counts are not
unique identifiers, and rounded compute estimates depend on crop-accounting
assumptions. The small-model JSON includes an explicitly labeled analytic
compute estimate, not a hardware measurement or a unique identifier.
See [checkpoint selection](CHECKPOINT_SELECTION.md).

## Inputs and outputs

Input: floating-point `(batch, ROI, time)` tensors. A standard smoke-test
shape is `(2, 100, 200)`. Output: `(batch, 256)` for `10m` and `(batch, 128)`
for the `2m` architecture. The public loader returns an evaluation-mode model.

The loader does not denoise, register, parcellate, resample, crop, or
normalize raw scans. Prepare ROI series according to your study protocol;
preserve the ROI/time axis order and avoid fitting preprocessing on test
data. Support for variable input shapes is not evidence of equivalent
performance across atlases or acquisition protocols.

## Limitations and reproducibility

No participant data, manifests, task labels, or trained diagnostic heads
are distributed. The training scripts and generic example configurations
are included for adaptation, not as a claim of complete paper reproduction.
Dataset provenance, preprocessing details, paper bibliographic metadata,
and definitive checkpoint-to-table mapping must be completed before a
public paper release. Exact optimizer-state resumption is unsupported by
the encoder-only export.
