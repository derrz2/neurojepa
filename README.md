# NeuroJEPA

Research code for fMRI representation learning, with a simple interface for
loading pretrained encoders and extracting downstream features.

**Private preparation release:** both 2.65M and 10.42M candidate encoders
are included, with strict-loading and output-equivalence checks. The small
model matches 0.128 EFLOPs under the documented estimation convention.
The definitive mapping from these candidates to the paper's table remains
to be confirmed; see [checkpoint selection](docs/CHECKPOINT_SELECTION.md).

## Quick start

Use Python 3.10 or later. The release was tested with Python 3.10,
PyTorch 2.4.0/2.4.1, and timm 1.0.22. After downloading or cloning this repository,
run the following from its root directory:

```bash
python -m pip install -e .
python examples/extract_features.py --variant 10m --device cpu
python examples/extract_features.py --variant 2m --device cpu
```

The checkpoint is a regular file in `checkpoints/`; no Hugging Face login,
Git LFS client, or model download service is needed after obtaining the repo.
For CUDA, install the appropriate PyTorch build for your machine and pass
`--device cuda`. Only the CPU inference path has been verified in this release.

```python
import torch
from neurojepa import load_model

encoder = load_model("10m", device="cpu")
x = torch.randn(2, 100, 200)  # synthetic example: batch, ROI, time
with torch.inference_mode():
    features = encoder(x)     # shape: (2, 256)
```

The loader checks SHA-256, parameter count, and all state-dictionary keys.
It uses safetensors and does not unpickle training checkpoints. It never
silently falls back to random weights. For weights stored elsewhere, pass
`checkpoint_dir="checkpoints"` explicitly to `load_model`.

## Models

| Variant | Parameters | Completed epochs | Availability |
| --- | ---: | ---: | --- |
| `2m` | 2,654,336 | 64 | Verified candidate, epoch 63, weights included |
| `10m` | 10,417,408 | 300 | Verified encoder weights included |

Weights exclude the pretraining projector, optimizer, logs, and serialized
training configuration. They support downstream initialization and feature
extraction, not exact pretraining resumption. See [model details](docs/MODEL_CARD.md)
and [checkpoint information](checkpoints/README.md).

## Your data

Supply preprocessed ROI-by-time series; the inference API does not process
raw NIfTI scans or normalize inputs. No participant data, identifiers, split
manifests, or label files are included. Match preprocessing, parcellation,
and time-axis conventions to your study. Synthetic inputs above only test
the software interface and are not a scientific evaluation.

## Training and downstream evaluation

Install the optional dependencies:

```bash
python -m pip install -e ".[train]"
```

Alternatively use `environment.yml` for the original CUDA-oriented Conda
environment. Adapt the example configs to your data and task before running:

```bash
python pretrain.py --config configs/pretrain_example.yaml --output_dir outputs/pretrain
python finetune.py --config configs/finetune_example.yaml --output_dir outputs/finetune
```

The downstream example selects the included 10.42M architecture and
safetensors weights. Its task and split paths are placeholders. Full
training/evaluation and paper metrics have not been reproduced by the
release checks. The original internal objective selector `lejepa` is
retained for compatibility; the public project name is NeuroJEPA.
Multi-GPU execution uses `torchrun --nproc_per_node=<NUM_GPUS>` in place of
`python`. Experiment tracking is disabled by default.

## Verification

```bash
python -m pip install -e ".[test]"
python -m pytest -q tests
```

The checks cover the included encoder, finite outputs, input/output shape,
missing weights, checksum rejection, and downstream initialization for both
variants. See [verification scope](docs/VERIFICATION.md).

## Release scope and attribution

This repository excludes experiment outputs, backups, data, credentials,
and private infrastructure paths. Third-party attribution is retained;
see [third-party notices](THIRD_PARTY_NOTICES.md). A project-wide license
and paper citation have not yet been assigned in this private preparation
release. A private repository is not an anonymous reviewer access link;
reviewer access and publication metadata need a separate release decision.
