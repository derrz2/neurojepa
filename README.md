# A SCALING STUDY FOR FMRI FOUNDATION MODELS

**NeuroJEPA** learns representations from fMRI ROI time series. This repository
provides pretrained encoders and code to extract features, train downstream
models, and evaluate held-out data.

**Start here:** [Install](#1-install) · [Load checkpoints](#2-load-a-pretrained-checkpoint) ·
[Extract features](#3-extract-and-save-features) · [Train and test a linear probe](#4-train-a-linear-probe-and-evaluate-a-test-set) ·
[Fine-tune](#5-run-the-downstream-training-pipeline) · [Software tests](#6-check-your-installation)

## 1. Install

Run from the repository root with Python 3.10:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -c requirements/constraints-py310.txt -e ".[train,test]"
python -m pip check
```

The Linux environment has been tested on CPU and an RTX 3090.
[Environment setup](docs/ENVIRONMENT.md) covers the exact dependency lock,
CPU-only wheels, mirrors, Conda, and Windows caveats.

## 2. Load a pretrained checkpoint

Both checkpoints are included as regular files: no Git LFS or Hugging Face
download is needed after obtaining this private repository.

| Variant | Encoder parameters | Feature dimension | Checkpoint in `checkpoints/` |
| --- | ---: | ---: | --- |
| `2m` | 2,654,336 | 128 | `neurojepa-2m-epoch63.safetensors` |
| `10m` | 10,417,408 | 256 | `neurojepa-10m-epoch299.safetensors` |

```python
import torch
from neurojepa import load_model

device = "cuda" if torch.cuda.is_available() else "cpu"
encoder = load_model("2m", device=device)  # use "10m" for the larger encoder
encoder.requires_grad_(False)

# Synthetic example. Replace with your preprocessed fMRI batch.
x = torch.randn(8, 100, 200, device=device)  # (samples, ROI, time)
with torch.inference_mode():
    features = encoder(x)                 # (8, 128); "10m" gives (8, 256)
```

`load_model` automatically selects the matching architecture, verifies SHA-256,
and strictly loads every encoder parameter. It returns an evaluation-mode model;
missing or incompatible weights raise an error, not a random-weight fallback.
For another checkpoint directory, use
`load_model("2m", checkpoint_dir="my_checkpoints", device=device)` with the
same release filenames. Arbitrary checkpoints need their own architecture.

These are **encoder-only** weights: outputs are features, not class predictions.
A downstream classifier must be trained for your labels. The weights do not
contain a trained task head or the optimizer/projector needed to resume pretraining.

## 3. Extract and save features

Prepare one floating-point NumPy array per split:

| File | Shape | Contents |
| --- | --- | --- |
| `data/X_train.npy` | `(N_train, 100, 200)` | Preprocessed training ROI time series |
| `data/y_train.npy` | `(N_train,)` | Integer class labels aligned with rows |
| `data/X_test.npy` | `(N_test, 100, 200)` | Held-out ROI time series |
| `data/y_test.npy` | `(N_test,)` | Held-out labels aligned with rows |

The `.npy` extraction path does **not** normalize, crop, or preprocess inputs.
Use your study's preprocessing and a consistent ROI/time convention. It does not
accept raw NIfTI volumes. The documented input shape is a tested example, not
a declaration that any parcellation is scientifically interchangeable.

```bash
python examples/extract_features.py --variant 2m --input data/X_train.npy --output outputs/2m/train_features.npy --batch-size 16 --device cuda
python examples/extract_features.py --variant 2m --input data/X_test.npy --output outputs/2m/test_features.npy --batch-size 16 --device cuda
```

Use `--device cpu` without a GPU. For the larger model, replace `--variant 2m`
with `--variant 10m` and save to a different output directory. Keep the same
variant for every split. Output rows preserve input order, so labels remain
aligned. Output shapes are `(N, 128)` or `(N, 256)`; no gradients are retained.
Input batches are memory-mapped; the feature matrix is accumulated in host RAM.

Prefer Python? For an already-loaded, suitably sized array:

```python
import numpy as np
import torch
from neurojepa import load_model

encoder = load_model("2m", device="cpu")
x = torch.from_numpy(np.load("data/X_test.npy", allow_pickle=False)).float()
with torch.inference_mode():
    features = encoder(x).numpy()
# Saving a new result file:
with open("test_features.npy", "xb") as stream:
    np.save(stream, features, allow_pickle=False)
```

## 4. Train a linear probe and evaluate a test set

Once features are saved, the encoder is no longer needed for probe fitting.
The following example fits feature standardization and a logistic-regression
classifier **using training rows only**, then evaluates the held-out test set:

```bash
python examples/linear_probe.py --train-features outputs/2m/train_features.npy --train-labels data/y_train.npy --test-features outputs/2m/test_features.npy --test-labels data/y_test.npy --C 1.0 --output-dir outputs/2m/probe
```

Results appear in the new output directory:

- `metrics.json`: test accuracy, balanced accuracy, and macro-F1.
- `test_predictions.npy`: predicted labels, in test-input order.
- `test_probabilities.npy`: class probabilities; columns follow `classes` in `probe.npz`.
- `probe.npz`: classifier coefficients/intercept, class order, and training-set scaler parameters.

This is a minimal **fixed-C classification example**, not the paper benchmark
protocol. Choose `C` before looking at test results; use a separate validation
split for tuning. For validation-based selection and the original training
entry point, use the next section.

Split by **subject before extraction**: sessions/windows from one participant
must not cross train/validation/test boundaries. Row-wise random splitting can
leak subject information. The simple array example cannot check subject overlap;
you must enforce it in your split construction. Do not interpret metrics from
random synthetic data as model performance.

The scripts refuse to overwrite existing result paths; choose a new output
file/directory when repeating an experiment.

## 5. Run the downstream training pipeline

Use `finetune.py` when you want manifest-based loading, validation-based probe
selection, a trainable task head, or end-to-end fine-tuning. The
[downstream guide](docs/DOWNSTREAM.md) explains the manifest format, preprocessing,
mode settings, and output files.

Create `data/train.csv`, `data/val.csv`, and `data/test.csv` with columns
`Path,subject,label`. Each `Path` points to one `(ROI, time)` NumPy sample.
Then generate a configuration whose architecture exactly matches the checkpoint:

```bash
python scripts/make_downstream_config.py --variant 2m --mode linear_probe --output outputs/configs/2m_probe.yaml
python finetune.py --config outputs/configs/2m_probe.yaml --output_dir outputs/2m_manifest_probe
```

The generated example keeps your validation/test manifests fixed, fits frozen
features, and selects `C` from `[0.01, 0.1, 1.0, 10.0]` by validation loss.
Test metrics are reported for the selected model. This user example does not
claim to reproduce the paper's experimental protocol.

For end-to-end fine-tuning:

```bash
python scripts/make_downstream_config.py --variant 10m --mode full_finetune --output outputs/configs/10m_finetune.yaml
python finetune.py --config outputs/configs/10m_finetune.yaml --output_dir outputs/10m_finetune
```

Use `--mode frozen_head` to train only the task head. Set `--num-classes` for
your classification task and use `--train-manifest`, `--val-manifest`, and
`--test-manifest` for different CSV locations. Inspect learning rate, epochs,
batch size, and task settings in the generated YAML before starting training.
The encoder-only release is used as **initialization**, not a resume checkpoint.

## 6. Check your installation

These are **software checks**, distinct from evaluating your scientific test set:

```bash
# No participant data required: load each checkpoint and check feature shapes.
python examples/extract_features.py --variant 2m --device cpu
python examples/extract_features.py --variant 10m --device cpu

# Check dependencies, both encoders, and synthetic downstream updates.
python scripts/verify_environment.py --device cpu
python scripts/verify_environment.py --device cuda --pretrain-smoke

# Automated loading, checksum, downstream, and configuration tests.
python -m pytest -q tests
```

The documented workflows passed 19 automated tests on Linux. Both encoders
passed CPU/CUDA loading and file extraction; synthetic downstream runs covered
both sizes and one epoch of 10m fine-tuning. These are software checks, not
reproductions of paper metrics. Windows and multi-GPU execution are not verified.

## Pretraining and release notes

For pretraining from scratch, adapt your data settings in
`configs/pretrain_example.yaml`, then run:

```bash
python pretrain.py --config configs/pretrain_example.yaml --output_dir outputs/pretrain
```

This compact release includes the NeuroJEPA/LeJEPA objective and ViT backbone;
unrelated baseline implementations are excluded. The internal selector `lejepa`
is retained for compatibility.

No participant data, private infrastructure paths, credentials, experiment
outputs, or backups are included. Preserve [third-party notices](THIRD_PARTY_NOTICES.md).
A project-wide license and final paper citation are pending. This private
repository is not an anonymous reviewer-access link.
