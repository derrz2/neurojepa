# A SCALING STUDY FOR FMRI FOUNDATION MODELS

NeuroJEPA: pretrained fMRI encoders, feature extraction, and downstream evaluation.

## Install

Linux / Python 3.10. Run from the repository root:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -c requirements/constraints-py310.txt -e ".[train,test]"
```

[CPU-only, Conda, and dependency lock](docs/ENVIRONMENT.md).

## Load checkpoints

Both encoder checkpoints are included in `checkpoints/`.

| Variant | Parameters | Feature dimension |
| --- | ---: | ---: |
| `2m` | 2.65M | 128 |
| `10m` | 10.42M | 256 |

```python
import torch
from neurojepa import load_model

encoder = load_model("2m", device="cpu")  # or "10m", device="cuda"
x = torch.randn(2, 100, 200)              # synthetic (samples, ROI, time)
with torch.inference_mode():
    features = encoder(x)               # (2, 128); 10m: (2, 256)
```

## Extract features

Prepare `data/X_{train,val,test}.npy` as preprocessed float arrays
`(N, 100, 200)`, and aligned `y_{train,val,test}.npy` integer labels `(N,)`.
Keep subjects disjoint across splits. This API does not crop or normalize inputs.

```bash
# Change 2m to 10m for the larger encoder; use --device cpu without a GPU.
for split in train val test; do
  python examples/extract_features.py --variant 2m \
    --input data/X_${split}.npy --output outputs/2m/${split}_features.npy \
    --batch-size 16 --device cuda
done
```

## Downstream: train → val selection → test

Reuse the original `finetune.py` pipeline: fit on train, select C on val,
then evaluate the selected model on test. No refitting on train+val.

```bash
python examples/linear_probe.py \
  --train-features outputs/2m/train_features.npy --train-labels data/y_train.npy \
  --val-features outputs/2m/val_features.npy --val-labels data/y_val.npy \
  --test-features outputs/2m/test_features.npy --test-labels data/y_test.npy \
  --output-dir outputs/2m/probe
```

Default C grid: `1e-5 … 1e3`; selection loss: `1 - val accuracy`.
Results: `metrics.json`, `test_predictions.npy`, `test_probabilities.npy`, `probe.npz`.
Use new output paths when rerunning.

### Manifest-based training / fine-tuning

Prepare `data/{train,val,test}.csv` with `Path,subject,label` columns.
Each Path points to a `(ROI, time)` NumPy sample. See [data format](docs/DOWNSTREAM.md).

```bash
python scripts/make_downstream_config.py --variant 2m --mode linear_probe --output outputs/configs/probe.yaml
python finetune.py --config outputs/configs/probe.yaml --output_dir outputs/downstream
```

Use `--mode full_finetune` to train encoder + head, or `--mode frozen_head`
to train only the head. Adjust task settings in the generated YAML.

## Tests

```bash
python -m pip check
python scripts/verify_environment.py --device cpu
python scripts/verify_environment.py --device cuda --pretrain-smoke
python -m pytest -q tests
```

22 tests passed on Linux; CPU/CUDA loading and synthetic downstream workflows verified.
These checks do not reproduce paper metrics. Windows/multi-GPU are not verified.

## Pretraining

Set your data paths in `configs/pretrain_example.yaml`, then run:

```bash
python pretrain.py --config configs/pretrain_example.yaml --output_dir outputs/pretrain
```

[Third-party notices](THIRD_PARTY_NOTICES.md) · Project-wide license and paper citation pending.
