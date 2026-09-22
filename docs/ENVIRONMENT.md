# Environment setup

Run every command from the repository root. Keep `checkpoints/` alongside the
source checkout: the editable installation loads the two bundled safetensors
files without downloading weights or requiring a model-hosting account.

## Recommended: Linux, Python 3.10, CUDA 12.1

Use 64-bit Python 3.10. GPU execution requires an NVIDIA GPU and a compatible
driver (`nvidia-smi` should work). The PyTorch wheel supplies the CUDA runtime;
a separate CUDA toolkit is not needed for these commands. Version-matched wheels
follow the [official PyTorch instructions](https://docs.pytorch.org/get-started/previous-versions/).

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -c requirements/constraints-py310.txt -e ".[train,test]"
python -m pip check
```

Do not use `--system-site-packages`: unrelated packages can mask missing
dependencies. Direct dependencies are pinned by the constraints file. The full
resolved set is recorded in
[`requirements/linux-cu121-py310.lock.txt`](../requirements/linux-cu121-py310.lock.txt).
For that exact Linux dependency set, use these installation commands after
creating and activating the environment:

```bash
python -m pip install -r requirements/linux-cu121-py310.lock.txt --extra-index-url https://download.pytorch.org/whl/cu121
python -m pip install --no-deps -e .
python -m pip check
```

The lock is platform-specific; do not use it for CPU-only wheels or Windows.

If PyPI downloads are slow, the reference installation used the
[Tsinghua PyPI mirror](https://mirrors.tuna.tsinghua.edu.cn/help/pypi/) for the
project dependencies after installing PyTorch from its official CUDA index:

```bash
python -m pip install -c requirements/constraints-py310.txt -e ".[train,test]" --index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

This is a per-command option, not a change to your global pip configuration.

## Verify the installation

```bash
python scripts/verify_environment.py --device cpu
python scripts/verify_environment.py --device cuda  # NVIDIA GPU required
python scripts/verify_environment.py --device cuda --pretrain-smoke
python -m pytest -q tests
python pretrain.py --help
python finetune.py --help
python examples/extract_features.py --variant 2m --device cpu
python examples/extract_features.py --variant 10m --device cpu
```

The verification script imports training and inference dependencies, checks both
checkpoint hashes, strictly loads both encoders, checks finite features, and
performs a synthetic downstream-head backward/optimizer step for each model.
Expected parameter counts are 2,654,336 and 10,417,408. Input `(2, 100, 200)`
produces features `(2, 128)` and `(2, 256)`. Failures cause a nonzero exit.
Tests also cover downstream initialization and pretraining model, optimizer,
and scheduler construction. No participant data is needed. See the actual
tested platform and limits in the [README](../README.md#6-check-your-installation).
The optional `--pretrain-smoke` also runs one pretraining forward/backward and
optimizer step with the example configuration and eight synthetic views. It
initializes a new training model; it does not modify the released checkpoints.

## CPU-only alternative

Create a separate Python 3.10 environment, replacing the PyTorch command with:

```bash
python -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -c requirements/constraints-py310.txt -e ".[train,test]"
python -m pip check
python scripts/verify_environment.py --device cpu
python -m pytest -q tests
```

CPU execution is tested using the CUDA-enabled wheel. The separate CPU-only
wheel installation is an alternative, not an independently validated
environment. CPU execution does not require an NVIDIA driver.

## Conda and Windows

Conda can provide Python instead of `venv`:

```bash
conda env create -f environment.yml
conda activate neurojepa
```

Then run the recommended pip installation and verification commands above.
`environment.yml` bootstraps Python and pip only. Fresh-install verification
used `venv`, not the Conda solver.

For Windows PowerShell, create and activate the environment with:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Use the same pip commands and direct constraints, not the Linux lock. Windows
has not been independently verified. Linux is recommended for training and
distributed execution. If activation is blocked, invoke
`.\.venv\Scripts\python.exe` directly instead of `python`.

## Troubleshooting and data-dependent features

- No matching torch wheel: use Python 3.10, not the newest Python release.
- CUDA unavailable: check `nvidia-smi`, the installed PyTorch CUDA build, and
  whether `CUDA_VISIBLE_DEVICES` hides GPUs. Try `--device cpu` to isolate this.
- Missing checkpoint or checksum mismatch: obtain the complete repository;
  do not rename or edit the safetensors files to bypass validation.
- Lance datasets and external experiment tracking are optional integrations,
  disabled in the examples and not covered by this environment. Default
  examples require your own ROI time series and split files.
- Passing software checks does not reproduce paper metrics. Full data
  preparation, multi-GPU training, and evaluation require your data and
  task-specific configuration.
