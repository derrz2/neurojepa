# Verification scope

## User workflows: version 0.1.3

Verified on 2026-09-22 in the isolated Linux reference environment described
below (same dependencies, updated source). **19 automated tests passed**, none
skipped, in 19.49 seconds. New tests cover batched file extraction and row order
for both encoders, refusal to overwrite feature files, saved linear-probe
parameters/predictions, training-only standardization, and generated architecture
configurations for both model sizes.

Additional CLI runs used synthetic, subject-disjoint fixtures (8 training,
4 validation, 4 test samples), not participant data:

- Both 2m and 10m encoders extracted ordered `.npy` features on CUDA.
- Both feature sets trained the fixed-C classifier and produced held-out
  metrics, predictions, probabilities, and numeric probe parameters.
- Both model variants completed `finetune.py` with CSV manifests, fixed
  validation/test splits, a four-value C grid, and `probe_summary.json` output.
- The generated 10m full-finetune configuration completed one epoch (two
  training batches, batch size 4, warmup disabled for this smoke test),
  validation/test evaluation, and best-checkpoint saving on CUDA.

These runs check end-to-end software execution, not useful predictive
performance, convergence, or the paper's benchmark protocol. The new file-based
extraction path was tested on CPU and CUDA; the complete manifest pipeline was
tested on CUDA. `frozen_head` mode was not separately exercised end-to-end.

## Fresh environment: version 0.1.2

Verified on 2026-09-22 using a newly created Linux x86_64 virtual environment,
without system or user site-packages. Python 3.10.19, PyTorch 2.4.1+cu121,
torchvision 0.19.1+cu121, timm 1.0.22, and NumPy 1.26.4 were used. GPU checks
used an NVIDIA GeForce RTX 3090 with driver 565.57.01 and CUDA runtime 12.1.

PyTorch came from its official CUDA wheel index. Project dependencies were
installed with the direct constraints file and the Tsinghua PyPI mirror.
The full resolved package set is in
[`linux-cu121-py310.lock.txt`](../requirements/linux-cu121-py310.lock.txt).
See [environment setup](ENVIRONMENT.md) for commands.

- `pip check`: no broken requirements.
- Environment isolation: all 18 checked dependency imports came from the new
  virtual environment; system-site and user-site packages were disabled.
- `pytest -q tests`: **14 passed**, none skipped (5.61 seconds).
- Both training entry-point `--help` commands and both feature-extraction
  example commands completed successfully in the fresh environment.
- The 80-entry dependency lock matches the installed reference environment;
  an offline pip dry-run required no package changes. The lock was captured
  after the fresh constrained installation, not installed into a second venv.
- Both released encoders passed SHA-256, strict state-dictionary, exact parameter
  count, and finite-feature checks on **CPU and CUDA**.
- Both encoders supported a synthetic downstream-head backward/AdamW update
  on CPU and CUDA, without modifying the pretrained weights.
- One synthetic pretraining step using the example configuration and eight
  views passed on CUDA: finite loss (approximately 0.8682), finite gradients,
  and a confirmed parameter update. This uses a newly initialized training
  model, not exact resumption from the encoder-only checkpoints.

## Export and compatibility checks

Earlier export-parity checks used Python 3.10.20/PyTorch 2.4.1+cu121 for the
first 10m export and Python 3.10.19/PyTorch 2.4.0+cu121 for the 2m export,
with timm 1.0.22 and CPU execution. Those earlier environments reused existing
system packages; the clean installation checks above are new in version 0.1.2.

- Editable package installation succeeded.
- Fourteen automated tests passed: both checkpoint loads/forwards,
  invalid model name, missing file, checksum rejection,
  downstream-head initialization with exact encoder-feature agreement,
  rejection of missing encoder keys and incompatible tensor shapes,
  downstream release-checksum rejection, and pretraining model/optimizer/
  scheduler initialization from the example configuration.
- The included encoders have exactly 2,654,336 and 10,417,408 parameters.
- `(2, 100, 200)` input produces `(2, 128)` and `(2, 256)` outputs respectively.
- The downstream YAML example was loaded and checked against the exact
  10m architecture; initialization from its configured safetensors succeeded.
- Outputs from each exported encoder and its original source/checkpoint
  agreed exactly on fixed synthetic inputs with shapes `(2, 100, 200)` and
  `(1, 60, 160)`: maximum absolute error was **0.0** in both cases.
- The new 2m candidate has four register tokens; it is not the missing
  one-register-token epoch-199 hypothesis described in the initial release.
- The exported safetensors files contain only encoder tensors and generic
  format metadata; optimizer state and private training configuration were
  not exported.

10m artifact size: 41,690,016 bytes.
SHA-256: `f86e319268970aedbf5ec0af4b27c50a5497a060a561f5b06d1961bf2ea87dae`.

2m artifact size: 10,637,304 bytes.
SHA-256: `2da6afd688f2327595a1c16da1a4c246a05f0f3a30eef00043615eeae0fdbe92`.

These checks establish software loading and export equivalence for the
available candidates, not reproduction of scientific results. Multi-GPU,
complete data pipelines, training convergence, Windows installation, the
separate CPU-only wheel environment, and paper-table checkpoint identity remain
outside the verified scope. See `CHECKPOINT_SELECTION.md`.
