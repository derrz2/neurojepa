# Verification scope

Verified on 2026-09-22 using Python 3.10.20, PyTorch 2.4.1+cu121,
timm 1.0.22, and CPU execution. An isolated virtual environment reused
existing system packages; this was not a clean dependency-install test.

- Editable package installation succeeded.
- Ten automated tests passed: available checkpoint load/forward, explicit
  missing-2m error, invalid model name, missing file, checksum rejection,
  downstream-head initialization with exact encoder-feature agreement,
  rejection of missing encoder keys and incompatible tensor shapes,
  downstream release-checksum rejection, and pretraining model/optimizer/
  scheduler initialization from the example configuration.
- The included encoder has exactly 10,417,408 parameters.
- The supplied example ran with `(2, 100, 200)` input and `(2, 256)` output.
- The downstream YAML example was loaded and checked against the exact
  10m architecture; initialization from its configured safetensors succeeded.
- Outputs from the exported encoder and the original source/checkpoint
  agreed exactly on fixed synthetic inputs with shapes `(2, 100, 200)` and
  `(1, 60, 160)`: maximum absolute error was **0.0** in both cases.
- The missing 2m architecture was instantiated separately and has exactly
  2,653,568 parameters. This is **not** a pretrained-checkpoint load test.
- The exported safetensors file contains only encoder tensors and generic
  format metadata; optimizer state and private training configuration were
  not exported.

10m artifact size: 41,690,016 bytes.
SHA-256: `f86e319268970aedbf5ec0af4b27c50a5497a060a561f5b06d1961bf2ea87dae`.

These checks establish software loading and export equivalence for the
available candidate, not reproduction of scientific results. GPU, multi-GPU,
complete data pipelines, training convergence, and paper-table checkpoint
identity remain outside the verified scope. The target 2m weights are missing.
