# Verification scope

Verified on 2026-09-22. The first 10m export was tested with Python 3.10.20
and PyTorch 2.4.1+cu121. The two-model release and 2m export were tested with
Python 3.10.19 and PyTorch 2.4.0+cu121. Both environments used timm 1.0.22
and CPU execution. Isolated virtual environments reused existing system
packages; these were not clean dependency-install tests.

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
available candidates, not reproduction of scientific results. GPU, multi-GPU,
complete data pipelines, training convergence, and paper-table checkpoint
identity remain outside the verified scope. See `CHECKPOINT_SELECTION.md`.
