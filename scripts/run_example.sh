#!/usr/bin/env bash
set -euo pipefail

# Supply one or more GPU processes as appropriate for your environment.
NUM_GPUS="${NUM_GPUS:-1}"

torchrun --nproc_per_node="${NUM_GPUS}" pretrain.py \
  --config configs/pretrain_example.yaml \
  --output_dir outputs/pretrain
