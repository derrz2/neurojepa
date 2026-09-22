# Downstream training and held-out evaluation

Start with the [README](../README.md) for loading checkpoints and extracting
features from batched NumPy arrays. This guide covers `finetune.py`, which reads
individual samples through train/validation/test CSV manifests.

## Data and preprocessing

Create three subject-disjoint CSVs: `data/train.csv`, `data/val.csv`, `data/test.csv`.
Each contains the same columns, but different participants, for example:

```csv
Path,subject,label
data/roi/sample_001.npy,participant_001,0
data/roi/sample_002.npy,participant_002,1
```

Each `Path` points to a numeric **2D** NumPy array `(ROI, time)`, not the batched
3D array used by `examples/extract_features.py`. Paths are resolved from the
working directory (run at the repository root), not relative to the CSV.
`subject` is a study-local grouping key; do not publish participant identifiers.
Classification labels must be integers `0 .. num_classes-1`.

The generated configuration selects `data.mode: task_manifest`, 100 ROIs,
`raw_series_layout: channel_time`, and 200 time points. This loader takes the
first 200 time points, zero-pads shorter series, then z-scores each ROI across
time. **This differs from the direct array extraction API**, which makes no
preprocessing changes. To compare the two paths, supply identically processed
inputs. Confirm these choices are appropriate for your study; padding is not
a substitute for a valid acquisition/preprocessing protocol.

All sessions/windows from one subject must stay in the same split. The scripts
do not independently prove that your manifests are subject-disjoint. Validation
is for model selection; test labels must not guide tuning or preprocessing.

## Generate an architecture-matched configuration

```bash
python scripts/make_downstream_config.py --variant 2m --mode linear_probe --num-classes 2 --train-manifest data/train.csv --val-manifest data/val.csv --test-manifest data/test.csv --output outputs/configs/2m_probe.yaml
python finetune.py --config outputs/configs/2m_probe.yaml --output_dir outputs/2m_manifest_probe
```

Use `--variant 10m` with a new YAML/output directory for the larger encoder.
The generator obtains all architecture fields and the checkpoint filename from
the release metadata. Do not change only the checkpoint filename while keeping
the other model's width, attention heads, or register-token settings.

The generated example uses the CSV validation/test splits **without resampling**,
one seed, no subject-level prediction pooling, and this validation grid:

```yaml
probe:
  protocol_version: user_example_fixed_split
  num_runs: 1
  eval_split_group: fixed
  eval_pooling: none
  linear_probe:
    C_grid: [0.01, 0.1, 1.0, 10.0]
    max_iter: 2000
```

The encoder is frozen. For each candidate C, the scaler and classifier fit only
training features; validation loss selects C. The selected classifier is then
evaluated on the test split, without refitting on validation data. This is a
user-facing example protocol, not a reproduction of the paper's benchmark setup.
Cached features stay in memory in this route; use the README's extraction CLI
when you want reusable `.npy` feature files.

For single-label classification, this runner's selection field `loss` is
`1 - accuracy`, not cross-entropy. `log_loss` is reported separately. Ties keep
the first C in the grid. The full-finetune runner instead uses its task loss.

## Choose which parameters to train

| Generator `--mode` | YAML settings | What is optimized |
| --- | --- | --- |
| `linear_probe` | `experiment.mode: linear_probe` | A classifier on frozen, cached features |
| `frozen_head` | `experiment.mode: full_finetune`, `training.freeze_encoder: true` | Only the neural task head |
| `full_finetune` | `experiment.mode: full_finetune`, `training.freeze_encoder: false` | Encoder and task head |

For full fine-tuning, generate a separate configuration:

```bash
python scripts/make_downstream_config.py --variant 10m --mode full_finetune --output outputs/configs/10m_finetune.yaml
python finetune.py --config outputs/configs/10m_finetune.yaml --output_dir outputs/10m_finetune
```

The default task is binary classification. Before a real run, review
`task.num_classes`, `optim.epochs`, `optim.learning_rate`, `optim.head_lr`,
`data.batch_size`, and input conventions. GPU execution is recommended for
end-to-end training. `experiment.pretrained_checkpoint` initializes the encoder;
`experiment.resume`/`--resume` instead require a compatible downstream training
checkpoint, including a task head and training state.

The full-finetune runner logs validation and test metrics during training and
selects the best checkpoint by **validation loss**. Do not use intermediate test
metrics to select an epoch or adjust settings.

## Find the results

For manifest-based linear probing, inspect:

```text
outputs/2m_manifest_probe/
  probe_summary.json
  run_00_seed_42/
    config.yaml
    checkpoints/
      checkpoint_best.pth
```

`probe_summary.json` contains each run's validation/test metrics and selected C.
The linear-probe `.pth` artifact records configuration and evaluation metadata;
it is **not** a serialized fitted sklearn estimator. For portable saved
classifier coefficients, use `examples/linear_probe.py` and its `probe.npz`.

Full fine-tuning writes downstream checkpoints containing the encoder, trained
head, and training state under the run's checkpoint directory. Treat legacy
`.pth` files as trusted-only Python checkpoint artifacts; the released encoder
loading API uses pickle-free safetensors instead.

Training outputs can contain your local paths, task labels, and split settings.
Keep them out of commits; `outputs/` is ignored. Refer to
[verification scope](VERIFICATION.md) for what has actually been tested.
