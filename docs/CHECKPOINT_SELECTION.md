# Checkpoint selection and uncertainty

The `2m` release now contains the full-data p4 **epoch 63** candidate:
2,654,336 encoder parameters, 64 completed epochs, 15,808 optimizer steps,
and four register tokens. Its downstream manifest references the same
checkpoint and a preserved summary contains 24 rows across 12 tasks and
LP/MLP modes. A separate tuned-linear evaluation also references it.

Its analytic pretraining estimate is **0.12804869594913176 EFLOPs**, rounding
to 0.128. The convention is the same as used to reconstruct the 10m
candidate's 1.97 EFLOPs: three times estimated forward FLOPs, actual
projector widths `[2048, 2048, 512]`, and mean channel-crop scales. For 2m,
the global/local mean scales are 0.9/0.45, global batch size is 256, and
each epoch contains 247 optimizer steps. This is an estimate, not profiling.

The earlier preparation release hypothesized a different small-model run:
one register token, 2,653,568 parameters, epoch 199. Its weights were missing.
That hypothesis was not a unique identification of the table. The new 2m
artifact is a different, explicitly documented candidate, not a renamed
copy of that missing checkpoint.

Rounding does not uniquely identify a model: other data fractions and
epochs also approach 0.128, and fixed-output crop accounting can make
epoch 59 of the full-data run round to 0.128. Epoch 63 is selected here as
the closest full-data candidate under the convention shared with the 10m
estimate, with matching physical weights and downstream records.

These facts establish availability, compute plausibility, and downstream
use. They do not independently prove which row was copied into the final
paper. That last step requires the original paper-result selection record
or comparison with its downstream metrics. Private training paths and
participant information are intentionally excluded from this release.
