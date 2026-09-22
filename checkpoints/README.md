# Encoder weights

`neurojepa-10m-epoch299.safetensors` contains the pretrained encoder only:
10,417,408 parameters, 300 completed epochs, 74,100 optimizer steps.
The projector, optimizer, serialized Python configuration, and training paths
are excluded. This is suitable for feature extraction and downstream
initialization, not exact pretraining resumption.

`neurojepa-2m-epoch63.safetensors` contains a 2,654,336-parameter encoder,
64 completed epochs, and 15,808 optimizer steps. It has four register
tokens and matches 0.128 EFLOPs under the documented analytic proxy.
It is a newly located candidate, not a recovered copy of the earlier
one-register-token epoch-199 hypothesis. See `docs/CHECKPOINT_SELECTION.md`.

Checksums and exact architecture settings are in `neurojepa/configs/`.
Correspondence of these candidates to the paper's rounded EFLOPs table is
not yet independently verified. Do not treat the compute values as a
checkpoint identifier.
