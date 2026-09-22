# Encoder weights

`neurojepa-10m-epoch299.safetensors` contains the pretrained encoder only:
10,417,408 parameters, 300 completed epochs, 74,100 optimizer steps.
The projector, optimizer, serialized Python configuration, and training paths
are excluded. This is suitable for feature extraction and downstream
initialization, not exact pretraining resumption.

The 2,653,568-parameter target (epoch 199) has not been recovered. Its
architecture is documented, but no substitute weights are included.

Checksums and exact architecture settings are in `neurojepa/configs/`.
Correspondence of these candidates to the paper's rounded EFLOPs table is
not yet independently verified. Do not treat the compute values as a
checkpoint identifier.
