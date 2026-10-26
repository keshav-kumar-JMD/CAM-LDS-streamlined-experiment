Final structured telemetry snapshot.

For each logical run, this directory retains the latest completed telemetry record when one exists; otherwise it retains the latest available terminal record. Superseded failed attempts and append-only worker logs are omitted for completed runs.

Each retained JSON contains wall time, model/cache load time, training or inference time, software versions, GPU/CUDA/driver metadata, sampled GPU utilisation and power, estimated energy, and peak memory. Colab account compute units are only available when manually recorded in colab_billing.csv.

Active inference is resumed from results/**/records rather than trainer checkpoints. All final SFT adapter archives exist, so no trainer checkpoint is required for the currently incomplete evaluations.