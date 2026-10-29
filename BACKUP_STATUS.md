# Backup status

This repository is the portable backup of the streamlined CAM-LDS experiment. It includes source code, notebooks, frozen manifests and tokenizers, prepared datasets, response records, telemetry, and completed LoRA adapter archives.

## Included completed SFT adapters

- BitNet 2B: technique, explanation, joint and full
- Hermes: technique, explanation, joint and full
- Llama 3B: technique, explanation, joint and full
- Mistral 7B: technique, explanation, joint and full
- Qwen 1B: technique, explanation, joint and full
- Qwen 7B: technique, explanation, joint and full

The final adapter ZIP files are stored with Git LFS. Base-model weights are excluded because the frozen Hugging Face repository and revision are recorded and can be downloaded again.

All six model families have complete technique, explanation, joint, and full
adapter archives: 24 adapters in total.

## Completed API evaluation

- Haiku reference responses: 990 completed
- Uniform common judgments: 12,449 completed
- Haiku self-evaluations: 285 completed

## Excluded transient data

- Python virtual environments
- Downloaded base-model caches
- Superseded trainer checkpoints for models that already have final adapters
- Receiver tokens, heartbeat files, tunnel state and partial uploads
- Claude and Hugging Face credentials

The original parent `secret/` directory is outside this repository and must never be committed.
