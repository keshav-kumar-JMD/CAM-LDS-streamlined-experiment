# Streamlined experiment backup

Snapshot date: 26 October 2026 (Australia/Brisbane).

## Complete SFT evaluations

The following models have all four final adapters plus 140 validation and 145 test rows for every adapter:

- Llama 3B
- Mistral 7B
- Qwen 1B
- Qwen 7B

## Partial evaluations

- Hermes: all four adapters and all four 145-row test runs are complete; validation responses are absent.
- BitNet 2B: all four adapters and all validation runs are complete; technique and explanation tests are complete, joint test was active, and full test was pending at snapshot time.
- Untreated BitNet and untreated Hermes inference were active at snapshot time.

## Resume state

All 24 final adapter archives are included through Git LFS. No trainer checkpoint is included because no remaining work requires training: unfinished work is inference and resumes from the atomic per-row ledgers under `results/`.

## Telemetry policy

`telemetry/final_run_inventory.json` indexes the latest completed structured telemetry record for each available logical run. These records retain total runtime, model/cache load time, training or inference time, software versions, GPU/CUDA/driver details, sampled utilisation and power, estimated energy, and peak GPU memory. Superseded failed-attempt telemetry and completed-run append-only logs are excluded. Available logs for interrupted active runs are retained in `telemetry/in_progress_logs/`.

Colab compute-unit balances are only available if they were manually entered in notebook section 7; no billing file was present at this snapshot.

## Exclusions

- Downloaded base-model weights (frozen repository and commit remain recorded)
- Receiver tokens and tunnel state
- Partial upload files
- Superseded training checkpoints
- Python environments
- Claude and Hugging Face credentials