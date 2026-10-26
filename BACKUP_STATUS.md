# Streamlined experiment backup

Snapshot date: 27 October 2026 (Australia/Brisbane).

## Complete SFT evaluations

All four final adapters and all 140 validation plus 145 test rows are complete for:

- bitnet2b
- hermes
- llama3b
- mistral7b
- qwen1b
- qwen7b

## Complete untreated evaluation

- hermes: 990/990 finalized rows

## Incomplete, resumable untreated evaluations

- bitnet2b: 653/990 finalized rows; atomic record ledger retained.
- llama3b: 289/990 finalized rows; atomic record ledger retained.
- mistral7b, qwen1b and qwen7b: no untreated response folders were present at snapshot time.

The single `in_progress` marker in each partial CSV is deliberately retained and will be retried; it is not considered a finalized response.

## Resume and artifact policy

- All 24 final SFT adapter archives are included with Git LFS.
- No trainer checkpoints are included because every training condition has a final adapter.
- Incomplete inference resumes from `results/<model>/untreated_hf/all/records/`.
- Immutable base-model weights, Colab caches, receiver runtime state, partial uploads, environments and secrets are excluded.

## Report-grade telemetry

`telemetry/final_run_inventory.json` indexes 74 selected structured telemetry records. The selected records retain runtime, model/cache load time, training/inference time, software versions, GPU/CUDA/driver, sampled utilisation and power, estimated energy, peak GPU memory and token totals where available. Superseded progress attempts and completed-run logs are excluded. Interruption logs for the two incomplete untreated runs are retained separately.

Manual Colab billing data: not present in the source experiment. Device energy estimates are not equivalent to Colab compute-unit charges.
