# Streamlined experiment backup

Snapshot date: 27 October 2026 (Australia/Brisbane).

## Complete SFT evaluations

All four final adapters and all 140 validation plus 145 test rows are complete for bitnet2b, hermes, llama3b, mistral7b, qwen1b and qwen7b.

## Complete untreated evaluations

- bitnet2b: 990/990 finalized rows (847 completed responses; 143 context-unsupported records)
- hermes: 990/990 finalized rows
- llama3b: 990/990 finalized rows
- mistral7b: 990/990 finalized rows
- qwen1b: 990/990 finalized rows
- qwen7b: 990/990 finalized rows

## Incomplete, resumable untreated evaluations

None in this snapshot. Untreated models without a result folder have not started.

## Resume and artifact policy

- All 24 final SFT adapter archives are included through Git LFS.
- No trainer checkpoints are included because every SFT condition has a final adapter.
- Incomplete inference resumes from `results/<model>/untreated_hf/all/records/`.
- Completed runs retain their canonical response CSVs without redundant atomic ledgers.
- Base weights, caches, runtime/tunnel state, partial uploads, environments and secrets are excluded.

## Report-grade telemetry

`telemetry/final_run_inventory.json` indexes 78 selected structured records containing runtime, model/cache load, training/inference time, software, GPU/CUDA/driver, utilisation, power, energy, peak memory and token totals where available. Incomplete-run logs are retained separately. Manual Colab billing data: not present in the source experiment.
