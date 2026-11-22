# CAM-LDS exact-teacher experiment

This independent experiment retains historical files in their existing locations. Nothing in the old experiment is an input except the frozen scenario-step split and the original query identities used to verify mapping.

## Complete notebook and analysis order

Run notebooks from the `streamlined_experiment` project root so their relative paths resolve. Existing final outputs are preserved in `results/final/`; rerunning paid API or GPU notebooks can incur new usage. The numbered families are:

| Order | Notebook(s) | Purpose |
| --- | --- | --- |
| 00 | `notebooks/00_prepare_universal_dataset.ipynb` | Freeze the source prompts, teacher text, splits and manifests; document receiver setup. |
| 01 | Six `notebooks/01_sft_*.ipynb` files | Train four adapters per model and run BF16 validation/test inference. |
| 02 | Six `notebooks/02_untreated_hf_*.ipynb` files | Run each untreated HF model on all 990 prompts. |
| 03 | `notebooks/03_haiku_reference_and_judge.ipynb` | Collect Haiku references and Phase 1 judgments. |
| 04 | `notebooks/04_quantized_setup.ipynb`, then six `notebooks/04_untreated_*.ipynb` files | Prepare toolchains and run untreated TQ1_0 or native I2_S GGUF models. |
| 05 | Six `notebooks/05_sft_*.ipynb` files | Prepare and evaluate quantized SFT adapters. The separate Mistral 7B and Qwen 7B adapter notebooks are alternative split runners for the corresponding combined notebooks; do not run both for the same adapter concurrently. |
| 06 | Ten `notebooks/06_*iq1_s*.ipynb` files | Optional CAM-LDS-calibrated IQ1_S comparison; separate from the primary TQ1_0/I2_S study. |
| 07 | `notebooks/07_haiku_judge_quantized.ipynb` | Judge Phase 2 outputs, teacher responses and reference baselines. |
| 08 | `notebooks/08_consolidate_results.ipynb` | Export the authoritative response, judgment and manifest files to `results/final/`. |
| 09 | `notebooks/09_compute_telemetry_summary.ipynb` | Export per-invocation and per-stream compute records under `results/final/compute/`. |
| 10 | **Root** `10_analysis.ipynb` | Build the underlying analysis tables and first-pass figures in `analysis/`. |

For the **corrected figures used in the report**, run the root `10_analysis.ipynb` first so `analysis/tables/` is populated. Then, from the project root in the local environment, run `python analysis_round2/code/test_figure_qa.py` followed by `python analysis_round2/code/run_round2.py`. The latter is a script, **not another notebook**; it reads `analysis/tables/` and writes corrected figures, figure data, QA and manifests to `analysis_round2/`. Cite or inspect `analysis_round2/figures/` and its companion `figure_data/` for the latest reviewed presentation, while retaining `analysis/tables/` for calculation provenance. The duplicate `notebooks/10_analysis.ipynb` is not the documented entry point. See `analysis_round2/README.md` for its validation record.

## Start here

1. Select **`streamlined_experiment/.venv-local/Scripts/python.exe`** as the local VS Code kernel. It is Python 3.12, separate from the charts environment.
2. Open `notebooks/00_prepare_universal_dataset.ipynb`. Run sections 1–4 (already performed during implementation if the execution report says so). Section 5 documents receiver, tunnel and watchdog startup.
3. Start the independent watchdog, test the audible alert, and leave Windows awake/unmuted. Start receiver/tunnel for Colab only. Every applicable notebook cell repeats the terminal commands.
4. Run the six `01_sft_*.ipynb` notebooks on separate or successive Colab GPU runtimes. Run sections 1–6 for training and both validation/test; section 7 records optional billing observations. These train 24 independent adapters. All four are prespecified on both validation and test; no selection or tuning follows validation.
5. Run the six `02_untreated_hf_*.ipynb` notebooks, sections 1–4, using the identical frozen HF base revisions and BF16 backend.
6. Run `03_haiku_reference_and_judge.ipynb` **locally** in numbered order. Section 10 explicitly retries only confirmed failed/missing requests. Repeated polling never submits new work. Section 11 exports the new charts input and group-bootstrap comparisons.

Model/condition runs may operate in parallel across distinct runtimes. Do not run the same model/condition concurrently. Each notebook has separate file namespaces; a receiver lease rejects overlapping live owners.

## Preparation results

Preparation sections 1–4 have been executed and their outputs saved in notebook 00. All 990 extracted prompts and teacher response substrings match the source. The frozen split contains 705 training, 140 validation and 145 test rows. Complete-example context checks retain 604 common training examples and report 101 exclusions. Four teacher responses fail the schema audit; their original text is retained.

BitNet cannot fit 143 original prompts within its native context while leaving room for a response. These receive explicit `context_unsupported` records rather than shortened prompts or invented answers. Another 82 BitNet prompts fit with less than the full 1,024-token generation allowance. The other five models fit all prompts with that allowance. This missing-response coverage must accompany comparisons.

All 14 notebooks passed structure and Python syntax checks. Ten automated tests passed, including checkpoint transfer, duplicate-worker rejection, response preservation, and ambiguous API request handling. API tests use mocks; messages saying “Paid batch requests” in the test report do not represent actual API calls. GPU training/inference and paid Haiku requests have not been executed. The Windows sound mechanism was tested; confirm that it is audible from your listening location before starting a long run. Validation evidence is in `manifests/implementation_validation.json` and `manifests/test_results.txt`.

## Scientific contract

- Source: designated GPT-5.5 teacher physical rows 1981–2970 of the original `llm_responses.json`. All 990 identities are checked against `llm_queries.json`. Model identity, source hash, prompt and teacher hashes are recorded.
- Prompts are never cleaned, truncated, repaired or rewritten. Annotation warnings are recorded without changing inputs.
- Native chat role/control-token wrappers are rendered around unique placeholders, then the exact source strings are inserted. This prevents native template `trim` filters from silently removing source whitespace. The same construction is used for training and untreated/SFT inference.
- Teacher responses are structured JSON objects in the source. Their exact lexical JSON substring is retained, including field order, whitespace and escaped characters. This is exact preservation of the stored source, not a claim to recover the teacher API's original serialization.
- The source's train/validation/test group assignment is reused (705/140/145 before context exclusions). Scenario-step groups never cross partitions.
- Training eligibility uses the intersection of complete prompt + teacher response + chat ending fit across all six tokenizers. Training limits: 8,192 tokens for conventional models, 4,096 for BitNet. These are fixed experimental limits, independent of allocated GPU; they are not all models' native context lengths.
- All context-excluded rows remain in datasets and inference/evaluation coverage. Inference uses each model's verified native context, up to a 1,024-token output ceiling. A prompt that leaves no output space receives `context_unsupported`; a smaller output allowance is explicitly recorded. Prompt text stays unchanged.
- Conditional adapters mask loss on values in the same original JSON response: technique, explanation, joint, full. Structure/ending are supervised. Field-boundary tokens are masked. All teacher fields remain visible to later tokens; these are objective ablations, not information-isolation ablations. No ground-truth insertion or target correction occurs.
- BF16 base checkpoints are identical between untreated and SFT. BitNet's BF16 master checkpoint retains intrinsic online ternary-weight/INT8-activation forward computation (`AutoBitLinear`); both arms use this same native path. This is not a dense BF16 BitNet ablation or optimized packed-weight deployment. Removing its intrinsic quantization would alter the model. Separate deployment quantization and 2×2 comparisons are deferred.
- The checkpoint's tokenizer regex is retained (`fix_mistral_regex=False`). Transformers 4.57.6's local-config heuristic incorrectly labels BitNet's Llama-3 tokenizer as needing a Mistral regex repair when its config lacks a Transformers version; no such rewrite is applied.
- Precision, batch size 1, gradient accumulation 16, seed 42, three epochs and effective generation settings are frozen. BF16-unsupported GPUs fail clearly rather than silently changing precision. Smaller BF16 GPUs can still run out of memory; resume on a larger GPU rather than change the protocol.
- Each model response and each judge response is retained, including empty text, malformed JSON, refusals and length limits. Process failures are retryable; poor content is never a retry criterion. Audit columns are not filters.
- Each judge context contains exactly one candidate. General rubric is common to every condition. Three condition-specific diagnostic rubrics each have six dimensions. Haiku self-assessment is standalone on validation/test once per prompt. It is not independent human validation.
- Exact source prompts are used for the Haiku reference; Haiku provider-default temperature is recorded as a deviation from greedy HF decoding. No invented temperature=0 equivalence. Model alias, wire parameters, usage and receipts are saved.
- Test is a prespecified comparative replication, not used for model selection. Raw training-split judgments are descriptive and are not evidence of held-out performance. Grouped CIs are exploratory and not multiplicity-adjusted. Reliability repeats and independent judges are deferred.

## Resume and durable storage

Returned rows are stored as atomic JSON records and materialized CSVs. A response is accepted regardless of quality. A completed response with different content cannot overwrite an existing completed record. The local record ledger rebuilds CSVs after interruption.

Colab sends a row to the receiver and waits for durable acknowledgement before advancing. On transfer failure, work stops and the local watchdog alarms. The unsent row is spooled in Colab and resent if that runtime remains accessible. If an unacknowledged row is lost with the entire runtime, that row must be generated again; no system can recover output that was never durably saved.

Training checkpoints contain adapter, optimizer, scheduler, RNG and trainer state. The checkpoint archive is sent in resumable 4 MiB chunks with size/SHA256 verification; the latest pointer changes only after acknowledgement. A restart replays at most work since the latest acknowledged checkpoint. GPU changes are recorded and can affect numerical results. Training software-version mismatches halt with a recovery message.

Haiku uses an intent ledger, remote batch IDs and exact custom-ID reconciliation. Unknown submissions are held, not automatically resubmitted. Synchronous pilot requests can be irrecoverably ambiguous if the connection dies before a receipt; investigate before an explicit retry. Received malformed judgments remain completed with missing derived fields, not rejected. Old attempts/receipts stay auditable.

## Alerts

`python -m streamlined.monitor --root .` runs independently of VS Code/Jupyter. Worker heartbeats arrive every 30 seconds. Missing heartbeat threshold: 150 seconds. Progress thresholds: 30 minutes for model inference; 60 minutes for training; 25 hours for API batch processing (the API can legitimately queue long jobs). These thresholds are code settings, not model output constraints.

Alerts repeat every 15 seconds until acknowledged. Local jobs write heartbeat files directly and do not need a tunnel. Colab workers use the HTTPS receiver; runtime loss, a dead worker, or tunnel/network failure causes heartbeat loss. A live heartbeat with no stage progress triggers the separate progress threshold. Completion sounds once. The monitor cannot sound if Windows sleeps, loses power, is muted, or the watchdog itself exits. Test the sound in the actual listening setup.

## Credentials and model caches

Parent `secret/Claude_API_Key.txt` and `secret/HF_Token.txt` are read without printing. The API key is never sent to Colab. The HF read token is served only by an authenticated receiver route and held in worker memory/environment. Bootstrap bundles exclude secrets and original source JSON. Receiver tokens and connection files live in ignored `runtime/`; never commit them or notebook outputs containing them.

Base model snapshots, tokenizer caches, adapter archives and trainer checkpoints are local under `models/` and `checkpoints/`, ignored by Git because they are large. **Tokenizers and base model weights can be downloaded/cached locally once.** Colab installs Python GPU dependencies on a new runtime, but performs no custom CUDA compilation. Source code, notebooks, data exports, manifests and results can be backed up separately. No automatic Git push is performed.

## Telemetry

HF runs record device/driver/CUDA, library versions, wall/load time, per-row tokens/latency, PyTorch peak memory and sampled device utilization/power. NVML measurements are device-wide; utilization and energy integrals are estimates, not billed compute units. Per-row peak resets mean run-wide memory must also be interpreted with the row maxima. Colab billing requires manually entered account balances; concurrent sessions prevent precise attribution.

API records include input/output/cache tokens, batch receipts, stop reasons and estimated USD. Batch elapsed time includes queueing and is not inference latency. GPU/memory data is unavailable for the hosted API. Pricing snapshot: 2026-10-24, [Haiku documentation](https://platform.claude.com/docs/en/models/haiku-5-5/overview). Batch lifecycle follows the [Messages Batch API](https://platform.claude.com/docs/en/build-with-claude/batch-processing). Trainer checkpoint behaviour follows [Transformers](https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/trainer).

## Outputs

- A: `data/prompts.csv`; universal exact responses: `data/teacher_responses.csv`.
- B: `data/split_manifest.csv`; C: `data/teacher_responses_{train,validation,test}.csv`.
- `data/ground_truth.csv`, `dataset_audit.csv`, `context_feasibility.csv`, `training_exclusions.csv`.
- D: `results/compiled/D_all_responses.csv`; E: three `E_*_validation.csv` files.
- Final charts input: `results/compiled/all_judgments.csv`.
- Paired descriptive uncertainty: `results/compiled/grouped_paired_intervals.csv`.
- Raw candidate streams: `results/<model>/<condition>/<split>/responses.csv`.
- API ledgers: `results/api/<run>/`; receipts and intents are retained.
- Metrics: `telemetry/`, including manual billing observations.

Missing values stay missing; `failure_labels=[]` means no failures observed, while null/unparseable means unavailable. `status=completed` means a response was received, not correct JSON, good quality, or correct classification.
