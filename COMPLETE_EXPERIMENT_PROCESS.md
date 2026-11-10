# CAM-LDS streamlined experiment: complete process and file key

## Scope and immutable source

Run notebooks in numeric order. `00`–`03` are the frozen HF and Haiku study. `04`–`08` add deployment quantization and its matched Haiku judgment. The new Python code lives in `continuation/`; the frozen `streamlined/` package, source prompts, teacher responses, model revisions, split and policy are not changed. The experiment manifest and prompt hashes are checked at every new stage.

The input is the 990 designated GPT-5.5 teacher rows of `data/source/llm_responses.json`, **not** rows from `llm_queries.json` for model inference. That latter file is used only by notebook 00 as an identity cross-check. Prompts and teacher responses are preserved as found in the source. Scenario and step are kept together in the original scenario–step split: 705 training rows, 140 validation rows and 145 test rows. The common exact-distillation eligible subset is 604 training rows; the other training examples are excluded from training but remain in inference and audits. No validation or test row is used for training or calibration.

| Model key | Frozen HF base | Nominal size | Native inference context | SFT training context |
| --- | --- | ---: | ---: | ---: |
| `qwen1b` | Qwen2.5-1.5B-Instruct | 1.5B | 32,768 | 8,192 |
| `hermes` | OpenHermes-Qwen1.5-1.8B | 1.8B | 32,768 | 8,192 |
| `bitnet2b` | bitnet-b1.58-2B-4T-bf16 | 2B | 4,096 | 4,096 |
| `llama3b` | Llama-3.2-3B-Instruct | 3B | 131,072 | 8,192 |
| `mistral7b` | Mistral-7B-Instruct-v0.3 | 7B | 32,768 | 8,192 |
| `qwen7b` | Qwen2.5-7B-Instruct | 7B | 32,768 | 8,192 |

Notebook 01 trains four LoRA adapters per model over unchanged teacher response text. The `full` objective uses all answer tokens; `technique`, `explanation` and `joint` use their specified answer-field masks while leaving masked text visible as context. The frozen settings are three epochs, BF16, per-device batch 1, gradient accumulation 16, optimizer checkpoints every ten steps, LoRA rank 16 for the smaller models or 8 for the two 7B models, alpha twice rank and dropout 0.05. The frozen learning rate is 2×10⁻⁴ for the smaller conventional models and 1×10⁻⁴ for the 7B models and BitNet. The policy is in `configs/experiment.json`; rank and learning-rate choices are in the hash-frozen `streamlined/gpu.py`, and exact tokenizer/revision hashes are in `manifests/models.json`.

## Notebook sequence

| Notebook | Execution place | Work | Principal outputs |
| --- | --- | --- | --- |
| `00_prepare_universal_dataset.ipynb` | Laptop `.venv-local` | Extract exact teacher prompts/responses; freeze split, source hashes, model revisions and context feasibility | `data/*.csv`, `manifests/*.json`, `models/tokenizers/*` |
| `01_sft_<model>.ipynb` (six) | Colab GPU | Four exact-text LoRA adapters per model: technique, explanation, joint and full; validation and test inference | `models/adapters/<model>_<adapter>.zip`, `results/<model>/<adapter>/{validation,test}/responses.csv`, `telemetry/<model>/*` |
| `02_untreated_hf_<model>.ipynb` (six) | Colab GPU | Untreated frozen BF16 HF model on all 990 prompts | `results/<model>/untreated_hf/all/responses.csv`, telemetry |
| `03_haiku_reference_and_judge.ipynb` | Laptop `.venv-local` | 990 Haiku reference responses, uniform common judgment of HF candidates, separate Haiku self-judgment, audits and paired analysis | `results/api/*`, `results/compiled/*` |
| `04_quantized_setup.ipynb` | Laptop `.venv-local` | Start extended receiver, HTTPS tunnel and independent audible watchdog; verify 604 calibration IDs | No model outputs |
| `04_untreated_tq1_0_<model>.ipynb` (five) | Colab GPU | Condition B: untreated conventional HF base → GGUF TQ1_0 → 990 prompts | `models/quantized/*`, `results/quantized/*`, telemetry |
| `04_untreated_native_i2_s_bitnet2b.ipynb` | Colab GPU | Condition A: pinned official native BitNet I2_S GGUF → 990 prompts | Same namespaces |
| `05_sft_tq1_0_<model>.ipynb` (five) | Colab GPU | Each conventional SFT adapter → merged BF16 HF → GGUF TQ1_0 → 140 validation + 145 test prompts | Same namespaces |
| `05_sft_native_i2_s_bitnet2b.ipynb` | Colab GPU | Each BitNet SFT adapter → merged HF → native I2_S conversion, subject to a real converter/tokenizer compatibility gate → validation + test | Same namespaces; stops clearly if unsupported |
| `06_untreated_iq1_s_camlds_calibrated_<model>.ipynb` (five, optional) | Colab GPU | Condition C: untreated conventional model → GGUF IQ1_S with an importance matrix made only from the 604 eligible training prompts → validation + test | `data/calibration/*`, `models/quantized/*`, `results/quantized/*` |
| `06_sft_iq1_s_camlds_calibrated_<model>.ipynb` (five, optional) | Colab GPU | Four SFT adapters per conventional model → same IQ1_S process and same 604 prompt IDs, but a separate matrix per source checkpoint → validation + test | Same namespaces |
| `07_haiku_judge_quantized.ipynb` | Laptop `.venv-local` | Common rubric on every completed quantized candidate, one candidate per API context; optional IQ1_S uses a separate ledger | `results/api/quantized_common_cached_v1/*` and optionally `results/api/quantized_iq1_cached_v1/*` |
| `08_consolidate_results.ipynb` | Laptop `.venv-local` | Export explicitly named final files, joining validation and test per model/adapter | `results/final/*` |

`<model>` is `qwen1b`, `hermes`, `bitnet2b`, `llama3b`, `mistral7b` or `qwen7b`; conventional TQ1_0 and IQ1_S omit BitNet. There are **60 required candidate streams**: 30 HF (six untreated + 24 SFT) and 30 quantized A/B (six untreated + 24 SFT). Optional C adds **25** streams (five untreated + 20 SFT). These are stream/file counts, not judgment counts. Context-unsupported prompts remain explicit candidate records and are not sent to the judge, as in notebook 03.

## Inference and quantization controls

All conditions use the same unchanged teacher-row prompt, its frozen model-specific chat template, independent generation context per row, greedy decoding, seed 42, repetition penalty 1.0 and an output ceiling of 1,024 tokens (or the remaining native context). There is no response screening: empty text, invalid JSON, unusual stop reasons and poor answers remain saved as completed responses. A prompt with no generation allowance is recorded as `context_unsupported` rather than shortened. Format/schema fields are audits only.

### Native BitNet I2_S context-boundary reserve

On 3 November 2026, the untreated native-I2_S run had 431 locally received records: 331 completed, 62 `context_unsupported`, and 38 `retryable_error`. Every error came from the pinned BitNet backend reporting `Backend truncated the input or shifted context`; in every case, frozen input tokens plus the requested output allowance equalled the nominal 4,096-token context exactly. Nineteen of these rows failed twice, showing that an unchanged retry did not resolve the boundary condition. The corresponding untreated HF implementation had successfully generated these near-boundary rows by using the exact remaining allowance, so this is recorded as a backend-specific usable-context difference rather than a source-prompt failure.

From this cutover onward, native BitNet I2_S inference reserves **8 tokens** inside the nominal 4,096-token context. The unchanged prompt is always retained, and the allowance is `min(1024, 4096 - input_tokens - 8)`. A row is `context_unsupported` only if that calculation leaves no generation token. Each affected record stores `backend_context_reserve_tokens=8`, the actual `max_new_tokens`, and `output_allowance_reduced`; telemetry stores the same reserve. Earlier completed records remain unchanged and are skipped on resume. Retryable boundary records may be replaced by successful later attempts under this documented revision.

The eight-token value is a conservative engineering margin selected because schedule constraints did not permit a reserve-size pilot. It is 0.20% of the native context and can reduce only the maximum output allowance, by at most eight tokens, for context-bound rows; it does not shorten prompts, change weights, sampling, or screen response content. This differs operationally from HF inference, which used no reserve, and must be reported as an engine compatibility accommodation. Notebook 03 treats `context_unsupported` as a terminal, reportable coverage status and does not send such rows to Haiku; it does not delete or silently relabel them.

The previous BF16 HF run and new GGUF deployment runs use different inference engines. The new GGUF path pins the exact `llama.cpp` or `bitnet.cpp` source commit. Frozen HF token IDs define the experimental input. The worker compares GGUF and HF tokenization on pilot and longest prompts; if a GGUF pre-tokenizer segments identical rendered text differently, llama-server must detokenize the frozen HF IDs back to the exact rendered prompt before those IDs can be used. The mismatch and round-trip evidence are retained in a tokenizer audit. GGUF prompt cache is disabled, so rows cannot borrow a preceding row's KV state. Backend, context, token counts, stop reason, artifact hashes, model revision, GPU and time are recorded. Engine differences may still affect numeric outputs; comparisons should be described as **deployment-format comparisons**, not as purely weight-only changes.

- **A, BitNet native I2_S:** the official pinned GGUF is used as supplied. It has no CAM-LDS importance matrix. The BitNet SFT path converts the merged checkpoint to I2_S and must pass real export/tokenization/inference gates. No BitNet TQ1_0 or IQ1_S run is scheduled.

  **BitNet SFT multi-shard converter compatibility patch.** The four merged SFT checkpoints are approximately 4.8 GB and are saved as two safetensors shards. At pinned Microsoft BitNet commit `0b341e582afbf9e1011f24744b554c96a3477eb5`, `_get_part_names()` returns multi-shard filenames as a one-use generator while `BitnetModel.write_tensors()` deliberately makes two passes (scale discovery, then conversion). The first pass therefore exhausted the filename generator and the converter wrote a valid GGUF v3 header with 23 metadata entries but zero tensors (7,818,592 bytes). This was rejected by the artifact gate and produced no experimental responses. The deterministic local patch `multi_shard_filename_tuple_v1` changes only the safetensors/bin shard-name containers from generators to tuples, retaining the same filenames and order for both passes. It does not modify checkpoint values, I2_S quantization, prompts, sampling or inference. Every resulting artifact manifest records the patch identifier and actual tensor-type counts; zero-tensor outputs remain rejected.
- **B, TQ1_0:** direct post-training quantization of the conventional frozen base or the corresponding merged SFT checkpoint. No CAM-LDS importance matrix is used. The untreated run covers all 990 rows.
- **C, optional IQ1_S:** an importance matrix is generated separately for each conventional source checkpoint using the **same 604 frozen eligible training prompt texts**. The exact IDs, prompt hashes, matrix hash and GGUF artifact hash are recorded. The matrix differs by checkpoint because activation importance is model dependent. Both untreated and SFT IQ1_S inference cover only validation/test (285 rows), preventing calibration leakage into those comparisons. IQ1_S is an approximate one-bit deployment format; TQ1_0 is ternary. Neither is claimed to be identical to BitNet's native training quantization.

Quantized model files are uploaded to the laptop and verified before their manifests become usable. A new Colab runtime can restore those files. CUDA toolchain binaries are also saved locally by pinned commit, GPU architecture and CUDA toolkit version, so a compatible later runtime can fetch them; a different GPU architecture or toolkit gets a separate build. Source/model downloads and C++ builds are reused within a live runtime. CPU merge/export, C++ build and large model transfer may be slow. Their durations are retained in logs and artifact manifests.

If an artifact upload loses its tunnel after quantization, `artifact.pending.json` in the surviving Colab runtime allows the next preparation attempt to resume that upload without quantizing again. A destroyed Colab runtime cannot preserve its unfinished local GGUF, but already acknowledged laptop GGUFs and CUDA toolchain archives are reusable.

### Observed EOS and output-limit behaviour

The 1,024-token ceiling is a maximum allowance, not a required output length. A row recorded with `finish_reason=eos` ended because the model selected its configured end-of-sequence/end-of-generation token. A row recorded as `length` by the HF backend or `limit` by llama.cpp reached the allowance without selecting that token. The EOS token remains in the tokenizer and model metadata; a limit result means it was not selected during that generation. No artificial stop strings, response screening or early JSON-based termination are applied.

The following snapshot compares completed untreated BF16 HF results with the TQ1_0 records available on 3 November 2026. Partial TQ1_0 runs are labelled by their current row count.

| Model | Untreated HF EOS | Untreated HF limit | TQ1_0 EOS | TQ1_0 limit |
| --- | ---: | ---: | ---: | ---: |
| Hermes | 227/990 | 763/990 | 0/990 | 990/990 |
| Llama 3B | 986/990 | 4/990 | 0/990 | 990/990 |
| Mistral 7B | 716/990 | 274/990 | 0/320 completed | 320/320 completed |
| Qwen 1B | 989/990 | 1/990 | 118/990 | 872/990 |
| Qwen 7B | 990/990 | 0/990 | 0/205 completed | 205/205 completed |

This is a substantial deployment-condition difference, especially for Llama 3B and Qwen 7B, whose untreated HF models almost always selected EOS. TQ1_0 is an aggressive ternary post-training quantization and can materially change next-token probabilities; EOS may lose probability relative to ordinary tokens, and deterministic greedy decoding can then remain in repetitive or incoherent continuations until the ceiling. The neutral repetition penalty of 1.0 intentionally adds no intervention that would force termination. Qwen 1B TQ1_0 selected EOS on 118 rows, demonstrating that the GGUF inference path can detect and honour EOS rather than universally ignoring it.

This observation must not yet be reported as a purely weight-quantization causal effect. HF and GGUF use different inference engines, and each exported GGUF model's EOS/EOG metadata should be audited against its frozen HF tokenizer and generation configuration. Until that audit is complete, report the result as an observed termination difference between the untreated-HF and TQ1_0 deployment conditions, with aggressive quantization as the leading explanation and model-specific conversion metadata retained as a possible contributor.

## Checkpointing, interruptions and local CSV cost

The Colab worker saves each completed row first as an atomic local JSON record, then a background uploader sends receiver checkpoints in batches of up to five. Backpressure pauses new local saves if 25 rows await receiver acknowledgement, and normal completion forces a full upload flush. The record ID includes the experiment, model, condition, representation, artifact and calibration hashes, settings and prompt ID. A resumed run reconciles Colab-spooled and laptop records and skips terminal rows. A row whose outcome is uncertain can be retried; completed rows cannot be silently overwritten with a different response. The quantized response CSV is a materialized view of the laptop records, refreshed after 25 mutations and again at normal completion. The larger Haiku judge CSV refreshes every 100 record mutations and at cell completion; each judge record is still saved immediately. If a runtime dies before its background queue is uploaded, its atomic Colab records are replayed on a surviving runtime; if the runtime itself is discarded first, only receiver-acknowledged records are recoverable. Run the verification/snapshot cell or resume to refresh the CSV. The laptop receiver is required for Colab work; local Haiku API notebooks require no receiver or tunnel.

The 25-row snapshot reduces full-CSV rewrites and Windows file-lock exposure. Five-row asynchronous receiver calls remove most tunnel latency from the generation critical path while retaining bounded recovery exposure. Saved timing separates summed generation time, wall time, local atomic writes, receiver upload and CSV snapshot time. Independent rows are issued concurrently but remain separate llama-server requests with separate contexts, unchanged frozen prompt token IDs and unchanged decoding settings. General configured slots are Qwen 1B/Hermes 8, BitNet/Llama 3B 4, and Mistral/Qwen 7B 2. The untreated Mistral/Qwen 7B TQ1_0 continuation described below is deliberately scoped to four slots. The later L4 revision for unfinished 05 Llama 3B, Mistral 7B and Qwen 7B runs retains their general slot counts and changes only scheduling and log verbosity. A pilot or tail uses no more slots than remaining rows. The server allocates the original per-row context per slot and records both per-slot and total context. Reusing one URL/token across different model streams is safe because files and leases are stream-specific; never run the same model/condition twice simultaneously. The laptop watchdog alarms on lost Colab heartbeats independently of the notebook kernel; keep Windows awake and sound enabled.

## Quantized inference process revisions and completion at cutover

The inference process changed on **2 November 2026**. Revision 1 used one llama-server slot, synchronous receiver acknowledgement after every row and a CSV snapshot every ten mutations. Revision 2 uses the model-specific concurrent slots listed above, atomic Colab row records, background receiver uploads in batches of up to five, backpressure at 25 outstanding rows and a CSV snapshot every 25 mutations. Quantization artifacts, prompts, frozen HF token IDs, context allowance and generation settings did not change.

The following is the dated completion snapshot copied from the opening descriptions of the 04 and 05 notebooks. Counts were obtained from authoritative results/quantized record files; saved telemetry was used to corroborate completed runs. All records present at this cutover lacked the Revision 2 parallel_slots field, so they belong to Revision 1. Revision 2 starts by resuming the remaining prompt IDs in the same stream.

| Notebook / model | Required rows | Revision 1 rows saved before change | Revision 2 rows at cutover | Remaining for Revision 2 |
| --- | ---: | ---: | ---: | ---: |
| 04 untreated TQ1_0: Qwen 1B | 990 | 239 (24%) | 0 | 751 |
| 04 untreated TQ1_0: Hermes | 990 | 990 (100%) | 0 | 0 |
| 04 untreated TQ1_0: Llama 3B | 990 | 658 (66%) | 0 | 332 |
| 04 untreated TQ1_0: Mistral 7B | 990 | 252 (25%) | 0 | 738 |
| 04 untreated TQ1_0: Qwen 7B | 990 | 0 (0%) | 0 | 990 |
| 04 untreated native I2_S: BitNet 2B | 990 | 0 (0%) | 0 | 990 |
| 05 SFT TQ1_0: Qwen 1B, each of four adapters | 285 each | 0 each | 0 each | 285 each |
| 05 SFT TQ1_0: Hermes, each of four adapters | 285 each | 0 each | 0 each | 285 each |
| 05 SFT TQ1_0: Llama 3B, each of four adapters | 285 each | 0 each | 0 each | 285 each |
| 05 SFT TQ1_0: Mistral 7B, each of four adapters | 285 each | 0 each | 0 each | 285 each |
| 05 SFT TQ1_0: Qwen 7B, each of four adapters | 285 each | 0 each | 0 each | 285 each |
| 05 SFT native I2_S: BitNet 2B, each of four adapters | 285 each | 0 each | 0 each | 285 each |

The optional 06 IQ1_S notebooks had no saved inference rows at the cutover and therefore have no Revision 1 responses. Their first inference results will use Revision 2. These counts describe response inference only; an already prepared or uploaded GGUF artifact is not counted as a response row.

### Untreated Mistral/Qwen 7B TQ1_0 fixed continuation runtime (3 November 2026)

Only `04_untreated_tq1_0_mistral7b.ipynb` and `04_untreated_tq1_0_qwen7b.ipynb` use this fixed continuation configuration. BitNet, the 05 SFT notebooks and other representations retain their existing slot policy. The laptop receiver had **360/990 Mistral** and **257/990 Qwen 7B** records at the decision point. Completed records remain untouched in the same stream and are skipped on resume. Every newly generated row records `runtime_configuration_revision=untreated_tq1_0_7b_four_slots_length_scheduled_v1`, `parallel_slots`, prompt-length bucket, batch/microbatch, Flash Attention mode, CPU-thread mode and server log verbosity. Mixed runtime revisions must be disclosed in analysis.

The earlier mandatory training-only performance sweep and architecture-specific profile were retired before use for the remaining experimental rows. Its repeated server starts and long prompt evaluations were too costly, and the partial Qwen output (`4096 baseline tokens/s 0.28 error none`) established only that the original baseline completed in the up-to-4,096-token bucket; it did not establish a faster candidate. No performance-profile file is now required to resume inference.

The fixed continuation uses **one persistent llama-server**, **four independent request slots**, batch **2,048**, microbatch **512**, backend-default Flash Attention and backend-default CPU threads. Missing prompts are deterministically ordered into token-length scheduling groups of up to 2,048, 4,096, 8,192 and over 8,192 tokens, then by token count and prompt ID. These groups alter queue order only: they do not restart the server or change the prompt, per-row context, response allowance or sampling. The implementation retains the exact teacher prompts, frozen HF token IDs, separate contexts, greedy decoding, seed 42, repetition penalty 1, prompt caching disabled, natural EOS and the original **1,024-token** maximum. There is no loop detector, lower output cap, prompt truncation, KV quantization, speculative decoding or identical-prefix caching. Each completion still receives an immediate atomic local record and queued laptop checkpoint.

The four-slot choice uses the substantial unused A100 memory observed in saved telemetry while avoiding unverified changes to Flash Attention, batch/microbatch or CPU threads. It is a documented runtime revision, not a claim of guaranteed throughput improvement. Actual per-row and run telemetry remains the evidence for its effect.

### SFT TQ1_0 L4 runtime revision (4 November 2026)

This output-neutral revision applies only to `05_sft_tq1_0_llama3b.ipynb`, `05_sft_tq1_0_mistral7b.ipynb` and `05_sft_tq1_0_qwen7b.ipynb`. Hermes was already complete and Qwen 1B was already running, so their code paths and notebook documentation remain unchanged. At adoption, each affected adapter stream held **0/285** response rows.

The later single-adapter `05_sft_tq1_0_mistral7b_<adapter>.ipynb` and `05_sft_tq1_0_qwen7b_<adapter>.ipynb` split notebooks inherit this same runtime revision, stream identities and deterministic prompt ordering. They process only one adapter per Colab runtime; they do not define a new experimental condition or scheduling policy.

### Declared completion of partial 05 Mistral/Qwen 7B streams (10 November 2026)

The researcher declared all notebook 05 streams closed for collection except Mistral 7B `sft_joint`, which remained active. Three Mistral 7B streams were deliberately abandoned after attaining high coverage because the remaining long-prompt rows repeatedly exceeded the unchanged 30-minute request timeout. Their frozen final outcomes are:

| Stream | Successful responses | Saved retryable errors | No saved record | Available successful coverage |
|---|---:|---:|---:|---:|
| Mistral 7B `sft_technique` TQ1_0 | 277 | 7 | 1 | 277/285 (97.2%) |
| Mistral 7B `sft_explanation` TQ1_0 | 278 | 4 | 3 | 278/285 (97.5%) |
| Mistral 7B `sft_full` TQ1_0 | 279 | 6 | 0 | 279/285 (97.9%) |

The checkpoint records identify every Mistral `retryable_error` as `TimeoutError: timed out`; none is classified as `context_unsupported`. All affected prompts fall in the fixed `gt_8192` scheduling bucket and contain approximately 11,506–12,378 frozen input tokens. The failures therefore reflect exceptionally slow prompt evaluation/generation under the TQ1_0 llama.cpp deployment and the fixed request deadline, not prompt truncation, a native-context violation or response-quality filtering. Prompt IDs affected by a saved timeout are: technique—175, 373, 400, 571, 769, 796 and 967; explanation—175, 373, 571 and 967; full—175, 373, 571, 769, 796 and 967. Rows without a saved record are unavailable rather than assumed failures. The saved records, CSV snapshots, GGUF artifacts, telemetry and logs are retained. Comparative analysis must use and report the available-case denominator rather than describe these streams as 285/285 complete.

Qwen 7B `sft_technique`, `sft_explanation` and `sft_full` completed all 285 rows successfully. Qwen 7B `sft_joint` was declared a **terminal partial, abandoned run** at **247 successful / 285 expected (86.7%)**, with no saved retryable error and no `context_unsupported` record. The remaining 38 prompts have no saved outcome: collection stopped because no Colab GPU accelerator was available in time to resume the experiment. This is an infrastructure/availability limitation, not evidence that those prompts failed inference. The existing Qwen Joint telemetry documents the completed attempts and must be retained; downstream analysis must report 247 as its available-case denominator and must not impute the 38 unavailable outcomes.

Mistral 7B `sft_joint` is excluded from this closure statement because collection was still running when the other streams were declared complete.

The pinned llama.cpp backend continues to compile in Release mode for the allocated GPU architecture; an L4 therefore produces or restores an architecture-keyed `sm_89` toolchain. Completed GGUF artifacts and their manifests are restored from the laptop and checksum-verified instead of being rebuilt. Mistral 7B and Qwen 7B retain two slots. Batch remains 2,048, microbatch 512, Flash Attention remains backend-default and CPU threads remain backend-default. Only one inference notebook should run on each L4.

Missing rows are deterministically ordered by frozen prompt-token length using the same fixed buckets as the untreated 7B continuation, while one persistent server and a continuously filled request queue are retained. This changes completion order only. The five-row pilot retains server verbosity 2 for CUDA-offload and throughput diagnostics; the full run uses verbosity 1 while retaining errors, per-row records, stop reasons, token counts and final telemetry. Atomic Colab records, five-row asynchronous receiver batches, 25-row backpressure and resumable missing-ID selection are unchanged. New rows record `runtime_configuration_revision=sft_tq1_0_l4_length_scheduled_low_log_v1` and the server/scheduling settings. Prompts, frozen token IDs, weights, independent contexts, greedy decoding, seed, repetition penalty, natural EOS and the 1,024-token ceiling remain unchanged.

### Llama 3B SFT TQ1_0 single-slot correction (5 November 2026)

The four initial five-row adapter pilots used four slots and recorded approximately **5.94–6.05 aggregate output tokens/s**, **3.1–3.2% mean sampled GPU utilization**, and **5.62–5.63 GiB sampled peak GPU memory** on an NVIDIA L4. All 20 responses reached the unchanged 1,024-token ceiling. An earlier single-slot Llama 3B TQ1_0 server run using the same backend family recorded approximately **20.4–20.7 output tokens/s** and **36.7 prompt tokens/s**. Four-slot concurrency was therefore counterproductive for this model/backend combination.

Subsequent missing Llama 3B SFT TQ1_0 rows use one persistent server with **one slot**, batch 2,048, microbatch 512, backend-default Flash Attention and backend-default CPU threads. This also allocates only one per-row context instead of four. Pilot verbosity is 3 so explicit `offloaded N/N layers to GPU` evidence can be retained; full-run verbosity remains 1. New rows record `runtime_configuration_revision=sft_tq1_0_llama3b_single_slot_l4_v2`. Existing four-slot pilot rows are preserved rather than regenerated and remain distinguishable through their recorded configuration. This is an output-neutral runtime correction: prompts, weights, token IDs, context allowance, sampling, natural EOS and maximum output remain unchanged.

## Haiku judge and caching

Notebook 03 saves the 990 Haiku reference responses once. Notebook 07 reuses them and the **same six-dimension uniform common rubric** and judge model. The system rubric text is unchanged; it alone is marked as an ephemeral cache block. Each API request still contains one distinct candidate, its exact prompt, ground truth and corresponding Haiku reference response; no two candidates share a judgment context. Cache hits depend on provider scheduling, especially in batches. Actual `cache_creation_input_tokens` and `cache_read_input_tokens` are retained from each API response rather than assumed. This is a transport/billing optimization, not a changed experimental condition. The full raw judge output, parsed score fields, candidate audit, token usage, estimated cost and request/batch receipt remain available. The 07 required and optional ledgers are separate so optional work can be added later without changing existing request IDs.

Notebook 07 uses an explicitly documented **available-response semantic-judging policy**, consistent with notebook 03's substantive inclusion rule. Every record with `status=completed` is included verbatim, including successfully returned empty text, malformed JSON, extra prose, truncation at the output ceiling, repetition and poor answers. `missing`, `context_unsupported`, `retryable_error` and any other non-completed status are omitted from paid semantic judging because no candidate response was returned or captured; they are retained by model, condition and representation in `source_coverage.csv`, including `expected`, `judge_eligible`, `omitted_no_response` and individual status counts. They are not assigned zero scores or represented by synthetic error text. Consequently the judge request set is no longer blocked by terminal partial streams, and all comparisons must report the available-case denominator and operational coverage alongside score summaries. Section 3 freezes the exact set of currently completed candidates; responses completed only afterward cannot be silently added to that paid ledger and require a separately versioned judge run.

Notebook 07 batch-result polling is rate paced because the required run created 245 Message Batches and Anthropic enforces an organisation-level 50-requests-per-minute Message Batches API limit. The collector waits 2.6 seconds after each batch inspection, conservatively allowing for both status and result calls, and automatically waits at least 65 seconds before retrying a read-only call that returns HTTP 429. Atomic receipts and completed judgment records make such retries idempotent: rate-limit recovery neither resubmits paid requests nor changes already collected results.

Once a notebook 07 request manifest exists, every recovery run reloads its exact frozen job dictionary rather than rebuilding it from potentially newer inference files. Its original `source_coverage.csv` is likewise preserved. This permits safe polling recovery after kernel or API interruption without expanding, removing or repricing the paid request set; later candidate completions require a separately versioned judge ledger.

## File key

| File or pattern | Meaning |
| --- | --- |
| `manifests/source.json`, `manifests/experiment.json`, `manifests/models.json` | Frozen source, split/eligible IDs/code/policy hashes, and model revisions/context/tokenizer identities |
| `data/prompts.csv`, `data/teacher_responses.csv`, `data/split_manifest.csv`, `data/ground_truth.csv` | Universal prompt, teacher, split and label tables; join on prompt ID and check prompt hash |
| `data/context_feasibility.csv`, `data/training_exclusions.csv`, `data/dataset_audit.csv` | Training feasibility, exclusions and descriptive source audit |
| `data/calibration/camlds_604_train_prompts_only.{txt,json}` | Optional IQ1_S calibration text and exact training IDs/hashes |
| `models/tokenizers/<model>/` | Frozen local tokenizer needed for unchanged chat rendering and token checks |
| `models/adapters/<model>_<adapter>.zip` | Final LoRA adapter archive from 01; model-specific complete SFT artifact |
| `checkpoints/<model>/<adapter>/latest.json` and `step-*.zip` | Incomplete training resumption only; final adapters are the primary complete artifacts |
| `models/quantized/<model>/<condition>/<representation>/model.gguf` and `artifact.json` | Final local GGUF and hash, source revision, calibration and toolchain lineage |
| `models/toolchains/<backend>/<commit>/cuda_<architecture>_toolkit_<version>.zip` | Reusable compiled CUDA backend binaries; keyed by GPU architecture, toolkit and source commit |
| `results/<model>/untreated_hf/all/responses.csv` | 02 BF16 untreated 990-row output |
| `results/<model>/<adapter>/{validation,test}/responses.csv` | 01 BF16 SFT outputs; 140 and 145 rows respectively |
| `results/quantized/<model>/<condition>/<representation>/records/*.json` | Authoritative new per-row quantized checkpoints; each received on the laptop immediately |
| `results/quantized/<model>/<condition>/<representation>/responses.csv` | New quantized CSV snapshot; refreshes every 25 received mutations and at completion |
| `results/api/haiku/responses.csv` | 990 Haiku reference answers |
| `results/api/common_uniform_v2/`, `results/api/haiku_self_uniform_v2/` | Existing 03 judge ledgers, row records, raw/parsed responses, receipts and usage |
| `results/api/quantized_common_cached_v1/`, `results/api/quantized_iq1_cached_v1/` | New 07 required and optional judge ledgers, `records/`, `responses.csv`, `submissions/`, receipts, cache policy and coverage |
| `results/compiled/D_all_responses.csv`, `E_*_validation.csv`, `all_judgments.csv`, `grouped_paired_intervals.csv` | 03 compiled candidate tables, condition tables, HF judgments and paired analysis |
| `results/final/source__*.csv`, `reference__haiku__all.csv` | Final source and reference copies |
| `results/final/analysis__*.csv` | Byte-preserved completed notebook 03 compiled candidate/judge and coverage tables |
| `results/final/response__<model>__<condition>__<format>__<extent>.csv` | One candidate CSV per completed model/condition/format; SFT validation/test joined |
| `results/final/judgment__<run>__<model>__<condition>__<format>__<extent>.csv` | Judge outputs separated by run/model/condition/format; partial legacy ledgers are marked in manifest |
| `results/final/file_manifest.csv` | Final file tally, source, row counts, status coverage, completeness and SHA256 |
| `telemetry/<model>/train/<adapter>/*.json`, `telemetry/<model>/<condition>/<split>/*.json` | 01/02 runtime, GPU sample, peak memory and token/time observations |
| `telemetry/quantized/<model>/<condition>/<representation>/*.json` | 04–06 GPU samples, generation time, token totals, checkpoint/network/CSV time, artifact identity |
| `telemetry/logs/*` | Detailed worker, converter, quantizer, imatrix, server and tokenizer/debug logs |
| `telemetry/quantized/*/*/*/*_tokenizer_audit.json` | HF-versus-GGUF tokenization check and exact frozen-ID round-trip evidence |
| `telemetry/colab_billing.csv`, `telemetry/billing/*.json` | Manually recorded Colab balance deltas; concurrent sessions prevent automatic per-run billing attribution |

`results/final/file_manifest.csv` is the authoritative tally. It records any missing or nonterminal result so exported partial judge snapshots cannot be mistaken for complete coverage. API dollar estimates are based on the price snapshot in each API request manifest; actual charges should be checked against Anthropic billing. GPU power samples estimate energy, not Colab compute units. No code in 04–08 deletes the original HF responses, adapters or judge results.

## Recovery quick reference

For the revised **04 untreated Mistral/Qwen 7B** notebooks: there is no performance-profile stage. After a new runtime, rerun Sections 1–3. Rerun Section 4 only if the final GGUF artifact/manifest cannot be restored; Section 5 is an optional five-row experimental response pilot; Section 6 resumes all missing responses through the fixed four-slot persistent server; Section 7 verifies coverage and refreshes the CSV. Do not launch an old and new worker for the same stream together.

- **New Colab runtime:** restart the extended receiver/tunnel/watchdog if needed, rerun that notebook's 1–3, then 4 only if no final GGUF manifest is present, then 6. Section 5 is an optional pilot once the artifact exists.
- **Tunnel or receiver interruption:** restore the terminal services, rerun the Colab bootstrap/verification cells, then the interrupted stage. Laptop row records determine what remains. Do not run two workers for the same stream.
- **Generation interrupted after a save:** resume section 6; acknowledged rows are skipped. Run section 7 to force a final CSV snapshot.
- **Haiku API interrupted:** rerun 07 section 1, then 3. Reconcile unknown batch submissions in section 5 before any submission. Polling in 7 or 11 never creates paid requests. Preserve `submissions/` and `records/`.
- **Final export:** rerun notebook 08 section 2 after more streams or judgments complete. It reads local authoritative records and performs no GPU or paid API work.
