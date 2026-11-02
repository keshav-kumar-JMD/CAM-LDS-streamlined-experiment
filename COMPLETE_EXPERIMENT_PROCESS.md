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

The previous BF16 HF run and new GGUF deployment runs use different inference engines. The new GGUF path pins the exact `llama.cpp` or `bitnet.cpp` source commit. Frozen HF token IDs define the experimental input. The worker compares GGUF and HF tokenization on pilot and longest prompts; if a GGUF pre-tokenizer segments identical rendered text differently, llama-server must detokenize the frozen HF IDs back to the exact rendered prompt before those IDs can be used. The mismatch and round-trip evidence are retained in a tokenizer audit. GGUF prompt cache is disabled, so rows cannot borrow a preceding row's KV state. Backend, context, token counts, stop reason, artifact hashes, model revision, GPU and time are recorded. Engine differences may still affect numeric outputs; comparisons should be described as **deployment-format comparisons**, not as purely weight-only changes.

- **A, BitNet native I2_S:** the official pinned GGUF is used as supplied. It has no CAM-LDS importance matrix. The BitNet SFT path converts the merged checkpoint to I2_S and must pass real export/tokenization/inference gates. No BitNet TQ1_0 or IQ1_S run is scheduled.
- **B, TQ1_0:** direct post-training quantization of the conventional frozen base or the corresponding merged SFT checkpoint. No CAM-LDS importance matrix is used. The untreated run covers all 990 rows.
- **C, optional IQ1_S:** an importance matrix is generated separately for each conventional source checkpoint using the **same 604 frozen eligible training prompt texts**. The exact IDs, prompt hashes, matrix hash and GGUF artifact hash are recorded. The matrix differs by checkpoint because activation importance is model dependent. Both untreated and SFT IQ1_S inference cover only validation/test (285 rows), preventing calibration leakage into those comparisons. IQ1_S is an approximate one-bit deployment format; TQ1_0 is ternary. Neither is claimed to be identical to BitNet's native training quantization.

Quantized model files are uploaded to the laptop and verified before their manifests become usable. A new Colab runtime can restore those files. CUDA toolchain binaries are also saved locally by pinned commit, GPU architecture and CUDA toolkit version, so a compatible later runtime can fetch them; a different GPU architecture or toolkit gets a separate build. Source/model downloads and C++ builds are reused within a live runtime. CPU merge/export, C++ build and large model transfer may be slow. Their durations are retained in logs and artifact manifests.

If an artifact upload loses its tunnel after quantization, `artifact.pending.json` in the surviving Colab runtime allows the next preparation attempt to resume that upload without quantizing again. A destroyed Colab runtime cannot preserve its unfinished local GGUF, but already acknowledged laptop GGUFs and CUDA toolchain archives are reusable.

## Checkpointing, interruptions and local CSV cost

The Colab worker saves each completed row first as an atomic local JSON record, then a background uploader sends receiver checkpoints in batches of up to five. Backpressure pauses new local saves if 25 rows await receiver acknowledgement, and normal completion forces a full upload flush. The record ID includes the experiment, model, condition, representation, artifact and calibration hashes, settings and prompt ID. A resumed run reconciles Colab-spooled and laptop records and skips terminal rows. A row whose outcome is uncertain can be retried; completed rows cannot be silently overwritten with a different response. The quantized response CSV is a materialized view of the laptop records, refreshed after 25 mutations and again at normal completion. The larger Haiku judge CSV refreshes every 100 record mutations and at cell completion; each judge record is still saved immediately. If a runtime dies before its background queue is uploaded, its atomic Colab records are replayed on a surviving runtime; if the runtime itself is discarded first, only receiver-acknowledged records are recoverable. Run the verification/snapshot cell or resume to refresh the CSV. The laptop receiver is required for Colab work; local Haiku API notebooks require no receiver or tunnel.

The 25-row snapshot reduces full-CSV rewrites and Windows file-lock exposure. Five-row asynchronous receiver calls remove most tunnel latency from the generation critical path while retaining bounded recovery exposure. Saved timing separates summed generation time, wall time, local atomic writes, receiver upload and CSV snapshot time. Independent rows are issued concurrently but remain separate llama-server requests with separate contexts, unchanged frozen prompt token IDs and unchanged decoding settings. Configured slots are Qwen 1B/Hermes 8, BitNet/Llama 3B 4, and Mistral/Qwen 7B 2; the pilot or tail uses no more slots than remaining rows. The server allocates the original per-row context per slot and records both per-slot and total context. Reusing one URL/token across different model streams is safe because files and leases are stream-specific; never run the same model/condition twice simultaneously. The laptop watchdog alarms on lost Colab heartbeats independently of the notebook kernel; keep Windows awake and sound enabled.

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

## Haiku judge and caching

Notebook 03 saves the 990 Haiku reference responses once. Notebook 07 reuses them and the **same six-dimension uniform common rubric** and judge model. The system rubric text is unchanged; it alone is marked as an ephemeral cache block. Each API request still contains one distinct candidate, its exact prompt, ground truth and corresponding Haiku reference response; no two candidates share a judgment context. Cache hits depend on provider scheduling, especially in batches. Actual `cache_creation_input_tokens` and `cache_read_input_tokens` are retained from each API response rather than assumed. This is a transport/billing optimization, not a changed experimental condition. The full raw judge output, parsed score fields, candidate audit, token usage, estimated cost and request/batch receipt remain available. The 07 required and optional ledgers are separate so optional work can be added later without changing existing request IDs.

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

- **New Colab runtime:** restart the extended receiver/tunnel/watchdog if needed, rerun that notebook's 1–3, then 4 only if no final GGUF manifest is present, then 6. Section 5 is an optional pilot once the artifact exists.
- **Tunnel or receiver interruption:** restore the terminal services, rerun the Colab bootstrap/verification cells, then the interrupted stage. Laptop row records determine what remains. Do not run two workers for the same stream.
- **Generation interrupted after a save:** resume section 6; acknowledged rows are skipped. Run section 7 to force a final CSV snapshot.
- **Haiku API interrupted:** rerun 07 section 1, then 3. Reconcile unknown batch submissions in section 5 before any submission. Polling in 7 or 11 never creates paid requests. Preserve `submissions/` and `records/`.
- **Final export:** rerun notebook 08 section 2 after more streams or judgments complete. It reads local authoritative records and performs no GPU or paid API work.
