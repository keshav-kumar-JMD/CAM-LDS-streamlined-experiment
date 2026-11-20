# Final CSV file guide

## Authority and regeneration

`results/final/` is a reporting namespace rebuilt by notebooks 08 and 09. Immutable per-row JSON remains authoritative for raw text, nested metadata and recovery. Manifests are authoritative for expected membership, completeness and hashes. CSVs are sorted, flattened reporting views; if a CSV conflicts with JSON, verify the record/manifests and regenerate the CSV rather than editing it manually.

Notebook 09 section 2 refreshes both response/judgment exports and compute exports. It removes obsolete flat naming revisions before writing canonical names. Missing IDs are never synthesized. A partial stream CSV contains only available authoritative records and its incomplete coverage is recorded in `08__manifest__files.csv`.

## Notebook-prefix convention

Every filename begins with the notebook that produced the underlying data:

| Prefix | Origin |
| --- | --- |
| `00__` | Frozen prompts, teacher responses, ground truth and split tables |
| `01__` | BF16 SFT training/inference streams |
| `02__` | Untreated BF16 inference streams |
| `03__` | Haiku references, BF16 judgments, self-judgments and compiled Phase 1 tables |
| `04__` | Untreated TQ1_0/native-I2_S streams |
| `05__` | SFT TQ1_0/native-I2_S streams |
| `06__` | Optional IQ1_S streams |
| `07__` | Quantized Haiku judgments |
| `08__` | Cross-stream final-results manifest |
| `09__` | Cross-notebook compute tables, dictionary and compute manifest |
| `10` analysis namespace | Derived response-quality, reference-ladder and compute analyses under `analysis/`; these files do not modify `results/final/` inputs |

Per-stream compute files use `01__`–`06__` according to their source stream. Consolidated compute tables use `09__` because they combine multiple notebook phases.

## Root result files

```text
00__source__<table>.csv
01__response__<model>__sft_<adapter>__hf__validation_test.csv
02__response__<model>__untreated_hf__all.csv
03__reference__haiku__all.csv
03__analysis__<table>.csv
03__judgment__legacy_common__<model>__<condition>__hf__<extent>.csv
03__judgment__haiku_self__validation_test.csv
04__response__<model>__untreated__<representation>__all.csv
05__response__<model>__sft_<adapter>__<representation>__validation_test.csv
06__response__<model>__<condition>__iq1_s_camlds_calibrated__validation_test.csv
07__judgment__quantized_common__<model>__<condition>__<representation>__<extent>.csv
07__judgment__teacher_common__gpt55_teacher__teacher__source__validation_test.csv
07__judgment__reference_common__baseline_<null|modal>__baseline__reference__validation_test.csv
08__manifest__files.csv
```

`extent=all` denotes the 990-row source universe. `validation_test` denotes the joined 140- and 145-row legacy partitions, treated operationally as one 285-row evaluation pool while retaining each row's `split` value.

## Compute layout

```text
compute/
├── invocations/
│   ├── 09__compute__invocation_inventory.csv
│   └── <01--06>__compute__invocations__<stream>__<phase>.csv
├── rows/
│   ├── 09__compute__row_performance.csv
│   ├── 09__compute__coverage_and_retries.csv
│   └── <01--06>__compute__rows__<stream>.csv
├── stream_summary/
│   ├── 09__compute__stream_summary.csv
│   ├── 09__compute__model_resource_summary.csv
│   ├── 09__compute__api_usage_summary.csv
│   ├── 09__compute__data_dictionary.csv
│   └── <01--06>__compute__stream_summary__<stream>__<phase>.csv
├── 09__compute__aggregation_manifest.json
└── 09__manifest__compute_files.csv
```

## Common identifiers and statuses

`prompt_id` is the primary cross-experiment key. Use `model`, `condition`, `representation` and `split` to identify a candidate stream, and verify `prompt_sha256` when joining prompt-bearing tables.

- `completed`: a response was returned and preserved, regardless of correctness or JSON formatting.
- `context_unsupported`: the unchanged prompt left no defensible generation allowance; no candidate response exists.
- `retryable_error`: an operational failure such as timeout; error and attempt history are retained.
- `missing`: no authoritative row record exists for the expected ID.

Blank fields mean unavailable, not applicable, not returned or not instrumented in that runtime revision; they do not mean zero. Time is in seconds, sizes are bytes, utilization is percent, energy proxies are watt-hours and API cost estimates are USD.

## Response and judgment columns

Response CSVs preserve source identity, exact prompt/hash, model/condition/representation, raw response/hash, status, attempt count/history, token counts, output allowance, finish reason, elapsed time, runtime revision, deterministic JSON/schema audits and relevant artifact/backend metadata. Empty, malformed, repetitive or ceiling-terminated completed responses are retained.

Judgment CSVs additionally preserve the anonymous candidate case, Haiku reference where applicable, ground truth, raw judge text, API receipt/usage, parse diagnostics, six substantive rubric scores plus judge-derived `confidence_label_agreement`, similarity fields, confidence fields, extracted predictions, failure labels, reasons and null reasons. The six substantive dimensions are technique identification, prediction-evidence support, explanation grounding, prediction-reason alignment, confidence-evidence support and instruction adherence. Parseable but schema-invalid judge output is retained rather than coerced.

## Compute aggregation

Compute summaries group by exact `(model, condition, representation, phase)`. Retries and resumes are summed only within that exact stream/phase. Training, preparation/upload and inference remain separate. Pilots are reported separately. GPU-session hours are resource time, not calendar duration. Per-row elapsed times can overlap under parallel slots. Peaks use maxima; utilization is time-weighted. Whole-device NVML measurements are not process-isolated, and sampled energy remains an audit proxy rather than a primary energy result.

## Notebook 10 analysis namespace

Notebook 10 reads the canonical CSV exports above and writes derived files outside `results/final/`:

```text
analysis/
├── tables/                    # Analysis-ready estimates and audits
│   ├── figure_data/           # Exact rows plotted in each figure
│   ├── figure_manifest.csv    # Figure-to-data/source/function mapping
│   └── analysis_output_catalog.csv
├── figures/                   # Rendered main-text and supplementary figures
└── analysis_manifest.json     # Input hashes, seed, labels and analysis settings
```

The analysis tables are derived outputs, not new observations. Their authority is subordinate to the canonical `results/final/` CSVs and ultimately to the immutable row records and manifests. Regenerate them through `notebooks/10_analysis.ipynb`; do not hand-edit them. `analysis/analysis_manifest.json` records the exact source paths and hashes, bootstrap seed, replicate count, canonical stream labels and pre-specified non-inferiority margin. `analysis/tables/analysis_output_catalog.csv` inventories generated analytical files and hashes. `analysis/tables/figure_manifest.csv` maps each figure to its companion figure-data CSV, source tables and key.

### Analysis populations and terminology

- `all` means the 990-row source universe where a stream was designed for all prompts.
- `validation_test` means the joined 140 validation and 145 test rows. Notebook 10 treats these legacy partitions as one 285-row evaluation pool while retaining `split` for audit.
- Cross-treatment comparisons use exact common prompt IDs. Partial streams therefore contribute only rows observed in every member of that contrast.
- `intended_rows` or `pool_n` is the target population before score eligibility filtering.
- `candidate_rows` is the number of candidate records available to the judge export.
- `completed_candidate_responses` and `completed_judgments` record operational completion before dimension-specific score eligibility.
- `eligible_n`, `scalar_in_range_scores` or `n` is the denominator actually used for the stated dimension/statistic.
- `unavailable_n` is retained missing analytical support, not a score of zero.
- `independent_groups` is the number of scenario-step clusters contributing to an estimate.
- `BF16 HF` denotes Hugging Face inference. `one-bit frontier` is a thesis-level reporting group containing TQ1_0 and native BitNet I2_S, while `exact_representation` retains their technical distinction.
- Reference streams (`gpt55_teacher`, `baseline_null`, `baseline_modal`) are never classified as one of the six model families or included in representation-group summaries.

Notebook 10 does not newly parse candidate text or recover additional semantics from malformed JSON. A judge dimension is eligible only when its canonical judgment CSV already contains a scalar score from 0 to 4. Existing extracted predictions and deterministic audits are preserved only as secondary, coverage-limited evidence.

### Common statistical columns

The following names recur across analysis tables:

| Column | Meaning |
| --- | --- |
| `dimension`, `dimension_short` | Full rubric field and compact display label; dimensions remain separate and are never combined into a headline score |
| `mean`, `median`, `q1`, `q3` | Descriptive score summaries on the eligible 0–4 judge scores |
| `low95`, `high95`, `mean_low95`, `mean_high95` | 95% percentile interval from 10,000 scenario-step cluster-bootstrap replicates unless the table states otherwise |
| `proportion_ge3`, `proportion_ge_3` | Eligible-score proportion rated 3 or 4 |
| `proportion_ge_3_low95`, `proportion_ge_3_high95`, `ge3_low95`, `ge3_high95` | Cluster-bootstrap interval for that proportion |
| `mean_difference` | Mean paired row-level score difference in the named contrast; positive favours the first/named treatment |
| `improved_rate`, `unchanged_rate`, `worsened_rate` | Paired proportions with positive, zero and negative score change |
| `probability_superiority_with_ties` | `P(A>B) + 0.5P(A=B)` on matched rows |
| `p_value`, `p_adjusted_bh` | Exploratory paired test and Benjamini–Hochberg-adjusted value within the specified multiplicity family |
| `common_support_n`, `common_response_ids` | Number/list of prompt IDs available in every stream required by a contrast |
| `bootstrap_replicates`, `bootstrap_cluster` | Interval replication count and resampling unit |
| `sigma_g`, `mde` | Observed scenario-step effect dispersion and minimum detectable effect under the documented design approximation |
| `delta`, `groups_needed_delta_0_3`, `noninferior_delta_0_3` | Pre-specified 0.3-point margin, estimated cluster requirement and descriptive non-inferiority flag |

Intervals describe sampling variation across the observed scenario-step groups. They do not account for judge-model uncertainty, alternative training runs, model-seed variation or a different source corpus.

## Notebook 10 response-analysis table categories

### Score availability and overall performance

Files: `judge_score_availability.csv`, `judge_score_availability_bounds.csv`, `model_scores_by_dimension.csv`, `model_score_distributions.csv`, and `judge_scale_use.csv`.

These tables show, by exact model/treatment/representation and rubric dimension, how many candidate responses, judgments and scalar scores were available; their 0–4 distribution; mean/median/IQR; proportion scoring at least 3; and cluster-bootstrap intervals. Availability tables are essential denominators and must accompany score comparisons. `judge_scale_use.csv` describes how Haiku used the scale; it is a judge-behaviour diagnostic rather than candidate performance by itself.

### Paired treatment and adapter comparisons

Files: `treatment_contrasts.csv`, `treatment_transition_counts.csv`, `common_support_diagnostics.csv`, `adapter_specificity_profiles.csv`, `runtime_revision_sensitivity.csv`, and `scenario_step_variant_sensitivity.csv`.

These tables compare only identical prompt IDs. They report paired score changes, win/tie/loss-style transition counts, probability of superiority with ties, uncertainty and common-support loss. They cover BF16 SFT versus BF16 untreated, one-bit SFT versus one-bit untreated, BF16 versus one-bit deployment stacks and conditional adapters. A deployment-stack contrast includes both a weight-representation and inference-backend change and must not be described as a pure quantization causal effect.

### ATT&CK distribution and class-level analysis

Files: `source_attack_distribution.csv`, `tactic_distribution.csv`, `ground_truth_multiplicity.csv`, `attack_class_scores.csv`, `attack_class_macro_micro.csv`, `attack_class_treatment_effects.csv`, and `rare_attack_class_uncertainty.csv`.

`source_attack_distribution.csv` and `tactic_distribution.csv` describe the original 990 rows and 285-row evaluation pool. `attack_class_scores.csv` reports judge-score difficulty for rows containing a technique class. `attack_class_macro_micro.csv` distinguishes row-weighted estimates from class-macro estimates, so frequent classes do not silently define both summaries. Sparse classes should be interpreted with their counts and intervals.

### Judge-mediated ranked-technique metrics

Files: `judge_mediated_rank_metrics_by_row.csv`, `judge_mediated_rank_metric_coverage.csv`, `judge_mediated_rank_metrics_by_stream.csv`, `judge_mediated_rank_metric_contrasts.csv`, `judge_mediated_parent_rank_metrics_by_stream.csv`, `judge_mediated_parent_rank_metric_contrasts.csv`, and `judge_mediated_technique_recall.csv`.

These files compare the ordered ATT&CK IDs already returned in the canonical `judge_extracted_predictions` field with the source `ground_truth` IDs. They do not parse candidate response text. Matching is normalized exact ATT&CK-ID equality; parent and sub-technique IDs are distinct. Repeated predictions are removed after their first occurrence and only the first ten distinct predictions are used.

- `judge_mediated_rank_metrics_by_row.csv` is the audit table. It retains prompt/stream identity, serialized prediction and truth lists, extraction state, counts, and exact-ID plus parent-normalized `Hit@1` through `Hit@10`, fixed-denominator precision and row recall at `k=1,3,5,10`.
- `judge_mediated_rank_metric_coverage.csv` reports intended rows, observed judgments, and counts/rates for `nonempty_list`, `empty_list`, `missing_or_null`, and `malformed_non_list`. An explicit `[]` is a valid extraction and is scored zero; missing/null/malformed extraction is unavailable in the primary estimate.
- `judge_mediated_rank_metrics_by_stream.csv` reports all model/representation/condition estimates with 10,000-replicate scenario-step cluster-bootstrap intervals. It contains row-weighted Hit and Precision, macro Recall, micro Recall, and a sensitivity estimate that treats unavailable extraction as zero.
- `judge_mediated_rank_metric_contrasts.csv` contains exact-common-prompt paired SFT-minus-untreated and one-bit-minus-BF16 contrasts with clustered intervals and support counts.
- `judge_mediated_parent_rank_metrics_by_stream.csv` and `judge_mediated_parent_rank_metric_contrasts.csv` repeat the stream and contrast calculations after collapsing valid sub-techniques `Txxxx.xxx` to their parent `Txxxx`. These are sensitivity outputs, not primary accuracy tables.
- `judge_mediated_technique_recall.csv` is a populated backward-compatible view of the recall estimates; the richer stream table is authoritative for the full ranked-metric family.

`Hit@k` indicates whether at least one ground-truth technique appears within the first `k` predictions. `Precision@k` uses the fixed denominator `k`, so unfilled ranks count as non-relevant. Macro `Recall@k` averages row recall; micro `Recall@k` divides all matched ground-truth memberships by all ground-truth memberships. These are judge-mediated diagnostics: they measure candidate performance as interpreted by Haiku and remain dependent on extraction quality. They are not strict candidate-JSON accuracy or human-validated labels.

Exact-ID matching is authoritative: parent and sub-technique IDs are distinct. In the parent-normalized sensitivity files, valid sub-techniques are collapsed in both prediction and truth lists and duplicates introduced by that collapse are removed in first-appearance order. Invalid IDs remain non-matches. Parent-normalized estimates reveal sensitivity to ATT&CK hierarchy but must not be interpreted as sub-technique correctness or merged with exact estimates.

### Training-subset representativeness audit

Files: `training_subset_representativeness_rows.csv`, `training_subset_model_feasibility.csv`, `training_subset_length_comparison.csv`, `training_subset_scenario_comparison.csv`, and `training_subset_group_coverage.csv`.

- `training_subset_representativeness_rows.csv` maps every one of the 705 original training rows to `retained` or `excluded`, preserving prompt ID/hash, scenario, variant, step, scenario-step group and raw character/word/line length. Full prompt text is not duplicated in this derived table.
- `training_subset_model_feasibility.csv` reports each model's native and training context, number of complete examples that did not fit, share of the 101 universal exclusions and non-fit required-token range.
- `training_subset_length_comparison.csv` compares the 604 retained and 101 excluded rows using raw text measures and prompt/teacher/complete-example token counts under every model tokenizer. `standardized_mean_difference` is `(excluded mean - retained mean) / pooled standard deviation`; it describes magnitude rather than a causal or population-inference test.
- `training_subset_scenario_comparison.csv` reports retained, excluded and total rows and exclusion proportions by scenario.
- `training_subset_group_coverage.csv` reports retained/excluded row counts for each scenario-step group and classifies it as `all_retained`, `mixed` or `all_excluded`.

These files audit the implemented training population. They do not alter SFT, impute excluded examples or imply that the 604 rows are representative of the original 705. Figure 18F visualizes the same source calculations.

### Unseen-technique and split-balance audit

Files: `split_population_summary.csv`, `split_scenario_balance.csv`, `split_label_multiplicity.csv`, `split_prompt_length_comparison.csv`, `evaluation_unseen_techniques.csv`, and `evaluation_unseen_technique_rows.csv`.

- `split_population_summary.csv` reports rows, distinct `prompt_sha256` values, duplicate rows, scenario-step groups, scenarios, distinct exact technique IDs, label multiplicity and prompt-character summaries for original train, feasible train, validation, test and combined evaluation.
- `split_scenario_balance.csv` and `split_label_multiplicity.csv` retain raw counts and within-population proportions, preventing unequal split sizes from being compared using counts alone.
- `split_prompt_length_comparison.csv` gives raw character/word/line and each model tokenizer's prompt-token summaries for all five populations.
- `evaluation_unseen_techniques.csv` reports exact ATT&CK-ID frequency in the feasible training set and evaluation partitions. `unseen_in_feasible_training=True` means the ID appears in evaluation but zero times in the 604 SFT rows.
- `evaluation_unseen_technique_rows.csv` identifies every evaluation row containing at least one such ID and preserves its complete ground-truth list and unseen subset.

Distinct prompts are exact hash-distinct texts. Unseen labels describe supervised-training exposure only; they do not imply that the pretrained model had never encountered the technique. Figure 18G visualizes scenario, multiplicity and raw prompt-length balance.

### Related dimensions, failures and judge diagnostics

Files: `judge_dimension_relationships.csv`, `confidence_field_availability.csv`, `failure_rates.csv`, `failure_risk_differences.csv`, `failure_label_dictionary.csv`, `score_failure_label_coherence.csv`, `judge_confidence_rule_audit.csv`, `judge_output_consistency_audit.csv`, `judge_duplicate_case_consistency.csv`, `judge_reference_anchor_diagnostics.csv`, `model_item_difficulty_concordance.csv`, `shared_and_model_specific_hard_cases.csv`, and `qualitative_audit_cases.csv`.

Dimension-relationship tables preserve paired score fields rather than averaging different rubric constructs. Failure tables contain both raw counts and normalized rates; use counts to describe burden and rates/risk differences for comparisons with unequal denominators. Confidence and reference-anchor tables are internal consistency and possible self-preference diagnostics: the same Haiku pass produced the warranted label and agreement assessment. They do not validate judge reliability. There was no multi-judge panel, repeated independent judging design or human-coded validation subset.

### Repeated prompts, lexical analysis and long tails

Files: `repeated_prompt_consistency.csv`, `prompt_length_and_long_tail.csv`, and `token_ceiling_and_stop_reasons.csv`.

These report exact-response repetition, technique-set consistency where already available, lexical similarity measures, prompt-length buckets, operational completion, latency, output-token distributions, natural stopping and 1,024-token-ceiling rates. Lexical similarity measures surface-form change and are not correctness measures. In `token_ceiling_and_stop_reasons.csv`, EOS and ceiling rates use completed responses as their denominator; `context_unsupported` and `retryable_error` remain separate counts. Thus a zero-token unsupported row cannot be misclassified as a ceiling hit.

### Haiku self-evaluation

File: `haiku_self_evaluation_summary.csv`.

This table summarizes the separately judged Haiku reference responses using the same rubric. It is a circular descriptive anchor and judge-bias diagnostic, not independent evidence of Haiku correctness or a validated upper bound.

## Notebook 10 reference and distillation tables

### `reference_ladder.csv`

One row represents one stream × rubric dimension. Important columns are `model`, `condition`, `representation_group`, `exact_representation`, `stream_display_name`, `pool_n`, `eligible_n`, `unavailable_n`, `mean`, its interval, and `proportion_ge_3` with its interval. It places the null baseline, training-only modal-technique baseline, GPT-5.5 teacher and four core streams per model on the same 285-row pool. The baselines are weak reference anchors, not validated labels.

### `retention_index.csv`

One row represents one model × dimension. `retained_gap` is the one-bit SFT score minus the modal baseline; `deployment_loss` is BF16 SFT minus one-bit SFT; and `bf16_sft_gain_over_modal` is the BF16 SFT gap above the modal baseline. `retention_index` expresses the retained share of that BF16 SFT gain when estimable. Read `retention_estimable`, `n`, `independent_groups` and the interval before interpreting the ratio, especially when the denominator is small.

### `size_contrasts.csv`

`record_type` separates the primary paired Qwen2.5 7B-minus-1.5B family contrast from the secondary six-model descriptive scatter. The paired records include matched support, distributional change, interval, MDE and superiority. Scatter records include parameter count, `log10_parameter_count`, score and retention. No regression p-value across six heterogeneous models is supported.

### `mde_and_noninferiority.csv`

Each row describes one pre-defined contrast/dimension with observed cluster dispersion, contributing rows/groups, MDE, the fixed `delta=0.3` margin, estimated groups required and a descriptive non-inferiority flag. It is a design-sensitivity table, not proof of equivalence.

### `distillation_fidelity.csv`

Each row is an SFT-minus-GPT-5.5-teacher paired contrast for one model, adapter, representation group and rubric dimension on common evaluation rows. Negative values mean the candidate was scored below the teacher. This is descriptive fidelity to the distillation target, not a human-validated quality difference.

## Notebook 10 compute-reporting and compute-analysis tables

Notebook 10 reconciles notebook 09 exports with invocation telemetry, row records, training logs and artifact manifests. It does not reconstruct telemetry that was never saved. The key reporting tables are:

| File | Unit and purpose |
| --- | --- |
| `compute_invocation_audit.csv` | One saved invocation; exposes phase, completion state, duration, duplicate/retry classification and metric observed/applicable flags |
| `compute_source_reconciliation.csv` | Source inventory and whether each compute input contributed to the analytical ledger |
| `compute_stream_resource_summary.csv` | One exact stream; accumulated training/preparation/inference session time, overhead components, GPU samples, coverage, tokens and throughput |
| `compute_phase_resource_summary.csv` | Model/phase roll-up that keeps training, artifact preparation and inference distinct |
| `compute_sft_training_report.csv` | Training invocations, steps, recorded loss history, GPU-runtime session hours, utilization, memory and adapter size |
| `training_loss_history.csv` | Saved step/epoch/loss/learning-rate observations and source log |
| `compute_quantization_report.csv`, `quantized_artifact_inventory.csv` | Preparation/upload burden, artifact bytes/hash, tensor distribution, retries, hardware and representation lineage |
| `compute_inference_report.csv` | Inference coverage, session time, tokens, latency/throughput, GPU utilization/memory and runtime revision |
| `compute_api_report.csv` | Haiku reference, judging and self-evaluation request/token/cache/cost summaries |

Resource-analysis tables include `compute_size_efficiency.csv`, `deployment_stack_comparisons.csv`, `sft_quantization_efficiency.csv`, `time_allocation_and_overhead.csv`, `runtime_configuration_analysis.csv`, `compression_performance_frontier.csv`, and `preparation_amortization.csv`. These are descriptive analyses of the recorded deployments. Hardware/backend comparability flags must be respected; a different GPU, backend or runtime revision prevents a clean causal speed comparison.

Runtime reconciliation adds four tables without replacing the observed telemetry ledger:

| File | Meaning |
| --- | --- |
| `runtime_reconstruction_row_audit.csv` | One terminal response record with elapsed request time, recorded/effective slots, timeout attempts, unrecoverable prior-attempt count and inferred timestamp interval |
| `runtime_reconciliation_by_stream.csv` | Exact-stream observed telemetry, slot-adjusted reconstruction, timestamp-union sensitivity, completeness checks, selected low/high inference range and accounting classification |
| `runtime_reconstruction_validation.csv` | Telemetry-complete streams used to compare reconstructed request/inference time with observed generation and invocation timers |
| `runtime_reconciliation_project_total.csv` | Captured total, captured inference/non-inference portions, reconciled inference range and hybrid project range |

`accounting_method` is one of `observed_complete`, `hybrid_gap_filled`, `reconstructed_only` or `insufficient_for_reconstruction`. Reconstructed values cover request processing and known 1,800-second timeouts only. They do not estimate missing model loading, restoration, artifact transfer, idle allocation, GPU utilization, memory or energy. `timestamp_capacity_check_pass=false` means receiver-derived intervals implied concurrency beyond configured slot capacity, so the timestamp union is sensitivity evidence rather than a reliable request timeline.

For compute fields:

- `total_gpu_runtime_session_hours` is allocated invocation wall time divided by 3,600, not active floating-point compute.
- `training_session_hours`, `preparation_session_hours` and `inference_session_hours` are kept separate and sum retries/resumes only within their exact stream/phase.
- `model_load_seconds`, `restore_download_seconds`, `preparation_upload_seconds`, `receiver_seconds`, `local_checkpoint_seconds` and `csv_snapshot_seconds` are operational components only where instrumented. Blank does not imply zero.
- `sampled_peak_gpu_memory_bytes` is the largest observed sample; a peak between samples may be missed.
- `time_weighted_gpu_utilization_pct` is whole-device sampled utilization. The experiment used one active model process per allocated GPU, but it remains a device-level measure.
- `completed_rows_per_inference_session_hour` and token-throughput fields include the observed workload mix and should be read with prompt/output length and coverage.
- `estimated_energy_wh` is retained as an audit proxy and is not used for primary energy claims.
- Concurrent model streams ran on separate runtimes. Their resource hours may be summed as consumption, but not interpreted as sequential calendar elapsed time.

## Figure-data CSVs and auditability

Every Notebook 10 figure has a companion CSV under `analysis/tables/figure_data/`, conventionally named `<figure_id>__<short_name>.csv`. It contains the exact filtered and aggregated values plotted, including model/stream display names, representation group, condition, dimension, denominator, point estimate, interval and panel/order fields where applicable. These files are the preferred way to audit a visual value; do not estimate values from image pixels.

`figure_manifest.csv` provides:

| Column | Meaning |
| --- | --- |
| `figure_id` | Stable figure key used by the specification and filenames |
| `figure_file` | Rendered image path |
| `figure_data_csv` | Exact plotted-data path |
| `key` | Legend/colour/name key used by the figure |
| `source_tables` | Upstream analytical tables used to create the companion data |

Main-text figures emphasize the four core streams: BF16 untreated, BF16 full SFT, one-bit-frontier untreated and one-bit-frontier full SFT. Conditional-adapter and high-density figures use supplementary `S` identifiers. TQ1_0 and I2_S may appear together in the thesis-level one-bit-frontier group, but exact representation columns remain available for technical interpretation.

## Recommended join and interpretation order

1. Start from `08__manifest__files.csv` to establish intended and observed stream coverage.
2. Join source identity using `prompt_id`; verify `prompt_sha256` where available.
3. Use judgment exports for raw judge evidence and Notebook 10 availability tables for dimension-specific denominators.
4. For treatment contrasts, use only the common-support rows recorded in `common_support_diagnostics.csv`.
5. Read the relevant analysis table, then use `figure_manifest.csv` and its companion figure-data CSV to audit a plotted value.
6. For compute results, inspect `compute_invocation_audit.csv` before a stream summary when retries, resumes or failed starts materially contributed.
7. Preserve representation/backend caveats: BF16-versus-one-bit comparisons describe the observed deployment stacks, not weight format alone.

The full methodological rationale, protocol history and limitations are documented in [`../../COMPLETE_EXPERIMENT_PROCESS.md`](../../COMPLETE_EXPERIMENT_PROCESS.md). The exact analysis definitions and figure interpretation guide are in [`../../analysis/NOTEBOOK_10_ANALYSIS_SPECIFICATION.md`](../../analysis/NOTEBOOK_10_ANALYSIS_SPECIFICATION.md).
# Notebook 10 v8 corrected-analysis outputs

The following files are especially important for Chapter 5 and supersede earlier placeholder or incomplete analysis views:

- `analysis/tables/difference_in_differences.csv`: exact-common-support four-cell contrast measuring whether the SFT-minus-untreated gain changes after the one-bit deployment stack.
- `analysis/tables/attack_class_treatment_effects.csv`: paired effects by exact ATT&CK class, model, adapter, representation group, and judge dimension; `sufficiently_represented` means at least 15 matched rows.
- `analysis/tables/failure_label_dictionary.csv`: every raw judge label, its broad category, and the ordered lexical mapping rule. Raw labels are never deleted. Repetition/truncation/length covers `repet|repeat|loop|degenera`; `degenerate_or_incoherent_output` covers `garbl|gibber|incoheren|nonsens|corrupt`.
- `analysis/tables/failure_rates.csv`: row-level broad-category counts and rates by exact stream.
- `analysis/tables/failure_risk_differences.csv`: paired adapter-minus-untreated category-risk differences and cluster-bootstrap intervals.
- `analysis/tables/failure_profile_similarity.csv`: pairwise model similarity of broad failure-rate profiles, separated by representation and treatment.
- `analysis/tables/failure_category_cooccurrence.csv`: broad-category combinations and row counts.
- `analysis/tables/haiku_self_evaluation_summary.csv`: canonical scores from Haiku’s separate self-evaluation pass; these are circular descriptive diagnostics.
- `analysis/tables/retention_index.csv`: component gaps for all rows, but a retention ratio only when the BF16 SFT gain over modal is positive with a positive 95% lower bound. Read `retention_estimable` and `retention_estimability_reason` before using the ratio.
- `analysis/tables/model_item_difficulty_concordance.csv`: includes estimability-status columns. Null correlations caused by constant scores and null overlaps caused by complete floor saturation are intentional.
- `analysis/tables/compression_performance_frontier.csv`: `figure35_memory_bytes` is the sampled whole-device peak used by Figure 35 for every backend, and `figure35_memory_source` states its scope. `population_peak_allocated_bytes` remains the process-specific PyTorch measure where available and is legitimately unavailable/zero for llama.cpp and BitNet.cpp; it is not plotted as zero.
- `analysis/tables/figure_data/04X_bitnet_template_example.csv`: four reproducibly selected prompt/response panels (`prompt_P1`, `bf16_P1`, `i2s_P1`, `i2s_P2`) with source identities and hashes, verbatim head/tail excerpts, omitted-character counts, stop metadata and canonical judge fields. `natural_bf16_condition_dropped` records whether the predeclared natural-ending preference had to be relaxed.
- `analysis/tables/bitnet_template_prevalence.csv`: completed-response denominator, frozen-regex match count/rate, distinct 100-character prefix count and positive-score-register overlap for untreated/full I2_S and untreated BF16 control streams. `source_file_sha256` and `regex` make the calculation auditable; a low rate must not be rewritten as “almost all.”
- `analysis/tables/judge_mediated_rank_metrics.csv`: Hit@k/precision/recall outputs for model streams plus GPT-5.5 teacher, modal baseline, and null baseline.
- `analysis/tables/sft_quantization_efficiency.csv`: judge contrasts joined to reconciled stream time and GPU comparability fields.
- `analysis/tables/preparation_amortization.csv`: same-GPU-gated break-even results. The audited point-time basis is recorded per side; reconstructed bounds and `speedup_direction_robust_to_reconciled_range` show sensitivity to telemetry gaps. Paired median row latency is a separate diagnostic.

Figure numbers are not embedded in plot titles. The filename prefix and `analysis/tables/figure_manifest.csv` remain the authoritative figure key.

## Positive one-bit rows and training-label exposure

- `analysis/tables/one_bit_positive_score_row_register.csv`: exhaustive row-level register of completed one-bit-frontier judgments with any canonical score above zero. It contains all canonical scores, prompt and stream identifiers, ground truth, judge-extracted predictions/failure labels, response hash, original response character count, and a 600-character response excerpt. `excerpt_truncated` refers only to the displayed excerpt; it does not mean the stored candidate response was truncated.
- `analysis/tables/evaluation_training_label_exposure_rows.csv`: one row per validation/test prompt. `training_label_exposure` distinguishes all-seen, mixed, and all-unseen exact ground-truth IDs relative to the 604-row feasible training subset.
- `analysis/tables/scores_by_training_label_exposure.csv`: descriptive judge-score summaries and cluster-bootstrap intervals by exact stream, dimension, and exposure stratum. `eligible_score_rows` is the available-score denominator; `intended_stratum_rows` is the number of candidate rows in that stream/stratum before score availability.
- `analysis/tables/sft_effects_by_training_label_exposure.csv`: paired SFT-minus-untreated score differences within exposure strata. Use `common_support_n` as the denominator. Positive values favour SFT; intervals crossing zero indicate uncertain direction.

These files rely on canonical judge scores. They do not parse candidate prose and do not discard malformed candidate JSON.

## Population column used by Notebook 10

Notebook 10 comparison tables use `population`:

- `all_990`: full available untreated universe; retained as the untreated full-universe/sensitivity analysis. Adapter/reference streams have no training rows and therefore still contain only their designed evaluation records.
- `evaluation_285`: exact validation-plus-test prompt IDs; primary for untreated-versus-SFT, BF16-versus-one-bit, teacher/baseline, cross-model treatment panels, and non-inferiority comparisons.

Always filter `population == evaluation_285` for a like-for-like treatment figure. Use `all_990` for untreated-only full-universe reporting. `common_support_n`, `intended_rows`, or `denominator` remains the exact available/scored denominator within that population.

Compute tables retain full-stream session totals but add population-specific row-record fields prefixed `population_`, including completed rows, request-work seconds, slot-adjusted request seconds, token totals, median row latency, peak allocated/reserved bytes, and rates per slot-adjusted request hour. These fields exclude preparation/upload/idle time and must not be called GPU-runtime session hours.
### Full-universe MDE, size and floor fields

`analysis/tables/mde_and_noninferiority.csv` uses `population` to distinguish the primary 285 pool from the untreated 990 sensitivity analysis. Important fields are:

- `noninferior_delta_0_3`: raw margin result;
- `bf16_reference_mean`: paired BF16 mean for deployment contrasts;
- `modal_baseline_mean`, `null_baseline_mean`: fixed 285-row trivial-reference anchors;
- `reference_at_or_below_modal_baseline`, `reference_at_or_below_null_baseline`;
- `reference_floor_flag`: BF16 is no higher than the largest available trivial anchor **or its mean is no higher than the fixed 0.3 margin**;
- `reference_at_or_below_margin`, `baseline_floor_flag`: distinguish the two reasons for the combined floor flag;
- `minimum_noninferiority_n`, `support_sufficient_for_noninferiority`: require at least 15 paired scored rows for an interpretable verdict;
- `noninferior_above_reference_floor`: raw non-inferiority, adequate support and not floor-limited;
- `noninferiority_interpretation`: reader-facing classification;
- `floor_anchor_population`: provenance of the fixed anchor.

`analysis/tables/size_contrasts.csv` includes `all_990` only for untreated Qwen2.5 7B-minus-1.5B BF16 and one-bit contrasts. Other size rows remain `evaluation_285`.

Figures with `_990_` in their filename are untreated-only full-universe sensitivity figures. Their figure-data CSVs should contain only `population=all_990`.

`model_item_difficulty_concordance.csv` includes `low_score_rate_model_a`, `low_score_rate_model_b` and `floor_saturation_threshold`. When both rates are at least 0.95, `low_score_jaccard` is intentionally null and `low_score_overlap_estimability` records floor saturation; a near-perfect overlap created by near-universal low scores is not interpreted as model agreement.
