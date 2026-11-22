# Notebook 10 analysis specification

## Status and purpose

This document is a pre-analysis and implementation specification for notebook 10. It does not contain calculated results. Notebook 10 has three distinct aims:

1. analyse candidate-model responses and Haiku judgments;
2. report collected compute and API usage without mixing inference with operational work; and
3. analyse relationships between compute, model size, representation, treatment and response outcomes.

The notebook must preserve the multidimensional judge rubric. It must **never create a combined, weighted or headline score** from the judge dimensions. Technique identification, evidence support, explanation grounding, prediction-reason alignment, confidence evidence support, instruction adherence and confidence-label agreement measure different constructs and remain separate throughout.

The analysis is exploratory. Haiku is one proprietary judge, Haiku also generated the comparison reference, and Haiku self-evaluation is circular by design. Judge-derived scores therefore support structured exploratory comparison, not a claim of human-validated accuracy.

### How to read the methods below

Every analysis subsection follows the same order:

1. **Plain-language purpose:** what question the analysis answers and why it is useful.
2. **Population:** exactly which rows can be compared.
3. **Calculation:** the reproducible statistic or model.
4. **Representation:** the table or figure a reader will see.
5. **Interpretation boundary:** what the result does not establish.

This ordering is deliberate. The analysis should remain understandable to readers who are not statistical specialists, while the technical detail beneath it makes every result reproducible and auditable.

### Primary evidence hierarchy

The Haiku judge is the **primary analytical instrument** for response quality. This choice is deliberate: many small and quantized candidates do not reliably emit strict JSON, yet their free-form text can still contain an interpretable technique prediction, evidence, explanation or confidence statement. Notebook 10 therefore analyses the already-exported judge scores and judge labels rather than trying to reconstruct meaning with a new deterministic parser.

The evidence hierarchy is:

1. already-materialized Haiku judge scores, failure labels and similarity fields;
2. source metadata and ground-truth fields supplied by the frozen CAM-LDS inputs;
3. existing deterministic audit columns exported by notebooks 01--07, used only as secondary format/operational diagnostics; and
4. raw candidate and raw judge text, preserved for audit and qualitative examples but not reparsed by notebook 10.

Notebook 10 must not introduce a new JSON, regular-expression or semantic parser for candidate responses or raw judge responses. It must not recover additional scores from JSON-looking strings embedded in CSV cells. A judge-score cell is analytically eligible only when the canonical export already presents it as a scalar numeric value from 0 to 4. Non-scalar, null and malformed score cells remain retained in the audit data, are counted as unavailable for that dimension, and are never silently coerced. This conservative rule may reduce the usable judge-score denominator, but avoids model- and representation-dependent recovery rules.

There is one narrow distinction between **structural decoding** and **semantic parsing**:

- permitted: CSV decoding and schema-defined deserialization of a canonical field already exported as a list or object, such as `judge_failure_labels`, when the field contract in `CSV_FILE_GUIDE.md` explicitly declares that structure;
- prohibited: interpreting `candidate_response_raw`, `judge_response_raw`, free-text reasons or embedded objects to recover new predictions, scores, confidence labels or semantic categories.

Any structurally decoded field retains its original serialized value and a schema-validation flag. Failure to decode the declared structure makes that field unavailable; it never triggers a permissive fallback parser.

## Governing comparison, naming and audit rules

### Canonical model and stream name guide

This table is the single authoritative naming guide for Notebook 10. Figure titles, panel headings, legends, table display fields, captions and figure-data CSVs must use the **canonical display name** exactly as written. Internal keys and filename slugs remain machine-readable and must not appear as reader-facing model names.

| Internal key / filename slug | Frozen source repository | Nominal size | Canonical display name |
|---|---|---:|---|
| `qwen1b` | `Qwen/Qwen2.5-1.5B-Instruct` | 1.5B | `Qwen2.5 1.5B` |
| `hermes` | `vicgalle/OpenHermes-Qwen1.5-1.8B` | 1.8B | `OpenHermes 1.8B` |
| `bitnet2b` | `microsoft/bitnet-b1.58-2B-4T-bf16` | 2B | `BitNet b1.58 2B` |
| `llama3b` | `meta-llama/Llama-3.2-3B-Instruct` | 3B | `Llama 3.2 3B` |
| `mistral7b` | `mistralai/Mistral-7B-Instruct-v0.3` | 7B | `Mistral 7B v0.3` |
| `qwen7b` | `Qwen/Qwen2.5-7B-Instruct` | 7B | `Qwen2.5 7B` |
| `gpt55_teacher` | frozen `openai/gpt-5.5-20260423` rows in `llm_responses.json` | proprietary | `GPT-5.5 teacher | reference | distillation target` |
| `baseline_null` | deterministic empty completed response | not applicable | `Trivial baseline | reference | null response` |
| `baseline_modal` | deterministic `T1105` prediction; modal exact technique over 604 frozen training-only feasible rows | not applicable | `Trivial baseline | reference | modal technique` |

The last three rows are **reference streams**, not model-family or representation-group members. They are never counted among the six model families and never enter `BF16 HF` or `one-bit frontier` aggregates. In reference-ladder outputs only, the core untreated label is written `<Model> | BF16 HF | reference | untreated`; every pre-existing figure and table retains `<Model> | BF16 HF | untreated`.

Use this fixed model order in every six-model figure and table:

1. `Qwen2.5 1.5B`;
2. `OpenHermes 1.8B`;
3. `BitNet b1.58 2B`;
4. `Llama 3.2 3B`;
5. `Mistral 7B v0.3`; and
6. `Qwen2.5 7B`.

The four core stream labels use the exact pattern below, replacing `<Model>` with the canonical display name:

| Core stream | Required reader-facing label | Required short panel header |
|---|---|---|
| BF16 untreated | `<Model> | BF16 HF | untreated` | `BF16 HF | untreated` |
| BF16 SFT-full | `<Model> | BF16 HF | SFT-full` | `BF16 HF | SFT-full` |
| One-bit untreated, conventional models | `<Model> | one-bit frontier [TQ1_0 GGUF] | untreated` | `one-bit | untreated` |
| One-bit SFT-full, conventional models | `<Model> | one-bit frontier [TQ1_0 GGUF] | SFT-full` | `one-bit | SFT-full` |
| One-bit untreated, BitNet | `BitNet b1.58 2B | one-bit frontier [native I2_S GGUF] | untreated` | `one-bit | untreated` |
| One-bit SFT-full, BitNet | `BitNet b1.58 2B | one-bit frontier [native I2_S GGUF] | SFT-full` | `one-bit | SFT-full` |

Conditional-adapter treatment labels are exactly `SFT-technique`, `SFT-explanation` and `SFT-joint`. They replace `SFT-full` in the same representation-aware pattern. Do not abbreviate these treatments to `technique`, `explanation`, `joint` or `full` in reader-facing output.

Reader-facing figures may use the short panel header only when the model's canonical display name is already the panel title. Captions, companion CSVs and table rows always carry the full reader-facing label. Filenames continue to use stable lowercase slugs such as `qwen1b`, `sft_full`, `tq1_0` and `native_i2_s`; filenames are provenance identifiers, not display labels.

The implementation defines one shared model-label dictionary and one shared stream-label function. No plotting function may construct a model or stream label independently. `analysis/analysis_manifest.json` records the label dictionary, model order and stream-label function version so regenerated figures cannot silently change terminology.

### Thesis-level representation groups

The thesis studies the **one-bit frontier as a deployment category**. Therefore every thesis-level table and figure that compares BF16 with the one-bit frontier must include all six model families:

- Qwen2.5 1.5B, OpenHermes 1.8B, Llama 3.2 3B, Mistral 7B v0.3 and Qwen2.5 7B use `TQ1_0 GGUF` in the one-bit-frontier phase;
- BitNet b1.58 2B uses its native `I2_S GGUF` representation; and
- the one-bit-frontier category is the union of those exact representations, not an assertion that TQ1_0 and I2_S are the same encoding.

`representation_group` has exactly two analytical values: `BF16 HF` and `one-bit frontier`. `exact_representation` retains `BF16 HF`, `TQ1_0 GGUF` or `native I2_S GGUF`. Thesis-level plots group by `representation_group` so BitNet is never silently excluded from TQ1_0 panels and the other five models are never excluded from native-I2_S panels. Companion tables always retain `exact_representation`, artifact hash and backend provenance. Method-specific TQ1_0-versus-I2_S summaries may be supplied as diagnostics, but they do not replace the inclusive frontier analysis.

### Treatment populations

Every table, panel title, legend and figure-data export must identify one of these populations explicitly:

1. `BF16 HF | untreated`;
2. `BF16 HF | SFT-full`;
3. `one-bit frontier | untreated`; or
4. `one-bit frontier | SFT-full`.

These are the **four core streams** used for main figures outside adapter-specific analysis. Judge-score analysis additionally retains `SFT-technique`, `SFT-explanation` and `SFT-joint` for both representation groups in supplementary figures and tables. Compute reporting also retains every adapter because training and deployment costs differ by adapter, while its concise headline table uses the four core streams.

### Exact naming convention

Use `Model display name | representation group [exact representation] | treatment` in every table display field and every caption, for example `BitNet b1.58 2B | one-bit frontier [native I2_S GGUF] | SFT-full`. The bracketed exact representation is mandatory in table rows, figure subtitles or captions even when the panel is grouped as one-bit frontier. Never use an unqualified label such as `quantized`, `SFT`, `full`, `Qwen`, `Hermes` or `BitNet` where model version, representation or treatment could be ambiguous.

### Figure-level audit exports

Every figure has a companion CSV under `analysis/tables/figure_data/` named `<figure_id>__<short_name>.csv`. It contains the exact plotted rows after filtering and aggregation, including:

- figure and panel identifiers;
- internal model key, canonical model display name, representation group, exact representation, treatment and full stream display label;
- prompt population (`available_case`, `pair_common_support` or `universal_panel`);
- numerator, denominator and eligible `n`;
- plotted estimate and interval bounds;
- bootstrap seed/replicates where applicable; and
- canonical source-file identifiers or hashes.

`analysis/tables/figure_manifest.csv` maps each figure to its companion CSV, source files, calculation function, filters, grouping columns, interval method, publication tier and caption. A reader must be able to reproduce every plotted point from the companion CSV without extracting values from an image.

### Mandatory figure caption block

Every generated figure is labelled `Figure <number><letter>` (or `Figure S<number><letter>` for supplementary material) inside the figure and uses the same four-part caption structure:

1. **Shows:** the population, representation groups, treatments and metric;
2. **How to interpret:** what position, colour, bar length or cell colour means;
3. **Denominator/uncertainty:** eligible `n`, common-support rule and interval method; and
4. **Does not establish:** the main causal, judge or backend limitation.

The notebook places a short plain-language `How to interpret` Markdown block immediately before each figure family. Figure numbers, output filenames, notebook headings and `figure_manifest.csv` must agree.

## Authoritative inputs

Notebook 10 reads only canonical exports under `results/final/` and records their paths and SHA-256 hashes in an analysis manifest.

- `00__source__prompts.csv`, `00__source__ground_truth.csv`, `00__source__split_manifest.csv`: source identities, prompt text/hashes, labels and scenario-step groups.
- `01__response__*.csv`, `02__response__*.csv`, `04__response__*.csv`, `05__response__*.csv`: candidate responses and deterministic audits.
- `03__judgment__*.csv`, `07__judgment__*.csv`: Phase 1 and Phase 2 Haiku judgments, including the separately frozen GPT-5.5 teacher and trivial-reference ledgers.
- `03__judgment__haiku_self__validation_test.csv`: Haiku self-evaluation.
- `08__manifest__files.csv`: expected membership and coverage.
- `results/final/compute/invocations/09__compute__invocation_inventory.csv`: one row per collected telemetry invocation.
- `results/final/compute/rows/09__compute__row_performance.csv`: per-response timing and token data.
- `results/final/compute/rows/09__compute__coverage_and_retries.csv`: response coverage and retry totals.
- `results/final/compute/stream_summary/09__compute__stream_summary.csv`: exact-stream resource summaries.
- `results/final/compute/stream_summary/09__compute__model_resource_summary.csv`: model/representation/phase resource totals.
- `results/final/compute/stream_summary/09__compute__api_usage_summary.csv`: Haiku tokens, caching and estimated cost.
- `results/final/CSV_FILE_GUIDE.md` and `COMPLETE_EXPERIMENT_PROCESS.md`: schema, provenance and interpretation rules.

Raw JSON records and manifests remain authoritative for provenance and audit if a flattened CSV field is ambiguous. The canonical final CSVs are the analysis interface. Notebook 10 must not modify source CSVs and must not reparse raw JSON to enlarge analytical coverage.

## Reproducibility controls

The first analytical cell writes `analysis/analysis_manifest.json` containing:

- analysis version and Git commit;
- input paths and hashes;
- model-size mapping and model-family labels;
- pinned MITRE ATT&CK release used for tactic mapping;
- score dimensions;
- comparison families;
- missing-data rules;
- bootstrap seed and replicate count;
- the pre-specified non-inferiority margin `delta=0.3`, recorded before teacher or baseline scores are analysed;
- minimum class-size threshold;
- multiplicity-correction families;
- package versions; and
- every generated table and figure path/hash.

Default statistical settings:

- 10,000 bootstrap replicates;
- fixed seed `42` unless changed before results are inspected;
- 95% cluster-bootstrap percentile intervals, with BCa intervals used only if implemented and tested consistently;
- scenario-step `group_id` as the main resampling cluster;
- paired prompt-level contrasts wherever the same `prompt_id` exists in both conditions;
- Benjamini-Hochberg false-discovery-rate correction within a named family of related tests;
- raw and adjusted p-values retained, with effect sizes and uncertainty emphasized over thresholded significance;
- no missing-score imputation; and
- both available-case and common-support estimates where incomplete streams affect a comparison.

### Scope of uncertainty

**Plain-language explanation.** The intervals in notebook 10 show how results vary across the sampled CAM-LDS scenario-step groups. They do not capture every way the experiment could have varied.

Technically, the cluster bootstrap conditions on the one observed trained adapter, quantized artifact, runtime history and Haiku judgment pass. It estimates prompt-population uncertainty only. It does **not** estimate uncertainty from:

- repeating SFT with another random seed;
- creating another quantized artifact;
- rerunning the same cases through Haiku;
- choosing another judge model or rubric;
- changing CUDA, GPU, backend or server scheduling; or
- selecting another model family at the same nominal size.

Every interval caption must say `95% scenario-step cluster-bootstrap interval; conditional on the observed artifact and judge pass`. Do not call these intervals total experimental uncertainty, model-training uncertainty or judge reliability intervals.

### Multiplicity families

Benjamini-Hochberg adjustment is applied within predeclared, separate families rather than across every exploratory calculation in the notebook:

1. overall condition/representation contrasts within one judge dimension;
2. adapter contrasts within one judge dimension;
3. attack-class heterogeneity tests within one dimension and class level;
4. judge-bias/internal-consistency diagnostics; and
5. compute-outcome association models.

Descriptive tables, distributions and explicitly exploratory visual searches are not converted into significance claims. The manifest records the family name for every adjusted test.

`validation` and `test` are retained for provenance but are pooled as the established 285-row evaluation set. A sensitivity table reports them separately; no adapter selection is performed from validation.

# 1. Model-response analysis

## 1.1 Analytical table and eligibility

### Plain-language purpose

The first task is to establish which responses and scores genuinely exist. A missing response, an unscored judgment and a score that cannot be used numerically are different events. Keeping them separate prevents absent data from being interpreted as poor model performance.

Build one long table with one row per candidate judgment and these keys:

`prompt_id`, `prompt_sha256`, `group_id`, `scenario`, `variant`, `step`, `split`, `model`, `condition`, `representation`, `candidate_record_id`, `candidate_response_sha256`, `judge_protocol`.

Retain all judge dimensions independently:

- `score_technique_identification`;
- `score_prediction_evidence_support`;
- `score_explanation_grounding`;
- `score_prediction_reason_alignment`;
- `score_confidence_evidence_support`;
- `score_instruction_adherence`; and
- scalar numeric `confidence_label_agreement` where the canonical CSV already contains one.

Apply one uniform eligibility rule to every dimension:

1. retain the exported value unchanged as `<dimension>_raw`;
2. accept only an already-materialized scalar numeric value in the inclusive range 0--4;
3. set `<dimension>_available` and `<dimension>_valid_range` flags;
4. count objects, strings, nulls and out-of-range values as analytically unavailable; and
5. never extract an embedded `score`, reason or label during notebook 10.

The `candidate_confidence_label` field is itself returned by the Haiku judge after interpreting the candidate response; it is not a strict deterministic extraction performed by notebook 10. Pre-analysis audits showed that its availability is highly uneven across phases, so notebook 10 must recalculate and report the exact counts from the hashed canonical inputs rather than copying an earlier inventory count. Confidence-label agreement is conditional on a recoverable candidate confidence label and cannot serve as a standalone or population-wide calibration measure.

Each dimension has its own denominator. Null scores remain null. Candidate rows absent from the judgment exports are represented only in coverage tables, not as score zero.

Define two populations for every treatment comparison:

- **available case:** every valid score available for that exact dimension and stream;
- **common support:** prompt IDs with a valid score for every condition in that stated contrast.

Every table and figure caption states which population is used and reports `n`.

For representation or treatment comparisons, common support means the exact intersection of prompt IDs with completed candidate responses **and** eligible judge scores in every stream being contrasted. Partial streams are never padded with missing values. Separate available-case summaries may still describe every judged row in a 990-row untreated stream, but no cross-stream effect estimate may mix unmatched prompt populations.

### Judge-score availability analysis

Treat score availability as an observable outcome before analysing score values. For every model, condition, representation and dimension, report:

- intended candidate rows;
- completed candidate responses;
- completed Haiku judgments;
- scalar in-range scores;
- non-scalar, null, malformed and out-of-range score cells; and
- final common-support count for each contrast.

Model scalar-score availability descriptively with a clustered logistic regression using model, representation, condition, candidate output length, finish reason, prompt-length bucket and existing format flags. The purpose is to identify selective analytical loss, not to predict or impute missing scores.

For incomplete intended pools, calculate transparent worst/best bounds on the mean by assigning every unavailable score 0 and then 4. These are identification bounds, not estimates. They show how much the conclusion could change if unavailable cases were maximally poor or maximally strong.

Outputs:

- `analysis/tables/judge_score_availability.csv`
- `analysis/tables/judge_score_availability_bounds.csv`
- `analysis/figures/00A_judge_score_eligibility_counts.png`
- `analysis/figures/00B_initial_score_boxplots_core_streams.png`
- `analysis/figures/S00B_initial_score_boxplots_conditional_adapters.png`

**Figure 00A: eligibility and score-availability counts.** Use stacked horizontal bars for each exact stream and dimension: completed candidate response, completed judgment, scalar score and analytically unavailable score. Print the intended and scalar-score counts beside each bar. Facet first by representation group and then by dimension so BF16 and one-bit-frontier populations are never mixed without labels.

**Figure 00B: initial score boxplots with eligible counts.** For each dimension, show model-level boxplots of the already-materialized 0--4 scores, with the six models in a 2 by 3 grid and the four core streams as consistently coloured boxes. Print `n=<eligible scalar scores>` above or below every box. Supplementary Figure S00B repeats the design for technique, explanation and joint adapters. Because scores are ordinal and tied, overlay a light jitter/strip or category-frequency marks and retain the full ordinal plots in Section 1.2; the boxplot is a compact orientation view, not the primary distributional evidence.

**How to interpret.** Figure 00A shows where rows were lost before score analysis. Figure 00B shows the median, interquartile spread and broad separation of eligible scores, while its `n` labels reveal unequal analytical support. Neither figure treats an unavailable score as zero or proves that observed-score distributions represent missing rows.

Additional outputs:

- `analysis/figures/00A_judge_score_eligibility_counts.png`
- `analysis/figures/00B_initial_score_boxplots_core_streams.png`
- `analysis/figures/S00B_initial_score_boxplots_conditional_adapters.png`

## 1.2 Overall multidimensional performance

### Plain-language purpose

This analysis shows how models were scored without collapsing different qualities into one number. A model can identify a technique correctly while giving weak evidence or a poorly grounded explanation, so every rubric dimension remains visible.

### Question

How do judge-score distributions vary by model, condition, representation and rubric dimension?

### Calculations

For each exact stream and dimension calculate:

- eligible response count;
- judged count;
- valid numeric-score count;
- null/invalid count;
- mean;
- median;
- interquartile range;
- proportions at 0, 1, 2, 3 and 4;
- proportion scoring at least 3; and
- group-clustered 95% bootstrap interval for the mean and proportion at least 3.

Produce two explicitly separate summary layers:

- **stream description:** all available judged rows, including the 990-row untreated runs where available; and
- **comparative analysis:** exact matched prompt IDs on the 285-row evaluation pool, or the smaller exact common support for incomplete streams.

The first describes the breadth of collected behaviour. The second supports treatment/representation contrasts. Never place their means in the same comparative panel without labeling the different populations.

The mean is a compact descriptive statistic for the anchored 0--4 scale, not evidence of interval-level measurement. The full ordinal distribution is always retained.

### Figures

**Figures 1A--1B: overview score heatmaps.** Split the overview by deployment category: Figure 1A contains `BF16 HF`; Figure 1B contains the inclusive `one-bit frontier` and therefore contains all six models in the same order. BitNet is subtitled `[native I2_S]`; the other five models are subtitled `[TQ1_0]`. Within each figure, columns are untreated, technique, explanation, joint and full; models are rows and the six substantive score dimensions are columns inside each condition panel. Figures 1A and 1B use the identical 0--4 colour scale, cell order, dimensions, model order and physical panel dimensions so they remain directly comparable.

Do not annotate every cell. Annotate only panel-level `n` and values selected for discussion; exact values remain in the companion table. These are **supplementary index figures**, not main-paper evidence. Confidence-label agreement remains in a separate diagnostic companion because its denominator differs materially.

This heatmap is the best **index view** for locating broad patterns across many dimensions. It is not the main evidence for ordinal distributions.

**Figures 2A--2B per dimension: main ordinal distributions.** Produce separate BF16 and inclusive one-bit-frontier figures for every score dimension. Both use the same 0--4 Likert colours, model order, axes and panel sizes. Each main-paper figure shows only `untreated` and `SFT-full`, arranged as six model panels in a 2 by 3 grid with raw `n` at the bar end. Figure 2B contains BitNet native I2_S beside all five TQ1_0 models rather than creating method-specific figures that omit model families. This gives a readable primary before/after treatment comparison while preserving direct BF16/one-bit comparability through uniform scales.

Because many one-bit streams are dominated by score 0, every Figure 2B has a companion **Figure 2C** with two aligned views: the original zero rate on a fixed 0--100% axis, and the conditional composition of scores 1--4 among nonzero observations. The conditional panel prints both total eligible `n` and nonzero `n`; it never replaces the main 100% Likert bar. This exposes small nonzero differences without visually minimizing the scientifically important floor effect. Do not use a broken axis or silently remove zeros.

**Figures S2A--S2B per dimension: conditional-adapter distributions.** Retain technique, explanation and joint adapters in separate BF16 and one-bit supplementary figures, again with identical scales and ordering. The full adapter may be repeated as a visual reference. Promotion of a conditional-adapter figure into the main paper is allowed only under a predeclared relevance rule—directly answering a stated adapter-specific question—not because it happens to show the largest effect.

Supplementary **Figure S2C** repeats the zero-rate plus conditional-nonzero view for one-bit technique, explanation and joint adapters. It uses the same model order and axes as Figure 2C.

**How to interpret Figures 1--2.** Heatmap colour indicates the available-case mean and is used only to locate broad patterns. Likert-bar width gives the proportion at each actual score and is the primary distributional view. In Figures 2C/S2C, the left panel preserves the overall probability of a zero; the right panel answers the narrower question, "when the judge did not assign zero, which nonzero scores occurred?" A larger high-score share in the conditional panel is not evidence of better overall performance unless the zero rate is considered simultaneously.

Do not use radar charts: axes are not independent, polygon area is hard to compare, and ordering dimensions changes the visual impression. Do not place all dimensions, models, conditions and representations into one giant grouped bar chart.

### Rubric scale-use diagnostics

**Plain-language purpose.** Before interpreting small score differences, check whether Haiku actually used the full 0--4 scale or mostly selected a few categories. A compressed or strongly skewed scale has less power to distinguish candidates.

For every dimension and judge phase report:

- category utilization for 0--4;
- modal category;
- floor and ceiling rates;
- Shannon entropy and effective number of used categories;
- percentage concentrated in the two most common categories; and
- score distribution by API run/batch identifier where that identifier is already available.

This is a scale-use diagnostic, not proof of calibration or rubric-option position bias. The rubric order was fixed and was not experimentally permuted. Recent work shows that even pointwise rubric scoring can inherit preference for score-option positions; this study can document category compression but cannot identify its cause (Xu et al., 2026).

### Outputs

- `analysis/tables/model_scores_by_dimension.csv`
- `analysis/tables/model_score_distributions.csv`
- `analysis/tables/judge_scale_use.csv`
- `analysis/figures/01A_score_overview_bf16.png`
- `analysis/figures/01B_score_overview_one_bit_frontier.png`
- `analysis/figures/02A_<dimension>_bf16_untreated_full.png`
- `analysis/figures/02B_<dimension>_one_bit_untreated_full.png`
- `analysis/figures/02C_<dimension>_one_bit_zero_and_nonzero_composition.png`
- `analysis/figures/S02A_<dimension>_bf16_conditional_adapters.png`
- `analysis/figures/S02B_<dimension>_one_bit_conditional_adapters.png`
- `analysis/figures/S02C_<dimension>_one_bit_conditional_zero_and_nonzero.png`
- `analysis/figures/02G_judge_scale_use.png`

## 1.3 Full treatment-contrast analysis

### Plain-language purpose

This section asks whether the **same prompt rows** received different scores after SFT or one-bit-frontier deployment. Matching the rows matters: otherwise a treatment could appear better simply because it was evaluated on easier prompts.

### Question

For each model and rubric field, how do BF16 SFT, one-bit-frontier SFT, untreated BF16 and untreated one-bit-frontier responses differ on the same prompt rows?

### Contrasts

For every adapter and model, calculate paired prompt-level differences for:

1. BF16 SFT minus BF16 untreated;
2. one-bit-frontier SFT minus one-bit-frontier untreated;
3. one-bit-frontier untreated minus BF16 untreated;
4. one-bit-frontier SFT minus BF16 SFT; and
5. difference-in-differences: `(one-bit SFT - one-bit untreated) - (BF16 SFT - BF16 untreated)`.

All comparisons use the 285-row evaluation pool and exact common-support prompt IDs for the stated contrast. Contrast 5 is explicitly labelled exploratory because BF16 and GGUF phases also differ in inference engine, kernels, scheduling and serialized representation. It is not a pure causal estimate of quantization.

For incomplete streams, common support is recalculated for each exact contrast; it is never borrowed from another field or adapter. Report how that common-support subset differs from the intended 285 rows in prompt length, ground-truth class, scenario and split. Separate available-case summaries may use all judged rows, including up to 990 untreated rows, but are descriptive and are not substituted for paired effects.

Use two matched populations:

- **pair-specific common support:** the largest exact intersection for one stated contrast; this is the main effect estimate;
- **universal balanced panel:** the intersection shared by every stream in one predeclared comparison family; this is a smaller sensitivity panel that makes effects across models/adapters directly comparable.

Show the pair-specific and universal-panel estimates together when the latter is large enough to be informative. If the universal panel is very small, report it descriptively and explain the loss of coverage rather than treating it as the preferred estimate.

Treat TQ1_0 and native I2_S as members of the broad **one-bit-frontier deployment treatment** for every thesis-level model-suite table, figure and contrast so all six model families remain included. They are different encodings and runtime paths, so `exact_representation` and backend provenance remain visible in every detailed table and caption and method-specific diagnostics remain available. Likewise, `one-bit-frontier SFT` is an explicit compound treatment label meaning adapter merge plus GGUF conversion/quantization plus the relevant llama.cpp/BitNet inference stack; it is not a weight-only intervention.

### Statistics

For each contrast report:

- paired mean difference;
- paired median difference;
- the full paired difference distribution and interquartile range;
- percentage improved, unchanged and worsened;
- win, tie and loss proportions;
- probability of superiority with ties, `P(score_A > score_B) + 0.5 P(score_A = score_B)`;
- group-clustered 95% bootstrap interval; and
- cluster sign-flip permutation p-value, with BH adjustment within each score-dimension contrast family.

Means remain useful summaries of the anchored 0--4 scores, but they never appear without the ordinal distribution, median/IQR, percentage scoring 3--4 and common-support denominator. Raw counts are used for audit and sample-size context; normalized rates or proportions are used for comparisons. Counts from streams with different coverage are never compared as if they shared a denominator.

### Figures

**Figures 3A--3C per judge field: treatment-contrast forest plots.** Split the five contrasts into three low-density families: Figure 3A shows BF16 SFT-minus-untreated adapter effects; Figure 3B shows one-bit SFT-minus-untreated effects; Figure 3C shows BF16-to-one-bit deployment-stack contrasts and difference-in-differences. Each figure is a 2 by 3 model grid, uses the identical -4 to +4 scale, and places common-support `n` beside every estimate.

This is lower-dimensional and more interpretable than placing all fields in one forest plot. A point right of zero favours the first-named treatment; a point left favours the comparator. Put common-support `n` beside each contrast label.

**Figures 4A--4B: improved/unchanged/worsened bars.** Split BF16 and one-bit representations. For one score field, show a three-part horizontal bar per model/adapter using the same model order and percentage scale. The main-paper version uses untreated-to-full transitions; conditional-adapter transitions remain supplementary. **(Ignore or show in appendix)**

### Adapter-specificity profiles

**Plain-language purpose.** Conditional adapters were trained for different objectives. This analysis checks whether each adapter mainly changed its intended capability or produced broad changes across the whole rubric.

For every model and representation, calculate the paired SFT-minus-untreated effect separately for all six dimensions for the technique, explanation, joint and full adapters. Produce **one compact figure per model**, with separate BF16 and one-bit panels, four adapter profiles, identical -4 to +4 axes and intended dimensions visually outlined. This avoids placing all models, representations and adapters into one chart. Also report rescued rows (`0--1` to `3--4`), newly degraded rows (`3--4` to `0--1`) and unchanged rows for each dimension.

Cross-dimension differences are descriptive because rubric dimensions measure different constructs even though they share a 0--4 range. Do not test or claim that a one-point change in one dimension is intrinsically equivalent to a one-point change in another.

### Outputs

- `analysis/tables/treatment_contrasts.csv`
- `analysis/tables/treatment_transition_counts.csv`
- `analysis/tables/common_support_diagnostics.csv`
- `analysis/tables/adapter_specificity_profiles.csv`
- `analysis/figures/03A_<dimension>_bf16_sft_effects.png`
- `analysis/figures/03B_<dimension>_one_bit_sft_effects.png`
- `analysis/figures/03C_<dimension>_deployment_stack_effects.png`
- `analysis/figures/04A_<dimension>_bf16_treatment_transitions.png`
- `analysis/figures/04B_<dimension>_one_bit_treatment_transitions.png`
- `analysis/figures/04C_<model>_adapter_specificity_profiles.png`

## 1.4 ATT&CK distribution and class-level performance

### Plain-language purpose

Overall averages can hide that some ATT&CK techniques, tactics or multi-label cases are much harder than others. This section first shows what the benchmark contains, then asks where each model performs well or poorly.

### Descriptive source distribution

Before performance analysis, show what the benchmark contains in:

- all 990 source rows; and
- the pooled 285 validation/test rows.

Report exact technique, parent technique, tactic, scenario and scenario-step distributions. Pin one MITRE ATT&CK Enterprise release and store its STIX/hash metadata. A technique may map to more than one tactic, so report both:

- number of rows containing the class; and
- number of expanded row-class memberships.

### Tactic visualization

The main figure is **not** a full ATT&CK matrix. Use paired horizontal bars for each tactic: percentage of 990 rows and percentage of 285 evaluation rows, with raw counts labelled. This is easier to read and compare.

An ATT&CK-matrix-shaped tile plot may be generated as an appendix descriptive figure, with tiles coloured by source frequency and no performance claim. Because a technique can appear under multiple tactics, the caption must state that duplicated tactic placement is structural rather than extra independent observations.

### Performance calculations

Assign each prompt-level judge score to every ground-truth class attached to that row, then calculate class-specific means and bootstrap intervals. This is the primary class-associated difficulty analysis and does not require parsing a candidate response. Do not treat the expanded memberships as independent; resample `group_id` before expansion.

Use a predeclared minimum of 15 source rows for a named class in inferential plots. Classes below the threshold remain in a frequency appendix.

- **Row-weighted average:** average once per prompt row, so multi-label rows are not implicitly given extra weight. This is the main overall class-linked summary.
- **Class-macro average:** first calculate the mean for each eligible class, then average class means, so every eligible class has equal weight.
- **Membership-micro average:** average over expanded prompt-class memberships, so common and multi-label classes contribute more. Retain only as a secondary sensitivity estimate.

Show row-weighted and class-macro estimates as two labelled points per stream in a compact dumbbell plot; place membership-micro in the supplementary table. They answer different questions: row weighting describes the observed prompt population, whereas class macro describes the typical eligible class without frequency weighting.

Where canonical judge exports already contain a machine-usable judge-extracted prediction field, a separate judge-mediated hit/miss analysis may be reported. It must remain explicitly judge-mediated and must not be reconstructed by reparsing raw candidate or raw judge text. If that field is not already materialized in a directly usable form, omit technique-specific recall and rely on ground-truth-class-associated judge scores. Existing strict-parser technique metrics may appear only as a coverage-limited weak sensitivity analysis.

### Ground-truth multiplicity

Rows requiring several ATT&CK techniques may demand broader coverage than single-label rows. Report score distributions, failure-label rates, completion/timeout rates and treatment effects by the number of ground-truth technique IDs. Include label multiplicity in class-level descriptive models and in the common-support composition table.

This does not imply that every ground-truth technique is visibly supported by every sampled log window. The CAM-LDS ground truth identifies the annotated attack step, while the prompt contains only a sampled excerpt; absence of visible evidence remains a construct-validity limitation.

### Figures

- **Figure 5:** paired tactic-distribution bars for 990 versus 285;
- **Figures 6A-B:** model-by-technique judge-score heatmaps for BF16 and the inclusive one-bit frontier, one score dimension per figure;
- **Figures 6C-D:** technique-level paired SFT-effect forest plots for BF16 and the inclusive one-bit frontier;
- **Figures 6E-F:** row-weighted/class-macro dumbbell plots for BF16 and the inclusive one-bit frontier; and
- **Figures 7A-B:** bottom-technique dot-plot grids for BF16 and the inclusive one-bit frontier.

`Sufficiently represented` is fixed before examining scores: a technique or parent technique must occur in at least 15 unique prompt rows in the relevant source population. Main paired figures additionally require at least 10 eligible common-support judged rows for every stream shown; otherwise the class remains listed with its exact count and an `insufficient common support` marker rather than being silently removed. Selection never depends on the observed mean score. Main-paper technique figures use at most 10--15 such predeclared classes and show untreated versus SFT-full only. Conditional adapters and the full eligible technique set remain in supplementary figures. All main/supplementary pairs use identical technique ordering and axes. Heatmaps contain no cell text once they exceed 100 cells; exact estimates and denominators remain in tables.

Every class-performance figure is produced twice with identical scales: once for all six BF16 models and once for all six one-bit-frontier models. The one-bit panels include BitNet native I2_S and the five TQ1_0 models. Supplementary versions retain technique, explanation and joint adapters for both representation groups.

For the bottom-technique figure, select the bottom `K` techniques once using a pooled, predeclared reference estimate—not separately per model. Create one figure per judge dimension and representation. Use a 2 by 3 model grid; y-axis is the same selected techniques, x-axis is mean 0--4 judge score, and dots/connecting segments compare untreated with full. Conditional adapters use a supplementary version. This keeps selection fixed and makes model/condition comparison possible without hundreds of grouped bars.

**How to interpret Figures 5--7.** Figure 5 describes benchmark composition, not performance. Darker Figure 6A/B cells mean higher class-associated judge scores for the named stream; Figures 6C/D show whether the same class rows moved after SFT; Figures 6E/F show the difference between weighting prompt rows and weighting eligible classes equally. Figures 7A/B identify consistently low-scoring, sufficiently represented classes using one fixed class list. Intervals are 95% scenario-step cluster-bootstrap intervals using 10,000 replicates. A class-level score associates a row's ground-truth class with its judge score; it is not strict parsed technique recall.

### Outputs

- `analysis/tables/source_attack_distribution.csv`
- `analysis/tables/attack_class_scores.csv`
- `analysis/tables/attack_class_macro_micro.csv`
- `analysis/tables/ground_truth_multiplicity.csv`
- `analysis/figures/05_tactic_distribution_990_vs_285.png`
- `analysis/figures/06A_<dimension>_bf16_technique_heatmap.png`
- `analysis/figures/06B_<dimension>_one_bit_frontier_technique_heatmap.png`
- `analysis/figures/06C_<dimension>_bf16_technique_sft_effects.png`
- `analysis/figures/06D_<dimension>_one_bit_frontier_technique_sft_effects.png`
- `analysis/figures/06E_<dimension>_bf16_row_macro_comparison.png`
- `analysis/figures/06F_<dimension>_one_bit_frontier_row_macro_comparison.png`
- `analysis/figures/07A_<dimension>_bf16_lowest_techniques.png`
- `analysis/figures/07B_<dimension>_one_bit_frontier_lowest_techniques.png`
- `analysis/figures/S06A_<dimension>_bf16_all_techniques_and_adapters.png`
- `analysis/figures/S06B_<dimension>_one_bit_frontier_all_techniques_and_adapters.png`
- `analysis/figures/S07A_<dimension>_bf16_conditional_adapter_lowest_techniques.png`
- `analysis/figures/S07B_<dimension>_one_bit_frontier_conditional_adapter_lowest_techniques.png`
- `analysis/figures/07C_ground_truth_multiplicity.png`

## 1.5 Related judge dimensions: prediction, reasoning and confidence

### Plain-language purpose

These comparisons ask whether a response succeeds in one way but fails in another—for example, naming the right technique without supporting it from the log. Joint score distributions are more informative than a single correlation because they expose concrete high/low combinations.

### Questions

- Does technique identification occur without evidence support?
- Is an explanation grounded in the log but misaligned with the prediction, or vice versa?
- Is confidence evidentially supported, and does it agree with the judge's own warranted confidence category?
- Is apparent weakness mostly instruction/format adherence or substantive interpretation?
- Does similarity to the Haiku reference track ground-truth coverage or merely Haiku-like expression?

### Pairings

- technique identification versus prediction evidence support;
- explanation grounding versus prediction-reason alignment;
- confidence evidence support versus confidence-label agreement;
- instruction adherence versus each substantive score;
- prediction similarity versus technique identification; and
- reasoning similarity versus explanation grounding.

### Calculations and figures

For every pair:

- 5 by 5 joint-frequency heatmap using row percentages;
- Spearman correlation with group-clustered bootstrap interval;
- paired difference distribution;
- four quadrant proportions using low `0--1` versus high `3--4`, with score 2 shown as a middle category rather than forced into a quadrant.

The analysis is never aggregated across models. For each score pairing, produce one **Figure 8A** contact sheet for the four core streams: rows are the six models and columns are `BF16 untreated`, `BF16 SFT-full`, `one-bit-frontier untreated` and `one-bit-frontier SFT-full`. Each cell is a 5 by 5 row-percentage heatmap with a printed eligible `n`; all cells share the same 0--100% scale. This allows within-model and cross-representation comparison without blending model distributions.

Produce supplementary **Figures S8A/S8B** for every pairing: one BF16 and one inclusive one-bit-frontier figure with model rows and adapter columns (`technique`, `explanation`, `joint`, `full`; untreated may be repeated as reference). BitNet remains in the one-bit contact sheet with `[native I2_S]`; the other models retain `[TQ1_0]`. Correlations, intervals, difference distributions and quadrant proportions are written to the companion CSV rather than crowded into every heatmap. Do not aggregate counts across models and do not place all correlations into a network diagram.

**How to interpret Figure 8.** Each small heatmap describes one named model/representation/treatment stream. Mass near the upper-right means both related judge dimensions tend to be high; off-diagonal mass exposes mismatches such as correct identification with weak evidence. Compare cells only when their printed denominators are adequate. Spearman intervals use the same 10,000 scenario-step cluster bootstrap as the rest of the judge analysis.

Confidence evidence support versus confidence-label agreement may expose internal disagreement in the judge rubric, but it cannot prove judge bias: Haiku generated both `judge_warranted_confidence_label` and the agreement score in the same pass. Treat this as a self-consistency diagnostic and explicitly state the circularity.

Because candidate-confidence recovery is particularly sparse in Phase 2, always show a confidence-flow table: judged rows, rows with candidate label, rows with warranted label, rows with scalar agreement score and final paired denominator. Do not rank models by confidence-label agreement when coverage is too small or materially unequal; retain the result as a conditional diagnostic.

Judge reasons are produced in the same pass as the scores. Their presence and consistency can support an internal audit, but they are not independent validation evidence. Chiang and Lee (2023) show that asking for explanations can change agreement with human ratings; in this study, the frozen reason-generating protocol is part of the judge measurement itself.

### Outputs

- `analysis/tables/judge_dimension_relationships.csv`
- `analysis/tables/confidence_field_availability.csv`
- `analysis/figures/08A_<pair>_core_stream_joint_score_grid.png`
- `analysis/figures/S08A_<pair>_bf16_all_adapters.png`
- `analysis/figures/S08B_<pair>_one_bit_frontier_all_adapters.png`

## 1.6 Failure analysis

### Plain-language purpose

Average scores say how strongly models performed, but failure labels say **how** they failed. Rates rather than raw counts are required because streams have different completion and judge-score coverage.

### Normalization

Preserve raw `judge_failure_labels`. Add a versioned, many-to-one normalization dictionary with broad categories:

- wrong/missed technique;
- unsupported prediction;
- explanation absent or ungrounded;
- confidence error;
- format/schema error;
- truncation/length limit;
- degenerate repetition; and
- instruction failure.

No raw label is discarded. Unmapped labels remain `unmapped` and are listed.

### Calculations

- raw count;
- rate per 100 judged responses with the exact denominator;
- 95% scenario-step cluster-bootstrap interval using the globally fixed 10,000 replicates and seed 42;
- paired SFT-minus-untreated risk difference;
- co-occurrence counts and lift for common label pairs; and
- rate by prompt-length bucket and attack class.

Raw-count bar charts alone are insufficient when stream coverage differs. Use simple horizontal bars of failures per 100 responses and print the raw count and judged denominator at the bar end. **Figures 9A and 9B** separately show BF16 and the inclusive one-bit frontier, each as a 2 by 3 model grid comparing untreated with SFT-full. Supplementary **Figures S9A and S9B** repeat the exact design for technique, explanation and joint adapters. The one-bit figure includes BitNet native I2_S and all five TQ1_0 models. A compact heatmap is retained only as a supplementary index across all streams.

Use an UpSet plot only for the ten most common normalized combinations; do not use a dense Venn diagram.

**How to interpret Figures 9--10.** A longer Figure 9 bar means Haiku assigned that normalized failure category more often per 100 judged responses; the printed count shows the underlying number of labels. Its interval is the same 10,000-replicate cluster-bootstrap interval used for judge-score estimates. Figure 10 shows which failure categories co-occur within responses; it does not imply that one failure causes another. BF16 and one-bit panels have identical axes, and no raw count is compared without its denominator.

### Score–failure-label coherence

Check whether Haiku's structured failure labels agree with its own scores. For example, `wrong/missed technique` should generally accompany low technique-identification scores, `unsupported prediction` should align with low evidence support, and `instruction failure` should align with low instruction adherence.

For every normalized label, report the associated score distribution, median, proportion at `0--1`, and proportion at `3--4`. Flag high-score/failure-label and low-score/no-failure combinations for the qualitative audit. This measures internal output coherence only; it does not prove that either the score or label is correct.

### Outputs

- `analysis/tables/failure_label_dictionary.csv`
- `analysis/tables/failure_rates.csv`
- `analysis/tables/failure_risk_differences.csv`
- `analysis/tables/score_failure_label_coherence.csv`
- `analysis/figures/09A_bf16_failure_rate_bars.png`
- `analysis/figures/09B_one_bit_frontier_failure_rate_bars.png`
- `analysis/figures/S09A_bf16_conditional_adapter_failure_rates.png`
- `analysis/figures/S09B_one_bit_frontier_conditional_adapter_failure_rates.png`
- `analysis/figures/10_failure_cooccurrence_upset.png`

## 1.7 Repeated prompts and judge-grounding audits

### Plain-language purpose

Repeated and closely related prompts reveal whether a model behaves consistently and whether Haiku applies its rubric coherently. They do not replace a designed judge-reliability experiment, but they provide useful internal checks from cases already present in the dataset.

The source contains 990 prompt IDs but only 393 distinct prompt hashes. This enables candidate-consistency analysis and limited natural duplicate-case audits of the judge.

### Candidate response stability

Group by `model`, `condition`, `representation`, artifact/configuration identity and `prompt_sha256`.

Primary measures:

- exact response-hash agreement;
- exact judge score-vector agreement where the same response case was judged more than once;
- judge-score dispersion across repeated prompts;
- judge-mediated technique-set agreement only when an already-materialized canonical field is available;
- output-token range; and
- TF-IDF cosine similarity as a supplementary lexical measure.

Do not create a new technique extractor for this section. If canonical judge-extracted technique sets are already directly usable, Jaccard is suitable for identical prompts because it asks whether the judge interpreted the model as making the same claims. Label it `judge-mediated technique-set consistency`. When both sets are empty, report `both_empty` separately instead of assigning a misleading perfect Jaccard score. Do not use character n-gram cosine as a primary result; it adds little interpretable information beyond exact equality and TF-IDF for this study.

TF-IDF analyses are limited to:

- repeat-prompt lexical stability;
- lexical change from untreated to SFT; and
- candidate-to-teacher and candidate-to-Haiku lexical resemblance.

TF-IDF is never interpreted as correctness or semantic grounding.

The treatment comparison for lexical similarity uses only the four core streams and exact matched prompt IDs. For each model, calculate candidate-to-Haiku and candidate-to-teacher TF-IDF similarity for `BF16 untreated`, `BF16 SFT-full`, `one-bit-frontier untreated` and `one-bit-frontier SFT-full`. Report paired untreated-to-full changes within each representation and BF16-to-frontier changes within each treatment. **Figure 13A** is a 2 by 3 model grid; each panel shows the four core-stream estimates with paired prompt-level intervals. **Figure S13A** repeats lexical change for conditional adapters. This asks whether treatment moved surface expression toward either reference; it does not measure semantic quality.

### Scenario-step variant sensitivity

The benchmark contains several prompt variants within the same scenario-step group. For each model, condition, representation and judge dimension, report the within-`group_id` score range, median absolute deviation, proportion of groups spanning at least two score points, completion-status variation and output-length variation. Compare these measures across models and representations.

Use a variance-component or mixed descriptive model only as a secondary summary, with scenario-step as a grouping factor and variant/prompt length as fixed descriptors. The purpose is to quantify sensitivity to different excerpts or variants of the same attack step, not to claim that one variant caused the score change.

### Judge rubric and output-consistency audit

Notebook 10 does not reparse raw judge text or serialized objects to recreate rubric arithmetic. Use only already-materialized scalar judge/audit fields. Where both an exported rubric-implied value and returned score already exist, report exact agreement and mean absolute disagreement first, followed by weighted Cohen's kappa as a secondary statistic. If the required fields are not directly available, record the audit as `not_applicable` rather than inventing a parser.

This audit tests internal consistency of the existing judge outputs. It does not establish that Haiku interpreted the candidate correctly, and it is not a validation of judge reliability.

### Confidence-score arithmetic audit

Use the already-exported `candidate_confidence_label`, `judge_warranted_confidence_label` and scalar `confidence_label_agreement` values. Do not extract a candidate label from the raw response. On rows where all required fields are present, report exact rule agreement and mean absolute disagreement before weighted kappa, plus a confusion matrix. The eligible denominator and availability rate are mandatory because Phase 2 coverage is small and uneven. This exposes internal inconsistency in the same-pass confidence procedure without treating Haiku's warranted label as ground truth or implying population-wide calibration.

### Natural duplicate-case consistency

Construct a blinded case fingerprint from:

`prompt_sha256 + canonical ground_truth + candidate_response_sha256 + haiku_reference_response_sha256 + rubric_hash + judge_protocol`.

Where the same fingerprint was judged more than once, calculate exact score-vector agreement, per-dimension weighted kappa and maximum disagreement. Report how many duplicate fingerprints exist; do not generalize if the natural duplicate sample is small. This is not a substitute for a designed test-retest study.

### Reference-anchoring and self-preference diagnostics

The Haiku reference was visible during candidate judging. Assess whether similarity to that reference predicts higher scores beyond observable correctness proxies:

- stratify by ground-truth class and already-exported judge score where appropriate;
- estimate within-prompt associations between similarity and substantive score;
- fit a secondary model with prompt/group effects, candidate output length and candidate condition; include existing strict-format flags only as weak sensitivity covariates;
- compare Haiku self-evaluation score distributions with candidates on matched prompts; and
- report candidate/reference lexical similarity separately from judge scores.

Add a two-reference triangulation using the already stored GPT-5.5 teacher response and Haiku reference:

- candidate-to-teacher TF-IDF similarity;
- candidate-to-Haiku-reference TF-IDF similarity;
- paired change in both similarities after SFT;
- within-prompt association of each similarity with every substantive judge score; and
- models where judge scores rise mainly with Haiku similarity rather than teacher similarity.

These lexical measures do not establish semantic correctness. They help distinguish distillation-like surface convergence from possible reference anchoring. Self-preference research warns that a judge–candidate score gap is confounded by genuine response quality unless independent quality labels exist (Chen et al., 2025; Panickssery et al., 2024).

A positive residual association is evidence consistent with reference anchoring or stylistic preference, not proof of narcissism. Candidate quality and similarity are confounded, only one judge was used, and the judge may infer model family from style despite metadata blinding.

### Verbosity and format-sensitivity diagnostics

Within prompt IDs that have multiple candidates, model each judge dimension against candidate output tokens and length-limit status while controlling for candidate condition and ground-truth class. Existing strict-format indicators may be added only in a labelled secondary sensitivity model. Report marginal associations and prompt-clustered intervals. These are bias diagnostics, not causal estimates: longer answers may genuinely contain more evidence.

### Judge-bias and internal-consistency diagnostic panel

The study has no designed test-retest sample, independent judge panel or blinded human-coded subset. It therefore cannot establish judge reliability. Produce four separate diagnostic figures rather than one dense dashboard, covering:

- Haiku self-evaluation score distributions versus matched candidate distributions;
- within-prompt association between Haiku-reference similarity and substantive score;
- confidence-label availability and same-pass agreement diagnostics;
- scalar judge-score availability by phase, representation and model; and
- natural duplicate-case exact agreement where duplicate fingerprints exist.

These results help readers calibrate likely circularity, self/reference preference and output-format sensitivity. Call them `internal consistency and bias diagnostics`, never `judge reliability validation`.

### Reproducible qualitative audit

Quantitative scores cannot establish whether Haiku correctly interpreted malformed or unconventional text. Select a small audit set with a frozen algorithm rather than hand-picking favourable examples. Include cases from:

- high judge score with failed candidate format audits;
- low score despite successful candidate format audits;
- largest paired SFT improvements and degradations;
- high Haiku-reference similarity and low teacher similarity, and the reverse;
- output-ceiling or timeout-adjacent rows; and
- score/failure-label contradictions.

For each selected row display prompt ID, a bounded prompt/log excerpt, ground truth, candidate response, teacher response, Haiku reference, all judge scores, judge reasons, failure labels and provenance hashes. Reasons illustrate how Haiku justified a result; they do not independently verify it. Doddapaneni et al. (2024) found that evaluators may mention a defect without changing the associated score, which motivates showing score and reason together.

**How to interpret Figures 11--13.** Figure 11 shows whether identical prompt text produces identical or lexically similar responses and score vectors. Figure 12 diagnoses availability, natural duplicate agreement and Haiku's circular self-evaluation; it does not validate judge reliability. Figure 13 compares lexical proximity to the Haiku reference and GPT-5.5 teacher across the four core streams and separately shows its association with judge scores. An upward similarity shift is surface convergence, not proof of improved correctness or grounding.

### Outputs

- `analysis/tables/repeated_prompt_consistency.csv`
- `analysis/tables/scenario_step_variant_sensitivity.csv`
- `analysis/tables/judge_output_consistency_audit.csv`
- `analysis/tables/judge_confidence_rule_audit.csv`
- `analysis/tables/judge_duplicate_case_consistency.csv`
- `analysis/tables/judge_reference_anchor_diagnostics.csv`
- `analysis/tables/lexical_similarity_treatment_comparisons.csv`
- `analysis/tables/qualitative_audit_cases.csv`
- `analysis/figures/11A_repeated_response_stability.png`
- `analysis/figures/11B_repeated_judge_score_stability.png`
- `analysis/figures/12A_judge_score_availability_diagnostic.png`
- `analysis/figures/12B_judge_duplicate_consistency.png`
- `analysis/figures/12C_confidence_rule_diagnostic.png`
- `analysis/figures/12D_haiku_self_evaluation_diagnostic.png`
- `analysis/figures/13D_<dimension>_reference_similarity_vs_score.png`
- `analysis/figures/13A_core_stream_reference_lexical_similarity.png`
- `analysis/figures/S13A_conditional_adapter_reference_lexical_similarity.png`
- `analysis/figures/13B_scenario_step_score_sensitivity.png`
- `analysis/figures/13C_scenario_step_operational_sensitivity.png`

## 1.8 Long-tail analysis

### Plain-language purpose

The longest prompts and rarest ATT&CK classes may behave differently from typical rows. This section shows where results are sparse, slow, unsupported by context, or uncertain rather than allowing a global average to hide those cases.

### Prompt-length tail

Use the existing fixed operational buckets (`<=2048`, `2049--4096`, `4097--8192`, `>8192`) and deciles of input tokens. Preserve input tokens as a continuous `log1p` variable for models.

Analyze:

- completed, timeout, missing and `context_unsupported` rates;
- EOS versus output-limit rate;
- output tokens;
- row latency and throughput;
- each judge dimension;
- existing deterministic format-adherence flags as secondary diagnostics; and
- normalized failure categories.

Figures:

- empirical cumulative distribution of prompt lengths;
- completion-status stacked bars by fixed bucket;
- binned score means with group-bootstrap intervals;
- median and p95 latency by bucket; and
- one partial-effect plot from a predeclared regression model.

Every Figure 18 analysis uses a 2 by 3 model grid and explicitly compares the four core streams: BF16 untreated, BF16 SFT-full, one-bit-frontier untreated and one-bit-frontier SFT-full. Line style identifies representation group and colour identifies untreated versus SFT-full; exact one-bit representation remains in panel subtitles. Supplementary Figures S18B--S18D repeat completion, score and latency analyses for technique, explanation and joint adapters in both representation groups. Do not pool model families, representations or treatments into one unlabeled curve.

Use clustered logistic regression for completion/limit outcomes and an exploratory ordinal or robust regression for score. Length is associated with attack complexity, so do not claim a causal effect.

For partial streams, place available-case score summaries beside the 0--4 worst/best identification bounds defined in Section 1.1. Do not apply inverse-probability weighting as a primary correction: context exclusions, timeouts and GPU availability do not support a credible missing-at-random assumption.

### Rare attack-class tail

Plot class frequency against interval width and score. This shows where apparent performance differences are based on sparse labels. Classes below the inferential threshold remain descriptive.

**How to interpret Figure 18.** The ECDF shows what fraction of prompts fall below a given input length for each named stream. Completion and latency panels show whether longer prompts coincide with operational failure or delay; score panels show associations with judge outcomes. Wider rare-class intervals signal low information, not inherently unstable models. All score intervals use 10,000 scenario-step cluster-bootstrap replicates; regression lines remain non-causal because prompt length is associated with scenario complexity.

Outputs:

- `analysis/tables/prompt_length_and_long_tail.csv`
- `analysis/tables/rare_attack_class_uncertainty.csv`
- `analysis/figures/18A_prompt_length_ecdf.png`
- `analysis/figures/18B_completion_by_prompt_length.png`
- `analysis/figures/18C_scores_by_prompt_length.png`
- `analysis/figures/18D_latency_by_prompt_length.png`
- `analysis/figures/18E_rare_class_frequency_uncertainty.png`
- `analysis/figures/S18B_conditional_adapter_completion_by_prompt_length.png`
- `analysis/figures/S18C_conditional_adapter_scores_by_prompt_length.png`
- `analysis/figures/S18D_conditional_adapter_latency_by_prompt_length.png`

## 1.9 Model size and behavioural similarity

### Plain-language purpose

This section asks two distinct questions: whether larger models tend to receive different scores, and whether different models struggle on the **same prompt rows**. The second question is often more informative than comparing overall averages.

There are six non-identical model families; size is not independently randomized. Do not fit or claim a general scaling law.

Parameter count is confounded with architecture, tokenizer, pretraining corpus, instruction tuning, native context, quantization compatibility and runtime backend. Size plots are therefore weak cross-model comparisons used to describe the observed suite, not estimates of a causal size effect. Statements such as "larger models benefit more from SFT" require this caveat directly in the caption.

Analyses:

- stream-level score versus `log10(parameter_count)`;
- paired prompt-level score differences between model pairs;
- Spearman/Kendall agreement of model rankings across judge dimensions;
- correlation of attack-class score profiles across models;
- Jaccard/cosine similarity of normalized failure-rate profiles; and
- proportion of prompt rows on which small and larger models share the same normalized failure category.

Add an item-level concordance analysis for every judge dimension on exact matched prompts:

- pairwise Spearman correlation of prompt-level scores;
- Jaccard overlap of rows scoring `0--1`;
- overlap of each model's bottom score decile;
- agreement on normalized failure categories;
- rows difficult for nearly every model; and
- rows uniquely difficult for one model family or representation.

Produce one model-by-model concordance matrix per dimension and a concise table of shared versus model-specific hard cases. This responds directly to Dietz et al.'s warning that similar system-level rankings can conceal substantial item-level disagreement.

Figures:

- separate BF16 and inclusive one-bit-frontier log-size scatters per judge dimension, using identical axes and showing one condition per panel;
- model-by-model attack-profile correlation matrices for one condition/representation at a time; and
- failure-profile dendrograms as exploratory supplementary figures, with the distance definition stated.

The Figure 17 series is organized by the four core streams rather than by exact quantizer: Figure 17A is BF16 untreated, 17B is BF16 SFT-full, 17C is one-bit-frontier untreated and 17D is one-bit-frontier SFT-full. Every matrix includes all six models in the same order. In the frontier matrices, BitNet contributes native-I2_S scores and the other five models contribute TQ1_0 scores; exact representation remains in the companion CSV and subtitle. Supplementary Figures S17A--S17F repeat the concordance matrices for technique, explanation and joint adapters in both representation groups. No Figure 17 panel may exclude BitNet merely because it is not TQ1_0, or exclude other models because they are not I2_S.

Do not place all conditions and representations into one size scatter or matrix. Label only points required to identify models; exact condition/representation values remain in the table.

**How to interpret Figures 16--17.** Figure 16 describes whether score levels in this six-model suite co-vary with nominal parameter count; it is not a scaling law. Figure 17 asks whether two models find the same prompt rows easy or difficult: higher concordance means their item-level score ordering is more similar. Compare the four core matrices side by side using the same -1 to +1 scale. Differences between BF16 and frontier matrices include backend and representation effects and are exploratory.

The dendrogram and correlations are descriptive. The six models differ in architecture, training corpus, tokenizer, instruction tuning and backend, so they cannot identify a causal parameter-size effect.

Outputs:

- `analysis/tables/model_item_difficulty_concordance.csv`
- `analysis/tables/shared_and_model_specific_hard_cases.csv`
- `analysis/figures/16A_<dimension>_bf16_model_size_relationships.png`
- `analysis/figures/16B_<dimension>_one_bit_model_size_relationships.png`
- `analysis/figures/17A_<dimension>_bf16_untreated_item_difficulty_concordance.png`
- `analysis/figures/17B_<dimension>_bf16_sft_full_item_difficulty_concordance.png`
- `analysis/figures/17C_<dimension>_one_bit_frontier_untreated_item_difficulty_concordance.png`
- `analysis/figures/17D_<dimension>_one_bit_frontier_sft_full_item_difficulty_concordance.png`
- `analysis/figures/S17_<dimension>_<representation_group>_<adapter>_item_difficulty_concordance.png`

## 1.10 Haiku self-evaluation

### Plain-language purpose

Haiku's scores on its own responses show how the judge behaves when the candidate closely resembles its preferred style and reference. They help readers calibrate the later candidate scores, but they do not show how accurate Haiku is.

Repeat the same per-dimension, attack-class, failure, format and prompt-length summaries on the 285 Haiku self-evaluation rows. Compare with candidate models only on identical prompt IDs.

Do not calculate prediction/reasoning similarity for Haiku self-evaluation. Do not use Haiku self-scores to validate Haiku. Present them in a visually separate baseline panel labelled `judge self-evaluation (circular descriptive baseline)`.

The low self-evaluated instruction-adherence result must not be described as arising from an omitted output-format instruction. Haiku reference generation received each original CAM-LDS prompt verbatim, including the explicit requirement for one valid JSON object containing `techniques`, `confidence` and `explanation`. The stored references instead used Markdown-fenced JSON in 987/990 cases; strict whole-string auditing consequently marks all 990 as invalid/schema-nonadherent even though their inner JSON is generally recoverable. On the 285-row self-judgment pass, instruction adherence has mean 1.384, with 1 row scored 0, 171 scored 1, 109 scored 2, 4 null and none scored 3–4. Judge reasons frequently penalize the fence despite the frozen rubric saying minor fences must not cause recoverable substance to be discarded. Treat this as an internal judge-instruction inconsistency and circularity diagnostic, not evidence that the reference request omitted the required format.

The purpose is reader calibration: show how generously or harshly Haiku scores its own reference-like responses, whether self-scores occupy a different ordinal distribution, and whether similarity to Haiku expression is associated with candidate scores. This is a circular descriptive benchmark, a rubric-behaviour diagnostic and a possible indicator of self/reference preference, not a performance gold standard.

Dietz et al.'s circularity and "LLM narcissism" concerns and G-Eval's warning that LLM evaluators may prefer LLM-generated text must be stated in the figure caption and notebook discussion. Use existing format diagnostics only as secondary context.

Do not call the Haiku-versus-candidate score gap a self-preference effect. A higher self-score can reflect genuine quality differences, stylistic familiarity, reference anchoring or leniency. Chen et al. (2025) show that raw self-score gaps confound preference with response quality; without independent human quality labels, notebook 10 can only report a circular descriptive benchmark and bias-consistent associations.

## 1.11 Judge-independent response diagnostics

### Plain-language purpose

These diagnostics describe observable response behaviour—such as empty output, format adherence, repetition or reaching the token limit—without asking Haiku whether the answer is good. They are useful checks, but malformed formatting does not necessarily mean the semantic answer is wrong.

Judge-independent work is deliberately secondary and must not be described as a substitute quality evaluation. Notebook 10 performs no new parsing of model responses. It may summarize only deterministic columns already exported by notebooks 01--07.

Most defensible secondary diagnostics are:

1. exact response-hash stability on duplicated prompts;
2. EOS versus output-limit behaviour;
3. output length and already-recorded repetition/degeneration indicators;
4. already-exported JSON/schema/required-field rates, always with observed denominator; and
5. existing strict-parser ATT&CK metrics, if present, only on their original parseable subset.

Do not add permissive regular-expression extraction, new JSON recovery, new technique parsing or new confidence parsing. Strict-parser metrics are labelled `weak deterministic sensitivity analysis` because poor adherence is concentrated in particular models, treatments and representations; their parseable subset is therefore selected and cannot support a fair headline comparison. Report coverage beside every such metric and never treat an unavailable or malformed response as an incorrect semantic prediction.

Do not make unique-token, character-n-gram or strict-schema statistics headline quality measures. Repetition statistics quantify degeneration and length-limit behaviour only. TF-IDF remains a lexical-similarity appendix, not a correctness or semantic-grounding metric. Semantic embeddings would introduce another learned evaluator and are not required for the primary notebook.

## 1.12 Token-ceiling and stopping-behaviour analysis

### Plain-language purpose

Generating all 1,024 allowed tokens can make a run much slower and may indicate that the model did not naturally finish. This section separates that behaviour from prompt length, semantic score and GPU speed.

Quantized streams frequently reached the 1,024-token generation ceiling, increasing latency and sometimes ending otherwise usable text at the limit. Produce a dedicated table and figure by model, condition and representation containing:

- completed and observed row counts;
- natural-EOS count and rate;
- length/output-limit count and rate;
- recorded truncation and timeout count and rate;
- median, IQR and p95 output tokens;
- count and rate at exactly the configured ceiling; and
- median row latency for EOS versus ceiling-terminated rows.

Use a 2 by 3 model grid of horizontal rate bars with BF16 and one-bit-frontier representations adjacent, plus a concise table of token and latency summaries. This is an operational-behaviour result, not a semantic quality score. The notebook notes must record that final-row processing for some Mistral7B and Qwen7B TQ1_0 streams became infeasible within the available research time because repeated long generations/timeouts and later GPU availability constraints consumed disproportionate time; these streams were stopped with high but incomplete coverage so that the Haiku evaluation could still be completed.

Outputs:

- `analysis/tables/token_ceiling_and_stop_reasons.csv`
- `analysis/figures/14_token_ceiling_rates.png`

## 1.13 Runtime-revision sensitivity

### Plain-language purpose

Some long quantized runs changed slot counts or prompt ordering to improve completion time. Those changes were operational rather than learned-model treatments, but they could still coincide with different prompt populations or small numerical differences.

Within every stream containing more than one `runtime_configuration_revision`, report coverage, prompt-length distribution, output length, stop reason and judge-score distributions by revision. Repeat the main stream summary on the dominant revision only where sufficient common rows exist. Include revision and prompt-length bucket in secondary row-level models.

Do not interpret revision differences causally. Revisions were introduced after telemetry observations, and length-ordered scheduling means revision is confounded with prompt difficulty and collection time. Their purpose is sensitivity and provenance, not estimating the effect of parallel slots.

Outputs:

- `analysis/tables/runtime_revision_sensitivity.csv`
- `analysis/figures/15_runtime_revision_sensitivity.png`

## 1.14 Reference ladder

### Plain-language purpose

The reference ladder places each recorded model stream between deliberately weak baselines and the GPT-5.5 distillation target. It asks whether a stream performs above an empty answer or a simple training-set-frequency heuristic and how far it remains from the teacher under the same Haiku judge. The ladder is descriptive: the baselines are not validated labels or estimates of human performance.

### Judging and population

Notebook 07 judges `gpt55_teacher`, `baseline_null` and `baseline_modal` with exactly the same `judge_protocol`, rubric hash, Haiku reference response, ground truth, model alias, one-candidate context and eligibility rule used for all other candidates. The teacher candidate is the exact stored response. The null candidate is an empty but completed response. The modal candidate predicts `T1105`, selected before reference scores are analysed as the modal exact technique over the 604 frozen training-only feasible rows; validation/test labels do not define it.

The analysis pool is the frozen 285 validation+test prompt IDs. Reference rows receive no special score handling: only canonical scalar 0--4 judge fields are eligible under Section 1.1. For absolute ladder estimates, each stream uses its eligible rows within the 285-row pool and reports `eligible_n/285`. Pairwise quantities elsewhere continue to use exact common support.

### Calculation

For every score dimension and stream, calculate:

- mean judge score;
- 95% scenario-step cluster-bootstrap percentile interval using 10,000 replicates and seed 42;
- count and proportion with score at least 3; and
- eligible and unavailable counts.

Streams comprise the two trivial baselines, GPT-5.5 teacher, and the four core streams for each of the six model families. Reference streams are never pooled with either representation group.

### Representation and output

Write `analysis/tables/reference_ladder.csv`. Retain one low-dimensional dot-and-interval appendix panel per judge dimension under the Figure 19A series. Order the four core streams within model order, followed at the bottom by the teacher, modal baseline and null baseline. The exact canonical stream label must accompany every row. Draw vertical reference lines at the eligible-sample teacher and modal-baseline means for that dimension. A companion marker or aligned column shows the proportion scoring at least 3; it must not share an unlabeled dual axis.

Add the main-text `19A_all_reference_ladder_all_fields.png`: seven narrow, horizontally aligned panels for TI, PES, EG, PRA, CES, IA and CLA, sharing the same 27-stream y-axis and fixed 0--4 x-axis. Each point is the mean with its 95% scenario-step cluster-bootstrap interval. The three references remain at the bottom and every panel carries teacher and modal-baseline vertical reference lines. The combined view omits the proportion scoring at least 3 to preserve portrait-page legibility; that quantity remains in `reference_ladder.csv` and the per-dimension appendix figures. An exact zero is hollow, fewer than 30 scorable rows is grey, and no scorable rows has no point and is printed as an en dash. Colours distinguish BF16 untreated, BF16 SFT-full, one-bit untreated and one-bit SFT-full; the y-axis gives a short model name and a second-line representation/treatment label. A single legend appears below the figure. Notebook Section 6A2 renders this view from the completed 10,000-replicate ladder table in a lightweight process, avoiding a second load of the full judgment matrix.

The auditable companion file `analysis/tables/figure_data/19A_all__19A_all_reference_ladder_all_fields.csv` is a long-form 285-pool extract with one row per stream and dimension. It includes the interval inputs already calculated in `reference_ladder.csv`, the panel abbreviation, display order, short label, colour and marker state. The figure manifest records this file and the canonical judgment sources.

The caption states: 285-row validation+test pool; null/modal/teacher definitions; eligible denominators; scenario-step cluster-bootstrap method; and **Does not establish: the trivial baselines are not validated labels, and Haiku scores are not human ground truth.** Interpret left-to-right movement within a panel as a higher mean on that one rubric dimension only. Do not combine horizontal positions across dimensions into a composite score.

## 1.15 Retention and gap decomposition

### Plain-language purpose

This analysis asks how much of the SFT-full advantage above the modal baseline remains after one-bit deployment. It separates the observed advantage retained by the deployed stream from the portion lost between BF16 SFT-full and one-bit SFT-full, while showing how far the untreated BF16 model already sits above or below the modal baseline.

### Population and calculation

For each model and score dimension, restrict to exact prompt IDs with eligible scores for `baseline_modal`, BF16 untreated, BF16 SFT-full and one-bit SFT-full. Let their paired means be (M), (U), (S) and (Q). Calculate:

- retained gap: (Q-M);
- deployment loss: (S-Q);
- BF16 SFT gain above modal: (S-M);
- contextual untreated gap: (U-M); and
- retention index: (R=(Q-M)/(S-M)).

Do not clamp (R): values below 0 or above 1 reveal reversal or apparent amplification. If `S-M` is zero or numerically indistinguishable from zero, report `R` as not estimable and retain the component gaps. Calculate percentile intervals by resampling scenario-step groups and recomputing the complete ratio/decomposition within every replicate.

### Representation and output

Write `analysis/tables/retention_index.csv`, retaining model, dimension, common-support `n`, independent group count, all four paired means, all component gaps, (R), interval bounds and non-estimable reason. Figure 19B is a forest plot with one row per model for each dimension; the main mark is (R), and aligned context columns show `deployment_loss`, `retained_gap` and `BF16_untreated_minus_modal`. A vertical line at 1 means full observed retention and a line at 0 means no observed gain above modal remains.

This is a deployment-stack comparison, not a pure effect of numerical precision: BF16 and one-bit streams also differ in backend, kernels and scheduling.

## 1.16 Size analysis extension

### Plain-language purpose

The strongest available size comparison is within Qwen2.5: the 7B and 1.5B models share a family and are evaluated on identical prompts. Suite-level associations remain secondary because architecture, tokenizer, training corpus and model family change with size.

### Primary paired contrast

For every core stream and judge dimension, pair Qwen2.5 7B and Qwen2.5 1.5B on exact eligible prompt IDs and calculate `7B - 1.5B`, its scenario-step cluster-bootstrap interval, win/tie/loss proportions, probability of superiority with ties, common-support `n` and independent group count. Show the MDE for that contrast as a horizontal interval centred on zero in Figure 16C so readers can distinguish an imprecise null result from evidence of practical equivalence.

### Secondary suite-level association

For all six models, plot stream-specific mean score and retention index against `log10(parameter_count)`. Do not fit or report regression p-values across six points. Label every point and retain representation/treatment facets. OpenHermes 1.8B shares a Qwen 1.5 lineage but is not an identical architecture; it must not be described as a controlled Qwen size comparison.

Write `analysis/tables/size_contrasts.csv`, including the primary paired contrasts and clearly tagged secondary descriptive points. Existing Figure 16A/16B and all prior figure numbers remain unchanged; Figure 16C is additive.

## 1.17 MDE and non-inferiority planning table

Before teacher or baseline scores are inspected, set the practical non-inferiority margin to **delta = 0.3 judge-score points** and record it in `analysis/analysis_manifest.json`. For every paired contrast, form one mean paired difference per scenario-step group. Report:

- observed group-level standard deviation `sigma_g`;
- independent group count `G`;
- two-sided 5% alpha, 80% power approximate MDE, `MDE=(1.96+0.8416)*sigma_g/sqrt(G)`;
- pre-specified `delta=0.3`; and
- approximate groups required, `ceil(((1.96+0.8416)*sigma_g/delta)^2)`.

Write `analysis/tables/mde_and_noninferiority.csv`. The table is a design/sensitivity aid, not a post hoc proof of equivalence. Non-inferiority is supported only where the appropriately oriented 95% paired interval lies wholly above `-0.3`; otherwise report it as not established. Missing score pairs are never imputed.

## 1.18 Distillation fidelity

### Plain-language purpose

This table describes how far each SFT adapter remains from the exact teacher target under the shared judge. It is not a direct training-loss measure and does not imply that matching the teacher is always correct.

For each model, representation group, adapter and score dimension, pair the SFT candidate and GPT-5.5 teacher on exact eligible evaluation prompt IDs and calculate `SFT - teacher`, its scenario-step cluster-bootstrap interval, median and quartiles, win/tie/loss, probability of superiority with ties, common-support `n` and independent groups. Include BF16 and one-bit adapters separately, retaining exact representation. Write `analysis/tables/distillation_fidelity.csv`. No figure is required.

## 1.19 Judge-mediated ranked-technique accuracy

### Plain-language purpose

The rubric scores describe overall answer quality, but they do not directly answer whether a candidate named the ground-truth ATT&CK technique near the top of its answer. This secondary analysis uses the ordered technique list that Haiku already extracted during judging. It therefore remains robust to malformed candidate JSON without introducing a new parser, while still depending on Haiku's interpretation of the candidate.

### Input contract and matching rule

Use only the canonical `judge_extracted_predictions` and `ground_truth` fields already present in the judgment CSVs. Notebook 10 may deserialize these two schema-declared arrays; it must not inspect candidate prose, judge prose or embedded reasons to recover additional labels. Preserve prediction order, remove repeated identical IDs after their first occurrence, and retain at most the first ten distinct predictions. Matching is case-normalized exact ATT&CK-ID equality. Parent and sub-technique IDs remain distinct; for example, `T1059` does not match `T1059.001`. A syntactically invalid predicted ID remains a non-matching prediction rather than being silently repaired.

Exact-ID matching is the primary analysis. A separately labelled **parent-normalized sensitivity analysis** collapses every syntactically valid `Txxxx.xxx` sub-technique to `Txxxx` in both predictions and ground truth, then removes duplicates created by that collapse while preserving first-appearance prediction order. Invalid labels remain unchanged non-matches. This sensitivity analysis asks whether conclusions change when only parent-technique recognition is required; it must never be pooled with or substituted for exact sub-technique results.

Let the ordered predictions be `P=(p_1,...,p_m)`, the unique non-empty ground-truth set be `G`, and `P_k` be the first `min(k,m)` predictions. For each eligible row calculate:

- `Hit@k = 1` when `P_k` intersects `G`, otherwise `0`, for every `k=1,...,10`;
- fixed-denominator `Precision@k = |P_k intersect G| / k` for `k in {1,3,5,10}`; unfilled ranks count as non-relevant, so a one-label answer cannot receive `Precision@10=1`;
- `Recall@k = |P_k intersect G| / |G|` for `k in {1,3,5,10}`.

Report row-weighted means for Hit and Precision, both macro recall (mean row recall) and micro recall (`sum matched ground-truth IDs / sum ground-truth IDs`). The fixed precision denominator must be named in every table and caption because it intentionally penalizes short lists at larger `k`.

### Extraction states and denominators

Every row receives one of four extraction states:

| State | Meaning | Primary metric treatment |
|---|---|---|
| `nonempty_list` | Haiku returned one or more extracted predictions | included |
| `empty_list` | Haiku explicitly returned `[]`, meaning no affirmative candidate technique was recoverable | included as zero |
| `missing_or_null` | no usable extraction field was returned | unavailable and excluded |
| `malformed_non_list` | the canonical extraction field is structurally invalid | unavailable and excluded |

The coverage table must report all four states. A sensitivity column additionally treats missing/null/malformed states as zero, but this conservative result must not replace the primary estimate. Ground truth must contain at least one ID; otherwise the row is unavailable. Restricting analysis to non-empty prediction lists is prohibited as the primary analysis because it would condition on successful extraction and severely distort the one-bit-frontier results.

### Population, uncertainty and contrasts

Produce results for all six models, both representation groups, and all five conditions (`untreated`, `technique`, `confidence`, `explanation`, `full`). Absolute stream summaries use every available judged row, including partial streams, with intended and observed support reported together. Treatment contrasts use exact common prompt support:

1. each SFT adapter minus untreated within BF16 and within the one-bit frontier; and
2. one-bit minus BF16 for the same condition.

Intervals use 10,000 scenario-step cluster-bootstrap replicates with seed 42, consistent with the other judge analyses. The bootstrap resamples scenario-step groups and recomputes each metric; it does not treat five variants from the same scenario-step as independent observations.

### Outputs and figures

Write:

- `analysis/tables/judge_mediated_rank_metrics_by_row.csv`: auditable row-level predictions, truth, extraction state and all metric values;
- `analysis/tables/judge_mediated_rank_metric_coverage.csv`: extraction-state and eligible-denominator counts by stream;
- `analysis/tables/judge_mediated_rank_metrics_by_stream.csv`: stream estimates and cluster-bootstrap intervals;
- `analysis/tables/judge_mediated_rank_metric_contrasts.csv`: exact-common-support paired contrasts;
- `analysis/tables/judge_mediated_parent_rank_metrics_by_stream.csv`: parent-normalized stream estimates and intervals;
- `analysis/tables/judge_mediated_parent_rank_metric_contrasts.csv`: parent-normalized exact-common-support paired contrasts;
- `analysis/tables/judge_mediated_technique_recall.csv`: populated backward-compatible recall view replacing the former empty placeholder.

Main Figure 19C shows exact-ID Hit@1--10 for the four core streams per model. Supplementary Figure S19C extends exact-ID Hit@k to every adapter; Figure S19D shows exact-ID fixed-denominator Precision@1/3/5/10; Figure S19E shows exact-ID macro Recall@1/3/5/10; and Figure S19F plots parent-normalized minus exact Hit@k so the effect of relaxing sub-technique specificity is visible. Every figure has a plotted-data CSV and manifest entry.

### How to interpret

A higher Hit@k means the judge found at least one correct technique within the first `k` candidate predictions. Precision@k asks how much of a fixed top-`k` list is correct, while Recall@k asks how much of a potentially multi-label ground truth has been recovered. These are **judge-mediated classification diagnostics**, not deterministic parser accuracy: they combine candidate prediction quality with Haiku's ability to recover ordered labels from the answer. The extreme BF16/one-bit extraction imbalance must therefore be read alongside the coverage table. They remain secondary to the complete rubric-score analysis and do not validate the judge against human annotation.

Parent-normalized values can only equal or relax the semantic specificity of exact matching, although fixed-rank precision may also be affected by duplicate collapse. A positive parent-minus-exact difference indicates that some apparent errors were parent/sub-technique disagreements. It does not demonstrate correct sub-technique identification and must be described as sensitivity evidence rather than improved primary accuracy.

## 1.20 Representativeness of the context-feasible training subset

### Plain-language purpose

SFT used 604 of the 705 training rows because the complete prompt-plus-teacher example had to fit every model. This audit asks whether the 101 universally excluded rows differ systematically from the retained rows. It is necessary because a context-feasible subset can overrepresent short prompts or particular scenarios even though the exclusion rule was applied consistently across models.

### Inputs and calculations

Use the frozen `data/prompts.csv`, `data/split_manifest.csv`, `data/context_feasibility.csv` and `data/training_exclusions.csv`. Hash these files in the Notebook 10 manifest. Define `excluded` from the universal exclusion IDs and never infer it from prompt length alone.

Produce:

- `training_subset_representativeness_rows.csv`: one row per original training prompt with split identity, scenario-step group, prompt hash, retained/excluded status and raw text-length measures, but without duplicating full prompt text;
- `training_subset_model_feasibility.csv`: model context, training context, model-specific non-fit count, share of the 101 universal exclusions and required-token range;
- `training_subset_length_comparison.csv`: retained/excluded count, mean, median, 90th percentile and maximum for raw characters, words and lines, plus prompt, teacher-ending and complete-example tokens under every model tokenizer; include the standardized mean difference as a descriptive scale-free contrast;
- `training_subset_scenario_comparison.csv`: retained, excluded and total rows plus exclusion rate for every scenario;
- `training_subset_group_coverage.csv`: retained/excluded counts and `all_retained`, `mixed` or `all_excluded` status for every scenario-step group.

No significance test is required. These are the complete frozen training population rather than a random sample, so magnitude, coverage and distributional distortion are more informative than a null-hypothesis p-value.

### Figure 18F

Create one three-panel supplementary figure:

1. an ECDF of BitNet complete-example tokens for retained and excluded rows with the 4,096-token training-context boundary;
2. scenario exclusion proportions labelled with excluded/total counts; and
3. model-specific counts of training examples exceeding each model's limit.

All panels derive from the five audit tables and have a companion figure-data CSV. The figure shows whether exclusion was length-selective, whether it was concentrated by scenario and which model constraint drove the universal policy.

### Interpretation boundary

This audit does not repair the training subset or estimate what results would have been under long-context training. If scenarios or prompt lengths are unevenly removed, SFT findings apply to the implemented 604-row feasible population. The 285-row evaluation pool remains unchanged, so this is a training-exposure limitation rather than evaluation-row deletion.

## 1.21 Unseen techniques and split balance

### Plain-language purpose

This audit asks whether evaluation requires exact ATT&CK IDs absent from SFT training and whether train, validation and test contain comparable scenarios, numbers of labels and prompt lengths. It also reports how many genuinely distinct prompt texts are present, because repeated prompt rows should not be mistaken for independent textual diversity.

### Populations and exact-label rule

Report five named populations: original train (705), context-feasible train (604), validation (140), test (145), and combined evaluation (285). The unseen-technique analysis uses the 604-row feasible train set because this is what every adapter actually observed. Compare exact case-normalized ATT&CK IDs; parent normalization is not used for the primary unseen-label count.

### Outputs

- `split_population_summary.csv`: rows, exact distinct prompt hashes, duplicate rows, scenario-step groups, scenario count, distinct exact technique IDs, label multiplicity and prompt-character summaries for all five populations;
- `split_scenario_balance.csv`: scenario counts and within-population proportions;
- `split_label_multiplicity.csv`: counts and proportions by number of ground-truth IDs per row;
- `split_prompt_length_comparison.csv`: raw character/word/line and every-model tokenizer prompt-length summaries by population;
- `evaluation_unseen_techniques.csv`: exact technique ID, feasible-training frequency, evaluation/validation/test frequencies and unseen flag;
- `evaluation_unseen_technique_rows.csv`: every evaluation prompt containing at least one unseen exact ID, preserving row/group/split identity and the unseen subset.

Distinct prompts are defined by `prompt_sha256`, not by scenario-step identity. The technique tables count a technique at most once per row.

### Figure 18G

Produce a three-panel supplementary figure comparing original train, feasible train, validation and test:

1. scenario proportions;
2. label-multiplicity proportions; and
3. raw prompt-character ECDFs on a log x-axis.

All panels use population proportions where group sizes differ, while the companion tables retain raw counts. Validation and test remain visible separately even though downstream model analysis treats their union as the 285-row evaluation pool.

### Interpretation boundary

An unseen exact ID indicates zero SFT exposure to that label in the 604 feasible rows, not necessarily zero pretraining knowledge. Conversely, a seen label may have very little training support. Split imbalance is descriptive and should qualify later model comparisons; it does not establish that scenario or prompt length caused a performance difference.

# 2. Compute-usage reporting

This section reports what was consumed and where time was spent. It does not test performance hypotheses.

**Plain-language purpose.** A resumed run may include downloading a model, rebuilding an artifact, waiting, uploading checkpoints and performing actual inference. These activities must be reported separately so operational overhead is not mistaken for model generation cost.

## 2.0 Source-to-calculation contract

Compute figures must not be omitted merely because an already-aggregated table lacks the desired grouping. Notebook 10 reconstructs reportable aggregates from the most granular canonical source available, and records that source in `figure_manifest.csv`:

| Reporting need | Canonical source | Required calculation |
|---|---|---|
| retries, resumes, phase time, load/restore/preparation time | `results/final/compute/invocations/09__compute__invocation_inventory.csv` | deduplicate identical telemetry-content hashes; classify invocation phase; sum consumed wall time within exact stream and phase; retain invocation count and failed/noncompleted count |
| per-row latency, tokens and throughput | `results/final/compute/rows/09__compute__row_performance.csv` | aggregate completed experimental rows by exact stream; report median/p95 row time, token totals and rates using the corresponding invocation resource-time denominator |
| intended/completed/error coverage and retry burden | `results/final/compute/rows/09__compute__coverage_and_retries.csv` | preserve terminal-state counts; divide only by the exact intended stream denominator |
| exact-stream resource summary | `results/final/compute/stream_summary/09__compute__stream_summary.csv` | audit against invocation reconstruction; never use a pre-aggregated value when its phase definition conflicts with the invocation ledger |
| model-level phase totals | `results/final/compute/stream_summary/09__compute__model_resource_summary.csv` | use as a cross-check and concise model report, not as a replacement for exact-stream aggregation |
| API requests/tokens/cache/cost | `results/final/compute/stream_summary/09__compute__api_usage_summary.csv` | aggregate separately by reference generation, Phase 1 judging, self-evaluation and Phase 2 judging |
| raw samples or values absent from flattened exports | `telemetry/` records referenced by the inventory | read only the declared telemetry fields; do not infer absent utilization, memory, power or component timers |

Every compute table includes `source_file`, aggregation level, phase definition, observed/applicable flags and units. If a metric is not collected for a backend, the corresponding figure panel says `not observed`; Notebook 10 does not generate an empty unlabeled plot or substitute zero.

## 2.1 Invocation accounting and resumed runs

Each saved telemetry invocation represents one attempt. Successful resumptions and failed starts consumed real resources, so they remain in the audit while duplicated copies of the same telemetry record are excluded.

The invocation inventory is the audit base. Each telemetry file remains one invocation. Retries and resumes are **not deduplicated away**; they are summed within the exact `(model, condition, representation, phase)` stream because they consumed resources. Identical telemetry-content hashes are flagged by `duplicate_content_hash_count` and must be reviewed before aggregation so duplicated copies are not double-counted.

Separate:

- training;
- quantized artifact preparation and upload;
- artifact/model restore and synchronization;
- model/server load;
- experimental inference invocation wall time;
- pilot/benchmark time;
- failed-start/retry overhead; and
- Haiku API use.

Pilots remain separate from experimental totals. The model summary is resource consumption across streams, never calendar elapsed time across concurrent Colab sessions.

For every optional telemetry metric, derive three columns from the raw telemetry record rather than treating absent values as zero:

- `<metric>_observed`: the source record actually contained the metric;
- `<metric>_applicable`: the metric was expected for that phase/backend; and
- `<metric>_value`: the recorded numeric value, otherwise null.

This rule is essential because HF inference telemetry and quantized llama-server telemetry expose different component timings. An unobserved HF generation timer is unknown, not zero and not evidence that GGUF generation took longer.

## 2.2 Genuine inference versus operational work

The main reporting distinction is simple: preparation and movement make a model ready; inference processes experimental rows. Component timers may overlap, so they cannot all be added together.

The existing compute export supports separation through:

- `phase` and `invocation_kind`;
- `training_wall_seconds`;
- `preparation_and_upload_wall_seconds`;
- `model_cache_restore_download_sync_seconds`;
- `model_load_seconds`;
- `inference_and_io_wall_seconds`;
- `generation_seconds`;
- `local_checkpoint_seconds`;
- `receiver_seconds`; and
- `csv_snapshot_seconds`.

Important accounting rule: these component timers are not automatically mutually exclusive. Generation may occur concurrently across slots; receiver uploads may be asynchronous; generation is a subset of inference-and-I/O; and checkpoint/receiver work may overlap wall time. Therefore:

- use invocation `wall_seconds` for total resource time;
- use `phase` to produce mutually exclusive top-level totals;
- never sum generation, inference-and-I/O, receiver and CSV times into a second total unless the instrumentation proves they are non-overlapping;
- show overlapping component timers as diagnostic measures or ratios, not additive stacked segments; and
- calculate an `unattributed_or_overlapping` flag rather than forcing a residual when components exceed wall time.

Artifact transfer is assigned to quantized preparation only where instrumented. It is never assigned to inference throughput. Historical uninstrumented transfer remains unknown.

Use the term **GPU-runtime session hours** for invocation wall time divided by 3,600 while a GPU runtime was allocated. This is not active floating-point compute and may include downloads, CPU conversion, upload, synchronization, waiting or low-utilization intervals. Report sampled GPU utilization and any explicitly defined sampled-active proxy separately; never rename session time as active GPU hours.

## 2.3 SFT reporting

SFT reporting answers how much resource was used to create each adapter, separately from the later cost of asking that adapter to generate responses.

Report SFT training separately by model and adapter:

- invocation count, retries/resumes and final status;
- total training invocation wall time;
- checkpoint-transfer-inclusive training time where recorded;
- model restore/load time;
- GPU-runtime session hours;
- sampled mean utilization and peak memory;
- training examples, epochs and optimizer steps where present;
- final and minimum recorded training loss; and
- adapter artifact bytes.

Use a table plus two figures:

- horizontal bars of training GPU-runtime session hours by model/adapter; and
- loss curves by step, in a 2 by 3 model grid with adapters as separate lines.

Do not mix SFT response inference time with adapter training time.

## 2.4 Quantization/preparation reporting

Quantization reporting covers merging, conversion, toolchain work, upload and verification. It is deployment preparation, not response-generation speed.

Report by model, condition and representation:

- preparation invocation count and restarts;
- preparation and upload wall time;
- source restore/download synchronization time;
- toolchain compilation/restoration time where instrumented;
- GGUF exact bytes and compression ratio to BF16;
- target tensor-type counts;
- upload status and hash verification; and
- GPU-runtime session hours, sampled utilization and peak memory during preparation.

Use a table and simple horizontal bars for preparation time and artifact size. The main panel is an inclusive six-model one-bit-frontier view; exact TQ1_0/I2_S method is printed beside every bar. A supplementary method-specific panel separates TQ1_0 and I2_S preparation because their conversion procedures differ.

## 2.5 Inference reporting

Inference reporting shows how many experimental rows were processed, how much text was read and generated, how long it took, and why some rows did not complete.

Report per exact stream:

- expected, completed, `context_unsupported`, retryable-error and missing rows;
- invocation count and noncompleted invocations;
- attempts and retries;
- total experimental invocation wall time;
- inference-and-I/O wall time where recorded;
- model/server load and restore time;
- input and output tokens;
- successful rows per GPU-runtime session hour;
- operational output tokens per resource second;
- prompt/input tokens per GPU-runtime session hour;
- generated tokens per GPU-runtime session hour;
- median and p95 row latency;
- time-weighted GPU utilization;
- sampled peak GPU memory; and
- runtime revisions and GPU types.

Show candidate-generation coverage and judge-score coverage separately. Candidate coverage classifies completed response, `context_unsupported`, timeout/retryable error and missing. Judge coverage then shows, for each dimension, how many completed candidate rows received an already-materialized scalar score. This makes clear whether loss occurred during model inference, judgment, or score materialization.

For every partial stream, add an operational reason and evidence source from telemetry/status records. In particular, distinguish context-window exclusion from repeated long-generation timeouts, GPU/runtime loss and researcher stopping due to the remaining time budget. The Mistral7B and Qwen7B Phase 2 streams that were ended after high coverage are reported as abandoned/incomplete rather than silently treated as full samples. Compute tables report the resource cost of their retries/resumes and permit comparison of the common-support subset with the intended 285-row pool.

Primary figures:

1. coverage bars by stream;
2. inference resource-hours by stream;
3. output-token throughput dot plot;
4. GPU utilization versus peak-memory scatter; and
5. retry/resume overhead bars.

Use small multiples by representation and model. The main resource-hours figure aggregates to model/phase; exact-stream bars move to a supplementary figure. Throughput uses three aligned panels—rows/hour, prompt tokens/hour and generated tokens/hour—rather than multiple units on one axis. The GPU utilization/memory scatter labels only outliers and facets by GPU type. Coverage and retry burden use two aligned panels rather than an overlaid dual-scale chart. Do not combine preparation and inference into one bar without visibly separating them.

Always show rows/hour beside token throughput and realized output length. A model that stops early may complete more rows, while a model that repeatedly reaches 1,024 tokens performs more generation work. No single throughput measure is treated as definitive.

### Concise per-stream resource table

Produce `compute_stream_resource_summary.csv` with one row per model, representation group, exact representation and treatment. The headline view contains the four core streams; the same table retains technique, explanation and joint adapter rows for supplementary reporting. Each row contains one clearly defined value for:

- total GPU-runtime session hours across every retry/resume invocation;
- mean invocation GPU-runtime session minutes and invocation count;
- experimental inference session hours, excluding preparation/restore where phase instrumentation permits;
- preparation/upload, restore/load and failed-start session hours separately;
- completed/intended rows and retry count;
- successful rows per session hour;
- input and output token totals and generated tokens per session hour;
- mean, median and p95 experimental row latency;
- time-weighted mean sampled GPU utilization;
- maximum observed sampled peak GPU memory, in GiB;
- mean sampled power and sampled energy estimate only where actually observed and methodologically documented; and
- GPU type, runtime revision and metric-observed flags.

Totals answer how much resource the stream consumed; means describe a typical invocation; maximum memory answers capacity requirements. These statistics are not interchangeable. Artifact upload, restore and failed-start time remain visible and are never included in the numerator called `experimental inference session hours`.

## 2.6 API reporting

API reporting separates generating Haiku reference answers from using Haiku as a judge. Prompt-cache tokens are reported as an efficiency mechanism, not as fewer experimental cases.

Report Haiku reference generation, Phase 1 judging, Haiku self-evaluation and Phase 2 judging separately:

- requests completed;
- input and output tokens;
- cache creation/read tokens;
- estimated cost; and
- failed/held requests where available.

Prompt caching is a transport/computation optimization, not a changed evaluation context. Report cache token counts but do not treat cached static instructions as fewer experimental cases.

## 2.7 Reporting figures and tables

All Figures 20--26 carry their figure identifier in the rendered title. Main panels show the four core streams for all six models; supplementary panels repeat exact-stream values for technique, explanation and joint adapters. One-bit-frontier panels include native-I2_S BitNet and TQ1_0 models together while retaining exact representation in subtitles and companion CSVs.

**How to interpret Figures 20--26.** Figure 20 reports training resource time per adapter; Figure 21 shows recorded loss by training step where loss telemetry exists; Figure 22 reports one-time deployment preparation and artifact size; Figure 23 separates mutually exclusive phase-level session time; Figure 24 reports row, prompt-token and generated-token throughput together; Figure 25 shows observed utilization and capacity pressure rather than active compute; and Figure 26 distinguishes final coverage from retry/resume burden. Longer resource-time bars mean more allocated runtime was consumed, not necessarily more GPU arithmetic. Missing telemetry appears as `not observed`, not zero.

- `analysis/tables/compute_stream_resource_summary.csv`
- `analysis/tables/compute_invocation_audit.csv`
- `analysis/tables/compute_sft_training_report.csv`
- `analysis/tables/compute_quantization_report.csv`
- `analysis/tables/compute_inference_report.csv`
- `analysis/tables/compute_api_report.csv`
- `analysis/figures/20_training_resource_hours.png`
- `analysis/figures/21_training_loss_curves.png`
- `analysis/figures/22_quantization_time_and_artifact_size.png`
- `analysis/figures/23A_model_phase_resource_hours.png`
- `analysis/figures/S23B_exact_stream_resource_hours.png`
- `analysis/figures/24A_rows_per_hour.png`
- `analysis/figures/24B_prompt_tokens_per_hour.png`
- `analysis/figures/24C_generated_tokens_per_hour.png`
- `analysis/figures/25_gpu_utilization_and_memory.png`
- `analysis/figures/26A_stream_coverage.png`
- `analysis/figures/26B_retry_resume_burden.png`

# 3. Compute analysis

This section studies efficiency and resource/outcome relationships. It keeps SFT training, artifact preparation and inference analytically separate.

**Plain-language purpose.** Reporting says what resources were used. Compute analysis asks why some streams were slower, larger or more memory intensive and whether additional resource use corresponded with different judged outcomes.

Every Figure 30--36 is built from the Section 2 exact-stream resource table joined to response/judgment outcomes by `model + representation_group + exact_representation + treatment + prompt population`. For partial streams, response-outcome comparisons use exact common-support prompt IDs, while resource totals continue to report all consumed retries/resumes. The figure-data CSV retains both denominators so incomplete result coverage cannot make an expensive stream appear efficient.

Where an existing `09__compute__*.csv` does not contain the requested aggregate, calculate it from the invocation and row sources specified in Section 2.0. Do not leave a planned figure blank simply because it requires regrouping. A panel is omitted only when the underlying metric was never collected; the manifest then records `not observed`, the affected backend and the source files checked.

## 3.1 Model size versus inference efficiency

Larger parameter counts may require more memory and reduce throughput, but the observed models also differ in architecture and backend. These plots describe the suite rather than establish a universal scaling law.

### Measures

- output tokens per resource second;
- successful rows per GPU-runtime session hour;
- median and p95 latency;
- prompt-evaluation rate where backend timings are available;
- generation rate;
- peak GPU memory; and
- time-weighted GPU utilization.

### Method

Plot each measure against `log10(parameter_count)`. Use one figure per measure with a 2 by 2 grid of the four core streams and all six model points in every applicable panel. The one-bit panels include native-I2-S BitNet and TQ1_0 models under the shared frontier category, with exact representation encoded redundantly in the point annotation. Supplementary figures show technique, explanation and joint adapters. Keep identical axes across panels and label model points directly.

Because only six non-identical model families exist, regression lines are descriptive. Do not report them as scaling laws. A row-level secondary model may adjust latency for input tokens, output tokens, GPU type, slots and runtime revision, but the number of independent model-size units remains six.

## 3.2 BF16 versus one-bit-frontier deployment

This comparison asks what changed when moving from the HF BF16 deployment stack to TQ1_0 or native I2_S deployment. It includes both representation and backend effects.

Use paired prompt IDs and common support to compare:

- exact artifact bytes and compression ratio;
- peak memory;
- prompt and generation throughput;
- row latency;
- EOS versus limit behaviour;
- coverage/timeouts;
- every judge field separately; and
- GPU-runtime session hours per completed response.

Figures:

- paired slope plots from BF16 to quantized for memory, throughput and latency;
- paired treatment forest plots for judge fields, already defined in Section 1.3; and
- a coverage/EOS companion panel.

This is a deployment-stack comparison, not a weight-only causal effect, because backend, kernels, prompt scheduling and serialized representation differ. The principal suite-level view places all TQ1_0 models and native-I2_S BitNet in the inclusive one-bit-frontier category, while every plotted point retains its exact representation label. Optional method-specific diagnostics remain available, but they do not replace the inclusive six-model comparison. Figure captions repeat this qualification rather than hiding it in a methods footnote.

Resource-rate comparisons are valid only within comparable hardware strata. If BF16 and quantized runs used different GPU types, slot configurations or materially different prompt populations, report their observed deployment values separately. Do not call their ratio a representation speed-up. A secondary adjusted row-level model may include GPU type and workload, but it does not recreate a randomized same-hardware comparison.

## 3.3 SFT and quantization efficiency

This section keeps the cost of learning an adapter separate from the cost of converting and deploying it, then places those costs beside—not inside—the judge-score changes.

Keep two cost ledgers:

- **SFT ledger:** training and checkpoint-transfer resource use;
- **quantization ledger:** merge/export, preparation, upload and restore resource use.

Then relate each ledger to its separate response outcome contrasts:

- paired judge-score change per field;
- change in judge-derived technique-identification score;
- existing deterministic ATT&CK coverage only as a weak strict-subset sensitivity result;
- change in failure risk;
- change in completion/limit behaviour; and
- incremental resource hours.

Descriptive ratios such as `paired score change / training GPU-runtime session hour` or `/ total deployment session hour` may be calculated, but remain field-specific and are never combined into one utility score. Negative or near-zero effects and incomplete coverage remain visible.

Because score-change/resource ratios become unstable near zero and imply an arbitrary utility trade-off, treat them as supplementary. The primary representation is a Pareto-style plot of resource time, completion coverage and one judge field at a time. Do not rank treatments with a single cost-effectiveness number.

Use a two-axis-free scatter design: x-axis resource hours, y-axis paired score change for one field and colour adapter. Show coverage as a printed percentage or an aligned coverage strip rather than imprecise bubble area. Produce one figure per field. Do not use a dual-axis chart.

## 3.4 Time-allocation and operational-improvement analysis

This analysis identifies where elapsed runtime was spent so future experiments can target downloads, preparation, model loading, checkpointing or generation separately.

This is primarily diagnostic for discussion and future work.

At the mutually exclusive phase level, calculate the share of recorded invocation wall time spent in:

- training;
- artifact preparation/upload;
- restoration/model load; and
- inference invocation wall time.

Within inference, report generation, receiver, local checkpoint and CSV timers as potentially overlapping diagnostic ratios, not additive shares. Show an overlap warning when their sum exceeds invocation wall time.

Use:

- top-level 100% stacked bars only for mutually exclusive phases;
- separate dot plots for component-time-to-wall-time ratios;
- retry/resume resource-time bars; and
- a table of streams with the largest model-load, transfer or checkpoint burden.

This supports recommendations such as persistent model servers, artifact reuse and asynchronous checkpointing without pretending that operational overhead is model generation speed.

## 3.5 Prompt length, slots and runtime configuration

This section asks whether observed row time is explained mainly by prompt length, generated length, hardware or runtime configuration. It does not retrospectively declare one configuration optimal.

Model `log(row elapsed seconds)` using:

- `log1p(input_tokens)`;
- output tokens;
- model;
- representation;
- condition;
- GPU type;
- parallel slots;
- runtime revision; and
- prompt-length bucket.

Use cluster-robust uncertainty by `group_id` where response and source tables can be joined. Report partial-effect plots and raw binned medians. Mixed runtime revisions are not pooled without a revision indicator.

The aim is to explain observed resource variation, not to retrospectively claim that a configuration was optimal.

## 3.6 Compression-performance frontier

The frontier shows whether smaller artifacts also used less memory or completed work faster, while keeping response quality and coverage visible.

For every artifact, combine:

- exact GGUF bytes;
- bytes per nominal parameter;
- target tensor distribution;
- peak memory;
- throughput;
- completion coverage; and
- each judge field on common-support rows.

Produce one scatter per outcome with artifact bytes on a log x-axis, faceted by representation or model where needed. Label only Pareto-frontier points and notable outliers; the companion table identifies every point. A smaller file is not assumed to be faster: ternary unpacking/kernel efficiency and output length can dominate.

## 3.7 Preparation amortization

Amortization asks how many future rows would be needed to recover one-time preparation cost if a quantized deployment were genuinely faster per comparable row.

Where quantized inference is faster than BF16 on a common row population, estimate:

`break_even_rows = preparation_and_upload_seconds / (BF16_seconds_per_row - quantized_seconds_per_row)`.

Use bootstrap intervals by resampling prompt groups. If the denominator is zero or negative, report `not reached` rather than a negative break-even value. Provide scenarios with and without artifact restore/upload overhead because deployment reuse changes amortization.

Only calculate break-even rows when both rates come from the same GPU type, comparable slot policy, matched prompt population and common output policy. Otherwise report `not estimable from observed hardware` and show the two deployment measurements separately.

## 3.8 Runtime reconciliation across every stream

The captured telemetry total and the reconstructed inference total answer different questions and are retained side by side. Captured telemetry is the sum of successfully saved invocation wall time; it can include model loading, waiting, I/O and idle allocation, but it omits a runtime that ended before its telemetry was durably uploaded. Reconstruction asks how much request-processing wall time is implied by the saved terminal row records. It never reconstructs GPU utilization, memory, energy, artifact transfer, model loading or idle allocation.

For every exact model, representation and treatment stream, successful request work is reconstructed from each row's `elapsed_seconds`. The saturated lower estimate is `sum(elapsed_seconds / effective_slots)`. BF16 Transformers inference is assigned one slot because its frozen worker executes the rows serially. An explicit GGUF `parallel_slots` value is used where present; older GGUF rows without this field are assigned one slot and marked `missing_assumed_sequential`, rather than silently borrowing a later configuration.

The occupancy sensitivity estimate reconstructs `[record_modified_time - elapsed_seconds, record_modified_time]` for each completed row and takes the union of those intervals. This can capture under-filled final waves and uneven request durations, but receiver queuing can delay file modification times. The audit therefore reports maximum inferred concurrency, configured slot capacity, union segments, timestamp coverage and a capacity-check flag. A timestamp union that implies more concurrent requests than configured slots remains a sensitivity value and is not described as observed timing.

Known terminal `TimeoutError: timed out` attempts use the frozen 1,800-second llama-server request timeout and are divided by the applicable slots. Earlier attempts on a later-completed row lack saved duration/type; their count is reported as `unrecoverable_prior_attempts` and no duration is invented. Other failures without known duration are likewise unrecoverable.

Streams are classified as `observed_complete`, `hybrid_gap_filled`, `reconstructed_only` or `insufficient_for_reconstruction`. Completeness is assessed from observed generation/request-work coverage and `new_rows` coverage. Complete streams retain observed invocation inference time; their row reconstruction is used to validate the method. Incomplete streams use a low/high request-processing range from slot adjustment and timestamp union. The project hybrid total replaces incomplete inference components only; observed training and preparation remain unchanged and observed overhead is not extrapolated.

Validation reports row request work versus observed `generation_seconds`, and slot-adjusted/timestamp-union time versus observed inference invocation wall time. These comparisons are separated conceptually because request work, inference wall time and full session time are not interchangeable. The method supplies a measurement-reconciliation range, not a statistical confidence interval and not billing-grade GPU compute.

Outputs:

- `analysis/tables/runtime_reconstruction_row_audit.csv`
- `analysis/tables/runtime_reconciliation_by_stream.csv`
- `analysis/tables/runtime_reconstruction_validation.csv`
- `analysis/tables/runtime_reconciliation_project_total.csv`
- `analysis/figures/37A_runtime_reconstruction_validation.png`
- `analysis/figures/37B_runtime_reconciliation_by_stream.png`

**How to interpret Figure 37A.** Agreement with the diagonal on telemetry-complete streams supports the use of the slot-adjusted method as a gap reconstruction. Systematic departure identifies backend, occupancy or timer differences and must temper the hybrid total.

**How to interpret Figure 37B.** Blue points are captured invocation hours; orange ranges are the selected inference accounting. A wide difference identifies missing telemetry or material timestamp/occupancy uncertainty. It does not imply that the reconstructed portion had measured utilization equivalent to the observed portion.

## 3.9 Compute-analysis outputs

All Figures 30--36 render the identifier in the title and use the four core streams as the primary 2 by 2 treatment structure. Supplementary variants retain technique, explanation and joint adapters. The accompanying figure-data CSVs state whether an estimate uses all consumed resource invocations, completed response rows or paired/common-support judged rows.

**How to interpret Figures 30--36.** Figure 30 shows descriptive relationships between model size and observed efficiency within the collected suite. Figure 31 shows paired changes when moving from BF16 to the one-bit-frontier deployment stack; it is not a pure quantization effect. Figure 32 places separate SFT-training and deployment-preparation cost beside field-specific score changes without combining them into a utility score. Figure 33 shows where allocated runtime was consumed and which component timers overlap. Figure 34 describes associations with prompt length, slots and runtime revision rather than declaring a causal optimum. Figure 35 plots sampled **whole-device** peak GPU memory against row throughput because PyTorch process-allocation counters are unavailable for llama.cpp and BitNet.cpp; it must never place missing process peaks at zero. Whole-device peaks come from exclusive runtimes but remain hardware- and session-specific. Figure 36 reports break-even rows only when hardware, prompt population and output policy are comparable; otherwise it explicitly reports `not estimable`.

- `analysis/tables/compute_size_efficiency.csv`
- `analysis/tables/deployment_stack_comparisons.csv`
- `analysis/tables/sft_quantization_efficiency.csv`
- `analysis/tables/time_allocation_and_overhead.csv`
- `analysis/tables/runtime_configuration_analysis.csv`
- `analysis/tables/compression_performance_frontier.csv`
- `analysis/tables/preparation_amortization.csv`
- `analysis/figures/30_size_efficiency_relationships.png`
- `analysis/figures/31_deployment_stack_resource_changes.png`
- `analysis/figures/32_sft_quantization_resource_outcomes.png`
- `analysis/figures/33A_phase_time_allocation.png`
- `analysis/figures/33B_component_overhead_ratios.png`
- `analysis/figures/34_runtime_configuration_effects.png`
- `analysis/figures/35_compression_performance_frontier.png`
- `analysis/figures/36_preparation_amortization.png`
- `analysis/figures/37A_runtime_reconstruction_validation.png`
- `analysis/figures/37B_runtime_reconciliation_by_stream.png`

# Publication tiers and consistent visual code

## Publication tiers

Every figure manifest row contains `publication_tier = main | supplementary | diagnostic`. Tiering is frozen before substantive result interpretation except where the predeclared rule below permits promotion.

- **Main-paper candidates:** low-dimensional figures that directly answer the principal exploratory comparisons: Figures 2A--2C, 3A--3C, 4C, 5, 6C--6F, 8A, 9A--9B, 13A, 14, 17A--17D, 23A, 24A--24C, 31, 32 and 35. Main figures favour the four core streams; they do not omit any score dimension, but dimension-specific outputs may appear as labelled subfigures or consecutive figures. Final selection occurs after assessing readability and redundancy, not effect magnitude.
- **Supplementary/appendix:** complete conditional-adapter distributions and dense class-level detail: Figures 1A--1B, S2A--S2C, 4A--4B, 6A--6B, 7A--7C, S6--S9, 10--11, S13A, 13B--13D, 15--16, S17, 18, S18, 20--22, S23B, 25--26, 30, 33--34 and 36.
- **Diagnostic notebook only unless specifically discussed:** Figures 00A--00B, 2G and 12A--12D. These audit data availability or judge behaviour and must not be presented as model-performance evidence.

A supplementary conditional-adapter figure may be promoted only when it directly answers the declared adapter-specific exploration and the reason is stated in the analysis manifest. It cannot be promoted because its effect is large, significant or visually striking.

## Colour, marker and line encoding

Use colour-blind-safe, fixed encodings throughout. Never reuse one colour to mean different things in different figures.

### Model palette

Use model hue only when multiple models share a panel:

| Model | Colour | Hex |
|---|---|---|
| Qwen2.5 1.5B | blue | `#0072B2` |
| OpenHermes 1.8B | orange | `#E69F00` |
| BitNet b1.58 2B | bluish green | `#009E73` |
| Llama 3.2 3B | vermillion | `#D55E00` |
| Mistral 7B v0.3 | reddish purple | `#CC79A7` |
| Qwen2.5 7B | sky blue | `#56B4E9` |

### Adapter palette and markers

When a figure is already restricted to one model, hue identifies adapter instead:

| Condition | Colour | Hex | Marker |
|---|---|---|---|
| Untreated | dark grey | `#4D4D4D` | circle |
| Technique | blue | `#0072B2` | upward triangle |
| Explanation | orange | `#E69F00` | square |
| Joint | bluish green | `#009E73` | diamond |
| Full | reddish purple | `#CC79A7` | pentagon |

Never encode model and adapter simultaneously with competing hue palettes. If both vary, facet by model and use adapter colour/marker, or facet by adapter and use model colour.

### Representation encoding

- BF16 HF: hollow markers or solid line, always in a panel titled `BF16 HF`.
- One-bit frontier: filled markers or dashed line, always in an inclusive panel titled `one-bit frontier` containing all six models.
- Within a one-bit panel, TQ1_0 GGUF uses a dashed outline/line and native I2_S GGUF uses a dotted outline/line; direct labels state the exact representation.

Representation group is primarily separated by panels/figures. Exact-representation line style is a redundant provenance cue, not a reason to split or exclude models. All paired BF16/one-bit figures use identical axes, ordering, colour mapping and physical dimensions.

### Ordinal and outcome palettes

- Judge scores use a fixed diverging Likert palette: `0 #B2182B`, `1 #EF8A62`, `2 #D9D9D9`, `3 #67A9CF`, `4 #2166AC`.
- Treatment transitions use `improved #009E73`, `unchanged #BDBDBD`, `worsened #D55E00`.
- Missing/unavailable uses light grey `#E6E6E6` with hatching or an `NA` symbol; it is never assigned the score-0 colour.
- Uncertainty intervals use the same hue as the point with 45--60% opacity; the zero/reference line uses charcoal `#333333`.

### Labelling limits

- Label every panel and axis; prefer direct labels to legends when there are at most six series.
- Do not place numeric text in heatmap cells once a figure exceeds 100 cells.
- Label scatter points only for the Pareto frontier, declared outliers or cases discussed in text.
- Use one categorical colour dimension per figure; use facets, marker fill and line style for additional structure.
- The same construct keeps the same colour, marker, order and abbreviation in main, supplementary and diagnostic outputs.

# Output interpretation catalogue

This catalogue is mandatory notebook documentation. Each generated output must reproduce its corresponding explanation immediately above or below the displayed object, adapted with the actual population and denominator. `How to read` describes the marks or columns; `What it means` connects the output to the exploratory question and states the main limitation.

## Model-response tables

- **`analysis/tables/figure_manifest.csv`** — **How to read:** each row identifies a figure, its companion data CSV, source files, filters, grouping, interval method and caption. **What it means:** makes figure generation auditable rather than requiring values to be read from pixels.
- **`analysis/tables/figure_data/<figure_id>__<short_name>.csv`** — **How to read:** these are the exact post-filter, post-aggregation values plotted in the named figure, including denominators and uncertainty settings. **What it means:** permits independent checking of every plotted point or bar.
- **`analysis/tables/judge_score_availability.csv`** — **How to read:** compare intended rows, completed candidates, completed judgments and scalar in-range scores for each stream/dimension. **What it means:** identifies where analytical coverage was lost and whether some models or representations are disproportionately excluded before score comparison.
- **`analysis/tables/judge_score_availability_bounds.csv`** — **How to read:** the available-case mean sits between a worst-case mean assigning unavailable scores 0 and a best-case mean assigning them 4. **What it means:** narrow bounds indicate conclusions are relatively insensitive to missing scores; wide bounds mean the observed mean cannot represent the intended pool confidently.
- **`analysis/tables/model_scores_by_dimension.csv`** — **How to read:** each row is one exact model/condition/representation/dimension with its denominator and descriptive score summaries. **What it means:** supports dimension-specific comparisons without constructing a combined quality score.
- **`analysis/tables/model_score_distributions.csv`** — **How to read:** inspect counts and proportions at scores 0--4, median, IQR and percentage at 3--4. **What it means:** reveals whether equal means hide different failure/success shapes or polarized behaviour.
- **`analysis/tables/judge_scale_use.csv`** — **How to read:** examine category utilization, mode, floor/ceiling rates and effective number of categories by dimension and judge phase. **What it means:** shows score compression or leniency that limits discrimination; it does not prove rubric-option position bias.
- **`analysis/tables/treatment_contrasts.csv`** — **How to read:** positive paired effects favour the first-named treatment, negative effects favour the comparator; use the common-support `n` and interval. **What it means:** estimates how scores changed on the same prompts after SFT or deployment-stack change, conditional on the observed artifacts and judge pass.
- **`analysis/tables/treatment_transition_counts.csv`** — **How to read:** compare improved, tied, worsened, rescued and newly degraded prompt counts/rates. **What it means:** distinguishes broad modest shifts from improvements concentrated in a small subset.
- **`analysis/tables/common_support_diagnostics.csv`** — **How to read:** compare pair-specific and universal-panel membership against the intended 285 rows by length, scenario, class and multiplicity. **What it means:** shows whether matched comparisons have become a selected, easier or otherwise different prompt population.
- **`analysis/tables/adapter_specificity_profiles.csv`** — **How to read:** follow each adapter's paired effect across all six separate judge dimensions. **What it means:** indicates whether a conditional adapter changed its intended capability selectively or caused diffuse gains/losses; cross-dimension effect sizes are not interchangeable.
- **`analysis/tables/source_attack_distribution.csv`** — **How to read:** compare row counts and percentages for technique, parent technique, tactic, scenario and scenario-step in 990 versus 285 rows. **What it means:** establishes what the benchmark represents and whether the evaluation pool changes the source distribution.
- **`analysis/tables/attack_class_scores.csv`** — **How to read:** use class frequency, score, interval and stream identity together. **What it means:** locates ATT&CK classes associated with stronger or weaker judged responses without claiming that the class alone caused difficulty.
- **`analysis/tables/attack_class_macro_micro.csv`** — **How to read:** compare row-weighted, class-macro and secondary membership-micro estimates. **What it means:** differences reveal whether common classes dominate the observed average or rare classes have systematically different performance.
- **`analysis/tables/ground_truth_multiplicity.csv`** — **How to read:** compare scores, failures and completion by number of ground-truth technique IDs. **What it means:** tests whether multi-label attack steps impose greater coverage difficulty while acknowledging that sampled logs may omit some evidence.
- **`analysis/tables/judge_dimension_relationships.csv`** — **How to read:** use joint frequencies, rank correlation, difference distributions and high/low quadrants for one predeclared dimension pair. **What it means:** exposes patterns such as correct identification without evidence or grounded explanation without alignment; correlation is not redundancy or causation.
- **`analysis/tables/confidence_field_availability.csv`** — **How to read:** follow rows through candidate-label, warranted-label and scalar-agreement availability. **What it means:** determines whether confidence-agreement findings have enough and sufficiently balanced support to interpret.
- **`analysis/tables/failure_label_dictionary.csv`** — **How to read:** maps every raw judge label to the frozen normalized category while retaining unmapped values. **What it means:** makes failure aggregation reproducible and shows where analyst coding choices enter the analysis.
- **`analysis/tables/failure_rates.csv`** — **How to read:** compare failures per 100 judged responses with raw counts and denominators. **What it means:** identifies dominant failure modes without mistaking larger stream coverage for worse behaviour.
- **`analysis/tables/failure_risk_differences.csv`** — **How to read:** positive paired differences mean the first treatment has a higher failure rate; intervals crossing zero indicate uncertain direction. **What it means:** shows whether SFT or representation change altered specific failure risks on matched prompts.
- **`analysis/tables/score_failure_label_coherence.csv`** — **How to read:** examine the score distribution associated with each judge failure category and contradictory high-score/failure combinations. **What it means:** audits whether Haiku's structured labels and scores agree internally, not whether they are externally correct.
- **`analysis/tables/repeated_prompt_consistency.csv`** — **How to read:** compare exact response agreement, lexical similarity, score stability and output-length variation for repeated prompt hashes. **What it means:** measures reproducibility on identical text while preserving configuration and ground-truth context.
- **`analysis/tables/scenario_step_variant_sensitivity.csv`** — **How to read:** use within-group score ranges, dispersion and completion variation across variants. **What it means:** shows sensitivity to different excerpts/variants of the same attack step rather than ordinary between-scenario difficulty.
- **`analysis/tables/judge_output_consistency_audit.csv`** — **How to read:** compare already-exported rubric-implied values with returned scalar scores where both exist. **What it means:** measures internal rule application; non-applicable rows and disagreements prevent claims of judge reliability.
- **`analysis/tables/judge_confidence_rule_audit.csv`** — **How to read:** inspect exact agreement, mean absolute disagreement and confusion among candidate label, warranted label and returned agreement score. **What it means:** tests the same-pass confidence arithmetic on its limited eligible subset, not candidate calibration against truth.
- **`analysis/tables/judge_duplicate_case_consistency.csv`** — **How to read:** inspect exact score-vector agreement and maximum disagreement for identical case fingerprints judged more than once. **What it means:** provides a natural-repeat check only; a small duplicate sample cannot estimate test--retest reliability.
- **`analysis/tables/judge_reference_anchor_diagnostics.csv`** — **How to read:** compare within-prompt associations between scores and candidate similarity to Haiku versus GPT-5.5 teacher text. **What it means:** associations stronger for Haiku similarity are consistent with reference/style anchoring but remain confounded with genuine response quality.
- **`analysis/tables/lexical_similarity_treatment_comparisons.csv`** — **How to read:** compare matched candidate-to-Haiku and candidate-to-teacher similarity across the four core streams and conditional adapters. **What it means:** measures treatment-related lexical convergence, not correctness.
- **`analysis/tables/qualitative_audit_cases.csv`** — **How to read:** inspect the frozen, rule-selected cases with candidate, reference, teacher, ground truth, scores, reasons and hashes together. **What it means:** helps readers judge whether quantitative patterns reflect plausible evaluator behaviour; examples illustrate rather than estimate prevalence.
- **`analysis/tables/prompt_length_and_long_tail.csv`** — **How to read:** compare completion, stop reason, latency, output length and judge scores across fixed length buckets and deciles. **What it means:** reveals where long inputs are associated with operational or judged-performance degradation without establishing a causal length effect.
- **`analysis/tables/rare_attack_class_uncertainty.csv`** — **How to read:** read class frequency beside score and interval width. **What it means:** makes clear which class findings are supported and which are too sparse for stable interpretation.
- **`analysis/tables/model_item_difficulty_concordance.csv`** — **How to read:** examine prompt-level score correlations and low-score-set overlaps for each model pair/dimension. **What it means:** answers whether models fail on the same cases even when their overall averages are similar.
- **`analysis/tables/shared_and_model_specific_hard_cases.csv`** — **How to read:** rows list prompts difficult for most models or uniquely difficult for one model/representation. **What it means:** identifies shared benchmark difficulty versus architecture/treatment-specific weaknesses for later qualitative discussion.
- **`analysis/tables/token_ceiling_and_stop_reasons.csv`** — **How to read:** compare EOS and ceiling rates among completed responses, while reading `context_unsupported` and `retryable_error` as separate operational counts. A zero-token unsupported row is never a ceiling hit. **What it means:** quantifies how non-termination contributed to cost and incomplete streams, separately from semantic quality.
- **`analysis/tables/runtime_revision_sensitivity.csv`** — **How to read:** compare coverage, workload, stop reason and scores by recorded runtime revision. **What it means:** detects whether mixed operational configurations coincide with changed result distributions; differences are confounded and not causal configuration effects.

## Model-response figures

- **`analysis/figures/00A_judge_score_eligibility_counts.png`** — **How to read:** stacked stages show intended candidates, completed responses, judgments and scalar scores, with exact counts. **What it means:** immediately shows where analytical coverage was lost.
- **`analysis/figures/00B_initial_score_boxplots_core_streams.png`** — **How to read:** compare median/IQR only after checking the eligible `n` printed for every box. **What it means:** is a compact orientation to score location and spread across the four core streams; ordinal bars remain the primary evidence.
- **`analysis/figures/S00B_initial_score_boxplots_conditional_adapters.png`** — **How to read:** uses the same axes and model grid for technique, explanation and joint adapters. **What it means:** retains adapter-specific score spread without crowding the main figure.
- **`analysis/figures/01A_score_overview_bf16.png`** — **How to read:** compare colour within the same dimension using the fixed 0--4 scale and the same model/condition ordering used in Figure 01B; use annotated `n`. **What it means:** a BF16 navigation view of broad patterns, not sufficient evidence without ordinal distributions.
- **`analysis/figures/01B_score_overview_one_bit_frontier.png`** — **How to read:** use exactly the same score scale, dimension order and model order as Figure 01A. **What it means:** a one-bit-frontier navigation view that remains visually comparable with BF16 while keeping backend/representation differences explicit.
- **`analysis/figures/02A_<dimension>_bf16_untreated_full.png`** — **How to read:** each Likert bar is the percentage at scores 0--4 for untreated and full-SFT BF16 streams; the ordinal palette and 0--4 category widths match Figure 02B. **What it means:** the principal distributional comparison for the BF16 phase.
- **`analysis/figures/02B_<dimension>_one_bit_untreated_full.png`** — **How to read:** use the same axes, category order and ordinal colours as Figure 02A. **What it means:** the principal distributional comparison for the one-bit-frontier phase, with TQ1_0 and native I2_S identified rather than pooled as the same quantizer.
- **`analysis/figures/02C_<dimension>_one_bit_zero_and_nonzero_composition.png`** — **How to read:** read the overall zero rate first, then the score 1--4 composition conditional on a nonzero score. **What it means:** makes minority nonzero outcomes visible without hiding the dominant floor effect.
- **`analysis/figures/S02A_<dimension>_bf16_conditional_adapters.png`** — **How to read:** compare technique, explanation and joint adapters on the same BF16 ordinal scale used in Figure 02A. **What it means:** supplementary evidence about conditional-adapter behaviour without crowding the principal untreated/full comparison.
- **`analysis/figures/S02B_<dimension>_one_bit_conditional_adapters.png`** — **How to read:** compare conditional adapters using the same one-bit ordinal scale used in Figure 02B. **What it means:** supplementary evidence about conditional adapters after one-bit-frontier deployment; promote to the main text only for a predeclared adapter-specific finding, never because an effect appears large.
- **`analysis/figures/S02C_<dimension>_one_bit_conditional_zero_and_nonzero.png`** — **How to read:** combines each conditional adapter's total zero rate with its conditional nonzero composition. **What it means:** exposes small differences among nonzero outcomes while retaining the overall failure floor.
- **`analysis/figures/02G_judge_scale_use.png`** — **How to read:** compare category use, entropy and floor/ceiling concentration across dimensions/phases. **What it means:** reveals judge scale compression that can mask candidate differences; it cannot identify why compression occurred.
- **`analysis/figures/03A_<dimension>_bf16_sft_effects.png`** — **How to read:** points right of zero favour SFT over untreated BF16; whiskers show prompt-cluster uncertainty and labels show exact matched `n`. **What it means:** summarizes paired BF16 SFT effects for one rubric dimension.
- **`analysis/figures/03B_<dimension>_one_bit_sft_effects.png`** — **How to read:** points right of zero favour the SFT-derived one-bit artifact over its untreated one-bit counterpart; use the same x scale as Figure 03A. **What it means:** summarizes paired SFT effects after one-bit-frontier deployment while retaining representation labels.
- **`analysis/figures/03C_<dimension>_deployment_stack_effects.png`** — **How to read:** each point compares matched rows across the complete BF16-versus-one-bit deployment stacks; positive values favour the named first stack. **What it means:** describes combined weight-representation and backend differences, not an isolated quantization effect.
- **`analysis/figures/04A_<dimension>_bf16_treatment_transitions.png`** — **How to read:** improved, unchanged and worsened segments show row-level BF16 transitions on a shared percentage scale. **What it means:** distinguishes broad BF16 treatment gains from offsetting prompt-level gains and harms.
- **`analysis/figures/04B_<dimension>_one_bit_treatment_transitions.png`** — **How to read:** use the same transition colours, segment order and percentage scale as Figure 04A. **What it means:** provides the corresponding one-bit-frontier transition pattern without visually conflating backends.
- **`analysis/figures/04C_<model>_adapter_specificity_profiles.png`** — **How to read:** each compact model-specific figure contains separate, identically scaled BF16 and one-bit panels; trace each adapter across the rubric dimensions and use outlined targets to identify intended training dimensions. **What it means:** shows whether one model's adapter changed the targeted behaviour specifically or shifted several qualities together without forcing all models into one unreadable panel.
- **`analysis/figures/05_tactic_distribution_990_vs_285.png`** — **How to read:** compare the paired percentage bars and raw counts for every tactic. **What it means:** shows whether the 285-row evaluation pool preserves or changes the original tactic composition.
- **`analysis/figures/06A_<dimension>_bf16_technique_heatmap.png` / `06B_<dimension>_one_bit_frontier_technique_heatmap.png`** — **How to read:** compare model/condition cells within a technique using identical score scales; the one-bit figure includes all six models. **What it means:** locates class-associated strengths and weaknesses separately for BF16 and the inclusive frontier.
- **`analysis/figures/06C_<dimension>_bf16_technique_sft_effects.png` / `06D_<dimension>_one_bit_frontier_technique_sft_effects.png`** — **How to read:** points right/left of zero show paired SFT improvement/degradation for one technique. **What it means:** shows whether treatment effects are concentrated in particular ATT&CK classes.
- **`analysis/figures/06E_<dimension>_bf16_row_macro_comparison.png` / `06F_<dimension>_one_bit_frontier_row_macro_comparison.png`** — **How to read:** compare row-weighted and class-macro points for the same stream. **What it means:** a large gap indicates that common classes and the typical class tell different performance stories.
- **`analysis/figures/07A_<dimension>_bf16_lowest_techniques.png` / `07B_<dimension>_one_bit_frontier_lowest_techniques.png`** — **How to read:** all panels use the same preselected sufficiently represented classes and 0--4 axis. **What it means:** permits direct comparison of persistent weak classes without selecting a different worst list per model.
- **`analysis/figures/S06A_<dimension>_bf16_all_techniques_and_adapters.png` / `S06B_<dimension>_one_bit_frontier_all_techniques_and_adapters.png`** — **How to read:** locate a class by row and adapter by column using the fixed 0--4 scale. **What it means:** provides full adapter-specific class detail for both representation groups.
- **`analysis/figures/S07A_<dimension>_bf16_conditional_adapter_lowest_techniques.png` / `S07B_<dimension>_one_bit_frontier_conditional_adapter_lowest_techniques.png`** — **How to read:** dots and connectors compare conditional adapters on the same preselected difficult classes and common 0--4 axis. **What it means:** shows whether conditional adapters alter the persistent weak-class pattern in each representation group without using dense grouped bars.
- **`analysis/figures/07C_ground_truth_multiplicity.png`** — **How to read:** compare score distributions and coverage as the number of ground-truth labels increases. **What it means:** indicates whether broader multi-technique targets are associated with lower scores or more failures.
- **`analysis/figures/08A_<pair>_core_stream_joint_score_grid.png`** — **How to read:** each model row and core-stream column has its own 5-by-5 distribution; off-diagonal cells expose score mismatches. **What it means:** compares related dimensions without aggregating across models.
- **`analysis/figures/S08A_<pair>_bf16_all_adapters.png` / `S08B_<pair>_one_bit_frontier_all_adapters.png`** — **How to read:** compare adapter columns within the same named model and representation group. **What it means:** retains condition-specific joint-score structure as supplementary evidence.
- **`analysis/figures/09A_bf16_failure_rate_bars.png` / `09B_one_bit_frontier_failure_rate_bars.png`** — **How to read:** bar length is failures per 100 judged rows; endpoint labels give raw count and denominator. **What it means:** compares prevalence across all models while keeping representation groups explicit.
- **`analysis/figures/S09A_bf16_conditional_adapter_failure_rates.png` / `S09B_one_bit_frontier_conditional_adapter_failure_rates.png`** — **How to read:** use the same axes for technique, explanation and joint adapters. **What it means:** shows adapter-specific failure patterns without crowding the core-stream figure.
- **`analysis/figures/10_failure_cooccurrence_upset.png`** — **How to read:** top bars count common failure combinations and the dot matrix identifies included categories. **What it means:** shows which weaknesses commonly appear together without implying independence or causation.
- **`analysis/figures/11A_repeated_response_stability.png`** — **How to read:** higher exact/lexical/technique-set agreement and smaller output-length ranges mean more stable repeated candidate responses. **What it means:** assesses candidate behavioural consistency without mixing it with judge-score stability.
- **`analysis/figures/11B_repeated_judge_score_stability.png`** — **How to read:** smaller within-prompt score ranges and higher duplicate agreement mean more stable judgments on repeated cases. **What it means:** is an internal-consistency diagnostic, not external validation of judge reliability.
- **`analysis/figures/12A_judge_score_availability_diagnostic.png`** — **How to read:** compare canonical scalar-score availability by phase, stream and dimension. **What it means:** shows where score-based findings may be conditioned on selective judgment availability.
- **`analysis/figures/12B_judge_duplicate_consistency.png`** — **How to read:** higher agreement and smaller paired differences indicate more consistent outcomes for naturally duplicated cases. **What it means:** provides a limited internal check; it does not replace a designed test-retest or multi-judge study.
- **`analysis/figures/12C_confidence_rule_diagnostic.png`** — **How to read:** compare exact agreement and absolute disagreement between candidate confidence and the judge-warranted label only where canonical fields exist. **What it means:** describes the judge's confidence-rule behaviour and possible self-consistency, not independent confidence calibration.
- **`analysis/figures/12D_haiku_self_evaluation_diagnostic.png`** — **How to read:** compare Haiku self-score distributions with candidate score distributions on aligned rows. **What it means:** helps readers calibrate possible self-preference or circularity; it is not a fair model-performance baseline by itself.
- **`analysis/figures/13D_<dimension>_reference_similarity_vs_score.png`** — **How to read:** compare within-prompt score trends against similarity to Haiku and to the teacher for one dimension. **What it means:** stronger Haiku-linked trends are consistent with reference anchoring, while neither trend proves bias or correctness.
- **`analysis/figures/13A_core_stream_reference_lexical_similarity.png`** — **How to read:** within each model, compare candidate-to-Haiku and candidate-to-teacher similarity across the four core streams on matched prompts. **What it means:** shows treatment-related surface convergence, not semantic correctness.
- **`analysis/figures/S13A_conditional_adapter_reference_lexical_similarity.png`** — **How to read:** repeats the matched lexical comparison for technique, explanation and joint adapters. **What it means:** indicates whether conditional SFT shifts expression toward either stored reference.
- **`analysis/figures/13B_scenario_step_score_sensitivity.png`** — **How to read:** larger within-group judge-score dispersion means greater sensitivity across variants of the same scenario-step. **What it means:** identifies score instability attributable to variant choice without mixing in operational outcomes.
- **`analysis/figures/13C_scenario_step_operational_sensitivity.png`** — **How to read:** compare completion, timeout, ceiling and length variation within scenario-step groups. **What it means:** separates operational sensitivity from judged-quality sensitivity.
- **`analysis/figures/14_token_ceiling_rates.png`** — **How to read:** compare EOS, ceiling, timeout and truncation proportions across models and representations. **What it means:** shows which streams incur long generations or incomplete rows because models fail to stop naturally.
- **`analysis/figures/15_runtime_revision_sensitivity.png`** — **How to read:** compare distributions across revisions only after checking prompt-length composition. **What it means:** flags mixed-configuration sensitivity; it does not estimate a causal slot or scheduling effect.
- **`analysis/figures/16A_<dimension>_bf16_model_size_relationships.png`** — **How to read:** inspect direction and scatter against log parameter count within BF16, with one condition per panel. **What it means:** describes the BF16 suite's size-associated pattern, not a scaling law because architecture and pretraining remain confounded.
- **`analysis/figures/16B_<dimension>_one_bit_model_size_relationships.png`** — **How to read:** use the same y scale and condition order as Figure 16A, but interpret TQ1_0 and native I2_S labels explicitly. **What it means:** describes one-bit-frontier size-associated patterns without pretending the representations are algorithmically identical.
- **`analysis/figures/17A--17D_<dimension>_*_item_difficulty_concordance.png`** — **How to read:** compare the four identically scaled core-stream matrices; every matrix includes all six models, with native-I2_S BitNet included in frontier panels. **What it means:** answers whether models fail on the same cases across BF16/frontier and untreated/SFT-full rather than only whether averages differ.
- **`analysis/figures/S17_<dimension>_<representation_group>_<adapter>_item_difficulty_concordance.png`** — **How to read:** compare technique, explanation and joint adapter matrices with the same model order and scale. **What it means:** retains adapter-specific item-difficulty concordance as supplementary evidence.
- **`analysis/figures/18A_prompt_length_ecdf.png`** — **How to read:** at each token length, the curve gives the proportion of prompts no longer than that value. **What it means:** visualizes the long tail and differences between the 990-row and 285-row populations.
- **`analysis/figures/18B_completion_by_prompt_length.png`** — **How to read:** stacked proportions show completed, unsupported, timeout and missing outcomes within each length bucket. **What it means:** identifies length ranges driving coverage loss.
- **`analysis/figures/18C_scores_by_prompt_length.png`** — **How to read:** points/lines show binned scores with cluster intervals; inspect sample counts. **What it means:** reveals score deterioration or stability along the prompt-length tail without claiming length causality.
- **`analysis/figures/18D_latency_by_prompt_length.png`** — **How to read:** compare median and p95 latency across length buckets and representations. **What it means:** separates typical scaling from costly tail behaviour.
- **`analysis/figures/18E_rare_class_frequency_uncertainty.png`** — **How to read:** sparse classes should show wider intervals; frequency and uncertainty are read together. **What it means:** warns where apparent class rankings are too unstable for strong interpretation.
- **`analysis/figures/18F_training_subset_representativeness.png`** — **How to read:** Panel A contrasts retained/excluded complete-example lengths against BitNet's 4,096-token boundary; Panel B gives excluded/total rows by scenario; Panel C counts model-specific context failures. **What it means:** shows whether the universal 604-row SFT subset is length- and scenario-selective and identifies the limiting model.
- **`analysis/figures/18G_split_label_and_prompt_balance.png`** — **How to read:** compare population proportions for scenarios and label multiplicity, then compare prompt-character ECDFs on the common log scale. **What it means:** shows whether original/feasible train, validation and test differ in composition and textual length; exact distinct-prompt and unseen-technique counts remain in the companion tables.
- **`analysis/figures/S18B--S18D_conditional_adapter_*.png`** — **How to read:** repeat completion, score and latency views for conditional adapters using the same buckets and axes. **What it means:** reveals whether prompt-length associations differ by adapter without changing the core-stream comparison.

## Compute-reporting tables and figures

- **`analysis/tables/compute_stream_resource_summary.csv`** — **How to read:** each exact stream has total/mean session time, separated operational components, coverage, throughput, utilization, peak memory and observed flags. **What it means:** is the concise per-stream resource ledger for the four core streams and every conditional adapter.
- **`analysis/tables/compute_invocation_audit.csv`** — **How to read:** each row is one telemetry invocation with phase, status, wall time, duplicate flag and observed/applicable metric flags. **What it means:** is the auditable basis for all resource totals and shows exactly which retries/resumes were counted.
- **`analysis/tables/compute_sft_training_report.csv`** — **How to read:** compare training invocations, steps, loss, session hours, utilization, memory and adapter bytes by model/adapter. **What it means:** reports the resource cost of creating adapters separately from inference.
- **`analysis/tables/compute_quantization_report.csv`** — **How to read:** compare preparation/upload time, artifact size/hash, tensor types, retries and hardware. **What it means:** reports the one-time cost and compression outcome of the deployment representation.
- **`analysis/tables/compute_inference_report.csv`** — **How to read:** use coverage, session time, tokens, latency, throughput, utilization and runtime revision together. **What it means:** compares observed inference workloads without mistaking operational overhead or short outputs for intrinsic model speed.
- **`analysis/tables/compute_api_report.csv`** — **How to read:** separate reference generation, Phase 1 judging, self-evaluation and Phase 2 judging requests/tokens/cache/cost. **What it means:** reports API use by experimental purpose; cached tokens remain part of the same cases.
- **`analysis/figures/20_training_resource_hours.png`** — **How to read:** longer bars mean more GPU-runtime session time for one adapter training stream. **What it means:** identifies expensive training conditions, not active GPU compute or training effectiveness.
- **`analysis/figures/21_training_loss_curves.png`** — **How to read:** follow loss over optimizer steps within each model/adapter; compare shapes, not absolute loss across incompatible tokenizations blindly. **What it means:** documents optimization progress and anomalies, not validation performance.
- **`analysis/figures/22_quantization_time_and_artifact_size.png`** — **How to read:** compare preparation time and final bytes in separate aligned panels. **What it means:** shows conversion cost versus storage compression; smaller files are not automatically faster.
- **`analysis/figures/23A_model_phase_resource_hours.png`** — **How to read:** bars aggregate recorded GPU-runtime session time by model and phase while keeping training, preparation and inference distinct. **What it means:** is the concise main-text resource burden view; it does not represent active GPU computation alone.
- **`analysis/figures/S23B_exact_stream_resource_hours.png`** — **How to read:** each bar is one exact stream and includes its recorded retries/resumes. **What it means:** provides the auditable supplementary detail behind Figure 23A and exposes costly failed starts or incomplete streams.
- **`analysis/figures/24A_rows_per_hour.png`** — **How to read:** compare completed rows per recorded inference hour on a common log scale, alongside coverage. **What it means:** reports workload completion rate, which depends on prompt and output lengths.
- **`analysis/figures/24B_prompt_tokens_per_hour.png`** — **How to read:** compare processed input-token throughput on the same stream order and log-scale convention as Figures 24A and 24C. **What it means:** isolates prompt-processing volume from row count.
- **`analysis/figures/24C_generated_tokens_per_hour.png`** — **How to read:** compare generated-token throughput while checking output-ceiling rates. **What it means:** better reflects decoding throughput but should not be read as semantic performance.
- **`analysis/figures/25_gpu_utilization_and_memory.png`** — **How to read:** points combine time-weighted sampled utilization and sampled peak memory, labelled by hardware. **What it means:** shows whether memory or low utilization may constrain a stream; samples are whole-device and may miss short peaks.
- **`analysis/figures/26A_stream_coverage.png`** — **How to read:** stacked proportions show completed, unsupported, retryable-error, other saved-status and `not_reached` outcomes. Every untreated bar uses the intended 990 rows and every SFT bar the intended 285 rows; the denominator is never reduced to the number of saved records. **What it means:** shows the analytical support available for every stream without making partial streams appear complete.
- **`analysis/figures/26B_retry_resume_burden.png`** — **How to read:** compare retry/resume invocations and their session time separately from final coverage. **What it means:** identifies operational burden without embedding two different quantities in one dense figure.

## Compute-analysis tables and figures

- **`analysis/tables/compute_size_efficiency.csv`** — **How to read:** compare resource measures against parameter count within hardware/backend strata. **What it means:** describes suite-specific size/efficiency patterns, not a universal scaling relationship.
- **`analysis/tables/deployment_stack_comparisons.csv`** — **How to read:** paired rows contrast BF16 HF and exact TQ1_0/I2_S deployment values with hardware comparability flags. **What it means:** shows observed deployment-stack changes; flagged incomparable rows cannot support a speed-up claim.
- **`analysis/tables/sft_quantization_efficiency.csv`** — **How to read:** place training/preparation resource use beside one field-specific paired score change and coverage. **What it means:** supports cost/outcome discussion without constructing a composite utility score.
- **`analysis/tables/time_allocation_and_overhead.csv`** — **How to read:** top-level mutually exclusive phase times are additive; component ratios marked overlapping are not. **What it means:** identifies whether generation, preparation, loading or checkpoint operations dominate recorded runtime.
- **`analysis/tables/runtime_configuration_analysis.csv`** — **How to read:** inspect workload-adjusted associations for prompt length, output length, GPU, slots and revision. **What it means:** suggests operational bottlenecks but does not prove an optimal configuration.
- **`analysis/tables/compression_performance_frontier.csv`** — **How to read:** compare artifact bytes, memory, throughput, coverage and one judge dimension without combining them. **What it means:** identifies dominated and trade-off deployments on the observed data.
- **`analysis/tables/preparation_amortization.csv`** — **How to read:** finite break-even rows mean a comparable quantized stream was faster enough to repay preparation; `not reached` or `not estimable` states why no number is valid. **What it means:** frames whether one-time conversion cost could be justified under repeated use.
- **`analysis/figures/30_size_efficiency_relationships.png`** — **How to read:** compare one efficiency measure against log parameter count within clearly separated backend/hardware panels. **What it means:** visualizes suite-specific resource scaling with architecture/backend confounding retained.
- **`analysis/figures/31_deployment_stack_resource_changes.png`** — **How to read:** slopes connect the same model from BF16 HF to exact one-bit-frontier deployment; labels flag hardware mismatch. **What it means:** shows direction and magnitude of observed memory/throughput/latency change without calling it a weight-only effect.
- **`analysis/figures/32_sft_quantization_resource_outcomes.png`** — **How to read:** x is resource time and y is one judge-field effect; coverage is printed beside each point or shown in an aligned strip rather than encoded as bubble area. **What it means:** shows trade-offs and Pareto-like candidates without asking readers to estimate a third quantity from circle size; it does not define one best treatment.
- **`analysis/figures/33A_phase_time_allocation.png`** — **How to read:** stacked bars use only mutually exclusive training, preparation, restoration, inference and transfer categories. **What it means:** shows how recorded session time was allocated without double-counting overlapping timers.
- **`analysis/figures/33B_component_overhead_ratios.png`** — **How to read:** dots show separately defined component-to-wall ratios and their applicability/observation state. **What it means:** identifies operational overhead candidates while keeping non-additive measures out of the stacked time figure.
- **`analysis/figures/34_runtime_configuration_effects.png`** — **How to read:** partial-effect curves and raw binned medians show workload-adjusted associations with slots, GPU and revision. **What it means:** suggests configurations associated with better throughput; observational confounding prevents causal optimization claims.
- **`analysis/figures/35_compression_performance_frontier.png`** — **How to read:** points farther left are smaller; higher/lower y is better according to the single plotted outcome; coverage is shown explicitly. **What it means:** reveals whether compression coincides with a resource or quality trade-off rather than assuming smaller means faster.
- **`analysis/figures/36_preparation_amortization.png`** — **How to read:** the plotted break-even and interval apply only to hardware-comparable matched workloads; invalid comparisons are labelled not estimable. **What it means:** estimates the reuse volume needed to offset one-time preparation under observed deployment conditions.

# 4. Discussion boundaries and quantities not estimable

## 4.1 Plain-language boundary

Notebook 10 can compare the recorded models, prompts, artifacts and Haiku judgments. It cannot prove that Haiku is correct, that another training run would behave identically, or that one backend change alone caused an observed difference. These are design limits, not missing calculations.

## 4.2 Judge validity and reliability

The experiment contains no blinded expert-human subset, second independent judge model, deliberately repeated judge calls or rubric-order permutation. It therefore cannot estimate:

- human–Haiku agreement or judge accuracy;
- inter-judge agreement;
- test–retest judge reliability;
- rubric-option position bias;
- causal self-preference; or
- robustness to adversarial or quality-preserving perturbations.

Natural duplicate cases, self-evaluation, failure-label coherence, score availability and reference-similarity analyses are reported as **internal consistency and bias diagnostics only**. Dietz et al. (2025) warn that system-level agreement can conceal label-level disagreement and that circularity, evaluator self-preference and loss of opinion diversity threaten validity. Panickssery et al. (2024) and Chen et al. (2025) further show that self-recognition and genuine quality differences complicate self-preference measurement. Doddapaneni et al. (2024) demonstrate that evaluators may miss controlled quality degradations. None of those properties can be fully recovered retrospectively from this one frozen judge pass.

## 4.3 Training and artifact uncertainty

There is one observed adapter per model/condition and one final quantized artifact per deployment stream. Prompt-cluster bootstrap intervals do not estimate variability across SFT seeds, adapter checkpoints, merge implementations or repeated quantization. Claims apply to the recorded artifacts, not to an abstract distribution of all possible training runs.

## 4.4 Representation and backend confounding

BF16 Phase 1 used Transformers/PyTorch; Phase 2 used pinned llama.cpp or BitNet.cpp with TQ1_0/I2_S, different kernels and different scheduling. The one-bit-frontier comparison is scientifically informative as a deployment comparison, but it is not a pure causal estimate of changing only weight precision.

## 4.5 Missingness and stopped streams

`context_unsupported`, repeated 1,800-second timeouts, runtime loss and accelerator unavailability are related to prompt length, model context and deployment cost. Missingness is therefore not assumed random. Available-case results, exact common support, composition diagnostics and 0--4 identification bounds are shown together. No semantic response is invented for an absent row, and no unavailable judgment is scored as zero.

## 4.6 Future validation opportunities

The following would require new data collection and are not part of notebook 10:

- a blinded cybersecurity-expert sample for label-level judge validation;
- repeated Haiku calls to estimate test–retest stability;
- a diverse second-judge panel;
- balanced rubric-order permutations to test pointwise score-option bias;
- controlled error and quality-preserving perturbations following checklist-style evaluator audits; and
- repeated SFT seeds to estimate training variance.

Checklist frameworks can improve interpretability by decomposing subjective Likert judgments into traceable decisions (Lee et al., 2025), but applying a new rubric now would create a new experiment rather than reanalyse the frozen common rubric.

# Visual-design rules

- Prefer several low-dimensional figures to one high-dimensional figure.
- Use the overview heatmap only as a navigation/index figure.
- Use 100% Likert bars for ordinal score shape.
- Use forest plots for paired effects and uncertainty.
- Use horizontal bars for category frequency and failures.
- Use slope plots for BF16-to-one-bit-frontier deployment-stack changes.
- Use log axes only for strictly positive, strongly skewed size/time measures and label the transformation.
- Keep model colours, condition order, representation styling and score colours identical across all figures.
- Use colour-blind-safe palettes and retain readable grayscale contrast.
- Put denominators in captions or directly beside marks.
- Never encode missing values as zero.
- Mark available-case versus exact common-support populations in every comparative caption.
- Keep exact representation labels (`BF16 HF`, `TQ1_0 GGUF`, `native I2_S GGUF`) visible even when a broad `one-bit-frontier` panel is used.
- Label BF16-to-GGUF contrasts as deployment-stack comparisons because inference backend, kernels and scheduling changed with representation.
- Avoid radar, pie, 3-D, dual-axis and dense all-in-one figures.
- Export every plot as SVG/PDF for publication and PNG for notebook display.

# Interpretation safeguards from prior work

The analysis follows these principles:

- Dietz et al. identify circularity, LLM narcissism, loss of opinion diversity and the danger of relying on system-level rankings while label-level judgments disagree. This motivates separate field reporting, rubric-rule audits, self-evaluation separation, common-support analysis and judge-bias diagnostics.
- G-Eval and related work show that rubric-based form evaluation can correlate with human assessments, but also warn of bias toward LLM-generated text. This supports explicit dimensions and short reasons but not treatment of Haiku scores as truth.
- Chiang and Lee show that evaluation-prompt details and explanations can materially affect alignment. This motivates freezing the exact rubric, judge protocol and canonical export schema, retaining malformed judgments for audit and avoiding post hoc reparsing or regeneration.
- Zheng et al. and later judge-bias studies identify position, verbosity and self-enhancement concerns. Position bias is not directly applicable to this pointwise single-candidate design, but verbosity and self/reference preference remain relevant diagnostics.
- Multi-criteria NLG evaluation research shows that desirable dimensions need not correlate. This supports the prohibition on a combined score and the use of paired-dimension plots.
- Pointwise rubric scoring can still show score-option position effects; because this study did not permute rubric order, scale-use diagnostics are descriptive rather than a direct position-bias test (Xu et al., 2026).
- Controlled perturbation and checklist studies show that evaluators may overlook targeted defects or produce scores inconsistent with their explanations. This motivates score–failure-label coherence checks and a reproducible qualitative audit, not retrospective rubric changes (Doddapaneni et al., 2024; Lee et al., 2025).
- Human validation is absent. Notebook 10 therefore reports Haiku-based findings as exploratory and preserves existing deterministic audits only as complementary, coverage-limited weak evidence.

## References

- Dietz, L. et al. (2025). *Principles and Guidelines for the Use of LLM Judges*. ICTIR. https://doi.org/10.1145/3731120.3744588
- Liu, Y. et al. (2023). *G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment*. EMNLP. https://doi.org/10.18653/v1/2023.emnlp-main.153
- Chiang, C.-H. and Lee, H.-Y. (2023). *A Closer Look into Using Large Language Models for Automatic Evaluation*. Findings of EMNLP. https://aclanthology.org/2023.findings-emnlp.599/
- Zheng, L. et al. (2023). *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*. NeurIPS Datasets and Benchmarks. https://arxiv.org/abs/2306.05685
- Sai, A. B. et al. (2021). *Perturbation CheckLists for Evaluating NLG Evaluation Metrics*. EMNLP. https://aclanthology.org/2021.emnlp-main.575/
- Thakur, A. S. et al. (2025). *Judging the Judges: Evaluating Alignment and Vulnerabilities in LLMs-as-Judges*. GEM. https://aclanthology.org/2025.gem-1.33/
- Panickssery, A., Bowman, S. R. and Feng, S. (2024). *LLM Evaluators Recognize and Favor Their Own Generations*. NeurIPS. https://proceedings.neurips.cc/paper_files/paper/2024/hash/7f1f0218e45f5414c79c0679633e47bc-Abstract-Conference.html
- Doddapaneni, S. et al. (2024). *Finding Blind Spots in Evaluator LLMs with Interpretable Checklists*. EMNLP. https://aclanthology.org/2024.emnlp-main.911/
- Chen, Z.-Y. et al. (2025). *Beyond the Surface: Measuring Self-Preference in LLM Judgments*. EMNLP. https://aclanthology.org/2025.emnlp-main.86/
- Lee, Y. et al. (2025). *CheckEval: A Reliable LLM-as-a-Judge Framework for Evaluating Text Generation Using Checklists*. EMNLP. https://aclanthology.org/2025.emnlp-main.796/
- Xu, Y. et al. (2026). *Am I More Pointwise or Pairwise? Revealing Position Bias in Rubric-Based LLM-as-a-Judge*. Findings of EMNLP. https://aclanthology.org/2026.findings-emnlp.1444/

# Notebook 10 output structure

```text
analysis/
├── NOTEBOOK_10_ANALYSIS_SPECIFICATION.md
├── analysis_manifest.json
├── notebook10_analysis.py
├── tables/
│   ├── figure_manifest.csv
│   └── figure_data/
├── figures/
└── diagnostics/
notebooks/
└── 10_analysis.ipynb
```

The implementation should generate every numerical table before its figure and store the exact plotted rows in `analysis/tables/figure_data/`. Figures must never depend on hidden notebook state. Each caption states the population, denominator, score field, aggregation, uncertainty method and relevant limitation.
# Correctness revision before Chapter 5 (`notebook10-v8-correctness-reconciliation`)

This revision is authoritative wherever it is stricter than an earlier description in this specification.

## Retention estimability

The retention index is reported only when the paired BF16 SFT-full gain over the modal baseline is positive **and** its scenario-step cluster-bootstrap 95% lower bound is greater than zero. A merely non-zero denominator is insufficient. When this gate fails, `retention_index`, `retention_low95`, and `retention_high95` are null; the component gaps, their intervals, common-support count, and an explicit non-estimable reason remain reportable. This prevents negative or statistically unsupported BF16 gains from being described as retained deployment benefit.

## Efficiency, reconciliation, and GPU comparability

Efficiency tables use an audited per-completed-row time rather than blindly dividing incomplete captured sessions. Telemetry-complete streams retain observed session time. A partially observed stream may retain the observed point estimate only when at least half of its completed rows are represented by invocation `new_rows`; its row-reconstructed high bound remains alongside it. Streams below that majority-capture gate use reconstructed time. Each BF16/frontier comparison also reports paired median request latency, exact common completed rows, both GPU-type sets, and `same_gpu_flag`. A same-GPU comparison requires both streams to have exactly the same single recorded GPU type. Mixed-GPU streams are not used for break-even inference. Break-even is based on the audited point estimate and separately flags whether the direction remains positive across the reconciled range. The known clean untreated comparisons are the A100 pairs except OpenHermes BF16 (A100/T4 mixture) and Qwen2.5 7B frontier (A100/L4 mixture); all OpenHermes adapter pairs are also same-GPU comparisons. This retains the observed untreated BitNet A100 comparison (approximately 17.8 versus 28.7 seconds per completed row) while reconstruction prevents very sparsely instrumented 7B sessions from creating implausible speed-ups.

## Failure taxonomy and RQ2 profiles

Free-form judge failure labels are preserved verbatim in `failure_label_dictionary.csv` but mapped many-to-one into a fixed broad taxonomy: empty/missing response; repetition/truncation/length; degenerate/incoherent output; format/instruction non-adherence; confidence miscalibration; prediction-reason misalignment; explanation-grounding error; overprediction/unsupported prediction; underprediction/omission; incorrect technique/specificity; ambiguity/uncertainty handling; and other judge-identified failure. The repetition rule intentionally matches `repet|repeat|loop|degenera`, covering *repetition*, *repetitive*, loops and labels such as `degenerate_repetition`. The separate degenerate/incoherent rule matches `garbl|gibber|incoheren|nonsens|corrupt`. Rule order is frozen and auditable in the dictionary. Rates count rows containing a category, not arbitrary label-string frequency. Paired SFT-minus-untreated risk differences use exact prompt support and scenario-step cluster-bootstrap intervals. Between-model failure-profile similarity uses the broad category-rate vectors and reports cosine and Jensen-Shannon similarity with explicit non-estimable states.

## Added and corrected response analyses

- `difference_in_differences.csv` reports `(one-bit SFT - one-bit untreated) - (BF16 SFT - BF16 untreated)` for every model, adapter, and judge dimension on exact four-cell common support.
- `attack_class_treatment_effects.csv` contains paired SFT-minus-untreated effects by ATT&CK technique, representation group, model, adapter, and dimension. “Sufficiently represented” retains the pre-specified `n >= 15` matched-row rule.
- Haiku self-evaluation uses the canonical exported score columns directly. It is a circular descriptive diagnostic, not judge validation.
- Ranked technique metrics include GPT-5.5 teacher, modal baseline, and null baseline through the identical canonical judge-extracted-prediction pipeline.
- Item-difficulty Spearman correlation is not estimable when either model has constant scores. Lowest-score overlap is not estimable when both models place every common row in the low-score set; an overlap of 1 created solely by total floor saturation is never reported as perfect similarity.
- Figure identifiers remain in filenames and `figure_manifest.csv`; plot titles do not include chapter figure numbers.

The affected auditable outputs are `difference_in_differences.csv`, `attack_class_treatment_effects.csv`, `failure_label_dictionary.csv`, `failure_rates.csv`, `failure_risk_differences.csv`, `failure_profile_similarity.csv`, `failure_category_cooccurrence.csv`, `haiku_self_evaluation_summary.csv`, `model_item_difficulty_concordance.csv`, `judge_mediated_rank_metrics.csv`, `retention_index.csv`, `sft_quantization_efficiency.csv`, and `preparation_amortization.csv`.

## Section 1.22: complete positive-score frontier register

Write `analysis/tables/one_bit_positive_score_row_register.csv`. Include every completed one-bit-frontier judgment row for which at least one canonical scalar judge dimension is greater than zero. This is a census, not an extreme-case sample. Preserve prompt/scenario-step identity, split, model, adapter, exact GGUF representation, candidate record/hash, every scalar score, the positive-score dimension list, canonical judge failure labels and extracted predictions, ground truth, and source file. Add a whitespace-normalized 600-character excerpt of the candidate response, its original character count, and an excerpt-truncation flag. Do not parse candidate prose or require valid JSON. Native-I2_S and TQ1_0 rows follow the same rule.

**Interpretation.** The register permits manual inspection of every non-floor one-bit result and prevents qualitative examples from inadvertently excluding BitNet. It is not a prevalence estimate beyond the complete set defined by the rule.

## Section 1.23: deterministic BitNet template example and prevalence

Notebook 10 Section 6F reads only the canonical source, response and judgment exports and verifies `prompt_sha256` before selection. It writes `analysis/tables/figure_data/04X_bitnet_template_example.csv` with four rows: the selected P1 prompt, untreated BF16 P1 answer, native-I2_S P1 answer and native-I2_S P2 answer from a different scenario. P1 and P2 must have completed untreated BF16 and I2_S responses and completed judgments. The selection first requires the I2_S response to match the frozen template regex and BF16 to end naturally (`eos` or `stop`), then minimizes prompt characters and prompt ID. P2 must also share P1's explicit template subtype (`confidence_list` or `techniques_instruction_echo`), so the example supports a same-template claim rather than merely showing two members of a broad degenerate family. If no natural-ending row exists, the rule drops that requirement and records the fallback. Text excerpts are verbatim except literal tabs become four spaces; head/tail lengths and omitted characters are explicit.

The final evidence-based regex is:

```text
^1\.\s*(?:"?(?:Certain|Almost Certain|Somewhat Certain|Neutral|Uncertain)"?\s*[:"-]|"?Techniques"?\s*:\s*A\s+list\s+of\s+(?:top\s+10\s+)?(?:ATT&CK\s+)?techniques)
```

It covers the two directly observed prompt-insensitive opening families—confidence-list enumeration and a direct echo of the requested `Techniques` field—without treating every numbered answer as equivalent. `analysis/tables/bitnet_template_prevalence.csv` reports actual matches and distinct 100-character prefixes for untreated I2_S on 990 and 285 rows, full-SFT I2_S on 285 rows, and untreated BF16 on 990 rows. The result may support or contradict an “almost all” description; the implementation does not widen the rule merely to obtain a high rate. Both output rows in `figure_manifest.csv` preserve canonical source hashes, and the ordinary output catalog records their own hashes.

**How to interpret the externally typeset Figure 04X.** Compare BF16 and I2_S on P1 to see how the deployment representations respond to the identical prompt, then compare I2_S P1 with I2_S P2 to test whether the same opening template persists across scenarios. This is a reproducibly selected illustration; prevalence must be read from the full table rather than inferred from four panels.

## Section 1.23: score and SFT-effect stratification by training-label exposure

Use `evaluation_unseen_technique_rows.csv` as the authoritative unseen-label audit and join it by `prompt_id` to all 285 evaluation rows. Assign one mutually exclusive row status:

- `all_ground_truth_seen_in_604_train`;
- `mixed_seen_and_unseen_ground_truth`;
- `all_ground_truth_unseen_in_604_train`.

This is exact ATT&CK-ID exposure in the feasible distillation-training labels, not a statement about pretraining knowledge or parent-technique familiarity.

Write:

- `evaluation_training_label_exposure_rows.csv`: one row per evaluation prompt with ground-truth count, unseen count, and exposure status;
- `scores_by_training_label_exposure.csv`: available canonical score count, independent scenario-step groups, mean, median, IQR, proportion at least 3, and scenario-step cluster-bootstrap 95% interval for every model, adapter, representation, exposure status, and judge dimension;
- `sft_effects_by_training_label_exposure.csv`: exact-prompt paired SFT-minus-untreated differences, common support, independent groups, mean/median difference, win/tie/loss rates, and scenario-step cluster-bootstrap interval for every adapter and representation.

The paired table is the direct diagnostic for whether observed fine-tuning gains are concentrated among labels represented in the 604-row training subset. Sparse unseen strata must retain their denominator and interval; they must not be pooled silently with seen rows.

## Population alignment amendment: full universe and fixed evaluation pool

Notebook 10 version `notebook10-v11-population-alignment` separates two populations wherever untreated streams are compared with SFT, teacher, baseline, or partially completed streams:

- `all_990`: the full available untreated universe. Untreated-only comparisons across models remain valid and use the broader evidence base. Adapter and reference streams contain only their designed 285 evaluation rows even when retained in this view.
- `evaluation_285`: the frozen validation-plus-test prompt IDs used by every adapter, GPT-5.5 teacher, and trivial baseline. This is the primary population for every cross-treatment, cross-representation, teacher, baseline, and non-inferiority comparison.

The same `population` column is written into score summaries/distributions, availability tables, ranked metrics, stopping/length summaries, failure analyses, field relationships, reference-anchor diagnostics, repeat/variant stability, cross-model concordance, hard-case tables, and ATT&CK-class summaries. Main comparison figures filter `evaluation_285`. Full-universe untreated rows remain sensitivity evidence in the same CSVs rather than silently entering treatment panels.

The untreated one-bit-minus-BF16 deployment contrast is emitted once per model and dimension for `evaluation_285` and once for `all_990`; it is not repeated under four adapter names. The 285 estimate drives Figure 03C, Figure 32, ranked-metric deployment panels, and non-inferiority lookup. `common_support_diagnostics.csv` records the matching intended denominator (285 or 990) for each row.

Compute session totals continue to describe resources actually consumed by each full stream. Like-for-like performance rates are additionally reconstructed from completed row-performance records on `evaluation_285`: completed rows, input/output tokens, elapsed request work, slot-adjusted request seconds, median latency, and row-record peak allocated/reserved memory. Slot-adjusted rates divide each row's elapsed time by its recorded parallel slots, defaulting to one when absent. They are request-processing estimates, not captured GPU-runtime session hours. Figures 24A--C, 30, 31 and 35 use these evaluation-pool rates; preparation, restoration, upload and idle allocation remain separate full-stream accounting.
## Full-universe untreated sensitivity, floor-aware non-inferiority, and size evidence

Notebook 10 version `notebook10-v12-full-universe-sensitivity` extends—not replaces—the primary `evaluation_285` analysis.

`mde_and_noninferiority.csv` contains both `evaluation_285` and `all_990` rows for the untreated one-bit-minus-BF16 deployment contrast. MDE uses the standard deviation of scenario-step mean paired differences and the same frozen 0.3 margin in both populations. `size_contrasts.csv` likewise adds the untreated Qwen2.5 7B-minus-1.5B pair on `all_990` for BF16 and one-bit-frontier representations; SFT size contrasts remain evaluation-only.

Non-inferiority remains a statistical margin result, but it is not automatically interpreted as retained quality. For each one-bit-minus-BF16 row, the table reports the paired BF16 reference mean and the already judged modal/null trivial-baseline means. The fixed anchors come from the 285-row reference pass and are reused—not re-estimated—for both population labels. `reference_floor_flag` is true when the BF16 reference mean is at or below the largest available trivial-baseline mean **or at or below the pre-specified 0.3-point margin itself**: in either case, complete collapse to zero can satisfy the margin and cannot establish preserved useful quality. A statistically non-inferior row with that flag receives `noninferiority_interpretation=at_floor_noninferiority_not_quality_preservation` and `noninferior_above_reference_floor=false`. In addition, an interpretable verdict requires at least 15 paired scored rows; smaller samples receive `insufficient_support_noninferiority_not_interpretable`, even if their raw interval happens to clear the margin. The raw margin result remains in `noninferior_delta_0_3`; this preserves the statistical calculation while preventing floor-saturated or nearly unsupported comparisons from being described as preserved capability.

For item-difficulty overlap, a Jaccard value is not reported when **both** members of a model pair assign low scores (0–1) to at least 95% of common rows. Such a value would approach 1 mechanically because both models are almost always at the floor, not because the overlap is discriminating. `model_item_difficulty_concordance.csv` records both low-score rates, the 0.95 threshold and an explicit non-estimable status.

Section 5.2 full-universe sensitivity figures are saved with `_990_` in their filenames. They are untreated-only versions of score overview/distributions, deployment contrasts, attack-class scores and lowest classes, failure rates/profile similarity, stopping, item-difficulty concordance, prompt-length completion/technique scores, and Hit@k. Their companion figure-data CSVs contain only `population=all_990`. Primary cross-treatment figures remain `evaluation_285`.
