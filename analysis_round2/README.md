# Notebook 10 round-two outputs

This directory is an isolated regeneration of the Notebook 10 figure-fix specification. It reads the existing `analysis/tables/` files but never writes to `analysis/` or `report/`.

## Directory key

- `code/`: renderer, plotting conventions, QA checks, self-test and isolated runner.
- `figures/`: regenerated PNG figures using the original filenames.
- `figure_data/`: one companion CSV for every regenerated figure.
- `tables/`: corrected figure-level summaries and the old/new size register.
- `manifests/`: figure manifest and isolation audit.
- `qa/`: one-row-per-check QA report and the complete PDF contact sheet.

## Reproduction

From the `streamlined_experiment` project environment, run:

```powershell
python analysis_round2/code/test_figure_qa.py
python analysis_round2/code/run_round2.py
```

The runner hashes `analysis/` and `report/` before and after rendering. It raises an error if either existing tree changes.

## Validation state

- 72 regenerated figures and 72 figure-data CSVs.
- 936 final QA checks, all passing.
- Stable-content and explicit content-contract checks pass for every regenerated figure.
- Existing `analysis/` and `report/` trees were unchanged by the full isolated run.
