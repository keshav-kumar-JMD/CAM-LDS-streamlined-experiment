"""Regression checks for Notebook 10 population labels."""
from pathlib import Path
import pandas as pd


TABLES = [
    "model_item_difficulty_concordance.csv",
    "failure_profile_similarity.csv",
    "token_ceiling_and_stop_reasons.csv",
    "judge_mediated_rank_metrics_by_stream.csv",
    "judge_mediated_parent_rank_metrics_by_stream.csv",
]


def test_all_990_is_untreated_model_only():
    root = Path(__file__).resolve().parents[1]
    tables = root / "analysis" / "tables"
    references = {"gpt55_teacher", "baseline_modal", "baseline_null"}
    for name in TABLES:
        frame = pd.read_csv(tables / name, dtype=str, keep_default_na=False)
        full = frame[frame["population"].eq("all_990")]
        condition = "condition_short" if "condition_short" in full else "condition"
        assert full[condition].eq("untreated").all(), name
        if "model" in full:
            assert not full["model"].isin(references).any(), name


def test_compute_resource_tables_have_both_untreated_populations():
    root = Path(__file__).resolve().parents[1]
    tables = root / "analysis" / "tables"
    for name in ("compression_performance_frontier.csv", "deployment_stack_comparisons.csv"):
        frame = pd.read_csv(tables / name, dtype=str, keep_default_na=False)
        untreated = frame[frame["condition_short"].eq("untreated")]
        observed = set(untreated["population"])
        assert {"all_990", "evaluation_285"}.issubset(observed), (name, observed)
