"""Notebook 10: reproducible analysis of the streamlined CAM-LDS experiment.

This module is intentionally separate from the notebook.  The notebook is the
human-readable execution record; this file contains the tested calculations.
It reads only canonical exports under results/final and never reparses raw
candidate or judge text to recover semantic fields.
"""
from __future__ import annotations

import hashlib
import ast
import gc
import json
import math
import os
import re
import subprocess
import sys
import time
import textwrap
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform, jensenshannon
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import cohen_kappa_score
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ANALYSIS_VERSION = "notebook10-v15-reference-ladder-all-fields"
SEED = 42
BOOTSTRAP_REPLICATES = 10_000
MIN_CLASS_N = 15
MIN_NONINFERIORITY_N = 15
FLOOR_SATURATION_RATE = .95
NONINFERIORITY_MARGIN = 0.3
BITNET_TEMPLATE_REGEX = (
    r'^1\.\s*(?:'
    r'"?(?:Certain|Almost Certain|Somewhat Certain|Neutral|Uncertain)"?\s*[:"-]'
    r'|"?Techniques"?\s*:\s*A\s+list\s+of\s+(?:top\s+10\s+)?(?:ATT&CK\s+)?techniques'
    r')'
)
BITNET_CONFIDENCE_TEMPLATE_REGEX = (
    r'^1\.\s*"?(?:Certain|Almost Certain|Somewhat Certain|Neutral|Uncertain)"?\s*[:"-]'
)
BITNET_TECHNIQUES_ECHO_REGEX = (
    r'^1\.\s*"?Techniques"?\s*:\s*A\s+list\s+of\s+(?:top\s+10\s+)?(?:ATT&CK\s+)?techniques'
)

MODELS = ["qwen1b", "hermes", "bitnet2b", "llama3b", "mistral7b", "qwen7b"]
MODEL_LABELS = {
    "qwen1b": "Qwen2.5 1.5B", "hermes": "OpenHermes 1.8B",
    "bitnet2b": "BitNet b1.58 2B", "llama3b": "Llama 3.2 3B",
    "mistral7b": "Mistral 7B v0.3", "qwen7b": "Qwen2.5 7B",
}
REFERENCE_LABELS = {
    "gpt55_teacher": "GPT-5.5 teacher | reference | distillation target",
    "baseline_null": "Trivial baseline | reference | null response",
    "baseline_modal": "Trivial baseline | reference | modal technique",
}
MODEL_PARAMETERS = {
    "qwen1b": 1.5e9, "hermes": 1.8e9, "bitnet2b": 2.0e9,
    "llama3b": 3.0e9, "mistral7b": 7.0e9, "qwen7b": 7.0e9,
}
MODEL_COLORS = {
    "qwen1b": "#0072B2", "hermes": "#E69F00", "bitnet2b": "#009E73",
    "llama3b": "#D55E00", "mistral7b": "#CC79A7", "qwen7b": "#56B4E9",
}
CONDITIONS = ["untreated", "technique", "explanation", "joint", "full"]
CONDITION_COLORS = {
    "untreated": "#4D4D4D", "technique": "#0072B2", "explanation": "#E69F00",
    "joint": "#009E73", "full": "#CC79A7",
}
CONDITION_MARKERS = {"untreated": "o", "technique": "^", "explanation": "s", "joint": "D", "full": "p"}
TREATMENT_LABELS = {
    "untreated": "untreated", "technique": "SFT-technique",
    "explanation": "SFT-explanation", "joint": "SFT-joint", "full": "SFT-full",
}


def stream_display_name(model: str, representation: str, condition: str) -> str:
    """Return the canonical reader-facing stream label from the specification."""
    model_name = MODEL_LABELS.get(model, model)
    treatment = TREATMENT_LABELS.get(condition, condition)
    rep = str(representation).strip().lower()
    if rep in {"hf", "bf16 hf", "bf16", "bfloat16", "untreated_hf"}:
        return f"{model_name} | BF16 HF | {treatment}"
    exact = "native I2_S GGUF" if model == "bitnet2b" or "i2_s" in rep else "TQ1_0 GGUF"
    return f"{model_name} | one-bit frontier [{exact}] | {treatment}"
ORDINAL_COLORS = ["#B2182B", "#EF8A62", "#D9D9D9", "#67A9CF", "#2166AC"]
TRANSITION_COLORS = {"improved": "#009E73", "unchanged": "#BDBDBD", "worsened": "#D55E00"}
SCORE_COLUMNS = [
    "score_technique_identification",
    "score_prediction_evidence_support",
    "score_explanation_grounding",
    "score_prediction_reason_alignment",
    "score_confidence_evidence_support",
    "score_instruction_adherence",
    "confidence_label_agreement",
]
SUBSTANTIVE_SCORES = SCORE_COLUMNS[:6]
SHORT_SCORE = {
    "score_technique_identification": "technique_identification",
    "score_prediction_evidence_support": "prediction_evidence_support",
    "score_explanation_grounding": "explanation_grounding",
    "score_prediction_reason_alignment": "prediction_reason_alignment",
    "score_confidence_evidence_support": "confidence_evidence_support",
    "score_instruction_adherence": "instruction_adherence",
    "confidence_label_agreement": "confidence_label_agreement",
}
PROMPT_BUCKETS = [-np.inf, 2048, 4096, 8192, np.inf]
PROMPT_BUCKET_LABELS = ["<=2048", "2049-4096", "4097-8192", ">8192"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".partial")
    frame.to_csv(temp, index=False, encoding="utf-8-sig")
    os.replace(temp, path)


def safe_float(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    text = str(value).strip()
    if not re.fullmatch(r"[0-4](?:\.0+)?", text):
        return np.nan
    number = float(text)
    return number if 0 <= number <= 4 else np.nan


def boolish(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def canonical_list(value) -> list[str]:
    """Decode schema-declared list cells only; never inspect raw response text."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, list):
        return [str(x) for x in value]
    text = str(value).strip()
    if not text:
        return []
    try:
        decoded = json.loads(text)
        return [str(x) for x in decoded] if isinstance(decoded, list) else []
    except (json.JSONDecodeError, TypeError):
        return []


FAILURE_CATEGORY_RULES = [
    ("empty_or_missing_response", r"empty|no[_ -]?response|missing response|null response|blank response"),
    ("repetition_truncation_or_length", r"repet|repeat|loop|degenera|truncat|length limit|token limit|incomplete response|cut off"),
    ("degenerate_or_incoherent_output", r"garbl|gibber|incoheren|nonsens|corrupt"),
    ("format_or_instruction_nonadherence", r"json|format|markdown|code[_ -]?fence|schema|instruction|required field|unexpected text|list length|top.?10|prose outside"),
    ("confidence_miscalibration", r"confidence|certainty|overconfident|underconfident|calibrat"),
    ("prediction_reason_misalignment", r"reason.*align|align.*reason|unexplained prediction|not tied|fails to connect|disconnect"),
    ("explanation_grounding_error", r"explanation|grounding|fabricat|invent|hallucin|artifact|log evidence|factual|unsupported claim"),
    ("overprediction_or_unsupported_prediction", r"overpredict|unsupported|weak support|speculative|too broad|extraneous|irrelevant technique"),
    ("underprediction_or_omission", r"underpredict|missing|omitt|low[_ -]?ground[_ -]?truth|coverage|incomplete technique|fails to include"),
    ("incorrect_technique_or_specificity", r"incorrect|wrong technique|misclass|mismatch|subtechnique|parent technique|specificity|exact.?id"),
    ("ambiguity_or_uncertainty_handling", r"ambig|uncertain|hedg|insufficient evidence|benign alternative"),
]


def broad_failure_category(label: str) -> tuple[str, str]:
    """Map a free-form judge label to a small, auditable failure taxonomy."""
    text = re.sub(r"[_-]+", " ", str(label).strip().lower())
    for category, pattern in FAILURE_CATEGORY_RULES:
        if re.search(pattern, text):
            return category, pattern
    return "other_judge_identified_failure", "fallback: no broad-category lexical rule matched"


def declared_list_with_state(value) -> tuple[str, list[str]]:
    """Deserialize one schema-declared list cell without inspecting raw text.

    This deliberately distinguishes an explicit empty list from a missing/null
    field.  It is used only for canonical judge-extracted prediction and
    ground-truth columns, never to recover IDs from candidate/judge prose.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "missing_or_null", []
    if isinstance(value, list):
        decoded = value
    else:
        text = str(value).strip()
        if not text or text.lower() in {"null", "none", "nan"}:
            return "missing_or_null", []
        try:
            decoded = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return "malformed_non_list", []
    if not isinstance(decoded, list):
        return "malformed_non_list", []
    output = []
    seen = set()
    for item in decoded:
        token = str(item).strip().upper()
        if token and token not in seen:
            seen.add(token); output.append(token)
    return ("nonempty_list" if output else "empty_list"), output


def attack_parent_id(value: str) -> str:
    """Collapse a valid ATT&CK sub-technique to its parent for sensitivity analysis."""
    token = str(value).strip().upper()
    match = re.fullmatch(r"(T\d{4})(?:\.\d{3})?", token)
    return match.group(1) if match else token


def parent_normalized_unique(values) -> list[str]:
    """Parent-normalize an ordered list and retain first appearance only."""
    output, seen = [], set()
    for value in values:
        token = attack_parent_id(value)
        if token and token not in seen:
            seen.add(token); output.append(token)
    return output


def parse_filename(path: Path) -> dict[str, str]:
    parts = path.stem.split("__")
    meta = {"source_file": path.name, "notebook_phase": parts[0] if parts else ""}
    if "judgment" in parts:
        i = parts.index("judgment")
        tail = parts[i + 2 :]
        if len(tail) >= 4:
            meta.update(model=tail[0], condition_file=tail[1], representation_file=tail[2], split_file=tail[3])
    elif "response" in parts:
        i = parts.index("response")
        tail = parts[i + 1 :]
        if len(tail) >= 4:
            meta.update(model=tail[0], condition_file=tail[1], representation_file=tail[2], split_file=tail[3])
    return meta


def condition_short(value: str) -> str:
    text = str(value or "").lower()
    if "untreated" in text:
        return "untreated"
    for name in ("technique", "explanation", "joint", "full"):
        if name in text:
            return name
    return text


def representation_group(value: str) -> str:
    text = str(value or "").lower()
    if text in {"reference", "source"}: return "reference"
    return "BF16 HF" if text in {"hf", "bf16", "bfloat16"} or "hf" in text else "one-bit frontier"


def representation_label(value: str) -> str:
    text = str(value or "").lower()
    if text in {"reference", "source"}: return "reference"
    if "native_i2_s" in text or "i2_s" in text:
        return "native I2_S"
    if "tq1_0" in text:
        return "TQ1_0"
    return "BF16 HF"


def cluster_bootstrap(values: pd.Series, clusters: pd.Series, statistic=np.mean,
                      replicates: int = BOOTSTRAP_REPLICATES, seed: int = SEED) -> tuple[float, float]:
    data = pd.DataFrame({"value": pd.to_numeric(values, errors="coerce"), "cluster": clusters}).dropna()
    if data.empty:
        return np.nan, np.nan
    grouped = [g["value"].to_numpy(float) for _, g in data.groupby("cluster", sort=True)]
    if len(grouped) < 2:
        x = float(statistic(data["value"].to_numpy(float)))
        return x, x
    rng = np.random.default_rng(seed)
    estimates = np.empty(replicates, dtype=float)
    for i in range(replicates):
        selected = rng.integers(0, len(grouped), len(grouped))
        sample = np.concatenate([grouped[j] for j in selected])
        estimates[i] = statistic(sample)
    return tuple(np.quantile(estimates, [0.025, 0.975]))

def cluster_bootstrap_mean_fast(values: pd.Series, clusters: pd.Series,
                                replicates: int = BOOTSTRAP_REPLICATES, seed: int = SEED):
    """Vectorized cluster-bootstrap interval for a mean or binary proportion."""
    data=pd.DataFrame({"value":pd.to_numeric(values,errors="coerce"),"cluster":clusters}).dropna()
    if data.empty:return np.nan,np.nan
    grouped=data.groupby("cluster").value.agg(["sum","count"])
    sums=grouped["sum"].to_numpy(float);counts=grouped["count"].to_numpy(float);g=len(grouped)
    if g<2:
        value=sums.sum()/counts.sum();return value,value
    rng=np.random.default_rng(seed);estimates=np.empty(replicates,float);offset=0
    while offset<replicates:
        n=min(1000,replicates-offset);idx=rng.integers(0,g,size=(n,g))
        estimates[offset:offset+n]=sums[idx].sum(axis=1)/counts[idx].sum(axis=1);offset+=n
    return tuple(np.quantile(estimates,[.025,.975]))


def bh_adjust(pvalues: Sequence[float]) -> np.ndarray:
    p = np.asarray(pvalues, dtype=float)
    out = np.full(len(p), np.nan)
    valid = np.isfinite(p)
    if not valid.any():
        return out
    pv = p[valid]
    order = np.argsort(pv)
    ranked = pv[order]
    adjusted = np.minimum.accumulate((ranked * len(ranked) / np.arange(1, len(ranked) + 1))[::-1])[::-1]
    restored = np.empty_like(adjusted)
    restored[order] = np.minimum(adjusted, 1)
    out[valid] = restored
    return out


def sign_flip_pvalue(differences: np.ndarray, draws: int = 10_000, seed: int = SEED) -> float:
    x = np.asarray(differences, dtype=float)
    x = x[np.isfinite(x)]
    if not len(x):
        return np.nan
    observed = abs(x.mean())
    rng = np.random.default_rng(seed)
    sims = np.empty(draws)
    for i in range(draws):
        sims[i] = abs(np.mean(x * rng.choice([-1, 1], len(x))))
    return float((1 + np.sum(sims >= observed)) / (draws + 1))


def spearman_safe(a, b) -> tuple[float, float]:
    frame = pd.DataFrame({"a": a, "b": b}).dropna()
    if len(frame) < 3 or frame.a.nunique() < 2 or frame.b.nunique() < 2:
        return np.nan, np.nan
    result = stats.spearmanr(frame.a, frame.b)
    return float(result.statistic), float(result.pvalue)


def savefig(path: Path, fig=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = fig or plt.gcf()
    temp=path.with_name(f".{path.stem}.{os.getpid()}.partial{path.suffix}")
    try:
        fig.savefig(temp,dpi=220,bbox_inches="tight",facecolor="white")
        for attempt in range(5):
            try:os.replace(temp,path);break
            except PermissionError:
                if attempt==4:raise
                time.sleep(.25*(attempt+1))
    finally:
        if temp.exists():temp.unlink(missing_ok=True)
        plt.close(fig)


def empty_figure(path: Path, title: str, reason: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.axis("off")
    ax.text(.5, .64, title, ha="center", va="center", fontsize=15, weight="bold")
    ax.text(.5, .42, reason, ha="center", va="center", fontsize=10, wrap=True)
    savefig(path, fig)


@dataclass
class Paths:
    root: Path

    @property
    def final(self): return self.root / "results" / "final"
    @property
    def analysis(self): return self.root / "analysis"
    @property
    def tables(self): return self.analysis / "tables"
    @property
    def figures(self): return self.analysis / "figures"
    @property
    def figure_data(self): return self.tables / "figure_data"


class Notebook10Analysis:
    def __init__(self, root: str | Path, bootstrap_replicates: int = BOOTSTRAP_REPLICATES):
        self.p = Paths(Path(root).resolve())
        self.bootstrap_replicates = int(bootstrap_replicates)
        self.p.tables.mkdir(parents=True, exist_ok=True)
        self.p.figures.mkdir(parents=True, exist_ok=True)
        self.p.figure_data.mkdir(parents=True, exist_ok=True)
        sns.set_theme(style="whitegrid", context="notebook")
        self.generated: list[Path] = []
        figure_manifest = self.p.tables / "figure_manifest.csv"
        self.figure_records: list[dict] = (pd.read_csv(figure_manifest, dtype=str, keep_default_na=False).to_dict("records")
                                           if figure_manifest.exists() else [])
        self.prompts = self._read("00__source__prompts.csv")
        self.truth = self._read("00__source__ground_truth.csv")
        self.splits = self._read("00__source__split_manifest.csv")
        self.responses = self._load_family("0[1245]__response__*.csv")
        self.judgments = self._load_judgments()
        self.self_judgments = self._read_optional("03__judgment__haiku_self__validation_test.csv")
        self.compute_invocations = self._read_compute("invocations/09__compute__invocation_inventory.csv")
        self.compute_rows = self._read_compute("rows/09__compute__row_performance.csv")
        self.compute_coverage = self._read_compute("rows/09__compute__coverage_and_retries.csv")
        self.compute_streams = self._read_compute("stream_summary/09__compute__stream_summary.csv")
        self.compute_models = self._read_compute("stream_summary/09__compute__model_resource_summary.csv")
        self.api_usage = self._read_compute("stream_summary/09__compute__api_usage_summary.csv")
        all_long = self._build_long()
        self.reference_long = all_long[all_long.model.isin(REFERENCE_LABELS)].copy() if not all_long.empty else all_long.copy()
        self.long = all_long[all_long.model.isin(MODELS)].copy() if not all_long.empty else all_long.copy()
        self.evaluation_ids = set(self.splits.loc[self.splits["split"].isin(["validation", "test"]), "prompt_id"].astype(str))

    def _population_views(self, frame: pd.DataFrame):
        """Yield the full available universe and the fixed 285-row evaluation pool.

        Adapter streams contain only evaluation rows in both views; untreated
        streams retain their 990-row universe in `all_990` and are restricted
        to the same validation/test IDs as adapters in `evaluation_285`.
        """
        yield "all_990", frame
        if "prompt_id" in frame:
            ids = frame["prompt_id"].astype(str)
            yield "evaluation_285", frame[ids.isin(self.evaluation_ids)].copy()
        else:
            yield "evaluation_285", frame.iloc[0:0].copy()

    def _read(self, name: str) -> pd.DataFrame:
        return pd.read_csv(self.p.final / name, dtype=str, keep_default_na=False)

    def _read_optional(self, name: str) -> pd.DataFrame:
        path = self.p.final / name
        return pd.read_csv(path, dtype=str, keep_default_na=False) if path.exists() else pd.DataFrame()

    def _read_compute(self, suffix: str) -> pd.DataFrame:
        path = self.p.final / "compute" / suffix
        return pd.read_csv(path, low_memory=False) if path.exists() else pd.DataFrame()

    def _load_family(self, pattern: str) -> pd.DataFrame:
        frames = []
        for path in sorted(self.p.final.glob(pattern)):
            frame = pd.read_csv(path, dtype=str, keep_default_na=False, low_memory=False)
            for key, value in parse_filename(path).items():
                frame[key] = value
            frames.append(frame)
        return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()

    def _load_judgments(self) -> pd.DataFrame:
        frames = []
        for pattern in ("03__judgment__legacy_common__*.csv", "07__judgment__quantized_common__*.csv",
                        "07__judgment__teacher_common__*.csv", "07__judgment__reference_common__*.csv"):
            for path in sorted(self.p.final.glob(pattern)):
                frame = pd.read_csv(path, dtype=str, keep_default_na=False, low_memory=False)
                for key, value in parse_filename(path).items(): frame[key] = value
                frames.append(frame)
        return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()

    def _build_long(self) -> pd.DataFrame:
        j = self.judgments.copy()
        if j.empty:
            return j
        for col in ("model", "condition", "representation", "split", "group_id"):
            if col not in j: j[col] = ""
        model = j["model"].fillna("").astype(str)
        condition = j["condition"].fillna("").astype(str)
        representation = j["representation"].fillna("").astype(str)
        model_fallback = j.get("model_file", pd.Series("", index=j.index)).fillna("").astype(str)
        condition_fallback = j.get("condition_file", pd.Series("", index=j.index)).fillna("").astype(str)
        representation_fallback = j.get("representation_file", pd.Series("", index=j.index)).fillna("").astype(str)
        j["model"] = model.where(model.ne(""), model_fallback)
        j["condition_raw"] = condition.where(condition.ne(""), condition_fallback)
        j["condition_short"] = j["condition_raw"].map(condition_short)
        j["representation"] = representation.where(representation.ne(""), representation_fallback)
        j["representation_group"] = j["representation"].map(representation_group)
        j["representation_label"] = j["representation"].map(representation_label)
        j["group_id"] = j["group_id"].where(j["group_id"].ne(""), j["scenario"].astype(str) + "::" + j["step"].astype(str))
        j["judgment_completed"] = j["status"].eq("completed")
        j["candidate_completed"] = j.get("candidate_status", "").eq("completed")
        for score in SCORE_COLUMNS:
            if score not in j: j[score] = ""
            j[score + "_raw"] = j[score]
            j[score + "_numeric"] = j[score].map(safe_float)
            j[score + "_available"] = j[score + "_numeric"].notna()
            j[score + "_valid_range"] = j[score + "_available"]
        numeric_candidates = [
            "candidate_output_tokens", "candidate_input_tokens", "candidate_elapsed_seconds",
            "candidate_output_tokens_per_second", "candidate_max_new_tokens", "candidate_parallel_slots",
        ]
        for col in numeric_candidates:
            if col not in j: j[col] = np.nan
            j[col] = pd.to_numeric(j[col], errors="coerce")
        j["prompt_length_bucket"] = pd.cut(j["candidate_input_tokens"], PROMPT_BUCKETS, labels=PROMPT_BUCKET_LABELS)
        j["candidate_response_empty_bool"] = boolish(j.get("candidate_response_empty", pd.Series(False, index=j.index)))
        j["candidate_valid_json_bool"] = boolish(j.get("candidate_valid_json", pd.Series(False, index=j.index)))
        j["candidate_schema_adherent_bool"] = boolish(j.get("candidate_schema_adherent", pd.Series(False, index=j.index)))
        return j

    def _write(self, frame: pd.DataFrame, name: str) -> pd.DataFrame:
        path = self.p.tables / name
        atomic_csv(frame, path); self.generated.append(path); return frame

    def _figure(self, path: Path, fig, data: pd.DataFrame | None = None,
                key: str = "", sources: str = "", show_key: bool = True,
                key_position: str = "bottom") -> None:
        """Save a keyed figure and its auditable plotted-data CSV.

        Figure identifiers remain in filenames and the manifest, but are not
        injected into plot titles; chapter layout controls numbering.
        """
        match = re.match(r"^(S?\d+[A-Z]?(?:_all)?)", path.stem)
        figure_id = match.group(1) if match else path.stem
        if fig._suptitle is None:
            title = next((ax.get_title() for ax in fig.axes if ax.get_title()), path.stem.replace("_", " "))
            fig.suptitle(title)

        handles, labels = [], []
        for ax in fig.axes:
            h, l = ax.get_legend_handles_labels()
            for handle, label in zip(h, l):
                if label and not label.startswith("_") and label not in labels:
                    handles.append(handle); labels.append(label)
            legend = ax.get_legend()
            if legend is not None: legend.remove()
        if not handles:
            present = [] if data is None or "model" not in data else [m for m in MODELS if m in set(data["model"].astype(str))]
            handles = [Patch(facecolor=MODEL_COLORS[m], label=MODEL_LABELS[m]) for m in present]
            labels = [MODEL_LABELS[m] for m in present]
        if handles and show_key:
            if key_position == "top":
                fig.legend(handles, labels, title="Key", loc="upper center", ncol=min(4, len(labels)),
                           bbox_to_anchor=(.5, .945), frameon=True, fontsize=8)
                fig.subplots_adjust(top=max(.72,.84-.035*math.ceil(len(labels)/4)))
            else:
                fig.legend(handles, labels, title="Key", loc="lower center", ncol=min(4, len(labels)),
                           bbox_to_anchor=(.5, -.02), frameon=True, fontsize=8)
                fig.subplots_adjust(bottom=max(.12, .06 + .035 * math.ceil(len(labels) / 4)))

        key_text = key or ("Colours and symbols are defined by the visible figure key; "
                           "BF16 HF and one-bit-frontier panels use matched scales where stated.")
        data_path = self.p.figure_data / f"{figure_id}__{path.stem}.csv"
        plotted = data.copy() if isinstance(data, pd.DataFrame) else pd.DataFrame([
            {"figure_id": figure_id, "note": "Figure is derived from the source tables named in figure_manifest.csv."}
        ])
        atomic_csv(plotted, data_path)
        self.generated.append(data_path)
        self.figure_records = [r for r in self.figure_records if r.get("figure_file") != path.name]
        self.figure_records.append({"figure_id": figure_id, "figure_file": path.name,
                                    "figure_data_csv": str(data_path.relative_to(self.p.analysis)),
                                    "key": key_text, "source_tables": sources})
        savefig(path, fig); self.generated.append(path)
        atomic_csv(pd.DataFrame(self.figure_records), self.p.tables / "figure_manifest.csv")

    def _group_cols(self):
        return ["model", "condition_short", "representation_group", "representation_label"]

    def build_manifest(self, final: bool = False) -> dict:
        inputs = list(self.p.final.rglob("*.csv"))
        inputs += [self.p.root/"data"/name for name in ("prompts.csv","ground_truth.csv","split_manifest.csv","context_feasibility.csv","training_exclusions.csv")]
        inputs = sorted({path.resolve() for path in inputs if path.exists()})
        existing_outputs = sorted([p for directory in (self.p.tables, self.p.figures) for p in directory.rglob("*") if p.is_file()])
        try:
            commit = subprocess.check_output(["git", "-C", str(self.p.root), "rev-parse", "HEAD"], text=True).strip()
        except Exception:
            commit = "unavailable"
        manifest = {
            "analysis_version": ANALYSIS_VERSION, "git_commit": commit, "seed": SEED,
            "bootstrap_replicates": self.bootstrap_replicates, "confidence_level": .95,
            "resampling_cluster": "scenario-step group_id", "minimum_class_n": MIN_CLASS_N,
            "model_parameters": MODEL_PARAMETERS, "model_labels": MODEL_LABELS,
            "reference_labels": REFERENCE_LABELS, "noninferiority_margin": NONINFERIORITY_MARGIN,
            "noninferiority_margin_status": "pre-specified before teacher and trivial-baseline scores were analysed",
            "score_dimensions": SCORE_COLUMNS,
            "missing_data_rule": "no imputation; canonical scalar scores only; available-case and exact common support",
            "mitre_attack_release": "not bundled; tactic analysis marked unavailable unless a pinned mapping is supplied",
            "packages": {name: getattr(sys.modules.get(name), "__version__", "unknown") for name in ("pandas", "numpy", "matplotlib", "seaborn", "scipy", "sklearn")},
            "inputs": [{"path": str(x.relative_to(self.p.root)), "sha256": sha256(x)} for x in inputs],
            "outputs": [{"path": str(x.relative_to(self.p.root)), "sha256": sha256(x)} for x in existing_outputs] if final else [],
        }
        path = self.p.analysis / "analysis_manifest.json"
        path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        if final and path not in self.generated: self.generated.append(path)
        return manifest

    # ---------- Section 1.1: eligibility and availability ----------
    def availability(self) -> None:
        rows = []
        for population,data in self._population_views(self.long):
            for keys, g in data.groupby(self._group_cols(), dropna=False):
                intended_here = 285 if population == "evaluation_285" or keys[1] != "untreated" else 990
                for score in SCORE_COLUMNS:
                    numeric = g[score + "_numeric"]
                    raw = g[score + "_raw"].astype(str).str.strip()
                    rows.append(dict(zip(self._group_cols(), keys), population=population, dimension=score,
                        intended_rows=intended_here, candidate_rows=len(g),
                        completed_candidate_responses=int(g["candidate_completed"].sum()),
                        completed_judgments=int(g["judgment_completed"].sum()),
                        scalar_in_range_scores=int(numeric.notna().sum()),
                        unavailable_scores=int(numeric.isna().sum()),
                        null_or_empty=int(raw.eq("").sum()), non_scalar_or_malformed=int((raw.ne("") & numeric.isna()).sum())))
        avail = self._write(pd.DataFrame(rows), "judge_score_availability.csv")
        bounds = avail.copy()
        n = bounds["intended_rows"].clip(lower=1)
        sums = []
        lookup = {(population,*keys):g for population,data in self._population_views(self.long) for keys,g in data.groupby(self._group_cols(),dropna=False)}
        for _, row in bounds.iterrows():
            key = (row["population"],*(row[c] for c in self._group_cols()))
            values = lookup[key][row["dimension"] + "_numeric"].dropna()
            sums.append(values.sum())
        bounds["observed_score_sum"] = sums
        bounds["worst_case_mean"] = bounds["observed_score_sum"] / n
        bounds["best_case_mean"] = (bounds["observed_score_sum"] + 4 * (n - bounds["scalar_in_range_scores"]).clip(lower=0)) / n
        self._write(bounds, "judge_score_availability_bounds.csv")
        shown=avail[avail.population.eq("evaluation_285")]
        pivot = shown.pivot_table(index=["model", "representation_label"], columns="dimension", values="scalar_in_range_scores", aggfunc="sum")
        fig, ax = plt.subplots(figsize=(13, 6)); sns.heatmap(pivot, cmap="Blues", ax=ax, cbar_kws={"label": "Scalar scores"})
        ax.set_title("Judge-score availability (canonical scalar values only)"); ax.set_xlabel(""); ax.set_ylabel("")
        self._figure(self.p.figures / "00_judge_score_availability.png", fig)

    # ---------- Section 1.2: score summaries ----------
    def score_summaries(self) -> None:
        summaries, distributions = [], []
        for population,data in self._population_views(self.long):
            for keys, g in data.groupby(self._group_cols(), dropna=False):
                base = dict(zip(self._group_cols(), keys)) | {"population":population}
                for score in SCORE_COLUMNS:
                    x = g[score + "_numeric"].dropna()
                    lo, hi = cluster_bootstrap(g[score + "_numeric"], g["group_id"], np.mean, self.bootstrap_replicates)
                    hi3 = lambda a: np.mean(np.asarray(a) >= 3)
                    p_lo, p_hi = cluster_bootstrap(g[score + "_numeric"], g["group_id"], hi3, self.bootstrap_replicates)
                    summaries.append(base | dict(dimension=score, intended_rows=285 if population=="evaluation_285" or keys[1]!="untreated" else 990,
                        eligible_n=len(x), unavailable_n=int(g[score + "_numeric"].isna().sum()), mean=x.mean(), median=x.median(),
                        q1=x.quantile(.25), q3=x.quantile(.75), proportion_ge3=(x >= 3).mean(), mean_low95=lo, mean_high95=hi,
                        ge3_low95=p_lo, ge3_high95=p_hi))
                    counts=x.value_counts().reindex(range(5), fill_value=0)
                    for value in range(5):
                        distributions.append(base | dict(dimension=score, score=value, count=int(counts[value]),
                            proportion=counts[value] / len(x) if len(x) else np.nan, denominator=len(x)))
        summary = self._write(pd.DataFrame(summaries), "model_scores_by_dimension.csv")
        dist = self._write(pd.DataFrame(distributions), "model_score_distributions.csv")
        scale = []
        for phase, g in self.long.groupby("notebook_phase"):
            for score in SCORE_COLUMNS:
                x = g[score + "_numeric"].dropna().astype(int)
                counts = x.value_counts().reindex(range(5), fill_value=0)
                p = counts / counts.sum() if counts.sum() else counts.astype(float)
                nz = p[p > 0]
                entropy = float(-(nz * np.log(nz)).sum()) if len(nz) else np.nan
                scale.append(dict(judge_phase=phase, dimension=score, n=len(x), modal_category=(counts.idxmax() if len(x) else np.nan),
                    floor_rate=(x.eq(0).mean() if len(x) else np.nan), ceiling_rate=(x.eq(4).mean() if len(x) else np.nan),
                    entropy=entropy, effective_categories=(math.exp(entropy) if np.isfinite(entropy) else np.nan),
                    top_two_concentration=(p.nlargest(2).sum() if len(x) else np.nan), **{f"score_{i}_rate": p.get(i, np.nan) for i in range(5)}))
        scale_df = self._write(pd.DataFrame(scale), "judge_scale_use.csv")
        self._overview_heatmaps(summary)
        self._ordinal_figures(dist)
        fig, ax = plt.subplots(figsize=(12, 5))
        if not scale_df.empty:
            plot = scale_df.melt(id_vars=["judge_phase", "dimension"], value_vars=[f"score_{i}_rate" for i in range(5)], var_name="score", value_name="rate")
            sns.barplot(plot, x="dimension", y="rate", hue="score", palette=ORDINAL_COLORS, ax=ax)
            ax.tick_params(axis="x", rotation=25); ax.set_ylabel("Proportion"); ax.set_xlabel("")
        ax.set_title("Judge scale use by rubric dimension")
        self._figure(self.p.figures / "02G_judge_scale_use.png", fig)

    def _overview_heatmaps(self, summary: pd.DataFrame) -> None:
        summary=summary[summary.population.eq("evaluation_285")]
        for group, filename in [("BF16 HF", "01A_score_overview_bf16.png"), ("one-bit frontier", "01B_score_overview_one_bit_frontier.png")]:
            data = summary[(summary.representation_group == group) & summary.dimension.isin(SUBSTANTIVE_SCORES)]
            panels = []
            for condition in CONDITIONS:
                panel = data[data.condition_short == condition].pivot_table(index="model", columns="dimension", values="mean", aggfunc="mean").reindex(MODELS)
                panels.append((condition, panel))
            fig, axes = plt.subplots(1, 5, figsize=(25, 5.5), sharey=True)
            for ax, (condition, panel) in zip(axes, panels):
                sns.heatmap(panel, vmin=0, vmax=4, cmap="RdBu", center=2, ax=ax, cbar=condition == CONDITIONS[-1], annot=False)
                ax.set_title(condition.title()); ax.set_xlabel(""); ax.set_ylabel("")
                ax.set_xticklabels([SHORT_SCORE.get(x.get_text(), x.get_text()).replace("_", "\n") for x in ax.get_xticklabels()], rotation=0, fontsize=7)
            fig.suptitle(f"Judge-score overview — {group} (available-case means; 0–4 common scale)", y=1.02)
            self._figure(self.p.figures / filename, fig)

    def _ordinal_figures(self, dist: pd.DataFrame) -> None:
        dist=dist[dist.population.eq("evaluation_285")]
        for score in SCORE_COLUMNS:
            short = SHORT_SCORE[score]
            for group, code in [("BF16 HF", "bf16"), ("one-bit frontier", "one_bit")]:
                for conditions, prefix, suffix in [(["untreated", "full"], "02A" if code == "bf16" else "02B", "untreated_full"),
                                                   (["technique", "explanation", "joint", "full"], "S02A" if code == "bf16" else "S02B", "conditional_adapters")]:
                    subset = dist[(dist.dimension == score) & (dist.representation_group == group) & dist.condition_short.isin(conditions)]
                    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)
                    for ax, model in zip(axes.ravel(), MODELS):
                        m = subset[subset.model == model]
                        left = np.zeros(len(conditions))
                        for value in range(5):
                            vals=[]
                            for cond in conditions:
                                row=m[(m.condition_short==cond)&(m.score==value)]
                                vals.append(float(row.proportion.iloc[0]) if len(row) else 0)
                            ax.barh(conditions, vals, left=left, color=ORDINAL_COLORS[value], label=str(value)); left += vals
                        ax.set_xlim(0, 1); ax.set_title(MODEL_LABELS[model])
                    handles=[plt.Rectangle((0,0),1,1,color=c) for c in ORDINAL_COLORS]
                    fig.legend(handles, [str(i) for i in range(5)], title="Score", loc="lower center", ncol=5)
                    fig.suptitle(f"{short.replace('_',' ').title()} — {group}"); fig.tight_layout(rect=[0,.06,1,.96])
                    self._figure(self.p.figures / f"{prefix}_{short}_{code}_{suffix}.png", fig)

    # ---------- Section 1.3: paired contrasts ----------
    def contrasts(self) -> None:
        results, transitions, diagnostics = [], [], []
        lookup = {(m, c, r): g for (m,c,r), g in self.long.groupby(["model","condition_short","representation_group"])}
        for model in MODELS:
            for adapter in CONDITIONS[1:]:
                comparisons = [
                    ("bf16_sft_minus_untreated", (adapter,"BF16 HF"), ("untreated","BF16 HF")),
                    ("one_bit_sft_minus_untreated", (adapter,"one-bit frontier"), ("untreated","one-bit frontier")),
                    ("untreated_one_bit_minus_bf16", ("untreated","one-bit frontier"), ("untreated","BF16 HF")),
                    ("sft_one_bit_minus_bf16", (adapter,"one-bit frontier"), (adapter,"BF16 HF")),
                ]
                for cname, akey, bkey in comparisons:
                    # The untreated deployment contrast is model-level rather
                    # than adapter-level. Emit it once, with the 285 estimate
                    # primary and the 990 universe retained as sensitivity.
                    if cname=="untreated_one_bit_minus_bf16" and adapter!=CONDITIONS[1]:continue
                    a=lookup.get((model,*akey)); b=lookup.get((model,*bkey))
                    if a is None or b is None: continue
                    populations=("evaluation_285","all_990") if cname=="untreated_one_bit_minus_bf16" else ("evaluation_285",)
                    for population in populations:
                        aa=a if population=="all_990" else a[a.prompt_id.astype(str).isin(self.evaluation_ids)]
                        bb=b if population=="all_990" else b[b.prompt_id.astype(str).isin(self.evaluation_ids)]
                        adapter_label="untreated" if cname=="untreated_one_bit_minus_bf16" else adapter
                        for score in SCORE_COLUMNS:
                            x=aa[["prompt_id","group_id",score+"_numeric"]].rename(columns={score+"_numeric":"a"})
                            y=bb[["prompt_id",score+"_numeric"]].rename(columns={score+"_numeric":"b"})
                            pair=x.merge(y,on="prompt_id").dropna(subset=["a","b"]); d=pair.a-pair.b
                            lo,hi=cluster_bootstrap(d,pair.group_id,np.mean,self.bootstrap_replicates)
                            p=sign_flip_pvalue(d.to_numpy(), min(self.bootstrap_replicates,10_000))
                            results.append(dict(model=model,adapter=adapter_label,contrast=cname,population=population,dimension=score,common_support_n=len(pair),
                                mean_difference=d.mean(),median_difference=d.median(),q1_difference=d.quantile(.25),q3_difference=d.quantile(.75),
                                improved_rate=(d>0).mean(),unchanged_rate=(d==0).mean(),worsened_rate=(d<0).mean(),
                                probability_superiority_with_ties=(d.gt(0).mean()+.5*d.eq(0).mean()),low95=lo,high95=hi,p_value=p))
                            transitions.append(dict(model=model,adapter=adapter_label,contrast=cname,population=population,dimension=score,n=len(pair),
                                improved=int((d>0).sum()),unchanged=int((d==0).sum()),worsened=int((d<0).sum()),
                                rescued=int(((pair.b<=1)&(pair.a>=3)).sum()),degraded=int(((pair.b>=3)&(pair.a<=1)).sum())))
                        ids=set(aa.prompt_id)&set(bb.prompt_id);intended=990 if population=="all_990" else 285
                        diagnostics.append(dict(model=model,adapter=adapter_label,contrast=cname,population=population,intended_n=intended,common_response_ids=len(ids),
                            missing_from_intended=intended-len(ids)))
                # Direct RQ3 contrast: did the SFT gain survive the move from
                # BF16/HF to the one-bit deployment stack?  All four cells are
                # paired on the exact same prompt IDs before differencing.
                cells={
                    "bf16_sft":lookup.get((model,adapter,"BF16 HF")),
                    "bf16_untreated":lookup.get((model,"untreated","BF16 HF")),
                    "onebit_sft":lookup.get((model,adapter,"one-bit frontier")),
                    "onebit_untreated":lookup.get((model,"untreated","one-bit frontier")),
                }
                if all(frame is not None for frame in cells.values()):
                    common=set.intersection(*(set(frame.prompt_id) for frame in cells.values()))
                    diagnostics.append(dict(model=model,adapter=adapter,contrast="difference_in_differences_sft_survival",population="evaluation_285",
                        intended_n=285,common_response_ids=len(common),missing_from_intended=285-len(common)))
                    for score in SCORE_COLUMNS:
                        joined=None
                        for name,frame in cells.items():
                            part=frame[["prompt_id","group_id",score+"_numeric"]].drop_duplicates("prompt_id").rename(columns={score+"_numeric":name})
                            joined=part if joined is None else joined.merge(part[["prompt_id",name]],on="prompt_id")
                        joined=joined.dropna(subset=list(cells))
                        d=(joined.onebit_sft-joined.onebit_untreated)-(joined.bf16_sft-joined.bf16_untreated)
                        lo,hi=cluster_bootstrap_mean_fast(d,joined.group_id,self.bootstrap_replicates)
                        p=sign_flip_pvalue(d.to_numpy(),min(self.bootstrap_replicates,10_000))
                        results.append(dict(model=model,adapter=adapter,contrast="difference_in_differences_sft_survival",population="evaluation_285",dimension=score,
                            common_support_n=len(joined),mean_difference=d.mean(),median_difference=d.median(),q1_difference=d.quantile(.25),q3_difference=d.quantile(.75),
                            improved_rate=(d>0).mean(),unchanged_rate=(d==0).mean(),worsened_rate=(d<0).mean(),
                            probability_superiority_with_ties=(d.gt(0).mean()+.5*d.eq(0).mean()),low95=lo,high95=hi,p_value=p))
                        transitions.append(dict(model=model,adapter=adapter,contrast="difference_in_differences_sft_survival",population="evaluation_285",dimension=score,n=len(joined),
                            improved=int((d>0).sum()),unchanged=int((d==0).sum()),worsened=int((d<0).sum()),rescued=np.nan,degraded=np.nan))
        contrast=pd.DataFrame(results)
        if len(contrast):
            contrast["p_adjusted_bh"] = contrast.groupby(["dimension","contrast"])["p_value"].transform(lambda x: bh_adjust(x))
        self._write(contrast,"treatment_contrasts.csv")
        self._write(contrast[contrast.contrast.eq("difference_in_differences_sft_survival")].copy(),"difference_in_differences.csv")
        transition=self._write(pd.DataFrame(transitions),"treatment_transition_counts.csv")
        self._write(pd.DataFrame(diagnostics),"common_support_diagnostics.csv")
        profiles=contrast[contrast.contrast.isin(["bf16_sft_minus_untreated","one_bit_sft_minus_untreated"])].copy() if len(contrast) else pd.DataFrame()
        self._write(profiles,"adapter_specificity_profiles.csv")
        self._contrast_figures(contrast, transition)

    def _contrast_figures(self, contrast, transition):
        if contrast.empty: return
        contrast=contrast[contrast.population.eq("evaluation_285")]
        transition=transition[transition.population.eq("evaluation_285")]
        for score in SCORE_COLUMNS:
            short=SHORT_SCORE[score]
            families=[("03A","bf16_sft_minus_untreated"),("03B","one_bit_sft_minus_untreated")]
            for prefix,cname in families:
                sub=contrast[(contrast.dimension==score)&(contrast.contrast==cname)]
                fig,axes=plt.subplots(2,3,figsize=(15,9),sharex=True,sharey=True)
                for ax,model in zip(axes.ravel(),MODELS):
                    m=sub[sub.model==model].set_index("adapter").reindex(CONDITIONS[1:])
                    ax.errorbar(m.mean_difference,range(4),xerr=[m.mean_difference-m.low95,m.high95-m.mean_difference],fmt="none",ecolor="#555")
                    for y,(adapter,row) in enumerate(m.iterrows()):
                        ax.scatter(row.mean_difference,y,color=CONDITION_COLORS[adapter],marker=CONDITION_MARKERS[adapter],s=55)
                    ax.axvline(0,color="black",lw=.8);ax.set_xlim(-4,4);ax.set_yticks(range(4),CONDITIONS[1:]);ax.set_title(MODEL_LABELS[model])
                fig.suptitle(f"{short.replace('_',' ').title()} — {cname.replace('_',' ')}")
                self._figure(self.p.figures/f"{prefix}_{short}_{'bf16' if prefix=='03A' else 'one_bit'}_sft_effects.png",fig)
            deployment=contrast[(contrast.dimension==score)&contrast.contrast.isin(["untreated_one_bit_minus_bf16","sft_one_bit_minus_bf16"])]
            deployment=deployment.copy()
            deployment["contrast_display"]=deployment.contrast.map({
                "untreated_one_bit_minus_bf16":"Untreated: one-bit − BF16",
                "sft_one_bit_minus_bf16":"SFT: one-bit − BF16",
            }).fillna(deployment.contrast)
            fig,axes=plt.subplots(2,3,figsize=(15,9),sharex=True,sharey=True)
            for ax,model in zip(axes.ravel(),MODELS):
                m=deployment[deployment.model==model]
                sns.pointplot(m,x="mean_difference",y="contrast_display",hue="adapter",palette=CONDITION_COLORS,ax=ax,errorbar=None)
                ax.axvline(0,color="black",lw=.8);ax.set_xlim(-4,4);ax.set_title(MODEL_LABELS[model]);ax.set_xlabel("Mean judge-score difference");ax.set_ylabel("Deployment comparison");ax.get_legend().remove() if ax.get_legend() else None
            fig.suptitle(f"{short.replace('_',' ').title()} — deployment-stack contrasts")
            self._figure(self.p.figures/f"03C_{short}_deployment_stack_effects.png",fig,data=deployment,
                         key="Rows are fixed evaluation_285 paired deployment-stack contrasts. The full all_990 untreated sensitivity estimate remains in treatment_contrasts.csv.",
                         sources="treatment_contrasts.csv")
            for prefix,cname in [("04A","bf16_sft_minus_untreated"),("04B","one_bit_sft_minus_untreated")]:
                sub=transition[(transition.dimension==score)&(transition.contrast==cname)&(transition.adapter=="full")]
                fig,axes=plt.subplots(2,3,figsize=(14,7),sharex=True)
                for ax,model in zip(axes.ravel(),MODELS):
                    row=sub[sub.model==model]
                    if len(row):
                        r=row.iloc[0]; total=max(r.n,1); left=0
                        for name in ("improved","unchanged","worsened"):
                            value=r[name]/total; ax.barh([0],[value],left=left,color=TRANSITION_COLORS[name],label=name);left+=value
                    ax.set_xlim(0,1);ax.set_yticks([]);ax.set_title(MODEL_LABELS[model])
                fig.suptitle(f"Full-SFT row transitions — {short.replace('_',' ')}")
                self._figure(self.p.figures/f"{prefix}_{short}_{'bf16' if prefix=='04A' else 'one_bit'}_treatment_transitions.png",fig)
        profiles=contrast[contrast.contrast.isin(["bf16_sft_minus_untreated","one_bit_sft_minus_untreated"]) & contrast.dimension.isin(SUBSTANTIVE_SCORES)]
        for model in MODELS:
            fig,axes=plt.subplots(1,2,figsize=(14,5),sharex=True,sharey=True)
            for ax,(cname,title) in zip(axes,[("bf16_sft_minus_untreated","BF16 HF"),("one_bit_sft_minus_untreated","one-bit frontier")]):
                m=profiles[(profiles.model==model)&(profiles.contrast==cname)]
                for adapter in CONDITIONS[1:]:
                    a=m[m.adapter==adapter].set_index("dimension").reindex(SUBSTANTIVE_SCORES)
                    ax.plot(a.mean_difference,range(len(SUBSTANTIVE_SCORES)),marker=CONDITION_MARKERS[adapter],color=CONDITION_COLORS[adapter],label=adapter)
                ax.axvline(0,color="black",lw=.8);ax.set_xlim(-4,4);ax.set_title(title);ax.set_yticks(range(len(SUBSTANTIVE_SCORES)),[SHORT_SCORE[x].replace("_"," ") for x in SUBSTANTIVE_SCORES])
            axes[1].legend(loc="best");fig.suptitle(f"Adapter specificity — {MODEL_LABELS[model]}")
            self._figure(self.p.figures/f"04C_{model}_adapter_specificity_profiles.png",fig)

    # Further sections are implemented as compact, auditable analyses below.
    def attack_and_failures(self) -> None:
        truth=self.truth.copy(); truth["techniques"]=truth.ground_truth.map(canonical_list); truth["label_count"]=truth.techniques.map(len)
        truth=truth.merge(self.splits[[c for c in self.splits.columns if c in {"prompt_id","split","group_id"}]],on="prompt_id",how="left")
        memberships=truth.explode("techniques"); memberships["parent_technique"]=memberships.techniques.str.extract(r"^(T\d{4})")
        dist=(memberships.groupby(["split","techniques"],dropna=False).agg(rows=("prompt_id","nunique"),memberships=("prompt_id","size")).reset_index())
        source_rows=[]
        populations={"all_990":truth,"evaluation_285":truth[truth.split.isin(["validation","test"])]}
        for population,pop in populations.items():
            denom=max(pop.prompt_id.nunique(),1)
            for level,column in [("technique","techniques"),("parent_technique","parent_technique"),("scenario","scenario"),("scenario_step","group_id")]:
                expanded=(pop.explode("techniques") if level in {"technique","parent_technique"} else pop.copy())
                if level=="parent_technique":expanded["parent_technique"]=expanded.techniques.str.extract(r"^(T\d{4})")
                for value,g in expanded.groupby(column,dropna=False):source_rows.append(dict(population=population,class_level=level,class_value=value,row_count=g.prompt_id.nunique(),row_percentage=100*g.prompt_id.nunique()/denom,expanded_memberships=len(g)))
        source_dist=self._write(pd.DataFrame(source_rows),"source_attack_distribution.csv")
        # No pinned STIX mapping is bundled; preserve a transparent not-applicable tactic table.
        tactic=pd.DataFrame([{"status":"not_applicable","reason":"No pinned MITRE ATT&CK STIX mapping was bundled with the frozen inputs; technique and parent-technique analyses remain available."}])
        self._write(tactic,"tactic_distribution.csv")
        scores=self.long.merge(memberships[["prompt_id","techniques","parent_technique","label_count"]],on="prompt_id",how="left")
        class_rows=[]
        for population,pop_scores in self._population_views(scores):
            melted_scores=pop_scores.melt(id_vars=["prompt_id","group_id","model","condition_short","representation_label","techniques","parent_technique","label_count"],value_vars=[x+"_numeric" for x in SCORE_COLUMNS],var_name="dimension",value_name="score")
            for (model,cond,rep,dim,tech),g in melted_scores.groupby(["model","condition_short","representation_label","dimension","techniques"],dropna=False):
                x=g.score.dropna()
                lo,hi=(cluster_bootstrap_mean_fast(g.score,g.group_id,self.bootstrap_replicates) if len(x)>=MIN_CLASS_N else (np.nan,np.nan))
                class_rows.append(dict(population=population,model=model,condition=cond,representation=rep,dimension=dim.removesuffix("_numeric"),technique=tech,n=len(x),mean=x.mean(),low95=lo,high95=hi))
        classdf=self._write(pd.DataFrame(class_rows),"attack_class_scores.csv")
        treatment_rows=[]
        for model in MODELS:
            for rep in ("BF16 HF","one-bit frontier"):
                untreated=scores[(scores.model.eq(model))&(scores.condition_short.eq("untreated"))&(scores.representation_group.eq(rep))]
                for adapter in CONDITIONS[1:]:
                    treated=scores[(scores.model.eq(model))&(scores.condition_short.eq(adapter))&(scores.representation_group.eq(rep))]
                    for score in SCORE_COLUMNS:
                        left=treated[["prompt_id","group_id","techniques",score+"_numeric"]].rename(columns={score+"_numeric":"treated"})
                        right=untreated[["prompt_id","techniques",score+"_numeric"]].rename(columns={score+"_numeric":"untreated"})
                        pair=left.merge(right,on=["prompt_id","techniques"]).dropna(subset=["treated","untreated"])
                        for technique,g in pair.groupby("techniques",dropna=False):
                            d=g.treated-g.untreated
                            lo,hi=(cluster_bootstrap_mean_fast(d,g.group_id,self.bootstrap_replicates) if len(g)>=MIN_CLASS_N else (np.nan,np.nan))
                            treatment_rows.append(dict(model=model,adapter=adapter,representation_group=rep,dimension=score,
                                technique=technique,common_support_n=len(g),independent_groups=g.group_id.nunique(),
                                sufficiently_represented=bool(len(g)>=MIN_CLASS_N),mean_difference=d.mean(),median_difference=d.median(),
                                improved_rate=d.gt(0).mean(),unchanged_rate=d.eq(0).mean(),worsened_rate=d.lt(0).mean(),low95=lo,high95=hi))
        treatment_effects=self._write(pd.DataFrame(treatment_rows),"attack_class_treatment_effects.csv")
        macro=[]
        for population,pop_long in self._population_views(self.long):
            pop_scores=scores if population=="all_990" else scores[scores.prompt_id.astype(str).isin(self.evaluation_ids)]
            melted=pop_long.melt(id_vars=["model","condition_short","representation_label","prompt_id"],value_vars=[s+"_numeric" for s in SCORE_COLUMNS],var_name="dimension",value_name="score")
            for keys,g in melted.groupby(["model","condition_short","representation_label","dimension"]):
                classes=classdf[(classdf.population==population)&(classdf.model==keys[0])&(classdf.condition==keys[1])&(classdf.representation==keys[2])&(classdf.dimension==keys[3].removesuffix("_numeric"))&(classdf.n>=MIN_CLASS_N)]
                expanded_scores=pop_scores[(pop_scores.model==keys[0])&(pop_scores.condition_short==keys[1])&(pop_scores.representation_label==keys[2])][keys[3]].dropna()
                macro.append(dict(population=population,model=keys[0],condition=keys[1],representation=keys[2],dimension=keys[3].removesuffix("_numeric"),row_weighted_mean=g.score.mean(),class_macro_mean=classes["mean"].mean(),membership_micro_mean=expanded_scores.mean(),eligible_classes=len(classes),row_n=g.score.notna().sum(),membership_n=len(expanded_scores)))
        self._write(pd.DataFrame(macro),"attack_class_macro_micro.csv")
        mult=[]
        merged=self.long.merge(truth[["prompt_id","label_count"]],on="prompt_id",how="left")
        for (m,c,r,dim,nlab),g in merged.melt(id_vars=["model","condition_short","representation_label","prompt_id","group_id","label_count"],value_vars=[x+"_numeric" for x in SCORE_COLUMNS],var_name="dimension",value_name="score").groupby(["model","condition_short","representation_label","dimension","label_count"],dropna=False):
            mult.append(dict(model=m,condition=c,representation=r,dimension=dim.removesuffix("_numeric"),label_count=nlab,n=g.score.notna().sum(),mean=g.score.mean()))
        self._write(pd.DataFrame(mult),"ground_truth_multiplicity.csv")
        self._attack_figures(dist,classdf,merged,treatment_effects)
        self._failure_analysis()

    def _attack_figures(self, dist, classdf, merged, treatment_effects):
        classdf=classdf[classdf.population.eq("evaluation_285")]
        empty_figure(self.p.figures/"05_tactic_distribution_990_vs_285.png","Tactic distribution","Not generated: a pinned ATT&CK STIX tactic mapping is not present in the frozen source inputs. See tactic_distribution.csv."); self.generated.append(self.p.figures/"05_tactic_distribution_990_vs_285.png")
        for score in SCORE_COLUMNS:
            short=SHORT_SCORE[score]; sub=classdf[classdf.dimension==score]
            top=(sub.groupby("technique").n.sum().nlargest(15).index if len(sub) else [])
            heat=sub[(sub.technique.isin(top))&(sub.condition.isin(["untreated","full"]))].pivot_table(index="technique",columns=["model","condition"],values="mean")
            fig,ax=plt.subplots(figsize=(14,8)); sns.heatmap(heat,vmin=0,vmax=4,cmap="RdBu",center=2,ax=ax);ax.set_title(f"Technique-associated {short.replace('_',' ')}")
            self._figure(self.p.figures/f"06_{short}_technique_heatmap.png",fig)
            effects=treatment_effects[(treatment_effects.dimension.eq(score))&(treatment_effects.adapter.eq("full"))&treatment_effects.sufficiently_represented]
            effect_heat=effects.pivot_table(index="technique",columns=["representation_group","model"],values="mean_difference")
            if len(effect_heat):effect_heat=effect_heat.loc[effect_heat.abs().max(axis=1).nlargest(20).index]
            fig,ax=plt.subplots(figsize=(15,max(6,.32*len(effect_heat))));sns.heatmap(effect_heat,vmin=-4,vmax=4,cmap="vlag",center=0,ax=ax)
            ax.set_title(f"Paired full-SFT effects by sufficiently represented technique — {short}")
            self._figure(self.p.figures/f"06B_{short}_technique_sft_effects.png",fig,data=effects,
                         key=f"Cells are paired SFT-full minus untreated judge-score differences. Classes require at least {MIN_CLASS_N} matched rows; columns separate BF16 and one-bit frontier streams.",sources="attack_class_treatment_effects.csv")
            empty_figure(self.p.figures/f"06C_{short}_row_macro_comparison.png",f"Row versus class-macro — {short}","Companion class summaries are provided in attack_class_performance.csv.");self.generated.append(self.p.figures/f"06C_{short}_row_macro_comparison.png")
            bottom=sub[sub.technique.isin(top)].groupby("technique").mean(numeric_only=True).nsmallest(12,"mean").index if len(sub) else []
            plot=sub[sub.technique.isin(bottom)&sub.condition.isin(["untreated","full"])]
            fig,ax=plt.subplots(figsize=(12,7)); sns.pointplot(plot,x="mean",y="technique",hue="model",palette=MODEL_COLORS,errorbar=None,ax=ax);ax.set_xlim(0,4);ax.set_title(f"Lowest-scoring represented techniques — {short}")
            self._figure(self.p.figures/f"07_{short}_lowest_techniques.png",fig)
            fig,ax=plt.subplots(figsize=(16,10)); sns.heatmap(sub.pivot_table(index="technique",columns=["model","condition"],values="mean"),vmin=0,vmax=4,cmap="RdBu",center=2,ax=ax);ax.set_title(f"All eligible techniques/adapters — {short}")
            self._figure(self.p.figures/f"S06_{short}_all_techniques_and_adapters.png",fig)
            empty_figure(self.p.figures/f"S07_{short}_conditional_adapter_lowest_techniques.png",f"Conditional-adapter lowest techniques — {short}","See attack_class_performance.csv; displayed only when sufficiently represented class/adapter cells exist.");self.generated.append(self.p.figures/f"S07_{short}_conditional_adapter_lowest_techniques.png")
        fig,ax=plt.subplots(figsize=(10,5)); temp=merged.groupby("label_count")[SUBSTANTIVE_SCORES[0]+"_numeric"].mean();ax.plot(temp.index,temp.values,marker="o");ax.set(title="Technique score by ground-truth multiplicity",xlabel="Ground-truth technique count",ylabel="Mean score",ylim=(0,4));self._figure(self.p.figures/"07B_ground_truth_multiplicity.png",fig)

    def _failure_analysis(self):
        dictionary={};row_records=[];coherence=[]
        for idx,r in self.long.iterrows():
            mapped=[]
            for label in canonical_list(r.get("judge_failure_labels","")):
                category,rule=broad_failure_category(label);dictionary[label]=(category,rule);mapped.append(category)
            categories=sorted(set(mapped))
            identity={c:r.get(c,"") for c in self._group_cols()}
            row_records.append(identity|{"row_index":idx,"prompt_id":r.prompt_id,"group_id":r.group_id,
                                         "failure_categories":categories,"failure_category_count":len(categories)})
            for category in categories:
                for score in SCORE_COLUMNS:
                    coherence.append(identity|{"prompt_id":r.prompt_id,"group_id":r.group_id,"failure_category":category,"dimension":score,"score":r.get(score+"_numeric")})
        dictionary_df=pd.DataFrame([{"failure_label_raw":label,"failure_category":value[0],"mapping_rule":value[1]}
                                    for label,value in sorted(dictionary.items())])
        self._write(dictionary_df,"failure_label_dictionary.csv")
        rowdf=pd.DataFrame(row_records);categories=sorted(set(dictionary_df.failure_category)) if len(dictionary_df) else []
        for category in categories:rowdf[category]=rowdf.failure_categories.map(lambda x:category in x)
        rate_rows=[]
        for population,pop_rows in self._population_views(rowdf):
            for keys,g in pop_rows.groupby(self._group_cols(),dropna=False):
                for category in categories:
                    count=int(g[category].sum());rate_rows.append(dict(zip(self._group_cols(),keys),population=population,failure_category=category,
                        count=count,denominator=len(g),rate_per_100=100*count/len(g) if len(g) else np.nan))
        failure=self._write(pd.DataFrame(rate_rows),"failure_rates.csv")

        lookup={(m,c,rep):g for (m,c,rep),g in rowdf.groupby(["model","condition_short","representation_group"])}
        risk=[]
        for model in MODELS:
            for rep in ("BF16 HF","one-bit frontier"):
                untreated=lookup.get((model,"untreated",rep))
                if untreated is None:continue
                for adapter in CONDITIONS[1:]:
                    treated=lookup.get((model,adapter,rep))
                    if treated is None:continue
                    for category in categories:
                        pair=treated[["prompt_id","group_id",category]].merge(untreated[["prompt_id",category]],on="prompt_id",suffixes=("_treated","_untreated"))
                        d=pair[category+"_treated"].astype(float)-pair[category+"_untreated"].astype(float)
                        lo,hi=cluster_bootstrap_mean_fast(d,pair.group_id,self.bootstrap_replicates)
                        risk.append(dict(model=model,adapter=adapter,representation_group=rep,population="evaluation_285",failure_category=category,
                            common_support_n=len(pair),independent_groups=pair.group_id.nunique(),treated_risk=pair[category+"_treated"].mean(),
                            untreated_risk=pair[category+"_untreated"].mean(),risk_difference=d.mean(),low95=lo,high95=hi))
        self._write(pd.DataFrame(risk),"failure_risk_differences.csv")
        coherence_df=pd.DataFrame(coherence)
        coh=[]
        for population,pop in self._population_views(coherence_df):
            if len(pop):
                agg=pop.groupby([*self._group_cols(),"failure_category","dimension"],dropna=False).score.agg(n="count",mean="mean",median="median").reset_index();agg["population"]=population;coh.append(agg)
        coherence_df=pd.concat(coh,ignore_index=True) if coh else pd.DataFrame()
        self._write(coherence_df,"score_failure_label_coherence.csv")

        similarity=[]
        for (population,rep,condition),g in failure.groupby(["population","representation_group","condition_short"]):
            pivot=g.pivot_table(index="model",columns="failure_category",values="rate_per_100",fill_value=0).reindex(MODELS)
            for i,a in enumerate(MODELS):
                for b in MODELS[i+1:]:
                    if a not in pivot.index or b not in pivot.index:continue
                    x=pivot.loc[a].to_numpy(float);y=pivot.loc[b].to_numpy(float)
                    sx=x.sum();sy=y.sum()
                    if sx<=0 or sy<=0:
                        cosine=js=np.nan;status="not_estimable_empty_profile"
                    else:
                        cosine=float(np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y))) if np.linalg.norm(x)*np.linalg.norm(y)>0 else np.nan
                        js=float(1-jensenshannon(x/sx,y/sy,base=2));status="estimated"
                    similarity.append(dict(population=population,representation_group=rep,condition=condition,model_a=a,model_b=b,
                        failure_categories=len(categories),cosine_similarity=cosine,jensen_shannon_similarity=js,estimability_status=status))
        similarity_df=self._write(pd.DataFrame(similarity),"failure_profile_similarity.csv")
        cooccurrence=(rowdf.failure_categories.map(lambda x:" + ".join(x) if x else "no_failure_label").value_counts().rename_axis("failure_category_combination").reset_index(name="rows"))
        self._write(cooccurrence,"failure_category_cooccurrence.csv")

        shown=failure[failure.population.eq("evaluation_285")]
        fig,axes=plt.subplots(2,3,figsize=(16,9),sharex=True)
        for ax,model in zip(axes.ravel(),MODELS):
            m=shown[shown.model.eq(model)].groupby("failure_category").rate_per_100.mean().nlargest(8).sort_values()
            ax.barh(m.index,m.values,color=MODEL_COLORS[model]);ax.set_title(MODEL_LABELS[model]);ax.set_xlabel("per 100 judged rows")
        fig.suptitle("Broad judge failure-category rates");self._figure(self.p.figures/"09_failure_rate_bars.png",fig,data=failure,sources="failure_rates.csv; failure_label_dictionary.csv")
        fig,axes=plt.subplots(2,2,figsize=(13,11))
        for ax,(rep,condition) in zip(axes.ravel(),[("BF16 HF","untreated"),("BF16 HF","full"),("one-bit frontier","untreated"),("one-bit frontier","full")]):
            sub=similarity_df[(similarity_df.population.eq("evaluation_285"))&(similarity_df.representation_group.eq(rep))&(similarity_df.condition.eq(condition))]
            matrix=pd.DataFrame(np.eye(len(MODELS)),index=MODELS,columns=MODELS,dtype=float)
            for _,r in sub.iterrows():matrix.loc[r.model_a,r.model_b]=matrix.loc[r.model_b,r.model_a]=r.jensen_shannon_similarity
            sns.heatmap(matrix,vmin=0,vmax=1,cmap="viridis",ax=ax,cbar=ax is axes.ravel()[-1]);ax.set_title(f"{rep} | {condition}")
        fig.suptitle("Similarity of broad failure profiles between model families")
        self._figure(self.p.figures/"10_failure_profile_similarity.png",fig,data=similarity_df,
                     key="Cells are Jensen-Shannon similarities between broad failure-category rate profiles; 1 means identical proportions and 0 means maximally different profiles.",sources="failure_profile_similarity.csv")

    def related_dimensions(self):
        pairs=[(SUBSTANTIVE_SCORES[0],SUBSTANTIVE_SCORES[1]),(SUBSTANTIVE_SCORES[0],SUBSTANTIVE_SCORES[2]),(SUBSTANTIVE_SCORES[1],SUBSTANTIVE_SCORES[2]),(SUBSTANTIVE_SCORES[4],SCORE_COLUMNS[6])]
        rows=[]
        for a,b in pairs:
            for population,data in self._population_views(self.long):
                x=data[["prompt_id","model","condition_short","representation_label","group_id",a+"_numeric",b+"_numeric"]].dropna()
                for keys,g in x.groupby(["model","condition_short","representation_label"]):
                    rho,p=spearman_safe(g[a+"_numeric"],g[b+"_numeric"])
                    rows.append(dict(population=population,model=keys[0],condition=keys[1],representation=keys[2],dimension_a=a,dimension_b=b,n=len(g),spearman_rho=rho,p_value=p,mean_difference=(g[a+"_numeric"]-g[b+"_numeric"]).mean()))
            x=self.long[self.long.prompt_id.astype(str).isin(self.evaluation_ids)][["model","condition_short","representation_label","group_id",a+"_numeric",b+"_numeric"]].dropna()
            grid=pd.crosstab(x[a+"_numeric"],x[b+"_numeric"]).reindex(index=range(5),columns=range(5),fill_value=0)
            fig,ax=plt.subplots(figsize=(6,5));sns.heatmap(grid,annot=True,fmt="g",cmap="Blues",ax=ax);ax.set(xlabel=SHORT_SCORE[b],ylabel=SHORT_SCORE[a],title=f"Joint scores: {SHORT_SCORE[a]} vs {SHORT_SCORE[b]}")
            self._figure(self.p.figures/f"08_{SHORT_SCORE[a]}_vs_{SHORT_SCORE[b]}_joint_score_grid.png",fig)
        self._write(pd.DataFrame(rows),"judge_dimension_relationships.csv")
        confidence=[]
        for population,data in self._population_views(self.long):
            for keys,g in data.groupby(self._group_cols(),dropna=False):
                candidate=g.get("candidate_confidence_label",pd.Series("",index=g.index)).astype(str).str.strip().ne("")
                warranted=g.get("judge_warranted_confidence_label",pd.Series("",index=g.index)).astype(str).str.strip().ne("")
                agreement=g["confidence_label_agreement_numeric"].notna()
                confidence.append(dict(zip(self._group_cols(),keys),population=population,judged_rows=len(g),candidate_label_available=int(candidate.sum()),warranted_label_available=int(warranted.sum()),scalar_agreement_available=int(agreement.sum()),all_three_available=int((candidate&warranted&agreement).sum())))
        self._write(pd.DataFrame(confidence),"confidence_field_availability.csv")

    def repeat_judge_longtail_size(self):
        # Lexical similarity is permitted only as a secondary appendix; it never recovers semantic scores.
        data=self.long.copy(); texts=data.get("candidate_response_raw",pd.Series("",index=data.index)).fillna("").astype(str)
        ref=data.get("haiku_reference_response_raw",pd.Series("",index=data.index)).fillna("").astype(str)
        teacher=data.get("teacher_response_raw",pd.Series("",index=data.index)).fillna("").astype(str)
        sims_h=np.full(len(data),np.nan);sims_t=np.full(len(data),np.nan)
        nonempty=(texts.str.len()>0)&(ref.str.len()>0)
        if nonempty.any():
            corpus=pd.concat([texts[nonempty],ref[nonempty],teacher[nonempty]],ignore_index=True)
            matrix=TfidfVectorizer(min_df=2,max_features=20000,ngram_range=(1,2)).fit_transform(corpus)
            n=nonempty.sum(); a=matrix[:n];b=matrix[n:2*n];c=matrix[2*n:]
            sims_h[nonempty.to_numpy()]=np.asarray(a.multiply(b).sum(axis=1)).ravel()
            sims_t[nonempty.to_numpy()]=np.asarray(a.multiply(c).sum(axis=1)).ravel()
        data["tfidf_candidate_haiku"]=sims_h;data["tfidf_candidate_teacher"]=sims_t
        repeat=[]
        for population,pop in self._population_views(data):
            for keys,g in pop.groupby(["model","condition_short","representation_label","prompt_sha256"]):
                if len(g)<2:continue
                repeat.append(dict(population=population,model=keys[0],condition=keys[1],representation=keys[2],prompt_sha256=keys[3],n=len(g),exact_response_hashes=g.candidate_response_sha256.nunique() if "candidate_response_sha256" in g else np.nan,output_token_range=g.candidate_output_tokens.max()-g.candidate_output_tokens.min(),**{f"{SHORT_SCORE[s]}_range":g[s+"_numeric"].max()-g[s+"_numeric"].min() for s in SCORE_COLUMNS}))
        repeatdf=self._write(pd.DataFrame(repeat),"repeated_prompt_consistency.csv")
        variant=[]
        for population,pop in self._population_views(data):
            for keys,g in pop.groupby(["model","condition_short","representation_label","group_id"]):
                if len(g)<2:continue
                variant.append(dict(population=population,model=keys[0],condition=keys[1],representation=keys[2],group_id=keys[3],variants=g.prompt_id.nunique(),output_token_range=g.candidate_output_tokens.max()-g.candidate_output_tokens.min(),**{f"{SHORT_SCORE[s]}_range":g[s+"_numeric"].max()-g[s+"_numeric"].min() for s in SCORE_COLUMNS}))
        vardf=self._write(pd.DataFrame(variant),"scenario_step_variant_sensitivity.csv")
        self._judge_audits(data)
        self._longtail(data)
        self._size_concordance(data)
        self._repeat_figures(repeatdf,vardf,data)

    def _judge_audits(self,data):
        self._write(pd.DataFrame([{"status":"not_applicable","reason":"No independent rubric-implied scalar is exported for every score; notebook 10 does not reparse raw judge text."}]),"judge_output_consistency_audit.csv")
        conf=data[[c for c in ["model","condition_short","representation_label","prompt_id","candidate_confidence_label","judge_warranted_confidence_label","confidence_label_agreement_numeric"] if c in data]].copy()
        self._write(conf,"judge_confidence_rule_audit.csv")
        fpcols=[c for c in ["prompt_sha256","ground_truth","candidate_response_sha256","haiku_reference_response_sha256","judge_protocol"] if c in data]
        data["case_fingerprint"]=data[fpcols].astype(str).agg("|".join,axis=1).map(lambda x:hashlib.sha256(x.encode()).hexdigest())
        dup=[]
        for fp,g in data.groupby("case_fingerprint"):
            if len(g)<2:continue
            dup.append(dict(case_fingerprint=fp,n=len(g),score_vector_exact_agreement=int(g[[s+"_numeric" for s in SCORE_COLUMNS]].drop_duplicates().shape[0]==1),maximum_dimension_range=max((g[s+"_numeric"].max()-g[s+"_numeric"].min() for s in SCORE_COLUMNS),default=np.nan)))
        self._write(pd.DataFrame(dup),"judge_duplicate_case_consistency.csv")
        anchor=[]
        for population,pop in self._population_views(data):
            for keys,g in pop.groupby(["model","condition_short","representation_label"]):
                for score in SUBSTANTIVE_SCORES:
                    a=g[[score+"_numeric","tfidf_candidate_haiku","tfidf_candidate_teacher"]].dropna()
                    rho_h,_=spearman_safe(a.iloc[:,0],a.iloc[:,1]);rho_t,_=spearman_safe(a.iloc[:,0],a.iloc[:,2])
                    anchor.append(dict(population=population,model=keys[0],condition=keys[1],representation=keys[2],dimension=score,n=len(a),rho_haiku=rho_h,rho_teacher=rho_t))
        self._write(pd.DataFrame(anchor),"judge_reference_anchor_diagnostics.csv")
        # Haiku judged its own independently generated reference responses in a
        # separate API pass.  These are reported as a circular descriptive
        # baseline and never treated as validation of the judge.
        self_rows=[]
        if not self.self_judgments.empty:
            sj=self.self_judgments.copy()
            for score in SCORE_COLUMNS:
                # SCORE_COLUMNS already contains the canonical `score_`
                # prefix for six rubric fields; agreement is the sole
                # unprefixed canonical field.
                source=score
                sj[score+"_numeric"]=sj.get(source,pd.Series(index=sj.index,dtype=object)).map(safe_float)
            for score in SCORE_COLUMNS:
                values=sj[score+"_numeric"].dropna()
                self_rows.append(dict(
                    dimension=score,
                    dimension_short=SHORT_SCORE[score],
                    intended_rows=len(sj),
                    scored_rows=len(values),
                    score_availability_rate=len(values)/len(sj) if len(sj) else np.nan,
                    mean=values.mean(),
                    median=values.median(),
                    q1=values.quantile(.25),
                    q3=values.quantile(.75),
                    pct_3_or_4=values.ge(3).mean() if len(values) else np.nan,
                    **{f"score_{i}_n":int(values.eq(i).sum()) for i in range(5)},
                ))
        self._write(pd.DataFrame(self_rows),"haiku_self_evaluation_summary.csv")
        # Frozen algorithm: extremes from canonical score/format fields.
        qual=[]
        for score in SUBSTANTIVE_SCORES:
            temp=data.sort_values(score+"_numeric").dropna(subset=[score+"_numeric"])
            for label,subset in [("lowest",temp.head(3)),("highest",temp.tail(3))]:
                for _,r in subset.iterrows():qual.append({"selection_reason":f"{label}_{SHORT_SCORE[score]}","prompt_id":r.prompt_id,"model":r.model,"condition":r.condition_short,"representation":r.representation_label,"score":r[score+"_numeric"],"candidate_response_sha256":r.get("candidate_response_sha256","")})
        self._write(pd.DataFrame(qual).drop_duplicates(),"qualitative_audit_cases.csv")

    def _longtail(self,data):
        rows=[]
        for population,pop in self._population_views(data):
            for keys,g in pop.groupby(["model","condition_short","representation_label","prompt_length_bucket"],observed=True):
                rows.append(dict(population=population,model=keys[0],condition=keys[1],representation=keys[2],prompt_length_bucket=keys[3],n=len(g),completed_rate=g.candidate_completed.mean(),median_output_tokens=g.candidate_output_tokens.median(),median_elapsed_seconds=g.candidate_elapsed_seconds.median(),p95_elapsed_seconds=g.candidate_elapsed_seconds.quantile(.95),**{f"mean_{SHORT_SCORE[s]}":g[s+"_numeric"].mean() for s in SCORE_COLUMNS}))
        long=self._write(pd.DataFrame(rows),"prompt_length_and_long_tail.csv")
        classperf=pd.read_csv(self.p.tables/"attack_class_scores.csv") if (self.p.tables/"attack_class_scores.csv").exists() else pd.DataFrame()
        rare=classperf.assign(interval_width=classperf.high95-classperf.low95) if len(classperf) else classperf
        self._write(rare,"rare_attack_class_uncertainty.csv")
        fig,ax=plt.subplots(figsize=(10,5))
        shown=data[data.prompt_id.astype(str).isin(self.evaluation_ids)]
        for model in MODELS:
            vals=shown[shown.model==model].candidate_input_tokens.dropna().sort_values();ax.plot(vals,np.arange(1,len(vals)+1)/max(len(vals),1),label=MODEL_LABELS[model],color=MODEL_COLORS[model])
        ax.set(xscale="log",xlabel="Input tokens",ylabel="ECDF",title="Prompt-length distribution");ax.legend(ncol=3);self._figure(self.p.figures/"18A_prompt_length_ecdf.png",fig)
        for name,title,y in [("18B_completion_by_prompt_length.png","Completion by prompt length","completed_rate"),("18C_scores_by_prompt_length.png","Technique score by prompt length","mean_technique_identification"),("18D_latency_by_prompt_length.png","Latency by prompt length","median_elapsed_seconds")]:
            fig,ax=plt.subplots(figsize=(11,5));sns.pointplot(long[long.population.eq("evaluation_285")],x="prompt_length_bucket",y=y,hue="model",palette=MODEL_COLORS,errorbar=None,ax=ax);ax.set_title(title);self._figure(self.p.figures/name,fig)
        fig,ax=plt.subplots(figsize=(9,5));
        rare_shown=rare[rare.population.eq("evaluation_285")] if len(rare) and "population" in rare else rare
        if len(rare_shown):ax.scatter(rare_shown.n,rare_shown.interval_width,alpha=.5);ax.set(xscale="log",xlabel="Class n",ylabel="95% interval width",title="Rare-class frequency and uncertainty")
        self._figure(self.p.figures/"18E_rare_class_frequency_uncertainty.png",fig)

    def _size_concordance(self,data):
        rows=[]
        for population,pop in self._population_views(data):
            melted=pop.melt(id_vars=["model","condition_short","representation_group","prompt_id"],value_vars=[s+"_numeric" for s in SCORE_COLUMNS],var_name="dimension",value_name="score").dropna(subset=["score"])
            for (cond,rep,score),g in melted.groupby(["condition_short","representation_group","dimension"]):
                pivot=g.pivot_table(index="prompt_id",columns="model",values="score")
                for a in MODELS:
                    for b in MODELS:
                        if a not in pivot or b not in pivot:
                            pair=pd.DataFrame();rho=jaccard=np.nan;spearman_status=overlap_status="not_estimable_missing_model"
                        elif a == b:
                            pair=pivot[[a]].dropna();rho=1.0 if len(pair) else np.nan;jaccard=1.0 if (len(pair) and (pair[a]<=1).any()) else np.nan
                            low_rate_a=low_rate_b=float((pair[a]<=1).mean()) if len(pair) else np.nan
                            spearman_status=overlap_status="self_comparison"
                        else:
                            pair=pivot[[a,b]].dropna()
                            if len(pair)<2 or pair[a].nunique()<2 or pair[b].nunique()<2:
                                rho=np.nan;spearman_status="not_estimable_constant_or_insufficient_scores"
                            else:
                                rho,_=spearman_safe(pair[a],pair[b]);spearman_status="estimated"
                            low_a=set(pair.index[pair[a]<=1]);low_b=set(pair.index[pair[b]<=1])
                            low_rate_a=len(low_a)/len(pair) if len(pair) else np.nan;low_rate_b=len(low_b)/len(pair) if len(pair) else np.nan
                            if len(pair) and min(low_rate_a,low_rate_b)>=FLOOR_SATURATION_RATE:
                                jaccard=np.nan;overlap_status="not_estimable_floor_saturation_ge_95pct"
                            elif low_a|low_b:
                                jaccard=len(low_a&low_b)/len(low_a|low_b);overlap_status="estimated"
                            else:
                                jaccard=np.nan;overlap_status="not_estimable_no_low_score_rows"
                        if a not in pivot or b not in pivot:
                            low_rate_a=low_rate_b=np.nan
                        rows.append(dict(population=population,condition=cond,representation=rep,dimension=score.removesuffix("_numeric"),model_a=a,model_b=b,n=len(pair),
                            spearman=rho,spearman_estimability=spearman_status,low_score_jaccard=jaccard,low_score_overlap_estimability=overlap_status,
                            low_score_rate_model_a=low_rate_a,low_score_rate_model_b=low_rate_b,floor_saturation_threshold=FLOOR_SATURATION_RATE))
        concord=self._write(pd.DataFrame(rows),"model_item_difficulty_concordance.csv")
        hard=[]
        for population,pop in self._population_views(data):
            for score in SCORE_COLUMNS:
                pivot=pop.pivot_table(index="prompt_id",columns="model",values=score+"_numeric")
                for pid,row in pivot.iterrows():hard.append(dict(population=population,prompt_id=pid,dimension=score,models_scoring_0_or_1=int((row<=1).sum()),mean_score=row.mean(),available_models=row.notna().sum()))
        self._write(pd.DataFrame(hard),"shared_and_model_specific_hard_cases.csv")
        summaries=pd.read_csv(self.p.tables/"model_scores_by_dimension.csv");summaries=summaries[summaries.population.eq("evaluation_285")]
        for score in SCORE_COLUMNS:
            short=SHORT_SCORE[score]
            for group,prefix in [("BF16 HF","16A"),("one-bit frontier","16B")]:
                sub=summaries[(summaries.dimension==score)&(summaries.representation_group==group)]
                fig,axes=plt.subplots(2,3,figsize=(14,8),sharex=True,sharey=True)
                for ax,(cond,cg) in zip(axes.ravel(),sub.groupby("condition_short")):
                    for _,r in cg.iterrows():ax.scatter(MODEL_PARAMETERS.get(r.model,np.nan),r["mean"],color=MODEL_COLORS.get(r.model,"#777"));ax.text(MODEL_PARAMETERS.get(r.model,np.nan),r["mean"],MODEL_LABELS.get(r.model,r.model),fontsize=7)
                    ax.set_xscale("log");ax.set_ylim(0,4);ax.set_title(cond)
                fig.suptitle(f"Model size and {short.replace('_',' ')} — {group}");self._figure(self.p.figures/f"{prefix}_{short}_{'bf16' if prefix=='16A' else 'one_bit'}_model_size_relationships.png",fig)
            for cond in ["untreated","full"]:
                for rep in ["BF16 HF","one-bit frontier"]:
                    sub=concord[(concord.population.eq("evaluation_285"))&(concord.dimension==score)&(concord.condition==cond)&(concord.representation==rep)]
                    if sub.empty:continue
                    matrix=sub.pivot(index="model_a",columns="model_b",values="spearman").reindex(index=MODELS,columns=MODELS)
                    fig,ax=plt.subplots(figsize=(7,6));sns.heatmap(matrix,vmin=-1,vmax=1,cmap="vlag",center=0,ax=ax);ax.set_title(f"Item difficulty concordance — {cond}, {rep}");self._figure(self.p.figures/f"17_{short}_{cond}_{rep.lower().replace(' ','_')}_item_difficulty_concordance.png",fig,data=matrix.reset_index(),key="Rows and columns are the six model families. Cell colour is Spearman rank correlation: red is negative, white is zero, and blue is positive.",sources="model_item_difficulty_concordance.csv")

    def _repeat_figures(self,repeatdf,vardf,data):
        repeatdf=repeatdf[repeatdf.population.eq("evaluation_285")] if "population" in repeatdf else repeatdf
        vardf=vardf[vardf.population.eq("evaluation_285")] if "population" in vardf else vardf
        for path,title,frame,y in [("11A_repeated_response_stability.png","Repeated-response stability",repeatdf,"output_token_range"),("11B_repeated_judge_score_stability.png","Repeated judge-score stability",repeatdf,"technique_identification_range"),("13B_scenario_step_score_sensitivity.png","Scenario-step score sensitivity",vardf,"technique_identification_range"),("13C_scenario_step_operational_sensitivity.png","Scenario-step output sensitivity",vardf,"output_token_range")]:
            fig,ax=plt.subplots(figsize=(10,5));
            if len(frame) and y in frame:sns.boxplot(frame,x="model",y=y,hue="representation",ax=ax);ax.tick_params(axis="x",rotation=20)
            ax.set_title(title);self._figure(self.p.figures/path,fig)
        # 12A: score availability is a measurement diagnostic, not a score.
        available=pd.read_csv(self.p.tables/"judge_score_availability.csv") if (self.p.tables/"judge_score_availability.csv").exists() else pd.DataFrame()
        fig,ax=plt.subplots(figsize=(12,6))
        if len(available):
            plot=available[available.population.eq("evaluation_285")].copy()
            plot["availability_rate"]=pd.to_numeric(plot["scalar_in_range_scores"],errors="coerce")/pd.to_numeric(plot["intended_rows"],errors="coerce").replace(0,np.nan)
            plot["stream"]=plot["model"].map(MODEL_LABELS).fillna(plot["model"])+" / "+plot["condition_short"].astype(str)+" / "+plot["representation_label"].astype(str)
            matrix=plot.pivot_table(index="stream",columns="dimension",values="availability_rate",aggfunc="mean")
            matrix=matrix.rename(columns=SHORT_SCORE)
            sns.heatmap(matrix,vmin=0,vmax=1,cmap="mako",ax=ax,cbar_kws={"label":"Available scalar-score proportion"})
        ax.set_title("Judge-score availability by stream")
        self._figure(self.p.figures/"12A_judge_score_availability_diagnostic.png",fig)

        # 12B: naturally duplicated, byte-identical cases provide only an
        # internal consistency check because no designed repeat-judging study
        # was conducted.
        duplicates=pd.read_csv(self.p.tables/"judge_duplicate_case_consistency.csv") if (self.p.tables/"judge_duplicate_case_consistency.csv").exists() else pd.DataFrame()
        fig,axes=plt.subplots(1,2,figsize=(11,4))
        if len(duplicates):
            counts=duplicates["score_vector_exact_agreement"].value_counts().reindex([1,0],fill_value=0)
            axes[0].bar(["Exact agreement","Any disagreement"],counts.values,color=["#009E73","#D55E00"])
            axes[1].hist(pd.to_numeric(duplicates["maximum_dimension_range"],errors="coerce").dropna(),bins=np.arange(-.5,5.6,1),color="#0072B2",edgecolor="white")
        else:
            for ax in axes:ax.text(.5,.5,"No natural duplicate cases",ha="center",va="center",transform=ax.transAxes)
        axes[0].set(ylabel="Duplicate cases",title="Whole score-vector agreement")
        axes[1].set(xlabel="Maximum within-case score range",ylabel="Duplicate cases",title="Largest dimension disagreement")
        fig.suptitle("Natural duplicate-case consistency (not a reliability validation)")
        self._figure(self.p.figures/"12B_judge_duplicate_consistency.png",fig)

        # 12C: compare the judge-returned candidate and warranted labels.  Both
        # originate in the same pass, so this is a circular rule audit.
        conf=pd.read_csv(self.p.tables/"judge_confidence_rule_audit.csv") if (self.p.tables/"judge_confidence_rule_audit.csv").exists() else pd.DataFrame()
        fig,axes=plt.subplots(1,2,figsize=(12,4))
        if len(conf):
            candidate=conf.get("candidate_confidence_label",pd.Series(index=conf.index,dtype=object)).fillna("").astype(str).str.strip()
            warranted=conf.get("judge_warranted_confidence_label",pd.Series(index=conf.index,dtype=object)).fillna("").astype(str).str.strip()
            present=pd.DataFrame({"Candidate label":candidate.ne("").mean(),"Warranted label":warranted.ne("").mean()},index=[0]).T
            axes[0].bar(present.index,present[0],color=["#56B4E9","#E69F00"]);axes[0].set_ylim(0,1)
            agreement=pd.to_numeric(conf.get("confidence_label_agreement_numeric"),errors="coerce").dropna()
            axes[1].bar(range(5),agreement.value_counts().reindex(range(5),fill_value=0),color="#CC79A7")
        axes[0].set(ylabel="Proportion of judgment rows",title="Returned-label availability")
        axes[1].set(xlabel="Agreement score (0-4)",ylabel="Judgments",title="Judge-returned agreement distribution")
        fig.suptitle("Confidence-label rule diagnostic (same-pass, judge-derived fields)")
        self._figure(self.p.figures/"12C_confidence_rule_diagnostic.png",fig)

        # 12D: a separate judgment pass in which Haiku evaluates its own
        # reference responses.  This displays possible self-preference; it is
        # not an external accuracy benchmark.
        selfeval=pd.read_csv(self.p.tables/"haiku_self_evaluation_summary.csv") if (self.p.tables/"haiku_self_evaluation_summary.csv").exists() else pd.DataFrame()
        fig,axes=plt.subplots(1,2,figsize=(13,4))
        if len(selfeval):
            labels=selfeval["dimension_short"].str.replace("_"," ")
            axes[0].barh(labels,selfeval["mean"],color="#F0E442");axes[0].set_xlim(0,4)
            dist=selfeval.set_index("dimension_short")[[f"score_{i}_n" for i in range(5)]]
            denom=dist.sum(axis=1).replace(0,np.nan);left=np.zeros(len(dist))
            for i,color in zip(range(5),ORDINAL_COLORS):
                values=(dist[f"score_{i}_n"]/denom).fillna(0).to_numpy()
                axes[1].barh([x.replace("_"," ") for x in dist.index],values,left=left,label=str(i),color=color);left+=values
            axes[1].set_xlim(0,1);axes[1].legend(title="Score",ncol=5,loc="lower center",bbox_to_anchor=(.5,-.3))
        axes[0].set(xlabel="Mean self-evaluation score",title="Mean by rubric dimension")
        axes[1].set(xlabel="Proportion",title="Full ordinal distribution")
        fig.suptitle("Haiku self-evaluation: circular descriptive baseline")
        self._figure(self.p.figures/"12D_haiku_self_evaluation_diagnostic.png",fig,data=selfeval,
                     key="Yellow bars show mean self-judgment scores; stacked bars show the full 0–4 ordinal distribution from Haiku's separate self-evaluation pass.",
                     sources="haiku_self_evaluation_summary.csv; 03__judgment__haiku_self__validation_test.csv")
        eval_data=data[data.prompt_id.astype(str).isin(self.evaluation_ids)]
        for score in SUBSTANTIVE_SCORES:
            short=SHORT_SCORE[score];fig,ax=plt.subplots(figsize=(9,5));plot=eval_data[[score+"_numeric","tfidf_candidate_haiku","condition_short"]].dropna();sns.regplot(plot,x="tfidf_candidate_haiku",y=score+"_numeric",scatter_kws={"alpha":.08},line_kws={"color":"black"},ax=ax);ax.set(title=f"Haiku-reference lexical similarity vs {short.replace('_',' ')}",xlabel="TF-IDF cosine similarity",ylabel="Judge score (0–4)");self._figure(self.p.figures/f"13_{short}_reference_similarity_vs_score.png",fig)

    def stopping_runtime(self):
        r=self.responses.copy()
        if r.empty:return
        for col in ["output_tokens","elapsed_seconds","max_new_tokens"]:r[col]=pd.to_numeric(r.get(col),errors="coerce")
        r["condition_short"]=r.condition.map(condition_short);r["representation_label"]=r.representation.where(r.representation.ne(""),r.representation_file).map(representation_label) if "representation" in r else r.representation_file.map(representation_label)
        finish=r.get("finish_reason","").astype(str).str.lower();completed=r.status.eq("completed")
        r["natural_eos"]=completed&finish.str.contains("stop|eos")
        # A context-unsupported row can have output_tokens=max_new_tokens=0. It
        # is not a generated response and must never be counted as a ceiling hit.
        r["at_ceiling"]=completed&r.max_new_tokens.gt(0)&r.output_tokens.ge(r.max_new_tokens)
        r["timeout"]=r.get("generation_error","").astype(str).str.lower().str.contains("timeout")
        out=[]
        for population,pop in self._population_views(r):
            for keys,g in pop.groupby(["model","condition_short","representation_label"]):
                gc=g[g.status.eq("completed")];den=len(gc)
                out.append(dict(population=population,model=keys[0],condition=keys[1],representation=keys[2],observed=len(g),completed=den,context_unsupported=int(g.status.eq("context_unsupported").sum()),retryable_error=int(g.status.eq("retryable_error").sum()),natural_eos=int(g.natural_eos.sum()),natural_eos_rate=(g.natural_eos.sum()/den if den else np.nan),at_ceiling=int(g.at_ceiling.sum()),at_ceiling_rate=(g.at_ceiling.sum()/den if den else np.nan),timeout=int(g.timeout.sum()),timeout_rate=g.timeout.mean(),median_output_tokens=gc.output_tokens.median(),p95_output_tokens=gc.output_tokens.quantile(.95),median_latency_seconds=gc.elapsed_seconds.median()))
        df=self._write(pd.DataFrame(out),"token_ceiling_and_stop_reasons.csv")
        fig,axes=plt.subplots(2,3,figsize=(14,8),sharex=True)
        for ax,model in zip(axes.ravel(),MODELS):
            m=df[(df.population.eq("evaluation_285"))&(df.model==model)];x=np.arange(len(m));ax.barh(x,m.at_ceiling_rate,color="#D55E00",label="ceiling");ax.barh(x,m.natural_eos_rate,left=m.at_ceiling_rate,color="#009E73",label="EOS");ax.set_yticks(x,(m.condition+" / "+m.representation).tolist());ax.set_xlim(0,1);ax.set_title(MODEL_LABELS[model])
        fig.suptitle("Stopping behaviour (rates among completed responses)");self._figure(self.p.figures/"14_token_ceiling_rates.png",fig)
        cols=[c for c in ["model","condition_short","representation_label","candidate_runtime_configuration_revision","candidate_prompt_length_bucket","candidate_output_tokens","candidate_elapsed_seconds"] if c in self.long]
        rev=self.long[cols].copy();self._write(rev,"runtime_revision_sensitivity.csv")
        fig,ax=plt.subplots(figsize=(10,5));
        if "candidate_runtime_configuration_revision" in rev and rev.candidate_runtime_configuration_revision.ne("").any():sns.boxplot(rev,x="candidate_runtime_configuration_revision",y="candidate_elapsed_seconds",ax=ax)
        ax.set_title("Runtime-revision sensitivity (descriptive; not causal)");self._figure(self.p.figures/"15_runtime_revision_sensitivity.png",fig)

    # ---------- Sections 1.14--1.18: external reference ladder ----------
    def _eval_rows(self):
        combined=pd.concat([self.long,self.reference_long],ignore_index=True,sort=False)
        if combined.empty:return combined
        return combined[combined["split"].isin(["validation","test"]) & combined["judgment_completed"]].copy()

    def _stream_rows(self,data,model,condition=None,representation_group_value=None):
        out=data[data.model.eq(model)]
        if condition is not None:out=out[out.condition_short.eq(condition)]
        if representation_group_value is not None:out=out[out.representation_group.eq(representation_group_value)]
        return out

    def _reference_stream_label(self,model,condition,rep_group,rep_label):
        if model in REFERENCE_LABELS:return REFERENCE_LABELS[model]
        if rep_group=="BF16 HF" and condition=="untreated":
            return f"{MODEL_LABELS[model]} | BF16 HF | reference | untreated"
        return stream_display_name(model,rep_label,condition)

    def _paired(self,left,right,score):
        cols=["prompt_id","group_id",score+"_numeric"]
        a=left[cols].dropna(subset=[score+"_numeric"]).drop_duplicates("prompt_id")
        b=right[cols].dropna(subset=[score+"_numeric"]).drop_duplicates("prompt_id")
        pair=a.merge(b,on="prompt_id",suffixes=("_left","_right"))
        if len(pair) and not pair.group_id_left.astype(str).eq(pair.group_id_right.astype(str)).all():
            raise ValueError("Scenario-step group differs across paired reference rows")
        pair["group_id"]=pair.group_id_left
        pair["difference"]=pair[score+"_numeric_left"]-pair[score+"_numeric_right"]
        return pair

    def _paired_summary(self,pair,identity):
        if pair.empty:return dict(identity,n=0,independent_groups=0,mean_difference=np.nan,low95=np.nan,high95=np.nan,
                                  sigma_g=np.nan,mde=np.nan,groups_needed_delta_0_3=np.nan,noninferior_delta_0_3=False)
        d=pair.difference.astype(float);lo,hi=cluster_bootstrap_mean_fast(d,pair.group_id,self.bootstrap_replicates)
        gm=pair.groupby("group_id").difference.mean();sigma=float(gm.std(ddof=1)) if len(gm)>1 else np.nan
        mde=(1.96+.8416)*sigma/math.sqrt(len(gm)) if len(gm)>1 else np.nan
        needed=math.ceil(((1.96+.8416)*sigma/NONINFERIORITY_MARGIN)**2) if np.isfinite(sigma) else np.nan
        return dict(identity,n=len(pair),independent_groups=len(gm),mean_difference=d.mean(),median_difference=d.median(),
                    q1_difference=d.quantile(.25),q3_difference=d.quantile(.75),low95=lo,high95=hi,
                    improved_rate=(d>0).mean(),unchanged_rate=(d==0).mean(),worsened_rate=(d<0).mean(),
                    probability_superiority_with_ties=(d.gt(0).mean()+.5*d.eq(0).mean()),sigma_g=sigma,mde=mde,
                    delta=NONINFERIORITY_MARGIN,groups_needed_delta_0_3=needed,
                    noninferior_delta_0_3=bool(np.isfinite(lo) and lo>-NONINFERIORITY_MARGIN))

    def _bootstrap_retention(self,pair):
        def stats_for(frame):
            means=frame[["modal","untreated","bf16_full","onebit_full"]].mean()
            den=means.bf16_full-means.modal
            return np.array([means.onebit_full-means.modal,means.bf16_full-means.onebit_full,
                             den,means.untreated-means.modal,(means.onebit_full-means.modal)/den if den!=0 else np.nan])
        point=stats_for(pair);grouped=pair.groupby("group_id")[["modal","untreated","bf16_full","onebit_full"]].agg(["sum","count"])
        if len(grouped)<2:return point,np.full(5,np.nan),np.full(5,np.nan)
        sums=np.stack([grouped[(c,"sum")].to_numpy(float) for c in ["modal","untreated","bf16_full","onebit_full"]],axis=1);counts=grouped[("modal","count")].to_numpy(float);g=len(grouped);rng=np.random.default_rng(SEED);values=np.empty((self.bootstrap_replicates,5),float);offset=0
        while offset<self.bootstrap_replicates:
            n=min(1000,self.bootstrap_replicates-offset);idx=rng.integers(0,g,size=(n,g));means=sums[idx].sum(axis=1)/counts[idx].sum(axis=1)[:,None];modal,untreated,bf16,onebit=means.T;den=bf16-modal
            values[offset:offset+n]=np.column_stack([onebit-modal,bf16-onebit,den,untreated-modal,np.divide(onebit-modal,den,out=np.full(n,np.nan),where=den!=0)]);offset+=n
        return point,np.nanquantile(values,.025,axis=0),np.nanquantile(values,.975,axis=0)

    def reference_extensions(self):
        data=self._eval_rows()
        missing=sorted(set(REFERENCE_LABELS)-set(data.model.astype(str)))
        if missing:
            raise RuntimeError("Reference judgments are not yet present in results/final for: "+", ".join(missing)+". Complete Notebook 07 Sections 12--19 and rerun Notebook 08 first.")
        self._reference_ladder(data)
        retention=self._retention(data)
        size=self._size_extension(data,retention)
        fidelity=self._distillation_fidelity(data)
        self._mde_table(data,size,fidelity,retention)

    # ---------- Section 1.19: judge-mediated ranked-technique accuracy ----------
    def _rank_metric_rows(self):
        """Build the auditable row table from canonical structured judge fields."""
        data=pd.concat([self.long,self.reference_long],ignore_index=True,sort=False)
        data=data[data["judgment_completed"]].copy()
        key=["model","condition_short","representation_group","representation_label","prompt_id"]
        data["duplicate_stream_prompt"] = data.duplicated(key,keep=False)
        duplicate_counts=(data.groupby(key[:-1],dropna=False).duplicate_stream_prompt.sum().rename("duplicate_rows_detected").reset_index()
                          .rename(columns={"representation_label":"exact_representation"}))
        data=data.sort_values(key+["source_file"]).drop_duplicates(key,keep="last")
        rows=[]
        selected_k={1,3,5,10}
        for _,r in data.iterrows():
            prediction_state,predictions=declared_list_with_state(r.get("judge_extracted_predictions",""))
            truth_state,truth=declared_list_with_state(r.get("ground_truth",""))
            predictions=predictions[:10]
            parent_predictions=parent_normalized_unique(predictions)[:10]
            parent_truth=parent_normalized_unique(truth)
            truth_valid=truth_state=="nonempty_list"
            primary_eligible=prediction_state in {"empty_list","nonempty_list"} and truth_valid
            base={
                "prompt_id":r.get("prompt_id",""),"group_id":r.get("group_id",""),
                "scenario":r.get("scenario",""),"variant":r.get("variant",""),"step":r.get("step",""),
                "split":r.get("split",""),"model":r.get("model",""),
                "condition_short":r.get("condition_short",""),
                "representation_group":r.get("representation_group",""),
                "exact_representation":r.get("representation_label",""),
                "stream_display_name":self._reference_stream_label(r.get("model",""),r.get("condition_short",""),r.get("representation_group",""),r.get("representation_label","")),
                "source_file":r.get("source_file",""),"candidate_record_id":r.get("candidate_record_id",r.get("record_id","")),
                "prediction_field_state":prediction_state,"ground_truth_field_state":truth_state,
                "primary_eligible":primary_eligible,"prediction_count":len(predictions),"ground_truth_count":len(truth),
                "judge_extracted_predictions":json.dumps(predictions,separators=(",",":")),
                "ground_truth":json.dumps(truth,separators=(",",":")),
                "parent_normalized_predictions":json.dumps(parent_predictions,separators=(",",":")),
                "parent_normalized_ground_truth":json.dumps(parent_truth,separators=(",",":")),
                "parent_prediction_count":len(parent_predictions),"parent_ground_truth_count":len(parent_truth),
            }
            truth_set=set(truth)
            parent_truth_set=set(parent_truth)
            for k in range(1,11):
                matched=sorted(set(predictions[:k]) & truth_set)
                parent_matched=sorted(set(parent_predictions[:k]) & parent_truth_set)
                hit=float(bool(matched)) if truth_valid else np.nan
                parent_hit=float(bool(parent_matched)) if truth_valid else np.nan
                base[f"matched_count_at_{k}"]=len(matched) if truth_valid else np.nan
                base[f"parent_matched_count_at_{k}"]=len(parent_matched) if truth_valid else np.nan
                if k in selected_k:base[f"matched_ids_at_{k}"]=json.dumps(matched,separators=(",",":"))
                if k in selected_k:base[f"parent_matched_ids_at_{k}"]=json.dumps(parent_matched,separators=(",",":"))
                base[f"hit_at_{k}"]=hit if primary_eligible else np.nan
                base[f"conservative_hit_at_{k}"]=hit
                base[f"parent_hit_at_{k}"]=parent_hit if primary_eligible else np.nan
                base[f"parent_conservative_hit_at_{k}"]=parent_hit
                if k in selected_k:
                    precision=(len(matched)/k) if truth_valid else np.nan
                    recall=(len(matched)/len(truth)) if truth_valid else np.nan
                    parent_precision=(len(parent_matched)/k) if truth_valid else np.nan
                    parent_recall=(len(parent_matched)/len(parent_truth)) if truth_valid and parent_truth else np.nan
                    base[f"precision_at_{k}"]=precision if primary_eligible else np.nan
                    base[f"recall_at_{k}"]=recall if primary_eligible else np.nan
                    base[f"conservative_precision_at_{k}"]=precision
                    base[f"conservative_recall_at_{k}"]=recall
                    base[f"parent_precision_at_{k}"]=parent_precision if primary_eligible else np.nan
                    base[f"parent_recall_at_{k}"]=parent_recall if primary_eligible else np.nan
                    base[f"parent_conservative_precision_at_{k}"]=parent_precision
                    base[f"parent_conservative_recall_at_{k}"]=parent_recall
            rows.append(base)
        frame=pd.DataFrame(rows)
        frame=frame.merge(duplicate_counts,on=["model","condition_short","representation_group","exact_representation"],how="left")
        frame["duplicate_rows_detected"]=frame["duplicate_rows_detected"].fillna(0).astype(int)
        return frame

    def _parent_rank_view(self,row_table):
        """Expose parent-normalized columns through the common rank summarizers."""
        frame=row_table.copy()
        frame["judge_extracted_predictions"]=frame["parent_normalized_predictions"]
        frame["ground_truth"]=frame["parent_normalized_ground_truth"]
        frame["prediction_count"]=frame["parent_prediction_count"]
        frame["ground_truth_count"]=frame["parent_ground_truth_count"]
        for k in range(1,11):
            frame[f"matched_count_at_{k}"]=frame[f"parent_matched_count_at_{k}"]
            frame[f"hit_at_{k}"]=frame[f"parent_hit_at_{k}"]
            frame[f"conservative_hit_at_{k}"]=frame[f"parent_conservative_hit_at_{k}"]
            if k in (1,3,5,10):
                frame[f"matched_ids_at_{k}"]=frame[f"parent_matched_ids_at_{k}"]
                frame[f"precision_at_{k}"]=frame[f"parent_precision_at_{k}"]
                frame[f"recall_at_{k}"]=frame[f"parent_recall_at_{k}"]
                frame[f"conservative_precision_at_{k}"]=frame[f"parent_conservative_precision_at_{k}"]
                frame[f"conservative_recall_at_{k}"]=frame[f"parent_conservative_recall_at_{k}"]
        return frame

    def _rank_bootstrap_intervals(self,frame,mean_columns,ratio_specs):
        """Reuse one cluster resample for every rank metric in one stream."""
        result={}
        if frame.empty:return result
        grouped=frame.groupby("group_id",sort=True)
        sums=np.column_stack([grouped[c].sum().to_numpy(float) for c in mean_columns])
        counts=grouped.size().to_numpy(float);g=len(counts)
        ratio_num=np.column_stack([grouped[n].sum().to_numpy(float) for n,_ in ratio_specs]) if ratio_specs else np.empty((g,0))
        ratio_den=np.column_stack([grouped[d].sum().to_numpy(float) for _,d in ratio_specs]) if ratio_specs else np.empty((g,0))
        if g<2:
            for i,c in enumerate(mean_columns):result[("mean",c)]=(sums[:,i].sum()/counts.sum(),)*2
            for i,spec in enumerate(ratio_specs):
                den=ratio_den[:,i].sum();value=ratio_num[:,i].sum()/den if den else np.nan;result[("ratio",spec)]=(value,value)
            return result
        rng=np.random.default_rng(SEED);mean_est=np.empty((self.bootstrap_replicates,len(mean_columns)));ratio_est=np.empty((self.bootstrap_replicates,len(ratio_specs)));offset=0
        while offset<self.bootstrap_replicates:
            n=min(500,self.bootstrap_replicates-offset);idx=rng.integers(0,g,size=(n,g));den_rows=counts[idx].sum(axis=1)
            mean_est[offset:offset+n]=sums[idx].sum(axis=1)/den_rows[:,None]
            if ratio_specs:
                num=ratio_num[idx].sum(axis=1);den=ratio_den[idx].sum(axis=1)
                ratio_est[offset:offset+n]=np.divide(num,den,out=np.full_like(num,np.nan),where=den!=0)
            offset+=n
        for i,c in enumerate(mean_columns):result[("mean",c)]=tuple(np.nanquantile(mean_est[:,i],[.025,.975]))
        for i,spec in enumerate(ratio_specs):result[("ratio",spec)]=tuple(np.nanquantile(ratio_est[:,i],[.025,.975]))
        return result

    def _rank_stream_tables(self,row_table,matching_level="exact_id"):
        selected_k=(1,3,5,10);coverage=[];summary=[]
        groups=["model","condition_short","representation_group","exact_representation","stream_display_name"]
        for population,pop in self._population_views(row_table):
          for keys,g in pop.groupby(groups,dropna=False,sort=True):
            identity=dict(zip(groups,keys));intended=285 if population=="evaluation_285" or identity["condition_short"]!="untreated" else 990
            states=g.prediction_field_state.value_counts();primary=g[g.primary_eligible].copy();truth_valid=g[g.ground_truth_count.gt(0)].copy()
            coverage.append(dict(identity,population=population,intended_rows=intended,judgment_rows=len(g),valid_declared_list_rows=int(states.get("nonempty_list",0)+states.get("empty_list",0)),
                nonempty_list_rows=int(states.get("nonempty_list",0)),explicit_empty_list_rows=int(states.get("empty_list",0)),
                missing_or_null_rows=int(states.get("missing_or_null",0)),malformed_non_list_rows=int(states.get("malformed_non_list",0)),
                valid_ground_truth_rows=int(g.ground_truth_count.gt(0).sum()),primary_metric_rows=len(primary),independent_groups=primary.group_id.nunique(),
                duplicate_rows_detected=int(g.duplicate_rows_detected.max()),valid_list_rate=(len(primary)/len(g) if len(g) else np.nan)))
            mean_cols=[f"hit_at_{k}" for k in range(1,11)]+[f"precision_at_{k}" for k in selected_k]+[f"recall_at_{k}" for k in selected_k]
            ratio_specs=[(f"matched_count_at_{k}","ground_truth_count") for k in selected_k]
            intervals=self._rank_bootstrap_intervals(primary,mean_cols,ratio_specs)
            specs=[("hit_rate",k,f"hit_at_{k}","mean") for k in range(1,11)]
            specs += [("precision_at_k_fixed",k,f"precision_at_{k}","mean") for k in selected_k]
            specs += [("recall_at_k_macro",k,f"recall_at_{k}","mean") for k in selected_k]
            for metric,k,column,kind in specs:
                lo,hi=intervals.get((kind,column),(np.nan,np.nan));estimate=primary[column].mean() if len(primary) else np.nan
                conservative=truth_valid[f"conservative_{column}"].mean() if len(truth_valid) else np.nan
                summary.append(dict(identity,population=population,metric=metric,k=k,estimate=estimate,low95=lo,high95=hi,conservative_missing_as_zero_estimate=conservative,
                    n=len(primary),independent_groups=primary.group_id.nunique(),intended_rows=intended,
                    nonempty_list_rows=int(states.get("nonempty_list",0)),explicit_empty_list_rows=int(states.get("empty_list",0)),
                    unavailable_prediction_field_rows=int(states.get("missing_or_null",0)+states.get("malformed_non_list",0)),
                    matching_level=matching_level,
                    matching_rule=("exact ATT&CK ID; sub-techniques distinct" if matching_level=="exact_id" else "parent-normalized ATT&CK ID; Txxxx.xxx collapsed to Txxxx"),
                    precision_denominator_policy="fixed k; missing ranks non-relevant"))
            for k in selected_k:
                spec=(f"matched_count_at_{k}","ground_truth_count");lo,hi=intervals.get(("ratio",spec),(np.nan,np.nan))
                den=primary.ground_truth_count.sum();estimate=primary[f"matched_count_at_{k}"].sum()/den if den else np.nan
                cden=truth_valid.ground_truth_count.sum();conservative=truth_valid[f"matched_count_at_{k}"].sum()/cden if cden else np.nan
                summary.append(dict(identity,population=population,metric="recall_at_k_micro",k=k,estimate=estimate,low95=lo,high95=hi,conservative_missing_as_zero_estimate=conservative,
                    n=len(primary),independent_groups=primary.group_id.nunique(),intended_rows=intended,
                    nonempty_list_rows=int(states.get("nonempty_list",0)),explicit_empty_list_rows=int(states.get("empty_list",0)),
                    unavailable_prediction_field_rows=int(states.get("missing_or_null",0)+states.get("malformed_non_list",0)),
                    matching_level=matching_level,
                    matching_rule=("exact ATT&CK ID; sub-techniques distinct" if matching_level=="exact_id" else "parent-normalized ATT&CK ID; Txxxx.xxx collapsed to Txxxx"),
                    precision_denominator_policy="not applicable"))
        return pd.DataFrame(coverage),pd.DataFrame(summary)

    def _rank_contrasts(self,row_table,matching_level="exact_id"):
        selected_k=(1,3,5,10);metric_cols=[f"hit_at_{k}" for k in range(1,11)]+[f"precision_at_{k}" for k in selected_k]+[f"recall_at_{k}" for k in selected_k]
        rows=[]
        def compare(left,right,identity,population="evaluation_285"):
            if population=="evaluation_285":
                left=left[left.prompt_id.astype(str).isin(self.evaluation_ids)];right=right[right.prompt_id.astype(str).isin(self.evaluation_ids)]
            a=left[["prompt_id","group_id",*metric_cols]].drop_duplicates("prompt_id")
            b=right[["prompt_id","group_id",*metric_cols]].drop_duplicates("prompt_id")
            pair=a.merge(b,on="prompt_id",suffixes=("_left","_right"))
            for col in metric_cols:
                valid=pair[["group_id_left",col+"_left",col+"_right"]].dropna();d=valid[col+"_left"]-valid[col+"_right"]
                lo,hi=cluster_bootstrap_mean_fast(d,valid.group_id_left,self.bootstrap_replicates)
                metric=("hit_rate" if col.startswith("hit") else "precision_at_k_fixed" if col.startswith("precision") else "recall_at_k_macro")
                k=int(col.rsplit("_",1)[1]);rows.append(dict(identity,population=population,metric=metric,k=k,common_support_n=len(valid),independent_groups=valid.group_id_left.nunique(),
                    matching_level=matching_level,mean_difference=d.mean(),low95=lo,high95=hi,improved_rate=d.gt(0).mean(),unchanged_rate=d.eq(0).mean(),worsened_rate=d.lt(0).mean(),
                    probability_superiority_with_ties=d.gt(0).mean()+.5*d.eq(0).mean()))
        for model in MODELS:
            for rep in ("BF16 HF","one-bit frontier"):
                base=row_table[(row_table.model==model)&(row_table.representation_group==rep)&(row_table.condition_short=="untreated")]
                for cond in CONDITIONS[1:]:
                    treated=row_table[(row_table.model==model)&(row_table.representation_group==rep)&(row_table.condition_short==cond)]
                    compare(treated,base,dict(model=model,contrast="SFT minus untreated",representation_group=rep,condition=cond,left_stream=cond,right_stream="untreated"))
            for cond in CONDITIONS:
                one=row_table[(row_table.model==model)&(row_table.representation_group=="one-bit frontier")&(row_table.condition_short==cond)]
                bf=row_table[(row_table.model==model)&(row_table.representation_group=="BF16 HF")&(row_table.condition_short==cond)]
                compare(one,bf,dict(model=model,contrast="one-bit deployment minus BF16 HF",representation_group="deployment_stack",condition=cond,left_stream="one-bit frontier",right_stream="BF16 HF"),"evaluation_285")
                if cond=="untreated":compare(one,bf,dict(model=model,contrast="one-bit deployment minus BF16 HF",representation_group="deployment_stack",condition=cond,left_stream="one-bit frontier",right_stream="BF16 HF"),"all_990")
        return pd.DataFrame(rows)

    def _rank_metric_figures(self,summary):
        summary=summary[summary.population.eq("evaluation_285")]
        core=summary[(summary.metric=="hit_rate")&summary.condition_short.isin(["untreated","full"])].copy()
        styles={("BF16 HF","untreated"):("#4D4D4D","-"),("BF16 HF","full"):("#CC79A7","-"),("one-bit frontier","untreated"):("#4D4D4D","--"),("one-bit frontier","full"):("#CC79A7","--")}
        fig,axes=plt.subplots(2,3,figsize=(15,9),sharex=True,sharey=True)
        for ax,model in zip(axes.ravel(),MODELS):
            for (rep,cond),(colour,ls) in styles.items():
                p=core[(core.model==model)&(core.representation_group==rep)&(core.condition_short==cond)].sort_values("k")
                if len(p):ax.plot(p.k,p.estimate,marker="o",ms=3,color=colour,ls=ls,label=f"{rep} | {TREATMENT_LABELS[cond]}")
            ax.set_title(MODEL_LABELS[model]);ax.set_xlim(1,10);ax.set_ylim(0,1);ax.set_xticks(range(1,11));ax.set_xlabel("k");ax.set_ylabel("Judge-mediated Hit@k")
        fig.suptitle("Judge-mediated Hit@k across the four core streams")
        self._figure(self.p.figures/"19C_judge_mediated_hit_at_k_core_streams.png",fig,data=core,key="Solid lines are BF16 HF; dashed lines are one-bit-frontier deployment. Grey is untreated and purple is SFT-full.",sources="judge_mediated_rank_metrics_by_stream.csv")
        for metric,figure,title in [("hit_rate","S19C_judge_mediated_hit_at_k_all_streams.png","Hit@k"),("precision_at_k_fixed","S19D_judge_mediated_precision_at_k_all_streams.png","Precision@k (fixed-rank denominator)"),("recall_at_k_macro","S19E_judge_mediated_recall_at_k_all_streams.png","Macro Recall@k")]:
            p=summary[(summary.metric==metric)&summary.k.isin([1,3,5,10])].copy();p["row_label"]=p.model.map(MODEL_LABELS)+" | "+p.condition_short.map(TREATMENT_LABELS)
            order=[MODEL_LABELS[m]+" | "+TREATMENT_LABELS[c] for m in MODELS for c in CONDITIONS]
            fig,axes=plt.subplots(1,2,figsize=(12,12),sharey=True)
            for ax,rep in zip(axes,["BF16 HF","one-bit frontier"]):
                q=p[p.representation_group==rep].pivot_table(index="row_label",columns="k",values="estimate").reindex(order)
                sns.heatmap(q,vmin=0,vmax=1,cmap="viridis",annot=True,fmt=".2f",ax=ax,cbar=ax is axes[-1]);ax.set_title(rep);ax.set(xlabel="k",ylabel="")
            fig.suptitle(f"All models and adapters — judge-mediated {title}")
            self._figure(self.p.figures/figure,fig,data=p,key="Rows are model/treatment streams; columns are k. BF16 HF and one-bit-frontier panels use the same 0–1 scale.",sources="judge_mediated_rank_metrics_by_stream.csv")

    def _parent_rank_sensitivity_figure(self,exact_summary,parent_summary):
        exact_summary=exact_summary[exact_summary.population.eq("evaluation_285")];parent_summary=parent_summary[parent_summary.population.eq("evaluation_285")]
        keys=["population","model","condition_short","representation_group","exact_representation","metric","k"]
        exact=exact_summary[keys+["estimate"]].rename(columns={"estimate":"exact_estimate"})
        parent=parent_summary[keys+["estimate"]].rename(columns={"estimate":"parent_normalized_estimate"})
        data=exact.merge(parent,on=keys,how="inner")
        data["parent_minus_exact"]=data.parent_normalized_estimate-data.exact_estimate
        core=data[(data.metric=="hit_rate")&data.condition_short.isin(["untreated","full"])].copy()
        styles={("BF16 HF","untreated"):("#4D4D4D","-"),("BF16 HF","full"):("#CC79A7","-"),("one-bit frontier","untreated"):("#4D4D4D","--"),("one-bit frontier","full"):("#CC79A7","--")}
        fig,axes=plt.subplots(2,3,figsize=(15,9),sharex=True,sharey=True)
        for ax,model in zip(axes.ravel(),MODELS):
            for (rep,cond),(colour,ls) in styles.items():
                p=core[(core.model==model)&(core.representation_group==rep)&(core.condition_short==cond)].sort_values("k")
                if len(p):ax.plot(p.k,p.parent_minus_exact,marker="o",ms=3,color=colour,ls=ls,label=f"{rep} | {TREATMENT_LABELS[cond]}")
            ax.axhline(0,color="#777777",lw=.8);ax.set_title(MODEL_LABELS[model]);ax.set_xlim(1,10);ax.set_xticks(range(1,11));ax.set_xlabel("k");ax.set_ylabel("Parent-normalized minus exact Hit@k")
        fig.suptitle("Sensitivity of Hit@k to parent normalization")
        self._figure(self.p.figures/"S19F_judge_mediated_parent_normalization_sensitivity.png",fig,data=data,
                     key="Positive values show the increase obtained by collapsing sub-techniques Txxxx.xxx to parent Txxxx. Exact-ID matching remains primary.",
                     sources="judge_mediated_rank_metrics_by_stream.csv; judge_mediated_parent_rank_metrics_by_stream.csv")

    def judge_mediated_rank_metrics(self):
        """Calculate Hit@1--10 and precision/recall at 1,3,5,10."""
        rows=self._write(self._rank_metric_rows(),"judge_mediated_rank_metrics_by_row.csv")
        coverage,summary=self._rank_stream_tables(rows,"exact_id")
        self._write(coverage,"judge_mediated_rank_metric_coverage.csv")
        self._write(summary,"judge_mediated_rank_metrics_by_stream.csv")
        contrasts=self._write(self._rank_contrasts(rows,"exact_id"),"judge_mediated_rank_metric_contrasts.csv")
        parent_rows=self._parent_rank_view(rows)
        _,parent_summary=self._rank_stream_tables(parent_rows,"parent_normalized")
        self._write(parent_summary,"judge_mediated_parent_rank_metrics_by_stream.csv")
        parent_contrasts=self._write(self._rank_contrasts(parent_rows,"parent_normalized"),"judge_mediated_parent_rank_metric_contrasts.csv")
        legacy=summary[(summary.metric=="hit_rate")&(summary.k==10)].rename(columns={"estimate":"judge_mediated_hit_rate"})
        recall=summary[(summary.metric=="recall_at_k_macro")&(summary.k==10)][["population","model","condition_short","representation_group","exact_representation","estimate"]].rename(columns={"estimate":"judge_mediated_macro_recall_at_10"})
        legacy=legacy.merge(recall,on=["population","model","condition_short","representation_group","exact_representation"],how="left")
        self._write(legacy,"judge_mediated_technique_recall.csv")
        self._rank_metric_figures(summary)
        self._parent_rank_sensitivity_figure(summary,parent_summary)
        return {"row_metrics":rows,"coverage":coverage,"stream_metrics":summary,"contrasts":contrasts,
                "parent_stream_metrics":parent_summary,"parent_contrasts":parent_contrasts}

    # ---------- Section 1.20: representativeness of the 604-row training subset ----------
    def training_subset_representativeness(self):
        """Audit the universal context-feasible training subset without changing it."""
        data_dir=self.p.root/"data"
        prompts=pd.read_csv(data_dir/"prompts.csv")
        splits=pd.read_csv(data_dir/"split_manifest.csv")
        fit=pd.read_csv(data_dir/"context_feasibility.csv")
        exclusions=pd.read_csv(data_dir/"training_exclusions.csv")
        train=splits[splits.split.eq("train")].copy()
        excluded_ids=set(exclusions.prompt_id.astype(int))
        train["training_subset"]=np.where(train.prompt_id.astype(int).isin(excluded_ids),"excluded","retained")
        train=train.merge(prompts[["prompt_id","prompt_sha256","prompt_text"]],on="prompt_id",how="left",validate="one_to_one")
        train["prompt_characters"]=train.prompt_text.str.len()
        train["prompt_words"]=train.prompt_text.str.split().str.len()
        train["prompt_lines"]=train.prompt_text.str.count("\n")+1
        row_audit=train.drop(columns=["prompt_text"]).copy()
        self._write(row_audit,"training_subset_representativeness_rows.csv")

        model_rows=[]
        train_fit=fit[fit.split.eq("train")].copy()
        train_fit["fits_training"]=train_fit.fits_training.astype(str).str.lower().eq("true") if train_fit.fits_training.dtype==object else train_fit.fits_training.astype(bool)
        for model,g in train_fit.groupby("model",sort=True):
            failed=g[~g.fits_training]
            model_rows.append(dict(model=model,model_display=MODEL_LABELS.get(model,model),training_rows=len(g),training_context_tokens=int(g.training_context.iloc[0]),
                native_context_tokens=int(g.context.iloc[0]),model_specific_nonfit_rows=len(failed),universal_excluded_rows=len(excluded_ids),
                nonfit_share_of_universal_exclusions=len(set(failed.prompt_id.astype(int))&excluded_ids)/len(excluded_ids),
                minimum_nonfit_required_tokens=failed.required_tokens.min() if len(failed) else np.nan,
                maximum_nonfit_required_tokens=failed.required_tokens.max() if len(failed) else np.nan))
        model_table=self._write(pd.DataFrame(model_rows),"training_subset_model_feasibility.csv")

        def summarize(values,metric,source_model="raw_text"):
            frame=pd.DataFrame({"value":pd.to_numeric(values["value"],errors="coerce"),"training_subset":values["training_subset"]}).dropna()
            retained=frame[frame.training_subset.eq("retained")].value;excluded=frame[frame.training_subset.eq("excluded")].value
            pooled=math.sqrt(((len(retained)-1)*retained.var(ddof=1)+(len(excluded)-1)*excluded.var(ddof=1))/max(len(retained)+len(excluded)-2,1))
            return dict(metric=metric,source_model=source_model,retained_n=len(retained),excluded_n=len(excluded),
                retained_mean=retained.mean(),excluded_mean=excluded.mean(),retained_median=retained.median(),excluded_median=excluded.median(),
                retained_q90=retained.quantile(.9),excluded_q90=excluded.quantile(.9),retained_max=retained.max(),excluded_max=excluded.max(),
                standardized_mean_difference=(excluded.mean()-retained.mean())/pooled if pooled else np.nan)
        length_rows=[]
        for col in ("prompt_characters","prompt_words","prompt_lines"):
            length_rows.append(summarize(train[["training_subset",col]].rename(columns={col:"value"}),col))
        membership=train[["prompt_id","training_subset"]]
        joined=train_fit.merge(membership,on="prompt_id",how="inner",validate="many_to_one")
        for model,g in joined.groupby("model",sort=True):
            for col in ("prompt_tokens","teacher_and_ending_tokens","required_tokens"):
                length_rows.append(summarize(g[["training_subset",col]].rename(columns={col:"value"}),col,model))
        length_table=self._write(pd.DataFrame(length_rows),"training_subset_length_comparison.csv")

        scenario=train.groupby(["scenario","training_subset"]).size().unstack(fill_value=0).reset_index()
        for col in ("retained","excluded"):
            if col not in scenario:scenario[col]=0
        scenario["total"]=scenario.retained+scenario.excluded
        scenario["excluded_rate"]=scenario.excluded/scenario.total
        scenario["retained_rate"]=scenario.retained/scenario.total
        scenario_table=self._write(scenario,"training_subset_scenario_comparison.csv")

        groups=train.groupby(["group_id","scenario","step"],as_index=False).agg(group_rows=("prompt_id","size"),excluded_rows=("training_subset",lambda x:int(x.eq("excluded").sum())),retained_rows=("training_subset",lambda x:int(x.eq("retained").sum())))
        groups["group_status"]=np.select([groups.excluded_rows.eq(0),groups.retained_rows.eq(0)],["all_retained","all_excluded"],default="mixed")
        group_table=self._write(groups,"training_subset_group_coverage.csv")

        bitnet=joined[joined.model.eq("bitnet2b")].copy()
        fig,axes=plt.subplots(1,3,figsize=(17,5))
        for status,colour in [("retained","#0072B2"),("excluded","#D55E00")]:
            values=bitnet[bitnet.training_subset.eq(status)].required_tokens.sort_values().to_numpy()
            axes[0].plot(values,np.arange(1,len(values)+1)/len(values),label=f"{status} (n={len(values)})",color=colour)
        axes[0].axvline(4096,color="#222222",ls="--",label="BitNet training context: 4,096")
        axes[0].set(xscale="log",xlabel="Complete-example tokens under BitNet tokenizer",ylabel="ECDF",title="A. Length-selected training subset")
        axes[1].bar(scenario_table.scenario.astype(str),scenario_table.excluded_rate,color="#D55E00")
        for i,r in scenario_table.iterrows():axes[1].text(i,r.excluded_rate+.015,f"{int(r.excluded)}/{int(r.total)}",ha="center",fontsize=8)
        axes[1].set(xlabel="Scenario",ylabel="Excluded proportion",ylim=(0,1.12),title="B. Exclusion rate by scenario")
        order=model_table.sort_values("model_specific_nonfit_rows")
        axes[2].barh(order.model_display,order.model_specific_nonfit_rows,color=[MODEL_COLORS.get(m,"#777777") for m in order.model])
        for i,v in enumerate(order.model_specific_nonfit_rows):axes[2].text(v+1,i,str(int(v)),va="center",fontsize=8)
        axes[2].set(xlabel="Training rows exceeding model limit",title="C. Model-specific feasibility")
        fig.suptitle("Representativeness audit of the 604-row context-feasible training subset")
        figure_data=pd.concat([scenario_table.assign(panel="scenario"),model_table.assign(panel="model")],ignore_index=True,sort=False)
        self._figure(self.p.figures/"18F_training_subset_representativeness.png",fig,data=figure_data,
            key="Blue/orange distinguish retained and excluded rows; bars report scenario exclusion rates and model-specific context failures.",
            sources="data/context_feasibility.csv; data/training_exclusions.csv; data/prompts.csv; data/split_manifest.csv")
        return {"rows":row_audit,"model_feasibility":model_table,"length_comparison":length_table,"scenario_comparison":scenario_table,"group_coverage":group_table}

    # ---------- Section 1.21: unseen labels and split balance ----------
    def split_label_balance(self):
        """Describe technique overlap and balance across frozen data partitions."""
        data_dir=self.p.root/"data"
        prompts=pd.read_csv(data_dir/"prompts.csv")
        truth=pd.read_csv(data_dir/"ground_truth.csv")
        splits=pd.read_csv(data_dir/"split_manifest.csv")
        fit=pd.read_csv(data_dir/"context_feasibility.csv")
        excluded_ids=set(pd.read_csv(data_dir/"training_exclusions.csv").prompt_id.astype(int))
        data=splits.merge(prompts[["prompt_id","prompt_sha256","prompt_text"]],on="prompt_id",validate="one_to_one").merge(truth[["prompt_id","ground_truth"]],on="prompt_id",validate="one_to_one")
        data["labels"]=data.ground_truth.map(lambda value: declared_list_with_state(value)[1])
        data["label_count"]=data.labels.map(len)
        data["prompt_characters"]=data.prompt_text.str.len();data["prompt_words"]=data.prompt_text.str.split().str.len();data["prompt_lines"]=data.prompt_text.str.count("\n")+1
        masks={
            "train_original":data.split.eq("train"),
            "train_feasible":data.split.eq("train")&~data.prompt_id.astype(int).isin(excluded_ids),
            "validation":data.split.eq("validation"),
            "test":data.split.eq("test"),
            "evaluation_pool":data.split.isin(["validation","test"]),
        }
        population_order=list(masks)
        summary=[]
        for name,mask in masks.items():
            g=data[mask]
            summary.append(dict(population=name,rows=len(g),distinct_prompt_hashes=g.prompt_sha256.nunique(),duplicate_prompt_rows=len(g)-g.prompt_sha256.nunique(),
                scenario_step_groups=g.group_id.nunique(),scenarios=g.scenario.nunique(),distinct_exact_technique_ids=len(set().union(*g.labels) if len(g) else set()),
                mean_labels_per_row=g.label_count.mean(),median_labels_per_row=g.label_count.median(),
                mean_prompt_characters=g.prompt_characters.mean(),median_prompt_characters=g.prompt_characters.median(),p90_prompt_characters=g.prompt_characters.quantile(.9),maximum_prompt_characters=g.prompt_characters.max()))
        summary_table=self._write(pd.DataFrame(summary),"split_population_summary.csv")

        scenario_rows=[];multiplicity_rows=[];length_rows=[]
        for name,mask in masks.items():
            g=data[mask]
            for scenario,count in g.scenario.value_counts().sort_index().items():scenario_rows.append(dict(population=name,scenario=scenario,rows=count,proportion=count/len(g)))
            for labels,count in g.label_count.value_counts().sort_index().items():multiplicity_rows.append(dict(population=name,label_count=labels,rows=count,proportion=count/len(g)))
            for metric in ("prompt_characters","prompt_words","prompt_lines"):
                x=g[metric]
                length_rows.append(dict(population=name,metric=metric,source_model="raw_text",n=len(x),mean=x.mean(),median=x.median(),q10=x.quantile(.1),q90=x.quantile(.9),maximum=x.max()))
        scenario_table=self._write(pd.DataFrame(scenario_rows),"split_scenario_balance.csv")
        multiplicity_table=self._write(pd.DataFrame(multiplicity_rows),"split_label_multiplicity.csv")
        fit_membership=fit.merge(data[["prompt_id","split"]],on=["prompt_id","split"],validate="many_to_one")
        for model,g_model in fit_membership.groupby("model"):
            for name,mask in masks.items():
                ids=set(data.loc[mask,"prompt_id"]);x=g_model[g_model.prompt_id.isin(ids)].prompt_tokens
                length_rows.append(dict(population=name,metric="prompt_tokens",source_model=model,n=len(x),mean=x.mean(),median=x.median(),q10=x.quantile(.1),q90=x.quantile(.9),maximum=x.max()))
        length_table=self._write(pd.DataFrame(length_rows),"split_prompt_length_comparison.csv")

        train_labels=data[masks["train_feasible"]].explode("labels").groupby("labels").prompt_id.nunique()
        evaluation=data[masks["evaluation_pool"]].copy();eval_exploded=evaluation.explode("labels")
        eval_counts=eval_exploded.groupby("labels").prompt_id.nunique();val_counts=eval_exploded[eval_exploded.split.eq("validation")].groupby("labels").prompt_id.nunique();test_counts=eval_exploded[eval_exploded.split.eq("test")].groupby("labels").prompt_id.nunique()
        techniques=sorted(set(train_labels.index)|set(eval_counts.index));tech_rows=[]
        for technique in techniques:
            tech_rows.append(dict(technique_id=technique,train_feasible_rows=int(train_labels.get(technique,0)),evaluation_rows=int(eval_counts.get(technique,0)),
                validation_rows=int(val_counts.get(technique,0)),test_rows=int(test_counts.get(technique,0)),
                unseen_in_feasible_training=bool(eval_counts.get(technique,0)>0 and train_labels.get(technique,0)==0)))
        technique_table=self._write(pd.DataFrame(tech_rows),"evaluation_unseen_techniques.csv")
        unseen=set(technique_table.loc[technique_table.unseen_in_feasible_training,"technique_id"])
        unseen_rows=[]
        for _,r in evaluation.iterrows():
            row_unseen=sorted(set(r.labels)&unseen)
            if row_unseen:unseen_rows.append(dict(prompt_id=r.prompt_id,scenario=r.scenario,variant=r.variant,step=r.step,group_id=r.group_id,split=r.split,
                ground_truth=json.dumps(r.labels,separators=(",",":")),unseen_ground_truth=json.dumps(row_unseen,separators=(",",":")),unseen_count=len(row_unseen),prompt_sha256=r.prompt_sha256))
        unseen_row_table=self._write(pd.DataFrame(unseen_rows),"evaluation_unseen_technique_rows.csv")

        fig,axes=plt.subplots(1,3,figsize=(18,5))
        plotted_populations=["train_original","train_feasible","validation","test"]
        scenario_plot=scenario_table[scenario_table.population.isin(plotted_populations)].pivot(index="scenario",columns="population",values="proportion").fillna(0).reindex(columns=plotted_populations)
        scenario_plot.plot(kind="bar",ax=axes[0],color=["#999999","#0072B2","#E69F00","#009E73"]);axes[0].set(xlabel="Scenario",ylabel="Proportion of rows",title="A. Scenario composition")
        multiplicity_plot=multiplicity_table[multiplicity_table.population.isin(plotted_populations)].pivot(index="label_count",columns="population",values="proportion").fillna(0).reindex(columns=plotted_populations)
        multiplicity_plot.plot(kind="bar",ax=axes[1],color=["#999999","#0072B2","#E69F00","#009E73"]);axes[1].set(xlabel="Ground-truth IDs per row",ylabel="Proportion of rows",title="B. Label multiplicity")
        ecdf_rows=[]
        for name,colour in zip(plotted_populations,["#999999","#0072B2","#E69F00","#009E73"]):
            values=data[masks[name]].prompt_characters.sort_values().to_numpy();ecdf=np.arange(1,len(values)+1)/len(values);axes[2].plot(values,ecdf,label=name,color=colour)
            ecdf_rows.extend(dict(population=name,prompt_characters=value,ecdf=cumulative,panel="prompt_length") for value,cumulative in zip(values,ecdf))
        axes[2].set(xscale="log",xlabel="Prompt characters",ylabel="ECDF",title="C. Prompt-length distribution")
        fig.suptitle("Balance of frozen train, validation and test partitions")
        figure_data=pd.concat([scenario_table.assign(panel="scenario"),multiplicity_table.assign(panel="label_multiplicity"),pd.DataFrame(ecdf_rows)],ignore_index=True,sort=False)
        self._figure(self.p.figures/"18G_split_label_and_prompt_balance.png",fig,data=figure_data,
            key="Grey is original train, blue is the 604-row feasible train subset, orange is validation and green is test.",
            sources="data/ground_truth.csv; data/split_manifest.csv; data/prompts.csv; data/training_exclusions.csv; data/context_feasibility.csv")
        return {"population_summary":summary_table,"scenario_balance":scenario_table,"label_multiplicity":multiplicity_table,"prompt_length":length_table,
                "technique_overlap":technique_table,"unseen_rows":unseen_row_table}

    # ---------- Sections 1.22--1.23: positive frontier rows and label exposure ----------
    def positive_frontier_and_training_exposure(self):
        """Create qualitative and training-exposure audits from canonical fields.

        This method never reparses candidate text.  It uses the scalar scores
        and judge-derived fields already exported by Notebooks 03/07, and the
        unseen-label row audit produced by `split_label_balance()`.
        """
        score_numeric=[score+"_numeric" for score in SCORE_COLUMNS]
        frontier=self.long[(self.long.representation_group.eq("one-bit frontier"))&self.long.judgment_completed].copy()
        frontier["positive_score_count"]=frontier[score_numeric].gt(0).sum(axis=1)
        frontier=frontier[frontier.positive_score_count.gt(0)].copy()
        frontier["positive_score_dimensions"]=frontier.apply(
            lambda r:json.dumps([score for score in SCORE_COLUMNS if pd.notna(r.get(score+"_numeric")) and float(r.get(score+"_numeric"))>0],separators=(",",":")),axis=1)
        response=frontier.get("candidate_response_raw",pd.Series("",index=frontier.index)).fillna("").astype(str)
        frontier["candidate_response_characters"]=response.str.len()
        frontier["candidate_response_excerpt"]=response.str.replace(r"\s+"," ",regex=True).str.strip().str.slice(0,600)
        frontier["excerpt_truncated"]=frontier.candidate_response_characters.gt(600)
        register_cols=["prompt_id","group_id","scenario","variant","step","split","model","condition_short","representation_group","representation_label",
            "stream_display_name","candidate_status","candidate_record_id","candidate_response_sha256","candidate_response_characters",
            "candidate_response_excerpt","excerpt_truncated","positive_score_count","positive_score_dimensions",*score_numeric,
            "judge_failure_labels","judge_extracted_predictions","ground_truth","source_file"]
        for col in register_cols:
            if col not in frontier:frontier[col]=""
        frontier["stream_display_name"]=[stream_display_name(m,r,c) for m,r,c in zip(frontier.model,frontier.representation_label,frontier.condition_short)]
        register=frontier[register_cols].sort_values(["model","condition_short","representation_label","prompt_id"])
        register=self._write(register,"one_bit_positive_score_row_register.csv")

        unseen_path=self.p.tables/"evaluation_unseen_technique_rows.csv"
        if not unseen_path.exists():self.split_label_balance()
        unseen=pd.read_csv(unseen_path,dtype={"prompt_id":str})
        eval_ids=self.splits[self.splits.split.isin(["validation","test"])][["prompt_id","group_id","split"]].copy()
        eval_ids["prompt_id"]=eval_ids.prompt_id.astype(str)
        truth=self.truth[["prompt_id","ground_truth"]].copy();truth["prompt_id"]=truth.prompt_id.astype(str)
        exposure=eval_ids.merge(truth,on="prompt_id",validate="one_to_one")
        exposure["ground_truth_count"]=exposure.ground_truth.map(lambda x:len(declared_list_with_state(x)[1]))
        unseen_counts=unseen.set_index(unseen.prompt_id.astype(str)).unseen_count.to_dict() if len(unseen) else {}
        exposure["unseen_ground_truth_count"]=exposure.prompt_id.map(unseen_counts).fillna(0).astype(int)
        exposure["training_label_exposure"]=np.select(
            [exposure.unseen_ground_truth_count.eq(0),exposure.unseen_ground_truth_count.ge(exposure.ground_truth_count)],
            ["all_ground_truth_seen_in_604_train","all_ground_truth_unseen_in_604_train"],
            default="mixed_seen_and_unseen_ground_truth")
        exposure["contains_unseen_ground_truth"]=exposure.unseen_ground_truth_count.gt(0)
        exposure=self._write(exposure,"evaluation_training_label_exposure_rows.csv")

        evaluated=self.long[self.long.split.isin(["validation","test"])].copy()
        evaluated["prompt_id"]=evaluated.prompt_id.astype(str)
        evaluated=evaluated.merge(exposure[["prompt_id","training_label_exposure","contains_unseen_ground_truth","ground_truth_count","unseen_ground_truth_count"]],on="prompt_id",how="inner",validate="many_to_one")
        summary=[]
        group_cols=["model","condition_short","representation_group","representation_label","training_label_exposure"]
        for keys,g in evaluated.groupby(group_cols,dropna=False):
            for score in SCORE_COLUMNS:
                x=g[[score+"_numeric","group_id"]].dropna()
                lo,hi=cluster_bootstrap_mean_fast(x[score+"_numeric"],x.group_id,self.bootstrap_replicates)
                summary.append(dict(zip(group_cols,keys),dimension=score,dimension_short=SHORT_SCORE[score],
                    intended_stratum_rows=g.prompt_id.nunique(),eligible_score_rows=len(x),independent_groups=x.group_id.nunique(),
                    mean=x[score+"_numeric"].mean(),median=x[score+"_numeric"].median(),q1=x[score+"_numeric"].quantile(.25),q3=x[score+"_numeric"].quantile(.75),
                    proportion_ge3=x[score+"_numeric"].ge(3).mean() if len(x) else np.nan,low95=lo,high95=hi))
        summary_table=self._write(pd.DataFrame(summary),"scores_by_training_label_exposure.csv")

        lookup={(m,c,r):g for (m,c,r),g in evaluated.groupby(["model","condition_short","representation_group"])}
        effects=[]
        for model in MODELS:
            for rep in ("BF16 HF","one-bit frontier"):
                untreated=lookup.get((model,"untreated",rep))
                if untreated is None:continue
                for adapter in CONDITIONS[1:]:
                    treated=lookup.get((model,adapter,rep))
                    if treated is None:continue
                    for exposure_name in exposure.training_label_exposure.unique():
                        left=treated[treated.training_label_exposure.eq(exposure_name)]
                        right=untreated[untreated.training_label_exposure.eq(exposure_name)]
                        for score in SCORE_COLUMNS:
                            pair=left[["prompt_id","group_id",score+"_numeric"]].rename(columns={score+"_numeric":"treated"}).merge(
                                right[["prompt_id",score+"_numeric"]].rename(columns={score+"_numeric":"untreated"}),on="prompt_id").dropna(subset=["treated","untreated"])
                            d=pair.treated-pair.untreated;lo,hi=cluster_bootstrap_mean_fast(d,pair.group_id,self.bootstrap_replicates)
                            effects.append(dict(model=model,adapter=adapter,representation_group=rep,training_label_exposure=exposure_name,
                                dimension=score,dimension_short=SHORT_SCORE[score],common_support_n=len(pair),independent_groups=pair.group_id.nunique(),
                                mean_sft_minus_untreated=d.mean(),median_sft_minus_untreated=d.median(),low95=lo,high95=hi,
                                improved_rate=d.gt(0).mean() if len(d) else np.nan,unchanged_rate=d.eq(0).mean() if len(d) else np.nan,
                                worsened_rate=d.lt(0).mean() if len(d) else np.nan))
        effects_table=self._write(pd.DataFrame(effects),"sft_effects_by_training_label_exposure.csv")
        return {"positive_frontier_register":register,"exposure_rows":exposure,"score_summary":summary_table,"paired_sft_effects":effects_table}

    def _reference_ladder(self,data):
        definitions=[]
        for model in MODELS:
            for rep,cond in [("BF16 HF","untreated"),("BF16 HF","full"),("one-bit frontier","untreated"),("one-bit frontier","full")]:
                g=self._stream_rows(data,model,cond,rep)
                definitions.append((model,cond,rep,(g.representation_label.iloc[0] if len(g) else rep),g))
        for model in ("gpt55_teacher","baseline_modal","baseline_null"):
            definitions.append((model,"teacher" if model=="gpt55_teacher" else "baseline","reference","reference",self._stream_rows(data,model)))
        rows=[]
        for score in SCORE_COLUMNS:
            for order,(model,cond,rep,exact,g) in enumerate(definitions):
                values=g[[score+"_numeric","group_id"]].dropna()
                lo,hi=cluster_bootstrap_mean_fast(values[score+"_numeric"],values.group_id,self.bootstrap_replicates)
                prop=(values[score+"_numeric"]>=3).mean() if len(values) else np.nan
                p_lo,p_hi=cluster_bootstrap_mean_fast(values[score+"_numeric"].ge(3).astype(float),values.group_id,self.bootstrap_replicates)
                rows.append(dict(dimension=score,dimension_short=SHORT_SCORE[score],display_order=order,model=model,
                    condition=cond,representation_group=rep,exact_representation=exact,
                    stream_display_name=self._reference_stream_label(model,cond,rep,exact),pool_n=285,
                    eligible_n=len(values),unavailable_n=285-len(values),mean=values[score+"_numeric"].mean(),
                    low95=lo,high95=hi,proportion_ge_3=prop,proportion_ge_3_low95=p_lo,proportion_ge_3_high95=p_hi,
                    bootstrap_replicates=self.bootstrap_replicates,bootstrap_cluster="scenario-step group_id"))
        ladder=self._write(pd.DataFrame(rows),"reference_ladder.csv")
        for score in SCORE_COLUMNS:
            short=SHORT_SCORE[score];plot=ladder[ladder.dimension.eq(score)].sort_values("display_order",ascending=False)
            fig,(ax,bx)=plt.subplots(1,2,figsize=(17,max(9,.35*len(plot))),sharey=True,gridspec_kw={"width_ratios":[2.2,1]})
            y=np.arange(len(plot));colors=[MODEL_COLORS.get(m,"#555555") if m in MODELS else {"gpt55_teacher":"#7B3294","baseline_modal":"#F1A340","baseline_null":"#999999"}.get(m,"#777") for m in plot.model]
            for yi,(_,r),color in zip(y,plot.iterrows(),colors):
                ax.errorbar(r["mean"],yi,xerr=[[max(0,r["mean"]-r.low95)],[max(0,r.high95-r["mean"])]],fmt="none",ecolor=color,alpha=.8)
            ax.scatter(plot["mean"],y,c=colors,s=35);bx.scatter(plot.proportion_ge_3,y,c=colors,s=35)
            teacher=plot.loc[plot.model.eq("gpt55_teacher"),"mean"];modal=plot.loc[plot.model.eq("baseline_modal"),"mean"]
            if len(teacher):ax.axvline(teacher.iloc[0],color="#7B3294",ls="--",label="GPT-5.5 teacher mean")
            if len(modal):ax.axvline(modal.iloc[0],color="#F1A340",ls=":",label="Modal baseline mean")
            ax.set_yticks(y,plot.stream_display_name);ax.set_xlim(0,4);bx.set_xlim(0,1)
            ax.set(xlabel="Mean judge score with 95% cluster-bootstrap interval",title="Mean and interval")
            bx.set(xlabel="Proportion scoring >=3",title="High-score proportion")
            fig.suptitle(f"Reference ladder — {short.replace('_',' ')} (285-row evaluation pool)")
            self._figure(self.p.figures/f"19A_{short}_reference_ladder.png",fig,data=plot,key="Colour identifies model/reference stream; dashed purple and dotted amber lines mark teacher and modal-baseline means.",sources="07 teacher/baseline judgments and existing 03/07 candidate judgments")

    def _retention(self,data):
        rows=[]
        modal=self._stream_rows(data,"baseline_modal")
        for model in MODELS:
            for score in SCORE_COLUMNS:
                frames={"modal":modal,"untreated":self._stream_rows(data,model,"untreated","BF16 HF"),
                        "bf16_full":self._stream_rows(data,model,"full","BF16 HF"),
                        "onebit_full":self._stream_rows(data,model,"full","one-bit frontier")}
                joined=None
                for name,g in frames.items():
                    part=g[["prompt_id","group_id",score+"_numeric"]].dropna().drop_duplicates("prompt_id").rename(columns={score+"_numeric":name})
                    joined=part if joined is None else joined.merge(part[["prompt_id",name]],on="prompt_id")
                if joined is None or joined.empty:
                    rows.append(dict(model=model,dimension=score,dimension_short=SHORT_SCORE[score],n=0,independent_groups=0,
                        retention_estimable=False,retention_estimability_reason="not estimable: no exact four-stream common support"));continue
                point,lo,hi=self._bootstrap_retention(joined)
                gain_positive=bool(np.isfinite(point[2]) and point[2]>0)
                gain_lower_positive=bool(np.isfinite(lo[2]) and lo[2]>0)
                estimable=gain_positive and gain_lower_positive and np.isfinite(point[4])
                reason=("estimated: BF16 SFT gain over modal is positive and its 95% lower bound exceeds zero" if estimable else
                        "not estimable: BF16 SFT gain over modal is not positive with a 95% lower bound above zero")
                rows.append(dict(model=model,dimension=score,dimension_short=SHORT_SCORE[score],n=len(joined),independent_groups=joined.group_id.nunique(),
                    retained_gap=point[0],retained_gap_low95=lo[0],retained_gap_high95=hi[0],deployment_loss=point[1],deployment_loss_low95=lo[1],deployment_loss_high95=hi[1],
                    bf16_sft_gain_over_modal=point[2],bf16_sft_gain_low95=lo[2],bf16_sft_gain_high95=hi[2],
                    bf16_untreated_minus_modal=point[3],retention_index=point[4] if estimable else np.nan,
                    retention_low95=lo[4] if estimable else np.nan,retention_high95=hi[4] if estimable else np.nan,
                    retention_estimable=estimable,retention_estimability_reason=reason,
                    comparison="one-bit SFT-full retention relative to BF16 SFT-full above modal baseline"))
        result=self._write(pd.DataFrame(rows),"retention_index.csv")
        for score in SCORE_COLUMNS:
            p=result[result.dimension.eq(score)].set_index("model").reindex(MODELS).reset_index();y=np.arange(len(p));colors=[MODEL_COLORS[m] for m in p.model]
            fig,axes=plt.subplots(1,4,figsize=(18,6),sharey=True,gridspec_kw={"width_ratios":[1.6,1,1,1]})
            for yi,(_,r),color in zip(y,p.iterrows(),colors):
                if r.get("retention_estimable",False) is True or str(r.get("retention_estimable",False)).lower()=="true":
                    axes[0].errorbar(r.retention_index,yi,xerr=[[max(0,r.retention_index-r.retention_low95)],[max(0,r.retention_high95-r.retention_index)]],fmt="none",ecolor=color)
                    axes[0].scatter(r.retention_index,yi,color=color)
                else:axes[0].scatter(0,yi,color=color,marker="x")
            axes[0].axvline(0,color="black",lw=.7);axes[0].axvline(1,color="black",ls="--",lw=.8);axes[0].set_title("Retention index R")
            for ax,col,title in zip(axes[1:],["deployment_loss","retained_gap","bf16_untreated_minus_modal"],["Lost in deployment","Retained above modal","BF16 untreated − modal"]):
                ax.scatter(p[col],y,c=colors);ax.axvline(0,color="black",lw=.7);ax.set_title(title)
            axes[0].set_yticks(y,[MODEL_LABELS[m] for m in p.model]);fig.suptitle(f"Retention and gap decomposition — {SHORT_SCORE[score].replace('_',' ')}")
            self._figure(self.p.figures/f"19B_{SHORT_SCORE[score]}_retention_and_gap_decomposition.png",fig,data=p,key="The first column is unbounded R with 95% interval; aligned columns show deployment loss, retained gain above modal, and untreated context gap.",sources="retention_index.csv")
        return result

    def _size_extension(self,data,retention):
        rows=[];core=[("BF16 HF","untreated"),("BF16 HF","full"),("one-bit frontier","untreated"),("one-bit frontier","full")]
        for rep,cond in core:
            for score in SCORE_COLUMNS:
                pair=self._paired(self._stream_rows(data,"qwen7b",cond,rep),self._stream_rows(data,"qwen1b",cond,rep),score)
                rows.append(self._paired_summary(pair,dict(record_type="primary_paired_qwen",population="evaluation_285",model_large="qwen7b",model_small="qwen1b",representation_group=rep,condition=cond,dimension=score,dimension_short=SHORT_SCORE[score])))
        # Untreated Qwen family size sensitivity on the complete available
        # universe. Adapter, teacher and baseline streams do not exist there.
        for rep in ("BF16 HF","one-bit frontier"):
            for score in SCORE_COLUMNS:
                large=self._stream_rows(self.long,"qwen7b","untreated",rep)
                small=self._stream_rows(self.long,"qwen1b","untreated",rep)
                pair=self._paired(large,small,score)
                rows.append(self._paired_summary(pair,dict(record_type="primary_paired_qwen",population="all_990",model_large="qwen7b",model_small="qwen1b",representation_group=rep,condition="untreated",dimension=score,dimension_short=SHORT_SCORE[score])))
        for model in MODELS:
            for rep,cond in core:
                g=self._stream_rows(data,model,cond,rep)
                for score in SCORE_COLUMNS:
                    values=g[score+"_numeric"].dropna();rows.append(dict(record_type="secondary_suite_point",population="evaluation_285",model=model,parameter_count=MODEL_PARAMETERS[model],log10_parameter_count=np.log10(MODEL_PARAMETERS[model]),representation_group=rep,condition=cond,dimension=score,dimension_short=SHORT_SCORE[score],n=len(values),mean_score=values.mean()))
            for _,r in retention[retention.model.eq(model)].iterrows():rows.append(dict(record_type="secondary_retention_point",population="evaluation_285",model=model,parameter_count=MODEL_PARAMETERS[model],log10_parameter_count=np.log10(MODEL_PARAMETERS[model]),representation_group="deployment_stack",condition="SFT-full",dimension=r.dimension,dimension_short=SHORT_SCORE.get(r.dimension,r.dimension),n=r.n,retention_index=r.retention_index))
        result=self._write(pd.DataFrame(rows),"size_contrasts.csv")
        primary=result[(result.record_type.eq("primary_paired_qwen"))&result.population.eq("evaluation_285")]
        fig,axes=plt.subplots(4,2,figsize=(15,18));
        for ax,score in zip(axes.ravel(),SCORE_COLUMNS):
            p=primary[primary.dimension.eq(score)].copy();p["label"]=p.representation_group+" | "+p.condition;y=np.arange(len(p))
            ax.hlines(y,-p.mde,p.mde,color="#BBBBBB",lw=6,label="Approximate MDE around zero")
            for yi,(_,r) in zip(y,p.iterrows()):ax.errorbar(r.mean_difference,yi,xerr=[[max(0,r.mean_difference-r.low95)],[max(0,r.high95-r.mean_difference)]],fmt="o",color="#0072B2")
            ax.axvline(0,color="black",lw=.8);ax.set_yticks(y,p.label);ax.set_title(SHORT_SCORE[score].replace("_"," "))
        axes.ravel()[-1].axis("off");fig.suptitle("Qwen2.5 7B minus Qwen2.5 1.5B paired size contrasts")
        self._figure(self.p.figures/"16C_qwen_family_size_contrasts.png",fig,data=primary,key="Blue point/interval is 7B−1.5B; grey bar centred on zero is the approximate 80%-power MDE.",sources="size_contrasts.csv")
        secondary=result[result.record_type.eq("secondary_suite_point")]
        fig,axes=plt.subplots(4,2,figsize=(15,18));stream_markers={("BF16 HF","untreated"):"o",("BF16 HF","full"):"s",("one-bit frontier","untreated"):"^",("one-bit frontier","full"):"D"}
        for ax,score in zip(axes.ravel(),SCORE_COLUMNS):
            p=secondary[secondary.dimension.eq(score)]
            for _,r in p.iterrows():ax.scatter(r.log10_parameter_count,r.mean_score,color=MODEL_COLORS.get(r.model,"#777"),marker=stream_markers[(r.representation_group,r.condition)],alpha=.75)
            ax.set_ylim(0,4);ax.set(xlabel="log10(parameter count)",ylabel="Mean judge score",title=SHORT_SCORE[score].replace("_"," "))
        axes.ravel()[-1].axis("off");fig.suptitle("Supplementary suite-level size associations (descriptive; six models)")
        self._figure(self.p.figures/"S16C_suite_size_score_associations.png",fig,data=secondary,key="Colour identifies model; circle/square are BF16 untreated/full and triangle/diamond are one-bit untreated/full. No regression p-values are reported.",sources="size_contrasts.csv")
        rp=result[result.record_type.eq("secondary_retention_point")]
        fig,axes=plt.subplots(4,2,figsize=(15,18))
        for ax,score in zip(axes.ravel(),SCORE_COLUMNS):
            p=rp[rp.dimension.eq(score)]
            ax.scatter(p.log10_parameter_count,p.retention_index,c=[MODEL_COLORS.get(m,"#777") for m in p.model]);ax.axhline(0,color="black",lw=.7);ax.axhline(1,color="black",ls="--",lw=.7);ax.set(xlabel="log10(parameter count)",ylabel="Retention index",title=SHORT_SCORE[score].replace("_"," "))
        axes.ravel()[-1].axis("off");fig.suptitle("Supplementary suite-level size and retention associations")
        self._figure(self.p.figures/"S16D_suite_size_retention_associations.png",fig,data=rp,key="Colour identifies model; horizontal lines mark zero and full retention. This is descriptive across six heterogeneous architectures.",sources="size_contrasts.csv")
        return result

    def _distillation_fidelity(self,data):
        teacher=self._stream_rows(data,"gpt55_teacher");rows=[]
        for model in MODELS:
            for rep in ("BF16 HF","one-bit frontier"):
                for cond in CONDITIONS[1:]:
                    candidate=self._stream_rows(data,model,cond,rep)
                    for score in SCORE_COLUMNS:
                        pair=self._paired(candidate,teacher,score)
                        rows.append(self._paired_summary(pair,dict(model=model,representation_group=rep,condition=cond,dimension=score,dimension_short=SHORT_SCORE[score],contrast="SFT minus GPT-5.5 teacher")))
        return self._write(pd.DataFrame(rows),"distillation_fidelity.csv")

    def _mde_table(self,data,size,fidelity,retention):
        rows=[];existing=pd.read_csv(self.p.tables/"treatment_contrasts.csv") if (self.p.tables/"treatment_contrasts.csv").exists() else pd.DataFrame()
        ladder=pd.read_csv(self.p.tables/"reference_ladder.csv") if (self.p.tables/"reference_ladder.csv").exists() else pd.DataFrame()
        anchors={}
        for score in SCORE_COLUMNS:
            anchors[score]={name:(pd.to_numeric(ladder[(ladder.model.eq(name))&(ladder.dimension.eq(score))].get("mean"),errors="coerce").dropna().iloc[0]
                                  if len(pd.to_numeric(ladder[(ladder.model.eq(name))&(ladder.dimension.eq(score))].get("mean"),errors="coerce").dropna()) else np.nan)
                            for name in ("baseline_modal","baseline_null")}
        score_cols=[s+"_numeric" for s in SCORE_COLUMNS]
        stream_cache={}
        def wide(source,model,condition,rep):
            key=(id(source),model,condition,rep)
            if key not in stream_cache:stream_cache[key]=self._stream_rows(source,model,condition,rep)[["prompt_id","group_id",*score_cols]].drop_duplicates("prompt_id")
            return stream_cache[key]
        def add_merged(merged,meta,contrast_name,score):
            valid=merged[["group_id_left",score+"_numeric_left",score+"_numeric_right"]].dropna();d=valid[score+"_numeric_left"].astype(float)-valid[score+"_numeric_right"].astype(float);gm=d.groupby(valid.group_id_left).mean();sigma=float(gm.std(ddof=1)) if len(gm)>1 else np.nan;mde=(1.96+.8416)*sigma/math.sqrt(len(gm)) if len(gm)>1 else np.nan;needed=math.ceil(((1.96+.8416)*sigma/NONINFERIORITY_MARGIN)**2) if np.isfinite(sigma) else np.nan
            prior=existing[(existing.model.eq(meta["model"]))&(existing.dimension.eq(meta["dimension"]))&(existing.contrast.eq(contrast_name))] if len(existing) else pd.DataFrame()
            if len(prior) and "population" in prior:prior=prior[prior.population.eq(meta["population"])]
            if len(prior) and contrast_name!="untreated_one_bit_minus_bf16":prior=prior[prior.adapter.eq(meta["condition"])]
            lo=prior.low95.iloc[0] if len(prior) else np.nan;hi=prior.high95.iloc[0] if len(prior) else np.nan
            raw_noninferior=bool(np.isfinite(lo) and lo>-NONINFERIORITY_MARGIN)
            deployment=contrast_name in {"untreated_one_bit_minus_bf16","sft_one_bit_minus_bf16"}
            reference_mean=valid[score+"_numeric_right"].mean() if len(valid) else np.nan
            modal=anchors[score]["baseline_modal"];null=anchors[score]["baseline_null"]
            below_modal=bool(deployment and np.isfinite(reference_mean) and np.isfinite(modal) and reference_mean<=modal)
            below_null=bool(deployment and np.isfinite(reference_mean) and np.isfinite(null) and reference_mean<=null)
            below_margin=bool(deployment and np.isfinite(reference_mean) and reference_mean<=NONINFERIORITY_MARGIN)
            available=[x for x in (modal,null) if np.isfinite(x)];floor_anchor=max(available) if available else np.nan
            baseline_floor=bool(deployment and np.isfinite(reference_mean) and np.isfinite(floor_anchor) and reference_mean<=floor_anchor)
            floor_limited=bool(baseline_floor or below_margin);support_sufficient=bool(len(valid)>=MIN_NONINFERIORITY_N)
            status=("insufficient_support_noninferiority_not_interpretable" if not support_sufficient else
                    "at_floor_noninferiority_not_quality_preservation" if raw_noninferior and floor_limited else
                    "noninferior_above_reference_floor" if raw_noninferior else "noninferiority_not_established")
            rows.append(dict(meta,n=len(valid),independent_groups=len(gm),sigma_g=sigma,mde=mde,delta=NONINFERIORITY_MARGIN,groups_needed_delta_0_3=needed,low95=lo,high95=hi,
                noninferior_delta_0_3=raw_noninferior,bf16_reference_mean=reference_mean if deployment else np.nan,
                modal_baseline_mean=modal if deployment else np.nan,null_baseline_mean=null if deployment else np.nan,
                reference_at_or_below_modal_baseline=below_modal,reference_at_or_below_null_baseline=below_null,reference_at_or_below_margin=below_margin,
                baseline_floor_flag=baseline_floor,reference_floor_flag=floor_limited,minimum_noninferiority_n=MIN_NONINFERIORITY_N,
                support_sufficient_for_noninferiority=support_sufficient,
                noninferior_above_reference_floor=bool(raw_noninferior and not floor_limited and support_sufficient),
                noninferiority_interpretation=status,floor_anchor_population="evaluation_285_reference_baselines" if deployment else "not_applicable"))
        for model in MODELS:
            for rep in ("BF16 HF","one-bit frontier"):
                base=wide(data,model,"untreated",rep)
                for cond in CONDITIONS[1:]:
                    merged=wide(data,model,cond,rep).merge(base,on="prompt_id",suffixes=("_left","_right"))
                    for score in SCORE_COLUMNS:add_merged(merged,dict(population="evaluation_285",contrast_family="SFT versus untreated",model=model,representation_group=rep,condition=cond,dimension=score),"bf16_sft_minus_untreated" if rep=="BF16 HF" else "one_bit_sft_minus_untreated",score)
            for cond in CONDITIONS:
                merged=wide(data,model,cond,"one-bit frontier").merge(wide(data,model,cond,"BF16 HF"),on="prompt_id",suffixes=("_left","_right"))
                for score in SCORE_COLUMNS:add_merged(merged,dict(population="evaluation_285",contrast_family="one-bit versus BF16 deployment stack",model=model,representation_group="deployment_stack",condition=cond,dimension=score),"untreated_one_bit_minus_bf16" if cond=="untreated" else "sft_one_bit_minus_bf16",score)
            # Full-universe untreated deployment sensitivity.
            merged=wide(self.long,model,"untreated","one-bit frontier").merge(wide(self.long,model,"untreated","BF16 HF"),on="prompt_id",suffixes=("_left","_right"))
            for score in SCORE_COLUMNS:add_merged(merged,dict(population="all_990",contrast_family="one-bit versus BF16 deployment stack",model=model,representation_group="deployment_stack",condition="untreated",dimension=score),"untreated_one_bit_minus_bf16",score)
        for _,r in size[size.record_type.eq("primary_paired_qwen")].iterrows():rows.append(dict(population=r.get("population","evaluation_285"),contrast_family="Qwen family size",model="qwen7b_minus_qwen1b",representation_group=r.representation_group,condition=r.condition,dimension=r.dimension,n=r.n,independent_groups=r.independent_groups,sigma_g=r.sigma_g,mde=r.mde,delta=NONINFERIORITY_MARGIN,groups_needed_delta_0_3=r.groups_needed_delta_0_3,low95=r.low95,high95=r.high95,noninferior_delta_0_3=r.noninferior_delta_0_3,reference_floor_flag=False,noninferiority_interpretation="not_applicable_size_contrast"))
        for _,r in fidelity.iterrows():rows.append(dict(population="evaluation_285",contrast_family="SFT versus teacher",model=r.model,representation_group=r.representation_group,condition=r.condition,dimension=r.dimension,n=r.n,independent_groups=r.independent_groups,sigma_g=r.sigma_g,mde=r.mde,delta=NONINFERIORITY_MARGIN,groups_needed_delta_0_3=r.groups_needed_delta_0_3,low95=r.low95,high95=r.high95,noninferior_delta_0_3=r.noninferior_delta_0_3,reference_floor_flag=False,noninferiority_interpretation="not_applicable_teacher_contrast"))
        self._write(pd.DataFrame(rows),"mde_and_noninferiority.csv")

    # ---------- Sections 2 and 3: compute ----------
    def _compute_source_aggregates(self):
        """Rebuild stream/phase resources from Notebook 9's source-level exports.

        Notebook 9 summaries are retained for reconciliation, but figures use
        invocation and row records so restore, preparation, inference and
        retry costs remain separately auditable.
        """
        inv = self.compute_invocations.copy()
        rows = self.compute_rows.copy()
        coverage = self.compute_coverage.copy()
        if inv.empty:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        if "telemetry_sha256" in inv:
            inv = inv.drop_duplicates("telemetry_sha256", keep="last")
        inv["condition_short"] = inv["condition"].map(condition_short)
        inv["representation_group"] = inv["representation"].map(representation_group)
        inv["representation_label"] = inv["representation"].map(representation_label)
        inv["stream_display_name"] = [stream_display_name(m, r, c) for m,r,c in zip(inv.model, inv.representation, inv.condition_short)]
        numeric = ["wall_seconds","model_load_seconds","model_cache_restore_download_sync_seconds",
                   "training_wall_seconds","preparation_and_upload_wall_seconds","inference_and_io_wall_seconds",
                   "generation_seconds","local_checkpoint_seconds","receiver_seconds","csv_snapshot_seconds",
                   "input_tokens","output_tokens","new_rows","sampled_seconds","gpu_utilization_weighted_seconds",
                   "mean_gpu_utilization_pct","sampled_peak_gpu_memory_bytes","peak_allocated_bytes",
                   "peak_reserved_bytes","sampled_peak_server_rss_bytes","estimated_energy_wh"]
        for col in numeric:
            inv[col] = pd.to_numeric(inv.get(col), errors="coerce")
            inv[col+"_observed"] = inv[col].notna()
        inv["is_pilot"] = inv["invocation_kind"].astype(str).str.contains("pilot",case=False,na=False)
        inv["completed_invocation"] = inv["status"].astype(str).str.lower().eq("completed")
        keys=["model","condition_short","representation_group","representation_label","phase","stream_display_name"]
        records=[]
        for group, g in inv.groupby(keys,dropna=False):
            sampled=g.sampled_seconds.sum(min_count=1)
            util=(g.gpu_utilization_weighted_seconds.sum(min_count=1)/sampled) if sampled and sampled>0 else np.nan
            rec=dict(zip(keys,group))
            rec.update(invocation_count=len(g),pilot_invocation_count=int(g.is_pilot.sum()),
                       completed_invocations=int(g.completed_invocation.sum()),
                       noncompleted_invocations=int((~g.completed_invocation).sum()),
                       resource_wall_seconds=g.wall_seconds.sum(min_count=1),
                       gpu_runtime_session_hours=g.wall_seconds.sum(min_count=1)/3600,
                       mean_invocation_minutes=g.wall_seconds.mean()/60,
                       time_weighted_gpu_utilization_pct=util,
                       gpu_types=" | ".join(sorted(set(g.gpu.dropna().astype(str)))))
            for col in numeric:
                if col not in {"wall_seconds","mean_gpu_utilization_pct","gpu_utilization_weighted_seconds"}:
                    rec[col] = g[col].max() if col in {"sampled_peak_gpu_memory_bytes","peak_allocated_bytes","peak_reserved_bytes","sampled_peak_server_rss_bytes"} else g[col].sum(min_count=1)
                rec[col+"_observed"] = bool(g[col].notna().any())
            records.append(rec)
        phase=pd.DataFrame(records)

        stream_keys=["model","condition_short","representation_group","representation_label","stream_display_name"]
        streams=[]
        for group,g in phase.groupby(stream_keys,dropna=False):
            rec=dict(zip(stream_keys,group))
            rec.update(invocation_count=g.invocation_count.sum(), completed_invocations=g.completed_invocations.sum(),
                       noncompleted_invocations=g.noncompleted_invocations.sum(),
                       total_gpu_runtime_session_hours=g.gpu_runtime_session_hours.sum(min_count=1),
                       training_session_hours=g.loc[g.phase.astype(str).str.contains("train",case=False),"gpu_runtime_session_hours"].sum(),
                       preparation_session_hours=g.loc[g.phase.astype(str).str.contains("prepar",case=False),"gpu_runtime_session_hours"].sum(),
                       inference_session_hours=g.loc[g.phase.astype(str).str.contains("infer",case=False),"gpu_runtime_session_hours"].sum(),
                       model_load_seconds=g.model_load_seconds.sum(min_count=1),
                       restore_download_seconds=g.model_cache_restore_download_sync_seconds.sum(min_count=1),
                       preparation_upload_seconds=g.preparation_and_upload_wall_seconds.sum(min_count=1),
                       inference_io_seconds=g.inference_and_io_wall_seconds.sum(min_count=1),
                       generation_seconds=g.generation_seconds.sum(min_count=1),
                       local_checkpoint_seconds=g.local_checkpoint_seconds.sum(min_count=1),
                       receiver_seconds=g.receiver_seconds.sum(min_count=1),csv_snapshot_seconds=g.csv_snapshot_seconds.sum(min_count=1),
                       sampled_seconds=g.sampled_seconds.sum(min_count=1),
                       sampled_peak_gpu_memory_bytes=g.sampled_peak_gpu_memory_bytes.max(),
                       peak_allocated_bytes=g.peak_allocated_bytes.max(),peak_reserved_bytes=g.peak_reserved_bytes.max(),
                       estimated_energy_wh=g.estimated_energy_wh.sum(min_count=1),
                       gpu_types=" | ".join(sorted(set(" | ".join(g.gpu_types.dropna().astype(str)).split(" | ")))))
            denom=g.sampled_seconds.sum(min_count=1)
            rec["time_weighted_gpu_utilization_pct"]=(g.time_weighted_gpu_utilization_pct*g.sampled_seconds).sum()/denom if denom and denom>0 else np.nan
            streams.append(rec)
        stream=pd.DataFrame(streams)
        if not coverage.empty:
            c=coverage.copy();c["condition_short"]=c.condition.map(condition_short);c["representation_group"]=c.representation.map(representation_group);c["representation_label"]=c.representation.map(representation_label)
            keep=["model","condition_short","representation_group","representation_label","record_count","completed","context_unsupported","retryable_error","other_status","total_attempts","total_retries","completed_input_tokens","completed_output_tokens","completed_elapsed_seconds","completed_output_tokens_per_request_work_second"]
            stream=stream.merge(c[keep],on=["model","condition_short","representation_group","representation_label"],how="outer")
        # Rebuild reader-facing names from the final normalized identity.  Do
        # not carry a label from an earlier invocation/phase aggregation: that
        # previously allowed BF16 rows to inherit a one-bit display label.
        stream["stream_display_name"] = [stream_display_name(m, r, c) for m,r,c in
                                         zip(stream.model, stream.representation_group, stream.condition_short)]
        # Coverage is defined against the experimental design, not merely the
        # rows that reached the response store.  Untreated streams target all
        # 990 prompts; every SFT stream targets the 285-row evaluation pool.
        # A row with no saved record is therefore explicit `not_reached`, not
        # silently removed from the denominator.
        stream["intended_rows"] = np.where(stream.condition_short.eq("untreated"), 990, 285)
        recorded = pd.to_numeric(stream.get("record_count"), errors="coerce").fillna(0)
        stream["not_reached"] = (stream["intended_rows"] - recorded).clip(lower=0).astype(int)
        hours=stream.inference_session_hours.replace(0,np.nan)
        stream["completed_rows_per_inference_session_hour"]=stream.completed/hours
        stream["generated_tokens_per_inference_session_hour"]=stream.completed_output_tokens/hours
        stream["coverage_rate"]=stream.completed/stream.intended_rows.replace(0,np.nan)

        # Reconcile the source-derived wall time against Notebook 9 summaries.
        old=self.compute_streams.copy()
        recon=pd.DataFrame()
        if not old.empty:
            old_total=pd.to_numeric(old.get("experimental_resource_wall_seconds"),errors="coerce").sum()
            recon=pd.DataFrame([{"source_level_wall_seconds":inv.wall_seconds.sum(min_count=1),
                                 "notebook09_summary_wall_seconds":old_total,
                                 "difference_seconds":inv.wall_seconds.sum(min_count=1)-old_total,
                                 "note":"Differences reflect source de-duplication and Notebook 9 aggregation scope."}])
        return inv,phase,stream,recon

    def _training_loss_history(self):
        # Training logs can be replaced by a later resumability check that only
        # says "adapter already complete".  Search both the live log directory
        # and immutable recovered logs, then retain the richest history for each
        # model/condition.  This prevents a later no-op run from erasing an
        # earlier, complete Trainer loss history from the analysis view.
        candidates=[]
        log_root=self.p.root/"telemetry"/"logs"
        for path in sorted(log_root.rglob("*_train_*_all.log")):
            match=re.match(r"(.+?)_train_(technique|explanation|joint|full)_all\.log$",path.name)
            if not match: continue
            model,condition=match.groups();observations=[]
            for line in path.read_text(encoding="utf-8",errors="replace").splitlines():
                text=line.strip()
                if not (text.startswith("{") and "'loss'" in text): continue
                try: record=ast.literal_eval(text)
                except (SyntaxError,ValueError): continue
                if "loss" in record:
                    observations.append(record)
            candidates.append((model,condition,path,observations))
        rows=[]
        grouped={}
        for model,condition,path,observations in candidates:
            key=(model,condition)
            previous=grouped.get(key)
            # Prefer more observations; on a tie prefer the recovered immutable
            # copy so the selected provenance is stable across reruns.
            rank=(len(observations),"recovered" in path.parts)
            if previous is None or rank>previous[0]: grouped[key]=(rank,path,observations)
        for (model,condition),(_,path,observations) in sorted(grouped.items()):
            source=str(path.relative_to(self.p.root))
            provenance="git_recovered" if "recovered" in path.parts else "working_log"
            for step,record in enumerate(observations,1):
                rows.append({"model":model,"condition":condition,"step":step,"epoch":record.get("epoch"),"loss":record.get("loss"),"learning_rate":record.get("learning_rate"),"source_log":source,"source_provenance":provenance})
        return pd.DataFrame(rows)

    def _artifact_inventory(self):
        rows=[]
        for path in sorted((self.p.root/"models"/"quantized").rglob("artifact.json")):
            try:r=json.loads(path.read_text(encoding="utf-8"))
            except (OSError,json.JSONDecodeError):continue
            rows.append({"model":r.get("model"),"condition":condition_short(r.get("adapter") or "untreated"),"representation":r.get("representation"),"representation_label":representation_label(r.get("representation")),"representation_group":"one-bit frontier","artifact_bytes":r.get("artifact_bytes"),"artifact_gib":pd.to_numeric(r.get("artifact_bytes"),errors="coerce")/2**30,"artifact_sha256":r.get("artifact_sha256"),"source_revision":r.get("source_revision"),"toolchain_commit":r.get("toolchain_commit"),"preparation_seconds_manifest":r.get("preparation_seconds"),"quantization_tensor_counts":json.dumps(r.get("quantization_tensor_counts",{}),sort_keys=True),"manifest_path":str(path.relative_to(self.p.root))})
        return pd.DataFrame(rows)

    def compute_reporting(self):
        inv,phase,streams,recon=self._compute_source_aggregates()
        self.compute_invocations_derived=inv;self.compute_phases_derived=phase;self.compute_streams_derived=streams
        self._write(inv,"compute_invocation_audit.csv");self._write(phase,"compute_phase_resource_summary.csv");self._write(streams,"compute_stream_resource_summary.csv");self._write(recon,"compute_source_reconciliation.csv")
        losses=self._write(self._training_loss_history(),"training_loss_history.csv")
        artifacts=self._write(self._artifact_inventory(),"quantized_artifact_inventory.csv")
        train=phase[phase.phase.astype(str).str.contains("train",case=False,na=False)].copy();quant=phase[phase.phase.astype(str).str.contains("prepar",case=False,na=False)].copy();infer=phase[phase.phase.astype(str).str.contains("infer",case=False,na=False)].copy()
        self._write(train,"compute_sft_training_report.csv");self._write(quant,"compute_quantization_report.csv");self._write(infer,"compute_inference_report.csv");self._write(self.api_usage,"compute_api_report.csv")
        self._compute_reporting_figures(train,quant,infer,self._compute_population_rows(streams),losses,artifacts)

    def _compute_population_rows(self,streams):
        """Attach request-record rates for the full universe and fixed 285 pool.

        Session-level resource totals remain full-stream observations and are
        repeated only as provenance. Population-specific rates use completed
        row records; elapsed request work is divided by recorded parallel slots
        (or one when absent). It is not relabelled as captured session time.
        """
        rows=self.compute_rows.copy()
        if streams.empty:return streams
        if rows.empty:
            out=streams.copy();out["population"]="all_990";return out
        rows["condition_short"]=rows.condition.map(condition_short);rows["representation_group"]=rows.representation.map(representation_group);rows["representation_label"]=rows.representation.map(representation_label)
        for col in ("elapsed_seconds","parallel_slots","input_tokens","output_tokens","peak_allocated_bytes","peak_reserved_bytes"):rows[col]=pd.to_numeric(rows.get(col),errors="coerce")
        rows=rows[rows.status.astype(str).str.lower().eq("completed")].copy();rows["effective_slots"]=rows.parallel_slots.fillna(1).clip(lower=1);rows["slot_adjusted_seconds"]=rows.elapsed_seconds/rows.effective_slots
        keys=["model","condition_short","representation_group","representation_label"]
        views=[]
        for population,pop in self._population_views(rows):
            agg=pop.groupby(keys,dropna=False).agg(population_completed_rows=("prompt_id","nunique"),population_request_work_seconds=("elapsed_seconds","sum"),population_slot_adjusted_request_seconds=("slot_adjusted_seconds","sum"),population_median_row_latency_seconds=("elapsed_seconds","median"),population_input_tokens=("input_tokens","sum"),population_output_tokens=("output_tokens","sum"),population_peak_allocated_bytes=("peak_allocated_bytes","max"),population_peak_reserved_bytes=("peak_reserved_bytes","max")).reset_index()
            agg["population"]=population;den=agg.population_slot_adjusted_request_seconds.replace(0,np.nan)
            agg["population_rows_per_slot_adjusted_request_hour"]=agg.population_completed_rows/(den/3600)
            agg["population_prompt_tokens_per_slot_adjusted_request_hour"]=agg.population_input_tokens/(den/3600)
            agg["population_output_tokens_per_slot_adjusted_request_hour"]=agg.population_output_tokens/(den/3600)
            merged=streams.merge(agg,on=keys,how="left");merged["population"]=population
            merged["population_resource_scope"]="population rates from row records; session resource totals describe the full exact stream"
            views.append(merged)
        return pd.concat(views,ignore_index=True,sort=False)

    def _compute_reporting_figures(self,train,quant,infer,streams,losses,artifacts):
        def bar(frame,value,title,path):
            fig,ax=plt.subplots(figsize=(11,max(4,.22*len(frame))))
            x=frame.copy()
            if len(x) and value in x:
                x[value]=pd.to_numeric(x[value],errors="coerce");x["label"]=x.stream_display_name.astype(str)
                if "phase" in x:x["label"]=x["label"]+" / "+x.phase.astype(str)
                x=x.sort_values(value);ax.barh(x.label,x[value],color=[MODEL_COLORS.get(m,"#777") for m in x.model])
            ax.set_title(title);self._figure(self.p.figures/path,fig,data=x,key="Bar colour identifies the model family; each label states representation and treatment.",sources="compute_phase_resource_summary.csv")
        bar(train,"gpu_runtime_session_hours","SFT GPU-runtime session hours","20_training_resource_hours.png")
        # One panel per model prevents 24 histories from being overplotted.
        # Adapter colours are held constant across panels, while panel titles
        # identify the model family.
        fig,axes=plt.subplots(2,3,figsize=(17,10),sharex=True,sharey=False)
        for ax,model in zip(axes.flat,MODELS):
            subset=losses[losses.model.eq(model)]
            for condition in CONDITIONS[1:]:
                g=subset[subset.condition.eq(condition)].sort_values("step")
                if len(g):
                    ax.plot(g.step,g.loss,label=f"SFT-{condition}",color=CONDITION_COLORS[condition],marker=CONDITION_MARKERS[condition],markevery=max(len(g)//8,1),markersize=3,linewidth=1.4)
            ax.set_title(MODEL_LABELS.get(model,model));ax.set_xlabel("Logged training step");ax.set_ylabel("Training loss");ax.grid(alpha=.2)
        # _figure() collects the four line labels from the axes and creates one
        # shared bottom key; avoid a second top legend competing with the title.
        fig.suptitle("Observed SFT training-loss histories")
        self._figure(self.p.figures/"21_training_loss_curves.png",fig,data=losses,key="Each panel is one model; colour and marker identify the SFT adapter consistently across panels. Logged steps are shown without smoothing.",sources="telemetry/logs/*_train_*_all.log; telemetry/logs/recovered/*_train_*_all.log")
        q=quant.rename(columns={"condition_short":"condition"}).merge(artifacts,on=["model","condition","representation_label"],how="outer",suffixes=("_telemetry","_manifest")) if len(artifacts) else quant
        fig,axes=plt.subplots(1,2,figsize=(15,max(8,.28*len(q))),sharey=True);
        if len(q):
            q["label"]=q.model.map(MODEL_LABELS).fillna(q.model)+" / "+q.condition.astype(str);q=q.sort_values("label")
            axes[0].barh(q.label,pd.to_numeric(q.get("resource_wall_seconds"),errors="coerce")/60,color=[MODEL_COLORS.get(m,"#777") for m in q.model]);axes[1].barh(q.label,pd.to_numeric(q.get("artifact_gib"),errors="coerce"),color=[MODEL_COLORS.get(m,"#777") for m in q.model]);axes[1].tick_params(axis="y",labelleft=False)
        axes[0].set(xlabel="Preparation session minutes",title="Preparation time");axes[1].set(xlabel="GGUF GiB",title="Saved artifact size")
        self._figure(self.p.figures/"22_quantization_time_and_artifact_size.png",fig,data=q,key="Matched rows identify the same model/adapter artifact; colour identifies model family.",sources="compute_phase_resource_summary.csv; models/quantized/**/artifact.json")
        bar(phase_all:=pd.concat([train,quant,infer],ignore_index=True),"gpu_runtime_session_hours","Model/phase GPU-runtime session hours","23A_model_phase_resource_hours.png")
        bar(infer,"gpu_runtime_session_hours","Exact-stream inference GPU-runtime session hours","S23B_exact_stream_resource_hours.png")
        shown=streams[streams.population.eq("evaluation_285")]
        for col,path,title in [("population_rows_per_slot_adjusted_request_hour","24A_rows_per_hour.png","Completed rows per slot-adjusted request hour (evaluation pool)"),("population_prompt_tokens_per_slot_adjusted_request_hour","24B_prompt_tokens_per_hour.png","Prompt tokens per slot-adjusted request hour (evaluation pool)"),("population_output_tokens_per_slot_adjusted_request_hour","24C_generated_tokens_per_hour.png","Generated tokens per slot-adjusted request hour (evaluation pool)")]:bar(shown,col,title,path)
        fig,ax=plt.subplots(figsize=(10,6));
        if len(streams):
            for _,r in streams[streams.population.eq("all_990")].iterrows():ax.scatter(r.time_weighted_gpu_utilization_pct,pd.to_numeric(r.sampled_peak_gpu_memory_bytes,errors="coerce")/2**30,color=MODEL_COLORS.get(r.model,"#777"),marker=CONDITION_MARKERS.get(r.condition_short,"o"),label=MODEL_LABELS.get(r.model,r.model))
        ax.set(xlabel="Time-weighted sampled GPU utilization (%)",ylabel="Sampled peak GPU memory (GiB)",title="GPU utilization and memory")
        self._figure(self.p.figures/"25_gpu_utilization_and_memory.png",fig,data=streams,key="Colour identifies model; marker identifies treatment. Values are sampled whole-device telemetry during exclusive runtimes.",sources="compute_stream_resource_summary.csv")
        if len(streams):
            c=streams[streams.population.eq("all_990")].copy();c["label"]=c.stream_display_name;fig,ax=plt.subplots(figsize=(12,max(5,.22*len(c))));left=np.zeros(len(c))
            for col,color in [("completed","#009E73"),("context_unsupported","#E69F00"),("retryable_error","#D55E00"),("other_status","#999999"),("not_reached","#D9D9D9")]:
                vals=pd.to_numeric(c[col],errors="coerce").fillna(0)/pd.to_numeric(c.intended_rows,errors="coerce").replace(0,np.nan);ax.barh(c.label,vals,left=left,color=color,label=col);left+=vals.fillna(0).to_numpy()
            ax.set_xlim(0,1);ax.set_title("Stream coverage (intended denominator)");self._figure(self.p.figures/"26A_stream_coverage.png",fig,data=c,key="Green=completed; amber=context unsupported; red=retryable error; dark grey=other saved status; light grey=no saved record/not reached. Every bar uses 990 untreated or 285 SFT rows as its denominator.",sources="compute_stream_resource_summary.csv")
            bar(c,"total_retries","Retry/resume burden","26B_retry_resume_burden.png")

    @staticmethod
    def _interval_union_seconds(intervals):
        """Return union duration and maximum inferred concurrency.

        Intervals are reconstructed from receiver file modification times and
        row elapsed time.  They are a sensitivity estimate, not authoritative
        request start/end timestamps, because queued receiver uploads can move
        the apparent completion time.
        """
        cleaned=sorted((float(a),float(b)) for a,b in intervals if np.isfinite(a) and np.isfinite(b) and b>=a)
        if not cleaned:return 0.0,0,0
        merged=[];events=[]
        for start,end in cleaned:
            if not merged or start>merged[-1][1]:merged.append([start,end])
            else:merged[-1][1]=max(merged[-1][1],end)
            # End events sort before start events at an identical timestamp.
            events.extend([(start,1),(end,-1)])
        active=maximum=0
        for _,change in sorted(events,key=lambda item:(item[0],item[1])):
            active+=change;maximum=max(maximum,active)
        return sum(end-start for start,end in merged),maximum,len(merged)

    def _runtime_reconciliation(self,streams):
        """Reconcile observed telemetry with row-derived inference time.

        Observed invocation telemetry remains authoritative where it covers a
        stream.  Row reconstruction is used only as a gap audit/fill and is
        never relabelled as measured GPU utilization, memory, energy, loading,
        restoration or idle allocated-runtime time.
        """
        timeout_seconds=1800.0 # frozen llama-server request timeout in continuation/infer.py
        rows=self.compute_rows.copy()
        inv=self.compute_invocations_derived.copy() if hasattr(self,"compute_invocations_derived") else self.compute_invocations.copy()
        if rows.empty:return pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
        rows["condition_short"]=rows.condition.map(condition_short)
        rows["representation_group"]=rows.representation.map(representation_group)
        rows["representation_label"]=rows.representation.map(representation_label)
        for col in ("elapsed_seconds","parallel_slots","attempt_count","record_modified_ns"):
            rows[col]=pd.to_numeric(rows.get(col),errors="coerce")
        rows["attempt_count"]=rows.attempt_count.fillna(1).clip(lower=1)
        # Transformers/PyTorch inference is a serial Python row loop.  Older
        # GGUF records without an explicit slot field also predate parallel
        # scheduling and are conservatively treated as one-slot records.
        rows["effective_slots"]=rows.parallel_slots
        rows["slot_source"]="recorded"
        hf_missing=rows.effective_slots.isna()&rows.representation_group.eq("BF16 HF")
        gguf_missing=rows.effective_slots.isna()&~rows.representation_group.eq("BF16 HF")
        rows.loc[hf_missing,"effective_slots"]=1;rows.loc[hf_missing,"slot_source"]="hf_serial_backend"
        rows.loc[gguf_missing,"effective_slots"]=1;rows.loc[gguf_missing,"slot_source"]="missing_assumed_sequential"
        rows["effective_slots"]=rows.effective_slots.clip(lower=1)
        rows["completed_row"]=rows.status.astype(str).str.lower().eq("completed")
        rows["successful_request_seconds"]=np.where(rows.completed_row,rows.elapsed_seconds.fillna(0),0)
        rows["slot_adjusted_success_seconds"]=rows.successful_request_seconds/rows.effective_slots
        timeout=rows.generation_error.astype(str).str.contains("timed out",case=False,na=False)
        rows["known_timeout_attempts"]=np.where(timeout,rows.attempt_count,0)
        rows["timeout_request_seconds"]=rows.known_timeout_attempts*timeout_seconds
        rows["slot_adjusted_timeout_seconds"]=rows.timeout_request_seconds/rows.effective_slots
        # Earlier attempts on a later-completed row have no duration/type in
        # the terminal record.  Preserve their count as unrecoverable rather
        # than silently treating every retry as a 30-minute timeout.
        rows["unrecoverable_prior_attempts"]=np.where(rows.completed_row,(rows.attempt_count-1).clip(lower=0),0)
        usable=rows.completed_row&rows.elapsed_seconds.gt(0)&rows.record_modified_ns.gt(0)
        rows["inferred_end_epoch_seconds"]=np.where(usable,rows.record_modified_ns/1e9,np.nan)
        rows["inferred_start_epoch_seconds"]=np.where(usable,rows.inferred_end_epoch_seconds-rows.elapsed_seconds,np.nan)
        row_audit_cols=["model","condition","condition_short","representation","representation_group","representation_label","prompt_id","status","attempt_count","generation_error","elapsed_seconds","parallel_slots","effective_slots","slot_source","successful_request_seconds","slot_adjusted_success_seconds","known_timeout_attempts","timeout_request_seconds","slot_adjusted_timeout_seconds","unrecoverable_prior_attempts","record_modified_ns","inferred_start_epoch_seconds","inferred_end_epoch_seconds","record_path"]
        row_audit=rows[[c for c in row_audit_cols if c in rows]].copy()

        # Map observed inference invocation coverage to the same canonical
        # stream identity.  generation_seconds is request work; invocation wall
        # and inference_and_io are elapsed session components.
        inv["condition_short"]=inv.condition.map(condition_short)
        inv["representation_group"]=inv.representation.map(representation_group)
        inv["representation_label"]=inv.representation.map(representation_label)
        inf=inv[inv.phase.astype(str).str.contains("infer",case=False,na=False)].copy()
        ikeys=["model","condition_short","representation_group","representation_label"]
        observed={}
        for key,g in inf.groupby(ikeys,dropna=False):
            observed[key]={"observed_inference_invocations":len(g),
                "observed_inference_session_seconds":pd.to_numeric(g.wall_seconds,errors="coerce").sum(min_count=1),
                "observed_inference_and_io_seconds":pd.to_numeric(g.inference_and_io_wall_seconds,errors="coerce").sum(min_count=1),
                "observed_generation_seconds":pd.to_numeric(g.generation_seconds,errors="coerce").sum(min_count=1),
                "observed_new_rows":pd.to_numeric(g.new_rows,errors="coerce").sum(min_count=1),
                "observed_inference_noncompleted_invocations":int((~g.status.astype(str).str.lower().eq("completed")).sum()),
                "observed_inference_gpus":" | ".join(sorted(set(g.gpu.dropna().astype(str))))}

        records=[]
        stream_lookup={}
        for _,stream_row in streams.iterrows():
            skey=(stream_row.get("model"),stream_row.get("condition_short"),stream_row.get("representation_group"),stream_row.get("representation_label"))
            stream_lookup[skey]=stream_row
        for key,g in rows.groupby(ikeys,dropna=False):
            meta=dict(zip(ikeys,key));obs=observed.get(key,{});stream_meta=stream_lookup.get(key)
            intervals=list(zip(g.inferred_start_epoch_seconds.dropna(),g.loc[g.inferred_start_epoch_seconds.notna(),"inferred_end_epoch_seconds"]))
            union_seconds,max_concurrency,segments=self._interval_union_seconds(intervals)
            completed=int(g.completed_row.sum());record_count=len(g)
            request_seconds=g.successful_request_seconds.sum(min_count=1)
            slot_seconds=g.slot_adjusted_success_seconds.sum(min_count=1)
            timeout_request=g.timeout_request_seconds.sum(min_count=1)
            timeout_slot=g.slot_adjusted_timeout_seconds.sum(min_count=1)
            lower=slot_seconds+timeout_slot
            timestamp_with_timeout=union_seconds+timeout_slot
            upper=max(lower,timestamp_with_timeout)
            max_slots=int(g.effective_slots.max()) if len(g) else 0
            timestamp_rows=int(g.inferred_start_epoch_seconds.notna().sum())
            capacity_pass=bool(max_concurrency<=max_slots) if timestamp_rows else False
            observed_generation=float(obs.get("observed_generation_seconds",0) or 0)
            observed_wall=float(obs.get("observed_inference_session_seconds",0) or 0)
            new_rows=float(obs.get("observed_new_rows",0) or 0)
            row_capture=(observed_generation/request_seconds) if request_seconds else np.nan
            new_row_capture=(new_rows/completed) if completed else np.nan
            intended=990 if meta["condition_short"]=="untreated" else 285
            terminal_records=int(stream_meta.get("record_count",record_count)) if stream_meta is not None and pd.notna(stream_meta.get("record_count")) else record_count
            # Legacy BF16 telemetry did not record generation/new_rows, but
            # its worker is serial and its terminal row ledger is complete.
            # GGUF completeness requires both request-work and new-row capture
            # so one well-covered timer cannot hide a missing invocation.
            bf16_complete=(meta["representation_group"]=="BF16 HF" and terminal_records>=intended)
            gguf_complete=(np.isfinite(row_capture) and np.isfinite(new_row_capture) and row_capture>=.95 and new_row_capture>=.95)
            if observed_wall>0 and (bf16_complete or gguf_complete):method="observed_complete"
            elif observed_wall>0 and lower>0:method="hybrid_gap_filled"
            elif lower>0:method="reconstructed_only"
            else:method="insufficient_for_reconstruction"
            if method=="observed_complete":selected_low=selected_high=observed_wall
            elif method in {"hybrid_gap_filled","reconstructed_only"}:selected_low,selected_high=lower,upper
            else:selected_low=selected_high=np.nan
            records.append(meta|obs|{
                "stream_display_name":stream_display_name(meta["model"],meta["representation_group"],meta["condition_short"]),
                "intended_rows":intended,"record_count":record_count,"terminal_records_for_completeness":terminal_records,"completed_rows":completed,"timestamp_usable_rows":timestamp_rows,
                "configured_slot_values":"|".join(str(int(x)) for x in sorted(set(g.effective_slots.dropna()))),
                "missing_slot_rows":int(g.parallel_slots.isna().sum()),"max_configured_slots":max_slots,
                "successful_request_work_seconds":request_seconds,"slot_adjusted_success_seconds":slot_seconds,
                "known_timeout_attempts":int(g.known_timeout_attempts.sum()),"timeout_request_seconds":timeout_request,
                "slot_adjusted_timeout_seconds":timeout_slot,"unrecoverable_prior_attempts":int(g.unrecoverable_prior_attempts.sum()),
                "timestamp_union_success_seconds":union_seconds,"timestamp_union_segments":segments,
                "max_inferred_concurrency":max_concurrency,"timestamp_capacity_check_pass":capacity_pass,
                "telemetry_generation_capture_ratio":row_capture,"telemetry_new_row_capture_ratio":new_row_capture,
                "accounting_method":method,"reconstructed_inference_low_seconds":lower,
                "reconstructed_inference_timestamp_seconds":timestamp_with_timeout,
                "reconstructed_inference_high_seconds":upper,"selected_inference_low_seconds":selected_low,
                "selected_inference_high_seconds":selected_high,
                "reconstruction_scope":"request processing and known timeout attempts only",
                "unreconstructed_scope":"model loading, restore/upload, idle allocation, unknown prior retry duration, GPU utilization/memory/energy"})
        reconciliation=pd.DataFrame(records)
        numeric_hours=[c for c in reconciliation if c.endswith("_seconds")]
        for col in numeric_hours:reconciliation[col.replace("_seconds","_hours")]=pd.to_numeric(reconciliation[col],errors="coerce")/3600

        validation=reconciliation[reconciliation.accounting_method.eq("observed_complete")].copy()
        validation["slot_adjusted_vs_observed_wall_ratio"]=validation.slot_adjusted_success_seconds/validation.observed_inference_session_seconds.replace(0,np.nan)
        validation["timestamp_union_vs_observed_wall_ratio"]=validation.timestamp_union_success_seconds/validation.observed_inference_session_seconds.replace(0,np.nan)
        validation["request_work_vs_observed_generation_ratio"]=validation.successful_request_work_seconds/validation.observed_generation_seconds.replace(0,np.nan)
        validation["slot_adjusted_absolute_pct_error"]=100*(validation.slot_adjusted_success_seconds-validation.observed_inference_session_seconds).abs()/validation.observed_inference_session_seconds.replace(0,np.nan)

        validation_error=pd.to_numeric(validation.slot_adjusted_absolute_pct_error,errors="coerce")
        bf16_error=pd.to_numeric(validation.loc[validation.representation_group.eq("BF16 HF"),"slot_adjusted_absolute_pct_error"],errors="coerce")
        onebit_error=pd.to_numeric(validation.loc[validation.representation_group.eq("one-bit frontier"),"slot_adjusted_absolute_pct_error"],errors="coerce")

        observed_total=pd.to_numeric(streams.total_gpu_runtime_session_hours,errors="coerce").sum(min_count=1)
        observed_inference=pd.to_numeric(streams.inference_session_hours,errors="coerce").sum(min_count=1)
        noninference=observed_total-observed_inference
        selected_low=pd.to_numeric(reconciliation.selected_inference_low_seconds,errors="coerce").sum(min_count=1)/3600
        selected_high=pd.to_numeric(reconciliation.selected_inference_high_seconds,errors="coerce").sum(min_count=1)/3600
        project=pd.DataFrame([{"captured_gpu_runtime_session_hours":observed_total,
            "captured_inference_session_hours":observed_inference,"captured_noninference_session_hours":noninference,
            "reconciled_inference_wall_hours_low":selected_low,"reconciled_inference_wall_hours_high":selected_high,
            "hybrid_resource_time_hours_low":noninference+selected_low,"hybrid_resource_time_hours_high":noninference+selected_high,
            "observed_complete_streams":int(reconciliation.accounting_method.eq("observed_complete").sum()),
            "hybrid_gap_filled_streams":int(reconciliation.accounting_method.eq("hybrid_gap_filled").sum()),
            "reconstructed_only_streams":int(reconciliation.accounting_method.eq("reconstructed_only").sum()),
            "insufficient_streams":int(reconciliation.accounting_method.eq("insufficient_for_reconstruction").sum()),
            "validation_streams":len(validation),
            "validation_median_absolute_pct_error":validation_error.median(),
            "bf16_validation_median_absolute_pct_error":bf16_error.median(),
            "one_bit_validation_median_absolute_pct_error":onebit_error.median(),
            "timeout_seconds_assumption":timeout_seconds,
            "interpretation":"Captured total is observed invocation wall time. Hybrid range replaces incomplete inference components with row-derived request-processing time; it is not billing time or active GPU compute."}])
        return row_audit,reconciliation,validation,project

    def _runtime_reconciliation_figures(self,reconciliation,validation):
        fig,ax=plt.subplots(figsize=(8,7))
        if len(validation):
            x=pd.to_numeric(validation.observed_inference_session_seconds,errors="coerce")/3600
            y=pd.to_numeric(validation.slot_adjusted_success_seconds,errors="coerce")/3600
            for (_,r),xx,yy in zip(validation.iterrows(),x,y):
                ax.scatter(xx,yy,color=MODEL_COLORS.get(r.model,"#777"),marker=CONDITION_MARKERS.get(r.condition_short,"o"),alpha=.75)
            top=max(float(x.max()),float(y.max()),1);ax.plot([0,top],[0,top],color="black",lw=1,ls="--")
        ax.set(xlabel="Observed inference invocation hours",ylabel="Slot-adjusted row reconstruction hours",title="Runtime reconstruction validation on telemetry-complete streams")
        self._figure(self.p.figures/"37A_runtime_reconstruction_validation.png",fig,data=validation,key="Each point is a telemetry-complete exact stream; colour identifies model and marker identifies treatment. The dashed line is exact agreement.",sources="compute_invocation_audit.csv; 09__compute__row_performance.csv")
        plot=reconciliation.copy();plot["observed_hours"]=pd.to_numeric(plot.observed_inference_session_seconds,errors="coerce")/3600;plot["low_hours"]=pd.to_numeric(plot.selected_inference_low_seconds,errors="coerce")/3600;plot["high_hours"]=pd.to_numeric(plot.selected_inference_high_seconds,errors="coerce")/3600
        plot=plot.sort_values("high_hours").tail(40)
        fig,ax=plt.subplots(figsize=(12,max(6,.28*len(plot))))
        if len(plot):
            y=np.arange(len(plot));ax.scatter(plot.observed_hours,y,color="#0072B2",label="captured telemetry",zorder=3);ax.hlines(y,plot.low_hours,plot.high_hours,color="#D55E00",lw=2,label="selected/reconstructed range");ax.scatter(plot.low_hours,y,color="#D55E00",s=18);ax.set_yticks(y,plot.stream_display_name)
        ax.set(xlabel="Hours",title="Observed and reconciled inference time (largest 40 streams)")
        self._figure(self.p.figures/"37B_runtime_reconciliation_by_stream.png",fig,data=plot,key="Blue points are captured invocation hours; orange intervals are selected inference ranges. Complete streams retain observed time; incomplete streams use row reconstruction.",sources="runtime_reconciliation_by_stream.csv")

    @staticmethod
    def _same_gpu_types(left,right):
        parse=lambda value:{x.strip() for x in str(value).split("|") if x.strip() and x.strip().lower()!="nan"}
        a,b=parse(left),parse(right)
        return bool(len(a)==1 and a==b)

    def compute_analysis(self):
        if not hasattr(self,"compute_streams_derived"):self.compute_reporting()
        streams=self.compute_streams_derived.copy();rows=self.compute_rows.copy()
        if streams.empty:return
        streams["parameter_count"]=streams.model.map(MODEL_PARAMETERS)
        row_audit,reconciliation,validation,project=self._runtime_reconciliation(streams)
        self._write(row_audit,"runtime_reconstruction_row_audit.csv")
        self._write(reconciliation,"runtime_reconciliation_by_stream.csv")
        self._write(validation,"runtime_reconstruction_validation.csv")
        self._write(project,"runtime_reconciliation_project_total.csv")
        rec_cols=["model","condition_short","representation_group","representation_label","accounting_method",
                  "selected_inference_low_seconds","selected_inference_high_seconds","observed_inference_session_seconds",
                  "telemetry_new_row_capture_ratio","telemetry_generation_capture_ratio"]
        streams=streams.merge(reconciliation[rec_cols],on=["model","condition_short","representation_group","representation_label"],how="left")
        completed=pd.to_numeric(streams.completed,errors="coerce").replace(0,np.nan)
        streams["reconciled_inference_seconds_per_completed_row_low"]=streams.selected_inference_low_seconds/completed
        streams["reconciled_inference_seconds_per_completed_row_high"]=streams.selected_inference_high_seconds/completed
        streams["reconciled_inference_seconds_per_completed_row_midpoint"]=(streams.reconciled_inference_seconds_per_completed_row_low+streams.reconciled_inference_seconds_per_completed_row_high)/2
        streams["completed_rows_per_reconciled_inference_hour_low"]=completed/(streams.selected_inference_high_seconds/3600)
        streams["completed_rows_per_reconciled_inference_hour_high"]=completed/(streams.selected_inference_low_seconds/3600)
        streams["completed_rows_per_reconciled_inference_hour_midpoint"]=completed/(((streams.selected_inference_low_seconds+streams.selected_inference_high_seconds)/2)/3600)
        observed_usable=streams.accounting_method.eq("observed_complete")|pd.to_numeric(streams.telemetry_new_row_capture_ratio,errors="coerce").ge(.5)
        streams["efficiency_seconds_per_completed_row_point"]=np.where(observed_usable,streams.observed_inference_session_seconds/completed,streams.reconciled_inference_seconds_per_completed_row_midpoint)
        streams["efficiency_seconds_per_completed_row_low"]=np.where(observed_usable,streams.observed_inference_session_seconds/completed,streams.reconciled_inference_seconds_per_completed_row_low)
        streams["efficiency_seconds_per_completed_row_high"]=np.where(streams.accounting_method.eq("observed_complete"),streams.observed_inference_session_seconds/completed,streams.reconciled_inference_seconds_per_completed_row_high)
        streams["efficiency_time_basis"]=np.where(streams.accounting_method.eq("observed_complete"),"observed complete inference session / completed rows",
            np.where(observed_usable,"observed inference session / completed rows; majority new-row telemetry captured; reconciled high bound retained","row-reconstructed inference wall time / completed rows"))
        streams=self._compute_population_rows(streams)
        self._write(streams,"compute_size_efficiency.csv")
        self._write(streams[[c for c in streams if c not in {"estimated_energy_wh"}]],"deployment_stack_comparisons.csv")
        timecols=[c for c in streams if c.endswith("seconds") or c.endswith("session_hours") or c in ["population","model","condition_short","representation_group","representation_label","stream_display_name"]]
        time=self._write(streams[timecols],"time_allocation_and_overhead.csv")
        self._write(rows,"runtime_configuration_analysis.csv")
        frontier=streams[[c for c in ["population","model","condition_short","representation_group","representation_label","sampled_peak_gpu_memory_bytes","population_peak_allocated_bytes","population_rows_per_slot_adjusted_request_hour","population_prompt_tokens_per_slot_adjusted_request_hour","population_output_tokens_per_slot_adjusted_request_hour","completed_rows_per_reconciled_inference_hour_low","completed_rows_per_reconciled_inference_hour_high","generated_tokens_per_inference_session_hour","completed","record_count","accounting_method"] if c in streams]].copy()
        # PyTorch peak-allocation counters do not exist for llama.cpp/BitNet.cpp.
        # Figure 35 therefore uses one consistent, available quantity for every
        # backend: the sampled whole-device memory peak from exclusive runtimes.
        frontier["figure35_memory_bytes"]=pd.to_numeric(frontier.get("sampled_peak_gpu_memory_bytes"),errors="coerce").replace(0,np.nan)
        frontier["figure35_memory_source"]="sampled whole-device peak; not process-specific allocated memory"
        frontier=self._write(frontier,"compression_performance_frontier.csv")
        contrasts=pd.read_csv(self.p.tables/"treatment_contrasts.csv") if (self.p.tables/"treatment_contrasts.csv").exists() else pd.DataFrame()
        efficiency=pd.DataFrame()
        if len(contrasts):
            resource=streams[["population","model","condition_short","representation_group","gpu_types","preparation_session_hours","completed",
                              "selected_inference_low_seconds","selected_inference_high_seconds","accounting_method"]].rename(columns={"condition_short":"adapter"})
            wide=resource.pivot_table(index=["population","model","adapter"],columns="representation_group",values=["gpu_types","preparation_session_hours","completed","selected_inference_low_seconds","selected_inference_high_seconds","accounting_method"],aggfunc="first").reset_index()
            wide.columns=["_".join([str(x) for x in c if str(x)]) if isinstance(c,tuple) else c for c in wide.columns]
            wide=wide.rename(columns={c:c.replace("BF16 HF","bf16").replace("one-bit frontier","one_bit") for c in wide.columns})
            efficiency=contrasts.merge(wide,on=["population","model","adapter"],how="left")
            efficiency["same_gpu_flag"]=efficiency.apply(lambda r:self._same_gpu_types(r.get("gpu_types_bf16",""),r.get("gpu_types_one_bit","")),axis=1)
            efficiency["comparison_reconciled_inference_hours"] = np.where(
                efficiency.contrast.str.startswith("bf16_"),pd.to_numeric(efficiency.get("selected_inference_high_seconds_bf16"),errors="coerce")/3600,
                pd.to_numeric(efficiency.get("selected_inference_high_seconds_one_bit"),errors="coerce")/3600)
        self._write(efficiency,"sft_quantization_efficiency.csv")
        row_times=rows.copy()
        row_times["condition_short"]=row_times.condition.map(condition_short)
        row_times["representation_group"]=row_times.representation.map(representation_group)
        row_times["elapsed_seconds"]=pd.to_numeric(row_times.elapsed_seconds,errors="coerce")
        row_times=row_times[row_times.status.astype(str).str.lower().eq("completed")]
        amort=[]
        amort_streams=streams[streams.population.eq("all_990")]
        for (model,condition),g in amort_streams.groupby(["model","condition_short"]):
            bf=g[g.representation_group.eq("BF16 HF")];one=g[g.representation_group.eq("one-bit frontier")]
            if bf.empty or one.empty:continue
            b,z=bf.iloc[0],one.iloc[0]
            bp=row_times[(row_times.model.eq(model))&(row_times.condition_short.eq(condition))&(row_times.representation_group.eq("BF16 HF"))][["prompt_id","elapsed_seconds"]].drop_duplicates("prompt_id")
            zp=row_times[(row_times.model.eq(model))&(row_times.condition_short.eq(condition))&(row_times.representation_group.eq("one-bit frontier"))][["prompt_id","elapsed_seconds"]].drop_duplicates("prompt_id")
            paired=bp.merge(zp,on="prompt_id",suffixes=("_bf16","_one_bit")).dropna()
            b_low=float(b.efficiency_seconds_per_completed_row_low);b_high=float(b.efficiency_seconds_per_completed_row_high)
            z_low=float(z.efficiency_seconds_per_completed_row_low);z_high=float(z.efficiency_seconds_per_completed_row_high)
            bsec=float(b.efficiency_seconds_per_completed_row_point);zsec=float(z.efficiency_seconds_per_completed_row_point)
            saving=bsec-zsec;saving_low=b_low-z_high;saving_high=b_high-z_low
            prep=float(z.preparation_session_hours)*3600;same_gpu=self._same_gpu_types(b.gpu_types,z.gpu_types)
            if not same_gpu:status="not_estimable_gpu_mismatch"
            elif not np.isfinite(saving) or saving<=0:status="not_reached_one_bit_not_faster"
            else:status="estimated_same_gpu_point_time"
            amort.append({"model":model,"condition":condition,"bf16_gpu_types":b.gpu_types,"one_bit_gpu_types":z.gpu_types,
                "same_gpu_flag":same_gpu,"common_completed_prompt_rows":len(paired),
                "bf16_median_row_latency_seconds":paired.elapsed_seconds_bf16.median(),"one_bit_median_row_latency_seconds":paired.elapsed_seconds_one_bit.median(),
                "bf16_efficiency_seconds_per_completed_row":bsec,"one_bit_efficiency_seconds_per_completed_row":zsec,
                "bf16_time_basis":b.efficiency_time_basis,"one_bit_time_basis":z.efficiency_time_basis,
                "bf16_reconciled_seconds_per_completed_row_low":b_low,"bf16_reconciled_seconds_per_completed_row_high":b_high,
                "one_bit_reconciled_seconds_per_completed_row_low":z_low,"one_bit_reconciled_seconds_per_completed_row_high":z_high,
                "one_bit_preparation_seconds":prep,"seconds_saved_per_row":saving,"seconds_saved_per_row_low":saving_low,"seconds_saved_per_row_high":saving_high,
                "speedup_direction_robust_to_reconciled_range":bool(saving_low>0),
                "break_even_rows":prep/saving if status=="estimated_same_gpu_point_time" else np.nan,
                "status":status,"time_basis":"audited observed/reconstructed inference time per completed row; paired median latency shown separately"})
        amortdf=self._write(pd.DataFrame(amort),"preparation_amortization.csv")
        self._runtime_reconciliation_figures(reconciliation,validation)
        self._compute_analysis_figures(streams,time,frontier,efficiency,amortdf)

    def _compute_analysis_figures(self,streams,time,frontier,efficiency,amort):
        streams=streams[streams.population.eq("evaluation_285")]
        time=time[time.population.eq("all_990")] if "population" in time else time
        frontier=frontier[frontier.population.eq("evaluation_285")]
        efficiency=efficiency[efficiency.population.eq("evaluation_285")] if len(efficiency) and "population" in efficiency else efficiency
        measures=[("population_rows_per_slot_adjusted_request_hour","Rows/slot-adjusted request hour"),("population_output_tokens_per_slot_adjusted_request_hour","Output tokens/slot-adjusted request hour"),("population_peak_allocated_bytes","Peak allocated bytes in evaluation rows")]
        fig,axes=plt.subplots(1,3,figsize=(16,5))
        for ax,(col,title) in zip(axes,measures):
            for _,r in streams.dropna(subset=[col]).iterrows():ax.scatter(r.parameter_count,r[col],color=MODEL_COLORS.get(r.model,"#777"),marker=CONDITION_MARKERS.get(r.condition_short,"o"));ax.set_xscale("log");ax.set_title(title)
        self._figure(self.p.figures/"30_size_efficiency_relationships.png",fig,data=streams,key="Colour identifies model family; marker identifies treatment. The x-axis is model parameter count on a log scale.",sources="compute_stream_resource_summary.csv")
        fig,axes=plt.subplots(1,3,figsize=(16,5))
        for ax,(col,title) in zip(axes,measures):
            for (m,c),g in streams.groupby(["model","condition_short"]):
                gg=g.groupby("representation_group")[col].mean().reindex(["BF16 HF","one-bit frontier"]).dropna()
                if len(gg)>1:ax.plot(gg.index,gg.values,marker="o",color=MODEL_COLORS.get(m,"#777"),alpha=.7,label=MODEL_LABELS.get(m,m))
            ax.set_title(title)
        self._figure(self.p.figures/"31_deployment_stack_resource_changes.png",fig,data=streams,key="Each line connects BF16 HF and one-bit-frontier observations for the same model and treatment; colour identifies model.",sources="compute_stream_resource_summary.csv")
        fig,ax=plt.subplots(figsize=(10,6))
        if len(efficiency):
            e=efficiency.dropna(subset=["mean_difference","comparison_reconciled_inference_hours"])
            for _,r in e.iterrows():ax.scatter(r.comparison_reconciled_inference_hours,r.mean_difference,color=MODEL_COLORS.get(r.model,"#777"),marker=CONDITION_MARKERS.get(r.adapter,"o"),alpha=.7)
        ax.axhline(0,color="black",lw=.8);ax.set(xlabel="Reconciled inference wall hours",ylabel="Paired judge-score mean difference",title="Resource and judge-score treatment contrasts")
        self._figure(self.p.figures/"32_sft_quantization_resource_outcomes.png",fig,data=efficiency,key="Colour identifies model; marker identifies adapter. Positive y-values favour the treatment named by the contrast. Backend and hardware differences make this descriptive, not causal.",sources="treatment_contrasts.csv; compute_stream_resource_summary.csv")
        phase=time[[c for c in ["training_session_hours","preparation_session_hours","inference_session_hours"] if c in time]].sum().sort_values()
        fig,ax=plt.subplots(figsize=(9,5));ax.barh(phase.index,phase.values,color=["#0072B2","#E69F00","#009E73"][:len(phase)]);ax.set(xlabel="GPU-runtime session hours",title="Phase time allocation");self._figure(self.p.figures/"33A_phase_time_allocation.png",fig,data=phase.rename("hours").reset_index(),key="Blue=training; amber=artifact preparation/upload; green=inference. These are allocated runtime-session hours, not active floating-point GPU hours.",sources="compute_stream_resource_summary.csv")
        fig,ax=plt.subplots(figsize=(10,5));components=[c for c in ["model_load_seconds","restore_download_seconds","generation_seconds","receiver_seconds","csv_snapshot_seconds"] if c in time]
        for col in components:ax.scatter([col]*len(time),time[col]/(time.total_gpu_runtime_session_hours*3600),alpha=.25,label=col)
        ax.tick_params(axis="x",rotation=30);ax.set(ylabel="Component / stream session time",title="Component overhead ratios (timers may overlap)");self._figure(self.p.figures/"33B_component_overhead_ratios.png",fig,data=time,key="Each point is one exact stream. Ratios describe recorded components and may overlap; they must not be summed to 100%.",sources="compute_stream_resource_summary.csv")
        fig,ax=plt.subplots(figsize=(10,5));
        if len(self.compute_rows):sns.scatterplot(self.compute_rows,x="input_tokens",y="elapsed_seconds",hue="representation",alpha=.2,ax=ax)
        ax.set(xscale="log",yscale="log",title="Observed workload and elapsed time");self._figure(self.p.figures/"34_runtime_configuration_effects.png",fig,data=self.compute_rows,key="Each point is an observed row attempt; colour identifies the exact stored representation. Both axes are logarithmic.",sources="09__compute__row_performance.csv")
        fig,ax=plt.subplots(figsize=(9,6));
        if len(frontier):
            plotted=frontier[frontier.population.eq("evaluation_285")].copy()
            for _,r in plotted.iterrows():ax.scatter(pd.to_numeric(r.figure35_memory_bytes,errors="coerce")/2**30,r.population_rows_per_slot_adjusted_request_hour,color=MODEL_COLORS.get(r.model,"#777"),marker=CONDITION_MARKERS.get(r.condition_short,"o"))
        else:plotted=frontier
        ax.set(xlabel="Sampled whole-device peak GPU memory (GiB)",ylabel="Rows/slot-adjusted request hour");fig.suptitle("Compression-performance frontier")
        self._figure(self.p.figures/"35_compression_performance_frontier.png",fig,data=plotted,key="Colour identifies model; marker identifies treatment. Memory is sampled whole-device peak usage during an exclusive runtime, not a process-specific allocation counter. Rates use the fixed 285-row pool; backend and hardware differences remain explicit.",sources="compression_performance_frontier.csv",key_position="top")
        fig,ax=plt.subplots(figsize=(10,max(5,.25*len(amort))))
        if len(amort):
            a=amort.copy();a["label"]=a.model.map(MODEL_LABELS).fillna(a.model)+" / "+a.condition.astype(str);a=a.sort_values("break_even_rows");ax.barh(a.label,a.break_even_rows,color=[MODEL_COLORS.get(m,"#777") for m in a.model])
        ax.set(xlabel="Estimated completed rows to offset preparation session time",title="Preparation-time amortization")
        self._figure(self.p.figures/"36_preparation_amortization.png",fig,data=amort,key="Colour identifies model. Bars are shown only for same-GPU pairs with a positive conservative reconciled time saving; paired median latency and non-estimable reasons remain in the source table.",sources="preparation_amortization.csv")

    def _full_universe_figures(self):
        """Render untreated-only all_990 sensitivity figures for Section 5.2."""
        summary=pd.read_csv(self.p.tables/"model_scores_by_dimension.csv");dist=pd.read_csv(self.p.tables/"model_score_distributions.csv")
        full=summary[(summary.population.eq("all_990"))&summary.condition_short.eq("untreated")]
        for rep,code in (("BF16 HF","01A"),("one-bit frontier","01B")):
            p=full[(full.representation_group.eq(rep))&full.dimension.isin(SUBSTANTIVE_SCORES)]
            matrix=p.pivot_table(index="model",columns="dimension",values="mean").reindex(MODELS)
            fig,ax=plt.subplots(figsize=(12,5.5));sns.heatmap(matrix,vmin=0,vmax=4,cmap="RdBu",center=2,ax=ax,annot=True,fmt=".2f");ax.set(xlabel="",ylabel="")
            fig.suptitle(f"Untreated {rep} — full 990 universe")
            self._figure(self.p.figures/f"{code}_990_untreated_score_overview_{'bf16' if rep=='BF16 HF' else 'one_bit'}.png",fig,data=p,key="Rows are the six untreated model families; columns are separate judge dimensions on the common 0–4 scale.",sources="model_scores_by_dimension.csv",show_key=False)
        for score in SCORE_COLUMNS:
            short=SHORT_SCORE[score]
            for rep,code in (("BF16 HF","02A"),("one-bit frontier","02B")):
                p=dist[(dist.population.eq("all_990"))&dist.condition_short.eq("untreated")&dist.representation_group.eq(rep)&dist.dimension.eq(score)]
                fig,axes=plt.subplots(2,3,figsize=(14,7),sharex=True,sharey=True)
                for ax,model in zip(axes.ravel(),MODELS):
                    m=p[p.model.eq(model)].set_index("score").reindex(range(5));left=0
                    for value,color in zip(range(5),ORDINAL_COLORS):
                        width=float(m.loc[value,"proportion"]) if value in m.index and pd.notna(m.loc[value,"proportion"]) else 0
                        ax.barh([0],[width],left=left,color=color,label=str(value));left+=width
                    ax.set_xlim(0,1);ax.set_yticks([]);ax.set_title(MODEL_LABELS[model])
                fig.suptitle(f"Untreated {rep} — {short.replace('_',' ')} distribution (full 990 universe)")
                self._figure(self.p.figures/f"{code}_990_{short}_{'bf16' if rep=='BF16 HF' else 'one_bit'}_untreated_distribution.png",fig,data=p,key="Stack colours encode judge scores 0–4; every panel is an untreated model on the full-universe sensitivity population.",sources="model_score_distributions.csv")

        contrasts=pd.read_csv(self.p.tables/"treatment_contrasts.csv")
        for score in SCORE_COLUMNS:
            short=SHORT_SCORE[score];p=contrasts[(contrasts.population.eq("all_990"))&contrasts.contrast.eq("untreated_one_bit_minus_bf16")&contrasts.dimension.eq(score)].copy()
            p=p.set_index("model").reindex(MODELS).reset_index();fig,ax=plt.subplots(figsize=(9,5));y=np.arange(len(p))
            ax.errorbar(p.mean_difference,y,xerr=[p.mean_difference-p.low95,p.high95-p.mean_difference],fmt="o",color="#0072B2");ax.axvline(0,color="black",lw=.8);ax.set_yticks(y,[MODEL_LABELS[m] for m in p.model]);ax.set(xlabel="One-bit minus BF16 judge score")
            fig.suptitle(f"Untreated deployment contrast — {short.replace('_',' ')} (full 990 universe)")
            fig.subplots_adjust(left=.24,bottom=.14,top=.88)
            self._figure(self.p.figures/f"03C_990_{short}_untreated_deployment_stack_effects.png",fig,data=p,key="Points are paired untreated one-bit-minus-BF16 means; lines are 95% scenario-step cluster-bootstrap intervals.",sources="treatment_contrasts.csv",show_key=False)

        classes=pd.read_csv(self.p.tables/"attack_class_scores.csv")
        for score in SCORE_COLUMNS:
            short=SHORT_SCORE[score];p=classes[(classes.population.eq("all_990"))&classes.condition.eq("untreated")&classes.dimension.eq(score)]
            fig,axes=plt.subplots(1,2,figsize=(18,8),sharey=True)
            plotted=[]
            for ax,rep in zip(axes,("BF16 HF","one-bit frontier")):
                q=p[p.representation.str.contains("BF16",na=False)] if rep=="BF16 HF" else p[~p.representation.str.contains("BF16",na=False)]
                top=q.groupby("technique").n.sum().nlargest(20).index;matrix=q[q.technique.isin(top)].pivot_table(index="technique",columns="model",values="mean").reindex(columns=MODELS)
                sns.heatmap(matrix,vmin=0,vmax=4,cmap="RdBu",center=2,ax=ax,cbar=ax is axes[-1]);ax.set_title(rep);plotted.append(q[q.technique.isin(top)])
            fig.suptitle(f"Untreated attack-class scores — {short.replace('_',' ')} (full 990 universe)")
            self._figure(self.p.figures/f"06_990_{short}_untreated_attack_class_scores.png",fig,data=pd.concat(plotted),key="Cells are mean judge scores for the most represented exact ATT&CK classes; BF16 and one-bit panels share a 0–4 scale.",sources="attack_class_scores.csv")
            for rep,tag in (("BF16 HF","bf16"),("one-bit frontier","one_bit")):
                q=p[p.representation.str.contains("BF16",na=False)] if rep=="BF16 HF" else p[~p.representation.str.contains("BF16",na=False)]
                eligible=q[q.n.ge(MIN_CLASS_N)];lowest=eligible.groupby("technique").mean(numeric_only=True).nsmallest(12,"mean").index;plot=eligible[eligible.technique.isin(lowest)]
                fig,ax=plt.subplots(figsize=(11,7))
                if len(plot):sns.pointplot(plot,x="mean",y="technique",hue="model",palette=MODEL_COLORS,errorbar=None,ax=ax)
                else:ax.text(.5,.5,"Not estimable: no sufficiently represented classes",ha="center",va="center",transform=ax.transAxes)
                ax.set_xlim(0,4);ax.set_title(f"Lowest untreated classes — {rep}, {short.replace('_',' ')} (full 990 universe)")
                figure_data=plot if len(plot) else pd.DataFrame([{"population":"all_990","representation_group":rep,"dimension":score,"status":"not_estimable_no_sufficiently_represented_class","minimum_class_n":MIN_CLASS_N}])
                self._figure(self.p.figures/f"07_990_{short}_{tag}_lowest_attack_classes.png",fig,data=figure_data,key=f"Only exact classes with at least {MIN_CLASS_N} score observations are eligible; points are model-specific means.",sources="attack_class_scores.csv")

        failure=pd.read_csv(self.p.tables/"failure_rates.csv");p=failure[(failure.population.eq("all_990"))&failure.condition_short.eq("untreated")]
        fig,axes=plt.subplots(2,3,figsize=(20,11),sharex=True)
        for ax,model in zip(axes.ravel(),MODELS):
            m=p[p.model.eq(model)];top=m.groupby("failure_category").rate_per_100.mean().nlargest(8).index;wide=m[m.failure_category.isin(top)].pivot_table(index="failure_category",columns="representation_group",values="rate_per_100").fillna(0)
            wide.index=[textwrap.fill(str(x).replace("_"," "),width=26) for x in wide.index]
            wide.plot.barh(ax=ax,color=["#0072B2","#D55E00"]);ax.set_title(MODEL_LABELS[model]);ax.set_xlabel("per 100 judged rows");ax.tick_params(axis="y",labelsize=7)
        fig.suptitle("Untreated failure-category rates (full 990 universe)");fig.subplots_adjust(wspace=.95,hspace=.38,left=.16,right=.98,top=.86,bottom=.09)
        self._figure(self.p.figures/"09_990_untreated_failure_rates.png",fig,data=p,key="Blue and orange bars distinguish BF16 and one-bit-frontier rates for the same broad judge-assigned failure category.",sources="failure_rates.csv",key_position="top")
        sim=pd.read_csv(self.p.tables/"failure_profile_similarity.csv");p=sim[(sim.population.eq("all_990"))&sim.condition.eq("untreated")]
        fig,axes=plt.subplots(1,2,figsize=(13,6))
        for ax,rep in zip(axes,("BF16 HF","one-bit frontier")):
            q=p[p.representation_group.eq(rep)];matrix=pd.DataFrame(np.eye(len(MODELS)),index=MODELS,columns=MODELS,dtype=float)
            for _,row in q.iterrows():matrix.loc[row.model_a,row.model_b]=matrix.loc[row.model_b,row.model_a]=row.jensen_shannon_similarity
            sns.heatmap(matrix,vmin=0,vmax=1,cmap="viridis",ax=ax,cbar=ax is axes[-1]);ax.set_title(rep)
        fig.suptitle("Untreated failure-profile similarity (full 990 universe)");self._figure(self.p.figures/"10_990_untreated_failure_profile_similarity.png",fig,data=p,key="Cells are Jensen–Shannon similarities of broad failure-rate profiles; non-estimable cells remain blank.",sources="failure_profile_similarity.csv")

        stopping=pd.read_csv(self.p.tables/"token_ceiling_and_stop_reasons.csv");p=stopping[(stopping.population.eq("all_990"))&stopping.condition.eq("untreated")]
        fig,axes=plt.subplots(2,3,figsize=(14,8),sharex=True)
        for ax,model in zip(axes.ravel(),MODELS):
            m=p[p.model.eq(model)];x=np.arange(len(m));ax.barh(x,m.at_ceiling_rate,color="#D55E00",label="ceiling");ax.barh(x,m.natural_eos_rate,left=m.at_ceiling_rate,color="#009E73",label="EOS");ax.set_yticks(x,m.representation);ax.set_xlim(0,1);ax.set_title(MODEL_LABELS[model])
        fig.suptitle("Untreated stopping behaviour (full 990 universe; completed-response denominator)");self._figure(self.p.figures/"14_990_untreated_token_ceiling_rates.png",fig,data=p,key="Green is natural EOS and orange is the output ceiling; context-unsupported and error rows are separate table columns.",sources="token_ceiling_and_stop_reasons.csv")

        concord=pd.read_csv(self.p.tables/"model_item_difficulty_concordance.csv")
        for score in SCORE_COLUMNS:
            p=concord[(concord.population.eq("all_990"))&concord.condition.eq("untreated")&concord.dimension.eq(score)];fig,axes=plt.subplots(1,2,figsize=(13,6))
            for ax,rep in zip(axes,("BF16 HF","one-bit frontier")):
                q=p[p.representation.eq(rep)];matrix=q.pivot(index="model_a",columns="model_b",values="spearman").reindex(index=MODELS,columns=MODELS);sns.heatmap(matrix,vmin=-1,vmax=1,cmap="vlag",center=0,ax=ax,cbar=ax is axes[-1]);ax.set_title(rep)
            fig.suptitle(f"Untreated item-difficulty concordance — {SHORT_SCORE[score].replace('_',' ')} (full 990 universe)");self._figure(self.p.figures/f"17_990_{SHORT_SCORE[score]}_untreated_item_difficulty_concordance.png",fig,data=p,key="Cells are pairwise Spearman correlations on common untreated prompts; constant/floor-saturated pairs remain non-estimable.",sources="model_item_difficulty_concordance.csv")

        long=pd.read_csv(self.p.tables/"prompt_length_and_long_tail.csv");p=long[(long.population.eq("all_990"))&long.condition.eq("untreated")]
        for file_name,y,title in (("18B_990_untreated_completion_by_prompt_length.png","completed_rate","Completion by prompt length"),("18C_990_untreated_scores_by_prompt_length.png","mean_technique_identification","Technique-identification score by prompt length")):
            fig,axes=plt.subplots(1,2,figsize=(15,5),sharey=True)
            for ax,rep in zip(axes,("BF16 HF","one-bit frontier")):
                q=p[p.representation.str.contains("BF16",na=False)] if rep=="BF16 HF" else p[~p.representation.str.contains("BF16",na=False)];sns.pointplot(q,x="prompt_length_bucket",y=y,hue="model",palette=MODEL_COLORS,errorbar=None,ax=ax);ax.set_title(rep)
            fig.suptitle(f"Untreated {title.lower()} (full 990 universe)");self._figure(self.p.figures/file_name,fig,data=p,key="Panels separate BF16 and one-bit-frontier untreated streams; colours identify model families.",sources="prompt_length_and_long_tail.csv")

        rank=pd.read_csv(self.p.tables/"judge_mediated_rank_metrics_by_stream.csv");p=rank[(rank.population.eq("all_990"))&rank.condition_short.eq("untreated")&rank.metric.eq("hit_rate")]
        fig,axes=plt.subplots(2,3,figsize=(15,9),sharex=True,sharey=True)
        for ax,model in zip(axes.ravel(),MODELS):
            for rep,colour,ls in (("BF16 HF","#0072B2","-"),("one-bit frontier","#D55E00","--")):
                q=p[(p.model.eq(model))&p.representation_group.eq(rep)].sort_values("k");ax.plot(q.k,q.estimate,marker="o",ms=3,color=colour,ls=ls,label=rep)
            ax.set(xlim=(1,10),ylim=(0,1),xticks=range(1,11),title=MODEL_LABELS[model],xlabel="k",ylabel="Judge-mediated Hit@k")
        fig.suptitle("Untreated judge-mediated Hit@k (full 990 universe)");self._figure(self.p.figures/"19C_990_untreated_judge_mediated_hit_at_k.png",fig,data=p,key="Solid blue is BF16 HF and dashed orange is one-bit-frontier deployment; all rows are untreated full-universe sensitivity estimates.",sources="judge_mediated_rank_metrics_by_stream.csv")

    def bitnet_template_example(self):
        """Write the deterministic BitNet template example and prevalence audit.

        This is a read-only derivation from canonical final CSVs. It never
        regenerates a response or judgment and verifies every prompt hash
        before selecting the shortest eligible prompt.
        """
        final=self.p.final
        files={
            "prompts":final/"00__source__prompts.csv",
            "truth":final/"00__source__ground_truth.csv",
            "split":final/"00__source__split_manifest.csv",
            "bf16_response":final/"02__response__bitnet2b__untreated_hf__all.csv",
            "i2s_response":final/"04__response__bitnet2b__untreated__native_i2_s__all.csv",
            "i2s_full_response":final/"05__response__bitnet2b__sft_full__native_i2_s__validation_test.csv",
            "bf16_judgment":final/"03__judgment__legacy_common__bitnet2b__untreated__hf__all.csv",
            "i2s_judgment":final/"07__judgment__quantized_common__bitnet2b__untreated__native_i2_s__all.csv",
            "i2s_full_judgment":final/"07__judgment__quantized_common__bitnet2b__sft_full__native_i2_s__validation_test.csv",
            "positive_register":self.p.tables/"one_bit_positive_score_row_register.csv",
        }
        missing=[str(path) for path in files.values() if not path.exists()]
        if missing:raise FileNotFoundError("Missing canonical inputs: "+", ".join(missing))
        prompts=pd.read_csv(files["prompts"],dtype={"prompt_id":str})
        truth=pd.read_csv(files["truth"],dtype={"prompt_id":str})
        split=pd.read_csv(files["split"],dtype={"prompt_id":str})
        source=prompts.merge(truth,on=["prompt_id","scenario","variant","step"],validate="one_to_one").merge(
            split[["prompt_id","scenario","variant","step","split"]],on=["prompt_id","scenario","variant","step"],validate="one_to_one")
        calculated=source.prompt_text.fillna("").map(lambda x:hashlib.sha256(str(x).encode("utf-8")).hexdigest())
        if not calculated.eq(source.prompt_sha256).all():raise ValueError("Canonical prompt_sha256 verification failed")
        source["prompt_characters"]=source.prompt_text.fillna("").astype(str).str.len()
        pattern=re.compile(BITNET_TEMPLATE_REGEX,re.IGNORECASE)
        confidence_pattern=re.compile(BITNET_CONFIDENCE_TEMPLATE_REGEX,re.IGNORECASE)
        techniques_pattern=re.compile(BITNET_TECHNIQUES_ECHO_REGEX,re.IGNORECASE)
        def signature(text):
            value=str(text).lstrip()
            if confidence_pattern.match(value):return "confidence_list"
            if techniques_pattern.match(value):return "techniques_instruction_echo"
            return ""
        def match(text):return bool(pattern.match(str(text).lstrip()))
        def response(name):
            frame=pd.read_csv(files[name],dtype={"prompt_id":str})
            if not frame.prompt_sha256.astype(str).isin(set(source.prompt_sha256)).all():raise ValueError(f"Prompt hash mismatch in {files[name].name}")
            return frame
        def judgment(name):
            frame=pd.read_csv(files[name],dtype={"prompt_id":str})
            if not frame.prompt_sha256.astype(str).isin(set(source.prompt_sha256)).all():raise ValueError(f"Prompt hash mismatch in {files[name].name}")
            return frame
        bf=response("bf16_response");i2=response("i2s_response")
        jb=judgment("bf16_judgment");ji=judgment("i2s_judgment")
        eligible=source[source.split.isin(["validation","test"])].copy()
        bcols=["prompt_id","prompt_sha256","record_id","response_raw","response_hash","status","finish_reason","output_tokens"]
        icols=bcols.copy()
        eligible=eligible.merge(bf[bcols].rename(columns={c:f"bf16_{c}" for c in bcols if c not in {"prompt_id","prompt_sha256"}}),on=["prompt_id","prompt_sha256"])
        eligible=eligible.merge(i2[icols].rename(columns={c:f"i2s_{c}" for c in icols if c not in {"prompt_id","prompt_sha256"}}),on=["prompt_id","prompt_sha256"])
        eligible=eligible[(eligible.bf16_status.eq("completed"))&(eligible.i2s_status.eq("completed"))]
        judge_ids_b=set(jb.loc[jb.status.eq("completed"),"prompt_id"]);judge_ids_i=set(ji.loc[ji.status.eq("completed"),"prompt_id"])
        eligible=eligible[eligible.prompt_id.isin(judge_ids_b&judge_ids_i)].copy()
        eligible["template_match"]=eligible.i2s_response_raw.map(match)
        eligible["template_signature"]=eligible.i2s_response_raw.map(signature)
        eligible=eligible[eligible.template_match]
        natural=eligible[eligible.bf16_finish_reason.fillna("").str.lower().isin(["eos","stop"])]
        natural_fallback=False
        if natural.empty:natural=eligible;natural_fallback=True
        natural=natural.assign(_prompt_id_numeric=pd.to_numeric(natural.prompt_id,errors="raise")).sort_values(["prompt_characters","_prompt_id_numeric"])
        if natural.empty:raise RuntimeError("No eligible BitNet template row exists")
        p1=natural.iloc[0]
        p2pool=natural[(natural.scenario.ne(p1.scenario))&(natural.template_signature.eq(p1.template_signature))]
        if p2pool.empty:raise RuntimeError("No eligible different-scenario P2 with the same template subtype exists")
        p2=p2pool.iloc[0]
        def judge_row(frame,pid):
            found=frame[(frame.prompt_id.eq(pid))&frame.status.eq("completed")]
            return found.iloc[0] if len(found) else pd.Series(dtype=object)
        def first(row,*names):
            for name in names:
                if name in row and pd.notna(row[name]):return row[name]
            return np.nan
        def excerpts(text):
            original="" if pd.isna(text) else str(text);shown=original.replace("\t","    ")
            return original,shown[:600],shown[-200:] if len(shown)>800 else "",max(len(original)-800,0)
        def log_excerpt(prompt):
            parts=str(prompt).split("\n\n",1);lines=(parts[1] if len(parts)>1 else str(prompt)).splitlines()
            return "\n".join(lines[:8]).replace("\t","    ")
        panel_rows=[]
        def add(panel,selected,kind,representation=""):
            if kind=="prompt":
                text=selected.prompt_text;jrow=pd.Series(dtype=object);record_id=response_hash=finish=out_tokens="";source_file=files["prompts"]
            elif kind=="bf16":
                text=selected.bf16_response_raw;jrow=judge_row(jb,selected.prompt_id);record_id=selected.bf16_record_id;response_hash=selected.bf16_response_hash;finish=selected.bf16_finish_reason;out_tokens=selected.bf16_output_tokens;source_file=files["bf16_response"]
            else:
                text=selected.i2s_response_raw;jrow=judge_row(ji,selected.prompt_id);record_id=selected.i2s_record_id;response_hash=selected.i2s_response_hash;finish=selected.i2s_finish_reason;out_tokens=selected.i2s_output_tokens;source_file=files["i2s_response"]
            original,head,tail,omitted=excerpts(text)
            panel_rows.append(dict(panel=panel,prompt_id=selected.prompt_id,prompt_sha256=selected.prompt_sha256,scenario=selected.scenario,step=selected.step,split=selected.split,
                ground_truth=selected.ground_truth,representation=representation,source_file=str(source_file.relative_to(self.p.root)),candidate_record_id=record_id,
                candidate_response_sha256=response_hash,text_total_characters=len(original),output_tokens=out_tokens,finish_reason=finish,excerpt_head=head,excerpt_tail=tail,
                omitted_characters=omitted,log_excerpt=log_excerpt(text) if kind=="prompt" and len(str(text).split("\n\n",1)[0])>600 else "",
                score_TI=first(jrow,"score_technique_identification"),score_EG=first(jrow,"score_explanation_grounding"),
                score_IA=first(jrow,"score_instruction_adherence","score_ instruction_adherence"),score_CLA=first(jrow,"score_confidence_label_agreement","confidence_label_agreement"),
                judge_extracted_predictions=first(jrow,"judge_extracted_predictions"),judge_failure_labels=first(jrow,"judge_failure_labels"),template_match=match(text),
                template_signature=signature(text),natural_bf16_condition_dropped=natural_fallback,regex=BITNET_TEMPLATE_REGEX))
        add("prompt_P1",p1,"prompt");add("bf16_P1",p1,"bf16","BF16 HF");add("i2s_P1",p1,"i2s","native I2_S");add("i2s_P2",p2,"i2s","native I2_S")
        examples=pd.DataFrame(panel_rows)
        example_path=self.p.figure_data/"04X_bitnet_template_example.csv";atomic_csv(examples,example_path);self.generated.append(example_path)

        positive=pd.read_csv(files["positive_register"],dtype={"prompt_id":str})
        stream_specs=[
            ("BitNet I2_S untreated","all_990","untreated",response("i2s_response"),files["i2s_response"]),
            ("BitNet I2_S untreated","evaluation_285","untreated",response("i2s_response"),files["i2s_response"]),
            ("BitNet I2_S SFT-full","evaluation_285","full",response("i2s_full_response"),files["i2s_full_response"]),
            ("BitNet BF16 untreated","all_990","untreated",response("bf16_response"),files["bf16_response"]),
        ]
        prevalence=[]
        for label,population,condition,frame,path in stream_specs:
            completed=frame[frame.status.eq("completed")].copy()
            if population=="evaluation_285":completed=completed[completed.prompt_id.isin(set(source[source.split.isin(["validation","test"])].prompt_id))]
            completed["template_match"]=completed.response_raw.fillna("").map(match)
            prefixes=completed.response_raw.fillna("").map(lambda x:str(x).replace("\t","    ").lstrip()[:100])
            is_one_bit="I2_S" in label
            if is_one_bit:
                preg=positive[(positive.model.eq("bitnet2b"))&(positive.condition_short.eq(condition))&positive.prompt_id.isin(set(completed.prompt_id))]
                positive_n=len(preg);positive_template=sum(match(x) for x in preg.candidate_response_excerpt.fillna(""))
            else:positive_n=positive_template=np.nan
            prevalence.append(dict(stream=label,population=population,completed_responses=len(completed),template_match_n=int(completed.template_match.sum()),
                template_match_rate=completed.template_match.mean() if len(completed) else np.nan,distinct_first_100_char_prefixes=prefixes.nunique(),
                positive_score_rows_n=positive_n,positive_score_rows_template_n=positive_template,regex=BITNET_TEMPLATE_REGEX,
                source_file=str(path.relative_to(self.p.root)),source_file_sha256=sha256(path)))
        prevalence=pd.DataFrame(prevalence);prevalence_path=self.p.tables/"bitnet_template_prevalence.csv";atomic_csv(prevalence,prevalence_path);self.generated.append(prevalence_path)

        source_hashes=json.dumps({str(path.relative_to(self.p.root)):sha256(path) for path in files.values()},sort_keys=True)
        manifest_path=self.p.tables/"figure_manifest.csv"
        manifest=pd.read_csv(manifest_path) if manifest_path.exists() else pd.DataFrame()
        if len(manifest) and "figure_id" in manifest:manifest=manifest[~manifest.figure_id.astype(str).isin(["04X","04X_prevalence"])]
        additions=pd.DataFrame([
            dict(figure_id="04X",figure_file="chapter_typeset_externally",figure_data_csv=str(example_path.relative_to(self.p.analysis)),key="Deterministically selected untreated BitNet BF16/native-I2_S prompt-response contrast.",source_tables="canonical results/final response, judgment and prompt CSVs",source_table_hashes=source_hashes),
            dict(figure_id="04X_prevalence",figure_file="chapter_typeset_externally",figure_data_csv=str(prevalence_path.relative_to(self.p.analysis)),key="Template prevalence and prefix diversity for BitNet control streams.",source_tables="canonical results/final response CSVs; one_bit_positive_score_row_register.csv",source_table_hashes=source_hashes),
        ])
        manifest=pd.concat([manifest,additions],ignore_index=True,sort=False);atomic_csv(manifest,manifest_path);self.generated.append(manifest_path)
        return {"example":examples,"prevalence":prevalence,"P1":str(p1.prompt_id),"P2":str(p2.prompt_id),"regex":BITNET_TEMPLATE_REGEX,"natural_bf16_condition_dropped":natural_fallback}

    def run_model_response(self):
        self.build_manifest(False);self.availability();self.score_summaries();self.contrasts();self.attack_and_failures();self.related_dimensions();self.repeat_judge_longtail_size();self.stopping_runtime();self.reference_extensions();self.judge_mediated_rank_metrics();self.training_subset_representativeness();self.split_label_balance();self.positive_frontier_and_training_exposure();self._full_universe_figures()

    def run_compute(self):
        self.compute_reporting();self.compute_analysis()

    def finalize(self) -> dict:
        manifest=self.build_manifest(True)
        catalog=[]
        existing_outputs=sorted([p for directory in (self.p.tables,self.p.figures) for p in directory.rglob("*") if p.is_file()])
        for path in existing_outputs:
            if path.exists():catalog.append({"path":str(path.relative_to(self.p.root)),"bytes":path.stat().st_size,"sha256":sha256(path)})
        self._write(pd.DataFrame(catalog),"analysis_output_catalog.csv")
        return self.build_manifest(True)


def render_reference_ladder_all_fields(root: str | Path) -> dict:
    """Render the compact seven-panel reference ladder from its canonical table.

    This lightweight renderer deliberately does not instantiate
    ``Notebook10Analysis`` or reload the full judgment matrix. It therefore
    remains safe to rerun after the larger per-dimension reference analysis.
    """
    root=Path(root).resolve();analysis_dir=root/"analysis";tables=analysis_dir/"tables";figures=analysis_dir/"figures"
    source=tables/"reference_ladder.csv"
    if not source.exists():raise FileNotFoundError("Run Notebook 10 Section 6A reference_extensions() first: "+str(source))
    ladder=pd.read_csv(source)
    dimension_abbreviations={
        "score_technique_identification":"TI","score_prediction_evidence_support":"PES",
        "score_explanation_grounding":"EG","score_prediction_reason_alignment":"PRA",
        "score_confidence_evidence_support":"CES","score_instruction_adherence":"IA",
        "confidence_label_agreement":"CLA",
    }
    missing=set(SCORE_COLUMNS)-set(ladder.dimension.astype(str))
    if missing:raise ValueError("reference_ladder.csv is missing dimensions: "+", ".join(sorted(missing)))
    if any(len(ladder[ladder.dimension.eq(score)])!=27 for score in SCORE_COLUMNS):
        raise ValueError("Figure 19A_all requires exactly 27 streams for every score dimension")
    model_short={
        "qwen1b":"Qwen 1.5B","hermes":"Hermes 1.8B","bitnet2b":"BitNet 2B",
        "llama3b":"Llama 3B","mistral7b":"Mistral 7B","qwen7b":"Qwen 7B",
    }
    core_style={
        ("BF16 HF","untreated"):("#0072B2","BF16 / untreated"),
        ("BF16 HF","full"):("#56B4E9","BF16 / SFT-full"),
        ("one-bit frontier","untreated"):("#D55E00","1-bit / untreated"),
        ("one-bit frontier","full"):("#CC79A7","1-bit / SFT-full"),
    }
    reference_style={
        "gpt55_teacher":("#7B3294","GPT-5.5 teacher\nreference"),
        "baseline_modal":("#E69F00","Modal baseline\nreference"),
        "baseline_null":("#777777","Null baseline\nreference"),
    }
    all_fields=ladder.copy();all_fields["population"]="evaluation_285"
    all_fields["dimension_abbreviation"]=all_fields.dimension.map(dimension_abbreviations)
    def ladder_style(row):
        if row.model in reference_style:return reference_style[row.model]
        color,second=core_style[(row.representation_group,row.condition)]
        return color,f"{model_short[row.model]}\n{second}"
    styles=all_fields.apply(ladder_style,axis=1)
    all_fields["stream_color"]=[item[0] for item in styles]
    all_fields["short_stream_label"]=[item[1] for item in styles]
    def marker_state(row):
        if int(row.eligible_n)==0:return "no_scorable_rows"
        exact_zero=bool(np.isfinite(row["mean"]) and np.isclose(float(row["mean"]),0.0,atol=1e-12))
        low_support=int(row.eligible_n)<30
        if exact_zero and low_support:return "exact_zero_low_support"
        if exact_zero:return "exact_zero"
        if low_support:return "low_support"
        return "estimated"
    all_fields["marker_state"]=all_fields.apply(marker_state,axis=1)
    all_fields=all_fields.sort_values(["dimension","display_order"],kind="stable").reset_index(drop=True)
    data_path=tables/"figure_data"/"19A_all__19A_all_reference_ladder_all_fields.csv"
    atomic_csv(all_fields,data_path)

    plt.close("all");gc.collect()
    fig,axes=plt.subplots(1,7,figsize=(11.0,15.2),sharey=False,gridspec_kw={"wspace":.15})
    ordered=all_fields[all_fields.dimension.eq(SCORE_COLUMNS[0])].sort_values("display_order",ascending=False)
    y=np.arange(len(ordered));labels=ordered.short_stream_label.tolist()
    for index,(ax,score) in enumerate(zip(axes,SCORE_COLUMNS)):
        plot=all_fields[all_fields.dimension.eq(score)].sort_values("display_order",ascending=False).reset_index(drop=True)
        for yi,row in plot.iterrows():
            state=row.marker_state
            if state=="no_scorable_rows":
                ax.text(.10,yi,"\N{EN DASH}",ha="center",va="center",color="#777777",fontsize=8)
                continue
            colour="#969696" if int(row.eligible_n)<30 else row.stream_color
            if np.isfinite(row.low95) and np.isfinite(row.high95):
                ax.plot([row.low95,row.high95],[yi,yi],color=colour,lw=.8,alpha=.9,zorder=1)
            exact_zero=state in {"exact_zero","exact_zero_low_support"}
            ax.scatter(row["mean"],yi,s=24,marker="o",facecolors="none" if exact_zero else colour,
                       edgecolors=colour,linewidths=1.0,zorder=2,clip_on=False)
        teacher=plot.loc[plot.model.eq("gpt55_teacher"),"mean"].dropna()
        modal=plot.loc[plot.model.eq("baseline_modal"),"mean"].dropna()
        if len(teacher):ax.axvline(teacher.iloc[0],color="#7B3294",ls="--",lw=.8)
        if len(modal):ax.axvline(modal.iloc[0],color="#E69F00",ls=":",lw=.9)
        ax.set_xlim(0,4);ax.set_ylim(-.5,len(plot)-.5);ax.set_xticks([0,2,4]);ax.set_yticks(y)
        ax.grid(axis="x",color="#E6E6E6",lw=.5);ax.set_title(dimension_abbreviations[score],fontsize=9,fontweight="bold")
        ax.set_xlabel("Mean score",fontsize=8);ax.tick_params(axis="x",labelsize=7)
        if index==0:ax.set_yticklabels(labels,fontsize=7)
        else:ax.set_yticklabels([]);ax.tick_params(axis="y",left=False)
    handles=[
        Patch(facecolor=core_style[("BF16 HF","untreated")][0],label="BF16 / untreated"),
        Patch(facecolor=core_style[("BF16 HF","full")][0],label="BF16 / SFT-full"),
        Patch(facecolor=core_style[("one-bit frontier","untreated")][0],label="1-bit / untreated"),
        Patch(facecolor=core_style[("one-bit frontier","full")][0],label="1-bit / SFT-full"),
        Line2D([0],[0],color="#7B3294",ls="--",lw=.9,label="Teacher mean"),
        Line2D([0],[0],color="#E69F00",ls=":",lw=1,label="Modal-baseline mean"),
        Line2D([0],[0],marker="o",mfc="none",mec="#333333",ls="none",label="Exact mean = 0"),
        Line2D([0],[0],marker="o",color="#969696",ls="none",label="Fewer than 30 scorable rows"),
        Line2D([0],[0],marker="$-$",color="#777777",ls="none",label="No scorable rows"),
    ]
    fig.suptitle("Reference ladder across all judge dimensions (285-row evaluation pool)",fontsize=12,y=.985)
    fig.text(.5,.075,"TI technique identification | PES prediction-evidence support | EG explanation grounding | PRA prediction-reason alignment | CES confidence-evidence support | IA instruction adherence | CLA confidence-label agreement",ha="center",va="center",fontsize=6.5)
    fig.legend(handles=handles,loc="lower center",ncol=3,bbox_to_anchor=(.5,.012),frameon=True,fontsize=7,title="Key",title_fontsize=7)
    fig.subplots_adjust(left=.235,right=.99,top=.955,bottom=.115,wspace=.15)
    figure_path=figures/"19A_all_reference_ladder_all_fields.png";savefig(figure_path,fig)

    key=("TI=technique identification; PES=prediction-evidence support; EG=explanation grounding; "
         "PRA=prediction-reason alignment; CES=confidence-evidence support; IA=instruction adherence; "
         "CLA=confidence-label agreement. Colours encode the four core streams; hollow markers are exact zero, "
         "grey markers have <30 scorable rows, and an en dash denotes no scorable rows. Dashed purple and dotted "
         "amber lines are the dimension-specific teacher and modal-baseline means.")
    manifest_path=tables/"figure_manifest.csv"
    manifest=pd.read_csv(manifest_path,dtype=str,keep_default_na=False) if manifest_path.exists() else pd.DataFrame()
    if len(manifest) and "figure_file" in manifest:manifest=manifest[manifest.figure_file.ne(figure_path.name)]
    addition=pd.DataFrame([dict(figure_id="19A_all",figure_file=figure_path.name,
        figure_data_csv=str(data_path.relative_to(analysis_dir)),key=key,
        source_tables="reference_ladder.csv; 07 teacher/baseline judgments; 03/07 candidate judgments",
        source_table_hashes=json.dumps({str(source.relative_to(root)):sha256(source)},sort_keys=True))])
    atomic_csv(pd.concat([manifest,addition],ignore_index=True,sort=False),manifest_path)
    catalog_path=tables/"analysis_output_catalog.csv"
    if catalog_path.exists():
        catalog=pd.read_csv(catalog_path,dtype=str,keep_default_na=False)
        relative={str(path.relative_to(root)) for path in (data_path,figure_path,manifest_path)}
        if "path" in catalog:catalog=catalog[~catalog.path.isin(relative)]
        new=pd.DataFrame([{"path":str(path.relative_to(root)),"bytes":path.stat().st_size,"sha256":sha256(path)} for path in (data_path,figure_path,manifest_path)])
        atomic_csv(pd.concat([catalog,new],ignore_index=True,sort=False),catalog_path)
    return {"figure":figure_path,"figure_data":data_path,"manifest":manifest_path,"rows":len(all_fields)}


def run_all(root: str | Path, bootstrap_replicates: int = BOOTSTRAP_REPLICATES) -> Notebook10Analysis:
    analysis=Notebook10Analysis(root,bootstrap_replicates=bootstrap_replicates)
    analysis.run_model_response();analysis.run_compute();analysis.finalize();return analysis
