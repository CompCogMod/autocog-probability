"""Shared trial-level schema for probability-updating datasets.

One row = one elicited probability judgment (a posterior, or a prior when the
row is a prior-only elicitation). All probabilities are on [0, 1] and refer to
P(H), where H is the hypothesis named in `hypothesis` for that row.

Each converter in this folder reads files under data/raw/<dataset>/ and writes
data/processed/<dataset>.csv.gz via `write_dataset`, which validates the schema.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
RAW = REPO / "data" / "raw"
PROCESSED = REPO / "data" / "processed"

# column -> description (kept in sync with data/processed/SCHEMA.md)
SCHEMA: dict[str, str] = {
    # provenance
    "dataset": "dataset key, e.g. 'hill2017'",
    "study": "study / experiment / wave label within the dataset",
    "source_file": "path of the raw file relative to data/raw/",
    "source_row": "0-based row index in the raw file (or identifying key if reshaped)",
    # who / when
    "subject_id": "globally unique subject id: '<dataset>:<study>:<original id>'",
    "trial": "0-based order of this judgment within the subject (NaN if unknown)",
    "round": "0-based update step within an item/sequence (0 = first posterior; -1 = prior-only row)",
    "condition": "between- or within-subject manipulation label(s); '' if none",
    # item
    "item_id": "identifier of the item / question / urn configuration",
    "item_desc": "human-readable item info (question text, urn composition, etc.)",
    "task_domain": "one of: abstract (urns/balls/bags/cards), ego (own ability/rank), political_fact, other",
    "hypothesis": "what the probability is about, e.g. 'urn A', 'statement true', 'top half'",
    # inputs
    "prior_objective": "P(H) given by the design before this step's evidence (NaN if none)",
    "prior_reported": "subject's own reported P(H) before this step's evidence (NaN if not elicited)",
    "signal_desc": "the evidence shown at this step, as text",
    "signal_dir": "+1 evidence favours H, -1 favours not-H, 0 neutral/uninformative, NaN unknown",
    "n_signals": "number of signals in this step's evidence",
    "p_signal_h": "P(observed evidence at this step | H)",
    "p_signal_not_h": "P(observed evidence at this step | not-H)",
    "log_lr": "ln[p_signal_h / p_signal_not_h] of this step's evidence",
    # outputs
    "posterior_reported": "subject's reported P(H) after the evidence (the judgment of this row)",
    "posterior_bayes": "Bayesian P(H) from prior_objective (or cumulative objective prior) and log_lr",
    "posterior_bayes_rep_prior": "Bayesian P(H) from prior_reported and log_lr",
    "response_scale": "original response format, e.g. 'percent 0-100', 'slider 0-1', 'distribution'",
    "rt_sec": "response time in seconds (NaN if unavailable)",
    "extra": "JSON object with any remaining trial/subject info (covariates, incentives, raw codes, ...)",
}
COLUMNS = list(SCHEMA)
TASK_DOMAINS = {"abstract", "ego", "political_fact", "other"}
PROB_COLS = ["prior_objective", "prior_reported", "p_signal_h", "p_signal_not_h",
             "posterior_reported", "posterior_bayes", "posterior_bayes_rep_prior"]


def logit(p):
    p = np.asarray(p, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(p) - np.log1p(-p)


def expit(x):
    x = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-x))


def bayes_posterior(prior, log_lr):
    """Posterior P(H) from prior P(H) and ln likelihood ratio (vectorised)."""
    return expit(logit(prior) + np.asarray(log_lr, dtype=float))


def to_extra(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    """Pack the given columns into a JSON string per row (NaN -> null)."""
    def clean(v):
        if v is None:
            return None
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating, float)):
            return None if math.isnan(v) else float(v)
        if isinstance(v, (pd.Timestamp,)):
            return v.isoformat()
        if v is pd.NA or v is pd.NaT:
            return None
        return v if isinstance(v, (int, str, bool, list, dict)) else str(v)
    recs = df[cols].to_dict(orient="records")
    return pd.Series([json.dumps({k: clean(v) for k, v in r.items()}, ensure_ascii=False)
                      for r in recs], index=df.index)


def finalize(df: pd.DataFrame) -> pd.DataFrame:
    """Add missing schema columns, fill derived Bayes columns, order columns."""
    df = df.copy()
    for c in COLUMNS:
        if c not in df:
            df[c] = np.nan
    for c in ["condition", "item_desc", "signal_desc", "hypothesis", "response_scale"]:
        df[c] = df[c].fillna("").astype(str)
    df["extra"] = df["extra"].fillna("{}").astype(str)
    lr_missing = df["log_lr"].isna() & df["p_signal_h"].notna() & df["p_signal_not_h"].notna()
    df.loc[lr_missing, "log_lr"] = np.log(df.loc[lr_missing, "p_signal_h"].astype(float)) - \
        np.log(df.loc[lr_missing, "p_signal_not_h"].astype(float))
    fill = df["posterior_bayes"].isna() & df["prior_objective"].notna() & df["log_lr"].notna()
    df.loc[fill, "posterior_bayes"] = bayes_posterior(df.loc[fill, "prior_objective"], df.loc[fill, "log_lr"])
    fill = df["posterior_bayes_rep_prior"].isna() & df["prior_reported"].notna() & df["log_lr"].notna()
    df.loc[fill, "posterior_bayes_rep_prior"] = bayes_posterior(df.loc[fill, "prior_reported"], df.loc[fill, "log_lr"])
    return df[COLUMNS]


def validate(df: pd.DataFrame) -> list[str]:
    problems = []
    if list(df.columns) != COLUMNS:
        problems.append(f"columns differ from schema: {set(df.columns) ^ set(COLUMNS)}")
        return problems
    for c in PROB_COLS:
        v = pd.to_numeric(df[c], errors="coerce")
        bad = v.notna() & ((v < 0) | (v > 1))
        if bad.any():
            problems.append(f"{c}: {bad.sum()} values outside [0,1]")
    bad_dom = ~df["task_domain"].isin(TASK_DOMAINS)
    if bad_dom.any():
        problems.append(f"task_domain: invalid values {df.loc[bad_dom, 'task_domain'].unique()[:5]}")
    for c in ["dataset", "study", "subject_id", "source_file", "item_id"]:
        if df[c].isna().any():
            problems.append(f"{c}: {df[c].isna().sum()} missing")
    if df["posterior_reported"].isna().all() and df["prior_reported"].isna().all():
        problems.append("no reported probabilities at all")
    for i, s in enumerate(df["extra"].head(200)):
        try:
            json.loads(s)
        except Exception as e:  # noqa: BLE001
            problems.append(f"extra row {i} not JSON: {e}")
            break
    return problems


def write_dataset(df: pd.DataFrame, name: str) -> Path:
    df = finalize(df)
    problems = validate(df)
    if problems:
        raise ValueError(f"{name} failed validation:\n  " + "\n  ".join(problems))
    PROCESSED.mkdir(parents=True, exist_ok=True)
    out = PROCESSED / f"{name}.csv.gz"
    df.to_csv(out, index=False)
    print(f"wrote {out.relative_to(REPO)}: {len(df)} rows, "
          f"{df['subject_id'].nunique()} subjects, {df['item_id'].nunique()} items")
    return out
