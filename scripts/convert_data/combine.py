"""Validate every per-dataset file and stack them into data/processed/all_datasets.csv.gz.

Usage (from repo root):
    uv run python scripts/convert_data/combine.py            # combine existing outputs
    uv run python scripts/convert_data/combine.py --rebuild  # rerun every converter first
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pandas as pd

from common import COLUMNS, PROCESSED, REPO, validate

DATASETS = ["hill2017", "tappin2020", "buser2018", "augenblick2025",
            "enke_graeber2023", "xie_hayes2022", "howe2022"]


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    def corr(g: pd.DataFrame, col: str) -> float:
        ok = g[col].notna() & g["posterior_reported"].notna() & np.isfinite(g[col])
        return np.corrcoef(g.loc[ok, "posterior_reported"], g.loc[ok, col])[0, 1] if ok.sum() > 2 else np.nan

    def per(g: pd.DataFrame) -> pd.Series:
        return pd.Series({
            "rows": len(g),
            "subjects": g["subject_id"].nunique(),
            "items": g["item_id"].nunique(),
            "domains": ",".join(sorted(g["task_domain"].unique())),
            "pct_prior_obj": 100 * g["prior_objective"].notna().mean(),
            "pct_prior_rep": 100 * g["prior_reported"].notna().mean(),
            "pct_lr": 100 * g["log_lr"].notna().mean(),
            "pct_post_rep": 100 * g["posterior_reported"].notna().mean(),
            "pct_bayes": 100 * g["posterior_bayes"].notna().mean(),
            "r_rep_bayes": corr(g, "posterior_bayes"),
            "r_rep_bayes_rep_prior": corr(g, "posterior_bayes_rep_prior"),
        })
    return df.groupby(["dataset", "study"]).apply(per, include_groups=False).round(2)


def main() -> None:
    if "--rebuild" in sys.argv:
        for name in DATASETS:
            subprocess.run([sys.executable, str(REPO / "scripts/convert_data" / f"{name}.py")], check=True)
    frames = []
    for name in DATASETS:
        path = PROCESSED / f"{name}.csv.gz"
        if not path.exists():
            print(f"MISSING {path.relative_to(REPO)}")
            continue
        d = pd.read_csv(path, low_memory=False)
        problems = validate(d)
        if problems:
            raise SystemExit(f"{name}: " + "; ".join(problems))
        frames.append(d)
    allx = pd.concat(frames, ignore_index=True)[COLUMNS]
    # wide raw files (one row per subject) map many judgments to one source_row, so key on item too
    # (and on trial: a subject can get the same randomly generated practice item twice)
    key = ["dataset", "source_file", "source_row", "subject_id", "item_id", "round", "trial"]
    dup = allx.duplicated(key).sum()
    if dup:
        print(f"WARNING: {dup} duplicate {key} keys")
    out = PROCESSED / "all_datasets.csv.gz"
    allx.to_csv(out, index=False)
    summ = summarize(allx)
    summ.to_csv(PROCESSED / "summary.csv")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(summ)
    print(f"\nwrote {out.relative_to(REPO)}: {len(allx)} rows")


if __name__ == "__main__":
    main()
