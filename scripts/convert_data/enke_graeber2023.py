"""Enke & Graeber (2023), "Cognitive Uncertainty", QJE -- belief-updating tasks.

Two bags: bag A (the hypothesis) contains a share `diagnosticity` of balls of the
"A colour" (black in A/B main files, red in the default-manipulation file); bag B
the complement. Prior P(bag A) is given; subject sees a sample of 1-5 balls and
states P(bag A) in percent, then a per-trial cognitive-uncertainty (CU) rating.

Studies:
  A_beliefs          A Experiments/.../Beliefs_cleaned.dta (500 subj x 12 rounds;
                     rounds 7-12 in 'complex numbers' arm show prior/diag as arithmetic)
  A_beliefs_default  A Experiments/.../Beliefs_Default_Manipulation_cleaned.dta
                     (108 subj x 12; 2-bag vs 10-bag framing, within subject, blocked)
  B_beliefs          B Experiments/.../Beliefs_cleaned.dta (1000 subj, ~6 tasks;
                     baseline, compound-diagnosticity (ROCL) and two/ten-state default arms)
Run from repo root: uv run python scripts/convert_data/enke_graeber2023.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common import RAW, to_extra, write_dataset

DS = "enke_graeber2023"
BASE = "enke_graeber2023/replication_files_rev2"


def read(rel: str) -> pd.DataFrame:
    return pd.read_stata(RAW / BASE / rel, convert_categoricals=False)


def common_cols(df, *, study, rel, sid, prior_pct, diag, n_a, n_b, col_a, col_b, belief):
    """Build the schema columns shared by all three files (probabilities -> [0,1])."""
    out = pd.DataFrame(index=df.index)
    out["dataset"] = DS
    out["study"] = study
    out["source_file"] = f"{BASE}/{rel}"
    out["source_row"] = df.index.astype(int)
    out["subject_id"] = f"{DS}:{study}:" + sid.astype(str)
    out["round"] = 0
    out["task_domain"] = "abstract"
    out["hypothesis"] = "bag A"
    prior = np.round(prior_pct.astype(float), 2) / 100
    p = np.round(diag.astype(float), 4)
    n_a = n_a.astype(int)
    n_b = n_b.astype(int)
    out["prior_objective"] = prior
    out["n_signals"] = n_a + n_b
    out["p_signal_h"] = p ** n_a * (1 - p) ** n_b
    out["p_signal_not_h"] = (1 - p) ** n_a * p ** n_b
    out["signal_dir"] = np.sign(n_a - n_b)
    out["signal_desc"] = n_a.astype(str) + f" {col_a}, " + n_b.astype(str) + f" {col_b}"
    out["posterior_reported"] = belief.astype(float) / 100
    out["response_scale"] = "percent 0-100"
    out["_diag"] = p
    out["_prior"] = prior
    out["_na"] = n_a
    out["_nb"] = n_b
    out["item_desc"] = ("bag A: " + (p * 100).round(0).astype(int).astype(str) + f"% {col_a} / bag B: "
                        + ((1 - p) * 100).round(0).astype(int).astype(str) + f"% {col_a}; P(A)="
                        + (prior * 100).round(2).astype(str) + "%; drew " + out["signal_desc"])
    return out


def item_id(out, variant):
    return (variant + "|pr" + (out["_prior"] * 100).round(2).astype(str)
            + "_d" + (out["_diag"] * 100).round(0).astype(int).astype(str)
            + "_" + out["_na"].astype(str) + "a" + out["_nb"].astype(str) + "b")


SUBJ_COVS = ["age", "female", "college", "education", "raven_score"]


def conv_a_main():
    rel = "A Experiments/data/Beliefs/Beliefs_cleaned.dta"
    d = read(rel)
    out = common_cols(d, study="A_beliefs", rel=rel, sid=d["participantcode"], prior_pct=d["prior"],
                      diag=d["diagnosticity"], n_a=d["number_blacks"], n_b=d["number_whites"],
                      col_a="black", col_b="white", belief=d["belief"])
    out["trial"] = d["round"].astype(int) - 1
    cx = d["is_complex"] == 1
    out["condition"] = np.where(cx, "complex_numbers", "baseline")
    variant = pd.Series(np.where(cx, "complex", "simple"), index=d.index)
    out["item_id"] = item_id(out, variant)
    out.loc[cx, "item_desc"] += "; prior shown as '" + d.loc[cx, "complex_prob"] + "', diagnosticity as '" + d.loc[cx, "complex_diag"] + "'"
    out["rt_sec"] = d["time_spent_belief"].astype(float)
    e = d.assign(
        cognitive_uncertainty=d["cognitive_uncertainty"].astype(float),
        cu_scale="0-1 (steps .05); A-wave: 1 - stated certainty (0-100%) that optimal answer is within +/-1pp of own answer",
        complex_numbers_arm=d["treatment"], is_complex_trial=d["is_complex"],
        rt_cu_sec=d["time_spent_Confidence"], bayesian_posterior_orig=d["bayesian_posterior"],
        bag_A_drawn=d["blackish_urn_drawn"], subj_mean_abs_inconsistency=d["avg_inconsis"],
        paper_main_sample=(d["treatment"] == 0).astype(int),
    )
    cols = ["cognitive_uncertainty", "cu_scale", "complex_numbers_arm", "is_complex_trial", "complex_prob",
            "complex_diag", "rt_cu_sec", "bayesian_posterior_orig", "bag_A_drawn", "subj_mean_abs_inconsistency",
            "clicked_payoff_info", "app_name", "session", "total_time_whole_study", "paper_main_sample", *SUBJ_COVS]
    out["extra"] = to_extra(e, cols)
    return out


def conv_a_default():
    rel = "A Experiments/data/Beliefs/Beliefs_Default_Manipulation_cleaned.dta"
    d = read(rel)
    out = common_cols(d, study="A_beliefs_default", rel=rel, sid=d["participantcode"], prior_pct=d["prior"],
                      diag=d["diagnosticity"], n_a=d["number_reds"], n_b=d["number_blues"],
                      col_a="red", col_b="blue", belief=d["belief"])
    out["trial"] = d["round"].astype(int) - 1
    out["condition"] = np.where(d["bags"] == 10, "ten_bags_low_default", "two_bags")
    out["item_id"] = item_id(out, pd.Series(np.where(d["bags"] == 10, "bags10", "bags2"), index=d.index))
    out["item_desc"] += "; draw sequence " + d["draw"] + "; framed with " + d["bags"].astype(str) + " bags"
    e = d.assign(
        cognitive_uncertainty=d["cognitive_uncertainty"].astype(float),
        cu_scale="0-1 (steps .05); same elicitation as A_beliefs (1 - certainty within +/-1pp)",
        bayesian_posterior_orig=d["bayesian_posterior"], bag_A_drawn=d["redish_urn_drawn"],
    )
    cols = ["cognitive_uncertainty", "cu_scale", "bags", "bags_first", "low_default", "draw",
            "bayesian_posterior_orig", "bag_A_drawn", "paying_round", "clicked_payoff_info", *SUBJ_COVS]
    out["extra"] = to_extra(e, cols)
    return out


def conv_b():
    rel = "B Experiments/data/Beliefs/Beliefs_cleaned.dta"
    d = read(rel)
    out = common_cols(d, study="B_beliefs", rel=rel, sid=d["id"].astype(int), prior_pct=d["prior"],
                      diag=d["diagnosticity"] / 100, n_a=d["number_blacks"], n_b=d["number_whites"],
                      col_a="black", col_b="white", belief=d["belief"])
    out["trial"] = np.nan  # no task-order variable in the B file
    rocl = d["rocl"] == 1
    out["condition"] = np.where(rocl, "compound_diagnosticity",
                                np.where(d["default_condition"] == "ten", "ten_states_low_default", "two_states"))
    out["item_id"] = item_id(out, pd.Series(np.where(rocl, "compound", "simple"), index=d.index))
    out.loc[rocl, "item_desc"] += "; diagnosticity described as a compound lottery"
    paper_excl = (d["outlier"] == 1)
    e = d.assign(
        cognitive_uncertainty=d["cognitive_uncertainty"].astype(float),
        cu_scale="0-1 = uncertainty_slider/30; B-wave: slider 0-30 = +/- interval (pp) around own answer within which subject is sure the optimal answer lies",
        bayesian_posterior_orig=d["bayesian_posterior"], wtp_for_bayes=d["wtp"],
        paper_excluded=paper_excl.astype(int),
        total_time_beliefs_subject=d["total_time_beliefs"],
    )
    cols = ["cognitive_uncertainty", "cu_scale", "uncertainty_slider", "cognitive_sd", "censored_cu",
            "censored_choice", "bayesian_posterior_orig", "wtp_for_bayes", "expected_payoff", "default_condition",
            "rocl", "treatment", "baseline_set", "rocl_set", "default_set", "outlier", "speeder",
            "paper_excluded", "total_time_beliefs_subject", "session_no", *SUBJ_COVS]
    out["extra"] = to_extra(e, cols)
    return out


def main():
    df = pd.concat([conv_a_main(), conv_a_default(), conv_b()], ignore_index=True)
    with np.errstate(divide="ignore"):
        df["log_lr"] = np.log(df["p_signal_h"]) - np.log(df["p_signal_not_h"])
    df = df.drop(columns=[c for c in df if c.startswith("_")])
    write_dataset(df, DS)


if __name__ == "__main__":
    main()
