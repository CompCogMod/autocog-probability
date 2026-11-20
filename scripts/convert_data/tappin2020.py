"""Convert Tappin, Pennycook & Rand (2020, Cognition) "Bayesian or Biased?" data.

Design (Studies 1, 2, Control): 16 political statements (fact-checked true/false,
pro-Democratic/pro-Republican). Phase 1: per statement, subject rates P(true)
0-100 (prior), then sees a TRUE/FALSE signal/clue. Phase 2 (after all 16): rates
P(true) again without signals (posterior). One row per posterior judgment, with the
phase-1 prior stored as prior_reported.

- S1: signals described as accurate 2 of 3 times -> P(sig|H)=2/3 vs 1/3.
- S2: accuracy unstated ("between 51% and 99%"); subject rated per trial the
  likelihood that clues are accurate (LHR, 51-99). The paper's Bayesian benchmark
  uses that subjective LHR, so p_signal_* = subject's own LHR/100.
- Control: same as S1 but signals hidden (still randomly assigned) -> no evidence
  shown, signal_dir=0, log_lr=0; the paper's (hidden-signal) benchmark kept in extra.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common import RAW, to_extra, write_dataset

DS = "tappin2020"
FILES = {
    "S1": "tappin2020/studies1_2/S1_data.csv",
    "S2": "tappin2020/studies1_2/S2_data.csv",
    "control": "tappin2020/control_study/Control-Study_data.csv",
}
MOJIBAKE = {"\x89Ûª": "’", "\x89ÛÏ": "“", "\x89Û\x9d": "”"}
STYPE = {"FalseRep": (False, "pro-Republican"), "FalseDem": (False, "pro-Democratic"),
         "TrueRep": (True, "pro-Republican"), "TrueDem": (True, "pro-Democratic")}
CONDITION = {"S1": "signal_shown_acc2of3", "S2": "clue_shown_acc_unstated",
             "control": "signal_hidden"}

SUBJ_COLS = ["CRT_sum", "age", "gender", "education", "party", "party_choose", "social_issues",
             "econ_issues", "X2016vote", "Clinton_Trump", "media_trust", "God", "random",
             "CRT_familiarity", "conserv_rep", "party_z"]
TRIAL_COLS = ["statement_type", "internal_statementID", "statement_true", "statement_partisan",
              "signal_raw", "prior_raw", "post_raw", "prior_recode", "post_recode",
              "Bayes_posterior_paper", "difference", "ratio", "log_ratio", "capped",
              "signal_favors_REP", "politically_concordant", "prior_extremity",
              "prior_concordance", "mem", "memory_error", "signalRate", "signalRT", "LHR",
              "LHRRT", "priorRT", "signal_true_accuracy_design", "excl_capped",
              "flag_self_report_random", "flag_crt_familiar"]


def fix_text(s: str) -> str:
    for k, v in MOJIBAKE.items():
        s = s.replace(k, v)
    return s


def convert(study: str, rel: str) -> pd.DataFrame:
    raw = pd.read_csv(RAW / rel, encoding="latin-1")
    d = raw.copy()
    d["source_row"] = np.arange(len(d))
    sig_true = d["signal"].astype(str).str.upper().eq("TRUE")
    d["statement_true"] = d["statement_type"].map(lambda t: STYPE[t][0])
    d["statement_partisan"] = d["statement_type"].map(lambda t: STYPE[t][1])
    d["signal_raw"] = np.where(sig_true, "TRUE", "FALSE")
    d["prior_raw"], d["post_raw"] = d["prior"], d["post"]
    d["Bayes_posterior_paper"] = d["Bayes_posterior"] / 100
    d["signal_true_accuracy_design"] = 2 / 3  # S1/control stated; S2 empirically ~2/3
    d["excl_capped"] = d["capped"].eq(1)  # paper drops capped trials (update toward signal impossible)
    d["flag_self_report_random"] = d["random"].eq(1)
    d["flag_crt_familiar"] = d["CRT_familiarity"].eq(1)

    out = pd.DataFrame(index=d.index)
    out["dataset"] = DS
    out["study"] = study
    out["source_file"] = rel
    out["source_row"] = d["source_row"]
    out["subject_id"] = f"{DS}:{study}:" + d["PPID"].astype(str)
    out["trial"] = np.nan  # presentation order not recorded (rows sorted by statementID)
    out["round"] = 0
    out["condition"] = CONDITION[study]
    out["item_id"] = "stmt" + d["statementID"].astype(int).astype(str).str.zfill(2)
    out["item_desc"] = (d["statementtext"].map(fix_text) + " [fact-check: "
                        + np.where(d["statement_true"], "true", "false") + "; "
                        + d["statement_partisan"] + "]")
    out["task_domain"] = "political_fact"
    out["hypothesis"] = "statement true"
    out["prior_reported"] = d["prior"] / 100
    out["posterior_reported"] = d["post"] / 100
    out["n_signals"] = 1
    if study == "control":
        out["signal_desc"] = "no signal shown (signal hidden; assigned " + d["signal_raw"] + ")"
        out["signal_dir"] = 0
        out["log_lr"] = 0.0
    else:
        word = "signal" if study == "S1" else "clue"
        out["signal_desc"] = f"{word}: " + d["signal_raw"]
        out["signal_dir"] = np.where(sig_true, 1, -1)
        if study == "S1":
            acc = pd.Series(2 / 3, index=d.index)
        else:
            acc = d["LHR"] / 100  # subject's own judged clue accuracy on this trial
        out["p_signal_h"] = np.where(sig_true, acc, 1 - acc)
        out["p_signal_not_h"] = np.where(sig_true, 1 - acc, acc)
    out["response_scale"] = "percent 0-100 (posterior elicited in delayed 2nd phase, no signal)"
    out["rt_sec"] = d["postRT"]
    for c in SUBJ_COLS + TRIAL_COLS:
        if c not in d:
            d[c] = np.nan
    out["extra"] = to_extra(d, SUBJ_COLS + TRIAL_COLS)
    return out


def main():
    df = pd.concat([convert(s, f) for s, f in FILES.items()], ignore_index=True)
    write_dataset(df, DS)


if __name__ == "__main__":
    main()
