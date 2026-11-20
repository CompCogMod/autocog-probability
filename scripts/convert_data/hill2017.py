"""Convert Hill (2017) "Learning Together Slowly" (J. Politics) to the shared schema.

Design (from 01_Analyze.R / 99_CallData.R / 03_RegressionModels-Exper2.do):
  * Exp 1 (MTurk, N=990): three "contests" per subject -- two partisan political
    facts (contest1, contest2; drawn from 6 statements) and one ego contest
    ("your IQ-quiz score is in the top half", stub selfcon). Each contest: an
    initial belief P(statement true) in 0-100 (round 1), then 4 binary signals
    (TRUE/FALSE), each 75% accurate (log LR = +/- ln 3), with a new belief after each.
  * Exp 2 (N=395): two contests per subject -- one abstract/non-political fact
    (Doha day length) and one partisan fact; each statement shown in a randomly
    assigned true or false wording. Same 4 x 75%-accurate signals.

One output row per posterior judgment (rounds 2-5 -> round 0-3). The round-1
(initial) belief is carried as prior_reported of round 0, so it is not a row.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import RAW, to_extra, write_dataset

DS = "hill2017"
RAWDIR = RAW / DS
ACC = 0.75  # signal accuracy

# Full statement texts and truth values (from 99_CallData.R `facts`).
FACTS = {
    "romney_share": ("In the 2012 Presidential Election, Barack Obama defeated the Republican Mitt Romney. In the nation as a whole, of all the votes cast for Obama and Romney, Romney won less than 48 percent.", 0, "dems"),
    "reagan_debt": ("The total public debt of the United States federal government more than doubled from quarter 2 in 1981 to quarter 1 in 1989 while Ronald Reagan was president.", 1, "dems"),
    "bush_unemploy": ("From January 2001, when President Bush first took office, to January 2005, when President Bush started his second term in office, the civilian unemployment rate increased by more than 1 percentage point.", 1, "dems"),
    "abortion": ("The rate at which American women aged 15-44 had legal abortions fell more between 1980 and 1988, while Ronald Reagan was president, than between 1992 and 2000, while Bill Clinton was president.", 0, "reps"),
    "obama_income": ("From 2009, when President Obama took office, to 2012, median household income adjusted for inflation in the United States fell by more than 4 percent.", 1, "reps"),
    "kerry_share": ("In the 2004 Presidential Election, John Kerry was defeated by George W. Bush. In the nation as a whole, of all the votes cast for Kerry and Bush, Kerry won less than 48 percent.", 0, "reps"),
}
TEXT2KEY = {v[0]: k for k, v in FACTS.items()}
IQ_TEXT = "Your score on the IQ quiz you just took is in the top half of respondents."


def signal_cols(df: pd.DataFrame, sig_true: pd.Series) -> pd.DataFrame:
    """sig_true: 1 if signal says hypothesis TRUE, 0 if FALSE, NaN if none."""
    df["signal_dir"] = np.where(sig_true.isna(), np.nan, np.where(sig_true == 1, 1.0, -1.0))
    df["p_signal_h"] = np.where(sig_true.isna(), np.nan, np.where(sig_true == 1, ACC, 1 - ACC))
    df["p_signal_not_h"] = np.where(sig_true.isna(), np.nan, np.where(sig_true == 1, 1 - ACC, ACC))
    df["signal_desc"] = np.where(sig_true.isna(), "",
                                 np.where(sig_true == 1, "signal: TRUE (75% accurate)",
                                          "signal: FALSE (75% accurate)"))
    df["n_signals"] = np.where(sig_true.isna(), np.nan, 1.0)
    return df


def exp1() -> pd.DataFrame:
    f = "AnalysisData-long.tab"
    long = pd.read_csv(RAWDIR / f, sep="\t")
    long["source_row"] = np.arange(len(long))
    wide = pd.read_csv(RAWDIR / "AnalysisData-wide.tab", sep="\t")
    pew = pd.read_csv(RAWDIR / "PewWeights.tab", sep="\t").rename(columns={"weight": "pew_weight"})
    wide = wide.merge(pew, on="resp.id", how="left")
    wide["quiz.score"] = pd.to_numeric(wide["quiz.score"].astype(str).str.strip(), errors="coerce")

    # fact text per contest
    ft = pd.concat([
        wide[["resp.id", "fact_contest1"]].rename(columns={"fact_contest1": "fact"}).assign(stub="contest1"),
        wide[["resp.id", "fact_contest2"]].rename(columns={"fact_contest2": "fact"}).assign(stub="contest2"),
    ])
    long = long.merge(ft, on=["resp.id", "stub"], how="left")
    subj_cols = [c for c in wide.columns if c not in ("fact_contest1", "fact_contest2",
                                                       "fact.pro.dem_contest1", "fact.pro.dem_contest2")]
    long = long.merge(wide[subj_cols], on="resp.id", how="left")
    long = long.sort_values(["resp.id", "stub", "round.num"]).reset_index(drop=True)

    p = long["pr.true"] / 100.0
    long["prior_reported"] = p.groupby([long["resp.id"], long["stub"]]).shift(1)
    long["posterior_reported"] = p

    is_iq = long["stub"] == "selfcon"
    key = long["fact"].map(TEXT2KEY)
    assert key[~is_iq].notna().all(), "unmatched fact text"
    long["item_id"] = np.where(is_iq, "iq_top_half", key)
    long["item_desc"] = np.where(is_iq, IQ_TEXT, long["fact"])
    long["task_domain"] = np.where(is_iq, "ego", "political_fact")
    long["hypothesis"] = np.where(is_iq, "own IQ-quiz score in top half", "statement true")
    truth = key.map(lambda k: FACTS[k][1] if isinstance(k, str) else np.nan)
    long["item_truth"] = np.where(is_iq, (long["quizAboveBelow"] == "Above").astype(float), truth)
    long["true_favors"] = key.map(lambda k: FACTS[k][2] if isinstance(k, str) else None)
    long["true_favors_party"] = np.select(
        [long["pid.summ"].eq("Democrat") & long["true_favors"].eq("dems"),
         long["pid.summ"].eq("Republican") & long["true_favors"].eq("reps"),
         long["pid.summ"].isin(["Democrat", "Republican"]) & long["true_favors"].notna()],
        [1.0, 1.0, 0.0], default=np.nan)
    long["condition"] = np.where(
        is_iq, np.where(long["iq.orig.qx.bottom"] == 1, "iq_question_asked_as_bottom_half",
                        "iq_question_asked_as_top_half"), "")

    sig = long["signal"].map({True: 1.0, False: 0.0, "TRUE": 1.0, "FALSE": 0.0})
    long = signal_cols(long, sig)
    long["rt_sec"] = long["time.submit"].where(long["time.submit"] >= 0)

    stub_order = {"contest1": 0, "contest2": 1, "selfcon": 2}
    long["sequence"] = long["stub"]
    long["round_num_raw"] = long["round.num"]
    long["flag_negative_rt"] = (long["time.submit"] < 0).astype(int)
    long["flag_missing_pew_weight"] = long["pew_weight"].isna().astype(int)
    long["flag_pew_weight_zero"] = (long["pew_weight"] == 0).astype(int)

    ex = ["sequence", "round_num_raw", "time.submit", "pr.true.prev", "pr.true.first.round",
          "did.not.update", "perf.bayes.cert", "cum.sig.comb", "item_truth", "true_favors",
          "true_favors_party", "pew_weight", "flag_negative_rt", "flag_missing_pew_weight",
          "flag_pew_weight_zero"] + [c for c in subj_cols if c not in ("resp.id", "pew_weight")]
    long["study"] = "exp1"
    long["source_file"] = f"{DS}/{f}"
    long["subject_id"] = DS + ":exp1:" + long["resp.id"].astype(str)
    long["extra"] = to_extra(long, ex)
    return long


def exp2() -> pd.DataFrame:
    f = "Exper2-DataForStata.tab"
    d = pd.read_csv(RAWDIR / f, sep="\t")
    d["source_row"] = np.arange(len(d))
    d["contest"] = np.where(d["abstrcont"] == 1, "abstract", "partisan")
    d = d.sort_values(["respid", "contest", "round"], kind="stable").reset_index(drop=True)

    # Orient to the statement as displayed: belieforig / signalorig.
    p = d["belieforig"] / 100.0
    d["prior_reported"] = p.groupby([d["respid"], d["contest"]]).shift(1)
    d["posterior_reported"] = p
    sig = d["signalorig"].map({True: 1.0, False: 0.0, "True": 1.0, "False": 0.0})
    d = signal_cols(d, sig)

    def item(s: str) -> tuple[str, str]:
        if s.startswith("On January 8, 2012"):
            return ("doha_daylength_" + s.split()[-1], s + " ... [text truncated in raw data]")
        if s.startswith("From 2009"):
            return ("obama_income", FACTS["obama_income"][0])
        if "more than doubled" in s:
            return ("reagan_debt", FACTS["reagan_debt"][0])
        if "cut by more than 50%" in s:
            return ("reagan_debt_cut50", s + " ... [text truncated in raw data]")
        raise ValueError(s)

    it = d["statem"].map(item)
    d["item_id"] = it.str[0]
    d["item_desc"] = it.str[1]
    d["task_domain"] = np.where(d["contest"] == "abstract", "other", "political_fact")
    d["hypothesis"] = "statement true"
    d["condition"] = d["contest"] + "_contest;statement_version=" + np.where(d["statemtrue"] == 1, "true", "false")
    d["item_truth"] = d["statemtrue"]
    ex = ["contest", "round", "statemtrue", "item_truth", "prtrue", "signalrecoded", "prtrueprev",
          "prtruefirstround", "didnotupdate", "perfbayescert", "cumsigcomb", "signalconswfirst",
          "pidsumm", "weight"]
    d["round_num_raw"] = d["round"]
    d["study"] = "exp2"
    d["source_file"] = f"{DS}/{f}"
    d["subject_id"] = DS + ":exp2:" + d["respid"].astype(str)
    d["rt_sec"] = np.nan
    d["round.num"] = d["round"]
    d["extra"] = to_extra(d, ex)
    return d


def main() -> None:
    out = []
    for d in (exp1(), exp2()):
        d = d[d["round.num"] >= 2].copy()  # round 1 = initial belief -> prior_reported
        d["round"] = d["round.num"] - 2
        d["trial"] = d.groupby("subject_id").cumcount()
        d["dataset"] = DS
        d["response_scale"] = "percent 0-100"
        d["prior_objective"] = np.nan
        out.append(d)
    df = pd.concat(out, ignore_index=True)
    write_dataset(df, DS)


if __name__ == "__main__":
    main()
