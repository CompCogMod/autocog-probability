"""Augenblick, Lazarus & Thaler (2025), "Overinference from Weak Signals and
Underinference from Strong Signals", QJE -- experimental studies 1a, 1b, 2.

Study 1a (Prolific, 2021): a card is drawn from a Green or a Purple deck (prior 1/2);
  subject states P(Green) in percent. 263 possible questions, each subject sees
  12 single-card (symmetric or asymmetric decks), 5 three-card, 2 uncertain-composition
  questions (+ attention check + 5 WTP questions, not beliefs). Built from the raw
  Qualtrics CSV (experiment1a_data.csv), replicating 01study1aCleaning.do, so that the
  52 attention-check failures are kept (flagged) and per-page timers are available.
  The Bayesian posterior per question comes from the authors' "bayesian" pseudo-respondent row.
Study 1b (Prolific, 2024): 2/3/4 decks (1 Green, 1-3 Blue), prior P(Green)=1/num_decks,
  one card drawn, 5 signal strengths x direction; 15 rounds. Authors' intermediate
  file experiment1b_data.dta (already restricted to 500 attention-check passers).
Study 2 (Prolific, 2024): NBA win-probability updating; 4 quarters x 4 updates.
  Authors' cleaned expt2_cleaned.dta (500 subjects); "Bayes" = NBA win-probability
  calculator (inpredictable.com) used by the paper as the benchmark.

Run from repo root: uv run python scripts/convert_data/augenblick2025.py
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from common import RAW, expit, logit, to_extra, write_dataset

DS = "augenblick2025"
BASE = "augenblick2025/Experiment/data"
N_LARGE, N_SMALL = 1665, 337


# --------------------------------------------------------------------------- 1a
def q1a_info(q: int) -> dict:
    """Design info for Study 1a question id q (1..258), following the do-files."""
    if q <= 194:
        m = (q - 1) % 97  # 0..96 within suit block
        suit = "Diamond" if q <= 97 else "Spade"
        large = q <= 33 or 66 <= q <= 81 or 98 <= q <= 130 or 163 <= q <= 178
        return dict(qtype="single_symmetric" if m <= 64 else "single_asymmetric", suit=suit,
                    N=N_LARGE if large else N_SMALL, n=1,
                    n_diamonds=1 if suit == "Diamond" else 0)
    if q <= 250:
        return dict(qtype="three_cards", suit=None, N=N_LARGE, n=3, n_diamonds=3 - (q + 1) % 4)
    return dict(qtype="uncertain_composition", suit=None, N=N_LARGE, n=1, n_diamonds=np.nan)


# uncertainty loops: question id -> raw loop column (01study1aCleaning.do renames)
UNC_COLS = {251: "1_uncertainty-d", 252: "4_uncertainty-s", 253: "5_uncertainty-d", 254: "8_uncertainty-s",
            255: "2_uncertainty-d", 256: "3_uncertainty-s", 257: "6_uncertainty-d", 258: "7_uncertainty-s"}
UNC_CODE = {1: 251, 2: 255, 3: 256, 4: 252, 5: 253, 6: 257, 7: 258, 8: 254}  # round24/25 code -> question


def q1a_cols(q: int) -> tuple[str, str, str]:
    """(green answer col, purple answer col, page-submit timer col) in the raw CSV."""
    if q <= 97:
        return f"{q}_Diamond_4", f"{q}_Diamond_5", f"{q}_time-diamond_Page Submit"
    if q <= 194:
        return f"{q}_Spade_4", f"{q}_Spade_5", f"{q}_time-spade_Page Submit"
    if q <= 250:
        k = q - 194
        return f"{k}_Q105_4", f"{k}_Q105_5", f"{k}_Q106_Page Submit"
    c = UNC_COLS[q]
    k, rest = c.split("_", 1)
    return f"{c}_4", f"{c}_5", f"{k}_time-{rest}_Page Submit"


def conv_1a() -> pd.DataFrame:
    rel = "experiment1a_data.csv"
    raw = pd.read_csv(RAW / BASE / rel, low_memory=False)
    raw = raw[raw["ResponseId"].notna()]
    bayes_row = raw[raw["ResponseId"] == "bayesian"].iloc[0]
    subj = raw[raw["ResponseId"] != "bayesian"]

    qs = list(range(1, 259))
    bayes = {}
    for q in qs:
        g, _, _ = q1a_cols(q)
        if q <= 250:
            bayes[q] = float(bayes_row[g])
        else:  # do-file hard-codes these
            bayes[q] = 100 * 2185 / 3330 if q <= 254 else 100 - 100 * 2185 / 3330

    # round (page) number for each subject x question (01study1aCleaning.do)
    recs = []
    for idx, r in subj.iterrows():
        rn = {}
        for k in list(range(1, 13)):
            rn[int(r[f"round{k}"])] = k
        for k in range(14, 19):
            rn[int(r[f"round{k}"]) + 194] = k
        for k in (24, 25):
            rn[UNC_CODE[int(r[f"round{k}"])]] = k
        for q in qs:
            g, p, t = q1a_cols(q)
            if pd.isna(r[g]):
                continue
            recs.append(dict(source_row=idx, rid=r["ResponseId"], question=q, answer=float(r[g]),
                             answer_purple=r[p], rt=r.get(t, np.nan), round_number=rn.get(q, np.nan)))
    d = pd.DataFrame(recs)
    d = d.merge(subj[["ResponseId", "attention_4", "attention_5", "age", "gender", "party", "education", "race",
                      "newscons", "newshelp", "crt-1", "crt-2", "crt-3", "Duration (in seconds)"]]
                .rename(columns={"ResponseId": "rid"}), on="rid", how="left")
    d["attention_pass"] = ((d["attention_4"] == 3) & (d["attention_5"] == 14)).astype(int)

    info = pd.DataFrame([dict(question=q, bayes_pct=bayes[q], **q1a_info(q)) for q in qs])
    d = d.merge(info, on="question", how="left")
    b = d["bayes_pct"] / 100
    N = d["N"].astype(float)

    # per-card P(diamond | Green), recovered from the Bayesian posterior (prior 1/2, mirrored decks)
    green_diam = pd.Series(np.nan, index=d.index)
    sym = d["qtype"] == "single_symmetric"
    dia = d["suit"] == "Diamond"
    green_diam[sym & dia] = np.round(b[sym & dia] * N[sym & dia])
    green_diam[sym & ~dia] = N[sym & ~dia] - np.round(b[sym & ~dia] * N[sym & ~dia])
    three = d["qtype"] == "three_cards"
    k3 = 2 * d.loc[three, "n_diamonds"] - 3
    green_diam[three] = np.round(expit(logit(b[three]) / k3) * N[three])
    uninf = sym & (d["bayes_pct"].round(6) == 50)  # q33/q130: uninformative card (Bayes = 50 exactly)
    green_diam[uninf] = np.nan
    d["green_diamonds"] = green_diam
    g = (green_diam / N).where(~uninf, 0.5)

    p_h = pd.Series(np.nan, index=d.index)
    p_nh = pd.Series(np.nan, index=d.index)
    p_h[sym & dia], p_nh[sym & dia] = g[sym & dia], 1 - g[sym & dia]
    p_h[sym & ~dia], p_nh[sym & ~dia] = 1 - g[sym & ~dia], g[sym & ~dia]
    nd = d.loc[three, "n_diamonds"]
    p_h[three] = g[three] ** nd * (1 - g[three]) ** (3 - nd)
    p_nh[three] = (1 - g[three]) ** nd * g[three] ** (3 - nd)
    unc = d["qtype"] == "uncertain_composition"
    # mirrored 50/50 mixtures of compositions: P(card|Green) = 1 - P(card|Purple) = Bayes posterior
    p_h[unc] = np.where(b[unc] > .5, 2185 / 3330, 1145 / 3330)
    p_nh[unc] = 1 - p_h[unc]
    unc_suit = d.loc[unc, "question"].map(lambda q: "Diamond" if UNC_COLS[q].endswith("-d") else "Spade")
    d.loc[unc, "suit"] = unc_suit

    out = pd.DataFrame(index=d.index)
    out["dataset"] = DS
    out["study"] = "1a"
    out["source_file"] = f"{BASE}/{rel}"
    out["source_row"] = d["source_row"].astype(int).astype(str) + ":q" + d["question"].astype(str)
    out["subject_id"] = f"{DS}:1a:" + d["rid"]
    out["round"] = 0
    out["condition"] = d["qtype"]
    out["task_domain"] = "abstract"
    out["hypothesis"] = "Green deck"
    out["prior_objective"] = 0.5
    out["p_signal_h"] = p_h
    out["p_signal_not_h"] = p_nh
    asym = d["qtype"] == "single_asymmetric"
    # asymmetric deck compositions are not recoverable from the data: LR from Bayes posterior (prior 1/2)
    out["log_lr"] = np.where(asym, logit(b), np.nan)
    out["posterior_bayes"] = b
    out["n_signals"] = d["n"]
    lr = np.where(asym, logit(b), np.log(p_h) - np.log(p_nh))
    out["signal_dir"] = np.sign(np.round(lr, 10))

    def sig_desc(r):
        if r.qtype == "three_cards":
            return f"3 cards drawn with replacement: {int(r.n_diamonds)} Diamond(s), {3 - int(r.n_diamonds)} Spade(s)"
        return f"1 card: {r.suit}"
    out["signal_desc"] = [sig_desc(r) for r in d.itertuples()]

    def desc(r):
        n = int(r.N)
        if r.qtype in ("single_symmetric", "three_cards"):
            if pd.isna(r.green_diamonds):
                comp = f"uninformative: Green and Purple decks have the same Diamond share ({n} cards)"
            else:
                k = int(r.green_diamonds)
                comp = f"Green deck {k} Diamonds/{n - k} Spades; Purple deck {n - k} Diamonds/{k} Spades"
        elif r.qtype == "single_asymmetric":
            comp = f"asymmetric Green/Purple decks ({n} cards; exact composition not in data)"
        else:
            comp = ("uncertain composition: each deck is one of two equally likely compositions (Purple mirrors Green); "
                    "exact compositions not in data; P(card|Green) = Bayes posterior")
        return f"{comp}; prior 1/2; {sig_desc(r)}; Bayes P(Green) = {r.bayes_pct:.2f}%"
    out["item_desc"] = [desc(r) for r in d.itertuples()]
    out["item_id"] = "1a_q" + d["question"].astype(str)
    out["posterior_reported"] = d["answer"] / 100
    out["response_scale"] = "percent 0-100 (Green and Purple boxes summing to 100)"
    out["rt_sec"] = pd.to_numeric(d["rt"], errors="coerce")
    d["trial"] = d.groupby("rid")["round_number"].rank(method="first") - 1
    out["trial"] = d["trial"]
    d["paper_excluded"] = 1 - d["attention_pass"]
    d["main_analysis_item"] = (sym & (d["bayes_pct"].round(4) != 50)).astype(int)
    d["uncertainty_same"] = np.where(unc, d["question"].isin([251, 252, 257, 258]).astype(float), np.nan)
    d["deck_size"] = d["N"]
    d["duration_sec"] = d["Duration (in seconds)"]
    d = d.rename(columns={"crt-1": "crt_1", "crt-2": "crt_2", "crt-3": "crt_3"})
    out["extra"] = to_extra(d, ["question", "round_number", "deck_size", "green_diamonds", "answer_purple",
                                "attention_pass", "paper_excluded", "main_analysis_item", "uncertainty_same",
                                "age", "gender", "party", "education", "race", "newscons", "newshelp",
                                "crt_1", "crt_2", "crt_3", "duration_sec"])
    return out


# --------------------------------------------------------------------------- 1b
# question -> (Green-favouring count, other count) out of 1665, from 03study1bAnalysis.do
LR1B = {1: (853, 812), 3: (916, 749), 5: (1036, 629), 7: (1294, 371), 9: (1419, 246)}


def conv_1b() -> pd.DataFrame:
    rel = "experiment1b_data.dta"
    full = pd.read_stata(RAW / BASE / rel, convert_categoricals=False)
    d = full[full["answer"].notna()].copy()
    q = d["question"].astype(int)
    nd = d["num_decks"].astype(int)
    base = ((q - 1) % 20) // 2 * 2 + 1  # 1,3,5,7,9 for each signal-strength pair
    favours = np.where(q <= 10, q % 2 == 1, q % 2 == 0)  # q1,3,..,9 & q22,24,..,30 favour Green
    hi = base.map(lambda x: LR1B[x][0])
    lo = base.map(lambda x: LR1B[x][1])
    p_h = np.where(favours, hi, lo) / 1665
    p_nh = np.where(favours, lo, hi) / 1665
    d["signal_cat"] = (base + 1) // 2

    # per-trial confidence and RT come from wide loop columns of this subject
    def pick(row, pat):
        c = pat.format(q=int(row.question), d=int(row.num_decks))
        return row[c] if c in row.index else np.nan
    d["confidence"] = [pick(r, "loop{q}_decks{d}_certainty_1") for _, r in d.iterrows()]
    d["rt"] = [pick(r, "loop{q}_time_decks{d}_pagesubmit") for _, r in d.iterrows()]
    d["rt_confidence"] = [pick(r, "loop{q}_time_decks{d}_certain_pagesu") for _, r in d.iterrows()]
    for j, s in enumerate(["5", "7", "8"], start=1):
        d[f"answer_blue{j}"] = [pick(r, "loop{q}_decks{d}_" + s) for _, r in d.iterrows()]

    out = pd.DataFrame(index=d.index)
    out["dataset"] = DS
    out["study"] = "1b"
    out["source_file"] = f"{BASE}/{rel}"
    out["source_row"] = d.index.astype(int)
    out["subject_id"] = f"{DS}:1b:" + d["responseid"].astype(str)
    out["trial"] = d["round_number"].astype(int) - 1
    out["round"] = 0
    out["condition"] = nd.astype(str) + "_decks"
    out["task_domain"] = "abstract"
    out["hypothesis"] = "Green deck"
    out["prior_objective"] = 1 / nd
    out["p_signal_h"] = p_h
    out["p_signal_not_h"] = p_nh
    out["n_signals"] = 1
    out["signal_dir"] = np.where(favours, 1, -1)
    out["signal_desc"] = np.where(favours, "1 card of the suit that is more common in the Green deck",
                                  "1 card of the suit that is less common in the Green deck")
    out["item_desc"] = ("1 Green deck + " + (nd - 1).astype(str) + " Blue deck(s), each 1665 cards; Green has "
                        + hi.astype(str) + " of one suit / " + lo.astype(str)
                        + " of the other, every Blue deck the mirror image; prior P(Green)=1/" + nd.astype(str)
                        + "; drawn card " + np.where(favours, "favours", "disfavours") + " Green (P(card|Green)="
                        + np.where(favours, hi, lo).astype(str) + "/1665, P(card|Blue)="
                        + np.where(favours, lo, hi).astype(str) + "/1665)")
    out["item_id"] = "1b_q" + q.astype(str) + "_d" + nd.astype(str)
    out["posterior_reported"] = d["answer"].astype(float) / 100
    out["response_scale"] = "percent 0-100 (one box per deck, summing to 100)"
    out["rt_sec"] = pd.to_numeric(d["rt"], errors="coerce")
    d["paper_excluded"] = 0
    d["deck_order"] = d["fl_27_do"]
    out["extra"] = to_extra(d, ["question", "question_deck", "round_number", "num_decks", "signal_cat",
                                "answer_blue1", "answer_blue2", "answer_blue3", "confidence", "rt_confidence",
                                "paper_excluded", "deck_order", "age", "gender", "education", "race",
                                "confusedlikert", "durationinseconds"])
    return out


# --------------------------------------------------------------------------- 2
S2_COLS = ["responseid", "quarter", "event", "loop", "bin_loop", "made_shot", "initial_score", "score_diff",
           "truth", "guess", "round_number", "block_number", "quarter_order", "time1", "time2", "time3", "time4",
           "time5", "nba_follow", "nba_champ", "nba_min", "nba_team", "nba_knowledge", "big_fan", "demo_age",
           "demo_gender", "demo_educ", "confidence_likert", "durationinseconds"]
QNAME = {1: "first", 2: "second", 3: "third", 4: "fourth"}


def conv_2() -> pd.DataFrame:
    rel = "expt2_cleaned.dta"
    full = pd.read_stata(RAW / BASE / rel, columns=S2_COLS, convert_categoricals=False)
    # keep the event-1 row (shown calculator probability) + the 4 elicited events of the shown sequence
    full = full[(full["guess"].notna()) | ((full["event"] == 1) & full["loop"].notna())]
    full = full.sort_values(["responseid", "quarter", "event"])
    full["truth_prev"] = full.groupby(["responseid", "quarter"])["truth"].shift(1)
    full["guess_prev"] = full.groupby(["responseid", "quarter"])["guess"].shift(1)
    full["score_prev"] = full.groupby(["responseid", "quarter"])["score_diff"].shift(1)
    d = full[full["guess"].notna()].copy()
    ev = d["event"].astype(int)
    shooter_is_h = ev.isin([3, 5])  # 04study2Cleaning.do: made shots add +2 at events 3,5, -2 at 2,4
    d["shooting_team"] = np.where(shooter_is_h, "H", "opponent")
    d["time_left"] = [r[f"time{int(r['event'])}"] for _, r in d.iterrows()]
    made = d["made_shot"].astype(int) == 1

    out = pd.DataFrame(index=d.index)
    out["dataset"] = DS
    out["study"] = "2"
    out["source_file"] = f"{BASE}/{rel}"
    out["source_row"] = d.index.astype(int)
    out["subject_id"] = f"{DS}:2:" + d["responseid"].astype(str)
    out["trial"] = d["round_number"].astype(int) - 1 - (d["block_number"].astype(int))  # 5 pages/block, 4 elicited
    out["round"] = ev - 2
    out["condition"] = "quarter" + d["quarter"].astype(int).astype(str)
    out["task_domain"] = "other"
    out["hypothesis"] = "team leading at start of quarter wins the NBA game"
    out["prior_objective"] = d["truth_prev"]
    out["prior_reported"] = np.where(ev >= 3, d["guess_prev"] / 100, np.nan)
    out["log_lr"] = logit(d["truth"]) - logit(d["truth_prev"])
    out["posterior_bayes"] = d["truth"]
    out["n_signals"] = 1
    out["signal_dir"] = np.sign(out["log_lr"].round(10))
    shooter = np.where(shooter_is_h, "team H", "opponent")
    out["signal_desc"] = ("with " + d["time_left"].astype(str) + " left in Q" + d["quarter"].astype(int).astype(str)
                          + ", " + shooter + np.where(made, " made a 2-point shot", " missed a shot")
                          + "; team H now leads by " + d["score_diff"].astype(int).astype(str))
    seq = d.apply(lambda r: r["bin_loop"][1:int(r["event"])], axis=1)
    out["item_id"] = ("2_q" + d["quarter"].astype(int).astype(str) + "_start" + d["initial_score"].astype(int).astype(str)
                      + "_e" + ev.astype(str) + "_" + seq)
    out["item_desc"] = ("NBA game, " + d["quarter"].astype(int).map(QNAME) + " quarter; team H made a shot to lead by "
                        + d["initial_score"].astype(int).astype(str) + " with 2:40 left (calculator P(win) shown); event "
                        + ev.astype(str) + "/5, made(1)/missed(0) sequence for events 2.." + ev.astype(str) + " = " + seq
                        + "; previous calculator P(H wins) = " + d["truth_prev"].round(3).astype(str)
                        + ", current = " + d["truth"].round(3).astype(str))
    out["posterior_reported"] = d["guess"].astype(float) / 100
    out["response_scale"] = "slider percent 0-100"
    d["paper_excluded"] = 0
    d["calculator_prob_prev"] = d["truth_prev"]
    d["calculator_prob"] = d["truth"]
    out["extra"] = to_extra(d, ["quarter", "event", "loop", "bin_loop", "initial_score", "score_diff", "made_shot",
                                "shooting_team", "time_left", "calculator_prob_prev", "calculator_prob",
                                "round_number", "block_number", "quarter_order", "paper_excluded",
                                "nba_follow", "nba_champ", "nba_min", "nba_team", "nba_knowledge", "big_fan",
                                "demo_age", "demo_gender", "demo_educ", "confidence_likert", "durationinseconds"])
    return out


if __name__ == "__main__":
    df = pd.concat([conv_1a(), conv_1b(), conv_2()], ignore_index=True)
    write_dataset(df, DS)
