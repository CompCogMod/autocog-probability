"""Convert Xie & Hayes (2022, Cognitive Science) "Sensitivity to evidential
dependencies in judgments under uncertainty" to the shared trial-level schema.

Task (all experiments): two jars, 100/120 balls of the jar's colour (P(ball of
jar colour) = 5/6). n = 3/5/7 "friends" each privately draw a ball and all
guess the x-coloured jar; then the participant draws one ball of the opposite
colour y. A rating follows each testimony and the own ball (n + 1 ratings).
  independent: friends only see their own ball
  sequential : friends see their own ball AND hear all previous friends' guesses
               (an information cascade -> later guesses carry less evidence)
Practice trials (expt2, expt3): 12 draws with replacement from a hidden jar,
then one rating.

Hypothesis for main-task rows: H = the y jar (colour of the participant's own
ball; the jar *against* the unanimous testimony). Authors' reverse-coded scale
(responseR, 10 = y jar) matches this. Practice rows: H = the jar the 12 balls
were actually drawn from.

Normative model: Whalen, Griffiths & Buchsbaum (2017) probability-matching
informants, as implemented by the authors (Model Code & Predictions/*.Rmd).
Step likelihoods are computed here by recursion and checked against the
authors' modelPreds csvs.
"""

from __future__ import annotations

import glob
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RAW, to_extra, write_dataset  # noqa: E402

DS = "xie_hayes2022"
BASE = RAW / DS / "Data & Data Analysis"
Q = 5 / 6  # P(ball of jar's colour | jar)


def rel(p: Path) -> str:
    return str(Path(p).relative_to(RAW))


# --------------------------------------------------------------------------- #
# normative model
# --------------------------------------------------------------------------- #
def testimony_likelihoods(n: int, dep: str) -> list[tuple[float, float]]:
    """For k = 1..n: (P(guess_k = x | x jar, hist), P(guess_k = x | y jar, hist)),
    with probability-matching informants. dep in {'Independent','Sequential'}."""
    ball = {"X": {"X": Q, "Y": 1 - Q}, "Y": {"X": 1 - Q, "Y": Q}}  # ball[urn][ball]
    cum = {"X": 1.0, "Y": 1.0}  # product of previous testimony likelihoods
    out = []
    for _ in range(n):
        L = {}
        for urn in ("X", "Y"):
            tot = 0.0
            for b in ("X", "Y"):
                if dep == "Sequential":
                    wx, wy = ball["X"][b] * cum["X"], ball["Y"][b] * cum["Y"]
                else:
                    wx, wy = ball["X"][b], ball["Y"][b]
                tot += ball[urn][b] * wx / (wx + wy)  # P(ball b | urn) * P(say x | b, hist)
            L[urn] = tot
        out.append((L["X"], L["Y"]))
        cum = {u: cum[u] * L[u] for u in cum}
    return out


def design_rows(n: int, dep: str) -> pd.DataFrame:
    """One row per rating q = 1..n+1 with normative step quantities (H = y jar)."""
    lik = testimony_likelihoods(n, dep)
    lik_naive = testimony_likelihoods(n, "Independent")
    rows, post, post_naive = [], 0.5, 0.5
    for q in range(1, n + 2):
        if q <= n:
            p_h, p_nh = lik[q - 1][1], lik[q - 1][0]  # P(guess x | y jar), P(guess x | x jar)
            pn_h, pn_nh = lik_naive[q - 1][1], lik_naive[q - 1][0]
            dirn = -1
        else:
            p_h, p_nh = Q, 1 - Q  # own ball is y
            pn_h, pn_nh = Q, 1 - Q
            dirn = 1
        prior = post
        post = prior * p_h / (prior * p_h + (1 - prior) * p_nh)
        post_naive = post_naive * pn_h / (post_naive * pn_h + (1 - post_naive) * pn_nh)
        rows.append(dict(informants=n, dependency=dep, question=q, prior_objective=prior,
                         p_signal_h=p_h, p_signal_not_h=p_nh, signal_dir=dirn,
                         posterior_model=post, posterior_naive_independent=post_naive,
                         cum_log_lr=float(np.log(post / (1 - post)))))
    return pd.DataFrame(rows)


DESIGN = pd.concat([design_rows(n, d) for n in (3, 5, 7) for d in ("Independent", "Sequential")],
                   ignore_index=True)


def authors_preds() -> pd.DataFrame:
    parts = []
    for n in (3, 5, 7):
        f = pd.read_csv(BASE / "csv" / f"modelPreds_{n}-1xtime.csv")
        parts.append(pd.DataFrame(dict(informants=n, dependency=f["dependency"], question=f["time"],
                                       model_posterior_authors=f["posterior"])))
    p = pd.concat(parts)
    # the 3/5-informant csvs are padded to 8/12 rows; keep the ones that fit the design
    p = p[p["question"] <= p["informants"] + 1]
    # expt3 file has corrected final-rating predictions for 3 and 5 informants
    e3 = pd.read_csv(BASE / "expt3" / "csv2" / "expt3_modelpredictions.csv")
    e3 = e3.assign(dependency=e3["dependency"].map({"i": "Independent", "sq": "Sequential"}))
    e3 = e3.rename(columns={"posterior": "model_posterior_authors_e3"})
    p = p.merge(e3, on=["informants", "dependency", "question"], how="left")
    p["model_posterior_authors"] = p["model_posterior_authors_e3"].fillna(p["model_posterior_authors"])
    p = p.drop(columns="model_posterior_authors_e3")
    e2 = pd.read_csv(BASE / "expt2" / "csv" / "expt12_modelpreds_log-odds.csv")
    e2 = e2.rename(columns={"in-js-scale": "model_pred_odds_scale_point"})
    p = p.merge(e2[["informants", "dependency", "question", "model_pred_odds_scale_point"]],
                on=["informants", "dependency", "question"], how="left")
    return p


DESIGN = DESIGN.merge(authors_preds(), on=["informants", "dependency", "question"], how="left")
_chk = DESIGN.dropna(subset=["model_posterior_authors"])
_err = (_chk["posterior_model"] - _chk["model_posterior_authors"]).abs()
print(f"normative recursion vs authors' modelPreds: max |diff| = {_err.max():.2e} over {len(_chk)} cells")
if _err.max() > 1e-6:
    print(_chk.loc[_err > 1e-6, ["informants", "dependency", "question", "posterior_model",
                                 "model_posterior_authors"]])


def add_design(df: pd.DataFrame) -> pd.DataFrame:
    """df has informants, dependency, question, y_colour, x_colour."""
    keys = ["informants", "dependency", "question"]
    df = df.drop(columns=[c for c in DESIGN.columns if c in df.columns and c not in keys])
    df = df.merge(DESIGN, on=["informants", "dependency", "question"], how="left", validate="m:1")
    n, q = df["informants"].astype(int), df["question"].astype(int)
    dep_short = df["dependency"].map({"Independent": "ind", "Sequential": "seq"})
    df["item_id"] = n.astype(str) + "inf-" + dep_short + "-q" + q.astype(str)
    dep_txt = df["dependency"].map({
        "Independent": "independent: each friend saw only their own ball",
        "Sequential": "sequential: each friend saw own ball and heard all previous friends' guesses"})
    df["item_desc"] = ("Two jars (100/120 balls of jar colour). " + n.astype(str)
                       + " friends each draw a ball and all guess the x jar (" + dep_txt
                       + "); then participant draws one y ball. Rating " + q.astype(str) + " of "
                       + (n + 1).astype(str) + ".")
    own = q > n
    df["signal_desc"] = np.where(
        own, "participant's own ball is " + df["y_colour"] + " (y)",
        "friend " + q.astype(str) + " of " + n.astype(str) + " guesses the " + df["x_colour"]
        + " jar (" + dep_short + ")")
    df["stage"] = np.where(own, "own_ball", "testimony")
    df["round"] = q - 1
    df["n_signals"] = 1
    df["hypothesis"] = "y jar (colour of participant's own ball; against the friends' guesses)"
    df["task_domain"] = "abstract"
    df["condition"] = df["dependency"].str.lower() + "|" + n.astype(str) + "_informants"
    return df


def add_prior_reported(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    df = df.sort_values(keys + ["question"])
    df["prior_reported"] = df.groupby(keys)["posterior_reported"].shift(1)
    return df


def practice_row(balls: list[str], jar: str) -> dict:
    k = sum(b == jar for b in balls)
    m = len(balls) - k
    return dict(item_id=f"practice-12draws-{k}jar-{m}other",
                item_desc=f"Practice: 12 draws with replacement from a hidden jar (two jars, 100/120 of "
                          f"own colour); {k} of the jar's colour, {m} of the other colour.",
                signal_desc="12 draws: " + ",".join(balls),
                signal_dir=float(np.sign(k - m)), n_signals=len(balls),
                p_signal_h=Q ** k * (1 - Q) ** m, p_signal_not_h=(1 - Q) ** k * Q ** m,
                prior_objective=0.5, round=0, stage="practice",
                hypothesis="the jar the practice balls were drawn from",
                task_domain="abstract", condition="practice")


EXTRA_DESIGN = ["informants", "dependency", "question", "stage", "x_colour", "y_colour",
                "posterior_model", "model_posterior_authors", "posterior_naive_independent",
                "model_pred_odds_scale_point", "cum_log_lr"]

# --------------------------------------------------------------------------- #
# Experiment 1: 0-10 likelihood scale, between-subjects dependency x informants
# --------------------------------------------------------------------------- #
def expt1() -> pd.DataFrame:
    parts = []
    for tag, fname in (("a", "expt1_data_1of2.csv"), ("b", "expt1_data_2of2.csv")):
        f = BASE / "csv" / fname
        d = pd.read_csv(f)
        d["source_row"] = d.index
        d["source_file"] = rel(f)
        d["file_part"] = tag
        d = d.dropna(subset=["responseR"])  # 3-informant subjects padded with NA rows to Q8
        parts.append(d)
    d = pd.concat(parts, ignore_index=True)
    d["dependency"] = d["dependency"].map({"i": "Independent", "sq": "Sequential"})
    d["y_colour"] = d["colourPart"]
    d["x_colour"] = np.where(d["colourPart"] == "red", "blue", "red")
    d = add_design(d)
    d["study"] = "expt1"
    d["subject_id"] = f"{DS}:expt1:" + d["file_part"] + "-" + d["subj_id"].astype(str)
    d["trial"] = d["question"] - 1
    d["posterior_reported"] = d["responseR"] / 10
    d["response_scale"] = "likert 0-10 (0 = definitely x jar, 10 = definitely y jar; reverse-coded from red/blue)"
    d = add_prior_reported(d, ["subject_id"])
    d["response_raw"] = d["response"]
    d["response_r"] = d["responseR"]
    d["excluded_by_authors"] = False
    d["exclusion_note"] = "authors' csv already excludes 5 attention-check failures"
    d["extra"] = to_extra(d, EXTRA_DESIGN + ["gender", "age", "language", "country", "file_part",
                                             "response_raw", "response_r", "excluded_by_authors",
                                             "exclusion_note"])
    return d


# --------------------------------------------------------------------------- #
# Experiment 2: log-odds scale, incentivised; raw jsPsych files per participant
# --------------------------------------------------------------------------- #
ODDS = {0: 1 / 40, 1: 1 / 20, 2: 1 / 10, 3: 1 / 5, 4: 1 / 2, 5: 1.0,
        6: 2.0, 7: 5.0, 8: 10.0, 9: 20.0, 10: 40.0}  # odds(y jar : x jar) for responseR point


def odds_point_to_p(pt):
    o = pd.Series(pt).map(ODDS)
    return (o / (1 + o)).to_numpy()


def expt2() -> pd.DataFrame:
    rows = []
    files = sorted(glob.glob(str(BASE / "expt2" / "csv-participants" / "*.csv")))
    for fp in files:
        d = pd.read_csv(fp)
        if "response" not in d:
            continue  # no slider ratings recorded
        meta = d.iloc[0]
        attn = d.loc[d["trial_type"] == "survey-text", "responses"] if "responses" in d else pd.Series([], dtype=object)
        attn_resp = None
        if len(attn):
            m = re.search(r'"Q0":"(.*)"', str(attn.iloc[0]))
            attn_resp = m.group(1) if m else str(attn.iloc[0])
        sl = d[d["trial_type"] == "image-slider-response"]
        n_main = int(meta["informants"]) + 1
        main_labels = [f"Q{i}" for i in range(1, n_main)] + ["Q_final"]
        complete = all(any(lab in str(s) for s in sl["stimulus"]) for lab in main_labels)
        base = dict(turkcode=str(meta["turkcode"]), gender=meta["gender"], age=meta["age"],
                    language=meta["language"], country=meta["country"], colourPart=meta["colourPart"],
                    informants=int(meta["informants"]), dependency=meta["dependency"],
                    incentive=bool(meta["incentive"]), scale=meta["scale"], attn_response=attn_resp,
                    complete=complete, source_file=rel(Path(fp)))
        inc = sl[sl["stimulus"].astype(str).str.contains("explain-incentive")]
        base["incentive_check_raw"] = float(inc["response"].iloc[0]) if len(inc) else np.nan
        # practice trials
        balls, trial_no = [], 0
        for idx, r in d.iterrows():
            s = str(r["stimulus"])
            mb = re.search(r"You sampled a <font color=(\w+)>", s)
            if mb:
                balls.append(mb.group(1))
                continue
            mp = re.search(r"PQ(\d)", s)
            if r["trial_type"] == "image-slider-response" and mp:
                # find revealed jar on the next "you sampled from the X jar" screen
                after = d.loc[idx + 1:, "stimulus"].astype(str)
                jar = after.str.extract(r"you sampled from the <font color=(\w+)>")[0].dropna()
                jar = jar.iloc[0] if len(jar) else None
                raw = float(r["response"])
                rec = dict(base, **practice_row(balls[-12:], jar), source_row=idx,
                           trial=trial_no, rt_sec=float(r["rt"]) / 1000, response_raw=raw,
                           practice_jar=jar, practice_trial=int(mp.group(1)),
                           posterior_reported=odds_point_to_p([raw if jar == "red" else 10 - raw])[0])
                rows.append(rec)
                trial_no += 1
                balls = []
                continue
            mq = re.search(r">Q(\d|_final)<", s)
            if r["trial_type"] == "image-slider-response" and mq:
                q = n_main if mq.group(1) == "_final" else int(mq.group(1))
                raw = float(r["response"])
                rr = 10 - raw if meta["colourPart"] == "blue" else raw
                rows.append(dict(base, question=q, source_row=idx, trial=trial_no,
                                 rt_sec=float(r["rt"]) / 1000, response_raw=raw, response_r=rr,
                                 posterior_reported=odds_point_to_p([rr])[0],
                                 time_elapsed_ms=float(r["time_elapsed"])))
                trial_no += 1
    d = pd.DataFrame(rows)
    d["study"] = "expt2"
    d["subject_id"] = f"{DS}:expt2:" + d["turkcode"]
    d["attn_pass"] = d["attn_response"] == "20"
    d["excluded_by_authors"] = ~(d["attn_pass"] & d["complete"])
    d["incentive_check_correct"] = d["incentive_check_raw"] == 2
    d["response_scale"] = ("odds slider 0-10 (>40x,20x,10x,5x,2x,1,2x,...,>40x); converted to P via "
                           "odds, endpoints set to 40:1")
    main = d[d["question"].notna()].copy()
    prac = d[d["question"].isna()].copy()
    main["question"] = main["question"].astype(int)
    main["y_colour"] = main["colourPart"]
    main["x_colour"] = np.where(main["colourPart"] == "red", "blue", "red")
    main = add_design(main)
    main = add_prior_reported(main, ["subject_id"])
    d = pd.concat([main, prac], ignore_index=True)
    d["condition"] = np.where(d["condition"] == "practice",
                              "practice|" + d["dependency"].str.lower() + "|" + d["informants"].astype(str) + "_informants",
                              d["condition"])
    d["extra"] = to_extra(d, EXTRA_DESIGN + [
        "turkcode", "gender", "age", "language", "country", "colourPart", "incentive", "scale",
        "response_raw", "response_r", "time_elapsed_ms", "attn_response", "attn_pass", "complete",
        "incentive_check_raw", "incentive_check_correct", "excluded_by_authors", "practice_jar",
        "practice_trial"])
    return d


# --------------------------------------------------------------------------- #
# Experiment 3: within-subject dependency (both orders), 0-10 scale, preference q
# --------------------------------------------------------------------------- #
PAIR = {"red": "blue", "blue": "red", "purple": "green", "green": "purple"}


def expt3() -> pd.DataFrame:
    f = BASE / "expt3" / "csv2" / "expt3_data.csv"
    w = pd.read_csv(f)
    bx = pd.read_csv(BASE / "expt3" / "csv2" / "expt3_data_why_BX.csv")[["turkcode", "rule", "category", "cascade"]]
    bx = bx.rename(columns={"rule": "why_rule_BX", "category": "why_category_BX", "cascade": "why_cascade_BX"})
    w = w.merge(bx, on="turkcode", how="left")
    w["source_row"] = w.index
    subj_cols = ["turkcode", "gender", "age", "language", "country", "informants", "colourPart1",
                 "colourPart2", "dependency1", "dependency2", "slider", "batch", "attn", "which",
                 "why_rule_BX", "why_category_BX", "why_cascade_BX", "source_row"]
    rows = []
    for _, r in w.iterrows():
        base = {c: r[c] for c in subj_cols}
        n = int(r["informants"])
        # practice: PQ1 jar = colourPart1 (colourOrder[0]); PQ2 jar = its pair partner.
        for k, jar in ((1, r["colourPart1"]), (2, PAIR[r["colourPart1"]])):
            balls = str(r[f"P{k}"]).split(",")
            rows.append(dict(base, **practice_row(balls, jar), trial=k - 1, response_r=r[f"PQ{k}"],
                             posterior_reported=r[f"PQ{k}"] / 10, practice_jar=jar, practice_trial=k,
                             dependency=np.nan))
        for t, dep in enumerate((r["dependency1"], r["dependency2"])):
            pre = "i" if dep == "Independent" else "sq"
            for q in range(1, n + 2):
                v = r[f"{pre}-{q}"]
                if pd.isna(v):
                    continue
                rows.append(dict(base, dependency=dep, question=q, task_order=t + 1,
                                 trial=2 + t * (n + 1) + (q - 1), response_r=v,
                                 posterior_reported=v / 10))
    d = pd.DataFrame(rows)
    d["study"] = "expt3"
    d["source_file"] = rel(f)
    d["subject_id"] = f"{DS}:expt3:" + d["turkcode"].astype(str)
    d["response_scale"] = "slider 0-10 integer (0 = definitely x jar, 10 = definitely y jar)"
    d["excluded_by_authors"] = False
    main = d[d["question"].notna()].copy()
    prac = d[d["question"].isna()].copy()
    main["question"] = main["question"].astype(int)
    main["x_colour"], main["y_colour"] = "x", "y"  # colours per task not recoverable from cleaned csv
    main = add_design(main)
    main["condition"] = main["condition"] + "|within|order" + main["task_order"].astype(int).astype(str)
    main = add_prior_reported(main, ["subject_id", "task_order"])
    d = pd.concat([main, prac], ignore_index=True)
    d["informants"] = d["informants"].astype(int)
    d["condition"] = np.where(d["condition"] == "practice",
                              "practice|" + d["informants"].astype(str) + "_informants", d["condition"])
    d["item_desc"] = np.where(d["stage"] == "practice", d["item_desc"],
                              d["item_desc"].str.replace("x jar (", "x jar [colour pair varies] (", regex=False))
    d["extra"] = to_extra(d, EXTRA_DESIGN + [
        "turkcode", "gender", "age", "language", "country", "colourPart1", "colourPart2",
        "dependency1", "dependency2", "task_order", "slider", "batch", "attn", "which",
        "why_rule_BX", "why_category_BX", "why_cascade_BX", "response_r", "excluded_by_authors",
        "practice_jar", "practice_trial"])
    return d


def main():
    df = pd.concat([expt1(), expt2(), expt3()], ignore_index=True)
    df["dataset"] = DS
    df = df.sort_values(["study", "subject_id", "trial"]).reset_index(drop=True)
    write_dataset(df, DS)


if __name__ == "__main__":
    main()
