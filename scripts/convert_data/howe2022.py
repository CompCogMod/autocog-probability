"""Convert Howe, Perfors et al. (2022, JDM) "Base rate neglect and conservatism
in probabilistic reasoning: insights from eliciting full distributions".

Task: a bag of chips has an unknown proportion of red chips theta in
{0, .1, ..., 1}. Subjects saw chips drawn from the bag (red/blue) and gave a
full probability distribution over theta (11 bins, weights sum to 1). Some
subjects also gave a distribution over theta before seeing any chips (prior).

Mapping to the binary schema: H = "the next chip drawn is red". P(H) under a
distribution over theta is its mean, so
  posterior_reported = mean of the elicited posterior distribution,
  prior_reported     = mean of the elicited prior distribution,
  prior_objective    = mean of the design prior (given histogram, or uniform).
Evidence D = the ordered chip sequence (r red, b blue). With design prior pi,
  P(D | H)     = E_pi[theta^(r+1) (1-theta)^b] / E_pi[theta]
  P(D | not H) = E_pi[theta^r (1-theta)^(b+1)] / E_pi[1-theta]
so expit(logit(prior_objective) + log_lr) is exactly the Bayesian posterior
mean under pi. posterior_bayes_rep_prior is the exact full-distribution Bayes
posterior mean under the subject's reported prior (not the binary formula).
Full distributions are kept in `extra`.

Raw files: data/raw/howe2022/repo/Experiment{1,2,3}_Andy.csv (rawest files;
the datafiles/*_adj.csv versions add 0.01 smoothing and drop excluded people).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common import RAW, logit, expit, to_extra, write_dataset

NAME = "howe2022"
RAWDIR = RAW / NAME / "repo"
BINS = np.round(np.arange(0, 1.01, 0.1), 1)
ECOLS = [f"e{int(round(x * 100))}" for x in BINS]

UNIFORM = np.full(11, 1 / 11)
# Exp2 PEAKED and Exp3 GIVEN priors, as hard-coded in data_creation_exp{2,3}.Rmd
PEAKED = np.array([0, 0, 0, 0, 0, 0.08, 0.08, 0.2, 0.6, 0.04, 0])
EXP3_GIVEN = np.array([0.043561, 0.049242, 0.064394, 0.104167, 0.132576, 0.212121,
                       0.132576, 0.104167, 0.064394, 0.049242, 0.043561])
EXP3_GIVEN = EXP3_GIVEN / EXP3_GIVEN.sum()


def dist_mean(w):
    w = np.asarray(w, float)
    return float((w * BINS).sum() / w.sum())


def bayes_dist(prior, r, b):
    lik = BINS ** r * (1 - BINS) ** b
    post = prior * lik
    return post / post.sum()


def predictive_lr(prior, r, b):
    """P(D|next red), P(D|next blue) for ordered sequence D under prior over theta."""
    prior = np.asarray(prior, float) / np.sum(prior)
    m = (prior * BINS).sum()
    ph = (prior * BINS ** (r + 1) * (1 - BINS) ** b).sum() / m
    pn = (prior * BINS ** r * (1 - BINS) ** (b + 1)).sum() / (1 - m)
    return ph, pn


def decode_chips(s):
    """chipsOrder: '2' = red, '1' = blue. Long sequences were stored as numbers in
    the raw CSV and are truncated to ~15 digits (rest zero-padded), so the
    decoded sequence can be shorter than seenRed+seenBlue (flagged in extra)."""
    if not isinstance(s, str) or s == "" or s.lower() == "nan":
        return ""
    return "".join({"2": "R", "1": "B"}.get(c, "") for c in s)


def load(exp):
    f = RAWDIR / f"Experiment{exp}_Andy.csv"
    d = pd.read_csv(f, dtype={"chipsOrder": str})
    d["source_row"] = np.arange(len(d))
    d["source_file"] = str(f.relative_to(RAW))
    if "section" not in d:
        d["section"] = "posterior"
        d["seenRed"], d["seenBlue"] = 4, 1
        d["chipsOrder"] = np.nan
        d["time"] = np.nan
    d["excluded_paper"] = d["pID"].isin(
        d.loc[(d.botCheck == 0) | (d.actuallyEstimated == 0), "pID"])
    return d


ORDER = {  # elicitation order within subject
    "exp1": {"prior": 0, "five": 1, "unlimited": 2},
    "exp3": {"prior": 0, "posterior": 1},
}


PRIOR_LABEL = {"given_peaked": "peaked", "given_uniform": "uniform",
               "given_histogram": "humped", "learned_by_experience_assumed_same_as_given": "humped",
               "none_given_assumed_uniform": "noprior"}


def design_prior(study, cond):
    if study == "exp2":
        return (PEAKED, "given_peaked") if cond == "peaked" else (UNIFORM, "given_uniform")
    if study == "exp3":
        src = "given_histogram" if cond == "given" else "learned_by_experience_assumed_same_as_given"
        return EXP3_GIVEN, src
    return UNIFORM, "none_given_assumed_uniform"


def convert():
    out = []
    for exp in (1, 2, 3):
        study = f"exp{exp}"
        d = load(exp)
        priors = {r.pID: r for r in d[d.section == "prior"].itertuples()}
        for r in d[d.section != "prior"].itertuples():
            w = np.array([getattr(r, c) for c in ECOLS], float)
            nr, nb = int(r.seenRed), int(r.seenBlue)
            pri, pri_src = design_prior(study, r.condition)
            ph, pn = predictive_lr(pri, nr, nb)
            rep = priors.get(r.pID)
            rep_w = np.array([getattr(rep, c) for c in ECOLS], float) if rep is not None else None
            order = ORDER.get(study, {}).get(r.section, 0) if rep is not None else (
                0 if study != "exp1" or r.condition != "all" else ORDER["exp1"][r.section])
            chips = decode_chips(r.chipsOrder)
            bayes_post_dist = bayes_dist(pri, nr, nb)
            row = dict(
                dataset=NAME, study=study, source_file=r.source_file, source_row=r.source_row,
                subject_id=f"{NAME}:{study}:{r.pID}", trial=order,
                round=1 if (study == "exp1" and r.condition == "all" and r.section == "unlimited") else 0,
                condition=r.condition if study != "exp1" else f"{r.condition}/{r.section}",
                item_id=f"{study}:{PRIOR_LABEL[pri_src]}:r{nr}b{nb}",
                item_desc=(f"Bag of chips with unknown proportion of red (0-100% in 10% steps); "
                           f"prior over proportion: {pri_src}; saw {nr} red, {nb} blue chips"
                           f"{' (' + chips + ')' if chips else ''}; report distribution over proportion red"),
                task_domain="abstract",
                hypothesis="next chip drawn is red (= mean proportion red)",
                prior_objective=dist_mean(pri),
                prior_reported=dist_mean(rep_w) if rep_w is not None else np.nan,
                signal_desc=f"{nr} red, {nb} blue" + (f" [{chips}]" if chips else ""),
                n_signals=nr + nb,
                p_signal_h=ph, p_signal_not_h=pn,
                posterior_reported=dist_mean(w),
                response_scale="distribution",
                rt_sec=np.nan,
                # extra
                dist_bins=BINS.tolist(), dist_weights=w.tolist(),
                prior_dist_weights=rep_w.tolist() if rep_w is not None else None,
                objective_prior_weights=[round(x, 6) for x in pri.tolist()],
                objective_prior_source=pri_src,
                bayes_post_dist_weights=[round(x, 6) for x in bayes_post_dist.tolist()],
                posterior_mode=float(BINS[np.argmax(w)]),
                posterior_sd=float(np.sqrt((w * (BINS - dist_mean(w)) ** 2).sum() / w.sum())),
                section=r.section, chips_seq=chips, chips_seq_complete=len(chips) == nr + nb, chipsOrder_raw=r.chipsOrder,
                pID=int(r.pID), age=r.age, gender=r.gender, botCheck=r.botCheck,
                actuallyEstimated=r.actuallyEstimated, excluded_paper=bool(r.excluded_paper),
                time_total_sec=r.time,
            )
            if rep_w is not None:
                row["posterior_bayes_rep_prior"] = dist_mean(bayes_dist(rep_w / rep_w.sum(), nr, nb))
            out.append(row)
    df = pd.DataFrame(out)
    df["log_lr"] = np.log(df.p_signal_h) - np.log(df.p_signal_not_h)
    df["signal_dir"] = np.sign(df["log_lr"])
    df["posterior_bayes"] = expit(logit(df.prior_objective) + df.log_lr)
    xcols = ["dist_bins", "dist_weights", "prior_dist_weights", "objective_prior_weights",
             "objective_prior_source", "bayes_post_dist_weights", "posterior_mode", "posterior_sd",
             "section", "chips_seq", "chips_seq_complete", "chipsOrder_raw", "pID", "age", "gender", "botCheck",
             "actuallyEstimated", "excluded_paper", "time_total_sec"]
    df["extra"] = to_extra(df, xcols)
    return df.drop(columns=xcols)


if __name__ == "__main__":
    write_dataset(convert(), NAME)
