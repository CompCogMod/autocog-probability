"""Convert Buser, Gerhards & van der Weele (2018, JRU) belief-updating data.

Raw: data/raw/buser2018/Data.dta (one row per subject, wide). Each subject does
three ego-relevant tasks (Anagrams 'A', Matrices 'M', Raven 'R') in a
randomised order (OrderA/OrderM/OrderR = position 1..3). After each task they
report P(top half of their group of 8) (Beliefs0<pos>), then receive 6 binary
signals (Signals<k><pos>, 1 = "top half") of accuracy 0.7 and report a belief
after each (Beliefs<k><pos>, k=1..6). Round/position 4 (competition choice)
has no belief data. Bayesian benchmark in Codes.do: 0.7/0.3 likelihoods.

Output: one row per elicited posterior (297 x 3 x 6 = 5346 rows).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common import RAW, bayes_posterior, to_extra, write_dataset

NAME = "buser2018"
SRC = RAW / NAME / "Data.dta"
ACC = 0.7
N_SIG = 6
TASKS = {
    "A": ("anagrams", "Anagrams task (word puzzles); rank vs. 7 others in group of 8 by score"),
    "M": ("matrices", "Matrices task (numerical/arithmetic matrix puzzles); rank vs. 7 others in group of 8 by score"),
    "R": ("raven", "Raven progressive matrices (IQ test); rank vs. 7 others in group of 8 by score"),
}
SUBJ_COVS = ["Gender", "Age", "major", "CompensationScheme", "DifficultyNr", "Risk",
             "CompetitionScale", "NarcicismScale4", "SelfIQ", "SelfGrades",
             "PastExperiments", "PaymentRound", "Score4", "Rank4", "Top504", "Winner"]
RELEVANCE = {"A": "RelevanceAnagrams", "M": "RelevanceMatrices", "R": "RelevanceRaven"}


def main() -> None:
    raw = pd.read_stata(SRC, convert_categoricals=False)
    rows = []
    for idx, s in raw.iterrows():
        sid = f"{NAME}:main:{int(s.session_id)}_{int(s.Subject)}"
        for t, (tname, tdesc) in TASKS.items():
            pos = int(s[f"Order{t}"])  # 1..3 task position
            beliefs = [round(float(s[f"Beliefs{k}{pos}"]), 4) for k in range(N_SIG + 1)]
            signals = [int(s[f"Signals{k}{pos}"]) for k in range(1, N_SIG + 1)]
            # task-level flags used in the paper (Codes.do): any boundary prior
            # (0 or 1 -> Bayes update = 0) and any wrongly-signed update
            missing_task = any(b in (0.0, 1.0) for b in beliefs[:N_SIG])
            wrong = [(sg == 1 and beliefs[k + 1] < beliefs[k]) or
                     (sg == 0 and beliefs[k + 1] > beliefs[k])
                     for k, sg in enumerate(signals)]
            chained = beliefs[0]
            for k, sg in enumerate(signals):
                llr = np.log(ACC / (1 - ACC)) * (1 if sg == 1 else -1)
                chained = float(bayes_posterior(chained, llr))
                rows.append(dict(
                    dataset=NAME, study="main", source_file=f"{NAME}/Data.dta",
                    source_row=f"{idx}:{t}:{k + 1}", subject_id=sid,
                    trial=(pos - 1) * N_SIG + k, round=k,
                    condition="",
                    item_id=tname, item_desc=tdesc, task_domain="ego",
                    hypothesis="own score in top half (rank 5-8) of group of 8 on this task",
                    prior_reported=beliefs[k],
                    signal_desc="signal: top half" if sg == 1 else "signal: bottom half",
                    signal_dir=1 if sg == 1 else -1, n_signals=1,
                    p_signal_h=ACC if sg == 1 else 1 - ACC,
                    p_signal_not_h=1 - ACC if sg == 1 else ACC,
                    posterior_reported=beliefs[k + 1],
                    posterior_bayes=chained,
                    response_scale="probability 0-1 (2 decimals)",
                    # extra
                    x_initial_prior=beliefs[0], x_task_position=pos,
                    x_top_half=int(s[f"Top50{pos}"]), x_rank=int(s[f"Rank{pos}"]),
                    x_score=float(s[f"Score{pos}"]), x_signal=sg,
                    x_task_relevance=int(s[RELEVANCE[t]]),
                    x_wrong_direction=bool(wrong[k]),
                    x_boundary_prior=beliefs[k] in (0.0, 1.0),
                    x_paper_task_missing=missing_task,
                    x_paper_task_all_correct=not any(wrong),
                    x_n_pos_signals_task=sum(signals),
                    x_belief_payment_pos=int(s[f"BeliefPayment{pos}"]),
                    **{f"x_{c}": s[c] for c in SUBJ_COVS},
                ))
    df = pd.DataFrame(rows)
    xcols = [c for c in df if c.startswith("x_")]
    df["extra"] = to_extra(df.rename(columns={c: c[2:] for c in xcols}), [c[2:] for c in xcols])
    df = df.drop(columns=xcols)
    write_dataset(df, NAME)


if __name__ == "__main__":
    main()
