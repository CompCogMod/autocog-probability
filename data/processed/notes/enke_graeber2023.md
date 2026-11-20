# enke_graeber2023 — Enke & Graeber (2023), "Cognitive Uncertainty", QJE

Converter: `scripts/convert_data/enke_graeber2023.py`. Output: 13,130 rows, 1,608 subjects, 952 item_ids. All rows are single-step balls-and-bags updating (`round`=0, `task_domain`=abstract, `hypothesis`="bag A").

## Design
Two bags. Bag A holds a share `diagnosticity` (p) of the "A colour" balls (black in A/B main files, red in the default file); bag B holds 1-p. The subject is told the prior P(bag A), sees 1–5 balls drawn with replacement, and states P(bag A) in percent (0–100). After each answer comes a per-trial **cognitive uncertainty (CU)** rating.

| study | file | subjects × tasks | manipulations |
|---|---|---|---|
| A_beliefs | A Experiments/data/Beliefs/Beliefs_cleaned.dta | 500 × 12 = 6000 | priors {1,5,10,30,50,70,90,95,99}%, p ∈ {.65,.75,.90}, n ∈ {1,3,5}. Between-subject "complex numbers" arm: in rounds 7–12, prior and diagnosticity are shown as arithmetic expressions (1410 rows, `condition`=complex_numbers). The paper's main analyses use `treatment==0` (flagged as `paper_main_sample` in `extra`). |
| A_beliefs_default | A Experiments/data/Beliefs/Beliefs_Default_Manipulation_cleaned.dta | 108 × 12 = 1296 | Same task. Within subject, blocked: framed with 2 bags vs. 10 bags (`low_default`; the ignorance default is 1/10 instead of 1/2). `bags_first` gives the block order. |
| B_beliefs | B Experiments/data/Beliefs/Beliefs_cleaned.dta | 1000 × ~6 = 5834 | priors {10,30,50,70,90}%, p ∈ {.70,.90,1.00}, n ∈ {1,3}. Conditions: two_states (baseline), ten_states_low_default (between-subject default arm), compound_diagnosticity (ROCL: p described as a compound lottery, prior 50%). |

## Column derivations
- `prior_objective` = prior/100. Float noise such as 30.000002 is rounded.
- `p_signal_h` = p^nA (1-p)^nB and `p_signal_not_h` = (1-p)^nA p^nB, where nA and nB are the counts of A-colour and other-colour balls. The likelihood is that of the counts; the binomial coefficient cancels.
- `signal_dir` = sign(nA − nB). `n_signals` = sample size.
- `posterior_reported` = belief/100. `posterior_bayes` is computed by `finalize`. It matches the raw `bayesian_posterior` (kept in `extra.bayesian_posterior_orig`, in percent) to within 1e-6 for the A files and 5e-4 for B, where the raw value is rounded to 0.1pp.
- B items with p = 1.00 (166 rows) have `log_lr` = ±inf and a Bayes posterior of 0 or 1. All draws are unanimous.
- `trial` = round−1 for the A files. It is NaN for B, which has no task-order variable.
- `rt_sec` = `time_spent_belief` (seconds) for A_beliefs only. The A_beliefs `extra` also has `rt_cu_sec`. B has only a per-subject `total_time_beliefs_subject` (looks like minutes).
- `item_id` = `<variant>|pr<prior%>_d<p%>_<nA>a<nB>b`. The variant is simple/complex (A), bags2/bags10 (default file) or simple/compound (B).

## Cognitive uncertainty (`extra.cognitive_uncertainty`, 0–1, higher = more uncertain)
- **A files:** 0–1 in steps of .05. Per the paper, CU = 1 − the stated certainty (0–100%) that the optimal answer lies within ±1pp of one's own answer.
- **B file:** CU = `uncertainty_slider`/30. The slider (0–30) is a ± interval in pp around one's own answer. `extra` also keeps the raw slider, `cognitive_sd` and `censored_cu`.
- Because the elicitations differ, CU should not be pooled across waves without a wave effect.

## Exclusions (all rows kept)
- B: `extra.outlier` (= `paper_excluded`; the B tables use `outlier==0`, 301 rows), `speeder` (583 rows) and `censored_choice`.
- A structural fits drop rows with belief/Bayes ratio >10 or <0.1. This is not flagged and can be recomputed.
- All subjects passed the attention checks.
- `across_trial_inconsistency.dta` (A) is a derived table of repeated-problem answer pairs. It is not converted; `extra.subj_mean_abs_inconsistency` carries the subject-level mean.

## Sanity checks
Regressing belief on Bayes, Bayes×CU and CU (non-complex, non-compound rows) gives:

| study | slope on Bayes | Bayes×CU |
|---|---|---|
| A_beliefs | 0.71 | −0.42 |
| A_beliefs_default | 0.74 | −0.33 |
| B_beliefs | 0.74 | −0.48 |

The belief-on-Bayes slope by CU quartile falls from 0.70 to 0.42 (A) and from 0.73 to 0.41 (B): higher CU goes with flatter, more compressed responses, as in the paper. The ten-states/10-bags framing lowers mean beliefs (B: 45.6 vs 51.8).

## Caveats
- `bag_A_drawn` (in `extra`) is the realized true state. In A it appears unrelated to the drawn counts, so it is kept for reference only.
- In the default file, some subjects appear to invert the scale (for example belief 99 when Bayes ≈ 0). The rows are kept as is.

## Not converted
- **Risk** (`*/data/Risk/*.dta`): lottery valuation via price lists. Each trial has a payout probability (`lottery_probability`, A: 1–99%, B: 0–100% with gains/losses), a payout amount, a normalized certainty equivalent (`ce`) and a per-trial CU. A also has a complex-numbers arm and a default-manipulation file; B has ROCL and high/low default arms. This is probability weighting, not updating. It is a candidate future addition as a separate "valuation" dataset.
- **Survey** (A `Survey_Beliefs/Survey_cleaned_final.dta`, 1000 × 1; B `Survey_exp/Survey_cleaned.dta`, 2002 subjects × 3): stated probabilities about stock returns, income and inflation thresholds, benchmarked against historical or objective probabilities (`true_percentile`), with CU. There is no prior and no signal, so these are not updating tasks and are excluded. They could serve as a CU-compression check on one-shot probability judgments.
