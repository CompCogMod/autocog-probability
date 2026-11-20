# buser2018 — Buser, Gerhards & van der Weele (2018), "Responsiveness to Feedback as a Personal Trait", JRU

Converter: `scripts/convert_data/buser2018.py`. Raw: `data/raw/buser2018/Data.dta` (one wide row per subject), `Codes.do` (analysis code, read only).

## Design
- 297 subjects, 22 sessions, ranked within groups of 8. Three ego-relevant tasks in randomised order:
  Anagrams (A), Matrices (M), Raven matrices (R). `Order{A,M,R}` gives each task's position (1-3); raw
  variables suffixed 1/2/3 are by position, not by task.
- After each task: prior P(top half of group of 8) (`Beliefs0<pos>`), then 6 binary signals
  (`Signals<k><pos>`, 1 = "top half"), belief re-elicited after each (`Beliefs<k><pos>`), probabilities 0-1.
- Signal accuracy 0.7 (Codes.do Bayes benchmark uses 0.7/0.3; data check: P(signal=1 | top) = 0.698,
  P(signal=1 | bottom) = 0.301).
- Position 4 (competition / piece-rate choice) has no belief data and is not included.

## Rows and columns
- One row per posterior: 297 x 3 tasks x 6 signals = 5346 rows. `item_id` = anagrams / matrices / raven.
- `round` = signal index within task (0-5); `trial` = (task position - 1) * 6 + round.
- `prior_reported` = belief just before the signal; `posterior_reported` = belief just after.
- `prior_objective` = NaN (no objective base rate; true top-half rate is 0.49).
- `posterior_bayes` = **chained** Bayesian posterior starting from the subject's own initial prior
  (`Beliefs0`) for that task and applying all signals so far (LR 7/3 per signal). `posterior_bayes_rep_prior`
  = one-step Bayes from the prior reported just before (the paper's benchmark).
- `condition` = "" (no between-subject manipulation in belief stage).
- `extra`: initial_prior, task_position, top_half (truth of H), rank (1-8, 8 = best), score, signal,
  task_relevance (1-7 self-rated), wrong_direction, boundary_prior, paper_task_missing,
  paper_task_all_correct, n_pos_signals_task, belief_payment_pos, and subject covariates (Gender 1/2 with 2 = female
  per Codes.do, Age, major, CompensationScheme (2 = chose competition in round 4), DifficultyNr (meaning not
  documented in code), Risk, CompetitionScale, NarcicismScale4, SelfIQ, SelfGrades, PastExperiments,
  PaymentRound, Score4/Rank4/Top504/Winner).

## Exclusions (none applied)
Paper flags, reproduced in `extra` at task level: `paper_task_missing` (some prior in the task is 0 or 1,
so Bayes update is 0) and `paper_task_all_correct` (no wrongly-signed update in the task). Only 41% of
subject-tasks pass both. Codes.do also recodes wrongly-signed updates to 0 for ratio measures; not done here.
Codes.do references `Missingfour`/`Correctfour` variables whose definition is not in the file.

## Sanity
Non-boundary priors: mean update after good signal +0.055 vs one-step Bayes +0.152; after bad signal
-0.045 vs -0.163 (strong conservatism, mild asymmetry). 13% wrongly-signed updates, 25% zero updates.
Mean initial prior 0.54 vs 0.49 actually in top half.

## Caveats
- Task descriptions (esp. "Matrices") are inferred from variable names; check paper.
- Beliefs stored as float32 in Stata; rounded to 4 decimals.
- No response times.
