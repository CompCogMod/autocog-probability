# tappin2020 — Tappin, Pennycook & Rand (2020), Cognition, "Bayesian or Biased?"

Converter: `scripts/convert_data/tappin2020.py`. Output: 31,776 rows (S1 7,872 / S2 15,872 / control 8,032),
1,986 subjects (492 / 992 / 502), 16 items; 16 rows per subject.

## Design
16 fact-checked political statements (8 true / 8 false x pro-Dem / pro-Rep; texts in SI). Phase 1, per
statement: rate P(statement true) 0-100% (prior), then see a TRUE/FALSE signal. Phase 2 (after all 16):
rate each statement again with no signal (posterior). So the posterior is a delayed judgment.
- **S1**: told signals are accurate "two out of three times". S1 also had a memory test for the signal seen.
- **S2**: told only that clues are accurate somewhere between 51% and 99%; on each trial subjects rated the
  clue's accuracy (1-5, `signalRate`) and the overall clue accuracy (`LHR`, 51-99). The paper's Bayesian
  benchmark uses the subject's `LHR`.
- **control**: like S1 but signals hidden (still randomly assigned) — a regression-to-the-mean control.
Signals were actually accurate ~2/3 of the time in all studies (66-67% match with ground truth).

## Columns
- `hypothesis` = "statement true"; `task_domain` = political_fact; `round` = 0; `n_signals` = 1.
- `prior_reported` = prior/100, `posterior_reported` = post/100 (raw, NOT the paper's 0/100 -> 0.5/99.5 recode;
  recoded values are in `extra`). `prior_objective` NaN (no design prior; ground truth in `extra.statement_true`).
- `signal_dir` = +1 for TRUE, -1 for FALSE; control = 0 (nothing shown) with `log_lr` = 0.
- `p_signal_h/p_signal_not_h`: S1 = 2/3 vs 1/3 (stated likelihood). S2 = the subject's own per-trial `LHR`/100
  (subjective likelihood, as in the paper's benchmark; not a design value). Control: NaN.
- `posterior_bayes_rep_prior` is computed from the raw prior; equals the paper's `Bayes_posterior` (in `extra`,
  rescaled to 0-1) except where prior was 0/100. For control it equals the prior; the paper's benchmark (computed
  as if the hidden signal were seen) is `extra.Bayes_posterior_paper`.
- `item_id` = stmt01-16 (`statementID`); `item_desc` = statement text (mojibake quotes fixed) + truth + partisanship.
- `trial` NaN: presentation order not recorded (rows are sorted by statementID). `rt_sec` = posterior RT.
- `condition`: signal_shown_acc2of3 / clue_shown_acc_unstated / signal_hidden.
- `extra`: CRT_sum, demographics, party (1-7), party_choose, issues, vote, media trust, God, random, CRT_familiarity,
  conserv_rep / party_z, statement_type, signal_favors_REP, politically_concordant, prior_concordance,
  prior/post RTs, signalRate/LHR (+RTs), mem/memory_error (S1), paper's difference/ratio/log_ratio, capped.

## Exclusions
Raw files are already post-exclusion (control N=502 matches SI). All rows kept. Flags in `extra`:
`excl_capped` (paper drops trials where updating toward the signal was impossible: prior 0/100 at the wrong end),
`flag_self_report_random`, `flag_crt_familiar`, `memory_error` (S1).

## Sanity
Among non-zero updates, fraction in signal direction: S1 0.656, S2 0.596, control 0.501 (vs hidden assigned signal,
i.e. chance). Zero updates: 10% / 11% / 17%.

## Caveats
Posterior is elicited after all 16 priors+signals (memory demands; S1 memory errors ~28% of trials). Statements are
politically charged so priors/updates interact with partisanship (`politically_concordant`).
