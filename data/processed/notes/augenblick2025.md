# augenblick2025 — Augenblick, Lazarus & Thaler (2025), "Overinference from Weak Signals and Underinference from Strong Signals", QJE

Converter: `scripts/convert_data/augenblick2025.py`. Raw: `data/raw/augenblick2025/Experiment/` (do-files 01–05
read as documentation only; PAPs; screenshots). `Empirical.zip` (betting/options data) is out of scope.

25,988 rows, 1,552 subjects, 558 items. One row per elicited posterior.

| study | source | rows | subjects | items |
|---|---|---|---|---|
| 1a | `experiment1a_data.csv` (raw Qualtrics) | 10,488 | 552 (500 pass attention) | 258 |
| 1b | `experiment1b_data.dta` (authors' intermediate) | 7,500 | 500 | 60 |
| 2  | `expt2_cleaned.dta` (authors' cleaned) | 8,000 | 500 | 240 |

## Study 1a (2021): card from Green vs Purple deck, prior 1/2
- Each subject answers 19 belief questions drawn from 258: 12 single-card, 5 three-card, 2 uncertain-composition
  (plus an attention check and 5 WTP-for-cards questions, not included: they aren't beliefs).
- Question ids follow `01study1aCleaning.do`: q1–97 Diamond drawn, q98–194 Spade drawn (within each block of 97:
  positions 1–65 symmetric decks, 66–97 "asymmetric" decks; large decks of 1,665 cards for q1–33, 66–81, 98–130,
  163–178, small decks of 337 otherwise); q195–250 three cards drawn with replacement, `n_diamonds = 3 - mod(q+1, 4)`;
  q251–258 uncertain deck composition.
- Built from the raw CSV, reproducing the do-file's reshape and round mapping (checked: all 9,500 rows in
  `expt1a_cleaned.dta` match on answer and round). This keeps the 52 attention-check failures and the per-page timers.
- `posterior_bayes` comes from the authors' `bayesian` pseudo-respondent row (q1–250); q251–258 are hard-coded as 2185/3330.
- Signal likelihoods: the deck compositions are not stored in the data, so they are worked back from the Bayes
  posterior. For symmetric mirrored decks (Purple is Green with the suits swapped), P(Green | card) = P(card | Green),
  so Green's Diamond count is `round(bayes × N)` (e.g. 853/812, matching the screenshots and Study 1b). For three
  cards, `logit(g) = logit(bayes) / (2·n_diamonds − 3)`. Rebuilding the Bayes posterior from `p_signal_h/p_signal_not_h`
  is accurate to 3e-4. For uncertain composition, the decks are 50/50 mixtures with mirrored Purple, so P(card | Green) = Bayes
  posterior (2185/3330 or 1145/3330). The component compositions aren't in the data; the screenshot example doesn't match any loop.
- Asymmetric questions (q66–97, q163–194; not analysed in the paper): compositions can't be recovered, so
  `p_signal_*` = NaN and `log_lr = logit(bayes)`.
- q33 and q130 are uninformative (Bayes = 50%): `p_signal_h = p_signal_not_h = 0.5`, `signal_dir = 0`.
- `trial` = rank of the page number (`round_number`: 1–12 single, 14–18 three-card, 24–25 uncertain) among the
  subject's belief rows. `rt_sec` = Qualtrics page-submit timer. `condition` = question type.
- `extra`: question, round_number, deck_size, green_diamonds, answer_purple, attention_pass, **paper_excluded**
  (failed the attention check), **main_analysis_item** (single symmetric informative card, the paper's `signal_cat`
  sample), uncertainty_same, age, gender, party, education, race, newscons, newshelp, crt_1-3, duration_sec.

## Study 1b (2024): 1 Green + 1–3 Blue decks, prior 1/num_decks
- 15 rounds per subject, 5 per deck count (2/3/4 decks; the order of deck-count blocks is in `deck_order`). Decks have 1,665 cards; the Green deck has
  853/812, 916/749, 1036/629, 1294/371 or 1419/246 of the two suits (`03study1bAnalysis.do`), and each Blue deck is the mirror image.
  Questions 1,3,..,9 and 22,24,..,30 favour Green; the others disfavour it. The drawn suit isn't identified in the
  data, so `signal_desc` describes the card by direction only.
- `posterior_reported` = P(Green). `extra.answer_blue1-3` = stated probabilities for the Blue decks, `confidence` =
  per-trial certainty slider (cognitive uncertainty), `rt_confidence`, `signal_cat` (1–5), demographics, confusedlikert.
- `trial` = round_number − 1. `rt_sec` = page-submit timer.

## Study 2 (2024): NBA win probabilities
- 4 quarters (blocks, random order). Each quarter opens with team H having just made a shot to lead by 1 or 5 with
  2:40 left, with the calculator's P(H wins) shown. That's followed by 4 events (opponent shot at events 2 and 4, H's shot at events 3 and 5; made or missed,
  randomised), each followed by an elicitation of P(H wins).
- No likelihoods: `prior_objective` = the calculator probability before the event, `posterior_bayes` = the calculator probability after the event (`truth` in
  `04study2Cleaning.do`, from inpredictable.com), `log_lr` = the difference in logits. `p_signal_*` = NaN.
  `prior_reported` = the subject's previous guess (events 3–5; at event 2 the prior is the shown calculator value).
  `posterior_bayes_rep_prior` = previous guess + the calculator's logit change.
- `round` = event − 2 (0–3) within quarter; `trial` = 0–15 across the session; `condition` = quarter.
  `item_id` = quarter × starting lead × event × made/missed sequence. `task_domain` = other. No RT.
- `extra`: quarter, event, loop, bin_loop, initial_score, score_diff, made_shot, shooting_team, time_left,
  calculator probabilities, block/round numbers, NBA knowledge items, demographics.

## Exclusions and caveats
- All 1a subjects kept; the paper drops `paper_excluded == 1` (52 subjects). The 1b and 2 intermediate files are
  already restricted by the authors to 500 subjects. The raw CSVs have 511 and 525 respondents, so 11 and 25
  excluded respondents (attention or screening failures) aren't included, because the authors' reshaping of those files isn't in
  the package. `paper_excluded = 0` there.
- The paper winsorises at 5/95% (by signal category) and maps answers of 0/100 to 1/99 (or 0.5/99.5 in Study 2) before
  taking logits. None of this is applied here: raw answers are kept and include 0 and 1.
- `benjamin2019_data.dta` isn't converted. It holds 70 aggregate points (signal_cat, abs_bayes, relative_deviation) from
  Benjamin's (2019) literature review, not trial-level data.

## Sanity (median of signed logit update ÷ Bayesian log LR)
- 1a main items, attention passers: |log LR| ≈ 0.02–0.3 → 1.0–1.35 (means 1.1–3.2); |log LR| ≥ 1.5 → 0.92–0.99
  (means 0.74–0.80).
- 1b: |log LR| = 0.05, 0.20, 0.50, 1.25, 1.75 → medians 1.62, 1.25, 1.12, 0.92, 0.86.
- 2: mean winsorised directional logit update vs. calculator by quarter: Q1 0.139 vs 0.087; Q2 0.159 vs 0.106; Q3
  0.168 vs 0.162; Q4 0.336 vs 0.522. That's overreaction early and underreaction late, as in the paper.
