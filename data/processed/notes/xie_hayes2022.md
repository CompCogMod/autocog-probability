# xie_hayes2022 — Xie & Hayes (2022), "Sensitivity to Evidential Dependencies in Judgments Under Uncertainty", Cognitive Science

Converter: `scripts/convert_data/xie_hayes2022.py` → `data/processed/xie_hayes2022.csv.gz` (6,174 rows, 743 subjects, 44 item ids).

## Design (all experiments)
Two jars, each with 100/120 balls of its own colour, so P(ball of jar colour) = 5/6. The prior is 0.5. n "friends" (3, 5 or 7) each
privately draw a ball (with replacement) and all guess the **x** jar. The participant then draws one ball of the other colour, **y**.
A rating follows each friend's guess and the own ball, giving n+1 ratings per sequence.
- **independent**: each friend sees only their own ball.
- **sequential**: each friend sees their own ball and has heard all previous friends' guesses (an information cascade). Under the
  normative model, later guesses carry less evidence.
- **expt1** (`csv/expt1_data_{1,2}of2.csv`, N=338 = 219 + 119): 2 dependency × 3 informant counts (3/5/7), between-subjects. Ratings are on
  a 0–10 likert scale. The authors' csvs already exclude 5 attention-check failures. No RTs.
- **expt2** (`expt2/csv-participants/*.csv`, raw jsPsych, one file per participant): 2 dependency × 5/7 informants, between-subjects, on an
  odds scale (11 points: >40x, 20x, 10x, 5x, 2x, equal, 2x … >40x), with a $0.90 accuracy bonus. There are 215 files. 10 files have no
  ratings and are dropped, leaving 205 subjects. The paper's N=194 is the subjects who passed the attention check (answer "20") and
  completed the task. Two practice trials precede the main task: 12 draws, then a rating. Slider RTs are available.
- **expt3** (`expt3/csv2/expt3_data.csv`, cleaned wide file, N=200): 3 or 5 informants, between-subjects. Dependency is within-subjects:
  both tasks, in random order (`dependency1/2`), each with a different colour pair. Ratings use a 0–10 slider whose start position was
  above or below the scale. Before the tasks, subjects stated which kind of evidence is stronger (`which`). There are two practice
  trials (12 draws). No RTs.
- The manipulation-check follow-up exists only as JASP archives (binary `data.bin`) and is not converted. The replication in
  "Experiment A1" has no data in the folder.

## Column derivation
- `hypothesis` for main-task rows is the **y jar**: the colour of the participant's own ball, against the unanimous testimony. This is
  the authors' reverse-coded `responseR`, where 10 = y. `posterior_reported` = responseR/10 for expt1 and expt3. For expt2,
  `posterior_reported` = odds/(1+odds) from the odds label, with the ">40x" endpoints set to 40:1 (an assumption). Raw and
  reverse-coded scale points are in `extra` (`response_raw`, `response_r`).
- Rows are step-wise: `round` = question−1 and `n_signals` = 1. The signal is friend q's guess (`signal_dir` −1) or the own ball
  (`signal_dir` +1). `prior_objective` is the normative posterior after the previous step (0.5 at round 0). `prior_reported` is the
  subject's previous rating within the same sequence.
- `p_signal_h` / `p_signal_not_h` are normative likelihoods under the authors' model (Whalen, Griffiths & Buchsbaum, 2017), which
  assumes probability-matching informants. In the independent condition, P(guess x | x jar) = 26/36 and P(guess x | y jar) = 10/36. In
  the sequential condition, informant k's guess probability is computed by recursion, given their ball and the earlier guesses. The own
  ball has likelihoods 5/6 vs 1/6. `posterior_bayes` is the cumulative normative posterior.
- `extra` also holds `posterior_naive_independent` (all guesses treated as independent; equals `posterior_bayes` in the independent
  condition), `model_posterior_authors` (joined from `modelPreds_*-1xtime.csv` and `expt3_modelpredictions.csv`), and
  `model_pred_odds_scale_point` (expt2 preds file). It also holds `cum_log_lr`, `stage` (testimony/own_ball/practice), `informants`,
  `dependency`, and demographics.
- Practice rows (expt2/expt3) are labelled `condition` "practice|…". H is the jar the 12 balls were actually drawn from. The jar is
  read from the reveal screen in expt2; in expt3 it is `colourPart1` for PQ1 and its pair partner for PQ2. The authors' PQ2
  recode makes 10 = true jar. `p_signal_h` = (5/6)^k (1/6)^(12−k), where k = number of balls of the true jar's colour; `n_signals` = 12.
- `item_id` = `{n}inf-{ind|seq}-q{q}` (36 cells) or `practice-12draws-{k}jar-{m}other`. `trial` = order within the subject: practice
  first, then the main ratings; for expt3, task 1 then task 2. `rt_sec` = slider rt/1000 (expt2 only).
- `condition` = `{independent|sequential}|{n}_informants`, with `|within|order{1,2}` added for expt3.

## Exclusions
All rows are kept. `extra.excluded_by_authors` is True for expt2 subjects who failed the attention check or did not finish (11
subjects). `attn_pass`, `complete` and `incentive_check_correct` (answered "2" when instructed) are also in `extra`. Expt1 and expt3
are already cleaned by the authors.

## Sanity checks
- The recursion reproduces the authors' predictions to within 1e-9, except for 5-informant sequential q5/q6. There the authors'
  5-informant code gives 0.0655/0.259, but their own 7-informant file (and this recursion) gives 0.0699 at q5. The authors' value is
  kept in `extra.model_posterior_authors`.
- Mean `posterior_reported` (main task) vs normative:

  | Study | Independent (normative) | Sequential (normative) |
  |---|---|---|
  | expt1 | .232 (.098) | .219 (.171) |
  | expt2 | .212 (.075) | .190 (.147) |
  | expt3 | .274 (.119) | .306 (.194) |

  Subjects under-update (they are conservative) and are largely insensitive to dependency, as the paper reports. Practice mean is
  .88 vs Bayes ≈ .99.

## Caveats
- The normative values depend on the probability-matching assumption. If informants maximize, sequential guesses after the first two
  become uninformative.
- The colours shown in each expt3 task are not recoverable from the cleaned csv, so `x_colour`/`y_colour` are "x"/"y".
- In the expt2 odds scale, the conversion of the endpoints to probabilities is approximate (">40x" is open-ended).
