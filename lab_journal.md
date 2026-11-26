# Lab journal: probability updating with autocog

**Question.** When people update probabilities, is the distortion of their reports a *bias applied at decision time* (e.g., misweighting prior and evidence in log-odds) or a by-product of *noisy sampling* (e.g., the Bayesian sampler)?

**Plan.**
1. Compare existing theories on belief-updating datasets.
2. Use autocog's adversarial loop to propose experiments that separate the theories.
3. Revise the theories based on what those experiments show.

---

## 2026-11-16: Data collection

We targeted 9 public belief-updating datasets and downloaded 7 of them into `data/raw/`. Raw data is git-ignored; [data/README.md](data/README.md) lists sources and re-download commands.

| Dataset | Paper | Notes |
|---|---|---|
| hill2017 | Hill (2017), *Learning Together Slowly* | political facts and ego-relevant updating |
| tappin2020 | Tappin, Pennycook & Rand (2020), *Bayesian or Biased?* | political facts, S1/S2/control |
| buser2018 | Buser, Gerhards & van der Weele (2018) | ego-relevant feedback |
| augenblick2025 | Augenblick, Lazarus & Thaler (2025), *Overinference from Weak Signals* | abstract urn tasks, Exp 1a/1b/2 |
| enke_graeber2023 | Enke & Graeber (2023), *Cognitive Uncertainty* | abstract urn tasks, A/B experiments |
| xie_hayes2022 | Xie & Hayes (2022), *Sensitivity to Evidential Dependencies* | 3 experiments |
| howe2022 | Howe et al. (2022), *Base rate neglect and conservatism* | 3 experiments |

**Still missing; both need a manual download:**
- **Möbius et al. (2022).** The INFORMS terms limit use to replication, so we should email the authors before using it for model comparison.
- **Loibl & Leuders (2025).** Needs a PsychArchives login.

## 2026-11-19: Shared trial-level format (commit `e5d6cabe`)

- The scripts in `scripts/convert_data/` convert each dataset to one schema ([data/processed/SCHEMA.md](data/processed/SCHEMA.md)), with one row per elicited probability. Each row records:
  - provenance, subject, item and condition;
  - the prior, which is either objective or reported;
  - the evidence likelihoods;
  - the reported posterior and the Bayesian posterior.
- The combined file is `data/processed/all_datasets.csv.gz`: **98,656 rows, 8,623 subjects, 18 studies**.
- We kept the authors' exclusion criteria as flags and dropped no rows.
- Each dataset has caveats in `data/processed/notes/<dataset>.md`. Coverage is in [data/processed/summary.csv](data/processed/summary.csv).
- **Two kinds of studies:**
  - Urn and abstract studies give an objective prior, so the Bayesian posterior is known.
  - Political-fact and ego studies (Hill, Tappin, Buser) only have a reported prior, so the "Bayesian" benchmark is computed from what each subject said their prior was.
- How closely reported posteriors track Bayes varies widely: r ≈ 0.8 in Augenblick 1b but r ≈ 0 in Howe Exp 3.

## 2026-11-19: Theories and model comparison (commit `ae1b542c`)

**New autocog domain** in `src/probability_updating/`. It has three parts:
- a data loader;
- `ProbabilityUpdatingExperiment`, the experiment class the autocog loop uses;
- a maximum-likelihood fitter scored by cross-validation that holds out whole subjects.

**10 theories** in `theories/probability_updating/`. Each combines a *mean rule* with a *noise type*. Nine theories fill a 3 × 3 grid of these factors; `weak_strong_evidence` adds a fourth mean rule.

| mean rule ↓ / noise → | log-odds Gaussian | probability Gaussian | binomial sampling |
|---|---|---|---|
| Bayes | bayes_noise | bayes_prob_noise | bayes_sampler |
| log-odds weighting, i.e. a linear-in-log-odds (LLO) bias (α prior, β evidence) | llo_bias | llo_prob_noise | llo_sampler |
| compression toward 50% | compression_logodds_noise | linear_compression | bayesian_sampler (Beta(b,b) correction) |
| weak/strong evidence (concave in the log likelihood ratio, \|LLR\|^γ) | weak_strong_evidence | | |

**Every theory has a lapse rate ε.**

**All theories share one reporting stage:**
- Responses use each study's own response scale.
- A fitted fraction of responses is rounded to multiples of 10 or 5. The median rates across studies are about 13% and 10%.

Without this stage, a theory could win simply by putting probability mass on round numbers.

**Results** are in [results/probability_updating/comparison/summary.md](results/probability_updating/comparison/summary.md). Scores are cross-validated negative log-likelihood (NLL) per trial, fitted separately for each study.

**No single theory wins everywhere.** Number of studies won, out of 18:

| Theory | Studies won |
|---|---|
| weak_strong_evidence | 6 |
| llo_sampler | 5 |
| bayesian_sampler | 2 |
| llo_prob_noise | 2 |
| llo_bias | 1 |
| linear_compression | 1 |
| compression_logodds_noise | 1 |
| Bayes-mean theories | 0 |

**Noise type**, taking the best theory with each noise type in each study:

| Noise type | Studies won | Summed loss vs. best (NLL/trial) |
|---|---|---|
| log-odds Gaussian | 8 | **0.228** |
| sampling | 7 | 0.623 |
| probability Gaussian | 3 | 1.071 |

- When sampling loses, it loses by more than log-odds noise does.
- Sampling wins Tappin (all three studies), Buser, Augenblick 1b, Howe 3 and Xie & Hayes 3.
- Log-odds noise wins Enke & Graeber, Hill, Howe 1–2 and Xie & Hayes 2.

**Mean rule:**

| Mean rule | Studies won | Summed loss vs. best (NLL/trial) |
|---|---|---|
| log-odds weighting | 8 | **0.212** |
| weak/strong | 6 | 0.345 |
| compression | 4 | 1.221 |
| Bayes | 0 | 3.069 |

The Bayes mean rule is clearly worst.

**Takeaway.** The mean is clearly distorted: log-odds weighting fits better than compression, and pure Bayes is ruled out. The noise question splits by study. Sampling wins in the political-fact and ego studies; log-odds noise wins in many of the urn studies. The existing designs don't settle it, which motivates designing new experiments.

## 2026-11-24: Adversarial experiment design (commit `c744c761`)

**Wiring.** The theories are now connected to autocog's propose → simulate → test loop through the entry point [main_probability_updating.py](main_probability_updating.py). In each round:
1. An LLM proposes an experiment (the cells of priors and evidence).
2. It then proposes a metric.
3. Both theories are simulated.
4. A Welch t-test at n = 25 subjects with α = .01 decides whether the metric separates the two theories.

There are two modes:
- **Design-only (`--design_only`):** proposes and scores designs only.
- **Full loop (`--ground_truth`):** treats one theory as the "true" subjects, then runs an arbiter and an improver or generator that revise the theories.

**Calibration** (`src/probability_updating/calibrate.py`):
- The theory YAMLs declare wide ranges because they double as fitting bounds. Drawing simulated subjects uniformly from those ranges would let every theory predict almost anything.
- So for simulation, each parameter range is narrowed to the 10th–90th percentile of that theory's per-study estimates. For example:
  - llo_bias: α ∈ [0.73, 1.02], β ∈ [0.14, 0.79];
  - llo_sampler: N ∈ [10, 200].
- The fitted rounding rates are applied inside the simulated experiment, so LLM-written theories get them too.

**Default comparison:** `llo_bias` (decision-time bias) vs. `llo_sampler` (the same mean rule with sampling noise). Within the grid, this pair isolates the noise question.

**Infrastructure fixes:**
- OpenRouter support for Claude Opus 5.5. We dropped the fp8 quantization pin and the `temperature` parameter for first-party models; with either, requests failed with a 404.
- 41 domain tests pass, including end-to-end loop tests with a mocked LLM.

**Runs so far: no complete design yet.**
- **Reasoning effort "max":** the model used up its full reasoning-token budget three times while writing the metric. Opus's first experiment proposal did succeed:
  - **Design:** 20 cells crossing priors .05/.2/.5/.8/.95 with a single draw from 9:1, 99:1 and 50:50 urns.
  - **What it targets:** how the spread of responses, and the share of answers at exactly 0% or 100%, change at extreme posteriors. That is where binomial sampling noise (variance vanishes near the ends) and log-odds noise differ most.
  - **Not saved:** no metric for it got written, so the design was never scored.
- **Reasoning effort "high":** the OpenRouter key ran out of credit on the first call.

## 2026-11-24: Which studies does each theory win, and why?

We compared the studies each theory wins with (a) the paper that proposed the theory and (b) the design features of the studies. The features were computed from `data/processed/all_datasets.csv.gz` and the parameters from `fits.csv`.

**No theory wins the studies from its own paper.** Two of them swap datasets.

| Theory | Proposed in | Its own paper's studies | Where it wins |
|---|---|---|---|
| `weak_strong_evidence` | Augenblick, Lazarus & Thaler 2025 | loses all 3 (by .075, .014, .031 NLL/trial) | Enke & Graeber (all 3), Hill exp2, Howe 1–2 |
| `linear_compression` (cognitive uncertainty) | Enke & Graeber 2023 | loses all 3 (by .12–.39 NLL/trial) | Augenblick 1a |
| `llo_bias` (Grether-style log-odds regression) | used for the main analysis in Buser, Hill and Tappin | wins Hill exp1 only, and it is tied there | Hill exp1 |
| `bayesian_sampler` | Zhu, Sanborn & Chater 2020 (not in our data) | — | Augenblick 1b, Howe 3 |

- **Augenblick's model fits Enke & Graeber's data.** Enke & Graeber's evidence spans a wide range of strengths (|log likelihood ratio| up to 11–12), so the curvature γ is identifiable. It comes out at 0.44–0.66 in all three studies: overreaction to weak evidence and underreaction to strong evidence, as Augenblick et al. propose.
- **Augenblick's data favors noise on the probability scale.** All three Augenblick studies go to models whose noise is Gaussian on the probability scale, or to models that behave like that. Their responses are precise (noise SD 3–7 points) and heavily rounded: 43–44% of answers are multiples of 10 in 1a and 1b.
- **Buser and Tappin get the log-odds weighting their papers used.** The mean is the one from their papers' regressions, but they prefer *sampling* noise over the log-odds Gaussian noise that an OLS regression on log-odds implies.

**What the winners have in common: the response format matters more than the topic or the paper.**

1. **Real sampling wins: `llo_sampler` with a small number of samples.** It wins Tappin S1, S2 and control, Buser, and Xie & Hayes 3, with N ≈ 8–55 samples.
   - **Shared feature: many answers of exactly 0% or 100%** (5–27%, compared with under 1% in Augenblick and most of Enke & Graeber). A small-N sampler predicts these as point masses.
   - **Tappin and Buser also both use subject-reported priors** on political or ego topics.
   - **Xie & Hayes 1 and 3 use an 11-point scale (0–10).** That coarse discreteness also suits sampling: in expt1 the sampler is within .0002 of the winner.
2. **Log-odds noise wins: Enke & Graeber, Xie & Hayes 2.**
   - **Xie & Hayes 2 is the clearest case.** It is the only study whose slider is laid out in odds steps. Its winner, `compression_logodds_noise`, has log-odds noise, so noise shaped like the response scale fits best.
   - **Enke & Graeber pose many items, each with an objective prior,** and very strong evidence. Log-odds noise handles the far tails of that evidence range.
3. **Probability-scale noise wins: Augenblick 1a, 1b and 2, Xie & Hayes 1.**
   - All give objective priors, mostly near 50%, and use precise response formats (boxes that sum to 100, or sliders).

**Wins that aren't evidence for any mechanism:**
- **Hill exp1 and exp2.** The winning fits give almost zero weight to the evidence (β ≈ 0.005), hit the lower bound on noise (σ = 0.05), and set the lapse rate at 43–50%. That describes "most people repeat their own (often extreme) prior; the rest answer at random." It is about people not updating, not about bias versus sampling.
- **Howe 1–3.** One trial per subject, the answer is the mean of a reported distribution, and there are only 300–600 trials. The top 3–4 theories are tied to the fourth decimal.
- **Augenblick 1b's "sampling" win.** The sampler hits the upper bound N = 200, where sampling noise is negligible, so this is probability-Gaussian noise under another name.

**Implication for design.** The noise type seems to track the response format. Two design dimensions follow:
- Vary the response scale (percent, odds, a coarse 0–10 scale) within subjects on the same items.
- Measure how answers pile up at 0% and 100% as the posterior moves toward the ends of the scale.

Opus's one surviving design proposal already targets the second.

**Caveats.**
- Each study is fitted with pooled parameters.
- Several margins are tiny.
- Hill's and Howe's wins should be left out of any "who wins where" tally.

## 2026-11-26: First design round (Opus 5.5, `llo_bias` vs `llo_sampler`)

Results are in `results/probability_updating/autocog/design_llo_bias_vs_llo_sampler_anthropic-claude-opus-5.5_run1/designs.md`. The run used reasoning effort "high" via OpenRouter, with 50 simulated subjects per theory. The first metric each side proposed passed the test (Welch p < .01 at N = 25 subjects per theory).

| Proposed by | Design (12 cells × 5 reps) | Metric | llo_bias | llo_sampler | p |
|---|---|---|---|---|---|
| `llo_bias` | mixes near-50%, conflict, extreme-prior and pure-prior cells | prior weight α estimated from median log-odds in prior-dominated cells | 0.77 | 1.00 | .0007 |
| `llo_sampler` | pure-prior cells (.15/.3/.7/.85, uninformative evidence), evidence-only cells at prior .5, conflict cells | median folded (report − prior) in the pure-prior cells | −0.05 | +0.03 | .00005 |

**Problem: both accepted metrics separate the theories by the prior weight α, not by the noise mechanism.**
- The two theories have the same mean rule, but their calibrated α ranges differ:
  - llo_bias: [0.73, 1.02];
  - llo_sampler: [0.77, 1.94].
- The upper tail of llo_sampler's range comes from fits where α is unidentified or at its bound:
  - Augenblick 1a, where every prior is 50%, so α = 1.5 is unconstrained;
  - Howe 1 and 3, with α ≈ 3 on 300–600 one-shot trials.
- So the "distinguishing" experiment tests a calibration artifact: does the prior get over-weighted? That is not the question of bias vs. sampling.

**The design rationales do target the right signatures, but the metrics didn't use them.** Those signatures are:
- how the spread of reports changes with how extreme the belief is: constant spread in log-odds vs. binomial spread;
- answers piling up at 0% and 100%;
- whether averaging pulls the mean toward 50%.

The LLMs went for the easiest t-test win.

**Fix before the next round** (open items, item 1).

## 2026-11-26: Calibration fix and second design round

**Calibration change** in `calibrate.py`, now the default in `main_probability_updating.py`:
- **Unidentified estimates are dropped** before calibrating. These are estimates of how the belief is formed that a study could not pin down:
  - α where the prior never varies: Augenblick 1a, Howe 1 and 3;
  - β where the evidence never varies: Tappin control, Howe 3;
  - γ with fewer than 3 evidence strengths;
  - any of these sitting on the fitting bound.
- **Shared parameters get one range.** Parameters declared by both seed theories are calibrated from their pooled estimates, giving one range:
  - α ∈ [0.73, 1.04];
  - β ∈ [0.12, 0.88];
  - ε (lapse rate) shared too.
- The old behaviour is available with `--keep_unidentified` and `--no_shared_params`.

**Round 2 results** are in `.../design_llo_bias_vs_llo_sampler_anthropic-claude-opus-5.5_run2/designs.md`. The run made 4 Opus calls, and the first metric each side proposed passed the test.

| Proposed by | Design | Metric | llo_bias | llo_sampler | p |
|---|---|---|---|---|---|
| `llo_bias` | 12 cells × 5 reps; priors .03–.97 crossed with single signals, log LR −2.9 to +3.9; zero-evidence cells at 3 priors | per-subject slope of log(within-cell trimmed range) on log(m(1−m)) | 0.45 | 0.19 | .0024 |
| `llo_sampler` | 10 cells × 6 reps; a 50%/no-evidence anchor cell plus extreme cells (beliefs ≈ .02–.995) | log ratio of median absolute deviation in tail cells vs. center cells | −1.13 | −0.38 | .0011 |

**Both metrics now test the noise mechanism.** Both measure how fast spread around the belief shrinks as the belief moves toward 0 or 1:
- **Constant log-odds noise:** SD ∝ p(1−p), so spread collapses quickly in the tails.
- **Binomial sampling:** SD ∝ √(p(1−p)/N), so spread shrinks only like the square root.

The two designs converged on essentially the same idea.

**Caveats:**
- **The effects are smaller than Opus expected.** It predicted slopes near 1 vs. 0.5; the simulations give 0.45 vs. 0.19. Rounding, lapses, motor noise τ and the many fits with N ≈ 200 (where sampling is nearly noiseless) all dilute the difference.
- **Between-subject variance is large** (t ≈ 3.2–3.5 at N = 25), so a real study needs more subjects or more reps.
- **Repeated identical cells (5–6 reps) invite remembered answers,** which would shrink within-cell spread under either theory. A real study would need disguised repeats, e.g. different cover stories or colours with the same prior and LR.
- **Both designs assume a 0–100% response scale.** The cross-study analysis suggests the response format itself shapes the noise, which these designs don't vary.

## 2026-11-26: Full loop, `weak_strong_evidence` as ground truth (1 round)

**Setup.**
- `weak_strong_evidence` stands in for people. Its calibrated ranges are α [0.64, 1.00], β [0.32, 1.01], γ [0.44, 1.01] and σ [0.17, 1.13].
- Seed theories: `llo_bias` (π1) and `llo_sampler` (π2).
- Model: Opus 5.5, reasoning effort "high".
- Results: `results/probability_updating/autocog/gt_weak_strong_evidence_llo_bias_vs_llo_sampler_anthropic-claude-opus-5.5_run1/`, in `designs.md`, `leaderboard.md` and `rounds/round_000/`.

**Designs.** Both target the noise again, and both passed the test at p < 10⁻⁶.

| Proposed by | Metric | llo_bias | llo_sampler | ground truth |
|---|---|---|---|---|
| π1 | share of answers at exactly 0% or 100%, among near-extreme answers (≥ 90% after folding) | 0.17 | 0.37 | **0.14** |
| π2 | log ratio of spread in extreme vs. mid cells | −1.61 | −0.41 | **−1.50** |

**Arbitration.** The arbiter scored π1 = 1.0 and π2 = 0.0. That is correct on the noise question: the ground truth has log-odds Gaussian noise, as `llo_bias` does, and produces few exact 0%/100% answers, with spread that collapses toward the ends of the scale.

**New theory (π3, replaced π2).** The theory generator wrote a *noisy log-odds encoding* model:
- The prior log-odds and the evidence LLR are each encoded with Gaussian noise that has a constant part and a part growing with magnitude (Weber-like): sd = s0 + w·|x|.
- Each is then decoded with Bayesian shrinkage toward 0 (gain k = λ²/(λ² + sd²)).
- The report is expit of the decoded log-odds, plus a small amount of response noise and a lapse rate.

It scored 0.991, close to π1. Its second attempt kept the code and changed only the parameter ranges, which cut its loss from 0.33 to 0.06.

**Interpretation.**
- **The loop answers the noise question correctly.** It discarded the sampler and kept log-odds noise.
- **It did not recover the ground truth's mean rule, and couldn't have.** Neither design varied the evidence strength enough to test curvature in |LLR|, so π1, with linear weights, scores 1.0 even though the truth has γ < 1.
- **π3 partly reinvents the weak/strong pattern from a mechanism.** Its shrinkage grows with magnitude, so strong evidence is under-weighted more than weak evidence, which is concave in |LLR|. But its gain k is at most 1, so it can never *over*-react to weak evidence, which is the core claim of Augenblick et al. A design that varies |LLR| finely, including weak signals at prior 50%, would separate π1, π3 and the truth. The arbiter itself recommended exactly this.
- **π3 predicts more variance in conflict cells than at 50%/no evidence, and a link between noise and conservatism.** It is a testable alternative to the hypothesis that the bias comes in at decision time.

**Next:** run more rounds (`--n_rounds 3`, resuming this run) so that π1 vs. π3 get designs that test curvature.

### Rounds 1–3 (same run, resumed)

Note: `--n_rounds` counts *additional* rounds when resuming. `--n_rounds 3` therefore ran rounds 1–3, giving 4 rounds in total, not the 2 more we intended.

**Theory succession.** In each round the loser was replaced:

| Round | Pair | Replaced | New theory |
|---|---|---|---|
| 1 | π1 vs π3 | π3 (Weber encoding) | **π4**: partial integration. Answers come from one cue with probability ρ, and output is soft-bounded in log-odds. |
| 2 | π1 vs π4 | π1 (`llo_bias`) | **π5**: π4's soft bound, without the cue selection, plus stimulus-independent off-task responding: uniform guessing, polarity reversal (100 − R) and "50%" answers. |
| 3 | π5 vs π4 | π4 | **π6**: sub-additive pooling of congruent cues (a p-norm), shrunk conflict differences, "salient value" answers (95/99/100) for near-certain beliefs, snapping to multiples of 5, and the off-task mixture. 11 parameters. |

**Final arbiter scores:** π6 0.82, π5 0.66, π1 0.58, π3 0.19, π2 0.07, π4 0.00.

**Designs, and how close each theory came to the ground truth:**
- **Round 1, π1: the right design, the wrong metric.**
  - The design is the magnitude sweep the arbiter asked for: |LLR| from 0.85 to 4.6 at prior 50%, matched prior-only cells and conflict cells.
  - But the metric pools the ratio β/α over all magnitudes, so the curvature disappears.
  - Ground truth 0.67; π1 0.53; π3 0.84.
- **Round 1, π3: falsified its own theory.** Its variance-coupling metric gave ground truth 0.15 and π1 0.07, against π3's predicted 0.54. That means no Weber-scaled noise in the ground truth.
- **Rounds 2–3: the designs drift to the lapse component.** They measured:
  - the rate of answers no route can reach without a lapse;
  - the off-task mass in single-source cells;
  - an excess of answers far from the theory's prediction in conflict cells.

  The ground truth matched π1 or π5 on these. They are tests of nuisance structure, not of how prior and evidence are combined.

**Assessment.**
- **The loop did not recover the ground truth's mechanism.** The true theory uses a power law in |LLR| (γ < 1). The surviving theories approximate that concavity indirectly:
  - π4–π6 use a soft bound at extreme log-odds;
  - π6 also pools congruent cues sub-additively.

  Neither is a per-cue power law, and π6 cannot produce overreaction to weak signals.
- **The surviving theories accumulate machinery the ground truth doesn't have:**
  - polarity reversals;
  - "don't know" 50% answers;
  - salient-value answers;
  - snapping to multiples of 5, which duplicates the shared rounding stage even though the prompt describes it.

  The scoring has no complexity penalty, so this extra machinery costs nothing.
- **Why the loop drifts.** The ground truth's calibrated lapse rate ε reaches up to 0.44. Lapses are then a large, easy-to-detect source of differences, and the design LLM goes after them once the noise question is settled.

**Fixes to consider before more full-loop runs:**
1. **Treat lapses like rounding.** Fix ε at a shared value, or tell the LLM it is a nuisance stage the theories shouldn't model. Narrowing the ground truth's lapse range would also help.
2. **Penalise complexity** in the arbiter or generator scoring, e.g. by the number of parameters or by cross-validated fit.
3. **Ask for metrics that keep the design's structure,** e.g. the slope of median log-odds against |LLR| at prior 50% rather than a pooled ratio. Alternatively, let a design carry several metrics.

---

## Open items / next steps

1. **Next design rounds.** The calibration is fixed (2026-11-26). Options:
   - pit other pairs against each other, e.g. `--seeds weak_strong_evidence bayesian_sampler`;
   - ask for designs that vary the response format and use disguised repeats;
   - run more rounds (`--n_rounds`).

   Command for a design round:
   ```bash
   uv run --with anthropic python main_probability_updating.py --design_only --llm_provider openrouter --llm_model anthropic/claude-opus-5.5 --reasoning_effort high --run_id 2
   ```
2. **Run the full loop** with a ground-truth theory, e.g. `--ground_truth weak_strong_evidence`. This tests whether the arbiter and improver recover the true theory.
3. **Get Möbius (2022) and Loibl & Leuders (2025)**, convert them, and refit.
4. **Reproducibility:** add `anthropic` to `pyproject.toml`; for now commands need `--with anthropic`. Optionally make the OpenRouter `max_tokens` (currently 128k) a setting; OpenRouter checks the full limit against the remaining credit before each call.
5. **Questions for discussion:**
   - The noise type splits by study, and the evidence so far points to how the response is elicited (see the 2026-11-24 analysis). Can we separate that from the domain and from whether the prior is reported or objective?
   - Should we tighten the parameter bounds, or flag degenerate fits automatically? Examples: the sampler hitting N = 200, and Hill's fits with β ≈ 0 and a lapse rate around 45%.
   - Should we fit hierarchically, with subject-level parameters, instead of pooling each study?
   - Within-subject response variability across repeated identical items is the most direct test of sampling vs. bias. Do any of the existing datasets have repeated items?
6. **Git:** nothing is pushed. Three local commits are ahead of `origin/main`.
