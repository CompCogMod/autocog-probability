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

---

## Open items / next steps

1. **Fix the calibration so that compared theories differ only in their noise, then rerun the design round.** Round 1 (2026-11-26) separated the theories through α ranges that are calibration artifacts. Options:
   - give both theories the same mean-rule parameter ranges (α, β);
   - calibrate only from studies where a parameter is identified (prior varies; estimate not at a bound);
   - or both.

   Command used for round 1:
   ```bash
   uv run --with anthropic python main_probability_updating.py --design_only --llm_provider openrouter --llm_model anthropic/claude-opus-5.5 --reasoning_effort high --run_id 1
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
