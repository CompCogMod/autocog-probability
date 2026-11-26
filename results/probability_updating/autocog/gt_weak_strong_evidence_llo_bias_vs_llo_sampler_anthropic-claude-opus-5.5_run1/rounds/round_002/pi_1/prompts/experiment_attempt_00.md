# experiment_attempt_00

## System Prompt

You are a renowned cognitive scientist designing an experiment in the Probability Updating domain.

Your goal is to be an adversarial collaborator: propose a design whose outcomes would be predicted by your advocated theory but NOT by the competing theory. Both are provided below.

A useful proposal targets a *quantitative* dissociation between the two theories — how they respond differently to specific stimuli in addition to differences in overall performance.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects are told a prior probability that hypothesis H is true (e.g. that a ball was drawn from urn A rather than urn B), then shown evidence whose probability under H and under not-H is known (e.g. the colours of several draws, with the urns' compositions stated). They report the probability that H is true on a 0-100% scale.

A belief-updating experiment. Each trial states a prior P(H) and shows evidence whose probability under H (`trial_p_signal_h`) and under not-H (`trial_p_signal_not_h`) is known to the subject; `trial_n_signals` is how many draws make up that evidence. The subject reports P(H) from 0 to 100%. Choose the unique trials (all four lists, same length) so that the candidate theories make different predictions: vary the prior and the strength of the evidence (log likelihood ratio) independently, and include cells where theories disagree about both the mean report and its trial-to-trial variability. Each unique trial is repeated K = max(1, 60 // n_unique) times per subject in a fresh random order. There is no feedback. As in human data, every response then passes through a shared reporting stage that does not depend on the theory: with probability 0.13 it is rounded to the nearest multiple of 10%, otherwise with probability 0.10 to the nearest multiple of 5%. Theories should not model this rounding themselves, and metrics should not rely on exact values that rounding would blur.

Subjects see the following instructions:
On each trial you will be told how likely it is, before any evidence, that a statement H is true. You will then see some evidence, together with how likely that evidence is if H is true and if it is false. Report how likely you think it is now that H is true, from 0% to 100%.

## ADVOCATED THEORY
**Description:** Probability distortion is a systematic bias applied when the judgment is formed: people combine prior and evidence additively in log-odds space, as Bayes prescribes, but with the wrong weights. The posterior log-odds equal alpha times the prior log-odds plus beta times the log likelihood ratio. alpha < 1 is base-rate neglect (the prior is underweighted), beta < 1 is conservatism (the evidence is underweighted) and beta > 1 is overreaction; alpha = beta = 1 recovers Bayes. Because the distortion is a deterministic transformation, the same stimulus always yields the same central judgment; trial-to-trial variability is a separate, constant response noise in log-odds space (sigma), plus a lapse rate epsilon. The theory therefore predicts that mean distortion and response variability are unrelated: changing how noisy responses are does not change their average, and the average is a fixed linear function of prior and evidence log-odds.


**Parameters:**
- alpha: [0.7325, 1.037]
- beta: [0.1248, 0.8789]
- sigma: [0.1955, 1.207]
- epsilon: [0.04915, 0.4149]

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    # See bayes_noise.yaml for the stimulus / output contract.
    from scipy.special import ndtr
    K = 101
    scalar = np.ndim(stimulus["prior"]) == 0
    prior = np.clip(np.atleast_1d(np.asarray(stimulus["prior"], dtype=float)), 1e-6, 1 - 1e-6)
    llr = np.atleast_1d(np.asarray(stimulus["log_lr"], dtype=float))
    alpha = float(parameters["alpha"])
    beta = float(parameters["beta"])
    mu = alpha * (np.log(prior) - np.log1p(-prior)) + beta * llr

    edges = np.clip((np.arange(K + 1) - 0.5) / (K - 1), 0.0, 1.0)
    with np.errstate(divide="ignore"):
        z_edges = np.log(edges) - np.log1p(-edges)
    sigma = float(parameters["sigma"])
    probs = np.diff(ndtr((z_edges[None, :] - mu[:, None]) / sigma), axis=1)

    eps = float(parameters["epsilon"])
    probs = (1.0 - eps) * probs + eps / K
    probs = probs / probs.sum(axis=1, keepdims=True)
    return probs[0] if scalar else probs
```

**`policy source code`:**
```python
def policy(probabilities):
    probabilities = probabilities / probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))
```


## COMPETING THEORY
**Description:** Stochastic partial integration with a soft-bounded response range.

On average, people combine prior and evidence additively in log-odds space with distorted weights: L = alpha*logit(prior) + beta*LLR. Here alpha<1 is base-rate neglect and beta<1 is conservatism. Integration is not guaranteed on every trial.

(a) Integration rule. When both cues are informative (prior differs from 50% and LLR differs from 0), the subject fails to integrate with probability rho and answers from a single cue:
- with probability q it anchors on the prior (alpha*L0);
- with probability 1-q it goes by the evidence alone (beta*LLR), a representativeness shortcut.
When only one cue is informative, the single-cue route coincides with integration, so single-source cells show no extra spread.

(b) Soft-bounded output. The chosen internal log-odds passes through a saturating transform: L' = sign(L)*Delta*tanh((|L|/Delta)^kappa)^(1/kappa). With kappa>1 the transform is nearly linear up to about |L|≈Delta and then saturates sharply at Delta. It therefore compresses only beyond about the 97-99.9% level, and it never makes encoding gains depend on magnitude in the moderate range. kappa=1 recovers the plain tanh bound.

(c) Noise. Constant Gaussian response noise sigma is added in log-odds around the bounded value, plus a uniform lapse epsilon.

Predictions:
- Single-source cells behave like a fixed-weight model, with compression only at the far extremes.
- Two-cue cells carry extra, possibly bimodal spread from the strategy mixture.
- Congruent medians are sub-additive.
- Exact-100 reports arise mostly via rounding of reports in the high 90s.

**Parameters:**
- alpha: [0.6, 1.1]
- beta: [0.3, 1.0]
- rho: [0.0, 0.4]
- q: [0.3, 0.8]
- Delta: [3.5, 7.0]
- kappa: [1.5, 4.0]
- sigma: [0.3, 1.0]
- epsilon: [0.0, 0.15]

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    from scipy.special import ndtr
    K = 101
    scalar = np.ndim(stimulus["prior"]) == 0
    prior = np.clip(np.atleast_1d(np.asarray(stimulus["prior"], dtype=float)), 1e-6, 1 - 1e-6)
    llr = np.atleast_1d(np.asarray(stimulus["log_lr"], dtype=float)) * np.ones_like(prior)
    L0 = np.log(prior) - np.log1p(-prior)

    alpha = float(parameters["alpha"])
    beta = float(parameters["beta"])
    rho = float(parameters["rho"])
    q = float(parameters["q"])
    Delta = max(float(parameters["Delta"]), 1e-3)
    kappa = max(float(parameters["kappa"]), 1e-3)
    sigma = max(float(parameters["sigma"]), 1e-3)
    eps = float(parameters["epsilon"])

    both = ((np.abs(L0) > 1e-6) & (np.abs(llr) > 1e-6)).astype(float)
    w_int = 1.0 - rho * both
    w_p = rho * q * both
    w_e = rho * (1.0 - q) * both

    mu_int = alpha * L0 + beta * llr
    mu_p = alpha * L0
    mu_e = beta * llr

    def bound(x):
        u = np.abs(x) / Delta
        return np.sign(x) * Delta * np.tanh(u ** kappa) ** (1.0 / kappa)

    edges = (np.arange(K + 1) - 0.5) / (K - 1)
    z_edges = np.empty(K + 1)
    z_edges[0] = -np.inf
    z_edges[-1] = np.inf
    inner = edges[1:-1]
    z_edges[1:-1] = np.log(inner) - np.log1p(-inner)

    def bins(mu):
        m = bound(mu)
        cdf = ndtr((z_edges[None, :] - m[:, None]) / sigma)
        return np.clip(np.diff(cdf, axis=1), 0.0, None)

    probs = (w_int[:, None] * bins(mu_int)
             + w_p[:, None] * bins(mu_p)
             + w_e[:, None] * bins(mu_e))
    probs = probs / np.maximum(probs.sum(axis=1, keepdims=True), 1e-300)
    probs = (1.0 - eps) * probs + eps / K
    probs = probs / probs.sum(axis=1, keepdims=True)
    return probs[0] if scalar else probs
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    p = np.asarray(probs, dtype=np.float64)
    p = np.clip(p, 0.0, None)
    p /= p.sum()
    return int(np.random.choice(len(p), p=p))
```


## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
[0] The design targets how response variability scales with the level of the distorted belief, while keeping the mean model (alpha times prior log-odds plus beta times LLR) identical in both theories. Prior log-odds and evidence LLR are crossed independently so alpha and beta are identifiable. Each trial uses a single signal (n=1) so the LLR is simply log(p_h/p_not_h), with values 0, +/-1.39, +/-2.94 and +/-4.6. Priors range over 0.01, 0.05, 0.2, 0.5, 0.8, 0.95 and 0.99.

The cells are:
(a) A pure 50% cell, where both theories predict maximal spread.
(b) Evidence-only cells at prior 0.5.
(c) Prior-only cells at 0.01 and 0.99 with uninformative evidence.
(d) Congruent extreme cells (0.99 with 0.99/0.01, 0.01 with 0.01/0.99, 0.8 with 0.95/0.05, 0.05 with 0.2/0.8). These push the distorted belief toward 0 or 100.
(e) Conflict cells (0.95 with 0.05/0.95, 0.2 with 0.8/0.2). These land near 50% but via large opposing components.

Advocated theory (logit-normal noise with constant sigma): the probability-scale SD is about sigma*p(1-p). It collapses quickly toward the extremes, so the SD at 95% is roughly 0.19 times the SD at 50%. The median sits at expit(mu) and the mean is pulled toward 50%. Exact 0% or 100% reports should be essentially absent, apart from lapses.

Competing theory (binomial sampling): the SD is sqrt(p(1-p)/N), so the SD at 95% is about 0.44 times the SD at 50%, a much flatter profile. The mean equals the target exactly. Extreme cells should produce a large mass of exact 0/100 (k=0 or k=N) reports, e.g. more than 50% of responses when the target is 97% and N=20.

Key diagnostics, all robust to rounding to 5 or 10%:
- The ratio of spread (IQR or SD) in the extreme cells versus the 50% and conflict cells.
- The frequency of 0/100 endpoint reports in extreme cells relative to the lapse baseline.
- Whether conflict cells, which share a near-50% belief, show the same variability as the pure 50% cell. Both theories say they should, which checks that variance depends only on the belief level.

The advocated theory predicts steep variance compression and few endpoint reports. The competing theory predicts a shallow sqrt-shaped profile and heavy endpoint mass. Twelve unique trials give K=5 repeats each.

[1] This design tests the advocated theory's distinctive signature: binomial report variability around a log-odds-distorted belief. The alternative is logit-normal noise with constant sigma around the same distorted belief. Both theories share the mean structure alpha*prior log-odds + beta*LLR. The design therefore crosses priors (0.05, 0.2, 0.5, 0.8, 0.95) with single-draw evidence (LLR 0, ±1.10, ±2.20, ±3.89) so alpha and beta are identified separately from the variability structure.

The design deliberately places many cells where the distorted belief lands in the 75-95% (or 5-25%) band rather than at the far extremes. That is where the two variance laws separate most clearly while the means remain matched.

Binomial prediction (advocated): SD = sqrt(p(1-p)/N). For example, with N≈20 the SD is about 0.067 at p=0.9 versus 0.112 at 0.5, a ratio of about 0.6.

Logit-normal prediction (competing): SD ≈ sigma*p(1-p), so the same comparison gives a ratio of about 0.36. Its spread therefore collapses much faster away from 50%.

Congruent cells (0.8 with 0.9/0.1; 0.95 with 0.98/0.02; mirrored low cells) push the belief toward the ends. There, binomial sampling predicts a substantial pile-up of exact 0%/100% reports (k=0 or k=N), e.g. P(k=N)=0.9^20≈12% at p=0.9, and more at higher p. Logit-normal noise yields a smooth tail with endpoint mass that grows only slowly and continuously. Exact 0 and 100 survive the 5/10% rounding stage, so endpoint frequency relative to the lapse floor is a rounding-robust diagnostic.

Prior-only cells (0.8, 0.95 with uninformative evidence) and evidence-only cells at prior 0.5 let us check that variability depends only on the level of the distorted belief, not on its source.

Conflict cells (0.95 with 0.1/0.9; 0.2 with 0.9/0.1) put the belief near 50-75% via opposing components. Both theories predict variance set by the belief level alone, which serves as a consistency check.

Key metrics:
(1) the slope of IQR/SD versus the model-implied belief level, comparing the binomial sqrt profile with the logit-normal p(1-p) profile;
(2) endpoint (0/100) frequency in congruent cells versus the 50% cell;
(3) mean versus median asymmetry. Binomial means equal the target, whereas logit-normal means are pulled toward 50% relative to medians.

Twelve unique trials give K=5 repeats each.

[2] Goal: separate a fixed linear log-odds mis-weighting with constant noise (advocated) from magnitude-dependent Weber encoding noise with Bayesian shrinkage (competing). Earlier designs compared variance laws on the probability scale at matched means. This design targets two different signatures: whether gain falls as inputs grow, and whether noise depends on the source of the belief. All trials use one draw, so LLR = log(p_h/p_nh).

(1) Linearity sweeps.
- Evidence-only, prior 0.5: LLR 0, 0.85, 1.39, 2.20, 2.94, 4.60.
- Prior-only, uninformative evidence 0.5/0.5: priors 0.7, 0.8, 0.9, 0.95, 0.99 (L0 = 0.85 to 4.6).
- Advocated: median response log-odds is exactly linear in LLR (slope beta) and in L0 (slope alpha), with constant marginal gain. Medians follow expit(beta*LLR) and expit(alpha*L0).
- Competing: gains k = lambda^2/(lambda^2+(s0+w|x|)^2) fall with magnitude, so the curves are concave and saturate. Increments shrink: 0.95 to 0.99 adds little, and extreme single-source cells stall near 93-97%.
- Metric: local slope of median log-odds (or of the median-report rank) for the upper half versus the lower half of each sweep. The advocated theory predicts equal slopes; the competing theory predicts the upper-half slope is clearly smaller. Medians at 5-10% resolution in the 60-95% range are robust to rounding.

(2) Source-dependent variability at matched belief.
- Conflict cells pit a strong prior against equally strong opposing evidence: 0.99 vs 0.01/0.99, 0.95 vs 0.05/0.95, 0.9 vs 0.1/0.9.
- Advocated: noise is a constant sigma around mu = (alpha - beta)*|x|. A conflict cell's spread should therefore equal that of any single-source cell with the same median. The low-magnitude single-source cells (prior 0.7 or 0.8; LLR 0.85 or 1.39) provide those median-matched comparisons.
- Competing: conflict cells carry encoding noise from two large components. The predicted log-odds SD is about 1.2 versus about 0.35 for the pure 50/50 cell, and it grows across the three conflict magnitudes.
- Metric: IQR of reports in each conflict cell divided by the IQR of the median-matched single-source cell. The advocated theory predicts a ratio of about 1, flat across conflict magnitude; the competing theory predicts a ratio well above 1 that increases with magnitude.
- A further check: under the advocated theory, the conflict-cell median shifts monotonically with magnitude at rate (alpha - beta). Under the competing theory it stays close to the mean-matched value but its variance inflates.

(3) Congruent moderate cell: prior 0.8 with evidence 0.8/0.2. This checks additivity against the two single-source cells (prior 0.8 alone; LLR 1.39 at prior 0.5).

Summary: 15 unique trials, K = 4 repeats each.

[3] Goal: test two signatures of Weber-scaled encoding noise with Bayesian shrinkage that earlier designs did not isolate. Both are framed so the linear model makes alpha/beta-free predictions. All trials use one draw (n=1), so LLR = log(p_h/p_nh).

(1) Split-versus-single-source contrast. This is the main new test. The same total log-odds is delivered either from one source or split equally between prior and evidence.
- Total 2.20: prior 0.9 alone, or LLR 2.20 alone (0.9/0.1), versus split prior 0.75 plus evidence 0.75/0.25 (1.10+1.10).
- Total 3.89: prior 0.98 alone, or LLR 3.89 alone (0.98/0.02), versus split prior 0.875 plus evidence 0.875/0.125 (1.95+1.95). A mirrored low split cell (0.125 with 0.125/0.875) is added for reliability.
- Linear model: split logit = (alpha+beta)*x, and the single-source cells give alpha*2x and beta*2x. So the split cell's median logit must equal exactly the average of the two single-source logits, whatever alpha and beta are.
- Advocated model: gains fall with magnitude, so splitting yields more total weight than one large source. The split cell should exceed the average. With typical parameters (s0=.3, w=.2, lambda_p=2, lambda_e=1.5) the excess is about 0.15 logits at total 2.2 and about 0.55 logits at total 3.9 (about 96.6% versus 94%). The excess should grow with total magnitude.

(2) Gain at small versus large magnitude.
- Sweeps: evidence-only LLR 0.41, 1.10, 2.20, 3.89 (prior 0.5), and prior-only 0.6, 0.75, 0.9, 0.98.
- Advocated: near-Bayesian gain for weak inputs (k about 0.9-0.95), falling to about 0.65-0.75 for strong inputs. The ratio of response logit to input should decline across each sweep.
- Linear model: the ratio is constant (beta and alpha respectively).
- Metric: per-cell mean response logit divided by input. This is computed on means over repeats and subjects, which is less sensitive to 5/10% rounding than medians.

(3) Variance coupling.
- Advocated: log-odds SD grows with input magnitude, and it is larger in split and single-source high-magnitude cells than in the pure 0.5/0.5 baseline.
- Linear model: constant log-odds sigma.
- Across subjects, the advocated model predicts that people with larger response variability in the 1.1-2.2 cells (where rounding hurts least) show stronger gain decline and stronger conservatism. The linear model predicts no correlation between variability and alpha/beta.

Twelve unique trials give K=5 repeats each.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- trial_priors: Prior P(H) for each unique trial, each strictly between 0 and 1.
- trial_p_signal_h: P(observed evidence | H) for each unique trial.
- trial_p_signal_not_h: P(observed evidence | not-H) for each unique trial.
- trial_n_signals: Number of signals (draws) making up each trial's evidence.


## Response

```json
{
  "rationale": "Goal: test whether two-cue trials are integrated deterministically on every trial, as the advocated fixed-weight log-odds model with constant noise claims, or through a stochastic mixture of strategies. All trials use one draw (n=1), so LLR = log(p_h/p_nh). The design mixes single-source anchor cells with two-cue cells. Unlike earlier designs, it tests mixture structure and additivity rather than variance-versus-level laws.\n\n(A) Single-source anchors. These cells calibrate alpha, beta and sigma without any mixture contribution.\n- Baseline: 0.5 with 0.5/0.5.\n- Evidence only at prior 0.5: LLR +/-1.39 and +2.94.\n- Prior only: 0.8 and 0.95.\n\n(B) Congruent cells: 0.8 with 0.8/0.2, 0.95 with 0.95/0.05, and 0.05 with 0.05/0.95.\n- Advocated: the median logit equals exactly the single-source prior logit plus the single-source evidence logit (strict additivity). Spread equals the baseline sigma.\n- Competing: medians are sub-additive. There is also a heavy tail or second mode at the single-cue values (about 80% or about 95%). The 0.95 congruent cell is further pulled down by the soft bound.\n\n(C) Conflict cells: 0.95 with 0.05/0.95, 0.05 with 0.95/0.05, 0.8 with 0.2/0.8, 0.95 with 0.2/0.8, and 0.8 with 0.05/0.95.\n- Advocated: one unimodal logit-normal distribution centred at (alpha*L0 - beta*|LLR|), with the same IQR as median-matched single-source cells. The proportion of reports on the far side of 50% is predictable from sigma alone.\n- Competing: responses mix a prior-anchored mode (near the prior-only cell's median) and an evidence-only mode (near the evidence-only cell's median, on the opposite side of 50%). This yields inflated IQR, bimodality, and an excess of opposite-side reports.\n\n(D) Near-single-cue probes: 0.95 with weak evidence 0.4/0.6, and 0.6 with strong evidence 0.95/0.05. These are the sharpest dissociation because both cues count as informative, which triggers the competing mixture.\n- Advocated: each distribution is a slightly shifted copy of its single-source partner (prior 0.95 alone; LLR 2.94 at prior 0.5), with identical spread.\n- Competing: an extra mode appears far from the partner. In the first probe, rho*(1-q) of responses fall near 40-45% (the evidence-only answer). In the second, rho*q of responses fall near 60% (the prior-only answer).\n\nMetrics, using 10%-wide bins so they are robust to rounding:\n1. IQR in each two-cue cell divided by the IQR of its single-source partner. Advocated predicts about 1; competing predicts more than 1.\n2. The fraction of two-cue responses falling within \u00b110% of the single-cue predictions taken from the anchor cells, compared against the logit-normal expectation.\n3. Additivity residuals in the congruent cells, computed from the empirical single-source medians. Advocated predicts 0; competing predicts negative residuals.\n4. Within-subject consistency: the advocated model says a subject's spread is the same in single-source and two-cue cells.\n\nFifteen unique trials give K=4 repeats each.",
  "trial_priors": [
    0.5,
    0.5,
    0.5,
    0.5,
    0.8,
    0.95,
    0.8,
    0.95,
    0.05,
    0.95,
    0.05,
    0.8,
    0.95,
    0.8,
    0.95,
    0.6
  ],
  "trial_p_signal_h": [
    0.5,
    0.8,
    0.2,
    0.95,
    0.5,
    0.5,
    0.8,
    0.95,
    0.05,
    0.05,
    0.95,
    0.2,
    0.2,
    0.05,
    0.4,
    0.95
  ],
  "trial_p_signal_not_h": [
    0.5,
    0.2,
    0.8,
    0.05,
    0.5,
    0.5,
    0.2,
    0.05,
    0.95,
    0.95,
    0.05,
    0.8,
    0.8,
    0.95,
    0.6,
    0.05
  ],
  "trial_n_signals": [
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1
  ]
}
```

## Usage

```json
{
  "input_tokens": 8312,
  "output_tokens": 3343,
  "reasoning_tokens": 1866,
  "provider": "Claude Platform on AWS"
}
```
