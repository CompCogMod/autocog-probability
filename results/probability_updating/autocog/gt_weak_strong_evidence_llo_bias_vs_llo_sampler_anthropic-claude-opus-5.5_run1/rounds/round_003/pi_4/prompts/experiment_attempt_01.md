# experiment_attempt_01

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


## COMPETING THEORY
**Description:** Bounded deterministic integration plus heterogeneous off-task responding.

(1) Engaged integration is deterministic. Every engaged trial combines prior and evidence additively in log-odds with distorted weights: L = alpha*logit(prior) + beta*LLR. alpha<1 is base-rate neglect and beta<1 is conservatism. There is no cue-selection mixture, so conflict cells are unimodal in the engaged state.

(2) The output is softly bounded. The integrated value passes through L' = sign(L)*Delta*tanh((|L|/Delta)^kappa)^(1/kappa), with kappa>1 and Delta about 3.5–5.5. The bound keeps engaged reports off the exact endpoints, compresses raw-scale spread near the extremes, and leaves logit-scale variability roughly constant.

(3) Response noise is constant Gaussian noise sigma in log-odds around the bounded value.

(4) A non-engaged component has subject-specific total probability lambda (about 0.1–0.42). Its probability does not depend on the stimulus. It is dominated by uniform guessing over 0–100. It has two minor parts:
- a polarity-reversal lapse that reports the mirror value 100−R, i.e. P(not-H);
- a rare 50% 'don't know' report.
Off-pattern mass is therefore equal in single-source, congruent and conflict cells.

**Parameters:**
- alpha: [0.7, 1.1]
- beta: [0.35, 1.0]
- Delta: [3.5, 5.5]
- kappa: [1.5, 4.0]
- sigma: [0.3, 0.9]
- lam: [0.1, 0.42]
- f_rev: [0.0, 0.15]
- f_half: [0.0, 0.05]

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
    Delta = max(float(parameters["Delta"]), 1e-3)
    kappa = max(float(parameters["kappa"]), 1e-3)
    sigma = max(float(parameters["sigma"]), 1e-3)
    lam = float(np.clip(parameters["lam"], 0.0, 1.0))
    f_rev = float(np.clip(parameters["f_rev"], 0.0, 1.0))
    f_half = float(np.clip(parameters["f_half"], 0.0, 1.0))
    tot = f_rev + f_half
    if tot > 1.0:
        f_rev /= tot
        f_half /= tot
    f_uni = max(0.0, 1.0 - f_rev - f_half)

    L = alpha * L0 + beta * llr
    u = np.abs(L) / Delta
    Lb = np.sign(L) * Delta * np.tanh(u ** kappa) ** (1.0 / kappa)

    edges = (np.arange(K + 1) - 0.5) / (K - 1)
    z_edges = np.empty(K + 1)
    z_edges[0] = -np.inf
    z_edges[-1] = np.inf
    inner = edges[1:-1]
    z_edges[1:-1] = np.log(inner) - np.log1p(-inner)

    cdf = ndtr((z_edges[None, :] - Lb[:, None]) / sigma)
    eng = np.clip(np.diff(cdf, axis=1), 0.0, None)
    eng = eng / np.maximum(eng.sum(axis=1, keepdims=True), 1e-300)

    rev = eng[:, ::-1]
    spike = np.zeros(K)
    spike[50] = 1.0
    uni = np.full(K, 1.0 / K)

    probs = (1.0 - lam) * eng + lam * (f_uni * uni[None, :] + f_rev * rev + f_half * spike[None, :])
    probs = np.clip(probs, 0.0, None)
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

[4] Goal: test whether two-cue trials are integrated deterministically on every trial, as the advocated fixed-weight log-odds model with constant noise claims, or through a stochastic mixture of strategies. All trials use one draw (n=1), so LLR = log(p_h/p_nh). The design mixes single-source anchor cells with two-cue cells. Unlike earlier designs, it tests mixture structure and additivity rather than variance-versus-level laws.

(A) Single-source anchors. These cells calibrate alpha, beta and sigma without any mixture contribution.
- Baseline: 0.5 with 0.5/0.5.
- Evidence only at prior 0.5: LLR +/-1.39 and +2.94.
- Prior only: 0.8 and 0.95.

(B) Congruent cells: 0.8 with 0.8/0.2, 0.95 with 0.95/0.05, and 0.05 with 0.05/0.95.
- Advocated: the median logit equals exactly the single-source prior logit plus the single-source evidence logit (strict additivity). Spread equals the baseline sigma.
- Competing: medians are sub-additive. There is also a heavy tail or second mode at the single-cue values (about 80% or about 95%). The 0.95 congruent cell is further pulled down by the soft bound.

(C) Conflict cells: 0.95 with 0.05/0.95, 0.05 with 0.95/0.05, 0.8 with 0.2/0.8, 0.95 with 0.2/0.8, and 0.8 with 0.05/0.95.
- Advocated: one unimodal logit-normal distribution centred at (alpha*L0 - beta*|LLR|), with the same IQR as median-matched single-source cells. The proportion of reports on the far side of 50% is predictable from sigma alone.
- Competing: responses mix a prior-anchored mode (near the prior-only cell's median) and an evidence-only mode (near the evidence-only cell's median, on the opposite side of 50%). This yields inflated IQR, bimodality, and an excess of opposite-side reports.

(D) Near-single-cue probes: 0.95 with weak evidence 0.4/0.6, and 0.6 with strong evidence 0.95/0.05. These are the sharpest dissociation because both cues count as informative, which triggers the competing mixture.
- Advocated: each distribution is a slightly shifted copy of its single-source partner (prior 0.95 alone; LLR 2.94 at prior 0.5), with identical spread.
- Competing: an extra mode appears far from the partner. In the first probe, rho*(1-q) of responses fall near 40-45% (the evidence-only answer). In the second, rho*q of responses fall near 60% (the prior-only answer).

Metrics, using 10%-wide bins so they are robust to rounding:
1. IQR in each two-cue cell divided by the IQR of its single-source partner. Advocated predicts about 1; competing predicts more than 1.
2. The fraction of two-cue responses falling within ±10% of the single-cue predictions taken from the anchor cells, compared against the logit-normal expectation.
3. Additivity residuals in the congruent cells, computed from the empirical single-source medians. Advocated predicts 0; competing predicts negative residuals.
4. Within-subject consistency: the advocated model says a subject's spread is the same in single-source and two-cue cells.

Fifteen unique trials give K=4 repeats each.

[5] Goal: separate two accounts. The advocated account is stochastic partial integration with a saturating response bound. The competing account is fixed-weight linear log-odds integration with constant Gaussian noise. Earlier designs [0-4] used single draws with |LLR| no larger than 4.6 and |L0| no larger than 4.6. This design uses multi-draw evidence, where I assume LLR = n*log(ph/pnh), and a 0.999 prior to push inputs beyond the soft bound Delta (3.5-7). It also tracks how the strategy mixture's modes separate as evidence grows.

Block A: saturation of single-source cells.
- Evidence-only sweep at prior 0.5 with draws of 0.8/0.2, n = 1, 2, 4 and 6 (LLR 1.39, 2.77, 5.55, 8.32).
- A mirrored low cell: 0.2/0.8 with n = 6.
- Prior-only cells at 0.9 and 0.999 (L0 2.2 and 6.9).
- Linear theory: median logit = beta*LLR (or alpha*L0), with no ceiling. The share of top-end reports keeps rising steeply from n=4 to n=6. For example, with beta=.6 and sigma=.6, P(report>99.5%) goes from about 0 to about 30%.
- Advocated theory: nearly linear up to about Delta, then a plateau. Medians at n=4, n=6 and prior 0.999 converge on a common ceiling near expit(Delta), and the top-tail share grows little.
- Rounding-robust metrics: the share of reports at or above 97.5%, and median rank, compared across n=4 vs n=6 and prior .9 vs .999. Under linear theory the n=1 to n=2 slope predicts the n=4 to n=6 slope; under the advocated theory the upper slope collapses.

Block B: mode separation at a fixed prior of 0.8 with congruent evidence of increasing strength (0.8/0.2, n = 1, 3, 6).
- Linear theory: one logit-normal distribution that shifts upward and tightens toward 100.
- Advocated theory: the integrated mode saturates near the ceiling. A prior-anchored mode stays fixed at expit(alpha*1.39), about 75-80%, carrying a constant mass of rho*q. A third component, the evidence-only mode at beta*LLR, has no fixed location but is absorbed into the ceiling.
- Predicted signature: a lower tail that becomes more separated, and more conspicuous, as evidence strengthens, with constant mass. Its fraction is unchanged with n, unlike constant-sigma noise, which predicts this tail mass vanishes with n.
- Metric: the fraction of reports at or below 85% in each cell.

Block C: extreme conflict cells with multi-draw evidence.
- Cells: 0.999 vs 0.2/0.8 with n=3; 0.2 vs 0.8/0.2 with n=4; 0.01 vs 0.8/0.2 with n=2.
- Linear theory: a single unimodal hump at alpha*L0 + beta*LLR. For the 0.999 cell this is about 97%. The mass beyond 50% on the opposite side is predictable from sigma fitted in Block A.
- Advocated theory: bimodal. For the 0.999 cell, one mode is pinned at the bound (about 99%) and another is near expit(-beta*4.16), about 5-20%. The 0.01 cell has a mode near 80-90%, and the 0.2 cell has a mode near 20%.
- Metrics: the opposite-side fraction, a bimodality coefficient, and IQR relative to single-source cells with matched medians. All are computed with 10%-wide bins.

Summary: 12 unique trials, giving K=5 repeats each. Both theories' noise and weight parameters are pinned by Blocks A and B. The critical out-of-sample tests are the ceiling plateau, a lower-tail mass that does not shrink with evidence, and bimodality in extreme conflicts.

[6] Goal: test where off-pattern responses come from and where they land. The advocated theory says one stimulus-independent off-task component of mass lambda, mostly uniform, plus a polarity-reversal mirror (100-R) and a small 50% spike. It should appear with equal mass in every cell type, and its non-uniform parts sit at the mirror of the engaged answer. The competing theory says off-pattern mass beyond a small uniform epsilon arises only in two-cue cells. There it is a strategy mixture located at the single-cue answers: prior-only alpha*L0 and evidence-only beta*LLR. Earlier designs studied variance laws, gain linearity, additivity, saturation and conflict bimodality. They did not separate mirror-located from single-cue-located extra mass, nor test equal off-pattern mass across cell types. LLR is assumed to be n*log(ph/pnh).

Block A, single-source cells:
- Baseline 0.5 with 0.5/0.5.
- Strong evidence at prior 0.5: 0.9/0.1 and 0.1/0.9, n=2, LLR +/-4.39. Engaged answer is about 95% or 5%.
- Prior-only cells: 0.95 and 0.05.
- Anchors: prior-only 0.6, and weak evidence-only 0.6/0.4 at prior 0.5. These place the single-cue modes empirically.
- Competing prediction: single-source cells contain only logit-normal mass plus uniform epsilon of at most 0.15. There is no excess in the mirror band.
- Advocated prediction: the mirror band (e.g. 0-15% when the engaged answer is about 95%) holds lambda*f_rev above the uniform floor. The total off-pattern mass can reach 0.3-0.4.

Block B, two-cue cells where the single-cue modes sit far from both the integrated answer and its mirror:
- 0.6 with 0.9/0.1, n=2 (integrated about 97%, prior-only mode about 59%, mirror about 3%).
- 0.4 with 0.9/0.1, n=2 (integrated about 94%, prior-only mode about 41%).
- 0.6 with 0.1/0.9, n=2 (integrated about 6%, prior-only mode about 59%, mirror about 94%).
- 0.95 with weak evidence 0.6/0.4 (integrated about 95%, evidence-only mode about 57%).
- 0.75 with 0.75/0.25 (integrated about 85%, single-cue modes about 68-73%, mirror about 15%).
- Competing prediction: a bump of mass rho*q or rho*(1-q) in the 35-65% band, absent from the matched single-source cells.
- Advocated prediction: no bump, with off-pattern mass identical to the single-source cells.

Rounding-robust metrics (10%-wide bins):
1. Mass in the mirror band minus the uniform-floor estimate, in single-source versus two-cue cells.
2. Mass in the single-cue band (located from the anchor cells) in two-cue cells, relative to the same band in matched single-source cells.
3. Total off-pattern mass (more than about 2 engaged-sigma from the median) by cell type. Advocated predicts equality; competing predicts two-cue greater than single-source.
4. Per-subject correlation of off-pattern mass across cell types. Advocated predicts a high, near-identity relation, because lambda is a subject trait.

There are 12 unique trials with K=5 repeats each.

[7] Goal: a 'fixed anchor vs moving mirror' test of where extra, off-pattern response mass lands in two-cue cells. LLR is assumed to be n*log(ph/pnh).

What each theory predicts:
- Advocated: on a fraction rho of two-cue trials the subject answers from one cue. This puts a secondary mode at the prior-only answer expit(alpha*L0), with mass rho*q, and another at the evidence-only answer expit(beta*LLR), with mass rho*(1-q). These locations depend only on the single cue. They stay fixed when the other cue is varied, and they are absent in single-source cells.
- Competing: extra mass is stimulus-independent (lambda) and mostly uniform. Its only structured parts are a polarity-reversal mode at 100-R, the mirror of the engaged answer, which moves as the integrated answer moves, and a small 50% spike. Its mass is equal in single- and two-cue cells.

Design:
- Anchors, single-source cells:
  - 0.5 baseline;
  - prior-only 0.3 and 0.7, which empirically locate the prior-anchored modes near 30% and 70%;
  - evidence-only 0.8/0.2 with n=2 and n=4, which locate the evidence-only modes near 80-85% and 90-96%.
- Fixed-prior sweep:
  - Prior 0.3 with congruent-for-H evidence 0.8/0.2, n = 2, 3, 4.
  - The integrated answer moves from about 70% to about 93%, so the competing mirror moves from about 30% down to about 7%.
  - The advocated prior-only mode stays pinned near 30%. As n grows, it separates from both the integrated answer and the mirror.
  - Mirrored version: prior 0.7 with 0.2/0.8, n = 2 and 4. The prior-anchored mode stays near 70%, while the mirror moves toward 90-95%.
- Fixed-evidence sweep:
  - Evidence 0.8/0.2 with n=2, at priors 0.1, 0.3, 0.5 and 0.9.
  - The integrated answer ranges from about 45% to about 97%.
  - The advocated evidence-only mode stays fixed near 80-85%. In the 0.1 conflict cell it sits on the opposite side of 50% from the integrated answer. In the 0.9 congruent cell it sits below the integrated answer, giving a lower tail. Its mass should be the same across priors.
  - The competing theory predicts mirror mass at about 55% and about 3% for those two cells, and no fixed 80-85% mode.

Metrics (10%-wide bins, robust to rounding):
1. Mass in the band of +/-10 points around the anchor-cell median, in each two-cue cell, minus the same band's mass in the baseline or matched single-source cells. Advocated: positive and constant across the sweep. Competing: about 0, apart from the uniform floor.
2. Mass in the band around 100 minus the two-cue cell's median. Competing: excess of about lambda*f_rev that tracks the integrated answer. Advocated: only the uniform floor.
3. Total mass more than 2 engaged-sigma from the median. Advocated: two-cue greater than single-source. Competing: equal.
4. Regression of secondary-mode location on the varied cue. Advocated: slope 0. Competing: slope -1 in mirror coordinates.

12 unique trials, K=5 repeats each.

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
  "rationale": "Conflict-magnitude dose-response at near-matched integrated belief. All trials use one draw (n=1), so LLR = log(ph/pnh) under any reading of n.\n\nCells:\n- Baseline: 0.5 with 0.5/0.5.\n- Single-source anchors: prior-only 0.9 and 0.97; evidence-only LLR +2.20 and -3.48 at prior 0.5.\n- Symmetric conflict sweep: the prior and opposing evidence have equal magnitude x = 0.41, 1.10, 2.20, 3.48. That is 0.6 vs 0.4/0.6, 0.75 vs 0.25/0.75, 0.9 vs 0.1/0.9, 0.97 vs 0.03/0.97, plus a mirrored 0.03 vs 0.97/0.03.\n- Congruent cells: 0.75 with 0.75/0.25, and 0.9 with 0.9/0.1.\n\nAdvocated (stochastic partial integration):\n- In conflict cells, a fraction rho of trials answers from one cue. This gives modes at expit(alpha*x) and expit(-beta*x), on opposite sides of 50%, around a central integrated mode at (alpha-beta)*x.\n- Mode separation grows with x. So IQR, bimodality and the fraction of reports on the far side of 50% (relative to the median) should rise steeply across the sweep. Example: x=3.48 gives modes near 95% and 10-20%.\n- Single-source cells, including the extreme ones, show only logit-normal spread plus a small uniform floor, with no mirror-located mass.\n- Congruent cells show sub-additive medians and a lower tail at the single-cue values.\n\nCompeting (deterministic integration plus stimulus-independent lambda):\n- Conflict cells are unimodal at (alpha-beta)*x with constant logit spread.\n- Off-pattern mass is the same in every cell. It is uniform plus a small mirror mode at 100 minus the median, which is mild because lambda*f_rev is at most about 0.06.\n- Off-pattern mass therefore does not grow with x.\n- The extreme single-source cells (prior 0.97, LLR -3.48) should carry the same excess mirror mass (near 3-10%) as the conflict cells.\n\nRounding-robust metrics (10%-wide bins):\n1. The slope of conflict-cell IQR, and of opposite-side mass, against x. Advocated predicts a positive, accelerating slope; competing predicts flat.\n2. Mass within 10 points of the single-cue anchor medians in conflict cells, minus that band's mass in the baseline. Advocated predicts it grows with x; competing predicts about zero.\n3. Off-pattern mass in extreme single-source cells versus x=0.41 conflict. Competing predicts equal; advocated predicts both are small.\n4. Congruent additivity residuals computed from the anchor medians.\n\n12 unique trials, K=5.",
  "trial_priors": [
    0.5,
    0.9,
    0.97,
    0.5,
    0.5,
    0.6,
    0.75,
    0.9,
    0.97,
    0.03,
    0.75,
    0.9
  ],
  "trial_p_signal_h": [
    0.5,
    0.5,
    0.5,
    0.9,
    0.03,
    0.4,
    0.25,
    0.1,
    0.03,
    0.97,
    0.75,
    0.9
  ],
  "trial_p_signal_not_h": [
    0.5,
    0.5,
    0.5,
    0.1,
    0.97,
    0.6,
    0.75,
    0.9,
    0.97,
    0.03,
    0.25,
    0.1
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
    1
  ]
}
```

## Usage

```json
{
  "input_tokens": 13929,
  "output_tokens": 2654,
  "reasoning_tokens": 1411,
  "provider": "Claude Platform on AWS"
}
```
