# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Probability Updating domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_1" and "pi_4") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_1" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_4" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects are told a prior probability that hypothesis H is true (e.g. that a ball was drawn from urn A rather than urn B), then shown evidence whose probability under H and under not-H is known (e.g. the colours of several draws, with the urns' compositions stated). They report the probability that H is true on a 0-100% scale.

## THEORY 1 — pi_1
Probability distortion is a systematic bias applied when the judgment is formed: people combine prior and evidence additively in log-odds space, as Bayes prescribes, but with the wrong weights. The posterior log-odds equal alpha times the prior log-odds plus beta times the log likelihood ratio. alpha < 1 is base-rate neglect (the prior is underweighted), beta < 1 is conservatism (the evidence is underweighted) and beta > 1 is overreaction; alpha = beta = 1 recovers Bayes. Because the distortion is a deterministic transformation, the same stimulus always yields the same central judgment; trial-to-trial variability is a separate, constant response noise in log-odds space (sigma), plus a lapse rate epsilon. The theory therefore predicts that mean distortion and response variability are unrelated: changing how noisy responses are does not change their average, and the average is a fixed linear function of prior and evidence log-odds.


`predict(parameters, state, history) -> np.ndarray`:
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


`policy(probs) -> int`:
def policy(probabilities):
    probabilities = probabilities / probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))


## THEORY 2 — pi_4
Stochastic partial integration with a soft-bounded response range.

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

`predict(parameters, state, history) -> np.ndarray`:
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


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    p = np.asarray(probs, dtype=np.float64)
    p = np.clip(p, 0.0, None)
    p /= p.sum()
    return int(np.random.choice(len(p), p=p))


## EXPERIMENT 1 (proposed by pi_1)

### DESIGN
**Unique trials (n=16):**
  trial 1: prior=0.500  P(e|H)=0.5  P(e|not-H)=0.5  n_signals=1  log LR=+0.00
  trial 2: prior=0.500  P(e|H)=0.8  P(e|not-H)=0.2  n_signals=1  log LR=+1.39
  trial 3: prior=0.500  P(e|H)=0.2  P(e|not-H)=0.8  n_signals=1  log LR=-1.39
  trial 4: prior=0.500  P(e|H)=0.95  P(e|not-H)=0.05  n_signals=1  log LR=+2.94
  trial 5: prior=0.800  P(e|H)=0.5  P(e|not-H)=0.5  n_signals=1  log LR=+0.00
  trial 6: prior=0.950  P(e|H)=0.5  P(e|not-H)=0.5  n_signals=1  log LR=+0.00
  trial 7: prior=0.800  P(e|H)=0.8  P(e|not-H)=0.2  n_signals=1  log LR=+1.39
  trial 8: prior=0.950  P(e|H)=0.95  P(e|not-H)=0.05  n_signals=1  log LR=+2.94
  trial 9: prior=0.050  P(e|H)=0.05  P(e|not-H)=0.95  n_signals=1  log LR=-2.94
  trial 10: prior=0.950  P(e|H)=0.05  P(e|not-H)=0.95  n_signals=1  log LR=-2.94
  trial 11: prior=0.050  P(e|H)=0.95  P(e|not-H)=0.05  n_signals=1  log LR=+2.94
  trial 12: prior=0.800  P(e|H)=0.2  P(e|not-H)=0.8  n_signals=1  log LR=-1.39
  trial 13: prior=0.950  P(e|H)=0.2  P(e|not-H)=0.8  n_signals=1  log LR=-1.39
  trial 14: prior=0.800  P(e|H)=0.05  P(e|not-H)=0.95  n_signals=1  log LR=-2.94
  trial 15: prior=0.950  P(e|H)=0.4  P(e|not-H)=0.6  n_signals=1  log LR=-0.41
  trial 16: prior=0.600  P(e|H)=0.95  P(e|not-H)=0.05  n_signals=1  log LR=+2.94

**Rationale:** Goal: test whether two-cue trials are integrated deterministically on every trial, as the advocated fixed-weight log-odds model with constant noise claims, or through a stochastic mixture of strategies. All trials use one draw (n=1), so LLR = log(p_h/p_nh). The design mixes single-source anchor cells with two-cue cells. Unlike earlier designs, it tests mixture structure and additivity rather than variance-versus-level laws.

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

**Schedule:** 16 unique x 3 reps = 48 trials per subject.


### METRIC
Rationale:
Implausible-zone rate in cells where every component of both theories points clearly one way. I use single-source and congruent cells (plus the 0.6-prior/strong-evidence probe, whose competing prior-only mode is still on the same side). For each cell I count reports in a zone that no integration route, prior-only route or evidence-only route can reach without a lapse:
- the wrong side of 50% for strong one-directional cells;
- the opposite extreme for the weak-evidence cells;
- the far tails (<=10 or >=90) for the 50/50 baseline.

Conflict cells and the weak-conflict probes (0.95 with -1.39 or -0.41) are excluded. There the competing theory's evidence-only mode legitimately lands on the far side, and including them would cancel the contrast.

In these zones, responses come almost only from uniform lapses, plus a small Gaussian-tail contribution from very noisy or low-beta subjects. The two theories place outlying responses differently:
- The advocated fixed-weight model has no strategy mixture. Every off-pattern human response must be absorbed by a uniform lapse (epsilon 0.05-0.41) and by a wide sigma (up to 1.2) and low beta (down to 0.12). These produce wrong-side reports even in cells where both cues agree or only one is informative.
- The competing theory explains off-pattern reports mostly through cue-selection modes confined to two-cue conflict cells. It has small lapses (epsilon 0-0.15), sigma at most 1 and beta at least 0.3, so these cells stay clean.

Predicted values: about 0.10-0.13 under the advocated theory versus about 0.03-0.05 under the competing theory. With about 30 eligible trials per subject, the per-subject rate is reasonably tight. The count-based zones use 10%-or-wider margins, so the shared rounding stage barely affects them.

On human data, a clear excess of wrong-side reports in cells with no cue conflict would support a global lapse/noise account over a conflict-specific mixture.

Source:
import numpy as np
import pandas as pd

def metric(data: pd.DataFrame) -> float:
    p = data['prior'].values.astype(float)
    l = data['log_lr'].values.astype(float)
    r = data['response_bin'].values.astype(float)
    L4 = np.log(4.0)
    L19 = np.log(19.0)
    def cell(pv, lv):
        return (np.abs(p - pv) < 1e-3) & (np.abs(l - lv) < 0.05)
    specs = [
        (0.5, 0.0, (r <= 10) | (r >= 90)),
        (0.5, L4, r <= 20),
        (0.5, -L4, r >= 80),
        (0.5, L19, r < 50),
        (0.8, 0.0, r < 50),
        (0.95, 0.0, r < 50),
        (0.8, L4, r < 50),
        (0.95, L19, r < 50),
        (0.05, -L19, r > 50),
        (0.6, L19, r < 50),
    ]
    hits = 0.0
    total = 0.0
    for pv, lv, zone in specs:
        m = cell(pv, lv)
        total += float(m.sum())
        hits += float((m & zone).sum())
    if total <= 0:
        return float('nan')
    return float(hits / total)


### RESULTS
- Predicted under pi_1 (simulated): 0.1220 (var=0.0061)
- Predicted under pi_4 (simulated): 0.0447 (var=0.0015)
- Observed on real data: 0.1240 (var=0.0029)

## EXPERIMENT 2 (proposed by pi_4)

### DESIGN
**Unique trials (n=12):**
  trial 1: prior=0.500  P(e|H)=0.8  P(e|not-H)=0.2  n_signals=1  log LR=+1.39
  trial 2: prior=0.500  P(e|H)=0.8  P(e|not-H)=0.2  n_signals=2  log LR=+1.39
  trial 3: prior=0.500  P(e|H)=0.8  P(e|not-H)=0.2  n_signals=4  log LR=+1.39
  trial 4: prior=0.500  P(e|H)=0.8  P(e|not-H)=0.2  n_signals=6  log LR=+1.39
  trial 5: prior=0.500  P(e|H)=0.2  P(e|not-H)=0.8  n_signals=6  log LR=-1.39
  trial 6: prior=0.900  P(e|H)=0.5  P(e|not-H)=0.5  n_signals=1  log LR=+0.00
  trial 7: prior=0.999  P(e|H)=0.5  P(e|not-H)=0.5  n_signals=1  log LR=+0.00
  trial 8: prior=0.800  P(e|H)=0.8  P(e|not-H)=0.2  n_signals=1  log LR=+1.39
  trial 9: prior=0.800  P(e|H)=0.8  P(e|not-H)=0.2  n_signals=3  log LR=+1.39
  trial 10: prior=0.800  P(e|H)=0.8  P(e|not-H)=0.2  n_signals=6  log LR=+1.39
  trial 11: prior=0.999  P(e|H)=0.2  P(e|not-H)=0.8  n_signals=3  log LR=-1.39
  trial 12: prior=0.200  P(e|H)=0.8  P(e|not-H)=0.2  n_signals=4  log LR=+1.39

**Rationale:** Goal: separate two accounts. The advocated account is stochastic partial integration with a saturating response bound. The competing account is fixed-weight linear log-odds integration with constant Gaussian noise. Earlier designs [0-4] used single draws with |LLR| no larger than 4.6 and |L0| no larger than 4.6. This design uses multi-draw evidence, where I assume LLR = n*log(ph/pnh), and a 0.999 prior to push inputs beyond the soft bound Delta (3.5-7). It also tracks how the strategy mixture's modes separate as evidence grows.

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

**Schedule:** 12 unique x 5 reps = 60 trials per subject.


### METRIC
Rationale:
Several design features cannot separate the theories. Both predict() functions use only log_lr, which is +1.39 regardless of n_signals, so the n-sweeps are effectively replicates. The sharpest remaining difference is lapse behaviour.

The metric uses only cells where every component of both theories lies firmly on the H side:
- the prior-only cells at priors .9 and .999 (log-odds about 1.6-7);
- the congruent cells at prior .8 with LLR +1.39, where the integrated, prior-anchored and evidence-only modes are all positive.

This gives 25 trials per subject. In these cells a report at or below 40% is essentially impossible under each model's main components. It arises almost only from the uniform lapse, which places 41/101 of its mass in that region.

Expected values differ by lapse range:
- Competing theory: epsilon spans 0.05-0.41 (mean about 0.23), so the opposite-side fraction is about 0.41*eps, roughly 0.09-0.10.
- Advocated theory: epsilon spans 0-0.15 (mean about 0.075), so the fraction is about 0.03. It also has saturation: the bound only pulls high responses toward the high 90s and never toward the opposite side.

The low-beta prior-0.5 cells and the conflict cells are excluded. There, a near-50% center (competing) or a representativeness mode (advocated) could push mass below 40% for reasons other than lapse.

The metric is a simple proportion, so it is defined per subject. Between-subject variance comes mostly from binomial noise over 25 trials plus epsilon spread. That noise is small relative to the roughly 0.06 gap, which is about t≈3.5-4 at N=25.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    d = data
    mask = (d['prior'] >= 0.79) & (d['log_lr'] >= -1e-9)
    sub = d.loc[mask]
    if len(sub) == 0:
        return float('nan')
    return float(np.mean(sub['response_bin'].values <= 40))


### RESULTS
- Predicted under pi_1 (simulated): 0.1016 (var=0.0049)
- Predicted under pi_4 (simulated): 0.0408 (var=0.0022)
- Observed on real data: 0.1200 (var=0.0049)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
**Unique trials (n=12):**
  trial 1: prior=0.500  P(e|H)=0.5  P(e|not-H)=0.5  n_signals=1  log LR=+0.00
  trial 2: prior=0.500  P(e|H)=0.8  P(e|not-H)=0.2  n_signals=1  log LR=+1.39
  trial 3: prior=0.500  P(e|H)=0.99  P(e|not-H)=0.01  n_signals=1  log LR=+4.60
  trial 4: prior=0.500  P(e|H)=0.01  P(e|not-H)=0.99  n_signals=1  log LR=-4.60
  trial 5: prior=0.990  P(e|H)=0.5  P(e|not-H)=0.5  n_signals=1  log LR=+0.00
  trial 6: prior=0.010  P(e|H)=0.5  P(e|not-H)=0.5  n_signals=1  log LR=+0.00
  trial 7: prior=0.990  P(e|H)=0.99  P(e|not-H)=0.01  n_signals=1  log LR=+4.60
  trial 8: prior=0.010  P(e|H)=0.01  P(e|not-H)=0.99  n_signals=1  log LR=-4.60
  trial 9: prior=0.950  P(e|H)=0.05  P(e|not-H)=0.95  n_signals=1  log LR=-2.94
  trial 10: prior=0.200  P(e|H)=0.8  P(e|not-H)=0.2  n_signals=1  log LR=+1.39
  trial 11: prior=0.800  P(e|H)=0.95  P(e|not-H)=0.05  n_signals=1  log LR=+2.94
  trial 12: prior=0.050  P(e|H)=0.2  P(e|not-H)=0.8  n_signals=1  log LR=-1.39

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    d = data.copy()
    b = d['bayes'].astype(float).values
    # moderately extreme cells: Bayesian posterior strongly on one side but not ~certain
    # (excludes the near-certain 0.99/0.99 cells where both theories pile up at 100)
    sel = ((b >= 0.95) & (b <= 0.996)) | ((b <= 0.05) & (b >= 0.004))
    d = d[sel]
    if len(d) == 0:
        return 0.5
    rb = d['response_bin'].astype(float).values
    up = d['bayes'].astype(float).values > 0.5
    folded = np.where(up, rb, 100.0 - rb)
    window = folded >= 90
    n_win = int(window.sum())
    if n_win == 0:
        return 0.5
    n_end = int((folded >= 100).sum())
    return float(n_end) / float(n_win)

```

**Observed (real) value:** 0.1400 (var=0.0062)
**Predicted under pi_1:** 0.1675 (var=0.0068)
**Predicted under pi_4:** 0.1480 (var=0.0076)

### Experiment 4
**Design**
**Unique trials (n=12):**
  trial 1: prior=0.500  P(e|H)=0.5  P(e|not-H)=0.5  n_signals=1  log LR=+0.00
  trial 2: prior=0.500  P(e|H)=0.75  P(e|not-H)=0.25  n_signals=1  log LR=+1.10
  trial 3: prior=0.500  P(e|H)=0.9  P(e|not-H)=0.1  n_signals=1  log LR=+2.20
  trial 4: prior=0.500  P(e|H)=0.98  P(e|not-H)=0.02  n_signals=1  log LR=+3.89
  trial 5: prior=0.800  P(e|H)=0.5  P(e|not-H)=0.5  n_signals=1  log LR=+0.00
  trial 6: prior=0.950  P(e|H)=0.5  P(e|not-H)=0.5  n_signals=1  log LR=+0.00
  trial 7: prior=0.800  P(e|H)=0.9  P(e|not-H)=0.1  n_signals=1  log LR=+2.20
  trial 8: prior=0.950  P(e|H)=0.98  P(e|not-H)=0.02  n_signals=1  log LR=+3.89
  trial 9: prior=0.200  P(e|H)=0.1  P(e|not-H)=0.9  n_signals=1  log LR=-2.20
  trial 10: prior=0.050  P(e|H)=0.02  P(e|not-H)=0.98  n_signals=1  log LR=-3.89
  trial 11: prior=0.950  P(e|H)=0.1  P(e|not-H)=0.9  n_signals=1  log LR=-2.20
  trial 12: prior=0.200  P(e|H)=0.9  P(e|not-H)=0.1  n_signals=1  log LR=+2.20

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    d = data.copy()
    d['pk'] = d['prior'].astype(float).round(3)
    d['lk'] = d['log_lr'].astype(float).round(2)
    d['med'] = d.groupby(['subject_id', 'pk', 'lk'])['response'].transform('median')
    d['dev'] = (d['response'].astype(float) - d['med']).abs()
    def mask_for(cells):
        m = np.zeros(len(d), dtype=bool)
        pr = d['prior'].astype(float).values
        ll = d['log_lr'].astype(float).values
        for p, l in cells:
            m |= (np.abs(pr - p) < 0.01) & (np.abs(ll - l) < 0.05)
        return m
    ext_cells = [(0.95, 0.0), (0.95, 3.89), (0.05, -3.89)]
    mid_cells = [(0.5, 0.0), (0.5, 1.10), (0.2, 2.20)]
    ext = d.loc[mask_for(ext_cells), 'dev'].values
    mid = d.loc[mask_for(mid_cells), 'dev'].values
    if len(ext) == 0 or len(mid) == 0:
        return float('nan')
    return float(np.log((np.median(ext) + 0.01) / (np.median(mid) + 0.01)))

```

**Observed (real) value:** -1.5041 (var=0.4026)
**Predicted under pi_1:** -1.6094 (var=0.6514)
**Predicted under pi_4:** -1.5041 (var=0.3621)

### Experiment 5
**Design**
**Unique trials (n=15):**
  trial 1: prior=0.500  P(e|H)=0.5  P(e|not-H)=0.5  n_signals=1  log LR=+0.00
  trial 2: prior=0.500  P(e|H)=0.7  P(e|not-H)=0.3  n_signals=1  log LR=+0.85
  trial 3: prior=0.500  P(e|H)=0.8  P(e|not-H)=0.2  n_signals=1  log LR=+1.39
  trial 4: prior=0.500  P(e|H)=0.9  P(e|not-H)=0.1  n_signals=1  log LR=+2.20
  trial 5: prior=0.500  P(e|H)=0.95  P(e|not-H)=0.05  n_signals=1  log LR=+2.94
  trial 6: prior=0.500  P(e|H)=0.99  P(e|not-H)=0.01  n_signals=1  log LR=+4.60
  trial 7: prior=0.700  P(e|H)=0.5  P(e|not-H)=0.5  n_signals=1  log LR=+0.00
  trial 8: prior=0.800  P(e|H)=0.5  P(e|not-H)=0.5  n_signals=1  log LR=+0.00
  trial 9: prior=0.900  P(e|H)=0.5  P(e|not-H)=0.5  n_signals=1  log LR=+0.00
  trial 10: prior=0.950  P(e|H)=0.5  P(e|not-H)=0.5  n_signals=1  log LR=+0.00
  trial 11: prior=0.990  P(e|H)=0.5  P(e|not-H)=0.5  n_signals=1  log LR=+0.00
  trial 12: prior=0.990  P(e|H)=0.01  P(e|not-H)=0.99  n_signals=1  log LR=-4.60
  trial 13: prior=0.950  P(e|H)=0.05  P(e|not-H)=0.95  n_signals=1  log LR=-2.94
  trial 14: prior=0.900  P(e|H)=0.1  P(e|not-H)=0.9  n_signals=1  log LR=-2.20
  trial 15: prior=0.800  P(e|H)=0.8  P(e|not-H)=0.2  n_signals=1  log LR=+1.39

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    r = np.clip(data['response'].astype(float).values, 0.01, 0.99)
    lo = np.log(r) - np.log1p(-r)
    pr = data['prior'].astype(float).values
    ll = data['log_lr'].astype(float).values
    def cm(mask):
        v = lo[mask]
        if len(v) == 0:
            return np.nan
        return float(np.median(v))
    num = 0.0
    den = 0.0
    for p in [0.7, 0.8, 0.9, 0.95, 0.99]:
        L = np.log(p / (1 - p))
        E = cm((np.abs(pr - 0.5) < 1e-3) & (np.abs(ll - L) < 0.05))
        P = cm((np.abs(pr - p) < 1e-3) & (np.abs(ll) < 0.05))
        C = cm((np.abs(pr - p) < 1e-3) & (np.abs(ll + L) < 0.05))
        if np.isnan(P):
            continue
        if not np.isnan(E):
            num += E
            den += P
        if not np.isnan(C):
            num += (P - C)
            den += P
    if abs(den) < 1e-6:
        return 0.0
    return float(num / den)

```

**Observed (real) value:** 0.6707 (var=0.1719)
**Predicted under pi_1:** 0.5297 (var=0.1446)
**Predicted under pi_4:** 0.7687 (var=0.0764)

### Experiment 6
**Design**
**Unique trials (n=12):**
  trial 1: prior=0.500  P(e|H)=0.6  P(e|not-H)=0.4  n_signals=1  log LR=+0.41
  trial 2: prior=0.500  P(e|H)=0.75  P(e|not-H)=0.25  n_signals=1  log LR=+1.10
  trial 3: prior=0.500  P(e|H)=0.9  P(e|not-H)=0.1  n_signals=1  log LR=+2.20
  trial 4: prior=0.500  P(e|H)=0.98  P(e|not-H)=0.02  n_signals=1  log LR=+3.89
  trial 5: prior=0.600  P(e|H)=0.5  P(e|not-H)=0.5  n_signals=1  log LR=+0.00
  trial 6: prior=0.750  P(e|H)=0.5  P(e|not-H)=0.5  n_signals=1  log LR=+0.00
  trial 7: prior=0.900  P(e|H)=0.5  P(e|not-H)=0.5  n_signals=1  log LR=+0.00
  trial 8: prior=0.980  P(e|H)=0.5  P(e|not-H)=0.5  n_signals=1  log LR=+0.00
  trial 9: prior=0.750  P(e|H)=0.75  P(e|not-H)=0.25  n_signals=1  log LR=+1.10
  trial 10: prior=0.875  P(e|H)=0.875  P(e|not-H)=0.125  n_signals=1  log LR=+1.95
  trial 11: prior=0.125  P(e|H)=0.125  P(e|not-H)=0.875  n_signals=1  log LR=-1.95
  trial 12: prior=0.500  P(e|H)=0.5  P(e|not-H)=0.5  n_signals=1  log LR=+0.00

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    d = data.copy()
    r = np.clip(d['response'].astype(float).values, 0.005, 0.995)
    d['lg'] = np.log(r) - np.log1p(-r)
    d['pk'] = np.round(d['prior'].astype(float).values, 3)
    d['lk'] = np.round(d['log_lr'].astype(float).values, 2)
    d['dev'] = d['lg'] - d.groupby(['subject_id', 'pk', 'lk'])['lg'].transform('median')
    pr = d['prior'].astype(float).values
    ll = d['log_lr'].astype(float).values
    def sel(p, l):
        return (np.abs(pr - p) < 0.01) & (np.abs(ll - l) < 0.05)
    low = sel(0.5, 0.405) | sel(0.6, 0.0) | sel(0.5, 0.0)
    high = sel(0.5, 2.197) | sel(0.9, 0.0) | sel(0.75, 1.099) | sel(0.875, 1.946) | sel(0.125, -1.946)
    dv = np.abs(d['dev'].values)
    if low.sum() == 0 or high.sum() == 0:
        return float('nan')
    mad_low = float(np.median(dv[low])) + 0.05
    mad_high = float(np.median(dv[high])) + 0.05
    return float(np.log(mad_high / mad_low))

```

**Observed (real) value:** 0.1515 (var=0.1960)
**Predicted under pi_1:** 0.0709 (var=0.2473)
**Predicted under pi_4:** 0.1254 (var=0.2522)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across six experiments the two theories split along a clear line. pi_4 is more accurate on the structural metrics.\n- Exp 3 (exact-100 share among folded reports \u226590): observed .140; pi_4 .148, pi_1 .168.\n- Exp 4 (raw-scale dispersion, extreme vs mid cells): observed \u22121.504; pi_4 \u22121.504 exactly, pi_1 \u22121.609.\n- Exp 5 (additivity/weight ratio): observed .671; pi_4 .769 (off by .10), pi_1 .530 (off by .14).\n- Exp 6 (logit-MAD high vs low cells): observed .152; pi_4 .125, pi_1 .071.\nSo a soft-bounded, near-linear log-odds integration with constant logit noise describes the shape of central tendency and dispersion somewhat better than the unbounded fixed-weight model.\n\npi_4 fails decisively on the two newest experiments, which both target off-pattern responding in cells where every route points one way.\n- Exp 1: observed .124; pi_1 .122, pi_4 .045.\n- Exp 2: observed .120; pi_1 .102, pi_4 .041.\nThe Exp 1 metric counts implausible-zone reports; the Exp 2 metric counts reports \u226440% when the prior is \u2265.8 and the evidence is non-negative. Humans produce roughly 12% grossly off-pattern reports even with no cue conflict, which requires an effective lapse of about 0.3. pi_4's conflict-specific strategy mixture with small epsilon cannot produce this. pi_1 captures it only via a large uniform epsilon.\n\nThe Exp 1 metric also includes baseline-cell tails (\u226410 or \u226590 at 50/50 evidence). This favors a genuinely diffuse component rather than only directional reversals. pi_1 additionally gets these numbers right somewhat 'for the wrong reasons', since it also uses low beta and wide sigma.\n\nNeither theory is task-invariant. pi_4 has the better integration and output structure but the wrong account of non-integrative responding. pi_1 has the right lapse magnitude but no saturation and a worse dispersion pattern. pi_4's key distinctive claim \u2014 that off-pattern mass is confined to two-cue cells \u2014 is directly contradicted by the data. Yet pi_4 remains the better structural base. pi_1 is the weaker overall competitor and should be replaced by a theory that inherits what each gets right.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Replace pi_1 with a theory of 'bounded deterministic integration plus heterogeneous off-task responding'. It should be a strong competitor to pi_4 that isolates the question of whether a cue-selection mixture is needed at all.\n\nCore claims:\n(1) Every trial integrates deterministically: L = alpha*logit(prior) + beta*LLR. There is no rho/q strategy mixture; conflict cells are unimodal in the engaged state.\n(2) The output passes through the same soft saturating bound as pi_4, with Delta \u2248 3.5\u20137 and kappa \u2265 1. This retains pi_4's success on Exps 3\u20136: few exact-100s, compressed raw-scale spread at extremes, roughly constant or slightly rising logit MAD.\n(3) Constant Gaussian logit noise sigma (\u22480.3\u20131.0) around the bounded value.\n(4) A substantial, subject-heterogeneous non-engaged component with total mass lambda drawn over roughly 0.05\u20130.45 (mean \u22480.25). It splits into:\n   (a) uniform guessing over 0\u2013100, needed for baseline-cell tails;\n   (b) a polarity-reversal lapse that reports the mirror value 100 \u2212 R, i.e. P(not-H), which sends mass to the wrong side in clean cells;\n   (c) optionally, a 50% 'don't know' spike.\n   These components should apply equally to single-source, congruent and conflict cells.\n\nKey contrasts with pi_4:\n- Off-pattern mass is the same in single-cue and two-cue cells, versus pi_4's conflict-specific excess.\n- Wrong-side reports in clean cells sit near the mirror of the correct answer (reversal) rather than spreading uniformly.\n\nContrasts with old pi_1:\n- Saturation at the extremes.\n- A structured lapse, so beta and sigma need not be distorted to soak up outliers. Keep beta \u2248 0.4\u20131.0.\n\nSuggested next discriminating experiment: matched clean single-source cells and strong conflict cells. Compare the excess off-side mass in conflict cells over clean cells (\u22480 under the new theory, rho*(1\u2212q) under pi_4). Also test whether wrong-side reports in clean cells cluster at 100 \u2212 (expected report) or are flat.\n\nNote for later: if pi_4 survives, its epsilon range must be widened to ~0\u20130.45 to fit Exps 1\u20132."
}
```

## Usage

```json
{
  "input_tokens": 15931,
  "output_tokens": 2991,
  "reasoning_tokens": 1257,
  "provider": "Claude Platform on AWS"
}
```
