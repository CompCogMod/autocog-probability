# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Probability Updating domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_1" and "pi_3") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_1" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_3" (THEORY 2 below). Justify your choice.


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


## THEORY 2 — pi_3
Noisy log-odds encoding with Weber-scaled noise and Bayesian shrinkage. Distortion in probability updating comes from noise, not from fixed mis-weighting.

Encoding:
- The stated prior is converted to log-odds, L0 = logit(prior). The evidence is represented by its log likelihood ratio, LLR.
- Each quantity x_i is encoded as a Gaussian sample m_i ~ N(x_i, sd_i^2).
- The noise sd has a constant part plus a part proportional to magnitude (scalar, Weber-like variability): sd_i = s0 + w*|x_i|, with w strictly positive.

Decoding:
- The subject does not know the true magnitudes, so it decodes each measurement with Bayesian shrinkage toward 0.
- The decoder uses a zero-mean Gaussian prior over log-odds, with spread lambda_p for priors and lambda_e for evidence.
- Each decoded value is k_i*m_i, with gain k_i = lambda_i^2 / (lambda_i^2 + sd_i^2).
- The decoded posterior log-odds is the sum k_p*m_p + k_e*m_e.

Report:
- The subject reports expit of the decoded log-odds, after a small extra response noise in log-odds (sigma_r).
- With lapse probability epsilon, the report is random.

Predictions:
- Effective weights alpha_eff = k_p and beta_eff = k_e fall as stimulus magnitude grows. Response log-odds are therefore a concave, saturating function of |prior log-odds| and |LLR|, with more conservatism and base-rate neglect for extreme inputs.
- Single-source extreme cells (prior 0.99 or LLR 4.6 alone) are pulled to roughly 93–97%, not 99–100%. This keeps exact-100 reports rare.
- Congruent cells still add up to very high, tightly clustered beliefs.
- Log-odds variability grows with |L0| and |LLR|, so conflict cells are more variable than the pure 50/50 cell.
- Mean distortion and variability are coupled: subjects or cells with more encoding noise show more conservatism and more base-rate neglect.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    from scipy.special import ndtr
    K = 101
    scalar = np.ndim(stimulus["prior"]) == 0
    prior = np.clip(np.atleast_1d(np.asarray(stimulus["prior"], dtype=float)), 1e-6, 1 - 1e-6)
    llr = np.atleast_1d(np.asarray(stimulus["log_lr"], dtype=float)) * np.ones_like(prior)
    L0 = np.log(prior) - np.log1p(-prior)

    s0 = float(parameters["s0"])
    w = float(parameters["w"])
    lam_p = float(parameters["lambda_p"])
    lam_e = float(parameters["lambda_e"])
    sig_r = float(parameters["sigma_r"])
    eps = float(parameters["epsilon"])

    # Weber-scaled encoding noise in log-odds space
    sd_p = s0 + w * np.abs(L0)
    sd_e = s0 + w * np.abs(llr)
    # Bayesian shrinkage gains (zero-mean Gaussian prior over log-odds)
    k_p = lam_p ** 2 / (lam_p ** 2 + sd_p ** 2)
    k_e = lam_e ** 2 / (lam_e ** 2 + sd_e ** 2)

    mu = k_p * L0 + k_e * llr
    sd = np.sqrt((k_p * sd_p) ** 2 + (k_e * sd_e) ** 2 + sig_r ** 2)
    sd = np.maximum(sd, 1e-3)

    edges = (np.arange(K + 1) - 0.5) / (K - 1)
    z_edges = np.empty(K + 1)
    z_edges[0] = -np.inf
    z_edges[-1] = np.inf
    inner = edges[1:-1]
    z_edges[1:-1] = np.log(inner) - np.log1p(-inner)

    cdf = ndtr((z_edges[None, :] - mu[:, None]) / sd[:, None])
    probs = np.diff(cdf, axis=1)
    probs = np.clip(probs, 0.0, None)
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

**Rationale:** Goal: separate a fixed linear log-odds mis-weighting with constant noise (advocated) from magnitude-dependent Weber encoding noise with Bayesian shrinkage (competing). Earlier designs compared variance laws on the probability scale at matched means. This design targets two different signatures: whether gain falls as inputs grow, and whether noise depends on the source of the belief. All trials use one draw, so LLR = log(p_h/p_nh).

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

**Schedule:** 15 unique x 4 reps = 60 trials per subject.


### METRIC
Rationale:
Estimates the relative weight of evidence vs. prior (beta/alpha) at matched input magnitudes, using median response log-odds. For each magnitude x in {0.85,1.39,2.2,2.94,4.6}, the evidence-only cell (prior .5, LLR x) gives beta*x, the prior-only cell (prior expit(x), uninformative evidence) gives alpha*x, and conflict cells (prior expit(x), LLR -x) give (alpha-beta)*x, so P-C is a second estimate of the evidence contribution. Metric = (sum E + sum(P-C)) / (sum of matching P). Both theories are additive in log-odds, so this ratio cleanly reads off the relative evidence/prior weight. Advocated theory: weights are fixed with strong conservatism relative to prior use (beta ~0.12-0.88 vs alpha ~0.73-1.04), giving a ratio typically ~0.5-0.6 and conflict cells resolving clearly toward H. Competing theory: both gains come from the same shrinkage law at matched magnitude (k = lambda^2/(lambda^2+(s0+w|x|)^2)), with lambda_e and lambda_p both large relative to encoding sd, so effective weights are near-equal (ratio ~0.8-0.95) and conflict cells sit near 50-65%. Using sums of cell medians across 8 cell pairs (pooled over all trials of a cell) is robust to lapses and rounding, and the ratio is a per-subject quantity with moderate between-subject variance, unlike the previous spread-based metric whose noise swamped the effect.

Source:
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


### RESULTS
- Predicted under pi_1 (simulated): 0.5297 (var=0.1446)
- Predicted under pi_3 (simulated): 0.8425 (var=0.0442)
- Observed on real data: 0.6707 (var=0.1719)

## EXPERIMENT 2 (proposed by pi_3)

### DESIGN
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

**Rationale:** Goal: test two signatures of Weber-scaled encoding noise with Bayesian shrinkage that earlier designs did not isolate. Both are framed so the linear model makes alpha/beta-free predictions. All trials use one draw (n=1), so LLR = log(p_h/p_nh).

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

**Schedule:** 12 unique x 5 reps = 60 trials per subject.


### METRIC
Rationale:
Variance coupling test, which is different from the earlier mean-based homogeneity contrast that failed. The linear-weighting theory assumes one constant response noise sigma in log-odds space, so trial-to-trial spread should be about the same in every cell. The Weber-noise theory has encoding sd = s0 + w|x| (w>0). Its spread should therefore be smallest when both inputs are near zero: the 50/50 baseline, prior 0.6 alone, and LLR 0.41 alone. Here sd is about sqrt(2)*k*s0, roughly 0.3-0.5 logits. Spread should be clearly larger when an input is 2.2 logits or more: LLR 2.2 alone, prior 0.9 alone, the split 1.1+1.1 cell, and the split 1.95+1.95 cells. Here sd is roughly 0.7-0.9 logits. The near-50% baseline responses under the advocated theory also tend to collapse onto the 50% bin once rounding is applied, which further lowers low-cell spread. Spread is measured robustly to lapses (epsilon up to 0.4) and rounding. Each response logit is taken as a deviation from that subject's own cell median, so between-subject mean differences do not inflate the spread. The median absolute deviation is then pooled over 15 low-magnitude trials and 25 high-magnitude trials. The metric is log(MAD_high / MAD_low), with a small floor to avoid dividing by zero. Predictions: about 0 for the linear theory, whatever alpha, beta or sigma are; clearly positive (about 0.4-0.8) for the Weber-noise theory. It works identically on pooled data or on one subject's 40 trials.

Source:
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


### RESULTS
- Predicted under pi_1 (simulated): 0.0709 (var=0.2473)
- Predicted under pi_3 (simulated): 0.5447 (var=0.2320)
- Observed on real data: 0.1515 (var=0.1960)

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
**Predicted under pi_3:** 0.1675 (var=0.0091)

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
**Predicted under pi_3:** -1.3863 (var=0.5246)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across all four experiments, the fixed linear log-odds weighting theory (pi_1) is the better and more task-invariant account. The Weber-noise/shrinkage theory (pi_3) fails on its own signature prediction.\n\nExperiment 1 (relative evidence/prior weight):\n- Observed ratio: 0.67. pi_1 predicted 0.53; pi_3 predicted 0.84.\n- Neither theory captures it. pi_1 overstates conservatism relative to prior use; pi_3 overstates the equality of the two gains.\n- pi_1 is slightly closer, and its parameter range could plausibly reach 0.67.\n\nExperiment 2 (pi_3's own variance-coupling test):\n- Observed log(MAD_high/MAD_low): 0.15. pi_1 predicted about 0.07; pi_3 predicted 0.54.\n- The magnitude-scaled encoding noise that pi_3 requires is largely absent. Variability in log-odds is close to constant, with only a weak positive residual.\n- This is a direct falsification of pi_3's core mechanism. Its description says mean distortion and variability are coupled through w>0.\n\nExperiment 3 (exact-100 reports in moderately extreme cells):\n- Observed 0.14; both theories predicted 0.1675.\n- This is a tie, and both slightly overpredict endpoint reports.\n\nExperiment 4 (probability-scale spread, extreme vs mid cells):\n- Observed -1.50. pi_1 predicted -1.61; pi_3 predicted -1.39.\n- The two theories are roughly equal, with pi_1 marginally closer. Constant log-odds noise compressing spread near the extremes is consistent with the data.\n\nOverall:\n- pi_1 wins or ties everywhere.\n- pi_3's distinctive claims (falling gains with magnitude, Weber noise) are not supported: the variance test came out near the linear prediction, and the weight ratio is not near-equal.\n- Some residuals remain that pi_1 does not explain:\n  - a mildly positive variance growth with magnitude;\n  - an intermediate evidence/prior ratio;\n  - slightly fewer exact-100 reports than predicted.\n- These residuals leave room for a stronger competitor. pi_3 should be replaced rather than re-parameterized.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Replace pi_3 with a theory that keeps additive log-odds integration on average but differs from pi_1 in its trial-level generative structure. Proposed name: 'Stochastic partial integration with a bounded response range.'\n\nComponents:\n(a) Integration rule on each trial. With probability 1-rho the subject integrates: L = alpha*L0 + beta*LLR. With probability rho it fails to integrate and reports from a single cue. On those trials it uses the prior alone (alpha*L0) with probability q, or the evidence alone (beta*LLR) with probability 1-q. These are anchoring / representativeness shortcuts.\n(b) Bounded output. The internal log-odds passes through a Zhang-Maloney-style bounded transform: L' = Delta*tanh(L/Delta), with Delta around 2.5-4. This caps reports near 92-98% and keeps exact-100 reports rare, without making the encoding gains magnitude-dependent.\n(c) Noise. Constant response noise sigma in log-odds, plus a lapse rate epsilon.\n\nWhat it predicts:\n- Single-source cells behave like pi_1, with no gain decline in encoding. Only output compression appears at the far extremes.\n- Two-cue cells (split, congruent and especially conflict cells) gain extra spread from the strategy mixture. This produces the small positive MAD_high/MAD_low seen in Experiment 2, coming only from cells with two nonzero cues.\n- The Experiment 1 ratio lands at an intermediate value (about 0.6-0.75), because conflict-cell medians are pulled by prior-only shortcut trials.\n\nParameter ranges:\n- alpha 0.6-1.1, beta 0.4-1.0\n- rho 0-0.4, q 0.3-0.8\n- Delta 2-5, sigma 0.3-1.0, epsilon 0-0.15\n\nDiscriminating tests against pi_1:\n- Bimodality or excess spread in conflict cells relative to median-matched single-source cells. pi_1 predicts no excess.\n- Sub-additivity in congruent cells. The mixture places mass at single-cue values, so congruent medians fall below alpha*L0 + beta*LLR.\n- Variance in split cells exceeding variance in single-source cells of equal total log-odds."
}
```

## Usage

```json
{
  "input_tokens": 13087,
  "output_tokens": 2735,
  "reasoning_tokens": 1099,
  "provider": "Claude Platform on AWS"
}
```
