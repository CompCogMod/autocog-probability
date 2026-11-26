# Proposed experiments

Predictions are from 50 simulated subjects per theory; p is a Welch test at N = 25 real subjects per theory's prediction.

## Round 0, proposed by `pi_1`

**Unique trials (n=12):**
  trial 1: prior=0.500  P(e|H)=0.6  P(e|not-H)=0.4  n_signals=1  log LR=+0.41
  trial 2: prior=0.500  P(e|H)=0.7  P(e|not-H)=0.3  n_signals=5  log LR=+0.85
  trial 3: prior=0.500  P(e|H)=0.3  P(e|not-H)=0.7  n_signals=5  log LR=-0.85
  trial 4: prior=0.100  P(e|H)=0.7  P(e|not-H)=0.3  n_signals=3  log LR=+0.85
  trial 5: prior=0.900  P(e|H)=0.3  P(e|not-H)=0.7  n_signals=3  log LR=-0.85
  trial 6: prior=0.900  P(e|H)=0.7  P(e|not-H)=0.3  n_signals=3  log LR=+0.85
  trial 7: prior=0.100  P(e|H)=0.3  P(e|not-H)=0.7  n_signals=3  log LR=-0.85
  trial 8: prior=0.020  P(e|H)=0.6  P(e|not-H)=0.4  n_signals=1  log LR=+0.41
  trial 9: prior=0.980  P(e|H)=0.4  P(e|not-H)=0.6  n_signals=1  log LR=-0.41
  trial 10: prior=0.800  P(e|H)=0.5  P(e|not-H)=0.5  n_signals=2  log LR=+0.00
  trial 11: prior=0.300  P(e|H)=0.8  P(e|not-H)=0.2  n_signals=2  log LR=+1.39
  trial 12: prior=0.600  P(e|H)=0.55  P(e|not-H)=0.45  n_signals=1  log LR=+0.20

**Rationale:** Goal: separate constant Gaussian noise in log-odds space (advocated) from binomial sampling noise around a log-odds-distorted belief (competing). I assume log LR = n_signals * log(p_h / p_not_h), with p_h and p_not_h giving the per-draw probability of the observed colour; all draws in a trial share that colour. There are 12 unique trials, so K = 5 repetitions each, enough to estimate within-cell spread.

(1) Variability profile.
- Advocated theory: spread is constant in log-odds. In probability units the SD is about sigma*p(1-p), but the distribution is skewed, with tails reaching well into the interior even when the central judgment is extreme.
- Competing theory: SD in probability is sqrt(p(1-p)/N), so the log-odds SD blows up as 1/sqrt(N p(1-p)). Near 0 or 100, reports pile onto the endpoint with almost no spread.
- Cells used to probe this:
  - near-50% cells: prior .5 with LLR .41; prior .6 with LLR .20;
  - conflict cells, where a strong prior is opposed by strong evidence: .1 vs +2.54; .9 vs -2.54;
  - extreme consistent cells: prior .5 with LLR ±4.24; prior .9 with LLR +2.54; prior .1 with LLR -2.54; prior .3 with LLR +2.77.
- Key metric: the ratio of within-cell spread at extreme cells to spread at mid cells, in probability units and in log-odds. The two theories predict opposite trends in log-odds, and a robust measure such as IQR or MAD survives the 5/10% rounding.

(2) Mean shrinkage.
- Advocated theory: averaging expit over log-odds noise pulls the probability-space mean toward 50% at extreme targets. A small sigma would remove this, but then the large interior tails would not appear either.
- Competing theory: the binomial mean equals the distorted target exactly.

(3) Prior weighting, decoupled from evidence.
- Pure-prior cell: prior .8 with zero LLR (p_h = p_not_h).
- Very extreme priors with weak opposing evidence: .02 and .98 with LLR ∓0.41.
- These cells estimate alpha directly. The competing theory permits alpha > 1 (over-extremised priors, reports beyond Bayes); the advocated theory caps alpha near 1, predicting base-rate neglect or near-Bayesian use of the prior.

(4) Independence of prior and evidence.
- Prior and LLR are crossed independently (priors .02-.98; LLR from -4.24 to +4.24) so that alpha and beta are identifiable separately from the noise parameters.
- Advocated prediction: the median report in log-odds is linear in prior log-odds and LLR, and its spread is constant across cells.

**Schedule:** 12 unique x 5 reps = 60 trials per subject.


**Metric:**

```python
def metric(data: pd.DataFrame) -> float:
    p = data['prior'].values.astype(float)
    l = data['log_lr'].values.astype(float)
    r = np.clip(data['response'].values.astype(float), 0.005, 0.995)
    z = np.log(r / (1.0 - r))
    def close(a, b):
        return np.abs(a - b) < 1e-3
    def med(mask):
        if mask.sum() == 0:
            return None
        return float(np.median(z[mask]))
    def lo(x):
        return float(np.log(x / (1.0 - x)))
    num = 0.0
    den = 0.0
    # symmetric pairs at prior .9 and .1: evidence term cancels
    for pr in [0.9, 0.1]:
        a = med(close(p, pr) & (l > 1e-9))
        b = med(close(p, pr) & (l < -1e-9))
        if a is not None and b is not None:
            s = np.sign(lo(pr))
            num += s * 0.5 * (a + b)
            den += abs(lo(pr))
    # extreme priors .98 / .02 with weak opposing evidence
    a = med(close(p, 0.98))
    b = med(close(p, 0.02))
    if a is not None and b is not None:
        num += 0.5 * (a - b)
        den += abs(lo(0.98))
    # pure prior cell (.8, zero evidence)
    c = med(close(p, 0.8) & (np.abs(l) < 1e-9))
    if c is not None:
        num += c
        den += abs(lo(0.8))
    if den == 0:
        return float('nan')
    return float(num / den)
```

**Metric rationale:** Estimates the prior weight alpha directly and robustly, decoupled from the evidence weight: it uses per-cell MEDIAN responses in log-odds (robust to the large lapse rates and to 5/10% rounding). Beta is cancelled by averaging the symmetric pairs (prior .9 with +/-LLR, prior .1 with +/-LLR); it is nearly irrelevant for the .98/.02 pair with weak opposing evidence; and it is exactly absent for the prior-.8, zero-LLR cell. The estimate is the summed signed median log-odds divided by the summed |prior log-odds|. The advocated log-odds-noise theory caps prior weighting near or below 1: alpha lies in about [0.73, 1.02], i.e. base-rate neglect or near-Bayesian use of the prior, so the metric should land around 0.8-0.9. The competing binomial-sampling theory needs over-extremised priors (alpha in about [0.77, 1.94]) to account for the pile-up of reports at the endpoints. Its reports therefore sit further from 50% on prior-dominated cells, giving a metric well above 1 (clipping at 0.5%/99.5% only partially limits this). The metric is a median-based ratio that is valid on a single subject's slice and pools over 7 cells (about 35 trials), so per-subject variance should be moderate, while the predicted means differ by roughly 0.3-0.5.

| theory | predicted (var) |
|---|---|
| pi_1 | 0.7744 (var=0.0178) |
| pi_2 | 0.9952 (var=0.0708) |

Welch p (pi_1 vs pi_2): 0.000706

## Round 0, proposed by `pi_2`

**Unique trials (n=12):**
  trial 1: prior=0.150  P(e|H)=0.5  P(e|not-H)=0.5  n_signals=1  log LR=+0.00
  trial 2: prior=0.300  P(e|H)=0.5  P(e|not-H)=0.5  n_signals=1  log LR=+0.00
  trial 3: prior=0.700  P(e|H)=0.5  P(e|not-H)=0.5  n_signals=1  log LR=+0.00
  trial 4: prior=0.850  P(e|H)=0.5  P(e|not-H)=0.5  n_signals=1  log LR=+0.00
  trial 5: prior=0.500  P(e|H)=0.6  P(e|not-H)=0.4  n_signals=1  log LR=+0.41
  trial 6: prior=0.500  P(e|H)=0.8  P(e|not-H)=0.2  n_signals=1  log LR=+1.39
  trial 7: prior=0.500  P(e|H)=0.2  P(e|not-H)=0.8  n_signals=1  log LR=-1.39
  trial 8: prior=0.500  P(e|H)=0.95  P(e|not-H)=0.05  n_signals=1  log LR=+2.94
  trial 9: prior=0.500  P(e|H)=0.99  P(e|not-H)=0.01  n_signals=1  log LR=+4.60
  trial 10: prior=0.900  P(e|H)=0.1  P(e|not-H)=0.9  n_signals=1  log LR=-2.20
  trial 11: prior=0.200  P(e|H)=0.8  P(e|not-H)=0.2  n_signals=1  log LR=+1.39
  trial 12: prior=0.990  P(e|H)=0.01  P(e|not-H)=0.99  n_signals=1  log LR=-4.60

**Rationale:** Goal: separate a log-odds-distorted belief reported by binomial sampling (advocated) from the same distortion reported with constant Gaussian noise in log-odds (competing). Every trial uses a single signal (n=1), so the log LR is just log(p_h/p_not_h) and there is no ambiguity about how draws combine. There are 12 unique trials, giving K=5 repetitions each.

(A) Pure-prior cells. The evidence is uninformative (p_h = p_not_h = 0.5) and the priors are .15, .30, .70 and .85.
- These isolate alpha.
- Advocated: alpha may exceed 1 (up to about 1.9). Reports would then be more extreme than the stated prior, e.g. prior .85 reported near 90-95%.
- Competing: alpha is at most about 1.02, so reports are at or regressive toward 50%.
- Metric: median report minus prior, signed away from 50%.

(B) Evidence-only cells at prior .5. The log LRs are +0.41, +1.39, -1.39, +2.94 and +4.60.
- These trace beta, and they trace the variability profile as the distorted belief moves from about 55% to about 95%.
- Advocated, spread in probability units: SD = sqrt(p(1-p)/N), which falls slowly with extremity (about 0.6x from p=.5 to p=.9).
- Competing, spread in probability units: SD is roughly sigma*p(1-p), which falls faster (about 0.36x over the same range).
- In log-odds, the advocated spread grows toward the extremes while the competing spread is flat.
- Means: the binomial mean equals expit(target) exactly, whereas Gaussian log-odds noise shrinks the probability mean toward 50%.

(C) Conflict cells. Strong priors are opposed by evidence of equal strength, so the Bayesian posterior is 50%: .90 vs -2.20, .20 vs +1.39, .95 vs -2.94, .99 vs -4.60.
- Both theories shift these cells toward the prior direction when alpha > beta.
- Advocated: alpha can be well above beta, giving an extreme internal belief and tight, near-endpoint reports.
- Competing: alpha is at most 1, so the belief stays closer to 50%, with constant log-odds spread and therefore wide spread in probability units.
- These cells also test linear additivity. As the paired prior and LLR scale up together, the advocated theory, with alpha above beta, predicts the prior to dominate more and more. The competing theory, with both weights below 1, predicts much milder drift.

Key metrics, chosen to be robust to the 5/10% rounding:
- median report per cell;
- IQR per cell, in probability units and in log-odds;
- the slope of log-odds IQR against distance of the median from 50%;
- the sign of the pure-prior extremization.

**Schedule:** 12 unique x 5 reps = 60 trials per subject.


**Metric:**

```python
def metric(data: pd.DataFrame) -> float:
    d = data[(np.abs(data['log_lr']) < 1e-9) & (np.abs(data['prior'] - 0.5) > 0.05)]
    if len(d) == 0:
        return float('nan')
    prior = d['prior'].to_numpy(dtype=float)
    resp = d['response'].to_numpy(dtype=float)
    folded_resp = np.where(prior > 0.5, resp, 1.0 - resp)
    folded_prior = np.where(prior > 0.5, prior, 1.0 - prior)
    dev = folded_resp - folded_prior
    return float(np.median(dev))
```

**Metric rationale:** This metric uses only the pure-prior cells (uninformative evidence, priors .15/.30/.70/.85), so the report depends on alpha alone. Each response is folded so that 'away from 50% in the prior's direction' is positive. The metric is the median of the folded report minus the folded stated prior.

Advocated theory: alpha ranges up to about 1.94 (mean about 1.36), so internal beliefs are often more extreme than the stated prior. For example, prior .85 with alpha 1.36 gives about .91, and prior .70 gives about .76. The binomial reports are centred on that belief, so the median deviation is positive, around +0.04 to +0.06.

Competing theory: alpha is at most about 1.02, so beliefs are at or regressive toward 50%. The log-odds Gaussian noise is symmetric in log-odds, so the median stays at the regressed belief. The deviation is therefore zero or negative, around -0.02 to -0.03.

Choices that keep the estimate stable:
- The median resists lapses and the 5/10% rounding.
- Pooling 20 trials per subject (4 priors x 5 reps) helps.
- The competing theory's narrow alpha range keeps its between-subject variance small.

Together these should give a large t at N=25.

| theory | predicted (var) |
|---|---|
| pi_2 | 0.0300 (var=0.0041) |
| pi_1 | -0.0500 (var=0.0039) |

Welch p (pi_2 vs pi_1): 4.86e-05

