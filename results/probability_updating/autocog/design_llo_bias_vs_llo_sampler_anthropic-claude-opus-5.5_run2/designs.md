# Proposed experiments

Predictions are from 50 simulated subjects per theory; p is a Welch test at N = 25 real subjects per theory's prediction.

## Round 0, proposed by `pi_1`

**Unique trials (n=12):**
  trial 1: prior=0.500  P(e|H)=0.5  P(e|not-H)=0.5  n_signals=1  log LR=+0.00
  trial 2: prior=0.500  P(e|H)=0.8  P(e|not-H)=0.2  n_signals=1  log LR=+1.39
  trial 3: prior=0.500  P(e|H)=0.95  P(e|not-H)=0.05  n_signals=1  log LR=+2.94
  trial 4: prior=0.500  P(e|H)=0.98  P(e|not-H)=0.02  n_signals=1  log LR=+3.89
  trial 5: prior=0.900  P(e|H)=0.5  P(e|not-H)=0.5  n_signals=1  log LR=+0.00
  trial 6: prior=0.900  P(e|H)=0.9  P(e|not-H)=0.1  n_signals=1  log LR=+2.20
  trial 7: prior=0.970  P(e|H)=0.95  P(e|not-H)=0.05  n_signals=1  log LR=+2.94
  trial 8: prior=0.100  P(e|H)=0.5  P(e|not-H)=0.5  n_signals=1  log LR=+0.00
  trial 9: prior=0.100  P(e|H)=0.9  P(e|not-H)=0.1  n_signals=1  log LR=+2.20
  trial 10: prior=0.900  P(e|H)=0.05  P(e|not-H)=0.95  n_signals=1  log LR=-2.94
  trial 11: prior=0.030  P(e|H)=0.05  P(e|not-H)=0.95  n_signals=1  log LR=-2.94
  trial 12: prior=0.300  P(e|H)=0.2  P(e|not-H)=0.8  n_signals=1  log LR=-1.39

**Rationale:** Both theories share the same mean structure (alpha*prior log-odds + beta*LLR), so the design targets the noise structure. The theories make three different predictions. (1) Shape of spread across belief levels. Constant log-odds noise gives an SD in probability space of about sigma*p(1-p), which collapses fast toward the extremes. Binomial sampling gives an SD of sqrt(p(1-p)/N), which shrinks only like the square root. So the ratio of SD at a distorted belief near 0.9-0.97 to SD at 0.5 is about 0.1-0.36 under the advocated theory versus about 0.17-0.6 under the sampling theory. (2) Shape and mean of the report distribution. Logit-normal noise yields skewed distributions with a tail toward 50%, and means in probability space are pulled toward 50% as sigma grows. Binomial means sit exactly at the target and are nearly symmetric for moderate N. (3) Extreme reports. With large sigma, logit noise puts real mass on 0%/100% when the belief is moderately extreme. The sampling theory puts mass at 0/100 only when Np(1-p) is small. Trial structure: priors (0.03 to 0.97) and LLRs (-2.94 to +3.89) are crossed rather than confounded. This includes zero-evidence trials at three priors (isolating alpha), evidence-only trials at prior 0.5 (isolating beta), congruent trials driving beliefs to the extremes, and conflicting trials landing near 50%. That gives several cells with similar distorted beliefs but different prior/LLR compositions, and others spanning the full belief range, so the SD-vs-mean curve can be traced. All evidence is given as a single aggregated signal (n_signals=1), so the likelihoods are unambiguous. With 12 unique trials each is repeated 5 times, which allows within-cell variance estimates. Rounding to 5/10% blurs exact values, so analyses should use SDs and interquartile spreads across cells rather than exact response values.

**Schedule:** 12 unique x 5 reps = 60 trials per subject.


**Metric:**

```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    slopes = []
    for sid, d in data.groupby('subject_id'):
        key = d['prior'].round(4).astype(str) + '_' + d['log_lr'].round(3).astype(str)
        xs = []
        ys = []
        for k, g in d.groupby(key):
            r = np.sort(g['response'].values.astype(float))
            n = len(r)
            if n < 3:
                continue
            m = float(np.median(r))
            if n >= 5:
                s = r[n - 2] - r[1]
            else:
                s = r[-1] - r[0]
            mm = min(max(m, 0.01), 0.99)
            xs.append(np.log(mm * (1.0 - mm)))
            ys.append(np.log(s + 0.02))
        if len(xs) >= 4 and np.var(xs) > 1e-6:
            xs = np.asarray(xs)
            ys = np.asarray(ys)
            xc = xs - xs.mean()
            slope = float(np.sum(xc * (ys - ys.mean())) / np.sum(xc * xc))
            slopes.append(slope)
    if len(slopes) == 0:
        return float('nan')
    return float(np.mean(slopes))
```

**Metric rationale:** Both theories share the same mean structure (alpha*prior log-odds + beta*LLR), so they differ only in how within-cell response spread scales with the central belief m. Constant logit-normal noise gives spread proportional to m(1-m), so a log-log regression of within-cell spread on m(1-m) has slope near 1, a bit lower with large sigma or the floor effects. Binomial sampling gives spread proportional to sqrt(m(1-m)), so the slope is near 0.5. This holds whatever the noise scale (sigma or N), so parameter heterogeneity across subjects matters less. For each subject and each of the 12 cells (5 reps each), I take the cell median as m. The spread is a trimmed range: 2nd to 4th order statistic, which is robust to one lapse on each side. I then regress log(spread + 0.02) on log(m(1-m)), clipping m to [0.01, 0.99], and average the slopes across subjects. The 0.02 floor tames zero spreads caused by discretization and rounding. The 12 cells span beliefs from about 0.5 to above 0.99, which gives good leverage on the slope. I expect a slope of about 0.8-1 under the logit-noise theory and about 0.4-0.5 under the sampling theory.

| theory | predicted (var) |
|---|---|
| pi_1 | 0.4549 (var=0.0953) |
| pi_2 | 0.1850 (var=0.0811) |

Welch p (pi_1 vs pi_2): 0.00236

## Round 0, proposed by `pi_2`

**Unique trials (n=10):**
  trial 1: prior=0.500  P(e|H)=0.5  P(e|not-H)=0.5  n_signals=1  log LR=+0.00
  trial 2: prior=0.500  P(e|H)=0.75  P(e|not-H)=0.25  n_signals=1  log LR=+1.10
  trial 3: prior=0.500  P(e|H)=0.95  P(e|not-H)=0.05  n_signals=1  log LR=+2.94
  trial 4: prior=0.500  P(e|H)=0.01  P(e|not-H)=0.99  n_signals=1  log LR=-4.60
  trial 5: prior=0.900  P(e|H)=0.5  P(e|not-H)=0.5  n_signals=1  log LR=+0.00
  trial 6: prior=0.020  P(e|H)=0.5  P(e|not-H)=0.5  n_signals=1  log LR=+0.00
  trial 7: prior=0.950  P(e|H)=0.9  P(e|not-H)=0.1  n_signals=1  log LR=+2.20
  trial 8: prior=0.050  P(e|H)=0.05  P(e|not-H)=0.95  n_signals=1  log LR=-2.94
  trial 9: prior=0.990  P(e|H)=0.8  P(e|not-H)=0.2  n_signals=1  log LR=+1.39
  trial 10: prior=0.200  P(e|H)=0.8  P(e|not-H)=0.2  n_signals=1  log LR=+1.39

**Rationale:** Both theories share the mean structure expit(alpha*prior log-odds + beta*LLR), so this design targets how response variability scales with the distorted belief p. It differs from the earlier design in three ways: fewer unique trials (10, so K=6 repeats each) for tighter within-cell SD estimates, a parameter-free anchor cell, and denser sampling of the extreme tails, where the theories diverge most.

(1) Anchor: prior 0.5 with uninformative evidence (P=0.5 under both H and not-H). The belief is exactly 0.5 whatever alpha and beta are. So the SD here estimates 0.5/sqrt(N) under binomial sampling, and about 0.25*sigma under logit-normal noise. Every other cell's SD is then normalised by this one.

(2) Predicted normalised SD ratio. Under sampling it is 2*sqrt(p(1-p)): 0.6 at p=0.9, 0.28 at p=0.98, 0.2 at p=0.99. Under constant logit noise it is roughly 4p(1-p): 0.36, 0.08 and 0.04. The ratio is also skewed, with a long tail toward 50%. The gap widens sharply in the tails. That is why several cells drive beliefs to 0.9-0.99 and 0.01-0.1 even under strong conservatism:
- strong single evidence: LLR +2.94 and -4.6 at prior 0.5;
- extreme priors with null evidence: 0.9 and 0.02, which isolate alpha;
- congruent prior and evidence: 0.95 with LR 9, 0.05 with LR 1/19, and 0.99 with LR 4.

(3) Asymmetry and skew. Under logit noise, extreme cells show right-skewed (low side) or left-skewed (high side) reports with tails reaching toward 50%. Binomial sampling gives compact distributions that pile at 0 or 100 when Np(1-p) is small. Analyses should therefore compare tail-cell SDs and IQRs, and the fraction of reports at least 15 points from the cell median, normalised by the anchor cell. These spread-based measures are robust to the 5/10% rounding.

(4) Moderate cells: LLR +1.1 at prior 0.5, and the conflict cell prior 0.2 with LR 4. These trace the middle of the SD-vs-belief curve. They also pin down beta and alpha for mapping each cell to its p.

Evidence is a single aggregated signal (n_signals=1), so the likelihoods are unambiguous.

**Schedule:** 10 unique x 6 reps = 60 trials per subject.


**Metric:**

```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    d = data.copy()
    d['prior_r'] = d['prior'].astype(float).round(3)
    d['llr_r'] = d['log_lr'].astype(float).round(2)
    d['resp'] = d['response'].astype(float)
    med = d.groupby(['subject_id', 'prior_r', 'llr_r'])['resp'].transform('median')
    d['dev'] = (d['resp'] - med).abs()
    pr = d['prior_r'].values
    ll = d['llr_r'].values
    center = (np.isclose(pr, 0.5) & (np.abs(ll) < 0.01)) | (np.isclose(pr, 0.5) & np.isclose(ll, 1.10, atol=0.02)) | np.isclose(pr, 0.2)
    tail = np.isclose(pr, 0.9) | np.isclose(pr, 0.02) | np.isclose(pr, 0.95) | np.isclose(pr, 0.05) | np.isclose(pr, 0.99)
    if center.sum() == 0 or tail.sum() == 0:
        return float('nan')
    s_c = float(np.median(d.loc[center, 'dev'].values))
    s_t = float(np.median(d.loc[tail, 'dev'].values))
    return float(np.log((s_t + 0.005) / (s_c + 0.005)))
```

**Metric rationale:** Both theories share the same central tendency, expit(alpha*prior logodds + beta*LLR), but they scale trial-to-trial spread with belief differently. Binomial sampling gives SD of about sqrt(p(1-p)/N), plus a constant motor noise tau that does not shrink in the tails. Constant logit-normal noise gives SD of about sigma*p(1-p). The ratio of tail-cell spread to center-cell spread therefore cancels the subject's noise-level parameter (N or sigma) to first order. It is about 2*sqrt(q) (or more, given tau) under sampling and about 4q under logit noise, where q=p(1-p) is roughly 0.02-0.1 in the tail cells. At q around 0.03 that is roughly 0.35-0.6 versus 0.12.

How the metric is computed:
- Deviations are taken from each subject x cell median, so alpha and beta shifts in location are removed.
- Center cells are those with distorted beliefs near 0.5: prior .5 with null or weak evidence, and prior .2 with LR 4.
- Tail cells are those with extreme priors and null or congruent evidence, so beliefs fall near 0.02-0.995 for any plausible alpha and beta.
- The spread in each group is the pooled median absolute deviation. This is robust to lapses, which hit both groups alike, and to the 5/10% rounding.
- The metric is the log ratio of tail spread to center spread, with a small floor added to each.

Predicted values are about -0.7 to 0 under sampling and about -2 under logit noise. The same code works on single subjects and on pooled data.

| theory | predicted (var) |
|---|---|
| pi_2 | -0.3830 (var=0.4916) |
| pi_1 | -1.1260 (var=0.6560) |

Welch p (pi_2 vs pi_1): 0.00113

