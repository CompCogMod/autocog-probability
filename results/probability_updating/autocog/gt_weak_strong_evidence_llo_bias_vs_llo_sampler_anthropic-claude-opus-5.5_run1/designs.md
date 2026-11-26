# Proposed experiments

Predictions are from 50 simulated subjects per theory; p is a Welch test at N = 25 real subjects per theory's prediction.

## Round 0, proposed by `pi_1`

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

**Rationale:** The design targets how response variability scales with the level of the distorted belief, while keeping the mean model (alpha times prior log-odds plus beta times LLR) identical in both theories. Prior log-odds and evidence LLR are crossed independently so alpha and beta are identifiable. Each trial uses a single signal (n=1) so the LLR is simply log(p_h/p_not_h), with values 0, +/-1.39, +/-2.94 and +/-4.6. Priors range over 0.01, 0.05, 0.2, 0.5, 0.8, 0.95 and 0.99.

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

**Schedule:** 12 unique x 5 reps = 60 trials per subject.


**Metric:**

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

**Metric rationale:** In the moderately extreme cells (Bayes posterior 0.95-0.99 or 0.01-0.05: trials 3,4,5,6,11,12) the distorted belief typically sits around 91-98%. I fold responses toward the favoured side and, among responses in the near-extreme window (>=90%), compute the share that are exactly at the endpoint (100% / 0%). Logit-normal noise (advocated) is a smooth distribution in log-odds that can't pile up at the boundary: its mode sits at expit(mu) (e.g. 96-98%), and reaching the 100 bin requires z > logit(0.995)=5.3, so endpoint share stays low (mostly coming from the shared rounding stage). Binomial sampling (competing) puts mass at k=N, and its Gaussian motor smear tau has open outer bin edges, so every bit of smeared mass above 99.5% lands in the 100 bin; its folded near-extreme distribution therefore has its mode AT the endpoint. The conditional ratio (endpoint / all >=90) removes most dependence on alpha/beta and lapse rate (lapses contribute only ~11/101 of their mass to the window), and the shared rounding affects both theories identically. Pooling 30 trials per subject keeps per-subject estimates reasonably stable, with expected values roughly 0.2-0.3 for logit-normal versus ~0.45-0.6 for binomial-sampling.

| theory | predicted (var) |
|---|---|
| pi_1 | 0.1675 (var=0.0068) |
| pi_2 | 0.3688 (var=0.0217) |
| pi_3 | 0.1675 (var=0.0091) |
| pi_4 | 0.1480 (var=0.0076) |
| pi_5 | 0.1547 (var=0.0089) |
| pi_6 | 0.1236 (var=0.0056) |
| ground truth | 0.1400 (var=0.0062) |

Welch p (pi_1 vs pi_2): 6.52e-07

## Round 0, proposed by `pi_2`

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

**Rationale:** This design tests the advocated theory's distinctive signature: binomial report variability around a log-odds-distorted belief. The alternative is logit-normal noise with constant sigma around the same distorted belief. Both theories share the mean structure alpha*prior log-odds + beta*LLR. The design therefore crosses priors (0.05, 0.2, 0.5, 0.8, 0.95) with single-draw evidence (LLR 0, ±1.10, ±2.20, ±3.89) so alpha and beta are identified separately from the variability structure.

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

**Schedule:** 12 unique x 5 reps = 60 trials per subject.


**Metric:**

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

**Metric rationale:** Shape of the variance law, measured within subject as a ratio so that overall noise level (N, tau vs sigma), which overlaps between theories, cancels. For each subject x cell, take absolute deviations of responses from the cell median (robust to the large lapse rates), pool them over cells whose distorted belief is reliably extreme for ANY admissible alpha/beta (prior .95 with no evidence, .95 with LLR +3.89, .05 with LLR -3.89; belief ~0.90-0.998), and over mid-range cells (prior .5 with LLR 0 or +1.10, prior .2 with LLR +2.2; belief ~0.25-0.72). Metric = log((median extreme dev + .01)/(median mid dev + .01)). Logit-normal noise scales as sigma*p(1-p): spread at p~0.95-0.99 collapses to ~4-20% of mid spread (log ratio strongly negative, about -1.5 to -2). Binomial sampling scales as sqrt(p(1-p)/N) and the motor noise tau adds a probability-space floor that does not shrink near the ends, so extreme spread stays ~40-100% of mid spread (log ratio near -0.5 to 0). The +0.01 floor guards against 1-point discretization; the median-based deviations resist lapses and the 5/10% rounding stage affects both theories similarly.

| theory | predicted (var) |
|---|---|
| pi_2 | -0.4055 (var=0.4605) |
| pi_1 | -1.6094 (var=0.6514) |
| pi_3 | -1.3863 (var=0.5246) |
| pi_4 | -1.5041 (var=0.3621) |
| pi_5 | -1.0986 (var=0.6023) |
| pi_6 | -1.3863 (var=0.5680) |
| ground truth | -1.5041 (var=0.4026) |

Welch p (pi_2 vs pi_1): 7.58e-07

## Round 1, proposed by `pi_1`

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


**Metric:**

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

**Metric rationale:** Estimates the relative weight of evidence vs. prior (beta/alpha) at matched input magnitudes, using median response log-odds. For each magnitude x in {0.85,1.39,2.2,2.94,4.6}, the evidence-only cell (prior .5, LLR x) gives beta*x, the prior-only cell (prior expit(x), uninformative evidence) gives alpha*x, and conflict cells (prior expit(x), LLR -x) give (alpha-beta)*x, so P-C is a second estimate of the evidence contribution. Metric = (sum E + sum(P-C)) / (sum of matching P). Both theories are additive in log-odds, so this ratio cleanly reads off the relative evidence/prior weight. Advocated theory: weights are fixed with strong conservatism relative to prior use (beta ~0.12-0.88 vs alpha ~0.73-1.04), giving a ratio typically ~0.5-0.6 and conflict cells resolving clearly toward H. Competing theory: both gains come from the same shrinkage law at matched magnitude (k = lambda^2/(lambda^2+(s0+w|x|)^2)), with lambda_e and lambda_p both large relative to encoding sd, so effective weights are near-equal (ratio ~0.8-0.95) and conflict cells sit near 50-65%. Using sums of cell medians across 8 cell pairs (pooled over all trials of a cell) is robust to lapses and rounding, and the ratio is a per-subject quantity with moderate between-subject variance, unlike the previous spread-based metric whose noise swamped the effect.

| theory | predicted (var) |
|---|---|
| pi_1 | 0.5297 (var=0.1446) |
| pi_3 | 0.8425 (var=0.0442) |
| pi_2 | 0.5472 (var=0.0784) |
| pi_4 | 0.7687 (var=0.0764) |
| pi_5 | 0.7787 (var=0.0769) |
| pi_6 | 0.7027 (var=0.0676) |
| ground truth | 0.6707 (var=0.1719) |

Welch p (pi_1 vs pi_3): 0.00092

## Round 1, proposed by `pi_3`

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


**Metric:**

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

**Metric rationale:** Variance coupling test, which is different from the earlier mean-based homogeneity contrast that failed. The linear-weighting theory assumes one constant response noise sigma in log-odds space, so trial-to-trial spread should be about the same in every cell. The Weber-noise theory has encoding sd = s0 + w|x| (w>0). Its spread should therefore be smallest when both inputs are near zero: the 50/50 baseline, prior 0.6 alone, and LLR 0.41 alone. Here sd is about sqrt(2)*k*s0, roughly 0.3-0.5 logits. Spread should be clearly larger when an input is 2.2 logits or more: LLR 2.2 alone, prior 0.9 alone, the split 1.1+1.1 cell, and the split 1.95+1.95 cells. Here sd is roughly 0.7-0.9 logits. The near-50% baseline responses under the advocated theory also tend to collapse onto the 50% bin once rounding is applied, which further lowers low-cell spread. Spread is measured robustly to lapses (epsilon up to 0.4) and rounding. Each response logit is taken as a deviation from that subject's own cell median, so between-subject mean differences do not inflate the spread. The median absolute deviation is then pooled over 15 low-magnitude trials and 25 high-magnitude trials. The metric is log(MAD_high / MAD_low), with a small floor to avoid dividing by zero. Predictions: about 0 for the linear theory, whatever alpha, beta or sigma are; clearly positive (about 0.4-0.8) for the Weber-noise theory. It works identically on pooled data or on one subject's 40 trials.

| theory | predicted (var) |
|---|---|
| pi_3 | 0.5447 (var=0.2320) |
| pi_1 | 0.0709 (var=0.2473) |
| pi_2 | 0.5459 (var=0.2839) |
| pi_4 | 0.1254 (var=0.2522) |
| pi_5 | 0.2102 (var=0.2816) |
| pi_6 | 0.0631 (var=0.3021) |
| ground truth | 0.1515 (var=0.1960) |

Welch p (pi_3 vs pi_1): 0.00128

## Round 2, proposed by `pi_1`

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


**Metric:**

```python
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
```

**Metric rationale:** Implausible-zone rate in cells where every component of both theories points clearly one way. I use single-source and congruent cells (plus the 0.6-prior/strong-evidence probe, whose competing prior-only mode is still on the same side). For each cell I count reports in a zone that no integration route, prior-only route or evidence-only route can reach without a lapse:
- the wrong side of 50% for strong one-directional cells;
- the opposite extreme for the weak-evidence cells;
- the far tails (<=10 or >=90) for the 50/50 baseline.

Conflict cells and the weak-conflict probes (0.95 with -1.39 or -0.41) are excluded. There the competing theory's evidence-only mode legitimately lands on the far side, and including them would cancel the contrast.

In these zones, responses come almost only from uniform lapses, plus a small Gaussian-tail contribution from very noisy or low-beta subjects. The two theories place outlying responses differently:
- The advocated fixed-weight model has no strategy mixture. Every off-pattern human response must be absorbed by a uniform lapse (epsilon 0.05-0.41) and by a wide sigma (up to 1.2) and low beta (down to 0.12). These produce wrong-side reports even in cells where both cues agree or only one is informative.
- The competing theory explains off-pattern reports mostly through cue-selection modes confined to two-cue conflict cells. It has small lapses (epsilon 0-0.15), sigma at most 1 and beta at least 0.3, so these cells stay clean.

Predicted values: about 0.10-0.13 under the advocated theory versus about 0.03-0.05 under the competing theory. With about 30 eligible trials per subject, the per-subject rate is reasonably tight. The count-based zones use 10%-or-wider margins, so the shared rounding stage barely affects them.

On human data, a clear excess of wrong-side reports in cells with no cue conflict would support a global lapse/noise account over a conflict-specific mixture.

| theory | predicted (var) |
|---|---|
| pi_1 | 0.1220 (var=0.0061) |
| pi_4 | 0.0447 (var=0.0015) |
| pi_2 | 0.1013 (var=0.0040) |
| pi_3 | 0.0953 (var=0.0041) |
| pi_5 | 0.1127 (var=0.0038) |
| pi_6 | 0.1093 (var=0.0045) |
| ground truth | 0.1240 (var=0.0029) |

Welch p (pi_1 vs pi_4): 8.61e-05

## Round 2, proposed by `pi_4`

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


**Metric:**

```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    d = data
    mask = (d['prior'] >= 0.79) & (d['log_lr'] >= -1e-9)
    sub = d.loc[mask]
    if len(sub) == 0:
        return float('nan')
    return float(np.mean(sub['response_bin'].values <= 40))
```

**Metric rationale:** Several design features cannot separate the theories. Both predict() functions use only log_lr, which is +1.39 regardless of n_signals, so the n-sweeps are effectively replicates. The sharpest remaining difference is lapse behaviour.

The metric uses only cells where every component of both theories lies firmly on the H side:
- the prior-only cells at priors .9 and .999 (log-odds about 1.6-7);
- the congruent cells at prior .8 with LLR +1.39, where the integrated, prior-anchored and evidence-only modes are all positive.

This gives 25 trials per subject. In these cells a report at or below 40% is essentially impossible under each model's main components. It arises almost only from the uniform lapse, which places 41/101 of its mass in that region.

Expected values differ by lapse range:
- Competing theory: epsilon spans 0.05-0.41 (mean about 0.23), so the opposite-side fraction is about 0.41*eps, roughly 0.09-0.10.
- Advocated theory: epsilon spans 0-0.15 (mean about 0.075), so the fraction is about 0.03. It also has saturation: the bound only pulls high responses toward the high 90s and never toward the opposite side.

The low-beta prior-0.5 cells and the conflict cells are excluded. There, a near-50% center (competing) or a representativeness mode (advocated) could push mass below 40% for reasons other than lapse.

The metric is a simple proportion, so it is defined per subject. Between-subject variance comes mostly from binomial noise over 25 trials plus epsilon spread. That noise is small relative to the roughly 0.06 gap, which is about t≈3.5-4 at N=25.

| theory | predicted (var) |
|---|---|
| pi_4 | 0.0408 (var=0.0022) |
| pi_1 | 0.1016 (var=0.0049) |
| pi_2 | 0.1032 (var=0.0064) |
| pi_3 | 0.0872 (var=0.0052) |
| pi_5 | 0.1200 (var=0.0065) |
| pi_6 | 0.1168 (var=0.0041) |
| ground truth | 0.1200 (var=0.0049) |

Welch p (pi_4 vs pi_1): 0.000805

## Round 3, proposed by `pi_5`

**Unique trials (n=12):**
  trial 1: prior=0.500  P(e|H)=0.5  P(e|not-H)=0.5  n_signals=1  log LR=+0.00
  trial 2: prior=0.500  P(e|H)=0.9  P(e|not-H)=0.1  n_signals=2  log LR=+2.20
  trial 3: prior=0.500  P(e|H)=0.1  P(e|not-H)=0.9  n_signals=2  log LR=-2.20
  trial 4: prior=0.950  P(e|H)=0.5  P(e|not-H)=0.5  n_signals=1  log LR=+0.00
  trial 5: prior=0.050  P(e|H)=0.5  P(e|not-H)=0.5  n_signals=1  log LR=+0.00
  trial 6: prior=0.600  P(e|H)=0.5  P(e|not-H)=0.5  n_signals=1  log LR=+0.00
  trial 7: prior=0.500  P(e|H)=0.6  P(e|not-H)=0.4  n_signals=1  log LR=+0.41
  trial 8: prior=0.600  P(e|H)=0.9  P(e|not-H)=0.1  n_signals=2  log LR=+2.20
  trial 9: prior=0.400  P(e|H)=0.9  P(e|not-H)=0.1  n_signals=2  log LR=+2.20
  trial 10: prior=0.600  P(e|H)=0.1  P(e|not-H)=0.9  n_signals=2  log LR=-2.20
  trial 11: prior=0.950  P(e|H)=0.6  P(e|not-H)=0.4  n_signals=1  log LR=+0.41
  trial 12: prior=0.750  P(e|H)=0.75  P(e|not-H)=0.25  n_signals=1  log LR=+1.10

**Rationale:** Goal: test where off-pattern responses come from and where they land. The advocated theory says one stimulus-independent off-task component of mass lambda, mostly uniform, plus a polarity-reversal mirror (100-R) and a small 50% spike. It should appear with equal mass in every cell type, and its non-uniform parts sit at the mirror of the engaged answer. The competing theory says off-pattern mass beyond a small uniform epsilon arises only in two-cue cells. There it is a strategy mixture located at the single-cue answers: prior-only alpha*L0 and evidence-only beta*LLR. Earlier designs studied variance laws, gain linearity, additivity, saturation and conflict bimodality. They did not separate mirror-located from single-cue-located extra mass, nor test equal off-pattern mass across cell types. LLR is assumed to be n*log(ph/pnh).

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

**Schedule:** 12 unique x 5 reps = 60 trials per subject.


**Metric:**

```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    d = data.copy()
    d['pk'] = d['prior'].astype(float).round(3)
    d['lk'] = d['log_lr'].astype(float).round(2)
    single = (d['lk'].abs() < 1e-6) | ((d['pk'] - 0.5).abs() < 1e-6)
    d = d[single]
    if len(d) == 0:
        return float('nan')
    r = d['response_bin'].astype(float)
    med = d.groupby(['subject_id', 'pk', 'lk'])['response_bin'].transform('median').astype(float)
    far = (np.abs(r - med) > 35.0).astype(float)
    return float(far.mean())
```

**Metric rationale:** Restrict to single-source cells (prior=0.5 or LLR=0: trials 1-7, 35 trials/subject). Here the competing theory has no strategy mixture, so its only off-pattern mass is a small uniform epsilon (<=0.15, mean ~0.075). The advocated theory has a stimulus-independent off-task component lambda (0.1-0.42, mean ~0.26), mostly uniform plus mirror reversals, which land far from the engaged answer. Metric: fraction of responses more than 35 percentage points from the subject's own median for that cell (median of 5 reps is robust to the lapses and absorbs alpha/beta/bound idiosyncrasies). Engaged log-odds noise (sigma<=0.9) rarely produces 35-point deviations, and the 35-point threshold is far wider than any rounding shift. Expected roughly 0.13-0.15 under the advocated theory vs 0.04-0.05 under the competing theory. Per-subject SD is about 0.07 vs 0.04, giving t of about 5 at N=25.

| theory | predicted (var) |
|---|---|
| pi_5 | 0.0954 (var=0.0029) |
| pi_4 | 0.0411 (var=0.0011) |
| pi_1 | 0.0891 (var=0.0029) |
| pi_2 | 0.0709 (var=0.0032) |
| pi_3 | 0.0697 (var=0.0024) |
| pi_6 | 0.0983 (var=0.0032) |
| ground truth | 0.0869 (var=0.0029) |

Welch p (pi_5 vs pi_4): 0.000115

## Round 3, proposed by `pi_4`

**Unique trials (n=12):**
  trial 1: prior=0.500  P(e|H)=0.5  P(e|not-H)=0.5  n_signals=1  log LR=+0.00
  trial 2: prior=0.900  P(e|H)=0.5  P(e|not-H)=0.5  n_signals=1  log LR=+0.00
  trial 3: prior=0.970  P(e|H)=0.5  P(e|not-H)=0.5  n_signals=1  log LR=+0.00
  trial 4: prior=0.500  P(e|H)=0.9  P(e|not-H)=0.1  n_signals=1  log LR=+2.20
  trial 5: prior=0.500  P(e|H)=0.03  P(e|not-H)=0.97  n_signals=1  log LR=-3.48
  trial 6: prior=0.600  P(e|H)=0.4  P(e|not-H)=0.6  n_signals=1  log LR=-0.41
  trial 7: prior=0.750  P(e|H)=0.25  P(e|not-H)=0.75  n_signals=1  log LR=-1.10
  trial 8: prior=0.900  P(e|H)=0.1  P(e|not-H)=0.9  n_signals=1  log LR=-2.20
  trial 9: prior=0.970  P(e|H)=0.03  P(e|not-H)=0.97  n_signals=1  log LR=-3.48
  trial 10: prior=0.030  P(e|H)=0.97  P(e|not-H)=0.03  n_signals=1  log LR=+3.48
  trial 11: prior=0.750  P(e|H)=0.75  P(e|not-H)=0.25  n_signals=1  log LR=+1.10
  trial 12: prior=0.900  P(e|H)=0.9  P(e|not-H)=0.1  n_signals=1  log LR=+2.20

**Rationale:** Conflict-magnitude dose-response at near-matched integrated belief. All trials use one draw (n=1), so LLR = log(ph/pnh) under any reading of n.

Cells:
- Baseline: 0.5 with 0.5/0.5.
- Single-source anchors: prior-only 0.9 and 0.97; evidence-only LLR +2.20 and -3.48 at prior 0.5.
- Symmetric conflict sweep: the prior and opposing evidence have equal magnitude x = 0.41, 1.10, 2.20, 3.48. That is 0.6 vs 0.4/0.6, 0.75 vs 0.25/0.75, 0.9 vs 0.1/0.9, 0.97 vs 0.03/0.97, plus a mirrored 0.03 vs 0.97/0.03.
- Congruent cells: 0.75 with 0.75/0.25, and 0.9 with 0.9/0.1.

Advocated (stochastic partial integration):
- In conflict cells, a fraction rho of trials answers from one cue. This gives modes at expit(alpha*x) and expit(-beta*x), on opposite sides of 50%, around a central integrated mode at (alpha-beta)*x.
- Mode separation grows with x. So IQR, bimodality and the fraction of reports on the far side of 50% (relative to the median) should rise steeply across the sweep. Example: x=3.48 gives modes near 95% and 10-20%.
- Single-source cells, including the extreme ones, show only logit-normal spread plus a small uniform floor, with no mirror-located mass.
- Congruent cells show sub-additive medians and a lower tail at the single-cue values.

Competing (deterministic integration plus stimulus-independent lambda):
- Conflict cells are unimodal at (alpha-beta)*x with constant logit spread.
- Off-pattern mass is the same in every cell. It is uniform plus a small mirror mode at 100 minus the median, which is mild because lambda*f_rev is at most about 0.06.
- Off-pattern mass therefore does not grow with x.
- The extreme single-source cells (prior 0.97, LLR -3.48) should carry the same excess mirror mass (near 3-10%) as the conflict cells.

Rounding-robust metrics (10%-wide bins):
1. The slope of conflict-cell IQR, and of opposite-side mass, against x. Advocated predicts a positive, accelerating slope; competing predicts flat.
2. Mass within 10 points of the single-cue anchor medians in conflict cells, minus that band's mass in the baseline. Advocated predicts it grows with x; competing predicts about zero.
3. Off-pattern mass in extreme single-source cells versus x=0.41 conflict. Competing predicts equal; advocated predicts both are small.
4. Congruent additivity residuals computed from the anchor medians.

12 unique trials, K=5.

**Schedule:** 12 unique x 5 reps = 60 trials per subject.


**Metric:**

```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    d = data.copy()
    pr = d['prior'].astype(float).clip(1e-6, 1 - 1e-6).values
    L0 = np.log(pr) - np.log1p(-pr)
    ll = d['log_lr'].astype(float).values
    rb = d['response_bin'].astype(float).values
    rc = np.clip(rb, 0.5, 99.5) / 100.0
    d['_x'] = np.log(rc) - np.log1p(-rc)
    d['_cell'] = d['prior'].astype(float).round(4).astype(str) + '_' + d['log_lr'].astype(float).round(4).astype(str)
    inf_p = np.abs(L0) > 1e-6
    inf_e = np.abs(ll) > 1e-6
    conflict = inf_p & inf_e & (np.sign(L0) != np.sign(ll))
    big = conflict & (np.abs(ll) >= 1.9) & (np.abs(L0) >= 1.9)
    ctrl = ~conflict
    d['_big'] = big
    d['_ctrl'] = ctrl
    bins = np.arange(101, dtype=float)
    bc = np.clip(bins, 0.5, 99.5) / 100.0
    bx = np.log(bc) - np.log1p(-bc)
    thr = 1.8

    def cell_stats(cd):
        med = float(np.median(cd['response_bin'].astype(float).values))
        mc = min(max(med, 0.5), 99.5) / 100.0
        m = np.log(mc) - np.log1p(-mc)
        far = np.abs(cd['_x'].values - m) >= thr
        width = float(np.sum(np.abs(bx - m) >= thr))
        return int(far.sum()), len(cd), width

    vals = []
    for _, sd in d.groupby('subject_id'):
        cf = 0.0
        cw = 0.0
        for _, cd in sd[sd['_ctrl']].groupby('_cell'):
            nf, n, w = cell_stats(cd)
            cf += nf
            cw += n * w
        bf = 0.0
        bn = 0.0
        bw = 0.0
        for _, cd in sd[sd['_big']].groupby('_cell'):
            nf, n, w = cell_stats(cd)
            bf += nf
            bn += n
            bw += n * w
        if bn < 1 or cw <= 0:
            continue
        dens = cf / cw
        obs = bf / bn
        exp = dens * (bw / bn)
        v = obs - exp
        if np.isfinite(v):
            vals.append(v)
    if len(vals) == 0:
        return 0.0
    return float(np.mean(vals))
```

**Metric rationale:** Conflict-specific excess of far-from-core responses, with the stimulus-independent off-task floor cancelled within each subject. In the large-conflict cells (prior 0.9 vs LLR -2.2, prior 0.97 vs -3.48, prior 0.03 vs +3.48), the advocated cue-selection mixture puts a fraction rho of trials at the single-cue anchors, for example near 95% and near 10% when x=3.48. These anchors sit about 2-3 logit units from the subject's integrated core, so they land outside the band within 1.8 logits of the subject's cell median. The competing theory has no such mixture, so its only far mass is the uniform guessing component lambda (plus reversals and tails), and that has the same density per response bin in every cell.

The metric is built per subject in three steps:
1. Estimate the per-bin density of far responses from all non-conflict cells: baseline, prior-only, evidence-only and congruent cells, where the advocated theory predicts no anchor mass.
2. Multiply that density by the width, in bins, of the far region in each conflict cell. This gives the expected uniform-floor mass there.
3. Subtract that expectation from the observed far fraction in the conflict cells, then average across subjects.

Predictions:
- Competing: about 0 or slightly negative, because uniform mass cancels exactly. Reversal reports at the mirror of extreme single-source cells inflate the control density slightly.
- Advocated: about rho*0.7, roughly 0.12-0.15.

Robustness:
- The 1.8-logit threshold keeps Gaussian core tails small.
- Wide bin regions and subject-specific medians make the metric robust to 5%/10% rounding and to individual alpha/beta.
- Subtracting each subject's own floor removes lambda's large between-subject variance, which sank earlier count-based metrics.

| theory | predicted (var) |
|---|---|
| pi_4 | 0.0792 (var=0.0152) |
| pi_5 | -0.0198 (var=0.0099) |
| pi_1 | 0.0003 (var=0.0116) |
| pi_2 | -0.0341 (var=0.0092) |
| pi_3 | 0.0417 (var=0.0100) |
| pi_6 | -0.0395 (var=0.0056) |
| ground truth | -0.0223 (var=0.0048) |

Welch p (pi_4 vs pi_5): 0.00308

