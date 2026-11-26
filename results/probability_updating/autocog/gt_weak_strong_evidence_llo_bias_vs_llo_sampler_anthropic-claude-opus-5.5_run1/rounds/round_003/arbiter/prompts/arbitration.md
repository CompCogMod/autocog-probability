# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Probability Updating domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_5" and "pi_4") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_5" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_4" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects are told a prior probability that hypothesis H is true (e.g. that a ball was drawn from urn A rather than urn B), then shown evidence whose probability under H and under not-H is known (e.g. the colours of several draws, with the urns' compositions stated). They report the probability that H is true on a 0-100% scale.

## THEORY 1 — pi_5
Bounded deterministic integration plus heterogeneous off-task responding.

(1) Engaged integration is deterministic. Every engaged trial combines prior and evidence additively in log-odds with distorted weights: L = alpha*logit(prior) + beta*LLR. alpha<1 is base-rate neglect and beta<1 is conservatism. There is no cue-selection mixture, so conflict cells are unimodal in the engaged state.

(2) The output is softly bounded. The integrated value passes through L' = sign(L)*Delta*tanh((|L|/Delta)^kappa)^(1/kappa), with kappa>1 and Delta about 3.5–5.5. The bound keeps engaged reports off the exact endpoints, compresses raw-scale spread near the extremes, and leaves logit-scale variability roughly constant.

(3) Response noise is constant Gaussian noise sigma in log-odds around the bounded value.

(4) A non-engaged component has subject-specific total probability lambda (about 0.1–0.42). Its probability does not depend on the stimulus. It is dominated by uniform guessing over 0–100. It has two minor parts:
- a polarity-reversal lapse that reports the mirror value 100−R, i.e. P(not-H);
- a rare 50% 'don't know' report.
Off-pattern mass is therefore equal in single-source, congruent and conflict cells.

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


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    p = np.asarray(probs, dtype=np.float64)
    p = np.clip(p, 0.0, None)
    p /= p.sum()
    return int(np.random.choice(len(p), p=p))


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


## EXPERIMENT 1 (proposed by pi_5)

### DESIGN
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


### METRIC
Rationale:
Restrict to single-source cells (prior=0.5 or LLR=0: trials 1-7, 35 trials/subject). Here the competing theory has no strategy mixture, so its only off-pattern mass is a small uniform epsilon (<=0.15, mean ~0.075). The advocated theory has a stimulus-independent off-task component lambda (0.1-0.42, mean ~0.26), mostly uniform plus mirror reversals, which land far from the engaged answer. Metric: fraction of responses more than 35 percentage points from the subject's own median for that cell (median of 5 reps is robust to the lapses and absorbs alpha/beta/bound idiosyncrasies). Engaged log-odds noise (sigma<=0.9) rarely produces 35-point deviations, and the 35-point threshold is far wider than any rounding shift. Expected roughly 0.13-0.15 under the advocated theory vs 0.04-0.05 under the competing theory. Per-subject SD is about 0.07 vs 0.04, giving t of about 5 at N=25.

Source:
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


### RESULTS
- Predicted under pi_5 (simulated): 0.0954 (var=0.0029)
- Predicted under pi_4 (simulated): 0.0411 (var=0.0011)
- Observed on real data: 0.0869 (var=0.0029)

## EXPERIMENT 2 (proposed by pi_4)

### DESIGN
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


### METRIC
Rationale:
Conflict-specific excess of far-from-core responses, with the stimulus-independent off-task floor cancelled within each subject. In the large-conflict cells (prior 0.9 vs LLR -2.2, prior 0.97 vs -3.48, prior 0.03 vs +3.48), the advocated cue-selection mixture puts a fraction rho of trials at the single-cue anchors, for example near 95% and near 10% when x=3.48. These anchors sit about 2-3 logit units from the subject's integrated core, so they land outside the band within 1.8 logits of the subject's cell median. The competing theory has no such mixture, so its only far mass is the uniform guessing component lambda (plus reversals and tails), and that has the same density per response bin in every cell.

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

Source:
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


### RESULTS
- Predicted under pi_5 (simulated): -0.0198 (var=0.0099)
- Predicted under pi_4 (simulated): 0.0792 (var=0.0152)
- Observed on real data: -0.0223 (var=0.0048)

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
**Predicted under pi_5:** 0.1547 (var=0.0089)
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
**Predicted under pi_5:** -1.0986 (var=0.6023)
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
**Predicted under pi_5:** 0.7787 (var=0.0769)
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
**Predicted under pi_5:** 0.2102 (var=0.2816)
**Predicted under pi_4:** 0.1254 (var=0.2522)

### Experiment 7
**Design**
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

**Metric**
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

**Observed (real) value:** 0.1240 (var=0.0029)
**Predicted under pi_5:** 0.1127 (var=0.0038)
**Predicted under pi_4:** 0.0447 (var=0.0015)

### Experiment 8
**Design**
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

**Metric**
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

**Observed (real) value:** 0.1200 (var=0.0049)
**Predicted under pi_5:** 0.1200 (var=0.0065)
**Predicted under pi_4:** 0.0408 (var=0.0022)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across all eight experiments the two theories split along a clear seam, and pi_4 loses on the dimension the data treat as decisive.\n\n(1) Off-pattern mass. Here pi_5 is consistently right and pi_4 consistently wrong:\n- Exp1, single-source far-from-median mass: observed 0.087; pi_5 0.095; pi_4 0.041.\n- Exp7, opposite-side zone hits: observed 0.124; pi_5 0.113; pi_4 0.045.\n- Exp8, reports at or below 40 in high-prior cells with confirming evidence: observed 0.120; pi_5 0.120; pi_4 0.041.\n- Exp2, the conflict-specific excess that pi_4 itself proposed as its signature test: observed -0.022; pi_5 -0.020; pi_4 +0.079.\nSo there is substantial, stimulus-independent off-task mass of about 10-12% beyond the engaged core. There is no conflict-specific cue-selection mixture: the floor-corrected excess in big-conflict cells is zero or slightly negative, exactly as the deterministic-integration plus uniform-lambda account predicts. pi_4's two central claims (small epsilon only, bimodal single-cue modes in two-cue cells) are both falsified.\n\n(2) Variance structure. pi_4 does better here:\n- Exp4, raw-scale median absolute deviation in extreme cells versus mid cells: observed -1.50; pi_4 -1.50; pi_5 -1.10.\n- Exp6, logit-scale MAD for high- versus low-magnitude cells: observed 0.15; pi_4 0.125; pi_5 0.21.\nExtreme-cell reports are tighter than pi_5 predicts. pi_5's large uniform lambda, and possibly its sigma and bound settings, inflate typical deviations near the endpoints.\n\n(3) Additivity. Exp5's additivity ratio (observed 0.67) is overshot by both theories (0.78 and 0.77). Integration is more sub-additive or prior-dominated than either captures.\n\n(4) Endpoints. In Exp3 (endpoint stickiness) both theories are fine.\n\nOverall, pi_5 is the clear leader. pi_4 is degenerate because its distinctive mechanism, the strategy mixture, is refuted. But pi_5 has its own residual misses in Exp4, Exp5 and Exp6. The best move is to replace pi_4 with a stronger competitor that:\n- keeps what the data demand (stimulus-independent off-task mass, unimodal conflict cells);\n- offers a different account of the near-extreme tightness and of the sub-additive combination.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Replace pi_4 with a theory of 'sub-additive deterministic integration, salient-value response mapping, and trait-level off-task responding'.\n\n(a) Engaged integration is deterministic and unimodal in conflict cells. This is required by Exp2. It is sub-additive when both cues are informative: L = alpha*L0 + beta*LLR - gamma*sign-aware shrinkage. One implementation is |L_comb| = (|aL0|^p + |bLLR|^p)^(1/p) for congruent cues with p > 1, and a proportionally shrunk difference for conflict cues. Alternatively, use precision-weighted averaging with total weight below 1. Either version should bring the Exp5 ratio to about 0.65-0.70, whereas pi_5 is purely additive.\n\n(b) The response mapping is not plain logit-Gaussian. Reports are produced on a coarse salient-value grid (multiples of 5 and 10, plus 0, 50 and 100), and grid snapping gets stronger near the extremes. This makes extreme-cell raw deviations very small, giving an Exp4 ratio of about -1.5. It also yields the modest Exp6 logit-MAD increase of about 0.15 and the Exp3 endpoint ratio of about 0.14. This is a different mechanism from pi_5's tanh bound, so the two theories make distinguishable predictions: snapping implies discrete clustering and within-subject repeat identity at 95/99/100, whereas the tanh bound implies continuous compression.\n\n(c) Off-task responding has subject-specific mass lambda of about 0.08-0.2 and is equal across cell types, since the data require this. Its composition differs from pi_5:\n- mostly mirror or polarity reversals (100-R) and 50% 'don't know' reports;\n- a smaller uniform share than pi_5's.\nThis predicts the opposite-side mass seen in Exp7 and Exp8 while keeping the median deviations in extreme cells low.\n\nThe proposer should design discriminating experiments that:\n- test mirror-located versus uniform off-task mass (e.g., the mass in the 30-70% band versus the mirror band of extreme single-source cells);\n- test congruent sub-additivity against the bound, using moderate-magnitude congruent cells where the tanh bound is inactive."
}
```

## Usage

```json
{
  "input_tokens": 18683,
  "output_tokens": 3948,
  "reasoning_tokens": 2202,
  "provider": "Claude Platform on AWS"
}
```
