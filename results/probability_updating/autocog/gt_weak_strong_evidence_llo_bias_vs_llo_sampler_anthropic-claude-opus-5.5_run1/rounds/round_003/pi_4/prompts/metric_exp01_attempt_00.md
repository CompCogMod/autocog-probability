# metric_exp01_attempt_00

## System Prompt

You are a psychology researcher proposing a metric in the Probability Updating domain.

Your goal is adversarial: propose a metric that DISCRIMINATES the two theories — i.e., its value, computed on data simulated under your advocated theory, should be as far as possible from its value computed on data simulated under the competing theory. The direction of the gap does not matter; what matters is that the two theories produce visibly different numbers on this metric. The metric is computed on the data collected from the experimental design provided in the prompt. Produce a metric where you're prediction will be much more accurate than the competing theory's prediction on human data.

Your metric is a Python function

    metric(data: pd.DataFrame) -> float

Available imports inside `metric`:
- numpy as np
- pandas as pd

The system evaluates your metric in two ways and reports the pair as `point_estimate (var=between_subject_variance)` everywhere downstream:
- `point_estimate` is `metric(data)` applied to the FULL pooled DataFrame (all subjects together) — the canonical scalar;
- `between_subject_variance` is the population variance (`ddof=0`) of `metric(subj_df)` re-applied per `subject_id`, summarising how stable the metric is across subjects. If your metric only makes sense on multi-subject data this will fall back to `n/a` and the metric is rejected (the acceptance test below cannot run without it). Prefer metrics that work both on the pooled DataFrame and on a single subject's slice.

Acceptance rule: the system simulates each theory and runs Welch's two-sample t-test on `(point_estimate_self, between_subject_variance_self, N)` vs. `(point_estimate_adv, between_subject_variance_adv, N)`, where N is the number of HUMAN subjects the experiment will actually be run with (a fixed small number, currently 25). Your metric is admitted iff the two-sided p-value is below the significance level (currently alpha=0.01). Implication: a large between-theory gap is NOT enough — if either theory's metric is also highly variable across subjects, N humans won't reliably distinguish them and the metric will be rejected. Aim for contrasts that are both large in mean AND tight per subject.

Do NOT propose metrics that are trivially true for your theory.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects are told a prior probability that hypothesis H is true (e.g. that a ball was drawn from urn A rather than urn B), then shown evidence whose probability under H and under not-H is known (e.g. the colours of several draws, with the urns' compositions stated). They report the probability that H is true on a 0-100% scale.

Each subject completes about 60 trials in random order. On each trial `predict(parameters, stimulus, history)` receives `stimulus`, a dict with keys `prior`: P(H) before this step's evidence: the design prior if there is one, else the subject's own reported prior; `prior_is_objective`: 1 if `prior` comes from the design, 0 if it is the subject's report; `log_lr`: ln[P(evidence | H) / P(evidence | not-H)] of this step's evidence; `n_signals`: number of signals (draws, informants, clues) in this step's evidence; `signal_dir`: +1 evidence favours H, -1 favours not-H, 0 uninformative; `bayes`: Bayesian posterior P(H) given `prior` and `log_lr`. Values are floats for one trial; the same code is also called with equal-length numpy arrays (a batch of trials) when fitting to human data, so write it to broadcast. It must return probabilities over the 101 response bins 0%, 1%, ..., 100%: shape (101,) for one trial, (n, 101) for a batch. `history` is the subject's earlier trials (dict of lists) or None when fitting a batch. As in human data, every response then passes through a shared reporting stage that does not depend on the theory: with probability 0.13 it is rounded to the nearest multiple of 10%, otherwise with probability 0.10 to the nearest multiple of 5%. Theories should not model this rounding themselves, and metrics should not rely on exact values that rounding would blur.

## CHOSEN EXPERIMENTAL DESIGN
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


## DATA SCHEMA
Your metric receives a tidy per-trial pandas DataFrame stacking all subjects (rows grouped by `subject_id`, in trial order). Columns:
- subject_id: simulated or human subject identifier
- trial: 0-based trial index within the subject
- prior: prior P(H) stated to the subject
- p_signal_h: probability of the observed evidence if H is true
- p_signal_not_h: probability of the observed evidence if H is false
- n_signals: number of signals (draws) in the evidence
- log_lr: ln(p_signal_h / p_signal_not_h)
- bayes: Bayesian posterior P(H | evidence)
- response: reported P(H) in [0, 1] (response_bin / 100)
- response_bin: reported percentage as an integer 0-100

## IMPLEMENTATION GUARDRAILS
Any column in the schema above whose description names a list / tuple / np.ndarray (i.e. a per-trial sequence of values) holds non-scalar cells. Those cells are NOT hashable, so operations that hash row values fail with `TypeError: unhashable type: 'list'`. Treating `<seq_col>` as a placeholder for any such sequence-valued column:
- Avoid: `data.groupby('<seq_col>')`, `data['<seq_col>'].value_counts()`,     `data['<seq_col>'].nunique()`, `data['<seq_col>'].unique()` (returns     an object array but downstream `set()` / `in dict` will crash),     `set(data['<seq_col>'])`, `data['<seq_col>'].isin([...])` against list     values, or using a list cell as a dict key.
- If you need a hashable surrogate, project to one first, e.g.:
    - `data['<seq_col>_key'] = data['<seq_col>'].apply(tuple)` then group by `<seq_col>_key`
    - `data['<seq_col>_str'] = data['<seq_col>'].apply(lambda x: ''.join(map(str, x)))`
    Scalar columns (ints, floats, strings like `subject_id`, integer     responses, etc.) hash fine and can be used directly.
- Generator expressions inside function calls like `map()` or `join()` MUST be     parenthesized. For example:
    - WRONG: `map(str, int(v) for v in x)` → SyntaxError
    - RIGHT: `map(str, (int(v) for v in x))` or use a list comp: `[str(int(v)) for v in x]`
- Always verify your code is syntactically valid Python before returning it.

## METRICS YOU ALREADY TRIED AND FAILED ON
Each entry below is a metric you previously proposed in this round that did NOT discriminate the two theories at the human sample size — either it errored, its between-subject variance was unavailable, or Welch's t-test on `(self mean, self var, N)` vs. `(adv mean, adv var, N)` returned p ≥ alpha. The `outcome` line is the simulation result (means, between-subject variances, t-statistic and p-value at the human N) on the same `data_self` / `data_adv` your next metric will be evaluated on. Use the numbers to see where your hypothesised contrast collapsed — small mean gap, large per-subject variance, or both — and propose something qualitatively different. Don't repeat the same idea with cosmetic tweaks.
[0] rationale: Measures the rate of 'wrong-side' reports (response on the opposite side of 50%, beyond a 10-point margin, from the subject's own median for that cell) restricted to cells where every response route of the advocated model lands on the same side: single-source cells (prior 0.3/0.7 with no evidence; prior 0.5 with evidence) and the congruent cell (prior 0.9 + pro-H evidence). In these cells the advocated cue-selection mixture cannot produce opposite-side responses (prior-only and evidence-only modes are on the same side as the integrated answer, and single-source cells have no mixture), so wrong-side mass comes only from Gaussian logit tails plus the small uniform lapse epsilon (0-0.15, ~0.4*eps ≈ 0.03). The competing theory has a large stimulus-independent off-task component lambda (0.1-0.42), dominated by uniform guessing (~40% of which lands on the wrong side) plus polarity reversals (100-R, always wrong side), giving ≈ 0.26*(0.41+f_rev) ≈ 0.12-0.13 extra wrong-side mass. Expected: advocated ≈ 0.05-0.06, competing ≈ 0.15-0.17. Using the subject's own cell median makes it robust to individual alpha/beta distortions and LLR-scaling ambiguities; the 40/60 thresholds are coarse enough to be unaffected by 5%/10% rounding; pooling ~25 eligible trials per subject keeps per-subject variance manageable.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    d = data.copy()
    d['L0'] = np.round(np.log(d['prior'].clip(1e-6, 1-1e-6)) - np.log1p(-d['prior'].clip(1e-6, 1-1e-6)), 4)
    d['llr'] = np.round(d['log_lr'].astype(float), 4)
    # eligible cells: single-source (exactly one informative cue) or congruent two-cue cells
    p_inf = d['L0'].abs() > 1e-6
    e_inf = d['llr'].abs() > 1e-6
    single = p_inf ^ e_inf
    congruent = p_inf & e_inf & (np.sign(d['L0']) == np.sign(d['llr']))
    d = d[single | congruent]
    if len(d) == 0:
        return 0.0
    keys = ['subject_id', 'prior', 'p_signal_h', 'p_signal_not_h', 'n_signals']
    med = d.groupby(keys)['response_bin'].transform('median')
    r = d['response_bin'].astype(float)
    hi = med >= 60
    lo = med <= 40
    elig = hi | lo
    if elig.sum() == 0:
        return 0.0
    wrong = (hi & (r <= 40)) | (lo & (r >= 60))
    return float(wrong[elig].mean())

outcome: self_sim=0.0596 (var=0.0035) adversary_sim=0.0986 (var=0.0031) welch_t=-2.395 p=0.0206 (N=25, alpha=0.01) -> reject

[1] rationale: Estimates, per subject, the weight of a stimulus-independent uniform 'off-task' component by EM, fitting each response as a mixture of (a) a Gaussian in log-odds around the subject's own cell median (shared sigma, floored at 0.25 to tolerate rounding) and (b) a uniform over the 101 bins. It uses ONLY cells where at most one cue is informative (baseline 0.5/no evidence, prior 0.3 or 0.7 with uninformative evidence, prior 0.5 with 0.8/0.2 evidence; 25 trials per subject). In these cells the advocated cue-selection mixture is switched off by construction (both-informative flag = 0), so its only non-Gaussian mass is the small lapse epsilon (0-0.15). The competing theory's lambda (0.1-0.42) is stimulus-independent and dominated by uniform guessing, plus reversal reports at 100-R that land far from the median in the 30/70 cells, so it appears in single-source cells with full strength. The previous wrong-side count used only ~40% of the response range as its 'outlier' region. The EM likelihood instead uses every response's distance from the engaged core, which gives a far more efficient estimate of the uniform weight. Expected: advocated about 0.04-0.08; competing about 0.18-0.28. Medians are taken per subject and cell, so the metric is robust to individual alpha/beta and to how n_signals is scaled. It is computed per subject and averaged, so it works on pooled data and on a single subject's slice. Exact rounding values are not relied on.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    d = data.copy()
    pr = d['prior'].astype(float).clip(1e-6, 1 - 1e-6)
    L0 = np.log(pr) - np.log1p(-pr)
    llr = d['log_lr'].astype(float)
    single = (L0.abs() < 1e-6) | (llr.abs() < 1e-6)
    d = d[single.values]
    if len(d) == 0:
        return 0.0
    keys = ['prior', 'p_signal_h', 'p_signal_not_h', 'n_signals']

    def fit(sd):
        r = sd['response_bin'].astype(float).values / 100.0
        med = sd.groupby(keys)['response_bin'].transform('median').astype(float).values / 100.0
        rc = np.clip(r, 0.005, 0.995)
        mc = np.clip(med, 0.005, 0.995)
        x = np.log(rc) - np.log1p(-rc)
        m = np.log(mc) - np.log1p(-mc)
        jac = 0.01 / (rc * (1.0 - rc))
        dev = x - m
        s = max(1.4826 * float(np.median(np.abs(dev))), 0.25)
        w = 0.2
        for _ in range(150):
            g = (1.0 - w) * np.exp(-0.5 * (dev / s) ** 2) / (s * np.sqrt(2.0 * np.pi)) * jac
            u = w / 101.0
            post = u / (g + u + 1e-300)
            w = float(np.clip(post.mean(), 1e-4, 0.95))
            wt = 1.0 - post
            s = float(np.sqrt(np.sum(wt * dev ** 2) / max(np.sum(wt), 1e-9)))
            s = max(s, 0.25)
        return w

    ws = []
    for _, sd in d.groupby('subject_id'):
        if len(sd) >= 5:
            ws.append(fit(sd))
    if len(ws) == 0:
        return 0.0
    return float(np.mean(ws))

outcome: self_sim=0.1200 (var=0.0183) adversary_sim=0.2132 (var=0.0307) welch_t=-2.106 p=0.04077 (N=25, alpha=0.01) -> reject

[2] rationale: Per-subject EM mixture that pits the two theories' sources of off-core mass against each other, using all 60 trials rather than one subset. Each response, in logit space with a bin Jacobian, is modelled as a mixture of four parts. (1) A core Gaussian at the integrated answer. In two-cue cells this is alpha_hat*logit(prior) + beta_hat*LLR, with alpha_hat and beta_hat estimated from the subject's own prior-only and evidence-only cells. In single-source cells it is the cell median. (2) In two-cue cells only, 'anchor' Gaussians at the subject's prior-only location (alpha_hat*L0) and evidence-only location (beta_hat*LLR). (3) A stimulus-independent uniform over 101 bins. (4) A noise width sigma shared across all trials, so the anchors cannot be mistaken for a wider core: sigma is pinned by the single-source cells, where no anchors exist. The metric is w_anchor - w_uniform, averaged over subjects. Advocated prediction: w_anchor ≈ rho (cue-selection on two-cue trials) and w_uniform ≈ epsilon (small), so the metric is positive. Competing prediction: there is no cue-selection, so w_anchor ≈ 0, while the large off-task lambda (dominated by uniform guessing, plus reversals that also fall in the uniform) gives w_uniform ≈ lambda, so the metric is negative. The two earlier failed metrics measured only the uniform-weight half of this contrast, giving gaps of about 0.04-0.09. Taking the difference adds the rho signal on top, roughly doubling the expected gap. Using all trials also shrinks estimation noise. Logit-scale Gaussians with a sigma floor of 0.3 and coarse mixture components keep the metric insensitive to the 5%/10% rounding. It is computed per subject, so it works on a single subject's slice as well as on pooled data.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    d = data.copy()
    pr = d['prior'].astype(float).clip(1e-6, 1 - 1e-6).values
    L0 = np.log(pr) - np.log1p(-pr)
    llr = d['log_lr'].astype(float).values
    r = np.clip(d['response_bin'].astype(float).values / 100.0, 0.005, 0.995)
    d['_L0'] = L0
    d['_llr'] = llr
    d['_x'] = np.log(r) - np.log1p(-r)
    d['_jac'] = 0.01 / (r * (1.0 - r))
    keys = ['prior', 'p_signal_h', 'p_signal_not_h', 'n_signals']

    def npdf(x, m, s):
        return np.exp(-0.5 * ((x - m) / s) ** 2) / (s * np.sqrt(2.0 * np.pi))

    def fit(sd):
        x = sd['_x'].values
        jac = sd['_jac'].values
        l0 = sd['_L0'].values
        ll = sd['_llr'].values
        pi = np.abs(l0) > 1e-6
        ei = np.abs(ll) > 1e-6
        both = pi & ei
        single = ~both
        ponly = pi & ~ei
        eonly = ei & ~pi
        if single.sum() < 3 or both.sum() < 2:
            return None
        alpha = float(np.median(x[ponly] / l0[ponly])) if ponly.sum() > 0 else 0.85
        beta = float(np.median(x[eonly] / ll[eonly])) if eonly.sum() > 0 else 0.7
        alpha = float(np.clip(alpha, 0.2, 1.5))
        beta = float(np.clip(beta, 0.1, 1.5))
        med = sd.groupby(keys)['_x'].transform('median').values
        core = np.where(both, alpha * l0 + beta * ll, med)
        Ap = alpha * l0
        Ae = beta * ll
        dev0 = (x - core)[single]
        s = max(1.4826 * float(np.median(np.abs(dev0 - np.median(dev0)))), 0.3)
        wu, wa, q = 0.1, 0.1, 0.5
        u = 1.0 / 101.0
        bf = both.astype(float)
        for _ in range(200):
            g = npdf(x, core, s) * jac
            gp = npdf(x, Ap, s) * jac
            ge = npdf(x, Ae, s) * jac
            wc = np.where(both, 1.0 - wu - wa, 1.0 - wu)
            cp = bf * wa * q * gp
            ce = bf * wa * (1.0 - q) * ge
            cc = wc * g
            cu = wu * u * np.ones_like(x)
            den = cc + cp + ce + cu + 1e-300
            pc = cc / den
            pp = cp / den
            pe = ce / den
            pu = cu / den
            wu = float(np.clip(pu.mean(), 1e-4, 0.9))
            pa = (pp + pe)[both]
            wa = float(np.clip(pa.mean(), 1e-4, 0.9))
            if wu + wa > 0.98:
                sc = 0.98 / (wu + wa)
                wu *= sc
                wa *= sc
            q = float(np.clip(pp[both].sum() / max(pa.sum(), 1e-12), 0.02, 0.98))
            num = np.sum(pc * (x - core) ** 2 + pp * (x - Ap) ** 2 + pe * (x - Ae) ** 2)
            dd = np.sum(pc + pp + pe)
            s = max(float(np.sqrt(num / max(dd, 1e-9))), 0.3)
        return wa - wu

    vals = []
    for _, sd in d.groupby('subject_id'):
        v = fit(sd)
        if v is not None and np.isfinite(v):
            vals.append(v)
    if len(vals) == 0:
        return 0.0
    return float(np.mean(vals))

outcome: self_sim=0.1769 (var=0.0573) adversary_sim=0.0248 (var=0.0651) welch_t=+2.173 p=0.03477 (N=25, alpha=0.01) -> reject

[3] rationale: Earlier attempts each measured one half of the contrast with ad hoc counts or an unstable EM, and got gaps of only about 2 SE. This metric uses the statistically optimal discriminant instead: a per-subject, per-trial log-likelihood ratio between the two theories' response models, each fit by a small grid search within its own stated parameter ranges. The value is averaged over subjects, so it works identically on a single subject's slice.

Shared pieces for both models:
- Logit-Gaussian engaged core at the subject's own cell median, with a common sigma.
- The theory-independent reporting stage modelled exactly: each response is scored as a 0.783/0.087/0.13 mixture of unrounded, 5%-rounded and 10%-rounded bin masses. Rounding spikes (e.g. 90, 100 in the prior-0.9 cell) therefore do not masquerade as outliers.

Advocated model:
- Adds, in two-cue cells only, anchor modes. The prior-only anchor is at alpha_hat*logit(prior) and the evidence-only anchor at beta_hat*LLR, both estimated from the subject's own single-source cells.
- Free weights: rho in [0, 0.4] and q.
- Small uniform lapse eps in [0, 0.15].

Competing model:
- Stimulus-independent off-task mass lambda in [0.1, 0.42], spread uniformly over all cells.
- A polarity-reversal mode at the mirror of the engaged answer (logit -core), with f_rev in [0, 0.15].

Why the two theories should separate:
- On advocated-generated data, two-cue cells contain excess mass at cue-specific fixed anchors that the competing model can only absorb by widening sigma or adding uniform mass. Single-source cells are clean, which the competing model's forced lambda >= 0.1 penalises. The ratio should be positive.
- On competing-generated data, uniform contamination appears in every cell, including single-source ones, at levels the advocated eps cap cannot reach. Mirror responses (e.g. near 6% in the prior-0.9 cell) are explained only by the competing model. The ratio should be near zero or negative.

Because the metric pools both signatures (anchor mass and global uniform/reversal mass) over all 60 trials with likelihood weights, it should beat the separate counts in signal-to-noise. Its extra flexibility (two extra parameters) gives only a small positive bias.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    import math
    verf = np.vectorize(math.erf, otypes=[float])

    def Phi(z):
        return 0.5 * (1.0 + verf(z / np.sqrt(2.0)))

    def zedge(e):
        ec = np.clip(e, 0.5, 99.5) / 100.0
        z = np.log(ec) - np.log1p(-ec)
        z = np.where(e < 0.5, -np.inf, z)
        z = np.where(e > 99.5, np.inf, z)
        return z

    d = data.copy()
    d['_cell'] = (d['prior'].astype(float).round(4).astype(str) + '_' + d['log_lr'].astype(float).round(4).astype(str) + '_' + d['n_signals'].astype(float).round(2).astype(str))
    s_grid = [0.3, 0.4, 0.5, 0.6, 0.75, 0.9, 1.1, 1.3]
    rho_grid = [0.0, 0.1, 0.2, 0.3, 0.4]
    q_grid = [0.3, 0.55, 0.8]
    eps_grid = [0.0, 0.03, 0.06, 0.1, 0.15]
    lam_grid = [0.1, 0.15, 0.2, 0.25, 0.3, 0.36, 0.42]
    fr_grid = [0.0, 0.05, 0.1, 0.15]

    def fit(sd):
        r = sd['response_bin'].astype(float).values
        n = len(r)
        if n < 5:
            return None
        rc = np.clip(r / 100.0, 0.005, 0.995)
        x = np.log(rc) - np.log1p(-rc)
        pr = sd['prior'].astype(float).clip(1e-6, 1 - 1e-6).values
        L0 = np.log(pr) - np.log1p(-pr)
        ll = sd['log_lr'].astype(float).values
        pi = np.abs(L0) > 1e-6
        ei = np.abs(ll) > 1e-6
        b = (pi & ei).astype(float)
        ponly = pi & ~ei
        eonly = ei & ~pi
        med = pd.Series(x).groupby(sd['_cell'].values).transform('median').values
        a_hat = float(np.clip(np.median(x[ponly] / L0[ponly]), 0.2, 1.6)) if ponly.sum() > 0 else 0.85
        b_hat = float(np.clip(np.median(x[eonly] / ll[eonly]), 0.1, 1.6)) if eonly.sum() > 0 else 0.65
        Ap = a_hat * L0
        Ae = b_hat * ll
        core = med
        i5 = (np.round(r) % 5 == 0).astype(float)
        i10 = (np.round(r) % 10 == 0).astype(float)
        wts = [(0.5, 0.783 * np.ones(n)), (2.5, 0.087 * i5), (5.0, 0.13 * i10)]
        edges = [(zedge(r - h), zedge(r + h), w, (np.minimum(r + h, 100.5) - np.maximum(r - h, -0.5)) / 101.0) for h, w in wts]
        U = np.zeros(n)
        for za, zb, w, wid in edges:
            U = U + w * wid

        def comb(mu, s):
            out = np.zeros(n)
            for za, zb, w, wid in edges:
                out = out + w * np.clip(Phi((zb - mu) / s) - Phi((za - mu) / s), 0.0, None)
            return out

        best_a = -np.inf
        best_c = -np.inf
        for s in s_grid:
            Cc = comb(core, s)
            Cp = comb(Ap, s)
            Ce = comb(Ae, s)
            Cr = comb(-core, s)
            for rho in rho_grid:
                for q in q_grid:
                    eng = (1.0 - rho * b) * Cc + rho * b * (q * Cp + (1.0 - q) * Ce)
                    for eps in eps_grid:
                        P = (1.0 - eps) * eng + eps * U
                        LL = float(np.sum(np.log(np.maximum(P, 1e-300))))
                        if LL > best_a:
                            best_a = LL
            for lam in lam_grid:
                for fr in fr_grid:
                    P = (1.0 - lam) * Cc + lam * ((1.0 - fr) * U + fr * Cr)
                    LL = float(np.sum(np.log(np.maximum(P, 1e-300))))
                    if LL > best_c:
                        best_c = LL
        return (best_a - best_c) / n

    vals = []
    for _, sd in d.groupby('subject_id'):
        v = fit(sd)
        if v is not None and np.isfinite(v):
            vals.append(v)
    if len(vals) == 0:
        return 0.0
    return float(np.mean(vals))

outcome: self_sim=0.0165 (var=0.0014) adversary_sim=-0.0058 (var=0.0004) welch_t=+2.603 p=0.01313 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    import pandas as pd\n    d = data.copy()\n    pr = d['prior'].astype(float).clip(1e-6, 1 - 1e-6).values\n    L0 = np.log(pr) - np.log1p(-pr)\n    ll = d['log_lr'].astype(float).values\n    rb = d['response_bin'].astype(float).values\n    rc = np.clip(rb, 0.5, 99.5) / 100.0\n    d['_x'] = np.log(rc) - np.log1p(-rc)\n    d['_cell'] = d['prior'].astype(float).round(4).astype(str) + '_' + d['log_lr'].astype(float).round(4).astype(str)\n    inf_p = np.abs(L0) > 1e-6\n    inf_e = np.abs(ll) > 1e-6\n    conflict = inf_p & inf_e & (np.sign(L0) != np.sign(ll))\n    big = conflict & (np.abs(ll) >= 1.9) & (np.abs(L0) >= 1.9)\n    ctrl = ~conflict\n    d['_big'] = big\n    d['_ctrl'] = ctrl\n    bins = np.arange(101, dtype=float)\n    bc = np.clip(bins, 0.5, 99.5) / 100.0\n    bx = np.log(bc) - np.log1p(-bc)\n    thr = 1.8\n\n    def cell_stats(cd):\n        med = float(np.median(cd['response_bin'].astype(float).values))\n        mc = min(max(med, 0.5), 99.5) / 100.0\n        m = np.log(mc) - np.log1p(-mc)\n        far = np.abs(cd['_x'].values - m) >= thr\n        width = float(np.sum(np.abs(bx - m) >= thr))\n        return int(far.sum()), len(cd), width\n\n    vals = []\n    for _, sd in d.groupby('subject_id'):\n        cf = 0.0\n        cw = 0.0\n        for _, cd in sd[sd['_ctrl']].groupby('_cell'):\n            nf, n, w = cell_stats(cd)\n            cf += nf\n            cw += n * w\n        bf = 0.0\n        bn = 0.0\n        bw = 0.0\n        for _, cd in sd[sd['_big']].groupby('_cell'):\n            nf, n, w = cell_stats(cd)\n            bf += nf\n            bn += n\n            bw += n * w\n        if bn < 1 or cw <= 0:\n            continue\n        dens = cf / cw\n        obs = bf / bn\n        exp = dens * (bw / bn)\n        v = obs - exp\n        if np.isfinite(v):\n            vals.append(v)\n    if len(vals) == 0:\n        return 0.0\n    return float(np.mean(vals))\n",
  "rationale": "Conflict-specific excess of far-from-core responses, with the stimulus-independent off-task floor cancelled within each subject. In the large-conflict cells (prior 0.9 vs LLR -2.2, prior 0.97 vs -3.48, prior 0.03 vs +3.48), the advocated cue-selection mixture puts a fraction rho of trials at the single-cue anchors, for example near 95% and near 10% when x=3.48. These anchors sit about 2-3 logit units from the subject's integrated core, so they land outside the band within 1.8 logits of the subject's cell median. The competing theory has no such mixture, so its only far mass is the uniform guessing component lambda (plus reversals and tails), and that has the same density per response bin in every cell.\n\nThe metric is built per subject in three steps:\n1. Estimate the per-bin density of far responses from all non-conflict cells: baseline, prior-only, evidence-only and congruent cells, where the advocated theory predicts no anchor mass.\n2. Multiply that density by the width, in bins, of the far region in each conflict cell. This gives the expected uniform-floor mass there.\n3. Subtract that expectation from the observed far fraction in the conflict cells, then average across subjects.\n\nPredictions:\n- Competing: about 0 or slightly negative, because uniform mass cancels exactly. Reversal reports at the mirror of extreme single-source cells inflate the control density slightly.\n- Advocated: about rho*0.7, roughly 0.12-0.15.\n\nRobustness:\n- The 1.8-logit threshold keeps Gaussian core tails small.\n- Wide bin regions and subject-specific medians make the metric robust to 5%/10% rounding and to individual alpha/beta.\n- Subtracting each subject's own floor removes lambda's large between-subject variance, which sank earlier count-based metrics."
}
```

## Usage

```json
{
  "input_tokens": 15522,
  "output_tokens": 11888,
  "reasoning_tokens": 10216,
  "provider": "Claude Platform on AWS"
}
```
