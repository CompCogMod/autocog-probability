# metric_exp00_attempt_00

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
  trial 2: prior=0.300  P(e|H)=0.5  P(e|not-H)=0.5  n_signals=1  log LR=+0.00
  trial 3: prior=0.700  P(e|H)=0.5  P(e|not-H)=0.5  n_signals=1  log LR=+0.00
  trial 4: prior=0.500  P(e|H)=0.8  P(e|not-H)=0.2  n_signals=2  log LR=+1.39
  trial 5: prior=0.500  P(e|H)=0.8  P(e|not-H)=0.2  n_signals=4  log LR=+1.39
  trial 6: prior=0.300  P(e|H)=0.8  P(e|not-H)=0.2  n_signals=2  log LR=+1.39
  trial 7: prior=0.300  P(e|H)=0.8  P(e|not-H)=0.2  n_signals=3  log LR=+1.39
  trial 8: prior=0.300  P(e|H)=0.8  P(e|not-H)=0.2  n_signals=4  log LR=+1.39
  trial 9: prior=0.700  P(e|H)=0.2  P(e|not-H)=0.8  n_signals=2  log LR=-1.39
  trial 10: prior=0.700  P(e|H)=0.2  P(e|not-H)=0.8  n_signals=4  log LR=-1.39
  trial 11: prior=0.100  P(e|H)=0.8  P(e|not-H)=0.2  n_signals=2  log LR=+1.39
  trial 12: prior=0.900  P(e|H)=0.8  P(e|not-H)=0.2  n_signals=2  log LR=+1.39

**Rationale:** Goal: a 'fixed anchor vs moving mirror' test of where extra, off-pattern response mass lands in two-cue cells. LLR is assumed to be n*log(ph/pnh).

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
(none yet)

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    import pandas as pd\n    d = data.copy()\n    d['L0'] = np.round(np.log(d['prior'].clip(1e-6, 1-1e-6)) - np.log1p(-d['prior'].clip(1e-6, 1-1e-6)), 4)\n    d['llr'] = np.round(d['log_lr'].astype(float), 4)\n    # eligible cells: single-source (exactly one informative cue) or congruent two-cue cells\n    p_inf = d['L0'].abs() > 1e-6\n    e_inf = d['llr'].abs() > 1e-6\n    single = p_inf ^ e_inf\n    congruent = p_inf & e_inf & (np.sign(d['L0']) == np.sign(d['llr']))\n    d = d[single | congruent]\n    if len(d) == 0:\n        return 0.0\n    keys = ['subject_id', 'prior', 'p_signal_h', 'p_signal_not_h', 'n_signals']\n    med = d.groupby(keys)['response_bin'].transform('median')\n    r = d['response_bin'].astype(float)\n    hi = med >= 60\n    lo = med <= 40\n    elig = hi | lo\n    if elig.sum() == 0:\n        return 0.0\n    wrong = (hi & (r <= 40)) | (lo & (r >= 60))\n    return float(wrong[elig].mean())\n",
  "rationale": "Measures the rate of 'wrong-side' reports (response on the opposite side of 50%, beyond a 10-point margin, from the subject's own median for that cell) restricted to cells where every response route of the advocated model lands on the same side: single-source cells (prior 0.3/0.7 with no evidence; prior 0.5 with evidence) and the congruent cell (prior 0.9 + pro-H evidence). In these cells the advocated cue-selection mixture cannot produce opposite-side responses (prior-only and evidence-only modes are on the same side as the integrated answer, and single-source cells have no mixture), so wrong-side mass comes only from Gaussian logit tails plus the small uniform lapse epsilon (0-0.15, ~0.4*eps \u2248 0.03). The competing theory has a large stimulus-independent off-task component lambda (0.1-0.42), dominated by uniform guessing (~40% of which lands on the wrong side) plus polarity reversals (100-R, always wrong side), giving \u2248 0.26*(0.41+f_rev) \u2248 0.12-0.13 extra wrong-side mass. Expected: advocated \u2248 0.05-0.06, competing \u2248 0.15-0.17. Using the subject's own cell median makes it robust to individual alpha/beta distortions and LLR-scaling ambiguities; the 40/60 thresholds are coarse enough to be unaffected by 5%/10% rounding; pooling ~25 eligible trials per subject keeps per-subject variance manageable."
}
```

## Usage

```json
{
  "input_tokens": 8451,
  "output_tokens": 4714,
  "reasoning_tokens": 3667,
  "provider": "Claude Platform on AWS"
}
```
