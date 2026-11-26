# metric_exp00_attempt_01

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
**Description:** Noisy log-odds encoding with Weber-scaled noise and Bayesian shrinkage. Distortion in probability updating comes from noise, not from fixed mis-weighting.

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

**Parameters:**
- s0: [0.1, 0.5]
- w: [0.05, 0.35]
- lambda_p: [1.3, 2.7]
- lambda_e: [0.7, 2.5]
- sigma_r: [0.05, 0.4]
- epsilon: [0.04, 0.35]

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
[0] rationale: Source-dependent variability at matched (moderate) belief. Under the advocated linear log-odds model, response noise is a constant sigma in log-odds regardless of the stimulus, so the within-cell robust spread (median absolute deviation from each subject's cell median, in log-odds) in the conflict cells (prior .99/.95/.90 vs opposing LLR -4.6/-2.94/-2.2) equals that in the low-magnitude cells (50/50 at prior .5, LLR .85 at prior .5, prior .7 alone): log ratio ~0 (slightly below due to ceiling clipping since conflict medians sit ~85-90%). Under the Weber-encoding competing theory, conflict cells inherit encoding noise from two large-magnitude components (sd = s0 + w|x| each), giving a log-odds SD around 0.9-1.3 vs ~0.4-0.5 in low-magnitude cells, so log ratio ~0.6-1.0. Using log-odds and per-cell medians removes mean differences; the median-based pooled MAD over 12 trials per group per subject resists lapses and rounding outliers, and the log ratio stabilizes between-subject variance. Works on pooled data and single-subject slices.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    d = data.copy()
    r = np.clip(d['response'].astype(float).values, 0.005, 0.995)
    d['lo'] = np.log(r) - np.log1p(-r)
    pr = np.clip(d['prior'].astype(float).values, 1e-6, 1 - 1e-6)
    L0 = np.log(pr) - np.log1p(-pr)
    llr = d['log_lr'].astype(float).values
    d['cell'] = np.round(d['prior'].astype(float), 3).astype(str) + '_' + np.round(d['log_lr'].astype(float), 2).astype(str)
    conflict = (llr < -1.0) & (L0 > 1.0)
    low = (np.abs(L0) + np.abs(llr) < 1.0) & (llr >= 0)
    d['grp'] = np.where(conflict, 'c', np.where(low, 'l', 'x'))
    d = d[d['grp'] != 'x']
    if len(d) == 0:
        return 0.0
    med = d.groupby(['subject_id', 'cell'])['lo'].transform('median')
    d['dev'] = np.abs(d['lo'] - med)
    c = d.loc[d['grp'] == 'c', 'dev'].values
    l = d.loc[d['grp'] == 'l', 'dev'].values
    if len(c) == 0 or len(l) == 0:
        return 0.0
    mc = np.median(c) + 0.03
    ml = np.median(l) + 0.03
    return float(np.log(mc / ml))

outcome: self_sim=0.1140 (var=0.4338) adversary_sim=0.4197 (var=0.4975) welch_t=-1.584 p=0.1199 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    import pandas as pd\n    r = np.clip(data['response'].astype(float).values, 0.01, 0.99)\n    lo = np.log(r) - np.log1p(-r)\n    pr = data['prior'].astype(float).values\n    ll = data['log_lr'].astype(float).values\n    def cm(mask):\n        v = lo[mask]\n        if len(v) == 0:\n            return np.nan\n        return float(np.median(v))\n    num = 0.0\n    den = 0.0\n    for p in [0.7, 0.8, 0.9, 0.95, 0.99]:\n        L = np.log(p / (1 - p))\n        E = cm((np.abs(pr - 0.5) < 1e-3) & (np.abs(ll - L) < 0.05))\n        P = cm((np.abs(pr - p) < 1e-3) & (np.abs(ll) < 0.05))\n        C = cm((np.abs(pr - p) < 1e-3) & (np.abs(ll + L) < 0.05))\n        if np.isnan(P):\n            continue\n        if not np.isnan(E):\n            num += E\n            den += P\n        if not np.isnan(C):\n            num += (P - C)\n            den += P\n    if abs(den) < 1e-6:\n        return 0.0\n    return float(num / den)\n",
  "rationale": "Estimates the relative weight of evidence vs. prior (beta/alpha) at matched input magnitudes, using median response log-odds. For each magnitude x in {0.85,1.39,2.2,2.94,4.6}, the evidence-only cell (prior .5, LLR x) gives beta*x, the prior-only cell (prior expit(x), uninformative evidence) gives alpha*x, and conflict cells (prior expit(x), LLR -x) give (alpha-beta)*x, so P-C is a second estimate of the evidence contribution. Metric = (sum E + sum(P-C)) / (sum of matching P). Both theories are additive in log-odds, so this ratio cleanly reads off the relative evidence/prior weight. Advocated theory: weights are fixed with strong conservatism relative to prior use (beta ~0.12-0.88 vs alpha ~0.73-1.04), giving a ratio typically ~0.5-0.6 and conflict cells resolving clearly toward H. Competing theory: both gains come from the same shrinkage law at matched magnitude (k = lambda^2/(lambda^2+(s0+w|x|)^2)), with lambda_e and lambda_p both large relative to encoding sd, so effective weights are near-equal (ratio ~0.8-0.95) and conflict cells sit near 50-65%. Using sums of cell medians across 8 cell pairs (pooled over all trials of a cell) is robust to lapses and rounding, and the ratio is a per-subject quantity with moderate between-subject variance, unlike the previous spread-based metric whose noise swamped the effect."
}
```

## Usage

```json
{
  "input_tokens": 8900,
  "output_tokens": 4520,
  "reasoning_tokens": 3477,
  "provider": "Claude Platform on AWS"
}
```
