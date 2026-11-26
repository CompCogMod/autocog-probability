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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    d = data\n    mask = (d['prior'] >= 0.79) & (d['log_lr'] >= -1e-9)\n    sub = d.loc[mask]\n    if len(sub) == 0:\n        return float('nan')\n    return float(np.mean(sub['response_bin'].values <= 40))\n",
  "rationale": "Several design features cannot separate the theories. Both predict() functions use only log_lr, which is +1.39 regardless of n_signals, so the n-sweeps are effectively replicates. The sharpest remaining difference is lapse behaviour.\n\nThe metric uses only cells where every component of both theories lies firmly on the H side:\n- the prior-only cells at priors .9 and .999 (log-odds about 1.6-7);\n- the congruent cells at prior .8 with LLR +1.39, where the integrated, prior-anchored and evidence-only modes are all positive.\n\nThis gives 25 trials per subject. In these cells a report at or below 40% is essentially impossible under each model's main components. It arises almost only from the uniform lapse, which places 41/101 of its mass in that region.\n\nExpected values differ by lapse range:\n- Competing theory: epsilon spans 0.05-0.41 (mean about 0.23), so the opposite-side fraction is about 0.41*eps, roughly 0.09-0.10.\n- Advocated theory: epsilon spans 0-0.15 (mean about 0.075), so the fraction is about 0.03. It also has saturation: the bound only pulls high responses toward the high 90s and never toward the opposite side.\n\nThe low-beta prior-0.5 cells and the conflict cells are excluded. There, a near-50% center (competing) or a representativeness mode (advocated) could push mass below 40% for reasons other than lapse.\n\nThe metric is a simple proportion, so it is defined per subject. Between-subject variance comes mostly from binomial noise over 25 trials plus epsilon spread. That noise is small relative to the roughly 0.06 gap, which is about t\u22483.5-4 at N=25."
}
```

## Usage

```json
{
  "input_tokens": 7931,
  "output_tokens": 5412,
  "reasoning_tokens": 4669,
  "provider": "Claude Platform on AWS"
}
```
