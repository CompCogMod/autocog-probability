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


## ADVOCATED THEORY
**Description:** Probability distortion is a systematic bias applied when the judgment is formed: people combine prior and evidence additively in log-odds space, as Bayes prescribes, but with the wrong weights. The posterior log-odds equal alpha times the prior log-odds plus beta times the log likelihood ratio. alpha < 1 is base-rate neglect (the prior is underweighted), beta < 1 is conservatism (the evidence is underweighted) and beta > 1 is overreaction; alpha = beta = 1 recovers Bayes. Because the distortion is a deterministic transformation, the same stimulus always yields the same central judgment; trial-to-trial variability is a separate, constant response noise in log-odds space (sigma), plus a lapse rate epsilon. The theory therefore predicts that mean distortion and response variability are unrelated: changing how noisy responses are does not change their average, and the average is a fixed linear function of prior and evidence log-odds.


**Parameters:**
- alpha: [0.7321, 1.016]
- beta: [0.136, 0.7868]
- sigma: [0.1955, 1.207]
- epsilon: [0.04997, 0.4299]

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
**Description:** Both mechanisms operate: people misweight prior and evidence when they form an internal belief, and then report that belief by sampling. The internal belief is expit(alpha * prior log-odds + beta * log LR) (base-rate neglect for alpha < 1, conservatism for beta < 1). The subject draws N mental samples from that belief and reports the proportion k/N in which H holds. The mean report therefore carries the log-odds distortion, while trial-to-trial variability is binomial: largest when the distorted belief is near 50% and vanishing near 0% and 100%. A small motor/reporting noise tau smears the discrete values onto the 0-100 scale, and a lapse rate epsilon gives occasional random reports. Non-integer N mixes neighbouring integer sample sizes.


**Parameters:**
- alpha: [0.7697, 1.944]
- beta: [0.1168, 0.9097]
- N: [9.983, 200]
- tau: [0.005, 0.09912]
- epsilon: [0.0548, 0.3936]

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    # See bayes_noise.yaml for the stimulus / output contract.
    from scipy.special import ndtr
    from scipy.stats import binom
    K = 101
    scalar = np.ndim(stimulus["prior"]) == 0
    prior = np.clip(np.atleast_1d(np.asarray(stimulus["prior"], dtype=float)), 1e-6, 1 - 1e-6)
    llr = np.atleast_1d(np.asarray(stimulus["log_lr"], dtype=float))
    alpha = float(parameters["alpha"])
    beta = float(parameters["beta"])
    target = 1.0 / (1.0 + np.exp(-(alpha * (np.log(prior) - np.log1p(-prior)) + beta * llr)))

    tau = float(parameters["tau"])
    edges = (np.arange(K + 1) - 0.5) / (K - 1)
    edges[0], edges[-1] = -np.inf, np.inf

    def report_dist(n):
        k = np.arange(n + 1)
        pmf = binom.pmf(k[None, :], n, target[:, None])
        to_bins = np.diff(ndtr((edges[None, :] - (k / n)[:, None]) / tau), axis=1)
        return pmf @ to_bins

    N = float(parameters["N"])
    n0 = max(int(np.floor(N)), 1)
    w = min(max(N - n0, 0.0), 1.0)
    probs = report_dist(n0) if w == 0.0 else (1.0 - w) * report_dist(n0) + w * report_dist(n0 + 1)

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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    p = data['prior'].values.astype(float)\n    l = data['log_lr'].values.astype(float)\n    r = np.clip(data['response'].values.astype(float), 0.005, 0.995)\n    z = np.log(r / (1.0 - r))\n    def close(a, b):\n        return np.abs(a - b) < 1e-3\n    def med(mask):\n        if mask.sum() == 0:\n            return None\n        return float(np.median(z[mask]))\n    def lo(x):\n        return float(np.log(x / (1.0 - x)))\n    num = 0.0\n    den = 0.0\n    # symmetric pairs at prior .9 and .1: evidence term cancels\n    for pr in [0.9, 0.1]:\n        a = med(close(p, pr) & (l > 1e-9))\n        b = med(close(p, pr) & (l < -1e-9))\n        if a is not None and b is not None:\n            s = np.sign(lo(pr))\n            num += s * 0.5 * (a + b)\n            den += abs(lo(pr))\n    # extreme priors .98 / .02 with weak opposing evidence\n    a = med(close(p, 0.98))\n    b = med(close(p, 0.02))\n    if a is not None and b is not None:\n        num += 0.5 * (a - b)\n        den += abs(lo(0.98))\n    # pure prior cell (.8, zero evidence)\n    c = med(close(p, 0.8) & (np.abs(l) < 1e-9))\n    if c is not None:\n        num += c\n        den += abs(lo(0.8))\n    if den == 0:\n        return float('nan')\n    return float(num / den)\n",
  "rationale": "Estimates the prior weight alpha directly and robustly, decoupled from the evidence weight: it uses per-cell MEDIAN responses in log-odds (robust to the large lapse rates and to 5/10% rounding). Beta is cancelled by averaging the symmetric pairs (prior .9 with +/-LLR, prior .1 with +/-LLR); it is nearly irrelevant for the .98/.02 pair with weak opposing evidence; and it is exactly absent for the prior-.8, zero-LLR cell. The estimate is the summed signed median log-odds divided by the summed |prior log-odds|. The advocated log-odds-noise theory caps prior weighting near or below 1: alpha lies in about [0.73, 1.02], i.e. base-rate neglect or near-Bayesian use of the prior, so the metric should land around 0.8-0.9. The competing binomial-sampling theory needs over-extremised priors (alpha in about [0.77, 1.94]) to account for the pile-up of reports at the endpoints. Its reports therefore sit further from 50% on prior-dominated cells, giving a metric well above 1 (clipping at 0.5%/99.5% only partially limits this). The metric is a median-based ratio that is valid on a single subject's slice and pools over 7 cells (about 35 trials), so per-subject variance should be moderate, while the predicted means differ by roughly 0.3-0.5."
}
```

## Usage

```json
{
  "input_tokens": 6834,
  "output_tokens": 12127,
  "reasoning_tokens": 11037,
  "provider": "Claude Platform on AWS"
}
```
