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
**Description:** Both mechanisms operate: people misweight prior and evidence when they form an internal belief, and then report that belief by sampling. The internal belief is expit(alpha * prior log-odds + beta * log LR) (base-rate neglect for alpha < 1, conservatism for beta < 1). The subject draws N mental samples from that belief and reports the proportion k/N in which H holds. The mean report therefore carries the log-odds distortion, while trial-to-trial variability is binomial: largest when the distorted belief is near 50% and vanishing near 0% and 100%. A small motor/reporting noise tau smears the discrete values onto the 0-100 scale, and a lapse rate epsilon gives occasional random reports. Non-integer N mixes neighbouring integer sample sizes.


**Parameters:**
- alpha: [0.7325, 1.037]
- beta: [0.1248, 0.8789]
- N: [9.983, 200]
- tau: [0.005, 0.09912]
- epsilon: [0.04915, 0.4149]

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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    import pandas as pd\n    slopes = []\n    for sid, d in data.groupby('subject_id'):\n        key = d['prior'].round(4).astype(str) + '_' + d['log_lr'].round(3).astype(str)\n        xs = []\n        ys = []\n        for k, g in d.groupby(key):\n            r = np.sort(g['response'].values.astype(float))\n            n = len(r)\n            if n < 3:\n                continue\n            m = float(np.median(r))\n            if n >= 5:\n                s = r[n - 2] - r[1]\n            else:\n                s = r[-1] - r[0]\n            mm = min(max(m, 0.01), 0.99)\n            xs.append(np.log(mm * (1.0 - mm)))\n            ys.append(np.log(s + 0.02))\n        if len(xs) >= 4 and np.var(xs) > 1e-6:\n            xs = np.asarray(xs)\n            ys = np.asarray(ys)\n            xc = xs - xs.mean()\n            slope = float(np.sum(xc * (ys - ys.mean())) / np.sum(xc * xc))\n            slopes.append(slope)\n    if len(slopes) == 0:\n        return float('nan')\n    return float(np.mean(slopes))\n",
  "rationale": "Both theories share the same mean structure (alpha*prior log-odds + beta*LLR), so they differ only in how within-cell response spread scales with the central belief m. Constant logit-normal noise gives spread proportional to m(1-m), so a log-log regression of within-cell spread on m(1-m) has slope near 1, a bit lower with large sigma or the floor effects. Binomial sampling gives spread proportional to sqrt(m(1-m)), so the slope is near 0.5. This holds whatever the noise scale (sigma or N), so parameter heterogeneity across subjects matters less. For each subject and each of the 12 cells (5 reps each), I take the cell median as m. The spread is a trimmed range: 2nd to 4th order statistic, which is robust to one lapse on each side. I then regress log(spread + 0.02) on log(m(1-m)), clipping m to [0.01, 0.99], and average the slopes across subjects. The 0.02 floor tames zero spreads caused by discretization and rounding. The 12 cells span beliefs from about 0.5 to above 0.99, which gives good leverage on the slope. I expect a slope of about 0.8-1 under the logit-noise theory and about 0.4-0.5 under the sampling theory."
}
```

## Usage

```json
{
  "input_tokens": 6547,
  "output_tokens": 4485,
  "reasoning_tokens": 3576,
  "provider": "Claude Platform on AWS"
}
```
