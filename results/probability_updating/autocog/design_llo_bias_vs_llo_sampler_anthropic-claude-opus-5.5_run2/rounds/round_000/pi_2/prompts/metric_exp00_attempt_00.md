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


## ADVOCATED THEORY
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    import pandas as pd\n    d = data.copy()\n    d['prior_r'] = d['prior'].astype(float).round(3)\n    d['llr_r'] = d['log_lr'].astype(float).round(2)\n    d['resp'] = d['response'].astype(float)\n    med = d.groupby(['subject_id', 'prior_r', 'llr_r'])['resp'].transform('median')\n    d['dev'] = (d['resp'] - med).abs()\n    pr = d['prior_r'].values\n    ll = d['llr_r'].values\n    center = (np.isclose(pr, 0.5) & (np.abs(ll) < 0.01)) | (np.isclose(pr, 0.5) & np.isclose(ll, 1.10, atol=0.02)) | np.isclose(pr, 0.2)\n    tail = np.isclose(pr, 0.9) | np.isclose(pr, 0.02) | np.isclose(pr, 0.95) | np.isclose(pr, 0.05) | np.isclose(pr, 0.99)\n    if center.sum() == 0 or tail.sum() == 0:\n        return float('nan')\n    s_c = float(np.median(d.loc[center, 'dev'].values))\n    s_t = float(np.median(d.loc[tail, 'dev'].values))\n    return float(np.log((s_t + 0.005) / (s_c + 0.005)))\n",
  "rationale": "Both theories share the same central tendency, expit(alpha*prior logodds + beta*LLR), but they scale trial-to-trial spread with belief differently. Binomial sampling gives SD of about sqrt(p(1-p)/N), plus a constant motor noise tau that does not shrink in the tails. Constant logit-normal noise gives SD of about sigma*p(1-p). The ratio of tail-cell spread to center-cell spread therefore cancels the subject's noise-level parameter (N or sigma) to first order. It is about 2*sqrt(q) (or more, given tau) under sampling and about 4q under logit noise, where q=p(1-p) is roughly 0.02-0.1 in the tail cells. At q around 0.03 that is roughly 0.35-0.6 versus 0.12.\n\nHow the metric is computed:\n- Deviations are taken from each subject x cell median, so alpha and beta shifts in location are removed.\n- Center cells are those with distorted beliefs near 0.5: prior .5 with null or weak evidence, and prior .2 with LR 4.\n- Tail cells are those with extreme priors and null or congruent evidence, so beliefs fall near 0.02-0.995 for any plausible alpha and beta.\n- The spread in each group is the pooled median absolute deviation. This is robust to lapses, which hit both groups alike, and to the 5/10% rounding.\n- The metric is the log ratio of tail spread to center spread, with a small floor added to each.\n\nPredicted values are about -0.7 to 0 under sampling and about -2 under logit noise. The same code works on single subjects and on pooled data."
}
```

## Usage

```json
{
  "input_tokens": 6620,
  "output_tokens": 6773,
  "reasoning_tokens": 5691,
  "provider": "Claude Platform on AWS"
}
```
