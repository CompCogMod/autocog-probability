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
  "metric_source": "import numpy as np\nimport pandas as pd\n\ndef metric(data: pd.DataFrame) -> float:\n    p = data['prior'].values.astype(float)\n    l = data['log_lr'].values.astype(float)\n    r = data['response_bin'].values.astype(float)\n    L4 = np.log(4.0)\n    L19 = np.log(19.0)\n    def cell(pv, lv):\n        return (np.abs(p - pv) < 1e-3) & (np.abs(l - lv) < 0.05)\n    specs = [\n        (0.5, 0.0, (r <= 10) | (r >= 90)),\n        (0.5, L4, r <= 20),\n        (0.5, -L4, r >= 80),\n        (0.5, L19, r < 50),\n        (0.8, 0.0, r < 50),\n        (0.95, 0.0, r < 50),\n        (0.8, L4, r < 50),\n        (0.95, L19, r < 50),\n        (0.05, -L19, r > 50),\n        (0.6, L19, r < 50),\n    ]\n    hits = 0.0\n    total = 0.0\n    for pv, lv, zone in specs:\n        m = cell(pv, lv)\n        total += float(m.sum())\n        hits += float((m & zone).sum())\n    if total <= 0:\n        return float('nan')\n    return float(hits / total)\n",
  "rationale": "Implausible-zone rate in cells where every component of both theories points clearly one way. I use single-source and congruent cells (plus the 0.6-prior/strong-evidence probe, whose competing prior-only mode is still on the same side). For each cell I count reports in a zone that no integration route, prior-only route or evidence-only route can reach without a lapse:\n- the wrong side of 50% for strong one-directional cells;\n- the opposite extreme for the weak-evidence cells;\n- the far tails (<=10 or >=90) for the 50/50 baseline.\n\nConflict cells and the weak-conflict probes (0.95 with -1.39 or -0.41) are excluded. There the competing theory's evidence-only mode legitimately lands on the far side, and including them would cancel the contrast.\n\nIn these zones, responses come almost only from uniform lapses, plus a small Gaussian-tail contribution from very noisy or low-beta subjects. The two theories place outlying responses differently:\n- The advocated fixed-weight model has no strategy mixture. Every off-pattern human response must be absorbed by a uniform lapse (epsilon 0.05-0.41) and by a wide sigma (up to 1.2) and low beta (down to 0.12). These produce wrong-side reports even in cells where both cues agree or only one is informative.\n- The competing theory explains off-pattern reports mostly through cue-selection modes confined to two-cue conflict cells. It has small lapses (epsilon 0-0.15), sigma at most 1 and beta at least 0.3, so these cells stay clean.\n\nPredicted values: about 0.10-0.13 under the advocated theory versus about 0.03-0.05 under the competing theory. With about 30 eligible trials per subject, the per-subject rate is reasonably tight. The count-based zones use 10%-or-wider margins, so the shared rounding stage barely affects them.\n\nOn human data, a clear excess of wrong-side reports in cells with no cue conflict would support a global lapse/noise account over a conflict-specific mixture."
}
```

## Usage

```json
{
  "input_tokens": 7945,
  "output_tokens": 15752,
  "reasoning_tokens": 14560,
  "provider": "Claude Platform on AWS"
}
```
