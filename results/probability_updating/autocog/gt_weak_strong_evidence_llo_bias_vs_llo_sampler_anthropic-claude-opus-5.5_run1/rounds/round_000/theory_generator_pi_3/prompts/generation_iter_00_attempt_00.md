# generation_iter_00_attempt_00

## System Prompt

You are a renowned cognitive scientist and an expert Python programmer.

Your job is to propose a new theory and its model instantiation in the Probability Updating domain based on the feedback provided by an arbiter. The feedback contains diagnoses of mechanistic failures of the previous theory along with suggestions for a new theory family that overcomes those failures. The newly proposed theory and model should display human-like behavior when simulated on experiment(s). 
The goal of the theory generation process is to SURFACE theories that are EXPERIMENT-INVARIANT: that is,theories that explain data across the majority of experiments. 
You will see a list of theories that have been proposed in the past but you should only use them as inspiration and not to choose from them. Propose a new theory that is different. 
If they fail to do so, you will receive feedback on their performance on the same experiment(s) and you will have to propose another new theory and model that meet the requirements, iterating until you succeed.

If you think the failure to capture human behavior is due to arbiter feedback that is inaccurate or unhelpful, you can propose a new theory and model that ignore the feedback, but you must provide rationale for why you are ignoring it and how your proposal overcomes the identified mechanistic failures.

## ACCEPT GATE & LOSS TRAJECTORY — HOW THE LOOP HANDLES YOUR EDITS
This propose-loop has a programmatic accept gate: after every iteration the candidate's `aggregate_loss` is compared against the running-best loss; strict improvement -> ACCEPTED (the candidate becomes the new running-best base); otherwise -> REJECTED (the candidate is discarded and the base is unchanged). You do NOT need to manually "revert" a regressed edit — the gate already does that for you.

The block rendered below as `## PREVIOUS CANDIDATE (this loop)` is ALWAYS the running-best (last ACCEPTED) candidate, NEVER your most recent attempt if it was rejected. So:
  * Treat `## PREVIOUS CANDIDATE` as a known-good base. Build on it.
  * The `## LOSS TRAJECTORY` block tags every iteration ACCEPTED or REJECTED. Use this as ground truth on which past critic advice actually moved the loop forward and which didn't.
  * The `## PRIOR FEEDBACK ITERATIONS` block annotates each prior critique with the same ACCEPTED/REJECTED tag of the candidate it elicited. Down-weight critic advice whose previous candidates were REJECTED, and reinforce / extend advice whose candidates were ACCEPTED.
  * Treat the best ACCEPTED iteration's loss as a soft floor — the next edit should plausibly land at-or-below it, otherwise the gate will reject your attempt and the base stays put.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

## PARAMETER NOTATION
`parameters` is a JSON object mapping each parameter name (snake_case string) to a *string* value that specifies its domain. Every value MUST be a string — never a bare list, number, tuple, or expression. Use exactly one of these notations per parameter:

1. Continuous interval — square brackets, two numeric bounds:
   "[min, max]"
   Examples: "[0, 1]", "[1.0, 10.0]", "[10, 1000]"

2. Discrete set — curly braces, comma-separated values:
   "{v1, v2, ...}"
   Example: "{1, 2}"

3. Vector of intervals whose length is set by the experiment — a bracketed tuple repeated by a symbolic length variable:
   "[(min, max)] * length_var"
   Example: "[(0, 1)] * n_features"

4. Symbolic reference — a bare variable name (no brackets, no angle brackets), used when the parameter takes its value from an experiment-defined constant rather than a range:
   "variable_name"
   Example: "n_features"

Rules:
- Do not use parentheses for intervals; square brackets only. Tuples `(a, b)` are reserved for the vector-of-intervals notation in (3).
- Do not mix notations within a single value (e.g., no "[0, 1] or {2, 3}").
- Do not quote numbers inside the notation (write "[0, 1]", not "['0', '1']").
- Every parameter referenced by `predict` or `policy` must appear as a key in `parameters`, and vice versa.
- Notations 3 and 4 may ONLY reference the experiment-defined symbolic identifiers listed under "ALLOWED SYMBOLIC IDENTIFIERS" below. Do not invent new identifier names. If a parameter's shape doesn't fit any of those variables, fall back to a literal interval (notation 1) or discrete set (notation 2). Use these names so the model adapts to any experiment in this domain instead of hardcoding shapes.

## ALLOWED SYMBOLIC IDENTIFIERS (for notations 3 and 4 above)
- n_trials: number of trials per subject

## AVAILABLE IMPORTS inside `predict` and `policy`
- numpy as np
- pandas as pd
- scipy and its submodules
- torch and torch.nn.functional as F
- sklearn and its submodules
- math, random, and other standard Python libraries

## RUNTIME CONTRACT (function signatures and argument shapes)
`predict(parameters, state, history) -> np.ndarray`:
- `parameters`: dict[str, value]. One sample drawn from your declared `parameters` ranges, applied for the entire subject run.
- `state`: the per-trial input delivered by the experiment (shape is domain-specific — see the experiment description above and the `history` key list below, which mirrors the per-trial variables carried in `state`). Convert to an array with `np.asarray(state)` if you need array ops.
- `history`: dict-of-lists for past trials in this subject's run, NOT a list-of-dicts. The per-trial keys are:
  Each value below is a Python list in trial order; entry `i` is the value for trial `i`. On the first trial all lists are empty.
  - `"trial"`: 0-based trial index within the subject
  - `"prior"`: prior P(H) stated to the subject
  - `"p_signal_h"`: probability of the observed evidence if H is true
  - `"p_signal_not_h"`: probability of the observed evidence if H is false
  - `"n_signals"`: number of signals (draws) in the evidence
  - `"log_lr"`: ln(p_signal_h / p_signal_not_h)
  - `"bayes"`: Bayesian posterior P(H | evidence)
  - `"response"`: reported P(H) in [0, 1] (response_bin / 100)
  - `"response_bin"`: reported percentage as an integer 0-100
Iterating `for x in history:` iterates the dict KEYS (strings); to walk trials index the lists in lock-step, e.g. `for i in range(len(next(iter(history.values())))): ...`.
- Returns: 1-D `np.ndarray` of choice probabilities over the experiment's discrete action set, summing to 1.

`policy(probs) -> int`:
- Receives the probability vector produced by `predict`.
- Returns: integer index in `[0, len(probs))` identifying the chosen action. If you sample with `np.random.choice(..., p=probs)`, normalise first (`probs = np.asarray(probs, dtype=np.float64); probs /= probs.sum()`) to avoid the "probabilities do not sum to 1" ValueError from float drift.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects are told a prior probability that hypothesis H is true (e.g. that a ball was drawn from urn A rather than urn B), then shown evidence whose probability under H and under not-H is known (e.g. the colours of several draws, with the urns' compositions stated). They report the probability that H is true on a 0-100% scale.

Each subject completes about 60 trials in random order. On each trial `predict(parameters, stimulus, history)` receives `stimulus`, a dict with keys `prior`: P(H) before this step's evidence: the design prior if there is one, else the subject's own reported prior; `prior_is_objective`: 1 if `prior` comes from the design, 0 if it is the subject's report; `log_lr`: ln[P(evidence | H) / P(evidence | not-H)] of this step's evidence; `n_signals`: number of signals (draws, informants, clues) in this step's evidence; `signal_dir`: +1 evidence favours H, -1 favours not-H, 0 uninformative; `bayes`: Bayesian posterior P(H) given `prior` and `log_lr`. Values are floats for one trial; the same code is also called with equal-length numpy arrays (a batch of trials) when fitting to human data, so write it to broadcast. It must return probabilities over the 101 response bins 0%, 1%, ..., 100%: shape (101,) for one trial, (n, 101) for a batch. `history` is the subject's earlier trials (dict of lists) or None when fitting a batch. As in human data, every response then passes through a shared reporting stage that does not depend on the theory: with probability 0.13 it is rounded to the nearest multiple of 10%, otherwise with probability 0.10 to the nearest multiple of 5%. Theories should not model this rounding themselves, and metrics should not rely on exact values that rounding would blur.

## ARBITER GUIDE
The arbiter labelled this round's two theories in its recommendation as follows:
- THEORY 1 = `pi_1`
- THEORY 2 = `pi_2`
- The recommendation below acts on THEORY 2 (= `pi_2`).

Replace pi_2 with a noisy-log-odds-encoding theory in which distortion arises from noise.

Core mechanism:
- Prior log-odds and evidence LLR are each encoded with Gaussian noise in log-odds space.
- That noise has a constant part plus a part that scales with magnitude (scalar/Weber variability): sd_i = s0 + w*|x_i|.
- The subject combines the noisy encodings with Bayesian shrinkage toward 0 (a prior over log-odds with spread lambda), then reports expit of the decoded log-odds, plus small response noise and a lapse rate.

What this predicts, consistent with the data so far:
- Responses are roughly logit-normal, so the steep compression of probability-scale variance toward the ends and the scarcity of exact 0/100 reports are preserved.

Where it departs from pi_1:
- Effective alpha and beta are not constants. Shrinkage grows with stimulus magnitude, so response log-odds are a concave, saturating function of |prior log-odds| and |LLR|. Large LLRs are underweighted proportionally more, so there is more conservatism at extreme evidence.
- Log-odds variability grows with |prior log-odds| and |LLR|.
- Conflict cells, which have large opposing components but a near-50% belief, show more log-odds variance than the pure 50% cell. pi_1 predicts equal variance there.
- Mean distortion and variability are coupled: cells or subjects with more noise show more conservatism and base-rate neglect.

Parameters: s0, w, lambda (or separate shrinkage priors for prior and evidence), epsilon.

Suggested discriminating experiments:
- Vary LLR magnitude finely (e.g. 0.5, 1, 2, 3, 4.6) at prior 0.5, and vary prior log-odds at uninformative evidence. Test linearity of median response log-odds, which pi_1 says must be linear.
- Compare log-odds IQR in conflict cells against the pure 50% cell, and in large- versus small-magnitude congruent cells.
- Within subjects, test whether the per-cell log-odds spread correlates with shortfall from the Bayesian log-odds.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_1` (overall score: 1.000)

**Description**
Probability distortion is a systematic bias applied when the judgment is formed: people combine prior and evidence additively in log-odds space, as Bayes prescribes, but with the wrong weights. The posterior log-odds equal alpha times the prior log-odds plus beta times the log likelihood ratio. alpha < 1 is base-rate neglect (the prior is underweighted), beta < 1 is conservatism (the evidence is underweighted) and beta > 1 is overreaction; alpha = beta = 1 recovers Bayes. Because the distortion is a deterministic transformation, the same stimulus always yields the same central judgment; trial-to-trial variability is a separate, constant response noise in log-odds space (sigma), plus a lapse rate epsilon. The theory therefore predicts that mean distortion and response variability are unrelated: changing how noisy responses are does not change their average, and the average is a fixed linear function of prior and evidence log-odds.


`predict(parameters, state, history) -> np.ndarray`:
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


`policy(probs) -> int`:
def policy(probabilities):
    probabilities = probabilities / probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))


`parameters`:
- alpha: [0.7325, 1.037]
- beta: [0.1248, 0.8789]
- sigma: [0.1955, 1.207]
- epsilon: [0.04915, 0.4149]

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.1400 (var=0.0062) vs this=0.1675 (var=0.0068)
- Experiment 2: real=-1.5041 (var=0.4026) vs this=-1.6094 (var=0.6514)


---

### `pi_2` (overall score: 0.000)

**Description**
Both mechanisms operate: people misweight prior and evidence when they form an internal belief, and then report that belief by sampling. The internal belief is expit(alpha * prior log-odds + beta * log LR) (base-rate neglect for alpha < 1, conservatism for beta < 1). The subject draws N mental samples from that belief and reports the proportion k/N in which H holds. The mean report therefore carries the log-odds distortion, while trial-to-trial variability is binomial: largest when the distorted belief is near 50% and vanishing near 0% and 100%. A small motor/reporting noise tau smears the discrete values onto the 0-100 scale, and a lapse rate epsilon gives occasional random reports. Non-integer N mixes neighbouring integer sample sizes.


`predict(parameters, state, history) -> np.ndarray`:
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


`policy(probs) -> int`:
def policy(probabilities):
    probabilities = probabilities / probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))


`parameters`:
- alpha: [0.7325, 1.037]
- beta: [0.1248, 0.8789]
- N: [9.983, 200]
- tau: [0.005, 0.09912]
- epsilon: [0.04915, 0.4149]

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.1400 (var=0.0062) vs this=0.3688 (var=0.0217)
- Experiment 2: real=-1.5041 (var=0.4026) vs this=-0.4055 (var=0.4605)


## EXPERIMENTAL RESULTS
### Experiment 1
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
**Other theories' values on this metric (for reference):**
- pi_1: 0.1675 (var=0.0068)
- pi_2: 0.3688 (var=0.0217)

### Experiment 2
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
**Other theories' values on this metric (for reference):**
- pi_2: -0.4055 (var=0.4605)
- pi_1: -1.6094 (var=0.6514)


## IMPLEMENTATION GUARDRAILS
- The parameters should be within the specified ranges.
- The model's predictions should be valid probability distributions (non-negative and sum to 1).
- When converting logits to probabilities via softmax, always use the numerically stable form: subtract the max before exponentiating (`x = x - np.max(x); p = np.exp(x); p /= p.sum()`). A naive `np.exp(x) / np.sum(np.exp(x))` overflows to Inf/NaN for large logits. Alternatively, use `scipy.special.softmax`.


## PROPOSAL
Propose a novel theory from scratch based on all the information available, faithfully implementing the mechanism family the arbiter prescribed above. Do NOT simply reuse anything generated in past rounds.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- theory: The theoretical claim this theory makes about cognition.
- predict: Python source defining def predict(parameters, state, history) -> np.ndarray.
- policy: Python source defining def policy(probs) -> int.
- parameters: Parameter ranges (e.g. "[1.0, 10.0]") or choices (e.g. "{1, 2}"), sampled per subject.
- rationale: Reasoning behind this theory design; used to explain why this theory is better then the previous ones.


## Response

```json
```json
{
  "theory": "Noisy log-odds encoding with Weber-scaled noise and Bayesian shrinkage. Distortion in probability updating comes from noise, not from fixed mis-weighting. The stated prior is converted to log-odds L0 = logit(prior), and the evidence to its log likelihood ratio LLR. Each quantity x_i is encoded internally as a Gaussian sample m_i ~ N(x_i, sd_i^2). The noise sd has a constant part plus a part proportional to magnitude (scalar, Weber-like variability): sd_i = s0 + w*|x_i|. The subject does not know the true magnitudes, so it decodes each noisy measurement with Bayesian shrinkage toward 0. The decoder uses a zero-mean Gaussian prior over log-odds with spread lambda_p for priors and lambda_e for evidence. That gives decoded value k_i*m_i, with gain k_i = lambda_i^2 / (lambda_i^2 + sd_i^2). The decoded posterior log-odds is the sum k_p*m_p + k_e*m_e. It is reported as expit of that sum, after a small extra response noise in log-odds (sigma_r), and with lapse probability epsilon a random report. The theory makes four predictions. (i) Effective weights alpha_eff = k_p and beta_eff = k_e fall as stimulus magnitude grows, so response log-odds are a concave, saturating function of |prior log-odds| and |LLR|, with more conservatism for extreme evidence. (ii) Response variability in log-odds grows with |L0| and |LLR|, so conflict cells (large opposing components, belief near 50%) are more variable than the pure 50/50 cell. (iii) Mean distortion and variability are coupled: subjects or cells with more encoding noise show more conservatism and more base-rate neglect. (iv) Responses are roughly logit-normal, so probability-scale variability is compressed near the ends and exact 0/100 reports are rare.",
  "predict": "def predict(parameters, stimulus, history):\n    import numpy as np\n    from scipy.special import ndtr\n    K = 101\n    scalar = np.ndim(stimulus[\"prior\"]) == 0\n    prior = np.clip(np.atleast_1d(np.asarray(stimulus[\"prior\"], dtype=float)), 1e-6, 1 - 1e-6)\n    llr = np.atleast_1d(np.asarray(stimulus[\"log_lr\"], dtype=float)) * np.ones_like(prior)\n    L0 = np.log(prior) - np.log1p(-prior)\n\n    s0 = float(parameters[\"s0\"])\n    w = float(parameters[\"w\"])\n    lam_p = float(parameters[\"lambda_p\"])\n    lam_e = float(parameters[\"lambda_e\"])\n    sig_r = float(parameters[\"sigma_r\"])\n    eps = float(parameters[\"epsilon\"])\n\n    # Weber-scaled encoding noise in log-odds space\n    sd_p = s0 + w * np.abs(L0)\n    sd_e = s0 + w * np.abs(llr)\n    # Bayesian shrinkage gains (zero-mean Gaussian prior over log-odds)\n    k_p = lam_p ** 2 / (lam_p ** 2 + sd_p ** 2)\n    k_e = lam_e ** 2 / (lam_e ** 2 + sd_e ** 2)\n\n    mu = k_p * L0 + k_e * llr\n    sd = np.sqrt((k_p * sd_p) ** 2 + (k_e * sd_e) ** 2 + sig_r ** 2)\n    sd = np.maximum(sd, 1e-3)\n\n    edges = (np.arange(K + 1) - 0.5) / (K - 1)\n    z_edges = np.empty(K + 1)\n    z_edges[0] = -np.inf\n    z_edges[-1] = np.inf\n    inner = edges[1:-1]\n    z_edges[1:-1] = np.log(inner) - np.log1p(-inner)\n\n    cdf = ndtr((z_edges[None, :] - mu[:, None]) / sd[:, None])\n    probs = np.diff(cdf, axis=1)\n    probs = np.clip(probs, 0.0, None)\n    probs = (1.0 - eps) * probs + eps / K\n    probs = probs / probs.sum(axis=1, keepdims=True)\n    return probs[0] if scalar else probs\n",
  "policy": "def policy(probs):\n    import numpy as np\n    p = np.asarray(probs, dtype=np.float64)\n    p = np.clip(p, 0.0, None)\n    p /= p.sum()\n    return int(np.random.choice(len(p), p=p))\n",
  "parameters": {
    "s0": "[0.1, 1.0]",
    "w": "[0.0, 0.45]",
    "lambda_p": "[1.0, 8.0]",
    "lambda_e": "[0.5, 5.0]",
    "sigma_r": "[0.1, 0.7]",
    "epsilon": "[0.04, 0.35]"
  },
  "rationale": "This implements the arbiter's prescribed noisy-log-odds-encoding family and replaces the binomial sampling account, pi_2, which failed both experiments.\n\nWhy pi_2 failed:\n- Binomial k/N reports pile up at exactly 0 or 100 when the belief is extreme. It predicted 0.37 for the share of 100% reports among folded responses of at least 90 in the Experiment 1 window, against 0.14 observed.\n- It also predicted too little relative compression of spread in extreme cells (Experiment 2: -0.41 predicted vs -1.50 observed).\n\nWhat this model keeps from pi_1:\n- It keeps the logit-normal response distribution that let pi_1 score well. This preserves rare exact end-point reports and strongly compressed probability-scale deviations in extreme cells, so both metrics should sit near pi_1's values.\n\nHow it differs from pi_1:\n- The distortion is generated by noise rather than stipulated. Effective alpha and beta are shrinkage gains lambda^2/(lambda^2 + (s0 + w|x|)^2).\n- These gains decrease with magnitude, giving saturating conservatism at large LLRs, as in Experiment 1's 0.99/0.01 evidence.\n- Log-odds variance grows with magnitude, which slightly raises spread in extreme cells relative to pi_1. That nudges the Experiment 2 metric from pi_1's -1.61 toward the observed -1.50.\n- Lapse epsilon and residual response noise sigma_r cover motor and reporting variability.\n\nParameter ranges:\n- With w = 0, the model reduces to pi_1 with alpha = k_p, beta = k_e and constant noise, so it nests the running-best behaviour.\n- The ranges allow alpha_eff of roughly 0.7-1 for priors (large lambda_p) and beta_eff of roughly 0.2-0.9 for evidence (smaller lambda_e), matching the fitted pi_1 ranges.\n\nWhat it adds, and how to test it:\n- Magnitude-dependent weighting.\n- More variance in conflict cells than in the pure 50% cell.\n- A within-subject coupling between noise and conservatism, which can be tested in future discriminating experiments."
}
```
```

## Usage

```json
{
  "input_tokens": 9421,
  "output_tokens": 3551,
  "reasoning_tokens": 951,
  "provider": "Claude Platform on AWS"
}
```
