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
- THEORY 2 = `pi_4`
- The recommendation below acts on THEORY 1 (= `pi_1`).

Replace pi_1 with a theory of 'bounded deterministic integration plus heterogeneous off-task responding'. It should be a strong competitor to pi_4 that isolates the question of whether a cue-selection mixture is needed at all.

Core claims:
(1) Every trial integrates deterministically: L = alpha*logit(prior) + beta*LLR. There is no rho/q strategy mixture; conflict cells are unimodal in the engaged state.
(2) The output passes through the same soft saturating bound as pi_4, with Delta ≈ 3.5–7 and kappa ≥ 1. This retains pi_4's success on Exps 3–6: few exact-100s, compressed raw-scale spread at extremes, roughly constant or slightly rising logit MAD.
(3) Constant Gaussian logit noise sigma (≈0.3–1.0) around the bounded value.
(4) A substantial, subject-heterogeneous non-engaged component with total mass lambda drawn over roughly 0.05–0.45 (mean ≈0.25). It splits into:
   (a) uniform guessing over 0–100, needed for baseline-cell tails;
   (b) a polarity-reversal lapse that reports the mirror value 100 − R, i.e. P(not-H), which sends mass to the wrong side in clean cells;
   (c) optionally, a 50% 'don't know' spike.
   These components should apply equally to single-source, congruent and conflict cells.

Key contrasts with pi_4:
- Off-pattern mass is the same in single-cue and two-cue cells, versus pi_4's conflict-specific excess.
- Wrong-side reports in clean cells sit near the mirror of the correct answer (reversal) rather than spreading uniformly.

Contrasts with old pi_1:
- Saturation at the extremes.
- A structured lapse, so beta and sigma need not be distorted to soak up outliers. Keep beta ≈ 0.4–1.0.

Suggested next discriminating experiment: matched clean single-source cells and strong conflict cells. Compare the excess off-side mass in conflict cells over clean cells (≈0 under the new theory, rho*(1−q) under pi_4). Also test whether wrong-side reports in clean cells cluster at 100 − (expected report) or are flat.

Note for later: if pi_4 survives, its epsilon range must be widened to ~0–0.45 to fit Exps 1–2.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_1` (overall score: 0.655)

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
- Experiment 3: real=0.6707 (var=0.1719) vs this=0.5297 (var=0.1446)
- Experiment 4: real=0.1515 (var=0.1960) vs this=0.0709 (var=0.2473)
- Experiment 5: real=0.1240 (var=0.0029) vs this=0.1220 (var=0.0061)
- Experiment 6: real=0.1200 (var=0.0049) vs this=0.1016 (var=0.0049)


---

### `pi_4` (overall score: 0.208)

**Description**
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


`parameters`:
- alpha: [0.6, 1.1]
- beta: [0.3, 1.0]
- rho: [0.0, 0.4]
- q: [0.3, 0.8]
- Delta: [3.5, 7.0]
- kappa: [1.5, 4.0]
- sigma: [0.3, 1.0]
- epsilon: [0.0, 0.15]

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.1400 (var=0.0062) vs this=0.1480 (var=0.0076)
- Experiment 2: real=-1.5041 (var=0.4026) vs this=-1.5041 (var=0.3621)
- Experiment 3: real=0.6707 (var=0.1719) vs this=0.7687 (var=0.0764)
- Experiment 4: real=0.1515 (var=0.1960) vs this=0.1254 (var=0.2522)
- Experiment 5: real=0.1240 (var=0.0029) vs this=0.0447 (var=0.0015)
- Experiment 6: real=0.1200 (var=0.0049) vs this=0.0408 (var=0.0022)


---

### `pi_3` (overall score: 0.170)

**Description**
Noisy log-odds encoding with Weber-scaled noise and Bayesian shrinkage. Distortion in probability updating comes from noise, not from fixed mis-weighting.

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

`predict(parameters, state, history) -> np.ndarray`:
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


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    p = np.asarray(probs, dtype=np.float64)
    p = np.clip(p, 0.0, None)
    p /= p.sum()
    return int(np.random.choice(len(p), p=p))


`parameters`:
- s0: [0.1, 0.5]
- w: [0.05, 0.35]
- lambda_p: [1.3, 2.7]
- lambda_e: [0.7, 2.5]
- sigma_r: [0.05, 0.4]
- epsilon: [0.04, 0.35]

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.1400 (var=0.0062) vs this=0.1675 (var=0.0091)
- Experiment 2: real=-1.5041 (var=0.4026) vs this=-1.3863 (var=0.5246)
- Experiment 3: real=0.6707 (var=0.1719) vs this=0.8425 (var=0.0442)
- Experiment 4: real=0.1515 (var=0.1960) vs this=0.5447 (var=0.2320)
- Experiment 5: real=0.1240 (var=0.0029) vs this=0.0953 (var=0.0041)
- Experiment 6: real=0.1200 (var=0.0049) vs this=0.0872 (var=0.0052)


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
- pi_3: 0.1675 (var=0.0091)
- pi_4: 0.1480 (var=0.0076)

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
- pi_3: -1.3863 (var=0.5246)
- pi_4: -1.5041 (var=0.3621)

### Experiment 3
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
**Other theories' values on this metric (for reference):**
- pi_1: 0.5297 (var=0.1446)
- pi_3: 0.8425 (var=0.0442)
- pi_2: 0.5472 (var=0.0784)
- pi_4: 0.7687 (var=0.0764)

### Experiment 4
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
**Other theories' values on this metric (for reference):**
- pi_3: 0.5447 (var=0.2320)
- pi_1: 0.0709 (var=0.2473)
- pi_2: 0.5459 (var=0.2839)
- pi_4: 0.1254 (var=0.2522)

### Experiment 5
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
**Other theories' values on this metric (for reference):**
- pi_1: 0.1220 (var=0.0061)
- pi_4: 0.0447 (var=0.0015)
- pi_2: 0.1013 (var=0.0040)
- pi_3: 0.0953 (var=0.0041)

### Experiment 6
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
**Other theories' values on this metric (for reference):**
- pi_4: 0.0408 (var=0.0022)
- pi_1: 0.1016 (var=0.0049)
- pi_2: 0.1032 (var=0.0064)
- pi_3: 0.0872 (var=0.0052)


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
  "theory": "Bounded deterministic integration plus heterogeneous off-task responding.\n\n(1) Engaged trials integrate deterministically. Every engaged trial combines prior and evidence additively in log-odds with distorted weights: L = alpha*logit(prior) + beta*LLR. Here alpha<1 is base-rate neglect and beta<1 is conservatism. There is no cue-selection mixture, so conflict cells are unimodal in the engaged state.\n\n(2) Soft saturating bound. The integrated log-odds pass through L' = sign(L)*Delta*tanh((|L|/Delta)^kappa)^(1/kappa), with kappa>1. This is nearly linear up to about |L|≈Delta and then saturates. The bound keeps reports off the exact endpoints, compresses raw-scale spread near the extremes, and leaves logit-scale variability roughly constant.\n\n(3) Response noise. Constant Gaussian noise sigma is added in log-odds around the bounded value.\n\n(4) Non-engaged component. With subject-specific total probability lambda (≈0.05–0.4), the trial is not processed on-task. It splits into three parts:\n- uniform guessing over 0–100;\n- a polarity-reversal lapse, where the subject reports P(not-H), i.e. the mirror 100−R of what they would otherwise report;\n- a 'don't know' report of 50%.\nThis component is stimulus-independent in its probability. Off-pattern mass is therefore equal in single-source, congruent and conflict cells, and wrong-side reports in clean cells cluster at the mirror of the correct answer rather than spreading flat.",
  "predict": "def predict(parameters, stimulus, history):\n    import numpy as np\n    from scipy.special import ndtr\n    K = 101\n    scalar = np.ndim(stimulus[\"prior\"]) == 0\n    prior = np.clip(np.atleast_1d(np.asarray(stimulus[\"prior\"], dtype=float)), 1e-6, 1 - 1e-6)\n    llr = np.atleast_1d(np.asarray(stimulus[\"log_lr\"], dtype=float)) * np.ones_like(prior)\n    L0 = np.log(prior) - np.log1p(-prior)\n\n    alpha = float(parameters[\"alpha\"])\n    beta = float(parameters[\"beta\"])\n    Delta = max(float(parameters[\"Delta\"]), 1e-3)\n    kappa = max(float(parameters[\"kappa\"]), 1e-3)\n    sigma = max(float(parameters[\"sigma\"]), 1e-3)\n    lam = float(np.clip(parameters[\"lam\"], 0.0, 1.0))\n    f_rev = float(np.clip(parameters[\"f_rev\"], 0.0, 1.0))\n    f_half = float(np.clip(parameters[\"f_half\"], 0.0, 1.0))\n    tot = f_rev + f_half\n    if tot > 1.0:\n        f_rev /= tot\n        f_half /= tot\n    f_uni = max(0.0, 1.0 - f_rev - f_half)\n\n    L = alpha * L0 + beta * llr\n    u = np.abs(L) / Delta\n    Lb = np.sign(L) * Delta * np.tanh(u ** kappa) ** (1.0 / kappa)\n\n    edges = (np.arange(K + 1) - 0.5) / (K - 1)\n    z_edges = np.empty(K + 1)\n    z_edges[0] = -np.inf\n    z_edges[-1] = np.inf\n    inner = edges[1:-1]\n    z_edges[1:-1] = np.log(inner) - np.log1p(-inner)\n\n    cdf = ndtr((z_edges[None, :] - Lb[:, None]) / sigma)\n    eng = np.clip(np.diff(cdf, axis=1), 0.0, None)\n    eng = eng / np.maximum(eng.sum(axis=1, keepdims=True), 1e-300)\n\n    rev = eng[:, ::-1]\n    spike = np.zeros(K)\n    spike[50] = 1.0\n    uni = np.full(K, 1.0 / K)\n\n    probs = (1.0 - lam) * eng + lam * (f_uni * uni[None, :] + f_rev * rev + f_half * spike[None, :])\n    probs = np.clip(probs, 0.0, None)\n    probs = probs / probs.sum(axis=1, keepdims=True)\n    return probs[0] if scalar else probs\n",
  "policy": "def policy(probs):\n    import numpy as np\n    p = np.asarray(probs, dtype=np.float64)\n    p = np.clip(p, 0.0, None)\n    p /= p.sum()\n    return int(np.random.choice(len(p), p=p))\n",
  "parameters": {
    "alpha": "[0.7, 1.1]",
    "beta": "[0.35, 1.0]",
    "Delta": "[3.5, 7.0]",
    "kappa": "[1.5, 4.0]",
    "sigma": "[0.3, 1.0]",
    "lam": "[0.05, 0.4]",
    "f_rev": "[0.0, 0.4]",
    "f_half": "[0.0, 0.25]"
  },
  "rationale": "The theory follows the arbiter's prescription. It keeps pi_4's soft saturating bound, which explains why Exps 1–4 work:\n- Exp 1: a low exact-100 share among high reports (0.14).\n- Exp 2: strongly compressed raw-scale deviations at the extremes (−1.50).\n- Exp 4: a near-constant logit MAD ratio.\n\nIt drops pi_4's rho/q cue-selection mixture and its narrow epsilon. pi_4's failures on Exps 5–6 (0.045 and 0.041 vs observed 0.124 and 0.12) came from too little wrong-side mass in clean cells. pi_1 matched those two experiments only by inflating a uniform lapse while distorting beta.\n\nHere a heterogeneous non-engaged mass (lambda ≈0.05–0.4, mean ≈0.22) supplies baseline tail mass without distorting beta or sigma. It has three parts:\n- a uniform guess;\n- a polarity reversal, which puts mass at 100−R on the wrong side of clean cells (Exp 5 off-side zones, Exp 6 ≤40% reports under strong H-favouring stimuli);\n- a 50% spike.\n\nBecause the lapse is the same for every stimulus, the model isolates whether conflict-specific excess spread (pi_4's mixture) is needed. Medians in Exp 3 come from engaged integration. With beta ≈0.35–1.0 and alpha ≈0.7–1.1, the evidence-to-prior weight ratio lands near the observed 0.67, between pi_1 (0.53) and pi_4 (0.77). Reversal lapses pull the pooled medians slightly toward zero, which helps that ratio."
}
```
```

## Usage

```json
{
  "input_tokens": 17823,
  "output_tokens": 3888,
  "reasoning_tokens": 1408,
  "provider": "Claude Platform on AWS"
}
```
