# generation_iter_01_attempt_00

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
- THEORY 1 = `pi_5`
- THEORY 2 = `pi_4`
- The recommendation below acts on THEORY 2 (= `pi_4`).

Replace pi_4 with a theory of 'sub-additive deterministic integration, salient-value response mapping, and trait-level off-task responding'.

(a) Engaged integration is deterministic and unimodal in conflict cells. This is required by Exp2. It is sub-additive when both cues are informative: L = alpha*L0 + beta*LLR - gamma*sign-aware shrinkage. One implementation is |L_comb| = (|aL0|^p + |bLLR|^p)^(1/p) for congruent cues with p > 1, and a proportionally shrunk difference for conflict cues. Alternatively, use precision-weighted averaging with total weight below 1. Either version should bring the Exp5 ratio to about 0.65-0.70, whereas pi_5 is purely additive.

(b) The response mapping is not plain logit-Gaussian. Reports are produced on a coarse salient-value grid (multiples of 5 and 10, plus 0, 50 and 100), and grid snapping gets stronger near the extremes. This makes extreme-cell raw deviations very small, giving an Exp4 ratio of about -1.5. It also yields the modest Exp6 logit-MAD increase of about 0.15 and the Exp3 endpoint ratio of about 0.14. This is a different mechanism from pi_5's tanh bound, so the two theories make distinguishable predictions: snapping implies discrete clustering and within-subject repeat identity at 95/99/100, whereas the tanh bound implies continuous compression.

(c) Off-task responding has subject-specific mass lambda of about 0.08-0.2 and is equal across cell types, since the data require this. Its composition differs from pi_5:
- mostly mirror or polarity reversals (100-R) and 50% 'don't know' reports;
- a smaller uniform share than pi_5's.
This predicts the opposite-side mass seen in Exp7 and Exp8 while keeping the median deviations in extreme cells low.

The proposer should design discriminating experiments that:
- test mirror-located versus uniform off-task mass (e.g., the mass in the 30-70% band versus the mirror band of extreme single-source cells);
- test congruent sub-additivity against the bound, using moderate-magnitude congruent cells where the tanh bound is inactive.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_5` (overall score: 0.777)

**Description**
Bounded deterministic integration plus heterogeneous off-task responding.

(1) Engaged integration is deterministic. Every engaged trial combines prior and evidence additively in log-odds with distorted weights: L = alpha*logit(prior) + beta*LLR. alpha<1 is base-rate neglect and beta<1 is conservatism. There is no cue-selection mixture, so conflict cells are unimodal in the engaged state.

(2) The output is softly bounded. The integrated value passes through L' = sign(L)*Delta*tanh((|L|/Delta)^kappa)^(1/kappa), with kappa>1 and Delta about 3.5–5.5. The bound keeps engaged reports off the exact endpoints, compresses raw-scale spread near the extremes, and leaves logit-scale variability roughly constant.

(3) Response noise is constant Gaussian noise sigma in log-odds around the bounded value.

(4) A non-engaged component has subject-specific total probability lambda (about 0.1–0.42). Its probability does not depend on the stimulus. It is dominated by uniform guessing over 0–100. It has two minor parts:
- a polarity-reversal lapse that reports the mirror value 100−R, i.e. P(not-H);
- a rare 50% 'don't know' report.
Off-pattern mass is therefore equal in single-source, congruent and conflict cells.

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


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    p = np.asarray(probs, dtype=np.float64)
    p = np.clip(p, 0.0, None)
    p /= p.sum()
    return int(np.random.choice(len(p), p=p))


`parameters`:
- alpha: [0.7, 1.1]
- beta: [0.35, 1.0]
- Delta: [3.5, 5.5]
- kappa: [1.5, 4.0]
- sigma: [0.3, 0.9]
- lam: [0.1, 0.42]
- f_rev: [0.0, 0.15]
- f_half: [0.0, 0.05]

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=0.1400 (var=0.0062) vs this=0.1547 (var=0.0089)
- Experiment 2: real=-1.5041 (var=0.4026) vs this=-1.0986 (var=0.6023)
- Experiment 3: real=0.6707 (var=0.1719) vs this=0.7787 (var=0.0769)
- Experiment 4: real=0.1515 (var=0.1960) vs this=0.2102 (var=0.2816)
- Experiment 5: real=0.1240 (var=0.0029) vs this=0.1127 (var=0.0038)
- Experiment 6: real=0.1200 (var=0.0049) vs this=0.1200 (var=0.0065)
- Experiment 7: real=0.0869 (var=0.0029) vs this=0.0954 (var=0.0029)
- Experiment 8: real=-0.0223 (var=0.0048) vs this=-0.0198 (var=0.0099)


---

### `pi_1` (overall score: 0.656)

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
- Experiment 7: real=0.0869 (var=0.0029) vs this=0.0891 (var=0.0029)
- Experiment 8: real=-0.0223 (var=0.0048) vs this=0.0003 (var=0.0116)


---

### `pi_3` (overall score: 0.162)

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
- Experiment 7: real=0.0869 (var=0.0029) vs this=0.0697 (var=0.0024)
- Experiment 8: real=-0.0223 (var=0.0048) vs this=0.0417 (var=0.0100)


## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision — only ACCEPTED candidates have ever been used as the base for a subsequent iteration. Use this together with PRIOR FEEDBACK ITERATIONS below to grade which past critic advice actually paid off.

- iter 1: loss=0.1444 -> ACCEPTED
Running-best (last ACCEPTED) base: iter 1 at loss=0.1444 — this is the source shown verbatim below under `## PREVIOUS CANDIDATE (this loop)`. Push the next edit's loss strictly below that floor or the gate will reject it.

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
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.1180 (var=0.0052) (Δ vs real -0.0221)
**Other theories' values on this metric (for reference):**
- pi_1: 0.1675 (var=0.0068)
- pi_2: 0.3688 (var=0.0217)
- pi_3: 0.1675 (var=0.0091)
- pi_4: 0.1480 (var=0.0076)
- pi_5: 0.1547 (var=0.0089)

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
**Previous candidate values (this loop):**
  - iter 1 (most recent): -1.3863 (var=0.6307) (Δ vs real +0.1178)
**Other theories' values on this metric (for reference):**
- pi_2: -0.4055 (var=0.4605)
- pi_1: -1.6094 (var=0.6514)
- pi_3: -1.3863 (var=0.5246)
- pi_4: -1.5041 (var=0.3621)
- pi_5: -1.0986 (var=0.6023)

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
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.7141 (var=0.0195) (Δ vs real +0.0434)
**Other theories' values on this metric (for reference):**
- pi_1: 0.5297 (var=0.1446)
- pi_3: 0.8425 (var=0.0442)
- pi_2: 0.5472 (var=0.0784)
- pi_4: 0.7687 (var=0.0764)
- pi_5: 0.7787 (var=0.0769)

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
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.4359 (var=0.4376) (Δ vs real +0.2844)
**Other theories' values on this metric (for reference):**
- pi_3: 0.5447 (var=0.2320)
- pi_1: 0.0709 (var=0.2473)
- pi_2: 0.5459 (var=0.2839)
- pi_4: 0.1254 (var=0.2522)
- pi_5: 0.2102 (var=0.2816)

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
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.1240 (var=0.0059) (Δ vs real +0.0000)
**Other theories' values on this metric (for reference):**
- pi_1: 0.1220 (var=0.0061)
- pi_4: 0.0447 (var=0.0015)
- pi_2: 0.1013 (var=0.0040)
- pi_3: 0.0953 (var=0.0041)
- pi_5: 0.1127 (var=0.0038)

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
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.1064 (var=0.0050) (Δ vs real -0.0136)
**Other theories' values on this metric (for reference):**
- pi_4: 0.0408 (var=0.0022)
- pi_1: 0.1016 (var=0.0049)
- pi_2: 0.1032 (var=0.0064)
- pi_3: 0.0872 (var=0.0052)
- pi_5: 0.1200 (var=0.0065)

### Experiment 7
**Design**
**Unique trials (n=12):**
  trial 1: prior=0.500  P(e|H)=0.5  P(e|not-H)=0.5  n_signals=1  log LR=+0.00
  trial 2: prior=0.500  P(e|H)=0.9  P(e|not-H)=0.1  n_signals=2  log LR=+2.20
  trial 3: prior=0.500  P(e|H)=0.1  P(e|not-H)=0.9  n_signals=2  log LR=-2.20
  trial 4: prior=0.950  P(e|H)=0.5  P(e|not-H)=0.5  n_signals=1  log LR=+0.00
  trial 5: prior=0.050  P(e|H)=0.5  P(e|not-H)=0.5  n_signals=1  log LR=+0.00
  trial 6: prior=0.600  P(e|H)=0.5  P(e|not-H)=0.5  n_signals=1  log LR=+0.00
  trial 7: prior=0.500  P(e|H)=0.6  P(e|not-H)=0.4  n_signals=1  log LR=+0.41
  trial 8: prior=0.600  P(e|H)=0.9  P(e|not-H)=0.1  n_signals=2  log LR=+2.20
  trial 9: prior=0.400  P(e|H)=0.9  P(e|not-H)=0.1  n_signals=2  log LR=+2.20
  trial 10: prior=0.600  P(e|H)=0.1  P(e|not-H)=0.9  n_signals=2  log LR=-2.20
  trial 11: prior=0.950  P(e|H)=0.6  P(e|not-H)=0.4  n_signals=1  log LR=+0.41
  trial 12: prior=0.750  P(e|H)=0.75  P(e|not-H)=0.25  n_signals=1  log LR=+1.10

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    d = data.copy()
    d['pk'] = d['prior'].astype(float).round(3)
    d['lk'] = d['log_lr'].astype(float).round(2)
    single = (d['lk'].abs() < 1e-6) | ((d['pk'] - 0.5).abs() < 1e-6)
    d = d[single]
    if len(d) == 0:
        return float('nan')
    r = d['response_bin'].astype(float)
    med = d.groupby(['subject_id', 'pk', 'lk'])['response_bin'].transform('median').astype(float)
    far = (np.abs(r - med) > 35.0).astype(float)
    return float(far.mean())

```

**Observed (real) value:** 0.0869 (var=0.0029)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.0886 (var=0.0025) (Δ vs real +0.0017)
**Other theories' values on this metric (for reference):**
- pi_5: 0.0954 (var=0.0029)
- pi_4: 0.0411 (var=0.0011)
- pi_1: 0.0891 (var=0.0029)
- pi_2: 0.0709 (var=0.0032)
- pi_3: 0.0697 (var=0.0024)

### Experiment 8
**Design**
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

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    d = data.copy()
    pr = d['prior'].astype(float).clip(1e-6, 1 - 1e-6).values
    L0 = np.log(pr) - np.log1p(-pr)
    ll = d['log_lr'].astype(float).values
    rb = d['response_bin'].astype(float).values
    rc = np.clip(rb, 0.5, 99.5) / 100.0
    d['_x'] = np.log(rc) - np.log1p(-rc)
    d['_cell'] = d['prior'].astype(float).round(4).astype(str) + '_' + d['log_lr'].astype(float).round(4).astype(str)
    inf_p = np.abs(L0) > 1e-6
    inf_e = np.abs(ll) > 1e-6
    conflict = inf_p & inf_e & (np.sign(L0) != np.sign(ll))
    big = conflict & (np.abs(ll) >= 1.9) & (np.abs(L0) >= 1.9)
    ctrl = ~conflict
    d['_big'] = big
    d['_ctrl'] = ctrl
    bins = np.arange(101, dtype=float)
    bc = np.clip(bins, 0.5, 99.5) / 100.0
    bx = np.log(bc) - np.log1p(-bc)
    thr = 1.8

    def cell_stats(cd):
        med = float(np.median(cd['response_bin'].astype(float).values))
        mc = min(max(med, 0.5), 99.5) / 100.0
        m = np.log(mc) - np.log1p(-mc)
        far = np.abs(cd['_x'].values - m) >= thr
        width = float(np.sum(np.abs(bx - m) >= thr))
        return int(far.sum()), len(cd), width

    vals = []
    for _, sd in d.groupby('subject_id'):
        cf = 0.0
        cw = 0.0
        for _, cd in sd[sd['_ctrl']].groupby('_cell'):
            nf, n, w = cell_stats(cd)
            cf += nf
            cw += n * w
        bf = 0.0
        bn = 0.0
        bw = 0.0
        for _, cd in sd[sd['_big']].groupby('_cell'):
            nf, n, w = cell_stats(cd)
            bf += nf
            bn += n
            bw += n * w
        if bn < 1 or cw <= 0:
            continue
        dens = cf / cw
        obs = bf / bn
        exp = dens * (bw / bn)
        v = obs - exp
        if np.isfinite(v):
            vals.append(v)
    if len(vals) == 0:
        return 0.0
    return float(np.mean(vals))

```

**Observed (real) value:** -0.0223 (var=0.0048)
**Previous candidate values (this loop):**
  - iter 1 (most recent): -0.0191 (var=0.0073) (Δ vs real +0.0032)
**Other theories' values on this metric (for reference):**
- pi_4: 0.0792 (var=0.0152)
- pi_5: -0.0198 (var=0.0099)
- pi_1: 0.0003 (var=0.0116)
- pi_2: -0.0341 (var=0.0092)
- pi_3: 0.0417 (var=0.0100)

## PREVIOUS CANDIDATE (this loop)
The RUNNING-BEST (last ACCEPTED) candidate in this critique loop — i.e. the source the loop's accept gate kept as the best base so far. If your most recent attempt was REJECTED by the gate, this is NOT that attempt; it is the previously-accepted base the gate rolled back to. Iterate on this source — the next critic feedback should be applied on top of it.

**Description**
Sub-additive deterministic integration, salient-value response mapping, and trait-level off-task responding.

(1) ENGAGED INTEGRATION (deterministic, unimodal)
The stated prior and the evidence are weighted in log-odds: x = alpha*logit(prior) and y = beta*LLR.
- alpha is near 1 and beta is below 1, so evidence is underweighted relative to the prior.
- How the two cues combine depends on their signs, and the combination is sub-additive:
  - Congruent cues are pooled by a p-norm, |L| = (|x|^p + |y|^p)^(1/p) with p > 1. Agreeing cues add less than their sum, which keeps congruent extreme cells off certainty without any explicit output bound.
  - Conflicting cues give a proportionally shrunk difference, L = s_conf*(x + y) with s_conf < 1.
  - A single informative cue passes through unchanged.
- There is no cue selection, so conflict cells are unimodal.

(2) SALIENT-VALUE RESPONSE MAPPING
The integrated value is perturbed by constant Gaussian noise in log-odds and lands in a 1% bin. With probability g, the report is then snapped to the nearest salient value. The salient grid is {0, 1, 5, 10, ..., 95, 99, 100}.
- The snap probability rises with extremity: g = g_c + (g_e - g_c)*|2x - 1|^2.
- The grid is very coarse in log-odds near the extremes (90 -> 95 -> 99 -> 100). Extreme cells therefore collapse onto one repeated value, such as 95 or 99, giving near-zero raw within-cell deviation.
- Mid-range reports stay effectively continuous, with log-odds variability roughly constant.
- The 99 grid point, rather than 100, keeps exact-endpoint reports rare.

(3) TRAIT-LEVEL OFF-TASK RESPONDING
A subject-specific mass lambda of off-task reports is independent of the stimulus, so it is equal in single-source, congruent and conflict cells. It has three parts:
- uniform guessing;
- polarity reversal (100 - R), a larger share than in earlier theories;
- 'don't know' 50% reports.

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
    p_sub = max(float(parameters["p_sub"]), 1.0)
    s_conf = float(parameters["s_conf"])
    sigma = max(float(parameters["sigma"]), 1e-3)
    g_c = float(np.clip(parameters["g_c"], 0.0, 1.0))
    g_e = float(np.clip(parameters["g_e"], 0.0, 1.0))
    lam = float(np.clip(parameters["lam"], 0.0, 1.0))
    f_rev = float(np.clip(parameters["f_rev"], 0.0, 1.0))
    f_half = float(np.clip(parameters["f_half"], 0.0, 1.0))
    tot = f_rev + f_half
    if tot > 1.0:
        f_rev /= tot
        f_half /= tot
    f_uni = max(0.0, 1.0 - f_rev - f_half)

    # --- sub-additive deterministic integration ---
    x = alpha * L0
    y = beta * llr
    prod = x * y
    cong = prod > 0
    conf = prod < 0
    ax = np.abs(x)
    ay = np.abs(y)
    Lcong = np.sign(x + y) * (ax ** p_sub + ay ** p_sub) ** (1.0 / p_sub)
    Lconf = s_conf * (x + y)
    L = np.where(cong, Lcong, np.where(conf, Lconf, x + y))

    # --- continuous noisy latent over 1% bins (logit-Gaussian) ---
    edges = (np.arange(K + 1) - 0.5) / (K - 1)
    z_edges = np.empty(K + 1)
    z_edges[0] = -np.inf
    z_edges[-1] = np.inf
    inner = edges[1:-1]
    z_edges[1:-1] = np.log(inner) - np.log1p(-inner)
    cdf = ndtr((z_edges[None, :] - L[:, None]) / sigma)
    cont = np.clip(np.diff(cdf, axis=1), 0.0, None)
    cont = cont / np.maximum(cont.sum(axis=1, keepdims=True), 1e-300)

    # --- salient-value snapping ---
    grid = np.array([0, 1] + list(range(5, 100, 5)) + [99, 100], dtype=float)
    is5 = (np.mod(grid, 5) == 0).astype(float)
    ks = np.arange(K, dtype=float)
    dist = np.abs(ks[:, None] - grid[None, :]) + 0.01 * (1.0 - is5[None, :])
    target = grid[np.argmin(dist, axis=1)].astype(int)
    S = np.zeros((K, K))
    S[np.arange(K), target] = 1.0
    ext = np.abs(2.0 * ks / 100.0 - 1.0) ** 2
    g = g_c + (g_e - g_c) * ext
    g = np.clip(g, 0.0, 1.0)
    snapped = (cont * g[None, :]) @ S
    eng = cont * (1.0 - g[None, :]) + snapped
    eng = eng / np.maximum(eng.sum(axis=1, keepdims=True), 1e-300)

    # --- trait-level off-task responding ---
    rev = eng[:, ::-1]
    spike = np.zeros(K)
    spike[50] = 1.0
    uni = np.full(K, 1.0 / K)
    probs = (1.0 - lam) * eng + lam * (f_uni * uni[None, :] + f_rev * rev + f_half * spike[None, :])
    probs = np.clip(probs, 0.0, None)
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
- alpha: [0.82, 1.02]
- beta: [0.4, 0.8]
- p_sub: [1.2, 2.0]
- s_conf: [0.7, 1.0]
- sigma: [0.3, 0.7]
- g_c: [0.15, 0.5]
- g_e: [0.75, 1.0]
- lam: [0.1, 0.38]
- f_rev: [0.1, 0.3]
- f_half: [0.02, 0.1]

`rationale`: Where this improves on pi_5. pi_5 fits Experiments 1 and 5–8 well but misses three: Exp2 (-1.10 vs -1.50), Exp3 (0.78 vs 0.67) and Exp4 (0.21 vs 0.15). Each change below targets one of these.

1. Exp3: integration weights and conflict shrinkage.
- The Exp3 index is roughly (1 + r - s_conf*(1 - r))/2, where r = beta/alpha.
- With mean alpha ≈ 0.92, beta ≈ 0.6 (r ≈ 0.65) and s_conf ≈ 0.85, it comes out near 0.68.
- pi_5 had both a higher r and a compressing bound on the prior-only cells, which pushed its index to 0.78.
- The bound is replaced by congruent p-norm sub-additivity. This keeps congruent near-certain cells (e.g. 0.95 prior with 3.89 LLR) at about 98–99% without compressing single-source cells.

2. Exp2: salient-value snapping.
- Snapping probability rises toward the extremes, and the grid is coarse there in log-odds (90/95/99/100).
- Extreme cells therefore collapse onto a repeated value (95 for a 0.95 prior; 99 for congruent near-certain cells). Their median raw deviation goes to about 0.
- Mid cells keep continuous logit noise with a raw MAD of about 0.07.
- This gives a ratio near exp(-1.5).
- Including 99 (rather than 100) in the grid keeps exact-100 reports to what the shared rounding stage produces, so the Exp1 endpoint ratio stays about 0.15.

3. Exp4: no heteroscedastic noise.
- Log-odds noise is constant. Quantisation onto a 5%-grid makes log-odds steps larger at 85–95 than near 50.
- This yields only a mild increase in log-odds MAD for the high cells, rather than pi_3's large heteroscedastic jump.

4. Off-task mixture: partial departure from the arbiter.
Per unit of mass, the components hit the off-zones as follows:
- In Exp6 (≤40 in high cells): mirror reports hit 100% of the time, uniform guesses 41%.
- In Exp5's zones: mirror about 70%, uniform about 41%.
- In Exp7's far-from-median criterion: mirror about 57%, uniform about 33%, 50%-reports about 30%.
The observed values are Exp5 ≈ Exp6 ≈ 0.12 and Exp7 ≈ 0.087.
- A mirror-dominated mixture would push Exp5 well below Exp6, which contradicts the data.
- I therefore follow the arbiter in direction but not in degree. Mirror and 50% shares are raised (mirror about 20%, versus about 7.5% in pi_5). Uniform guessing is reduced to about 70%.
- With lambda ≈ 0.24, this predicts approximately Exp6 0.12, Exp7 0.09 and Exp5 0.105.
- The mirror component also makes Exp8 slightly negative, as observed. Mirrored moderate conflict responses (about 70% becoming 30%) fall within the 1.8-logit threshold, while mirrored extreme control responses fall outside it.
- Off-task mass is stimulus-independent, so conflict cells show no extra bimodality.

Discriminating predictions.
- Snapping implies within-subject repeat identity at 95 and 99 in extreme single-source cells. A tanh bound instead implies continuous compression.
- Congruent sub-additivity should appear even in moderate congruent cells (e.g. 0.75 prior with 1.1 LLR, reported below Bayesian additivity), where a bound would be inactive.

## PRIOR FEEDBACK ITERATIONS
The critic's verdicts on each previous in-loop candidate, in order. Each block ends with an **Outcome of this advice** line saying whether the candidate the proposer produced AFTER this advice was ACCEPTED (loss strictly beat the running best) or REJECTED (the gate discarded it). Address the most recent iteration's feedback in your next edit, but down-weight past advice whose candidates were REJECTED.

### Iteration 1 (most recent — address this)
Verdict: regenerate
Interpretation: The candidate follows the prescribed family closely and was accepted (loss 0.1444). It has deterministic unimodal sub-additive integration, salient-value snapping that rises with extremity, and stimulus-independent off-task mass with a larger mirror share.

It fits most experiments well:
- **Exp5:** 0.124 vs 0.124, exact.
- **Exp7:** 0.089 vs 0.087.
- **Exp8:** -0.019 vs -0.022, with the correct negative sign from the mirror component.
- **Exp2:** -1.39 vs -1.50. Snapping fixes pi_5's weakness (-1.10).
- **Exp3:** 0.714 vs 0.671, better than pi_4 and pi_5. It is still slightly high, and the between-subject variance is far too small (0.02 vs 0.17).
- **Exp6:** 0.106 vs 0.12, slightly low.
- **Exp1:** 0.118 vs 0.14, slightly low.

The clear failure is **Exp4**: 0.44 vs 0.15, worse than pi_4 (0.13) and pi_5 (0.21). The 'high' cells in Exp4 (0.5/2.2, 0.9/0, 0.75/1.1, 0.875/±1.95) have latent posteriors around 80–92%. At those values the extremity weight |2x-1|^2 is already about 0.4–0.7, so g is substantial. The 5-point grid there is coarse in log-odds (logit 85 = 1.73, 90 = 2.20, 95 = 2.94). Snapped and unsnapped trials therefore sit at log-odds 0.4–0.7 from the cell median, inflating the high-cell MAD. Near 50 the grid steps are only about 0.2 logits, so the low-cell MAD stays small. The same quantization that helps Exp2 is over-applied in the 80–92% band.
Rationale: Keep the mechanism family (sub-additive deterministic integration + salient-value snapping + trait-level mirror/50/uniform off-task). Retune where snapping bites; do not add bounds or cue selection.

(1) Make snap probability rise much more steeply, concentrated at the true extremes (≥95 / ≤5):
- Replace |2x-1|^2 with |2x-1|^q, either fixing q≈4–6 or adding q in [3, 8] as a free parameter.
- Lower g_c to [0.0, 0.25].
- This leaves the 80–92% band mostly continuous, bringing Exp4 back toward 0.15. Exp2's extreme cells (95, 99) still collapse onto single values.

(2) Optionally make snapping consistent within a subject-cell: snap to the grid point nearest the noiseless latent, rather than the nearest to each noisy draw. This yields repeat identity, which helps Exp2 further without adding off-median scatter in Exp4.

(3) Make small adjustments to recover Exp1 and Exp6:
- Exp1 (0.118 vs 0.14): let snapping to 100 occur from bins ≥98, or add a small endpoint share.
- Exp6 (0.106 vs 0.12): nudge the lambda range up slightly (e.g. [0.14, 0.40]). Hold the mirror fraction where it is, because Exp5 and Exp8 already match.

(4) Exp3 between-subject variance is far too low (0.02 vs 0.17):
- Widen the beta range (e.g. [0.3, 0.95]) so integration weights vary more across subjects.
- Lower the beta midpoint slightly (by about 0.03), or reduce s_conf toward [0.65, 0.95], to pull the Exp3 point estimate from 0.71 toward 0.67.

**Outcome of this advice:** iter 1 candidate loss=0.1444 -> next candidate accept-status=PENDING (this is the advice you are addressing now).


## IMPLEMENTATION GUARDRAILS
- The parameters should be within the specified ranges.
- The model's predictions should be valid probability distributions (non-negative and sum to 1).
- When converting logits to probabilities via softmax, always use the numerically stable form: subtract the max before exponentiating (`x = x - np.max(x); p = np.exp(x); p /= p.sum()`). A naive `np.exp(x) / np.sum(np.exp(x))` overflows to Inf/NaN for large logits. Alternatively, use `scipy.special.softmax`.


## PROPOSAL — MINIMAL-DIFF EDIT (do NOT rewrite from scratch)
The RUNNING-BEST (last ACCEPTED) candidate is shown verbatim above under `## PREVIOUS CANDIDATE (this loop)`. This is the base the loop's accept gate is currently keeping; the most recent critic feedback (see `## PRIOR FEEDBACK ITERATIONS`) should be applied on top of it. Apply the SMALLEST edit that addresses the critic's diagnosis while staying inside the arbiter's prescribed mechanism family:
  - Re-emit the previous source verbatim, then change ONLY the lines needed to address the critic (a parameter range, a normalization, a softmax temperature, an attention scheme, a gating term, a buggy indexing line, etc.).
  - Keep all unaffected functions, equations, parameter names, and the overall mechanism intact.
  - Do NOT rewrite `predict` / `policy` end-to-end if a few lines would do, and do NOT switch mechanism families — that is the arbiter's decision, not yours in this loop.
  - Briefly explain the minimal edit in `rationale`.
If you genuinely believe a larger rewrite is required, you may do one, but justify in `rationale` why the minimal-diff path was insufficient.

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
  "theory": "Sub-additive deterministic integration, salient-value response mapping, and trait-level off-task responding.\n\n(1) ENGAGED INTEGRATION (deterministic, unimodal)\nThe prior and the evidence are weighted in log-odds: x = alpha*logit(prior) and y = beta*LLR.\n- alpha is about 1 and beta is below 1 (conservatism), with beta varying widely across subjects.\n- Congruent cues pool by a p-norm, |L| = (|x|^p + |y|^p)^(1/p) with p > 1, so agreeing cues are sub-additive.\n- Conflicting cues give a proportionally shrunk difference, L = s_conf*(x + y).\n- A single informative cue passes through unchanged.\n- There is no cue selection, so conflict cells are unimodal.\n\n(2) SALIENT-VALUE RESPONSE MAPPING, CONCENTRATED AT THE TRUE EXTREMES\nThe latent value is perturbed by constant Gaussian noise in log-odds and lands in a 1% bin.\n- With probability g, the report is snapped to the nearest salient value in {0, 1, 5, 10, ..., 95, 99, 100}.\n- Ties go toward the more extreme value, so 97 snaps to 99 and 3 snaps to 1.\n- g rises steeply with extremity: g = g_c + (g_e - g_c)*|2x - 1|^q with q of about 4-8.\n- Near-certain reports (at or above 95%, or at or below 5%) therefore collapse onto 95 or 99.\n- The 80-92% band and the mid-range stay nearly continuous, giving roughly constant log-odds variability.\n\n(3) TRAIT-LEVEL OFF-TASK RESPONDING\nA subject-specific mass lambda is independent of the stimulus, so it is equal across single-source, congruent and conflict cells. It mixes three parts:\n- uniform guessing;\n- polarity reversal (100 - R);\n- 'don't know' reports of 50%.",
  "predict": "def predict(parameters, stimulus, history):\n    import numpy as np\n    from scipy.special import ndtr\n    K = 101\n    scalar = np.ndim(stimulus[\"prior\"]) == 0\n    prior = np.clip(np.atleast_1d(np.asarray(stimulus[\"prior\"], dtype=float)), 1e-6, 1 - 1e-6)\n    llr = np.atleast_1d(np.asarray(stimulus[\"log_lr\"], dtype=float)) * np.ones_like(prior)\n    L0 = np.log(prior) - np.log1p(-prior)\n\n    alpha = float(parameters[\"alpha\"])\n    beta = float(parameters[\"beta\"])\n    p_sub = max(float(parameters[\"p_sub\"]), 1.0)\n    s_conf = float(parameters[\"s_conf\"])\n    sigma = max(float(parameters[\"sigma\"]), 1e-3)\n    g_c = float(np.clip(parameters[\"g_c\"], 0.0, 1.0))\n    g_e = float(np.clip(parameters[\"g_e\"], 0.0, 1.0))\n    q_snap = max(float(parameters[\"q_snap\"]), 1.0)\n    lam = float(np.clip(parameters[\"lam\"], 0.0, 1.0))\n    f_rev = float(np.clip(parameters[\"f_rev\"], 0.0, 1.0))\n    f_half = float(np.clip(parameters[\"f_half\"], 0.0, 1.0))\n    tot = f_rev + f_half\n    if tot > 1.0:\n        f_rev /= tot\n        f_half /= tot\n    f_uni = max(0.0, 1.0 - f_rev - f_half)\n\n    # --- sub-additive deterministic integration ---\n    x = alpha * L0\n    y = beta * llr\n    prod = x * y\n    cong = prod > 0\n    conf = prod < 0\n    ax = np.abs(x)\n    ay = np.abs(y)\n    Lcong = np.sign(x + y) * (ax ** p_sub + ay ** p_sub) ** (1.0 / p_sub)\n    Lconf = s_conf * (x + y)\n    L = np.where(cong, Lcong, np.where(conf, Lconf, x + y))\n\n    # --- continuous noisy latent over 1% bins (logit-Gaussian) ---\n    edges = (np.arange(K + 1) - 0.5) / (K - 1)\n    z_edges = np.empty(K + 1)\n    z_edges[0] = -np.inf\n    z_edges[-1] = np.inf\n    inner = edges[1:-1]\n    z_edges[1:-1] = np.log(inner) - np.log1p(-inner)\n    cdf = ndtr((z_edges[None, :] - L[:, None]) / sigma)\n    cont = np.clip(np.diff(cdf, axis=1), 0.0, None)\n    cont = cont / np.maximum(cont.sum(axis=1, keepdims=True), 1e-300)\n\n    # --- salient-value snapping (steep in extremity; ties go to the more extreme value) ---\n    grid = np.array([0, 1] + list(range(5, 100, 5)) + [99, 100], dtype=float)\n    ks = np.arange(K, dtype=float)\n    dist = np.abs(ks[:, None] - grid[None, :]) - 0.001 * np.abs(grid[None, :] - 50.0) / 50.0\n    target = grid[np.argmin(dist, axis=1)].astype(int)\n    S = np.zeros((K, K))\n    S[np.arange(K), target] = 1.0\n    ext = np.abs(2.0 * ks / 100.0 - 1.0) ** q_snap\n    g = g_c + (g_e - g_c) * ext\n    g = np.clip(g, 0.0, 1.0)\n    snapped = (cont * g[None, :]) @ S\n    eng = cont * (1.0 - g[None, :]) + snapped\n    eng = eng / np.maximum(eng.sum(axis=1, keepdims=True), 1e-300)\n\n    # --- trait-level off-task responding ---\n    rev = eng[:, ::-1]\n    spike = np.zeros(K)\n    spike[50] = 1.0\n    uni = np.full(K, 1.0 / K)\n    probs = (1.0 - lam) * eng + lam * (f_uni * uni[None, :] + f_rev * rev + f_half * spike[None, :])\n    probs = np.clip(probs, 0.0, None)\n    probs = probs / probs.sum(axis=1, keepdims=True)\n    return probs[0] if scalar else probs\n",
  "policy": "def policy(probs):\n    import numpy as np\n    p = np.asarray(probs, dtype=np.float64)\n    p = np.clip(p, 0.0, None)\n    p /= p.sum()\n    return int(np.random.choice(len(p), p=p))\n",
  "parameters": {
    "alpha": "[0.82, 1.02]",
    "beta": "[0.3, 0.81]",
    "p_sub": "[1.2, 2.0]",
    "s_conf": "[0.7, 1.0]",
    "sigma": "[0.3, 0.7]",
    "g_c": "[0.0, 0.2]",
    "g_e": "[0.75, 1.0]",
    "q_snap": "[4.0, 8.0]",
    "lam": "[0.12, 0.4]",
    "f_rev": "[0.1, 0.3]",
    "f_half": "[0.02, 0.1]"
  },
  "rationale": "This is a minimal-diff edit of the accepted iter-1 candidate. The mechanism family is unchanged: sub-additive deterministic integration, salient-value snapping and trait-level off-task responding.\n\n1. Exp4 (0.44 vs 0.15), the main failure. Per-draw snapping was too strong in the 80-92% band.\n- At those values |2x-1|^2 is about 0.4-0.7, so g was about 0.6.\n- Draws were scattered onto the log-odds-coarse 85/90/95 grid, inflating the high-cell log-odds MAD.\n- The extremity weight is now |2x-1|^q with a new parameter q_snap in [4, 8], and g_c is lowered to [0, 0.2].\n- In the 85-92% band, g falls to about 0.15-0.35. Bins of 97% or more keep g above 0.7.\n- I kept per-draw snapping on purpose. A little residual snapping in the high band is what lifts the log-odds MAD modestly above the mid band; pi_1 with no snapping gives 0.07.\n- I did not switch to latent-consistent snapping. My estimate is that it would drive the high-cell MAD below the low-cell MAD, giving a negative ratio.\n- The ratio should therefore fall between pi_1 (0.07) and the old 0.44, near 0.15-0.2.\n\n2. Protecting Exp2 under steeper snapping.\n- Grid ties now resolve toward the more extreme value: 97 goes to 99 instead of 95.\n- The congruent near-certain cells (latent about 98%, draws 96-100) therefore collapse onto 99 rather than splitting between 95 and 99. Their within-cell raw deviation stays about 0.\n- These two cells dominate the pooled extreme-cell median.\n- Lowering g_c also de-clusters the mid cells (e.g. 0.5/0 no longer piles at 50), which raises the mid-cell MAD. Both effects push Exp2 toward -1.5.\n\n3. Exp1 (0.118 vs 0.14).\n- Sending 97 to 99 creates more 99 reports.\n- The shared rounding stage turns 99 into 100, which slightly raises the endpoint share.\n\n4. Exp3 (0.714 vs 0.671; variance 0.02 vs 0.17).\n- The index is about (r + 1 - s + s*r)/2 with r = beta/alpha, so its slope in r is about 0.92.\n- The beta midpoint is lowered from 0.60 to about 0.555 to cut about 0.04 from the index.\n- The range is widened to [0.3, 0.81] to raise between-subject variance.\n\n5. Exp6 (0.106 vs 0.12).\n- The lambda range is nudged up to [0.12, 0.40].\n- The mirror, 50% and uniform shares are held, because Exp5, Exp7 and Exp8 already match.\n\nDiscriminating predictions are unchanged.\n- There should be repeat identity at 95 and 99 in near-certain cells, but continuous reporting at 80-92%. This band-specific clustering distinguishes snapping from a tanh bound.\n- Congruent sub-additivity should appear in moderate congruent cells where a bound would be inactive."
}
```
```

## Usage

```json
{
  "input_tokens": 27557,
  "output_tokens": 16431,
  "reasoning_tokens": 12574,
  "provider": "Claude Platform on AWS"
}
```
