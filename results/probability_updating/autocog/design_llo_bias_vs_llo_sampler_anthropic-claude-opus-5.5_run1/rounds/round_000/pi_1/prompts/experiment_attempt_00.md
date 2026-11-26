# experiment_attempt_00

## System Prompt

You are a renowned cognitive scientist designing an experiment in the Probability Updating domain.

Your goal is to be an adversarial collaborator: propose a design whose outcomes would be predicted by your advocated theory but NOT by the competing theory. Both are provided below.

A useful proposal targets a *quantitative* dissociation between the two theories — how they respond differently to specific stimuli in addition to differences in overall performance.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects are told a prior probability that hypothesis H is true (e.g. that a ball was drawn from urn A rather than urn B), then shown evidence whose probability under H and under not-H is known (e.g. the colours of several draws, with the urns' compositions stated). They report the probability that H is true on a 0-100% scale.

A belief-updating experiment. Each trial states a prior P(H) and shows evidence whose probability under H (`trial_p_signal_h`) and under not-H (`trial_p_signal_not_h`) is known to the subject; `trial_n_signals` is how many draws make up that evidence. The subject reports P(H) from 0 to 100%. Choose the unique trials (all four lists, same length) so that the candidate theories make different predictions: vary the prior and the strength of the evidence (log likelihood ratio) independently, and include cells where theories disagree about both the mean report and its trial-to-trial variability. Each unique trial is repeated K = max(1, 60 // n_unique) times per subject in a fresh random order. There is no feedback. As in human data, every response then passes through a shared reporting stage that does not depend on the theory: with probability 0.13 it is rounded to the nearest multiple of 10%, otherwise with probability 0.10 to the nearest multiple of 5%. Theories should not model this rounding themselves, and metrics should not rely on exact values that rounding would blur.

Subjects see the following instructions:
On each trial you will be told how likely it is, before any evidence, that a statement H is true. You will then see some evidence, together with how likely that evidence is if H is true and if it is false. Report how likely you think it is now that H is true, from 0% to 100%.

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


## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
(none yet)

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- trial_priors: Prior P(H) for each unique trial, each strictly between 0 and 1.
- trial_p_signal_h: P(observed evidence | H) for each unique trial.
- trial_p_signal_not_h: P(observed evidence | not-H) for each unique trial.
- trial_n_signals: Number of signals (draws) making up each trial's evidence.


## Response

```json
{
  "rationale": "Goal: separate constant Gaussian noise in log-odds space (advocated) from binomial sampling noise around a log-odds-distorted belief (competing). I assume log LR = n_signals * log(p_h / p_not_h), with p_h and p_not_h giving the per-draw probability of the observed colour; all draws in a trial share that colour. There are 12 unique trials, so K = 5 repetitions each, enough to estimate within-cell spread.\n\n(1) Variability profile.\n- Advocated theory: spread is constant in log-odds. In probability units the SD is about sigma*p(1-p), but the distribution is skewed, with tails reaching well into the interior even when the central judgment is extreme.\n- Competing theory: SD in probability is sqrt(p(1-p)/N), so the log-odds SD blows up as 1/sqrt(N p(1-p)). Near 0 or 100, reports pile onto the endpoint with almost no spread.\n- Cells used to probe this:\n  - near-50% cells: prior .5 with LLR .41; prior .6 with LLR .20;\n  - conflict cells, where a strong prior is opposed by strong evidence: .1 vs +2.54; .9 vs -2.54;\n  - extreme consistent cells: prior .5 with LLR \u00b14.24; prior .9 with LLR +2.54; prior .1 with LLR -2.54; prior .3 with LLR +2.77.\n- Key metric: the ratio of within-cell spread at extreme cells to spread at mid cells, in probability units and in log-odds. The two theories predict opposite trends in log-odds, and a robust measure such as IQR or MAD survives the 5/10% rounding.\n\n(2) Mean shrinkage.\n- Advocated theory: averaging expit over log-odds noise pulls the probability-space mean toward 50% at extreme targets. A small sigma would remove this, but then the large interior tails would not appear either.\n- Competing theory: the binomial mean equals the distorted target exactly.\n\n(3) Prior weighting, decoupled from evidence.\n- Pure-prior cell: prior .8 with zero LLR (p_h = p_not_h).\n- Very extreme priors with weak opposing evidence: .02 and .98 with LLR \u22130.41.\n- These cells estimate alpha directly. The competing theory permits alpha > 1 (over-extremised priors, reports beyond Bayes); the advocated theory caps alpha near 1, predicting base-rate neglect or near-Bayesian use of the prior.\n\n(4) Independence of prior and evidence.\n- Prior and LLR are crossed independently (priors .02-.98; LLR from -4.24 to +4.24) so that alpha and beta are identifiable separately from the noise parameters.\n- Advocated prediction: the median report in log-odds is linear in prior log-odds and LLR, and its spread is constant across cells.",
  "trial_priors": [
    0.5,
    0.5,
    0.5,
    0.1,
    0.9,
    0.9,
    0.1,
    0.02,
    0.98,
    0.8,
    0.3,
    0.6
  ],
  "trial_p_signal_h": [
    0.6,
    0.7,
    0.3,
    0.7,
    0.3,
    0.7,
    0.3,
    0.6,
    0.4,
    0.5,
    0.8,
    0.55
  ],
  "trial_p_signal_not_h": [
    0.4,
    0.3,
    0.7,
    0.3,
    0.7,
    0.3,
    0.7,
    0.4,
    0.6,
    0.5,
    0.2,
    0.45
  ],
  "trial_n_signals": [
    1,
    5,
    5,
    3,
    3,
    3,
    3,
    1,
    1,
    2,
    2,
    1
  ]
}
```

## Usage

```json
{
  "input_tokens": 3693,
  "output_tokens": 2485,
  "reasoning_tokens": 1274,
  "provider": "Claude Platform on AWS"
}
```
