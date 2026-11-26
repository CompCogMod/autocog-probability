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


## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
[0] Both theories share the same mean structure (alpha*prior log-odds + beta*LLR), so the design targets the noise structure. The theories make three different predictions. (1) Shape of spread across belief levels. Constant log-odds noise gives an SD in probability space of about sigma*p(1-p), which collapses fast toward the extremes. Binomial sampling gives an SD of sqrt(p(1-p)/N), which shrinks only like the square root. So the ratio of SD at a distorted belief near 0.9-0.97 to SD at 0.5 is about 0.1-0.36 under the advocated theory versus about 0.17-0.6 under the sampling theory. (2) Shape and mean of the report distribution. Logit-normal noise yields skewed distributions with a tail toward 50%, and means in probability space are pulled toward 50% as sigma grows. Binomial means sit exactly at the target and are nearly symmetric for moderate N. (3) Extreme reports. With large sigma, logit noise puts real mass on 0%/100% when the belief is moderately extreme. The sampling theory puts mass at 0/100 only when Np(1-p) is small. Trial structure: priors (0.03 to 0.97) and LLRs (-2.94 to +3.89) are crossed rather than confounded. This includes zero-evidence trials at three priors (isolating alpha), evidence-only trials at prior 0.5 (isolating beta), congruent trials driving beliefs to the extremes, and conflicting trials landing near 50%. That gives several cells with similar distorted beliefs but different prior/LLR compositions, and others spanning the full belief range, so the SD-vs-mean curve can be traced. All evidence is given as a single aggregated signal (n_signals=1), so the likelihoods are unambiguous. With 12 unique trials each is repeated 5 times, which allows within-cell variance estimates. Rounding to 5/10% blurs exact values, so analyses should use SDs and interquartile spreads across cells rather than exact response values.

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
  "rationale": "Both theories share the mean structure expit(alpha*prior log-odds + beta*LLR), so this design targets how response variability scales with the distorted belief p. It differs from the earlier design in three ways: fewer unique trials (10, so K=6 repeats each) for tighter within-cell SD estimates, a parameter-free anchor cell, and denser sampling of the extreme tails, where the theories diverge most.\n\n(1) Anchor: prior 0.5 with uninformative evidence (P=0.5 under both H and not-H). The belief is exactly 0.5 whatever alpha and beta are. So the SD here estimates 0.5/sqrt(N) under binomial sampling, and about 0.25*sigma under logit-normal noise. Every other cell's SD is then normalised by this one.\n\n(2) Predicted normalised SD ratio. Under sampling it is 2*sqrt(p(1-p)): 0.6 at p=0.9, 0.28 at p=0.98, 0.2 at p=0.99. Under constant logit noise it is roughly 4p(1-p): 0.36, 0.08 and 0.04. The ratio is also skewed, with a long tail toward 50%. The gap widens sharply in the tails. That is why several cells drive beliefs to 0.9-0.99 and 0.01-0.1 even under strong conservatism:\n- strong single evidence: LLR +2.94 and -4.6 at prior 0.5;\n- extreme priors with null evidence: 0.9 and 0.02, which isolate alpha;\n- congruent prior and evidence: 0.95 with LR 9, 0.05 with LR 1/19, and 0.99 with LR 4.\n\n(3) Asymmetry and skew. Under logit noise, extreme cells show right-skewed (low side) or left-skewed (high side) reports with tails reaching toward 50%. Binomial sampling gives compact distributions that pile at 0 or 100 when Np(1-p) is small. Analyses should therefore compare tail-cell SDs and IQRs, and the fraction of reports at least 15 points from the cell median, normalised by the anchor cell. These spread-based measures are robust to the 5/10% rounding.\n\n(4) Moderate cells: LLR +1.1 at prior 0.5, and the conflict cell prior 0.2 with LR 4. These trace the middle of the SD-vs-belief curve. They also pin down beta and alpha for mapping each cell to its p.\n\nEvidence is a single aggregated signal (n_signals=1), so the likelihoods are unambiguous.",
  "trial_priors": [
    0.5,
    0.5,
    0.5,
    0.5,
    0.9,
    0.02,
    0.95,
    0.05,
    0.99,
    0.2
  ],
  "trial_p_signal_h": [
    0.5,
    0.75,
    0.95,
    0.01,
    0.5,
    0.5,
    0.9,
    0.05,
    0.8,
    0.8
  ],
  "trial_p_signal_not_h": [
    0.5,
    0.25,
    0.05,
    0.99,
    0.5,
    0.5,
    0.1,
    0.95,
    0.2,
    0.2
  ],
  "trial_n_signals": [
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1
  ]
}
```

## Usage

```json
{
  "input_tokens": 4367,
  "output_tokens": 2902,
  "reasoning_tokens": 1836,
  "provider": "Claude Platform on AWS"
}
```
