# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Probability Updating domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_1" and "pi_2") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_1" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_2" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects are told a prior probability that hypothesis H is true (e.g. that a ball was drawn from urn A rather than urn B), then shown evidence whose probability under H and under not-H is known (e.g. the colours of several draws, with the urns' compositions stated). They report the probability that H is true on a 0-100% scale.

## THEORY 1 — pi_1
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


## THEORY 2 — pi_2
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


## EXPERIMENT 1 (proposed by pi_1)

### DESIGN
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

**Rationale:** The design targets how response variability scales with the level of the distorted belief, while keeping the mean model (alpha times prior log-odds plus beta times LLR) identical in both theories. Prior log-odds and evidence LLR are crossed independently so alpha and beta are identifiable. Each trial uses a single signal (n=1) so the LLR is simply log(p_h/p_not_h), with values 0, +/-1.39, +/-2.94 and +/-4.6. Priors range over 0.01, 0.05, 0.2, 0.5, 0.8, 0.95 and 0.99.

The cells are:
(a) A pure 50% cell, where both theories predict maximal spread.
(b) Evidence-only cells at prior 0.5.
(c) Prior-only cells at 0.01 and 0.99 with uninformative evidence.
(d) Congruent extreme cells (0.99 with 0.99/0.01, 0.01 with 0.01/0.99, 0.8 with 0.95/0.05, 0.05 with 0.2/0.8). These push the distorted belief toward 0 or 100.
(e) Conflict cells (0.95 with 0.05/0.95, 0.2 with 0.8/0.2). These land near 50% but via large opposing components.

Advocated theory (logit-normal noise with constant sigma): the probability-scale SD is about sigma*p(1-p). It collapses quickly toward the extremes, so the SD at 95% is roughly 0.19 times the SD at 50%. The median sits at expit(mu) and the mean is pulled toward 50%. Exact 0% or 100% reports should be essentially absent, apart from lapses.

Competing theory (binomial sampling): the SD is sqrt(p(1-p)/N), so the SD at 95% is about 0.44 times the SD at 50%, a much flatter profile. The mean equals the target exactly. Extreme cells should produce a large mass of exact 0/100 (k=0 or k=N) reports, e.g. more than 50% of responses when the target is 97% and N=20.

Key diagnostics, all robust to rounding to 5 or 10%:
- The ratio of spread (IQR or SD) in the extreme cells versus the 50% and conflict cells.
- The frequency of 0/100 endpoint reports in extreme cells relative to the lapse baseline.
- Whether conflict cells, which share a near-50% belief, show the same variability as the pure 50% cell. Both theories say they should, which checks that variance depends only on the belief level.

The advocated theory predicts steep variance compression and few endpoint reports. The competing theory predicts a shallow sqrt-shaped profile and heavy endpoint mass. Twelve unique trials give K=5 repeats each.

**Schedule:** 12 unique x 5 reps = 60 trials per subject.


### METRIC
Rationale:
In the moderately extreme cells (Bayes posterior 0.95-0.99 or 0.01-0.05: trials 3,4,5,6,11,12) the distorted belief typically sits around 91-98%. I fold responses toward the favoured side and, among responses in the near-extreme window (>=90%), compute the share that are exactly at the endpoint (100% / 0%). Logit-normal noise (advocated) is a smooth distribution in log-odds that can't pile up at the boundary: its mode sits at expit(mu) (e.g. 96-98%), and reaching the 100 bin requires z > logit(0.995)=5.3, so endpoint share stays low (mostly coming from the shared rounding stage). Binomial sampling (competing) puts mass at k=N, and its Gaussian motor smear tau has open outer bin edges, so every bit of smeared mass above 99.5% lands in the 100 bin; its folded near-extreme distribution therefore has its mode AT the endpoint. The conditional ratio (endpoint / all >=90) removes most dependence on alpha/beta and lapse rate (lapses contribute only ~11/101 of their mass to the window), and the shared rounding affects both theories identically. Pooling 30 trials per subject keeps per-subject estimates reasonably stable, with expected values roughly 0.2-0.3 for logit-normal versus ~0.45-0.6 for binomial-sampling.

Source:
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


### RESULTS
- Predicted under pi_1 (simulated): 0.1675 (var=0.0068)
- Predicted under pi_2 (simulated): 0.3688 (var=0.0217)
- Observed on real data: 0.1400 (var=0.0062)

## EXPERIMENT 2 (proposed by pi_2)

### DESIGN
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

**Rationale:** This design tests the advocated theory's distinctive signature: binomial report variability around a log-odds-distorted belief. The alternative is logit-normal noise with constant sigma around the same distorted belief. Both theories share the mean structure alpha*prior log-odds + beta*LLR. The design therefore crosses priors (0.05, 0.2, 0.5, 0.8, 0.95) with single-draw evidence (LLR 0, ±1.10, ±2.20, ±3.89) so alpha and beta are identified separately from the variability structure.

The design deliberately places many cells where the distorted belief lands in the 75-95% (or 5-25%) band rather than at the far extremes. That is where the two variance laws separate most clearly while the means remain matched.

Binomial prediction (advocated): SD = sqrt(p(1-p)/N). For example, with N≈20 the SD is about 0.067 at p=0.9 versus 0.112 at 0.5, a ratio of about 0.6.

Logit-normal prediction (competing): SD ≈ sigma*p(1-p), so the same comparison gives a ratio of about 0.36. Its spread therefore collapses much faster away from 50%.

Congruent cells (0.8 with 0.9/0.1; 0.95 with 0.98/0.02; mirrored low cells) push the belief toward the ends. There, binomial sampling predicts a substantial pile-up of exact 0%/100% reports (k=0 or k=N), e.g. P(k=N)=0.9^20≈12% at p=0.9, and more at higher p. Logit-normal noise yields a smooth tail with endpoint mass that grows only slowly and continuously. Exact 0 and 100 survive the 5/10% rounding stage, so endpoint frequency relative to the lapse floor is a rounding-robust diagnostic.

Prior-only cells (0.8, 0.95 with uninformative evidence) and evidence-only cells at prior 0.5 let us check that variability depends only on the level of the distorted belief, not on its source.

Conflict cells (0.95 with 0.1/0.9; 0.2 with 0.9/0.1) put the belief near 50-75% via opposing components. Both theories predict variance set by the belief level alone, which serves as a consistency check.

Key metrics:
(1) the slope of IQR/SD versus the model-implied belief level, comparing the binomial sqrt profile with the logit-normal p(1-p) profile;
(2) endpoint (0/100) frequency in congruent cells versus the 50% cell;
(3) mean versus median asymmetry. Binomial means equal the target, whereas logit-normal means are pulled toward 50% relative to medians.

Twelve unique trials give K=5 repeats each.

**Schedule:** 12 unique x 5 reps = 60 trials per subject.


### METRIC
Rationale:
Shape of the variance law, measured within subject as a ratio so that overall noise level (N, tau vs sigma), which overlaps between theories, cancels. For each subject x cell, take absolute deviations of responses from the cell median (robust to the large lapse rates), pool them over cells whose distorted belief is reliably extreme for ANY admissible alpha/beta (prior .95 with no evidence, .95 with LLR +3.89, .05 with LLR -3.89; belief ~0.90-0.998), and over mid-range cells (prior .5 with LLR 0 or +1.10, prior .2 with LLR +2.2; belief ~0.25-0.72). Metric = log((median extreme dev + .01)/(median mid dev + .01)). Logit-normal noise scales as sigma*p(1-p): spread at p~0.95-0.99 collapses to ~4-20% of mid spread (log ratio strongly negative, about -1.5 to -2). Binomial sampling scales as sqrt(p(1-p)/N) and the motor noise tau adds a probability-space floor that does not shrink near the ends, so extreme spread stays ~40-100% of mid spread (log ratio near -0.5 to 0). The +0.01 floor guards against 1-point discretization; the median-based deviations resist lapses and the 5/10% rounding stage affects both theories similarly.

Source:
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


### RESULTS
- Predicted under pi_1 (simulated): -1.6094 (var=0.6514)
- Predicted under pi_2 (simulated): -0.4055 (var=0.4605)
- Observed on real data: -1.5041 (var=0.4026)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
(no other experiments)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across both experiments the logit-normal account (pi_1) clearly beats the binomial-sampling account (pi_2).\n\nExperiment 1 (endpoint share among near-extreme responses):\n- pi_1 predicted 0.17 and pi_2 predicted 0.37.\n- The observed value was 0.14, very close to pi_1 and well below pi_2.\n- Subjects do not pile up at exact 0/100 in moderately extreme cells, as k=0 or k=N sampling would require. Their extreme responses form a smooth tail that stops short of the boundary.\n\nExperiment 2 (log ratio of spread in extreme versus mid-range cells):\n- pi_1 predicted -1.61 and pi_2 predicted -0.41.\n- The observed value was -1.50, again nearly on pi_1.\n- Spread collapses steeply toward the ends, as in the sigma*p(1-p) law. It does not follow the flat sqrt(p(1-p)/N) profile with a tau floor.\n\nThe two experiments probe different signatures: endpoint mass and the variance-compression profile. Both say the same thing: variability is approximately constant in log-odds space, not binomial in probability space.\n\nThe binomial reporting mechanism in pi_2 is therefore falsified twice. Its shared mean structure gives it nothing distinctive. Re-tuning N or tau will not rescue it: making N large enough to remove endpoint mass makes it indistinguishable from Gaussian noise, and the tau floor contradicts the observed compression.\n\nThe dataset has so far only tested the noise law, not pi_1's other strong commitments:\n- that the mean is a fixed linear function of prior and evidence log-odds (constant alpha and beta regardless of magnitude);\n- that log-odds noise is constant across stimuli and unrelated to the size of the distortion.\n\nThe productive next step is to replace pi_2 with a competitor that matches the confirmed logit-normal-like signature but breaks those two commitments.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Replace pi_2 with a noisy-log-odds-encoding theory in which distortion arises from noise.\n\nCore mechanism:\n- Prior log-odds and evidence LLR are each encoded with Gaussian noise in log-odds space.\n- That noise has a constant part plus a part that scales with magnitude (scalar/Weber variability): sd_i = s0 + w*|x_i|.\n- The subject combines the noisy encodings with Bayesian shrinkage toward 0 (a prior over log-odds with spread lambda), then reports expit of the decoded log-odds, plus small response noise and a lapse rate.\n\nWhat this predicts, consistent with the data so far:\n- Responses are roughly logit-normal, so the steep compression of probability-scale variance toward the ends and the scarcity of exact 0/100 reports are preserved.\n\nWhere it departs from pi_1:\n- Effective alpha and beta are not constants. Shrinkage grows with stimulus magnitude, so response log-odds are a concave, saturating function of |prior log-odds| and |LLR|. Large LLRs are underweighted proportionally more, so there is more conservatism at extreme evidence.\n- Log-odds variability grows with |prior log-odds| and |LLR|.\n- Conflict cells, which have large opposing components but a near-50% belief, show more log-odds variance than the pure 50% cell. pi_1 predicts equal variance there.\n- Mean distortion and variability are coupled: cells or subjects with more noise show more conservatism and base-rate neglect.\n\nParameters: s0, w, lambda (or separate shrinkage priors for prior and evidence), epsilon.\n\nSuggested discriminating experiments:\n- Vary LLR magnitude finely (e.g. 0.5, 1, 2, 3, 4.6) at prior 0.5, and vary prior log-odds at uninformative evidence. Test linearity of median response log-odds, which pi_1 says must be linear.\n- Compare log-odds IQR in conflict cells against the pure 50% cell, and in large- versus small-magnitude congruent cells.\n- Within subjects, test whether the per-cell log-odds spread correlates with shortfall from the Bayesian log-odds."
}
```

## Usage

```json
{
  "input_tokens": 9351,
  "output_tokens": 2160,
  "reasoning_tokens": 672,
  "provider": "Claude Platform on AWS"
}
```
