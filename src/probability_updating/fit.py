"""Maximum-likelihood fitting and comparison of probability-updating theories.

A theory is any autocog `Theory` whose `predict(parameters, stimulus, history)`
returns a distribution over the 101 response bins and broadcasts over a batch
stimulus (see `data.py`). Its `parameters` ranges ("[lo, hi]") double as the
optimisation bounds, so theories proposed or improved by autocog can be scored
on the human data with no extra code.

Every theory's prediction passes through the same reporting stage
(`ReportModel`) before it is scored:

1. Rounding. With probability `round10` the subject reports the nearest
   multiple of 10%, with probability (1 - round10) * round5 the nearest
   multiple of 5%, and otherwise the unrounded value. People pile answers onto
   round numbers; without this, a theory whose outputs happen to sit on round
   numbers (e.g. a sampler with N = 20 gives multiples of 5%) wins for that
   reason alone.
2. Response scale. Studies whose scale only allows multiples of `step`% (e.g.
   an 11-point 0-100 scale) get the predicted mass snapped onto those points.

Fits are pooled per study ("representative agent"): one parameter vector per
(theory, dataset/study). Out-of-sample fit uses K-fold cross-validation that
holds out whole subjects, so theories with more parameters get no free credit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import qmc

from src.probability_updating.data import GRID, N_BINS, stimulus_from_frame
from src.sample_parameters import resolve_parameters
from src.theory import Theory

LOG_FLOOR = 1e-12  # floor on per-trial likelihood so one impossible response can't dominate
ROUNDING_BOUNDS = {"round10": (0.0, 1.0), "round5": (0.0, 1.0)}


def parameter_bounds(theory: Theory) -> dict[str, tuple[float, float]]:
    """Optimisation bounds from the theory's declared "[lo, hi]" parameter ranges."""
    resolved = resolve_parameters(theory.parameters, {})
    bounds: dict[str, tuple[float, float]] = {}
    for name, spec in resolved.items():
        if isinstance(spec, dict) and set(spec) == {"interval"}:
            lo, hi = (float(x) for x in spec["interval"])
            bounds[name] = (min(lo, hi), max(lo, hi))
        else:
            raise ValueError(
                f"parameter {name!r} must be an interval '[lo, hi]' to be fitted; got {spec!r}"
            )
    return bounds


@lru_cache(maxsize=None)
def snap_matrix(step: int) -> np.ndarray:
    """(101, 101) matrix moving each bin's mass to the nearest multiple of `step` (ties split)."""
    m = np.zeros((N_BINS, N_BINS))
    targets = np.arange(0, N_BINS, step)
    for i in range(N_BINS):
        d = np.abs(targets - i)
        nearest = targets[d == d.min()]
        m[i, nearest] = 1.0 / len(nearest)
    return m


@dataclass(frozen=True)
class ReportModel:
    """How a theory's internal 0-100 report becomes the observed response."""

    step: int = 1          # response-scale resolution in percent (1 = any whole percent)
    rounding: bool = True  # fit round10 / round5 rounding probabilities

    @classmethod
    def for_data(cls, df: pd.DataFrame, *, rounding: bool = True) -> "ReportModel":
        """Infer the scale step as the gcd of the observed responses (10 for an 11-point scale)."""
        step = int(np.gcd.reduce(df["response_bin"].to_numpy().astype(int)))
        step = step if step in (2, 5, 10, 20, 25, 50) else 1
        # On a scale of 10% steps, rounding to 5 or 10 is the scale itself.
        return cls(step=step, rounding=rounding and step < 10)

    @property
    def bounds(self) -> dict[str, tuple[float, float]]:
        return dict(ROUNDING_BOUNDS) if self.rounding else {}

    def matrix(self, params: dict[str, float]) -> np.ndarray | None:
        m = None
        if self.rounding:
            r10, r5 = params["round10"], params["round5"]
            m = ((1 - r10) * (1 - r5)) * np.eye(N_BINS) + ((1 - r10) * r5) * snap_matrix(5) \
                + r10 * snap_matrix(10)
        if self.step > 1:
            m = snap_matrix(self.step) if m is None else m @ snap_matrix(self.step)
        return m


def predict_responses(theory: Theory, params: dict[str, float], stim: dict,
                      report: ReportModel) -> np.ndarray:
    """(n, 101) predicted distribution of the observed response."""
    theory_params = {k: v for k, v in params.items() if k not in ROUNDING_BOUNDS}
    probs = np.asarray(theory.predict(theory_params, stim, None), dtype=float)
    if probs.ndim == 1:
        probs = probs[None, :]
    m = report.matrix(params)
    return probs if m is None else probs @ m


def trial_log_lik(theory: Theory, params: dict[str, float], stim: dict, bins: np.ndarray,
                  report: ReportModel = ReportModel(rounding=False)) -> np.ndarray:
    probs = predict_responses(theory, params, stim, report)
    p = probs[np.arange(len(bins)), bins]
    return np.log(np.maximum(p, LOG_FLOOR))


@dataclass
class FitResult:
    params: dict[str, float]
    nll: float
    n_trials: int
    success: bool
    report: ReportModel = ReportModel()
    message: str = ""
    starts: list[float] = field(default_factory=list)


def fit_theory(
    theory: Theory,
    df: pd.DataFrame,
    *,
    report: ReportModel | None = None,
    n_starts: int = 3,
    n_screen: int = 64,
    seed: int = 0,
    init: dict[str, float] | None = None,
) -> FitResult:
    """Minimise the negative log-likelihood of `df.response_bin` under `theory`.

    Screens `n_screen` quasi-random parameter points, runs L-BFGS-B from the best
    `n_starts` of them (plus `init`, if given), then polishes the best with Powell.
    `report` defaults to `ReportModel.for_data(df)` (rounding on, scale inferred).
    """
    report = report if report is not None else ReportModel.for_data(df)
    bounds = {**parameter_bounds(theory), **report.bounds}
    names = list(bounds)
    lo = np.array([bounds[n][0] for n in names])
    hi = np.array([bounds[n][1] for n in names])
    stim = stimulus_from_frame(df)
    bins = df["response_bin"].to_numpy()

    def nll(x: np.ndarray) -> float:
        params = dict(zip(names, (float(v) for v in x)))
        try:
            val = -float(trial_log_lik(theory, params, stim, bins, report).sum())
        except Exception:  # a bad corner of parameter space: steer away from it
            return 1e12
        return val if math.isfinite(val) else 1e12

    # Starts: the caller's init (e.g. the full-data fit, when cross-validating),
    # then the best points of a quasi-random screen over the whole box. The
    # screen matters for likelihoods with kinks or plateaus (e.g. the
    # Bayesian sampler's sample count), where one local start stalls.
    starts: list[np.ndarray] = []
    if init is not None:
        starts.append(np.array([init.get(n, 0.5 * (bounds[n][0] + bounds[n][1])) for n in names], dtype=float))
    if n_screen > 0:
        pts = lo + qmc.Sobol(d=len(names), seed=seed).random(n_screen) * (hi - lo)
        pts = np.vstack([lo + 0.5 * (hi - lo), pts])
        vals = np.array([nll(x) for x in pts])
        starts += [pts[i] for i in np.argsort(vals)[: max(n_starts, 1)]]
    if not starts:
        starts.append(lo + 0.5 * (hi - lo))

    best = None
    start_vals = []
    box = list(zip(lo, hi))
    for x0 in starts:
        res = minimize(nll, x0, method="L-BFGS-B", bounds=box)
        start_vals.append(float(res.fun))
        if best is None or res.fun < best.fun:
            best = res
    assert best is not None
    # Derivative-free polish: gets past kinks where finite-difference gradients mislead L-BFGS-B.
    polish = minimize(nll, best.x, method="Powell", bounds=box)
    if polish.fun < best.fun:
        best = polish
    return FitResult(
        params=dict(zip(names, (float(v) for v in best.x))),
        nll=float(best.fun),
        n_trials=len(df),
        success=bool(best.success),
        report=report,
        message=str(best.message),
        starts=start_vals,
    )


def subject_folds(df: pd.DataFrame, k: int, seed: int = 0) -> np.ndarray:
    """Fold index per row, assigning whole subjects to folds."""
    subjects = df["subject_id"].unique()
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(subjects))
    fold_of = {s: int(i % k) for s, i in zip(subjects, perm)}
    return df["subject_id"].map(fold_of).to_numpy()


def cross_validate(
    theory: Theory,
    df: pd.DataFrame,
    *,
    report: ReportModel | None = None,
    k: int = 5,
    init: dict[str, float] | None = None,
    n_starts: int = 1,
    seed: int = 0,
) -> float:
    """Total held-out negative log-likelihood over K subject folds."""
    if k < 2 or df["subject_id"].nunique() < k:
        return float("nan")
    report = report if report is not None else ReportModel.for_data(df)
    folds = subject_folds(df, k, seed)
    total = 0.0
    for f in range(k):
        train, test = df[folds != f], df[folds == f]
        # Warm-started from the full-data fit, so no screen is needed.
        fit = fit_theory(theory, train, report=report, n_starts=n_starts, n_screen=0,
                         seed=seed + f, init=init)
        total += -float(trial_log_lik(theory, fit.params, stimulus_from_frame(test),
                                      test["response_bin"].to_numpy(), report).sum())
    return total


def predictive_check(theory: Theory, params: dict[str, float], df: pd.DataFrame,
                     report: ReportModel | None = None, n_bins: int = 10) -> pd.DataFrame:
    """Observed vs. predicted response mean and SD, by decile of the Bayesian posterior.

    This is the signature that separates decision-time bias from noisy sampling:
    bias models put constant noise around a distorted mean, while sampling models
    tie the noise to the posterior (largest near 50%, vanishing near 0 and 1).
    """
    report = report if report is not None else ReportModel.for_data(df)
    probs = predict_responses(theory, params, stimulus_from_frame(df), report)
    pred_mean = probs @ GRID
    pred_var = probs @ GRID**2 - pred_mean**2
    edges = np.linspace(0, 1, n_bins + 1)
    cell = np.clip(np.digitize(df["bayes"].to_numpy(), edges) - 1, 0, n_bins - 1)
    out = pd.DataFrame({"cell": cell, "obs": df["response"].to_numpy(),
                        "pred_mean": pred_mean, "pred_var": pred_var})
    rows = []
    for c, g in out.groupby("cell"):
        # Predicted SD of a single response pooled over the trials in this cell:
        # within-trial variance plus spread of the trial means.
        pv = g["pred_var"].mean() + g["pred_mean"].var(ddof=0)
        rows.append({"bayes_lo": edges[c], "bayes_hi": edges[c + 1], "n": len(g),
                     "obs_mean": g["obs"].mean(), "obs_sd": g["obs"].std(ddof=0),
                     "pred_mean": g["pred_mean"].mean(), "pred_sd": math.sqrt(max(pv, 0.0))})
    return pd.DataFrame(rows)


def load_theories(paths: list[Path]) -> dict[str, Theory]:
    return {Path(p).stem: Theory.from_yaml(p) for p in paths}


def n_params(theory: Theory, report: ReportModel | None = None) -> int:
    return len(parameter_bounds(theory)) + (len(report.bounds) if report is not None else 0)


def information_criteria(nll: float, k: int, n: int) -> dict[str, Any]:
    return {"aic": 2 * nll + 2 * k, "bic": 2 * nll + k * math.log(max(n, 1))}
