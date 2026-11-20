"""Probability-updating experiment: report P(H) after seeing a prior and evidence.

This is the autocog domain for the probability-distortion question. A design is
a list of unique trials, each giving a prior P(H) and evidence whose
probability under H and under not-H is stated (for example, the share of black
balls in urn A vs urn B raised to the number of draws). On each trial the
subject reports P(H) on a 0-100% scale, recorded as one of 101 response bins.

Theories for this domain (theories/probability_updating/*.yaml) predict a
distribution over those 101 bins and accept either one trial's stimulus or a
batch (see src/probability_updating/data.py), so the same theory files are
used both for simulation here and for likelihood fits to the human datasets.
"""

from __future__ import annotations

import math
from typing import Annotated, Any, ClassVar

import numpy as np
import pandas as pd
from pydantic import Field, PrivateAttr

from src.experiment import Experiment
from src.logger import info
from src.probability_updating.data import N_BINS, STIMULUS_KEYS

Prob = Annotated[float, Field(gt=0.0, lt=1.0)]


class ProbabilityUpdatingExperiment(Experiment):
    """Belief updating with explicit priors and evidence of known diagnosticity."""

    name: ClassVar[str] = "Probability Updating"
    description: ClassVar[str] = (
        "Subjects are told a prior probability that hypothesis H is true (e.g. that a "
        "ball was drawn from urn A rather than urn B), then shown evidence whose "
        "probability under H and under not-H is known (e.g. the colours of several "
        "draws, with the urns' compositions stated). They report the probability "
        "that H is true on a 0-100% scale."
    )
    MAX_TRIALS: ClassVar[int] = 60

    output_columns: ClassVar[dict[str, str]] = {
        "subject_id": "simulated or human subject identifier",
        "trial": "0-based trial index within the subject",
        "prior": "prior P(H) stated to the subject",
        "p_signal_h": "probability of the observed evidence if H is true",
        "p_signal_not_h": "probability of the observed evidence if H is false",
        "n_signals": "number of signals (draws) in the evidence",
        "log_lr": "ln(p_signal_h / p_signal_not_h)",
        "bayes": "Bayesian posterior P(H | evidence)",
        "response": "reported P(H) in [0, 1] (response_bin / 100)",
        "response_bin": "reported percentage as an integer 0-100",
    }
    parameter_variables: ClassVar[dict[str, str]] = {
        "n_trials": "number of trials per subject",
    }
    introduction_text: ClassVar[str] = (
        "On each trial you will be told how likely it is, before any evidence, that a "
        "statement H is true. You will then see some evidence, together with how likely "
        "that evidence is if H is true and if it is false. Report how likely you think "
        "it is now that H is true, from 0% to 100%."
    )

    trial_priors: list[Prob] = Field(
        ..., description="Prior P(H) for each unique trial, each strictly between 0 and 1."
    )
    trial_p_signal_h: list[Prob] = Field(
        ..., description="P(observed evidence | H) for each unique trial."
    )
    trial_p_signal_not_h: list[Prob] = Field(
        ..., description="P(observed evidence | not-H) for each unique trial."
    )
    trial_n_signals: list[Annotated[int, Field(ge=1)]] = Field(
        ..., description="Number of signals (draws) making up each trial's evidence."
    )

    _trials: list[dict[str, float]] = PrivateAttr(default_factory=list)
    _idx: int = PrivateAttr(default=0)
    _n_repeats: int = PrivateAttr(default=1)

    def model_post_init(self, _context: Any) -> None:
        n = len(self.trial_priors)
        lengths = {len(self.trial_p_signal_h), len(self.trial_p_signal_not_h), len(self.trial_n_signals)}
        if n == 0 or lengths != {n}:
            raise ValueError(
                "trial_priors, trial_p_signal_h, trial_p_signal_not_h and trial_n_signals "
                f"must be non-empty and the same length; got {n} and {sorted(lengths)}."
            )
        unique = [self._stimulus(*t) for t in zip(self.trial_priors, self.trial_p_signal_h,
                                                  self.trial_p_signal_not_h, self.trial_n_signals)]
        self._n_repeats = max(1, self.MAX_TRIALS // n)
        sequence = unique * self._n_repeats
        np.random.default_rng().shuffle(sequence)
        self._trials = sequence
        info(f"ProbabilityUpdatingExperiment: {n} unique trials x {self._n_repeats} reps = "
             f"{len(self._trials)} trials")
        self._reset_history()

    @staticmethod
    def _stimulus(prior: float, p_h: float, p_nh: float, n_signals: int) -> dict[str, float]:
        log_lr = math.log(p_h) - math.log(p_nh)
        bayes = 1.0 / (1.0 + math.exp(-(math.log(prior) - math.log1p(-prior) + log_lr)))
        return {"prior": prior, "prior_is_objective": 1.0, "log_lr": log_lr,
                "n_signals": float(n_signals), "signal_dir": float(np.sign(log_lr)),
                "bayes": bayes, "p_signal_h": p_h, "p_signal_not_h": p_nh}

    def _reset_history(self) -> None:
        self._history = {k: [] for k in self.output_columns if k != "subject_id"}

    # --- prompt-facing summaries --------------------------------------------

    @classmethod
    def pretty_print_header(cls) -> str:
        return (
            "A belief-updating experiment. Each trial states a prior P(H) and shows evidence "
            "whose probability under H (`trial_p_signal_h`) and under not-H "
            "(`trial_p_signal_not_h`) is known to the subject; `trial_n_signals` is how many "
            "draws make up that evidence. The subject reports P(H) from 0 to 100%. Choose the "
            "unique trials (all four lists, same length) so that the candidate theories make "
            "different predictions: vary the prior and the strength of the evidence "
            "(log likelihood ratio) independently, and include cells where theories disagree "
            "about both the mean report and its trial-to-trial variability. Each unique trial "
            f"is repeated K = max(1, {cls.MAX_TRIALS} // n_unique) times per subject in a "
            "fresh random order. There is no feedback."
        )

    @classmethod
    def pretty_print_protocol(cls) -> str:
        keys = "; ".join(f"`{k}`: {v}" for k, v in STIMULUS_KEYS.items())
        return (
            f"Each subject completes about {cls.MAX_TRIALS} trials in random order. On each "
            "trial `predict(parameters, stimulus, history)` receives `stimulus`, a dict with "
            f"keys {keys}. Values are floats for one trial; the same code is also called with "
            "equal-length numpy arrays (a batch of trials) when fitting to human data, so "
            "write it to broadcast. It must return probabilities over the 101 response bins "
            "0%, 1%, ..., 100%: shape (101,) for one trial, (n, 101) for a batch. `history` is "
            "the subject's earlier trials (dict of lists) or None when fitting a batch."
        )

    def pretty_print(self) -> str:
        rows = "\n".join(
            f"  trial {i + 1}: prior={p:.3f}  P(e|H)={h:.4g}  P(e|not-H)={nh:.4g}  "
            f"n_signals={ns}  log LR={math.log(h / nh):+.2f}"
            for i, (p, h, nh, ns) in enumerate(zip(self.trial_priors, self.trial_p_signal_h,
                                                  self.trial_p_signal_not_h, self.trial_n_signals))
        )
        return (f"**Unique trials (n={len(self.trial_priors)}):**\n{rows}\n\n"
                f"**Rationale:** {self.rationale or '(no rationale)'}\n\n"
                f"**Schedule:** {len(self.trial_priors)} unique x {self._n_repeats} reps = "
                f"{len(self._trials)} trials per subject.\n")

    def pretty_print_design(self) -> str:
        return self.pretty_print().split("\n\n**Rationale")[0]

    # --- iteration interface (used by Experiment.simulate) ------------------

    def __iter__(self) -> "ProbabilityUpdatingExperiment":
        self._idx = 0
        self._reset_history()
        return self

    def __next__(self) -> tuple[dict[str, float], dict[str, list[Any]]]:
        if self._idx >= len(self._trials):
            raise StopIteration
        stim = self._trials[self._idx]
        self._idx += 1
        return {k: stim[k] for k in STIMULUS_KEYS}, self._history

    def update(self, response: int) -> None:
        stim = self._trials[self._idx - 1]
        b = int(np.clip(int(response), 0, N_BINS - 1))
        h = self._history
        h["trial"].append(self._idx - 1)
        for k in ("prior", "p_signal_h", "p_signal_not_h", "n_signals", "log_lr", "bayes"):
            h[k].append(stim[k])
        h["response"].append(b / (N_BINS - 1))
        h["response_bin"].append(b)

    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame(self._history)

    def parameter_context(self) -> dict[str, Any]:
        return {"n_trials": len(self._trials)}

    # --- human data ----------------------------------------------------------

    @classmethod
    def from_trials(cls, trials: pd.DataFrame, **kwargs: Any) -> "ProbabilityUpdatingExperiment":
        """Design made of the unique (prior, P(e|H), P(e|not-H), n_signals) cells in human data.

        Needs rows with design priors and both likelihoods (abstract urn-style studies);
        use `data.load_trials()` and filter to one dataset/study first.
        """
        cells = (trials[trials["prior_is_objective"] == 1]
                 .dropna(subset=["p_signal_h", "p_signal_not_h"])
                 [["prior", "p_signal_h", "p_signal_not_h", "n_signals"]]
                 .drop_duplicates())
        cells = cells[(cells["prior"].between(0, 1, inclusive="neither"))
                      & (cells["p_signal_h"].between(0, 1, inclusive="neither"))
                      & (cells["p_signal_not_h"].between(0, 1, inclusive="neither"))]
        if cells.empty:
            raise ValueError("no trials with a design prior and both likelihoods strictly in (0, 1)")
        return cls(trial_priors=cells["prior"].tolist(),
                   trial_p_signal_h=cells["p_signal_h"].tolist(),
                   trial_p_signal_not_h=cells["p_signal_not_h"].tolist(),
                   trial_n_signals=cells["n_signals"].fillna(1).astype(int).clip(lower=1).tolist(),
                   **kwargs)
