"""Load the combined human dataset as theory-ready stimuli and response bins.

Every theory in this domain predicts a distribution over a 101-point response
grid (0%, 1%, ..., 100%). A *stimulus* is a dict of equal-length numpy arrays
(or scalars, for one trial) with the keys in `STIMULUS_KEYS`. Theories are
written to broadcast over those arrays, so the fitter can score a whole study
in one `predict` call while `Experiment.simulate` calls it one trial at a time.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
COMBINED = REPO / "data" / "processed" / "all_datasets.csv.gz"

N_BINS = 101
GRID = np.linspace(0.0, 1.0, N_BINS)

# Evidence with |log LR| above this is treated as (near-)conclusive. Enke &
# Graeber include signals that are always right (log LR = +-inf); clipping keeps
# every theory's likelihood finite without changing any reasonable prediction.
MAX_ABS_LOG_LR = 12.0

STIMULUS_KEYS: dict[str, str] = {
    "prior": "P(H) before this step's evidence: the design prior if there is one, "
             "else the subject's own reported prior",
    "prior_is_objective": "1 if `prior` comes from the design, 0 if it is the subject's report",
    "log_lr": "ln[P(evidence | H) / P(evidence | not-H)] of this step's evidence",
    "n_signals": "number of signals (draws, informants, clues) in this step's evidence",
    "signal_dir": "+1 evidence favours H, -1 favours not-H, 0 uninformative",
    "bayes": "Bayesian posterior P(H) given `prior` and `log_lr`",
}


def response_bin(p: np.ndarray) -> np.ndarray:
    """Index of the nearest grid point (nearest whole percent) for responses in [0, 1]."""
    return np.clip(np.rint(np.asarray(p, dtype=float) * (N_BINS - 1)), 0, N_BINS - 1).astype(int)


def _expit(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p) - np.log1p(-p)


def load_trials(path: Path = COMBINED, *, datasets: list[str] | None = None) -> pd.DataFrame:
    """Rows of the combined file that a theory can be scored on.

    Keeps rows with a reported posterior, a usable prior (design prior, else the
    subject's reported prior) and a finite-or-clipped log likelihood ratio. Adds
    the stimulus columns (`STIMULUS_KEYS`), `response` (posterior in [0, 1]),
    `response_bin`, and `group` = '<dataset>/<study>'.
    """
    df = pd.read_csv(path, low_memory=False)
    if datasets:
        df = df[df["dataset"].isin(datasets)]
    df = df[df["posterior_reported"].notna() & df["log_lr"].notna()].copy()

    has_obj = df["prior_objective"].notna()
    df["prior"] = np.where(has_obj, df["prior_objective"], df["prior_reported"])
    df["prior_is_objective"] = has_obj.astype(int)
    df = df[df["prior"].notna()].copy()

    df["log_lr"] = df["log_lr"].clip(-MAX_ABS_LOG_LR, MAX_ABS_LOG_LR)
    df["n_signals"] = df["n_signals"].fillna(1).astype(float)
    df["signal_dir"] = df["signal_dir"].fillna(np.sign(df["log_lr"])).astype(float)
    df["bayes"] = _expit(_logit(df["prior"].to_numpy()) + df["log_lr"].to_numpy())
    df["response"] = df["posterior_reported"].clip(0, 1)
    df["response_bin"] = response_bin(df["response"].to_numpy())
    df["group"] = df["dataset"] + "/" + df["study"].astype(str)
    return df.reset_index(drop=True)


def stimulus_from_frame(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Batch stimulus (dict of arrays) for `Theory.predict`."""
    return {k: df[k].to_numpy(dtype=float) for k in STIMULUS_KEYS}


def stimulus_from_row(row: dict | pd.Series) -> dict[str, float]:
    return {k: float(row[k]) for k in STIMULUS_KEYS}


def extra_field(df: pd.DataFrame, key: str) -> pd.Series:
    """Pull one key out of the JSON `extra` column (NaN where absent)."""
    return df["extra"].map(lambda s: json.loads(s).get(key) if isinstance(s, str) else None)
