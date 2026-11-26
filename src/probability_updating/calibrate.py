"""Calibrate theories for simulation from their fits to the human datasets.

The theory YAMLs declare wide parameter ranges because they double as fitting
bounds (e.g. a lapse rate up to 0.6, a sample count up to 200). Drawn uniformly
per simulated subject, those ranges make every theory predict nearly anything,
so the adversarial loop would compare caricatures. This module narrows each
theory's ranges to the spread of its estimates across the human studies
(by default the 10th to 90th percentile over studies) and reads the shared
rounding rates off the same fits.

Input is `fits.csv` written by `main_probability_compare.py`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.probability_updating.fit import ROUNDING_BOUNDS, parameter_bounds
from src.theory import Theory

DEFAULT_FITS = (Path(__file__).resolve().parents[2]
                / "results" / "probability_updating" / "comparison" / "fits.csv")


class _BlockDumper(yaml.SafeDumper):
    """Writes multi-line strings (theory text, predict/policy source) as `|` blocks."""


_BlockDumper.add_representer(
    str, lambda d, v: d.represent_scalar("tag:yaml.org,2002:str", v, style="|" if "\n" in v else None))


def load_fit_params(fits_path: str | Path = DEFAULT_FITS) -> pd.DataFrame:
    """One row per (study, theory), one column per fitted parameter."""
    fits = pd.read_csv(fits_path)
    params = pd.DataFrame([json.loads(p) for p in fits["params"]], index=fits.index)
    return pd.concat([fits[["group", "theory"]], params], axis=1)


def stimulus_variation(trials: pd.DataFrame) -> pd.DataFrame:
    """Per study: how much the prior and the evidence vary, which decides what a fit can identify.

    Columns: `sd_prior_logodds`, `sd_log_lr`, `n_abs_log_lr` (distinct nonzero |log LR| values).
    `trials` is `data.load_trials()`; the index is the `dataset/study` group used in fits.csv.
    """
    df = trials.assign(group=trials["dataset"] + "/" + trials["study"])
    p = df["prior"].clip(1e-3, 1 - 1e-3)
    df["prior_logodds"] = np.log(p / (1 - p))
    df["abs_llr"] = df["log_lr"].abs().round(3)
    g = df.groupby("group")
    return pd.DataFrame({
        "sd_prior_logodds": g["prior_logodds"].std().fillna(0.0),
        "sd_log_lr": g["log_lr"].std().fillna(0.0),
        "n_abs_log_lr": g["abs_llr"].agg(lambda s: s[s > 0].nunique()),
    })


# Mean-rule parameters and the stimulus variation each needs to be estimable.
# Noise and lapse parameters are identified by the response distribution itself.
IDENTIFIED_BY = {
    "alpha": lambda v: v["sd_prior_logodds"] > 0.1,   # prior weight needs varying priors
    "beta": lambda v: v["sd_log_lr"] > 0.1,           # evidence weight needs varying evidence
    "gamma": lambda v: v["n_abs_log_lr"] >= 3,        # curvature needs >= 3 evidence strengths
}


def drop_unidentified(params: pd.DataFrame, theories: dict[str, Theory], variation: pd.DataFrame,
                      bound_tol: float = 0.005) -> pd.DataFrame:
    """Blank out mean-rule estimates that the study could not pin down.

    An estimate of a parameter in `IDENTIFIED_BY` is set to NaN when the study
    lacks the stimulus variation it needs (e.g. alpha when every prior is 50%)
    or when it sits on the declared fitting bound (within `bound_tol` of the
    range), which signals an estimate the likelihood did not constrain.
    """
    out = params.copy()
    for theory_name, theory in theories.items():
        rows = out["theory"] == theory_name
        for name, (blo, bhi) in parameter_bounds(theory).items():
            if name not in IDENTIFIED_BY or name not in out:
                continue
            ok = IDENTIFIED_BY[name](variation)
            unidentified = rows & ~out["group"].map(ok).fillna(False).astype(bool)
            tol = bound_tol * (bhi - blo)
            at_bound = rows & ((out[name] <= blo + tol) | (out[name] >= bhi - tol))
            out.loc[unidentified | at_bound, name] = np.nan
    return out


def calibrated_ranges(theory_name: str, theory: Theory, params: pd.DataFrame,
                      quantiles: tuple[float, float] = (0.1, 0.9),
                      pooled: dict[str, np.ndarray] | None = None) -> dict[str, str]:
    """`"[lo, hi]"` ranges spanning `quantiles` of the theory's estimates across studies.

    Parameters the theory declares but that were never fitted keep their
    declared range; ranges are clipped to the declared bounds. `pooled` maps a
    parameter name to estimates to use instead of this theory's own (used to
    give theories that share a parameter the same range).
    """
    bounds = parameter_bounds(theory)
    rows = params[params["theory"] == theory_name]
    out: dict[str, str] = {}
    for name, (blo, bhi) in bounds.items():
        if pooled and name in pooled:
            vals = pooled[name]
        else:
            vals = rows[name].dropna().to_numpy() if name in rows else np.array([])
        if len(vals) == 0:
            out[name] = str(theory.parameters[name])
            continue
        lo, hi = np.quantile(vals, quantiles)
        lo, hi = max(float(lo), blo), min(float(hi), bhi)
        out[name] = f"[{lo:.4g}, {hi:.4g}]"
    return out


def reporting_rates(params: pd.DataFrame) -> tuple[float, float]:
    """Median (round10, round5) across studies; each study's value is its median over theories."""
    cols = [c for c in ROUNDING_BOUNDS if c in params]
    if len(cols) < 2:
        return 0.0, 0.0
    per_study = params.dropna(subset=cols).groupby("group")[cols].median()
    if per_study.empty:
        return 0.0, 0.0
    med = per_study.median()
    return float(med["round10"]), float(med["round5"])


def write_calibrated_theories(theory_paths: list[Path], out_dir: Path, params: pd.DataFrame,
                              quantiles: tuple[float, float] = (0.1, 0.9),
                              variation: pd.DataFrame | None = None,
                              share: bool = True) -> dict[str, Path]:
    """Copy each theory YAML to `out_dir` with calibrated parameter ranges.

    - `variation` (from `stimulus_variation`): if given, mean-rule estimates a
      study could not identify are dropped first (`drop_unidentified`).
    - `share`: parameters declared by every theory (e.g. alpha and beta of two
      theories with the same mean rule) get one range, from the pooled
      estimates of all of them, so the theories differ only where their
      mechanisms differ rather than through calibration noise.

    The copy keeps every other key of the source YAML; the original ranges are
    kept under `fit_bounds` for reference. Returns {stem: calibrated path}.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [Path(p) for p in theory_paths]
    theories = {p.stem: Theory.from_yaml(p) for p in paths}
    if variation is not None:
        params = drop_unidentified(params, theories, variation)
    pooled: dict[str, np.ndarray] = {}
    if share and len(theories) > 1:
        common = set.intersection(*(set(t.parameters) for t in theories.values()))
        rows = params[params["theory"].isin(list(theories))]
        pooled = {n: rows[n].dropna().to_numpy() for n in sorted(common) if n in rows}
    note = (f"parameter ranges = {quantiles[0]:g}-{quantiles[1]:g} quantiles of per-study "
            "maximum-likelihood estimates on the human datasets")
    if variation is not None:
        note += ("; mean-rule estimates dropped where the study could not identify them "
                 "(no variation in prior/evidence, or estimate on the fitting bound)")
    if pooled:
        note += f"; shared with the other seed theories: {', '.join(pooled)} (pooled estimates)"
    written: dict[str, Path] = {}
    for path in paths:
        raw = yaml.safe_load(path.read_text())
        ranges = calibrated_ranges(path.stem, theories[path.stem], params, quantiles, pooled)
        raw["fit_bounds"] = raw.get("parameters", {})
        raw["parameters"] = ranges
        raw["calibration"] = note
        dest = out_dir / path.name
        dest.write_text(yaml.dump(raw, Dumper=_BlockDumper, sort_keys=False, width=100, allow_unicode=True))
        written[path.stem] = dest
    return written
