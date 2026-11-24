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


def calibrated_ranges(theory_name: str, theory: Theory, params: pd.DataFrame,
                      quantiles: tuple[float, float] = (0.1, 0.9)) -> dict[str, str]:
    """`"[lo, hi]"` ranges spanning `quantiles` of the theory's estimates across studies.

    Parameters the theory declares but that were never fitted keep their
    declared range; ranges are clipped to the declared bounds.
    """
    bounds = parameter_bounds(theory)
    rows = params[params["theory"] == theory_name]
    out: dict[str, str] = {}
    for name, (blo, bhi) in bounds.items():
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
                              quantiles: tuple[float, float] = (0.1, 0.9)) -> dict[str, Path]:
    """Copy each theory YAML to `out_dir` with calibrated parameter ranges.

    The copy keeps every other key of the source YAML; the original ranges are
    kept under `fit_bounds` for reference. Returns {stem: calibrated path}.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for path in theory_paths:
        path = Path(path)
        raw = yaml.safe_load(path.read_text())
        theory = Theory.from_yaml(path)
        ranges = calibrated_ranges(path.stem, theory, params, quantiles)
        raw["fit_bounds"] = raw.get("parameters", {})
        raw["parameters"] = ranges
        raw["calibration"] = (f"parameter ranges = {quantiles[0]:g}-{quantiles[1]:g} quantiles of "
                              "per-study maximum-likelihood estimates on the human datasets")
        dest = out_dir / path.name
        dest.write_text(yaml.dump(raw, Dumper=_BlockDumper, sort_keys=False, width=100, allow_unicode=True))
        written[path.stem] = dest
    return written
