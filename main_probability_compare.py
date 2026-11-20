"""Compare probability-updating theories on the combined human dataset.

Fits every theory in `--theory_dir` (autocog Theory YAMLs) to every
dataset/study in data/processed/all_datasets.csv.gz by maximum likelihood,
then cross-validates over held-out subjects and runs a predictive check of
response mean and variability against the Bayesian posterior.

Outputs (under --out_dir, default results/probability_updating/comparison/):
  fits.csv               one row per (study, theory): NLL, AIC, BIC, CV NLL, parameters
  predictive_check.csv   observed vs predicted mean/SD by decile of the Bayesian posterior
  summary.md             per-study ranking, winners, and key parameter estimates

Examples (run from the repo root):
  uv run python main_probability_compare.py
  uv run python main_probability_compare.py --groups enke_graeber2023/A_beliefs --cv_folds 0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from src.probability_updating.data import COMBINED, load_trials  # noqa: E402
from src.probability_updating.fit import (  # noqa: E402
    ReportModel,
    cross_validate,
    fit_theory,
    information_criteria,
    load_theories,
    n_params,
    predictive_check,
)

DEFAULT_THEORY_DIR = REPO / "theories" / "probability_updating"
DEFAULT_OUT = REPO / "results" / "probability_updating" / "comparison"


def _run_one(theory_path: str, group: str, df: pd.DataFrame, n_starts: int, cv_folds: int,
             seed: int, rounding: bool) -> tuple[dict, pd.DataFrame]:
    name, theory = next(iter(load_theories([Path(theory_path)]).items()))
    t0 = time.time()
    report = ReportModel.for_data(df, rounding=rounding)
    fit = fit_theory(theory, df, report=report, n_starts=n_starts, seed=seed)
    k = n_params(theory, report)
    cv_nll = (cross_validate(theory, df, report=report, k=cv_folds, init=fit.params, seed=seed)
              if cv_folds else float("nan"))
    check = predictive_check(theory, fit.params, df, report)
    check.insert(0, "theory", name)
    check.insert(0, "group", group)
    row = {
        "group": group,
        "dataset": df["dataset"].iat[0],
        "study": str(df["study"].iat[0]),
        "theory": name,
        "n_trials": fit.n_trials,
        "n_subjects": df["subject_id"].nunique(),
        "k": k,
        "response_step": report.step,
        "rounding": report.rounding,
        "nll": fit.nll,
        **information_criteria(fit.nll, k, fit.n_trials),
        "cv_nll": cv_nll,
        "cv_nll_per_trial": cv_nll / fit.n_trials if cv_folds else float("nan"),
        "converged": fit.success,
        "params": json.dumps({p: round(v, 4) for p, v in fit.params.items()}),
        "seconds": round(time.time() - t0, 1),
    }
    return row, check


def write_summary(fits: pd.DataFrame, out: Path, score: str) -> None:
    lines = [
        "# Probability-updating theory comparison",
        "",
        f"Score: `{score}` (lower is better). Fits are pooled per study; "
        + ("CV holds out whole subjects." if score.startswith("cv") else "no cross-validation was run.")
        + (" Every theory shares the same reporting stage: fitted rounding to multiples of 5 and 10, "
           "and each study's response scale." if fits["rounding"].any() else
           " No rounding stage (run with --no_rounding)."),
        "",
        "## Ranking per study (difference from the best theory, per trial)",
        "",
    ]
    theories = sorted(fits["theory"].unique())
    table = fits.pivot(index="group", columns="theory", values=score)
    n = fits.groupby("group")["n_trials"].first()
    per_trial = table.div(n, axis=0) if score != "cv_nll_per_trial" else table
    delta = per_trial.sub(per_trial.min(axis=1), axis=0)
    lines.append("| study | n | winner | " + " | ".join(theories) + " |")
    lines.append("|---|---|---|" + "---|" * len(theories))
    for g in delta.index:
        winner = delta.loc[g].idxmin()
        cells = " | ".join("**0**" if t == winner else f"{delta.loc[g, t]:.4f}" for t in theories)
        lines.append(f"| {g} | {n[g]} | {winner} | {cells} |")
    wins = delta.idxmin(axis=1).value_counts()
    lines += ["", "## Wins", ""] + [f"- {t}: {wins.get(t, 0)} of {len(delta)} studies" for t in theories]

    # Factorial view: theories tagged with `mean_rule` and `noise` in their YAML.
    for axis, other in (("noise", "mean_rule"), ("mean_rule", "noise")):
        if axis not in fits or fits[axis].isna().all():
            continue
        tagged = fits.dropna(subset=[axis]).copy()
        tagged["score"] = tagged[score] / (1 if score == "cv_nll_per_trial" else tagged["n_trials"])
        best = tagged.groupby(["group", axis])["score"].min().unstack(axis)
        d = best.sub(best.min(axis=1), axis=0)
        levels = list(d.columns)
        lines += ["", f"## Best theory per `{axis}` (minimum over `{other}`), difference per trial", "",
                  "| study | winner | " + " | ".join(levels) + " |", "|---|---|" + "---|" * len(levels)]
        for g in d.index:
            w = d.loc[g].idxmin()
            lines.append(f"| {g} | {w} | " + " | ".join(
                "**0**" if lv == w else f"{d.loc[g, lv]:.4f}" for lv in levels) + " |")
        counts = d.idxmin(axis=1).value_counts()
        lines += ["", "Wins: " + ", ".join(f"{lv} {counts.get(lv, 0)}" for lv in levels),
                  "Total difference per trial summed over studies: "
                  + ", ".join(f"{lv} {d[lv].sum():.3f}" for lv in levels)]

    lines += ["", "## Parameter estimates", ""]
    for t in theories:
        sub = fits[fits["theory"] == t]
        lines += [f"### {t}", "", "| study | " + " | ".join(json.loads(sub['params'].iat[0])) + " |",
                  "|---|" + "---|" * len(json.loads(sub["params"].iat[0]))]
        for _, r in sub.sort_values("group").iterrows():
            p = json.loads(r["params"])
            lines.append(f"| {r['group']} | " + " | ".join(f"{v:.3g}" for v in p.values()) + " |")
        lines.append("")
    (out / "summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=COMBINED)
    ap.add_argument("--theory_dir", type=Path, default=DEFAULT_THEORY_DIR)
    ap.add_argument("--theories", nargs="*", help="theory file stems to include (default: all in --theory_dir)")
    ap.add_argument("--datasets", nargs="*", help="dataset keys to include (default: all)")
    ap.add_argument("--groups", nargs="*", help="'<dataset>/<study>' groups to include (default: all)")
    ap.add_argument("--n_starts", type=int, default=3, help="optimizer restarts per fit")
    ap.add_argument("--cv_folds", type=int, default=5, help="subject folds for cross-validation (0 = skip)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_rounding", action="store_true",
                    help="drop the shared round-to-5/10 reporting stage (not recommended: theories "
                         "whose outputs sit on round numbers then win for that reason alone)")
    ap.add_argument("--out_dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    trials = load_trials(args.data, datasets=args.datasets)
    if args.groups:
        trials = trials[trials["group"].isin(args.groups)]
    paths = sorted(args.theory_dir.glob("*.yaml"))
    if args.theories:
        paths = [p for p in paths if p.stem in set(args.theories)]
    if trials.empty or not paths:
        raise SystemExit("nothing to fit: check --datasets/--groups/--theories")
    groups = {g: d for g, d in trials.groupby("group")}
    print(f"{len(trials)} trials, {len(groups)} studies, {len(paths)} theories, {args.workers} workers")

    rows, checks = [], []
    jobs = [(str(p), g, d) for g, d in groups.items() for p in paths]
    # biggest studies first so the slowest jobs don't start last
    jobs.sort(key=lambda j: -len(j[2]))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_run_one, p, g, d, args.n_starts, args.cv_folds, args.seed,
                               not args.no_rounding): (p, g)
                   for p, g, d in jobs}
        for i, fut in enumerate(as_completed(futures), 1):
            p, g = futures[fut]
            try:
                row, check = fut.result()
            except Exception as e:  # report and keep going: one broken theory shouldn't sink the run
                print(f"[{i}/{len(jobs)}] FAILED {g} {Path(p).stem}: {type(e).__name__}: {e}")
                continue
            rows.append(row)
            checks.append(check)
            print(f"[{i}/{len(jobs)}] {g:<36} {row['theory']:<22} nll/trial={row['nll'] / row['n_trials']:.4f} "
                  f"cv/trial={row['cv_nll_per_trial']:.4f} ({row['seconds']}s)")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fits = pd.DataFrame(rows).sort_values(["group", "theory"])
    # Optional factorial tags (`mean_rule`, `noise`) from each theory's YAML.
    tags = {p.stem: yaml.safe_load(p.read_text()) for p in paths}
    for key in ("mean_rule", "noise"):
        fits[key] = fits["theory"].map(lambda t: tags.get(t, {}).get(key))
    fits.to_csv(args.out_dir / "fits.csv", index=False)
    pd.concat(checks, ignore_index=True).to_csv(args.out_dir / "predictive_check.csv", index=False)
    score = "cv_nll_per_trial" if args.cv_folds else "aic"
    write_summary(fits, args.out_dir, score)
    print(f"\nwrote {args.out_dir}/fits.csv, predictive_check.csv, summary.md")
    if np.isfinite(fits[score]).all():
        print((args.out_dir / "summary.md").read_text().split("## Parameter estimates")[0])


if __name__ == "__main__":
    main()
