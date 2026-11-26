"""Adversarial experiment design for probability updating: decision-time bias vs. noisy sampling.

Two theories (default: `llo_bias`, constant log-odds noise around a re-weighted
posterior, vs. `llo_sampler`, the same mean rule with binomial sampling noise)
each ask the LLM for an experiment and a metric that would separate them with
REAL_N_SUBJECTS people. Before the loop starts, every theory's parameter ranges
are narrowed to the spread of its fits to the human datasets
(`src/probability_updating/calibrate.py`), and all simulated subjects share the
round-to-5/10 reporting stage estimated from the same fits. The calibrated
YAMLs are written to `<run_dir>/seed_theories/` for inspection.

Two modes:

  --design_only     Propose experiments only (no stand-in subjects, arbiter or
                    theory revision). Writes `<run_dir>/designs.md`: each
                    proposed design, its metric, both theories' predictions and
                    the expected Welch test at REAL_N_SUBJECTS. Use this to pick
                    an experiment to run on people.
  (default)         Full loop, as in the other autocog domains: a calibrated
                    `--ground_truth` theory stands in for people, the arbiter
                    picks the worse theory and the improver / theory generator
                    replaces it, for `--n_rounds` rounds. A recovery check: the
                    loop should move toward the ground truth.

Run layout under <run_dir>/: observations/ (the evidence pool), rounds/round_NNN/
(prompt logs per agent), seed_theories/, leaderboard.md, designs.md.

Examples (from the repo root):
  uv run python main_probability_updating.py --design_only --llm_provider anthropic --llm_model claude-opus-5-5
  uv run python main_probability_updating.py --n_rounds 3 --ground_truth weak_strong_evidence
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scipy import stats  # noqa: E402

from src.arbiter import Arbiter  # noqa: E402
from src.autocog import AutoCog  # noqa: E402
from src.config import LLMConfig  # noqa: E402
from src.improver import Improver, make_theory  # noqa: E402
from src.llm import make_client, model_path_tag  # noqa: E402
from src.logger import info  # noqa: E402
from src.metric import fmt_estimate  # noqa: E402
from src.observation import Observation, Observations  # noqa: E402
from src.probability_updating.calibrate import (  # noqa: E402
    DEFAULT_FITS,
    load_fit_params,
    reporting_rates,
    stimulus_variation,
    write_calibrated_theories,
)
from src.probability_updating.data import load_trials  # noqa: E402
from src.probability_updating.experiment import ProbabilityUpdatingExperiment  # noqa: E402
from src.run_config import REAL_N_SUBJECTS  # noqa: E402
from src.theory import Theory  # noqa: E402
from src.theory_generator import TheoryGenerator  # noqa: E402

THEORY_DIR = REPO / "theories" / "probability_updating"
EXPERIMENT_CLASS = ProbabilityUpdatingExperiment
LEADERBOARD_N = 3


# --- labels (same scheme as the other domains: pi_1, pi_1_1, pi_3, ...) -----

def _parse_label(label: str) -> tuple[int, int | None]:
    parts = label.split("_")
    if parts[0] != "pi" or len(parts) not in (2, 3):
        raise ValueError(f"unrecognised pi label: {label!r}")
    return int(parts[1]), (int(parts[2]) if len(parts) == 3 else None)


def _all_labels(pool: Observations, initial: tuple[str, str]) -> set[str]:
    seen = set(initial)
    for r in pool.rounds:
        seen.update(o.proposer_label for o in r.observations if o.proposer_label)
        if r.next_theory_label:
            seen.add(r.next_theory_label)
    return seen


def next_model_label(current: str, all_labels: set[str]) -> str:
    base, _ = _parse_label(current)
    versions = [_parse_label(L)[1] or 0 for L in all_labels if _parse_label(L)[0] == base]
    return f"pi_{base}_{max(versions) + 1}"


def next_fresh_label(all_labels: set[str]) -> str:
    return f"pi_{max(_parse_label(L)[0] for L in all_labels) + 1}"


def slot_agent(pool: Observations, slot: int, seed_yaml: Path, llm_client) -> AutoCog:
    """The theory currently in `slot`: the latest one admitted there, else the seed."""
    label, theory = f"pi_{slot}", None
    for r in pool.rounds:
        if r.next_theory_idx == slot and r.next_theory is not None:
            label, theory = r.next_theory_label or label, r.next_theory
    if theory is None:
        theory = Theory.from_yaml(seed_yaml)
    return AutoCog(label=label, theory=theory, experiment_class=EXPERIMENT_CLASS,
                   llm_client=llm_client)


# --- reporting ---------------------------------------------------------------

def welch_p(obs: Observation, a: str, b: str, n: int = REAL_N_SUBJECTS) -> float | None:
    pa, pb = obs.prediction_by_label(a), obs.prediction_by_label(b)
    if pa is None or pb is None or None in (pa.value, pb.value, pa.variance, pb.variance):
        return None
    _, p = stats.ttest_ind_from_stats(pa.value, math.sqrt(max(pa.variance, 0)), n,
                                      pb.value, math.sqrt(max(pb.variance, 0)), n, equal_var=False)
    return 1.0 if math.isnan(float(p)) else float(p)


def write_designs(pool: Observations, path: Path) -> None:
    """Every proposed design with its metric and the competing predictions."""
    lines = ["# Proposed experiments", "",
             f"Predictions are from {50} simulated subjects per theory; p is a Welch test "
             f"at N = {REAL_N_SUBJECTS} real subjects per theory's prediction.", ""]
    for i, r in enumerate(pool.rounds):
        for obs in r.observations:
            labels = [p.label for p in obs.predicted_values]
            others = [L for L in labels if L != obs.proposer_label]
            p = welch_p(obs, obs.proposer_label, others[0]) if others else None
            lines += [f"## Round {i}, proposed by `{obs.proposer_label}`", "",
                      obs.experiment.pretty_print(), "",
                      "**Metric:**", "", f"```python\n{obs.metric.metric_source.rstrip()}\n```", "",
                      f"**Metric rationale:** {obs.metric.rationale or '(none)'}", "",
                      "| theory | predicted (var) |", "|---|---|"]
            lines += [f"| {pr.label} | {fmt_estimate(pr.as_estimate())} |" for pr in obs.predicted_values]
            if obs.real_value is not None:
                lines.append(f"| ground truth | {fmt_estimate(obs.real_as_estimate())} |")
            lines += ["", f"Welch p ({obs.proposer_label} vs {others[0] if others else '-'}): "
                      + ("n/a" if p is None else f"{p:.3g}"), ""]
    path.write_text("\n".join(lines) + "\n")


def log_leaderboard(pool: Observations, round_idx: int, stage: str, path: Path) -> None:
    scores = pool.theory_scores()
    if not scores:
        return
    block = "\n".join(f"  {L:<10} {s:.3f}" for L, s in sorted(scores.items(), key=lambda kv: -kv[1]))
    info(f"[round {round_idx} | {stage}]\n{block}")
    with path.open("a") as f:
        f.write(f"## round {round_idx} — {stage}\n\n```\n{block}\n```\n\n")


# --- one round ----------------------------------------------------------------

def propose(pool: Observations, rd, pi_1: AutoCog, pi_2: AutoCog, n_runs: int, design_only: bool):
    """Fill the current round with one observation per pi, resuming a half-done round.

    A round is done once it has both observations (design-only) or once a
    replacement theory has been admitted (full loop); then a new one starts.
    """
    latest = pool.latest_round if len(pool) else None
    done = latest is None or (len(latest) == 2 if design_only else latest.next_theory is not None)
    round_obj = pool.start_round() if done else latest
    round_idx = len(pool) - 1
    for pi, adv in ((pi_1, pi_2), (pi_2, pi_1))[len(round_obj):]:
        round_obj.add(pi.propose_round(adversary=adv, pool=pool, workspace=rd(round_idx) / pi.label,
                                       n_runs=n_runs))
        pool.save()
    return round_idx, round_obj


def run_round(*, pool, run_dir, seeds, ground_truth, llm_client, arbiter, improver, generator,
              n_runs) -> int:
    def rd(i: int) -> Path:
        return run_dir / "rounds" / f"round_{i:03d}"

    pi_1 = slot_agent(pool, 1, seeds[0], llm_client)
    pi_2 = slot_agent(pool, 2, seeds[1], llm_client)
    round_idx, round_obj = propose(pool, rd, pi_1, pi_2, n_runs, design_only=ground_truth is None)
    info(f"[round {round_idx}] slots: 1={pi_1.label}, 2={pi_2.label}")
    if ground_truth is None:  # design-only: no subjects, no arbitration
        return round_idx

    for obs in round_obj.observations:
        if obs.real_value is None:
            obs.set_data(obs.experiment.simulate(ground_truth, n_runs=REAL_N_SUBJECTS))
    pool.backfill_predictions(n_runs=n_runs)
    log_leaderboard(pool, round_idx, "post-data", run_dir / "leaderboard.md")
    pool.save()

    verdict = arbiter.arbitrate(round_obj, pool=pool, workspace=rd(round_idx) / "arbiter")
    target = verdict.target_theory_idx
    target_label = round_obj[target - 1].proposer_label or f"pi_{target}"
    labels = (round_obj[0].proposer_label or pi_1.label, round_obj[1].proposer_label or pi_2.label)
    other_label = labels[2 - target]
    seen = _all_labels(pool, (pi_1.label, pi_2.label))
    registry = pool.theory_registry()
    scores = pool.theory_scores()

    def leaderboard(exclude: set[str]):
        ranked = sorted((L for L in scores if L in registry and L not in exclude), key=lambda L: -scores[L])
        return [(L, registry[L], scores[L]) for L in ranked[:LEADERBOARD_N]]

    if verdict.verdict == "new_model":
        killed = round_obj[target - 1].proposer_theory
        next_label = next_model_label(target_label, seen)
        model = improver.propose_model(
            theory=killed, arbiter_guide=verdict.recommendation, arbiter_theory_labels=labels,
            arbiter_target_idx=target, other_theory=round_obj[2 - target].proposer_theory,
            observations=pool.all_observations, workspace=rd(round_idx) / f"improver_{next_label}",
            leaderboard=leaderboard({target_label, other_label}))
        next_theory = make_theory(killed, model)
    elif verdict.verdict == "new_theory":
        next_label = next_fresh_label(seen)
        next_theory = generator.propose_theory(
            arbiter_guide=verdict.recommendation, arbiter_theory_labels=labels,
            arbiter_target_idx=target, observations=pool.all_observations,
            workspace=rd(round_idx) / f"theory_generator_{next_label}", leaderboard=leaderboard(set()))
    else:
        raise ValueError(f"unknown verdict: {verdict.verdict!r}")

    round_obj.set_next_theory(next_theory, idx=target, label=next_label)
    (rd(round_idx) / "next_theory.json").write_text(json.dumps(
        {"verdict": verdict.verdict, "replaced": target_label, "slot": target, "label": next_label,
         "theory": next_theory.model_dump()}, indent=2))
    pool.backfill_predictions(n_runs=n_runs)
    log_leaderboard(pool, round_idx, f"post-admit ({next_label})", run_dir / "leaderboard.md")
    pool.save()
    info(f"[round {round_idx}] verdict={verdict.verdict} replaced {target_label} -> {next_label}")
    return round_idx


# --- main ----------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", nargs=2, default=["llo_bias", "llo_sampler"], metavar=("PI_1", "PI_2"),
                    help="theory stems in theories/probability_updating/ for the two slots")
    ap.add_argument("--ground_truth", default="weak_strong_evidence",
                    help="theory stem standing in for people (ignored with --design_only)")
    ap.add_argument("--design_only", action="store_true",
                    help="only propose experiments; no stand-in subjects, arbiter or theory revision")
    ap.add_argument("--n_rounds", type=int, default=1)
    ap.add_argument("--n_runs", type=int, default=50, help="simulated subjects per theory per design")
    ap.add_argument("--fits", type=Path, default=DEFAULT_FITS,
                    help="fits.csv from main_probability_compare.py, used for calibration")
    ap.add_argument("--quantiles", type=float, nargs=2, default=[0.1, 0.9],
                    help="range of per-study estimates that becomes each parameter's range")
    ap.add_argument("--no_calibration", action="store_true",
                    help="use the theories' full fitting bounds and no rounding stage")
    ap.add_argument("--keep_unidentified", action="store_true",
                    help="calibrate from every study's estimates, including mean-rule parameters "
                         "the study could not identify (no prior/evidence variation, or on the bound)")
    ap.add_argument("--no_shared_params", action="store_true",
                    help="calibrate each seed theory's ranges separately, even for parameters "
                         "both declare (by default those get one range from the pooled estimates)")
    ap.add_argument("--llm_provider", default="gemini")
    ap.add_argument("--llm_model", default="gemini-3.1-pro-preview")
    ap.add_argument("--reasoning_effort", default=None)
    ap.add_argument("--run_id", default="0")
    ap.add_argument("--out_path", type=Path, default=REPO / "results" / "probability_updating" / "autocog")
    return ap


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    mode = "design" if args.design_only else f"gt_{args.ground_truth}"
    run_dir = args.out_path / (f"{mode}_{args.seeds[0]}_vs_{args.seeds[1]}_"
                               f"{model_path_tag(args.llm_model)}_run{args.run_id}")
    run_dir.mkdir(parents=True, exist_ok=True)

    stems = list(dict.fromkeys(args.seeds + ([] if args.design_only else [args.ground_truth])))
    sources = [THEORY_DIR / f"{s}.yaml" for s in stems]
    for p in sources:
        if not p.exists():
            raise SystemExit(f"no theory {p.stem!r} in {THEORY_DIR}")
    if args.no_calibration:
        paths = {p.stem: p for p in sources}
    else:
        fit_params = load_fit_params(args.fits)
        variation = None if args.keep_unidentified else stimulus_variation(load_trials())
        out_dir = run_dir / "seed_theories"
        seed_sources = [THEORY_DIR / f"{s}.yaml" for s in dict.fromkeys(args.seeds)]
        paths = write_calibrated_theories(seed_sources, out_dir, fit_params, tuple(args.quantiles),
                                          variation=variation, share=not args.no_shared_params)
        others = [p for p in sources if p.stem not in paths]
        if others:  # the stand-in ground truth keeps its own ranges
            paths |= write_calibrated_theories(others, out_dir, fit_params, tuple(args.quantiles),
                                               variation=variation, share=False)
        r10, r5 = reporting_rates(fit_params)
        EXPERIMENT_CLASS.set_reporting(r10, r5)
        info(f"[main] calibrated {', '.join(paths)} from {args.fits}; rounding: "
             f"round10={r10:.3f} round5={r5:.3f}")

    llm_cfg = LLMConfig(provider=args.llm_provider, model=args.llm_model,
                        reasoning_effort=args.reasoning_effort)
    llm_client = make_client(llm_cfg)
    seeds = (paths[args.seeds[0]], paths[args.seeds[1]])
    ground_truth = None if args.design_only else Theory.from_yaml(paths[args.ground_truth])
    arbiter = improver = generator = None
    if not args.design_only:
        arbiter = Arbiter.from_config(experiment_class=EXPERIMENT_CLASS, llm=llm_cfg)
        improver = Improver.from_config(experiment_class=EXPERIMENT_CLASS, llm=llm_cfg)
        generator = TheoryGenerator.from_config(experiment_class=EXPERIMENT_CLASS, llm=llm_cfg)

    pool = Observations.load(run_dir / "observations", experiment_class=EXPERIMENT_CLASS)
    info(f"[main] {mode}: {args.n_rounds} round(s), {len(pool)} existing, run_dir={run_dir}")
    try:
        for i in range(args.n_rounds):
            run_round(pool=pool, run_dir=run_dir, seeds=seeds, ground_truth=ground_truth,
                      llm_client=llm_client, arbiter=arbiter, improver=improver,
                      generator=generator, n_runs=args.n_runs)
            write_designs(pool, run_dir / "designs.md")
    finally:
        pool.save()
        write_designs(pool, run_dir / "designs.md")
    info(f"[main] done: {len(pool)} rounds; designs in {run_dir / 'designs.md'}")


if __name__ == "__main__":
    main()
