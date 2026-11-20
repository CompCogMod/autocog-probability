"""Probability-updating domain: theory contract, experiment simulation, and fitting."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.probability_updating.data import COMBINED, GRID, N_BINS, load_trials, stimulus_from_frame
from src.probability_updating.experiment import ProbabilityUpdatingExperiment
from src.probability_updating.fit import fit_theory, parameter_bounds
from src.theory import Theory

THEORY_DIR = Path(__file__).resolve().parents[1] / "theories" / "probability_updating"
THEORY_PATHS = sorted(THEORY_DIR.glob("*.yaml"))


def _design() -> ProbabilityUpdatingExperiment:
    return ProbabilityUpdatingExperiment(
        trial_priors=[0.2, 0.5, 0.8, 0.5],
        trial_p_signal_h=[0.7, 0.9, 0.3, 0.6],
        trial_p_signal_not_h=[0.3, 0.1, 0.7, 0.4],
        trial_n_signals=[1, 2, 1, 1],
    )


def _batch(n: int = 40, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    prior = rng.uniform(0.05, 0.95, n)
    llr = rng.normal(0, 1.5, n)
    bayes = 1 / (1 + np.exp(-(np.log(prior / (1 - prior)) + llr)))
    return {"prior": prior, "prior_is_objective": np.ones(n), "log_lr": llr,
            "n_signals": np.ones(n), "signal_dir": np.sign(llr), "bayes": bayes}


def _mid_params(theory: Theory) -> dict[str, float]:
    return {k: lo + 0.3 * (hi - lo) for k, (lo, hi) in parameter_bounds(theory).items()}


def test_seed_theories_exist():
    assert {p.stem for p in THEORY_PATHS} >= {
        "bayes_noise", "llo_bias", "weak_strong_evidence", "linear_compression", "bayesian_sampler"}


@pytest.mark.parametrize("path", THEORY_PATHS, ids=lambda p: p.stem)
def test_theory_contract_scalar_and_batch(path):
    theory = Theory.from_yaml(path)
    params = _mid_params(theory)
    stim = _batch()
    batch = theory.predict(params, stim, None)
    assert batch.shape == (40, N_BINS)
    assert np.allclose(batch.sum(axis=1), 1.0)
    assert (batch >= 0).all()
    for i in (0, 7, 39):
        one = theory.predict(params, {k: float(v[i]) for k, v in stim.items()}, None)
        assert one.shape == (N_BINS,)
        assert np.allclose(one, batch[i])
    assert 0 <= theory.policy(batch[0]) < N_BINS


def test_llo_with_unit_weights_is_bayes_noise():
    llo = Theory.from_yaml(THEORY_DIR / "llo_bias.yaml")
    bn = Theory.from_yaml(THEORY_DIR / "bayes_noise.yaml")
    stim = _batch()
    a = llo.predict({"alpha": 1.0, "beta": 1.0, "sigma": 0.4, "epsilon": 0.05}, stim, None)
    b = bn.predict({"sigma": 0.4, "epsilon": 0.05}, stim, None)
    assert np.allclose(a, b)


def test_bayes_noise_peaks_at_bayes():
    bn = Theory.from_yaml(THEORY_DIR / "bayes_noise.yaml")
    stim = _batch()
    probs = bn.predict({"sigma": 0.05, "epsilon": 0.0}, stim, None)
    assert np.all(np.abs(probs.argmax(axis=1) - np.rint(stim["bayes"] * 100)) <= 1)


def test_sampler_mean_shrinks_toward_half():
    bs = Theory.from_yaml(THEORY_DIR / "bayesian_sampler.yaml")
    stim = _batch()
    N, b = 10.0, 1.0
    probs = bs.predict({"N": N, "b": b, "tau": 0.005, "epsilon": 0.0}, stim, None)
    expected = (N * stim["bayes"] + b) / (N + 2 * b)
    assert np.allclose(probs @ GRID, expected, atol=0.01)


def test_experiment_simulate_canonical_columns():
    exp = _design()
    theory = Theory.from_yaml(THEORY_DIR / "bayesian_sampler.yaml")
    df = exp.simulate(theory, n_runs=3)
    assert list(df.columns) == list(ProbabilityUpdatingExperiment.output_columns)
    assert df["subject_id"].nunique() == 3
    assert df["response_bin"].between(0, 100).all()
    assert np.allclose(df["response"], df["response_bin"] / 100)


def test_experiment_rejects_mismatched_lists():
    with pytest.raises(ValueError):
        ProbabilityUpdatingExperiment(trial_priors=[0.5], trial_p_signal_h=[0.7, 0.6],
                                      trial_p_signal_not_h=[0.3], trial_n_signals=[1])


def test_fit_recovers_llo_parameters():
    theory = Theory.from_yaml(THEORY_DIR / "llo_bias.yaml")
    true = {"alpha": 0.6, "beta": 0.5, "sigma": 0.5, "epsilon": 0.05}
    stim = _batch(n=3000, seed=1)
    probs = theory.predict(true, stim, None)
    rng = np.random.default_rng(2)
    bins = np.array([rng.choice(N_BINS, p=p) for p in probs])
    df = pd.DataFrame({**stim, "response_bin": bins, "subject_id": np.arange(len(bins)) % 30})
    fit = fit_theory(theory, df, n_starts=2)
    for k in ("alpha", "beta", "sigma"):
        assert abs(fit.params[k] - true[k]) < 0.08, (k, fit.params)


@pytest.mark.skipif(not COMBINED.exists(), reason="combined dataset not built")
def test_human_data_loads_and_builds_designs():
    trials = load_trials()
    assert trials["response_bin"].between(0, 100).all()
    assert np.isfinite(stimulus_from_frame(trials)["log_lr"]).all()
    exp = ProbabilityUpdatingExperiment.from_trials(trials[trials["group"] == "enke_graeber2023/A_beliefs"])
    assert len(exp.trial_priors) > 10


def _load(stem: str) -> Theory:
    return Theory.from_yaml(THEORY_DIR / f"{stem}.yaml")


@pytest.mark.parametrize("path", THEORY_PATHS, ids=lambda p: p.stem)
def test_theories_carry_factorial_tags(path):
    import yaml
    raw = yaml.safe_load(path.read_text())
    assert raw.get("mean_rule") and raw.get("noise") in {"logodds_gaussian", "probability_gaussian", "sampling"}


@pytest.mark.parametrize("general, gen_params, special, spec_params", [
    # log-odds weighting with unit weights is Bayes, for every noise type
    ("llo_prob_noise", {"alpha": 1.0, "beta": 1.0, "tau": 0.08, "epsilon": 0.05},
     "bayes_prob_noise", {"tau": 0.08, "epsilon": 0.05}),
    ("llo_sampler", {"alpha": 1.0, "beta": 1.0, "N": 7.4, "tau": 0.02, "epsilon": 0.05},
     "bayes_sampler", {"N": 7.4, "tau": 0.02, "epsilon": 0.05}),
    # no compression is Bayes
    ("compression_logodds_noise", {"lam": 1.0, "sigma": 0.6, "epsilon": 0.05},
     "bayes_noise", {"sigma": 0.6, "epsilon": 0.05}),
    ("linear_compression", {"lam": 1.0, "tau": 0.08, "epsilon": 0.05},
     "bayes_prob_noise", {"tau": 0.08, "epsilon": 0.05}),
    # the Bayesian sampler without its Beta(b, b) correction is pure sampling
    ("bayesian_sampler", {"N": 7.4, "b": 0.0, "tau": 0.02, "epsilon": 0.05},
     "bayes_sampler", {"N": 7.4, "tau": 0.02, "epsilon": 0.05}),
])
def test_hybrids_nest_their_special_cases(general, gen_params, special, spec_params):
    stim = _batch()
    assert np.allclose(_load(general).predict(gen_params, stim, None),
                       _load(special).predict(spec_params, stim, None), atol=1e-6)


def test_sampling_noise_is_largest_near_half():
    # The signature that separates sampling from constant log-odds noise at matched means.
    stim = {k: np.array(v) for k, v in {"prior": [0.5, 0.97], "prior_is_objective": [1, 1], "log_lr": [0.0, 0.0],
                                        "n_signals": [1, 1], "signal_dir": [0, 0], "bayes": [0.5, 0.97]}.items()}
    probs = _load("bayes_sampler").predict({"N": 10.0, "tau": 0.005, "epsilon": 0.0}, stim, None)
    sd = np.sqrt(probs @ GRID**2 - (probs @ GRID) ** 2)
    assert sd[0] > 2 * sd[1]


def test_snap_matrices_conserve_mass_and_split_ties():
    from src.probability_updating.fit import snap_matrix
    for step in (1, 5, 10):
        m = snap_matrix(step)
        assert np.allclose(m.sum(axis=1), 1.0)
        assert np.all(m[:, np.arange(N_BINS) % step != 0] == 0)
    assert np.isclose(snap_matrix(10)[5, 0], 0.5) and np.isclose(snap_matrix(10)[5, 10], 0.5)


def test_report_model_detects_coarse_scale_and_rounding():
    from src.probability_updating.fit import ReportModel
    coarse = ReportModel.for_data(pd.DataFrame({"response_bin": [0, 10, 30, 100]}))
    fine = ReportModel.for_data(pd.DataFrame({"response_bin": [0, 13, 50, 99]}))
    assert coarse.step == 10 and not coarse.rounding
    assert fine.step == 1 and fine.rounding
    probs = np.full((1, N_BINS), 1.0 / N_BINS)
    m = fine.matrix({"round10": 1.0, "round5": 0.0})
    assert np.allclose((probs @ m)[0, np.arange(N_BINS) % 10 != 0], 0.0)


def test_rounding_stage_removes_sampler_round_number_advantage():
    # Responses piled on multiples of 5 around a Bayesian mean: without a rounding
    # stage a sampler with N=20 (outputs on multiples of 5%) beats log-odds noise for
    # that reason alone; with the shared stage it should lose most of that edge.
    from src.probability_updating.fit import ReportModel
    stim = _batch(n=1500, seed=3)
    rng = np.random.default_rng(4)
    z = np.log(stim["bayes"] / (1 - stim["bayes"])) + rng.normal(0, 0.5, 1500)
    bins = (np.rint(100 / (1 + np.exp(-z)) / 5) * 5).astype(int)
    df = pd.DataFrame({**stim, "response_bin": bins, "subject_id": np.arange(1500) % 30})
    gaps = {}
    for rounding in (False, True):
        rep = ReportModel(step=1, rounding=rounding)
        f_s = fit_theory(_load("bayes_sampler"), df, report=rep, n_starts=2)
        f_n = fit_theory(_load("bayes_noise"), df, report=rep, n_starts=2)
        gaps[rounding] = (f_n.nll - f_s.nll) / len(df)  # > 0: sampler better
    assert gaps[False] > 0.2
    assert gaps[True] < gaps[False] / 3
