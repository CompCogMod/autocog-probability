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


# --- simulation calibration and the adversarial loop ------------------------

FITS = Path(__file__).resolve().parents[1] / "results" / "probability_updating" / "comparison" / "fits.csv"


def test_reporting_stage_rounds_simulated_responses():
    E = ProbabilityUpdatingExperiment
    old = (E.REPORT_ROUND10, E.REPORT_ROUND5)
    try:
        E.set_reporting(1.0, 0.0)
        exp = _design()
        assert {exp._report(b) for b in (3, 4, 6, 96)} == {0, 10, 100}
        assert {exp._report(5) for _ in range(200)} == {0, 10}  # ties split
        E.set_reporting(0.0, 1.0)
        assert [exp._report(b) for b in (2, 3, 97, 100)] == [0, 5, 95, 100]
        df = _design().simulate(Theory.from_yaml(THEORY_DIR / "llo_bias.yaml"), n_runs=2)
        assert (df["response_bin"] % 5 == 0).all()
        assert "rounded to the nearest multiple of 10%" in E.pretty_print_protocol()
        with pytest.raises(ValueError):
            E.set_reporting(1.5, 0.0)
    finally:
        E.set_reporting(*old)
    assert "rounded" not in E.pretty_print_header() or old != (0.0, 0.0)


@pytest.mark.skipif(not FITS.exists(), reason="comparison fits not built")
def test_calibrated_theories_narrow_ranges_and_keep_predictions(tmp_path):
    from src.probability_updating.calibrate import load_fit_params, reporting_rates, write_calibrated_theories
    params = load_fit_params(FITS)
    r10, r5 = reporting_rates(params)
    assert 0 < r10 < 0.5 and 0 < r5 < 0.5
    written = write_calibrated_theories(THEORY_PATHS, tmp_path, params)
    assert set(written) == {p.stem for p in THEORY_PATHS}
    for stem, path in written.items():
        orig, cal = _load(stem), Theory.from_yaml(path)
        assert orig.predict_source == cal.predict_source
        full, narrow = parameter_bounds(orig), parameter_bounds(cal)
        assert set(full) == set(narrow)
        for k, (lo, hi) in narrow.items():
            assert full[k][0] <= lo <= hi <= full[k][1], (stem, k)
        assert sum(n[1] - n[0] for n in narrow.values()) < sum(f[1] - f[0] for f in full.values())
        p = _mid_params(cal)
        assert np.allclose(orig.predict(p, _batch(), None), cal.predict(p, _batch(), None))


class _SchemaMock:
    """LLM stand-in that answers by requested schema: a fixed design or a fixed metric."""

    def __init__(self):
        import json
        self.exp = json.dumps({"trial_priors": [0.5, 0.5, 0.2, 0.8], "trial_p_signal_h": [0.6, 0.9, 0.7, 0.3],
                               "trial_p_signal_not_h": [0.4, 0.1, 0.3, 0.7], "trial_n_signals": [1, 1, 1, 1],
                               "rationale": "near-50% cells separate sampling from constant log-odds noise"})
        self.metric = json.dumps({"metric": "def metric(data):\n    d = data[data['prior'] == 0.5]\n"
                                            "    return float(d['response'].std())",
                                  "rationale": "response SD where the posterior is near 50%"})
        self.calls = 0

    def chat(self, messages, system, response_schema=None):
        from src.llm import ChatResult
        self.calls += 1
        text = self.exp if response_schema is ProbabilityUpdatingExperiment else self.metric
        parsed = response_schema.model_validate_json(text) if response_schema else None
        return ChatResult(text=text, parsed=parsed, usage={})


def test_design_only_loop_runs_end_to_end(tmp_path, monkeypatch):
    import main_probability_updating as mpu
    monkeypatch.setattr(mpu, "make_client", lambda cfg: _SchemaMock())
    old = (ProbabilityUpdatingExperiment.REPORT_ROUND10, ProbabilityUpdatingExperiment.REPORT_ROUND5)
    try:
        mpu.main(["--design_only", "--n_runs", "10", "--out_path", str(tmp_path), "--llm_model", "mock",
                  *([] if FITS.exists() else ["--no_calibration"])])
    finally:
        ProbabilityUpdatingExperiment.set_reporting(*old)
    run_dir = next(tmp_path.iterdir())
    designs = (run_dir / "designs.md").read_text()
    assert designs.count("## Round 0, proposed by") == 2
    assert "pi_1" in designs and "pi_2" in designs and "Welch p" in designs
    if FITS.exists():
        assert {p.stem for p in (run_dir / "seed_theories").glob("*.yaml")} == {"llo_bias", "llo_sampler"}
    from src.observation import Observations
    pool = Observations.load(run_dir / "observations", experiment_class=ProbabilityUpdatingExperiment)
    assert len(pool) == 1 and len(pool.latest_round) == 2


def test_full_loop_collects_ground_truth_before_arbitration(tmp_path, monkeypatch):
    import main_probability_updating as mpu

    class _Stop(Exception):
        pass

    class _Arbiter:
        def arbitrate(self, round_obj, *, pool, workspace):
            raise _Stop

    monkeypatch.setattr(mpu, "make_client", lambda cfg: _SchemaMock())
    monkeypatch.setattr(mpu.Arbiter, "from_config", classmethod(lambda cls, **kw: _Arbiter()))
    monkeypatch.setattr(mpu.Improver, "from_config", classmethod(lambda cls, **kw: None))
    monkeypatch.setattr(mpu.TheoryGenerator, "from_config", classmethod(lambda cls, **kw: None))
    old = (ProbabilityUpdatingExperiment.REPORT_ROUND10, ProbabilityUpdatingExperiment.REPORT_ROUND5)
    try:
        with pytest.raises(_Stop):
            mpu.main(["--n_runs", "10", "--out_path", str(tmp_path), "--llm_model", "mock",
                      *([] if FITS.exists() else ["--no_calibration"])])
    finally:
        ProbabilityUpdatingExperiment.set_reporting(*old)
    from src.observation import Observations
    run_dir = next(tmp_path.iterdir())
    assert run_dir.name.startswith("gt_weak_strong_evidence_llo_bias_vs_llo_sampler")
    pool = Observations.load(run_dir / "observations", experiment_class=ProbabilityUpdatingExperiment)
    for obs in pool.latest_round.observations:
        assert obs.real_value is not None
        assert {p.label for p in obs.predicted_values} == {"pi_1", "pi_2"}
    assert "ground truth" in (run_dir / "designs.md").read_text()
