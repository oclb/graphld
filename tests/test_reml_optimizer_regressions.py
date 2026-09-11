"""Synthetic regressions for accepted optimizer state and one-step uncertainty."""

import numpy as np
import polars as pl
import pytest

from graphld.heritability import FLAGS, GraphREML, MethodOptions, ModelOptions
from graphld.multiprocessing_template import SharedData


class AnalyticManager:
    """Two analytic blocks with the same incremental covariance contract as workers."""

    def __init__(self, shared, trial_values=(), linear=False):
        self.shared = shared
        self.trial_values = iter(trial_values)
        self.linear = linear
        self.flags = []
        self.trial_params = []
        self.covariance_diagonal = np.ones(2)

    def start_workers(self, flag):
        self.flags.append(flag)
        x = self.shared["params"][0]
        h2 = np.full(2, np.exp(x / 10))
        # Production reconstructs covariance from its fixed residual diagonal.
        self.covariance_diagonal = 1 + h2
        np.testing.assert_allclose(self.covariance_diagonal, 1 + h2)
        self.shared["variant_data"] = h2
        likelihood = x if self.linear else -0.5 * (x - 10)**2
        if flag == FLAGS["COMPUTE_LIKELIHOOD_ONLY"]:
            self.trial_params.append(x)
            likelihood = next(self.trial_values, likelihood)
        else:
            self.shared["gradient"] = np.full(2, 0.5 if self.linear else (10 - x) / 2)
            self.shared["hessian"] = np.full(2, -0.5)
        self.shared["likelihood"] = np.full(2, likelihood / 2)

    def await_workers(self):
        pass


def run_analytic(monkeypatch, *, trials=(), linear=False, start=0., **options):
    shared = SharedData(dict(likelihood=2, gradient=2, hessian=2, variant_data=2, params=1))
    manager = AnalyticManager(shared, trials, linear)
    blocks = [
        dict(sumstats=pl.DataFrame(dict(SNP=[f"rs{i}"], CHR=[1], POS=[i], base=[1.])),
             variant_offset=i)
        for i in range(2)
    ]
    method_options = dict(num_iterations=1, trust_region_size=1., num_jackknife_blocks=2)
    method_options.update(options)
    method = MethodOptions(**method_options)
    captured = {}
    native_jackknife = GraphREML._compute_pseudojackknife

    def capture_jackknife(g, h, p):
        captured.update(gradient=g.copy(), hessian=h.copy(), params=p.copy())
        return native_jackknife(g, h, p)

    monkeypatch.setattr(GraphREML, "_compute_pseudojackknife", staticmethod(capture_jackknife))
    result = GraphREML.supervise(
        manager, shared, blocks, num_iterations=method.num_iterations, num_params=1,
        verbose=False, method=method,
        model=ModelOptions(params=np.full((1, 1), start), sample_size=1000),
    )
    return result, manager, shared, captured


def test_pseudojackknife_reaches_quadratic_delete_optima():
    # Block objectives are -.5*(theta-2)^2 and -(theta-4)^2.
    theta = np.array([1.])
    deleted = GraphREML._compute_pseudojackknife(
        np.array([[1.], [6.]]), np.array([[[-1.]], [[-2.]]]), theta
    )
    np.testing.assert_allclose(deleted[:, 0], [4., 2.])


@pytest.mark.parametrize("optimizer", ["ai"])
def test_final_uncertainty_uses_accepted_parameter_derivatives(monkeypatch, optimizer):
    result, manager, shared, captured = run_analytic(monkeypatch, optimizer=optimizer)
    theta = result["parameters"][0]
    assert 1.99 < theta < 2.0
    np.testing.assert_allclose(captured["gradient"].sum(axis=0), [10-theta])
    np.testing.assert_allclose(captured["params"], result["parameters"])
    np.testing.assert_allclose(result["jackknife_params"], [[10.], [10.]])
    np.testing.assert_allclose(result["variant_h2"], np.exp(theta/10))
    assert result["log"]["final_likelihood"] == pytest.approx(-.5*(theta-10)**2)
    assert manager.flags[-1] == FLAGS["COMPUTE_PRECISE"]
    assert result["log"]["termination_reason"] == "max_iterations"
    assert result["log"]["uncertainty_status"] == "provisional_unresolved_endpoint"
    np.testing.assert_allclose(result['log']['initial_parameter_scaling_diagonal'], [1.])


@pytest.mark.parametrize("bad_likelihood", [-100., np.nan, np.inf, -np.inf])
def test_exhaustion_restores_covariance_likelihood_and_h2(monkeypatch, bad_likelihood):
    result, manager, shared, captured = run_analytic(monkeypatch, trials=[bad_likelihood]*100, num_iterations=10)
    assert len([x for x in manager.trial_params if x != 0]) == 16
    np.testing.assert_allclose(result["parameters"], [0.])
    np.testing.assert_allclose(result["variant_h2"], [1., 1.])
    np.testing.assert_allclose(manager.covariance_diagonal, [2., 2.])
    np.testing.assert_allclose(captured["gradient"].sum(axis=0), [10.])
    assert result["log"]["final_likelihood"] == -50.
    assert not result["log"]["converged"]
    assert result["log"]["termination_reason"] == "stalled_exact_score"
    assert FLAGS["COMPUTE_PRECISE"] in manager.flags


@pytest.mark.parametrize("bad_likelihood", [-100., np.nan, np.inf, -np.inf])
def test_rejected_trial_retries_from_accepted_params(monkeypatch, bad_likelihood):
    result, manager, shared, captured = run_analytic(monkeypatch, trials=[bad_likelihood])
    first, second = manager.trial_params[:2]
    assert 1.99 < first < 2.0
    assert second == pytest.approx(first/2)
    assert result["parameters"][0] == second
    np.testing.assert_allclose(manager.covariance_diagonal, 1+np.exp(second/10))
    assert manager.flags[-1] == FLAGS["COMPUTE_PRECISE"]


@pytest.mark.parametrize("window,tolerance", [(1, .1), (3, .4)])
def test_legacy_window_cannot_override_precise_stopping(monkeypatch, window, tolerance):
    result, _, _, _ = run_analytic(monkeypatch, linear=True, num_iterations=6,
                                   convergence_window=window, convergence_tol=tolerance)
    np.testing.assert_allclose(np.diff(result["likelihood_history"]), 1.)
    assert not result["log"]["converged"]
    assert result["log"]["num_iterations"] == 6


@pytest.mark.parametrize("invalid_step", [np.nan, np.inf, 1e308])
def test_nonfinite_proposal_is_not_sent_to_workers(monkeypatch, invalid_step):
    import graphld._reml_optimizer as callback
    native = callback.limit_linear_predictor_step
    calls = 0
    def invalid_first(designs, step, cap):
        nonlocal calls
        calls += 1
        return np.array([invalid_step]) if calls <= 2 else native(designs,step,cap)
    monkeypatch.setattr(callback, 'limit_linear_predictor_step', invalid_first)
    result, manager, _, _ = run_analytic(monkeypatch)
    assert np.isfinite(manager.trial_params).all()
    assert all(abs(x) <= 2 for x in manager.trial_params)
    assert np.isfinite(result['parameters']).all()


def test_exact_zero_score_requires_precise_audit(monkeypatch):
    result, manager, _, _ = run_analytic(monkeypatch, start=10., num_iterations=10)
    assert result["log"]["termination_reason"] == "stationary"
    assert result["log"]["converged"]
    assert FLAGS['COMPUTE_PRECISE'] in manager.flags
    assert result['log']['stationarity']['exact_score']
    assert len(result['log']['stationarity']['signed_objective_gains']) == 4
    np.testing.assert_allclose(manager.trial_params,10.)


def test_nonfinite_trial_h2_fails_before_further_worker_calls(monkeypatch):
    native = AnalyticManager.start_workers
    calls = []
    def corrupt(self, flag):
        calls.append(flag)
        native(self, flag)
        if flag == FLAGS['COMPUTE_LIKELIHOOD_ONLY']:
            self.shared['variant_data'] = np.full(2,np.nan)
    monkeypatch.setattr(AnalyticManager,'start_workers',corrupt)
    with pytest.raises(FloatingPointError,match='Nonfinite trial heritability'):
        run_analytic(monkeypatch)
    assert calls == [FLAGS['INITIALIZE'],FLAGS['COMPUTE_LIKELIHOOD_ONLY']]


def test_singular_endpoint_withholds_ordinary_uncertainty(monkeypatch):
    native = AnalyticManager.start_workers
    def singular(self, flag):
        native(self,flag)
        self.shared['hessian'] = np.zeros(2)
    monkeypatch.setattr(AnalyticManager,'start_workers',singular)
    result,_,_,captured = run_analytic(monkeypatch,start=10.,num_iterations=5)
    assert result['log']['termination_reason']=='stationary_singular_information'
    assert not result['log']['converged']
    assert result['log']['uncertainty_status']=='singular_information'
    assert np.isnan(result['parameters_se']).all()
    assert not captured


def test_information_solve_failure_withholds_uncertainty(monkeypatch):
    def failed_solve(*args):
        raise np.linalg.LinAlgError('Unrepresentable information update')
    monkeypatch.setattr(GraphREML, '_compute_pseudojackknife', staticmethod(failed_solve))
    result, _, _, _ = run_analytic(monkeypatch, start=10., num_iterations=5)
    assert result['log']['uncertainty_status'] == 'invalid_information_solve'
    assert np.isnan(result['parameters_se']).all()
    assert np.isnan(result['enrichment_se']).all()
