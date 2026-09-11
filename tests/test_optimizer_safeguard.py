"""Sufficient-progress safeguards preserve useful BFGS proposals."""

import numpy as np
import pytest

from graphld._optimizer import maximize, limit_linear_predictor_step, select_capped_proposal


@pytest.mark.parametrize('fraction', [.01, .1, .5])
def test_adequate_uncapped_bfgs_is_retained(fraction):
    bfgs = np.array([.5])
    step, log = select_capped_proposal(
        np.ones(1), np.eye(1), [('bfgs', bfgs), ('ai', np.ones(1))],
        lambda x: x, minimum_relative_gain=fraction,
    )
    assert log['candidates']['ai']['predicted_gain'] > log['candidates']['bfgs']['predicted_gain']
    assert log['direction'] == 'bfgs'
    assert log['primary_relative_gain'] == pytest.approx(.75)
    np.testing.assert_array_equal(step, bfgs)


@pytest.mark.parametrize('fraction,expected', [(.01, 'bfgs'), (.1, 'ai'), (.5, 'ai')])
def test_fixed_fraction_sensitivity_is_explicit(fraction, expected):
    _, log = select_capped_proposal(
        np.ones(1), np.eye(1), [('bfgs', np.array([.01])), ('ai', np.ones(1))],
        lambda x: x, minimum_relative_gain=fraction,
    )
    assert log['direction'] == expected
    assert log['minimum_relative_gain'] == fraction
    assert log['primary_relative_gain'] == pytest.approx(.0199)


@pytest.mark.parametrize('strategy', ['bfgs', 'ai_line'])
def test_saturated_direction_cap_uses_productive_gradient(strategy):
    # Average information can vanish quadratically while its score vanishes
    # linearly, making the raw AI direction huge in a saturated coefficient.
    def objective(x):
        return -.5*(x[0]-1)**2 - np.exp(x[1])
    def derivatives(x):
        return objective(x), np.array([1-x[0], -np.exp(x[1])]), np.diag([1., np.exp(2*x[1])])
    x = np.array([0., -45.])
    result = maximize(x, objective, derivatives, exact_derivatives=derivatives,
                      strategy=strategy, max_iterations=1,
                      limit_step=lambda x, d: limit_linear_predictor_step([np.eye(2)], d, 2.))
    log = result.history[0]['proposal']
    assert log['direction'] == 'scaled_gradient'
    assert log['reason'] == 'primary_below_objective_resolution'
    primary = 'bfgs' if strategy == 'bfgs' else 'ai'
    assert log['candidates'][primary]['cap_multiplier'] < 1e-18
    assert result.likelihood - objective(x) > .49
    assert result.status == 'max_iterations'


def test_no_resolvable_candidate_audits_then_stalls():
    # The supplied exact score still has q=.5. A cap below parameter resolution
    # cannot establish convergence even though every evaluated value is equal.
    audits = []
    def derivatives(x):
        return -.5*(x[0]-1)**2, np.array([1-x[0]]), np.eye(1)
    def exact(x):
        audits.append(x.copy())
        return derivatives(x)
    result = maximize(np.zeros(1), lambda x: derivatives(x)[0], derivatives,
                      exact_derivatives=exact, strategy='bfgs', max_iterations=5,
                      audit_correction=True,
                      limit_step=lambda x, d: d*1e-30)
    assert audits
    assert result.status == 'stalled_exact_score'
    assert not result.stationarity
    assert result.metric_resets == [dict(iteration=0, reason='precise_refresh')]
    np.testing.assert_array_equal(result.parameters, [0.])
