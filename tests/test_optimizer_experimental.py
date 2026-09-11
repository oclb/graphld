"""Behavioral regressions for the experimental optimizer."""
import numpy as np
import pytest
from graphld._optimizer import maximize, finite_difference_block_score, limit_linear_predictor_step


@pytest.mark.parametrize('strategy',['trust','ai_line','bfgs'])
def test_large_damping_cannot_certify_a_distant_point(strategy):
    def objective(x):return -.5*(x[0]-10)**2
    def derivatives(x):return objective(x),10-x,np.eye(1)
    r=maximize(np.zeros(1),objective,derivatives,exact_derivatives=derivatives,
               initial_damping=1e8,strategy=strategy,max_iterations=80)
    if r.status=='stationary':assert r.likelihood>-.002
    else:assert r.status!='stationary'


def test_curvature_mismatch_is_repaired_by_objective_search():
    def objective(x):return -.5*x[0]**2-.25*x[0]**4
    def derivatives(x):return objective(x),-x-x**3,np.eye(1)
    r=maximize(np.array([4.]),objective,derivatives,exact_derivatives=derivatives)
    assert r.status=='stationary' and r.likelihood>-.002
    assert all(b['likelihood']>=a['likelihood']-1e-10 for a,b in zip(r.history,r.history[1:]))


def test_biased_score_requires_audit_and_exact_refinement():
    def objective(x):return -.5*np.dot(x,x)
    def exact(x):return objective(x),-x,np.eye(len(x))
    def noisy(x):return objective(x),-x+np.array([.7,-.4]),np.eye(2)
    x=np.array([.7,-.4])
    uncertified=maximize(x,objective,noisy)
    assert uncertified.status=='stalled_stochastic_score'
    certified=maximize(x,objective,noisy,exact_derivatives=exact)
    assert certified.status=='stationary' and certified.likelihood>-.002
    assert certified.exact_derivative_evaluations>0


def test_rejected_nonfinite_objective_restores_accepted_state():
    state={}
    def objective(x):
        state['x']=x.copy()
        return -float(x@x) if abs(x[0])<1 else np.nan
    def derivative(x):return objective(x),np.array([2.]),np.eye(1)
    r=maximize(np.zeros(1),objective,derivative,max_iterations=2)
    assert r.status=='stalled_stochastic_score'
    assert np.array_equal(state['x'],r.parameters) and r.likelihood==0


def test_model_step_cap_cannot_manufacture_stationarity():
    def objective(x):return -.5*(x[0]-10)**2
    def derivatives(x):return objective(x),10-x,np.eye(1)
    def cap(x,d):return d*min(1.,1e-8/max(abs(d[0]),1e-30))
    r=maximize(np.zeros(1),objective,derivatives,exact_derivatives=derivatives,
               limit_step=cap,max_iterations=12)
    assert r.status!='stationary' and r.likelihood < -49


def test_predictor_cap_handles_steps_beyond_single_precision_range():
    a=np.array([[1.,2.],[1.,-1.]])
    for size in [1.,1e45,1e300]:
        step=limit_linear_predictor_step([a],np.array([size,-2*size]),2.)
        assert np.isfinite(step).all()
        assert np.max(np.abs(a@step))<=2.
        assert np.max(np.abs(a@step))>1.99


@pytest.mark.parametrize('strategy',['ai_line','bfgs','trust'])
def test_initial_scale_and_numerical_ridge_cannot_manufacture_stationarity(strategy):
    def objective(x):return -.5*float(x@x)
    def derivative(x):
        return objective(x),-x,np.diag([1e20 if x[1]>5 else 1.,1.])
    def exact(x):return objective(x),-x,np.eye(2)
    result=maximize(np.array([10.,10.]),objective,derivative,exact_derivatives=exact,strategy=strategy)
    assert result.status=='stationary' and result.likelihood>-.001


def test_flat_parameter_has_separate_stationarity_classification():
    def objective(x):return -.5*x[0]**2
    def derivative(x):return objective(x),np.array([-x[0],0.]),np.diag([1.,0.])
    result=maximize(np.zeros(2),objective,derivative,exact_derivatives=derivative)
    assert result.status=='stationary_singular_information'
    assert result.stationarity['information_rank_deficient']


def test_block_finite_difference_matches_exact_nonlinear_score_and_restores_center():
    a=np.array([[1.,-1.],[1.,.5],[1.,2.]])
    y=np.array([.3,.5,.8]);t=np.array([-.7,.2]);state={}
    def blocks(theta):
        state['parameters']=theta.copy()
        variance=.1+np.exp(a@theta)
        return -.5*(np.log(variance)+y*y/variance)+np.array([10000.,20000.,30000.])
    v=.1+np.exp(a@t);jac=a*np.exp(a@t)[:,None]
    exact=.5*(y*y/v**2-1/v)[:,None]*jac
    information=.5*jac.T@((y*y/v**3)[:,None]*jac)
    calculated,audit=finite_difference_block_score(t,information,blocks)
    np.testing.assert_allclose(calculated,exact,atol=2e-7)
    np.testing.assert_array_equal(state['parameters'],t)
    assert audit['information_scaled_step_difference']<audit['threshold']


def test_failed_finite_difference_audit_restores_center():
    center=np.array([0.]);state={}
    def blocks(t):
        state['parameters']=t.copy()
        return np.array([np.exp(1000*t[0])-1000*t[0]])
    with pytest.raises(RuntimeError,match='unresolved'):
        finite_difference_block_score(center,np.eye(1),blocks)
    np.testing.assert_array_equal(state['parameters'],center)


def test_finite_difference_preserves_values_from_a_reused_oracle_buffer():
    buffer=np.zeros(2)
    def blocks(t):
        buffer[:]=10000.-.5*(t[0]-np.array([1.,-2.]))**2
        return buffer
    g,_=finite_difference_block_score(np.zeros(1),np.array([[2.]]),blocks)
    np.testing.assert_allclose(g.ravel(),[1.,-2.],atol=1e-5)


def test_matching_rounded_differences_cannot_certify_a_weak_coordinate():
    center=np.zeros(1);state={}
    # The true score is 1e-12 and the exact quadratic gain is 0.5. Both
    # finite-difference step sizes round to identical objective values.
    def blocks(t):
        state['parameters']=t.copy()
        return np.array([10000.+1e-12*t[0]-.5e-24*t[0]**2])
    with pytest.raises(RuntimeError,match='unresolved'):
        finite_difference_block_score(center,np.array([[1e-24]]),blocks)
    np.testing.assert_array_equal(state['parameters'],center)


@pytest.mark.parametrize('strategy',['ai_line','bfgs','trust'])
def test_audit_control_variate_requires_precise_final_score(strategy):
    audited=[]
    def objective(x):return -.5*float(x@x)-.1*float(np.sum(x**4))
    def precise(x):
        audited.append(x.copy())
        return objective(x),-x-.4*x**3,np.eye(len(x))
    def noisy(x):
        return objective(x),-x-.4*x**3+.6+.03*np.sin(x),np.eye(len(x))
    result=maximize(np.array([2.,-1.]),objective,noisy,exact_derivatives=precise,
                    strategy=strategy,audit_correction=True)
    assert result.status=='stationary' and result.likelihood>-.001
    np.testing.assert_array_equal(audited[-1],result.parameters)
    assert result.stationarity['exact_score']
    assert result.exact_derivative_evaluations==len(audited)


def test_audit_can_update_the_score_oracle_without_double_correction():
    anchor={'bias':.8};audits=[]
    def objective(x):return -.5*float(x@x)-.1*float(np.sum(x**4))
    def score(x):return -x-.4*x**3
    def approximate(x):return objective(x),score(x)+anchor['bias'],np.eye(len(x))
    def precise(x):
        audits.append(x.copy());anchor['bias']=0.
        return objective(x),score(x),np.eye(len(x))
    result=maximize(np.array([2.,-1.]),objective,approximate,exact_derivatives=precise,
                    audit_correction=True,score_oracle_updates_on_audit=True)
    assert result.status=='stationary' and result.likelihood>-.001
    np.testing.assert_array_equal(audits[-1],result.parameters)


def test_repeated_screen_rejection_retries_original_proposals_exactly():
    screen_calls=[];observed=[]
    def objective(t):return -.5*float((t[0]-3)**2)
    def derivative(t):return objective(t),np.array([3-t[0]]),np.eye(1)
    def screen(x,step,f,g):screen_calls.append((x.copy(),step.copy()));return False
    def observer(x,step,f,g,actual,previously_screened):observed.append((actual-f,previously_screened))
    result=maximize(np.zeros(1),objective,derivative,exact_derivatives=derivative,
                    screen_proposal=screen,trial_observer=observer,max_iterations=10)
    assert result.status=='stationary'
    np.testing.assert_allclose(result.parameters,[3.])
    assert len(screen_calls)==2
    assert result.history[0]['exact_screen_fallback']
    assert observed[0][0]>0 and observed[0][1]
    # Final signed checks bypass the screen entirely.
    assert len(result.stationarity['signed_objective_gains'])==4
