"""Precise inverse diagonals and native worker trace correction."""
import numpy as np
import polars as pl
import pytest
from scipy.sparse import csc_matrix
from scipy.special import expit
from graphld.precision import PrecisionOperator
from graphld._trace_anchor import InverseDiagonalAnchor
from graphld.heritability import GraphREML


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('scale',[1e-5,1.,1e5])
def test_precise_diagonal_permutation_selection_and_updates(dtype,scale):
    rng=np.random.default_rng(183);n=37
    edges=np.tril(rng.normal(size=(n,n))*(rng.uniform(size=(n,n))<.08),-1)
    matrix=edges+edges.T;matrix+=np.diag(abs(matrix).sum(0)+.5);matrix=(matrix*scale).astype(dtype)
    operator=PrecisionOperator(csc_matrix(matrix),pl.DataFrame({'index':np.arange(n)}))
    expected=np.diag(np.linalg.inv(matrix.astype(float)))
    np.testing.assert_allclose(operator.inverse_diagonal(method='exact'),expected,rtol=2e-12,atol=0)
    selected=operator[[31,2,17,0]]
    np.testing.assert_allclose(selected.inverse_diagonal(method='exact'),expected[[31,2,17,0]],rtol=2e-12,atol=0)
    selected.set_diagonal(selected.get_diagonal()+np.array([1.,3.,2.,4.])*scale)
    expected=np.diag(np.linalg.inv(operator.matrix.toarray().astype(float)))
    np.testing.assert_allclose(selected.inverse_diagonal(method='exact'),expected[[31,2,17,0]],rtol=2e-12,atol=0)
    np.testing.assert_allclose(operator.inverse_diagonal(method='exact'),expected,rtol=2e-12,atol=0)


def test_anchor_invalidates_raw_cache_and_incompatible_probes(monkeypatch):
    op=PrecisionOperator(csc_matrix(np.diag([2.,3.,4.])),pl.DataFrame({'index':[0,1,2]}));op.set_which_indices([2,0])
    calls=[]
    def inverse(self,method='xdiag',n_samples=100,seed=None):
        values=1./self.get_diagonal()
        calls.append(method)
        return values if method=='exact' else values+np.array([.2,-.1])
    monkeypatch.setattr(PrecisionOperator,'inverse_diagonal',inverse)
    anchor=InverseDiagonalAnchor()
    anchor.evaluate(op,seed=123)
    np.testing.assert_allclose(anchor.evaluate(op,precise=True,seed=123),1/op.get_diagonal())
    assert calls.count('xdiag')==1 and anchor.cache_hits==1
    op.set_diagonal([5.,2.5])
    np.testing.assert_allclose(anchor.evaluate(op,seed=123),1/op.get_diagonal())
    np.testing.assert_allclose(anchor.evaluate(op,seed=127),1/op.get_diagonal()+[.2,-.1])
    op.set_which_indices([1,0])
    np.testing.assert_allclose(anchor.evaluate(op,seed=123),1/op.get_diagonal()+[.2,-.1])
    prior=calls.count('xdiag');op.set_diagonal([3.3,2.8])
    anchor.evaluate(op,precise=True,seed=123)
    assert calls.count('xdiag')==prior+1
    with pytest.raises(ValueError,match='fixed probe seed'):anchor.evaluate(op,seed=None)


def test_native_block_contracts_anchor_with_current_link_jacobian():
    n=80;nv=100;rng=np.random.default_rng(871)
    residual=np.eye(n)*2+np.diag(np.full(n-1,.2),1)+np.diag(np.full(n-1,.2),-1)
    indices=np.arange(nv)%n
    A=np.column_stack([np.ones(nv),rng.binomial(1,.4,nv),rng.normal(size=nv)])
    y=rng.normal(size=n);op=PrecisionOperator(csc_matrix(residual),pl.DataFrame({'index':indices}))
    anchor=InverseDiagonalAnchor();diagonal=np.diag(residual).copy()
    def evaluate(t,precise=False):
        return GraphREML._compute_block_likelihood(op,y,A,t[:,None],30.,np.zeros(nv),12,False,seed=123,residual_diagonal=diagonal,trace_anchor=anchor,precise_score=precise)
    def expected(t,estimated):
        h=np.logaddexp(0,A@t)/30;M=residual+np.diag(np.bincount(indices,weights=h,minlength=n))
        b=np.linalg.solve(M,y);J=np.zeros((n,3));np.add.at(J,indices,A*expit(A@t)[:,None]/30)
        return .5*(b*b-estimated)@J
    theta=np.array([-.7,.2,-.1]);evaluate(theta)
    _,g,_,_=evaluate(theta,True)
    np.testing.assert_allclose(g.ravel(),expected(theta,np.diag(np.linalg.inv(op.matrix.toarray()))),rtol=1e-10,atol=1e-12)
    error=anchor.anchor[2].copy();theta=np.array([-.4,-.1,.3]);_,g,_,_=evaluate(theta)
    np.testing.assert_allclose(g.ravel(),expected(theta,anchor.raw_cache[2]-error),rtol=1e-10,atol=1e-12)
    assert anchor.refreshes==1 and anchor.uses==1 and anchor.cache_hits==1
