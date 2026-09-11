"""Unpenalized likelihood restoration after extreme genetic-variance trials."""
import numpy as np
import polars as pl
from scipy.sparse import csc_matrix
from graphld.precision import PrecisionOperator
from graphld.heritability import GraphREML


def test_extreme_trial_restores_identical_unpenalized_covariance_and_likelihood():
    operator=PrecisionOperator(csc_matrix([[2.,.2],[.2,1.5]]),
                               pl.DataFrame({'index':[0,1,1]}))
    residual=operator.get_diagonal()
    annotations=np.array([[1.,0.],[1.,1.],[1.,.3]])
    old_h2=np.zeros(3)
    def evaluate(theta):
        nonlocal old_h2
        result=GraphREML._compute_block_likelihood(
            operator,np.array([.3,-.2]),annotations,np.array(theta)[:,None],
            1.,old_h2,10,True,residual_diagonal=residual)
        old_h2=result[3].ravel()
        return result[0]
    original=evaluate([-.7,.2]);matrix=operator.matrix.toarray().copy()
    assert np.isfinite(evaluate([1e20,1e20]))
    restored=evaluate([-.7,.2])
    np.testing.assert_array_equal(operator.matrix.toarray(),matrix)
    assert restored==original


def test_zero_genetic_variance_preserves_valid_residual_covariance():
    operator=PrecisionOperator(csc_matrix([[2.,.2],[.2,1.5]]),
                               pl.DataFrame({'index':[0,1]}))
    residual=operator.get_diagonal();original=operator.matrix.toarray().copy()
    result=GraphREML._compute_block_likelihood(
        operator,np.array([.3,-.2]),np.ones((2,1)),np.array([[-1000.]]),
        1.,np.zeros(2),10,True,residual_diagonal=residual)
    np.testing.assert_array_equal(result[3],np.zeros((2,1)))
    np.testing.assert_array_equal(operator.matrix.toarray(),original)
    assert np.isfinite(result[0])
