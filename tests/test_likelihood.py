"""Tests for likelihood computation functions."""

import numpy as np
import polars as pl
import pytest
from scipy.sparse import csc_matrix
from scipy.stats import multivariate_normal

from graphld.likelihood import (
    gaussian_likelihood,
    gaussian_likelihood_gradient,
    gaussian_likelihood_hessian,
)
from graphld.precision import PrecisionOperator


def test_gaussian_likelihood_basic():
    """Test basic functionality of gaussian_likelihood."""
    # Create a simple positive definite matrix
    data = np.array([2.0, -1.0, -1.0, 2.0], dtype=np.float32)
    indices = np.array([0, 1, 0, 1])
    indptr = np.array([0, 2, 4])
    matrix = csc_matrix((data, indices, indptr), shape=(2, 2))

    # Create variant info
    variant_info = pl.DataFrame({
        'variant_id': ['rs1', 'rs2'],
        'position': [1, 2],
        'chromosome': ['1', '1'],
        'index': [0, 1]
    })

    # Create precision operator
    P = PrecisionOperator(matrix, variant_info)

    # Create test data
    z = np.array([0.5, 0.5], dtype=np.float32)
    pz = matrix @ z  # P @ z

    # Compute likelihood
    ll = gaussian_likelihood(pz, P)

    # Basic sanity checks
    assert np.isfinite(ll)

    # Test with subsetting
    which_indices = np.array([True, False])  # Only use the first element
    pz_sub = pz[which_indices]
    P_sub = P[which_indices]  # Subset the precision operator
    ll_sub = gaussian_likelihood(pz_sub, P_sub)
    assert np.isfinite(ll_sub)


def test_gaussian_likelihood_errors():
    """Test error handling in gaussian_likelihood."""
    # Create simple precision operator
    matrix = csc_matrix(np.eye(2), dtype=np.float32)
    variant_info = pl.DataFrame({
        'variant_id': ['rs1', 'rs2'],
        'position': [1, 2],
        'chromosome': ['1', '1'],
        'index': [0, 1]
    })
    P = PrecisionOperator(matrix, variant_info)

    z = np.array([0.5, 0.5])
    pz = matrix @ z

    # Test with wrong shape
    with pytest.raises(ValueError):
        gaussian_likelihood(pz[:-1], P)


def test_gaussian_likelihood_matches_scipy():
    """Test that our likelihood matches scipy's multivariate normal."""
    # Create a simple positive definite matrix with float64
    data = np.array([2.0, -1.0, -1.0, 2.0], dtype=np.float64)
    indices = np.array([0, 1, 0, 1])
    indptr = np.array([0, 2, 4])
    P = csc_matrix((data, indices, indptr), shape=(2, 2))
    sigmasq = np.array([0.1, 0.2], dtype=np.float64)

    # Create variant info
    variant_info = pl.DataFrame({
        'variant_id': ['rs1', 'rs2'],
        'position': [1, 2],
        'chromosome': ['1', '1'],
        'index': [0, 1]
    })

    # Create precision operator
    P_op = PrecisionOperator(P, variant_info)
    P_op.update_matrix(sigmasq)

    # Create test data with float64
    z = np.array([0.2, 0.5], dtype=np.float64)
    pz = P @ z

    # Compute our likelihood
    ll = gaussian_likelihood(pz, P_op)

    # Compute scipy likelihood
    scipy_ll = multivariate_normal.logpdf(pz, cov=P_op.matrix.toarray())

    np.testing.assert_allclose(ll, scipy_ll, rtol=1e-5)


def test_gaussian_likelihood_gradient():
    """Test that the likelihood gradient matches finite difference approximation."""
    # Create a simple positive definite matrix
    data = np.array([2.0, -1.0, -1.0, 2.0], dtype=np.float32)
    indices = np.array([0, 1, 0, 1])
    indptr = np.array([0, 2, 4])
    matrix = csc_matrix((data, indices, indptr), shape=(2, 2))

    # Create variant info
    variant_info = pl.DataFrame({
        'variant_id': ['rs1', 'rs2'],
        'position': [1, 2],
        'chromosome': ['1', '1'],
        'index': [0, 1]
    })

    # Create test data
    z = np.array([0.5, 0.5], dtype=np.float32)

    # Create precision operator with some diagonal terms
    sigmasq = np.array([0.1, 0.2], dtype=np.float32)
    nn = 10.0

    # Create M = nn*sigmasq + P
    M = PrecisionOperator(matrix.copy() / nn, variant_info)  # Use P directly
    M.update_matrix(sigmasq)  # M = nn*sigmasq + P

    # Compute pz = P @ z
    pz = matrix @ z / np.sqrt(nn)

    # Compute gradient
    grad = gaussian_likelihood_gradient(pz, M, n_samples=None)  # exact inversion

    # Verify gradient using finite differences
    eps = 1e-5
    for i in range(len(sigmasq)):
        # Create two nearby matrices differing only in the i-th diagonal element
        M_plus = PrecisionOperator(matrix.copy() / nn, variant_info)

        # Perturb the i-th diagonal element
        sigmasq_plus = sigmasq.copy()
        sigmasq_plus[i] += eps

        M_plus.update_matrix(sigmasq_plus)  # M = nn*sigmasq + P

        # Compute likelihoods
        ll_plus = gaussian_likelihood(pz, M_plus)
        ll = gaussian_likelihood(pz, M)

        # Finite difference approximation
        grad_fd = (ll_plus - ll) / (eps)  # Scale eps by 1/nn

        # Compare with analytical gradient
        np.testing.assert_allclose(grad[i], grad_fd, rtol=1e-2)


def test_gaussian_likelihood_hessian():
    """Test that the likelihood Hessian matches finite difference of gradient."""
    # Create a simple positive definite matrix
    data = np.array([2.0, -1.0, -1.0, 2.0], dtype=np.float32)
    indices = np.array([0, 1, 0, 1])
    indptr = np.array([0, 2, 4])
    matrix = csc_matrix((data, indices, indptr), shape=(2, 2))

    # Create variant info
    variant_info = pl.DataFrame({
        'variant_id': ['rs1', 'rs2'],
        'position': [1, 2],
        'chromosome': ['1', '1'],
        'index': [0, 1]
    })

    # Create test data
    z = np.array([0.1, 0.1], dtype=np.float32)

    # Create precision operator with some diagonal terms
    sigmasq = np.array([1.0, 1.0], dtype=np.float32)
    nn = 10.0

    # Create M = nn*sigmasq + P
    M = PrecisionOperator(matrix.copy() / nn, variant_info)  # Use P directly
    M.update_matrix(sigmasq)  # M = nn*sigmasq + P

    # Compute pz = P @ z
    pz = matrix @ z / np.sqrt(nn)

    # Create del_sigma_del_a matrix - each column is gradient of sigmasq w.r.t a parameter
    # For testing, use identity matrix (each parameter affects one sigmasq element)
    del_sigma_del_a = np.eye(len(sigmasq), dtype=np.float32)

    # Compute Hessian
    hess = gaussian_likelihood_hessian(pz, M, del_sigma_del_a)

    # Verify Hessian is symmetric
    np.testing.assert_allclose(hess, hess.T, rtol=1e-6)

    # Verify Hessian is negative definite
    eigenvals = np.linalg.eigvals(hess)
    np.testing.assert_array_less(eigenvals, np.zeros_like(eigenvals))

    # Verify against MATLAB values
    matlab_hess = np.array([[-0.000347, -0.000029],
                            [-0.000029, -0.000347]])
    np.testing.assert_allclose(hess, matlab_hess, rtol=5e-3)


def test_gaussian_likelihood_gradient_methods():
    """Test that different diagonal estimation methods give similar gradients."""
    # Create a smaller test matrix for more reliable estimation
    n = 50  # Reduced from 100
    rng = np.random.RandomState(42)

    # Create a random sparse positive definite matrix
    A = np.sqrt(2) * rng.randn(n, n)  # Match MATLAB's randn variance
    A = A @ A.T + np.diag(np.arange(1, n+1))  # Make it positive definite
    matrix = csc_matrix(A)

    # Create variant info
    variant_info = pl.DataFrame({
        'chrom': ['1'] * n,
        'pos': range(n),
        'ref': ['A'] * n,
        'alt': ['T'] * n,
    })

    # Create test data
    z = rng.randn(n)

    # Create precision operator with some diagonal terms
    sigmasq = np.ones(n)  # Unit variances
    nn = 10.0

    # Create M = nn*sigmasq + P
    M = PrecisionOperator(matrix.copy() / nn, variant_info)
    M.update_matrix(sigmasq)

    # Compute pz = P @ z
    pz = matrix @ z / np.sqrt(nn)

    # Compute gradients using different methods
    grad_exact = gaussian_likelihood_gradient(pz, M,
                            trace_estimator="exact")
    grad_hutch = gaussian_likelihood_gradient(pz, M,
                            trace_estimator="hutchinson",
                            n_samples=500, seed=42)
    grad_xdiag = gaussian_likelihood_gradient(pz, M,
                            trace_estimator="xdiag",
                            n_samples=50, seed=42)

    # Compare gradients - they should be approximately equal
    # Use 15% relative tolerance since these are stochastic estimators
    np.testing.assert_allclose(grad_hutch, grad_exact, rtol=0.15)
    np.testing.assert_allclose(grad_xdiag, grad_exact, rtol=0.15)
