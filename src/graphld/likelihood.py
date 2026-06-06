"""Functions for computing likelihoods in the LDGM model."""

import warnings
from typing import Optional

import numpy as np

from .precision import PrecisionOperator


class _TraceEstimatorDefault:
    def __repr__(self) -> str:
        return "'xdiag'"


_TRACE_ESTIMATOR_DEFAULT = _TraceEstimatorDefault()


def gaussian_likelihood(
    pz: np.ndarray,
    M: PrecisionOperator,
) -> float:
    """Compute log-likelihood of GWAS summary statistics under a Gaussian model.

    The model is:
        beta ~ MVN(0, D)
        z|beta ~ MVN(sqrt(n)*R*beta, R) where R is the LD matrix, n the sample size
        pz = inv(R) * z / sqrt(n)
        M = cov(pz) = D + inv(R)/n

    Args:
        pz: Array of precision-premultiplied GWAS effect size estimates
        M: PrecisionOperator. This should be the covariance of pz.

    Returns:
        Log-likelihood value

    """
    # Following scipy's convention:
    # log_pdf = -0.5 * (n * log(2π) + log|Σ| + x^T Σ^{-1} x)
    n = len(pz)
    logdet = M.logdet()

    # Compute quadratic form
    b = M.solve(pz)
    quad = np.sum(pz * b)

    # Compute log likelihood
    ll = -0.5 * (n * np.log(2 * np.pi) + logdet + quad)

    return ll


def gaussian_likelihood_gradient(
    pz: np.ndarray,
    M: PrecisionOperator,
    del_M_del_a: Optional[np.ndarray] = None,
    n_samples: int = 10,
    seed: Optional[int] = None,
    trace_estimator: Optional[str] = "xdiag",
) -> np.ndarray:
    """Computes the score under a Gaussian model.

        The model is:
        beta ~ MVN(0, D)
        z|beta ~ MVN(sqrt(n)*R*beta, R) where R is the LD matrix, n the sample size
        pz = inv(R) * z / sqrt(n)
        M = cov(pz) = D + inv(R)/n

    Args:
        pz: Array of precision-premultiplied GWAS effect size estimates
        M: PrecisionOperator. This should be the covariance of pz.
        del_M_del_a: Matrix of derivatives of M's diagonal elements wrt parameters a
        n_samples: Number of probe vectors for Hutchinson's method or xdiag
        seed: Random seed for generating probe vectors
        trace_estimator: Method for computing the trace estimator.
            Options: "exact", "hutchinson", "xdiag"

    Returns:
        Array of diagonal elements of the gradient wrt M's diagonal elements,
        or with respect to parameters a if del_M_del_a is provided
    """
    # Compute b = M^(-1) * pz
    b = M.solve(pz)

    # Compute diagonal elements of M^(-1)
    minv_diag = M.inverse_diagonal(method=trace_estimator,
                                    n_samples=n_samples,
                                    seed=seed)

    # Compute gradient diagonal elements
    node_grad = -0.5 * (minv_diag.flatten() - b.flatten()**2)

    if del_M_del_a is None:
        return node_grad
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        return node_grad @ del_M_del_a


def gaussian_likelihood_hessian(
    pz: np.ndarray,
    M: PrecisionOperator,
    del_M_del_a: Optional[np.ndarray] = None,
    trace_estimator: str = _TRACE_ESTIMATOR_DEFAULT,
    n_samples: int = 100,
    seed: Optional[int] = None,
    *,
    diagonal_method: Optional[str] = None,
) -> np.ndarray:
    """Computes the average information matrix of the Gaussian log-likelihood.

    The model is:
        beta ~ MVN(0, D)
        z|beta ~ MVN(sqrt(n)*R*beta, R) where R is the LD matrix, n the sample size
        pz = inv(R) * z / sqrt(n)
        M = cov(pz) = D + inv(R)/n

    Args:
        pz: Array of precision-premultiplied GWAS effect size estimates
        M: PrecisionOperator. This should be the covariance of pz.
        del_M_del_a: Matrix of derivatives of M's diagonal elements wrt parameters a.
            If None, only the diagonal elements are computed.
        trace_estimator: Method for computing the diagonal trace estimator when
            del_M_del_a is None. Options accepted by PrecisionOperator include
            "exact", "hutchinson", and "xdiag". Defaults to "xdiag".
        n_samples: Number of probe vectors for Hutchinson's method or xdiag
        seed: Random seed for generating probe vectors
        diagonal_method: Deprecated alias for trace_estimator.

    Returns:
        Matrix of second derivatives wrt parameters a, or array of diagonal elements
        if del_M_del_a is None
    """
    trace_estimator_was_supplied = trace_estimator is not _TRACE_ESTIMATOR_DEFAULT
    if diagonal_method is not None:
        if trace_estimator_was_supplied and trace_estimator != diagonal_method:
            raise ValueError(
                "trace_estimator and diagonal_method specify different Hessian "
                "diagonal estimators; use trace_estimator only."
            )
        warnings.warn(
            "diagonal_method is deprecated; use trace_estimator instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        trace_estimator = diagonal_method
    elif not trace_estimator_was_supplied:
        trace_estimator = "xdiag"

    # Compute b = M^(-1) * pz
    b = M.solve(pz)
    if b.ndim == 1:
        b = b.reshape(-1, 1)

    # If del_M_del_a is None, compute only the diagonal of the Hessian
    if del_M_del_a is None:
        minv_diag = M.inverse_diagonal(
            method=trace_estimator, n_samples=n_samples, seed=seed
        )
        hess_diag = -0.5 * minv_diag.flatten() * b.flatten()**2
        return hess_diag

    # Compute b_scaled = b .* del_sigma_del_a
    b_scaled = b * del_M_del_a

    # Compute M^(-1) * b_scaled
    minv_b_scaled = M.solve(b_scaled)

    # Compute Hessian: -1/2 * b_scaled^T * M^(-1) * b_scaled
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        hess = -0.5 * (b_scaled.T @ minv_b_scaled)

    return hess
