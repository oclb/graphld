"""Functions for computing likelihoods in the LDGM model."""

from typing import Optional

import numpy as np

from .precision import PrecisionOperator


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
    #        = -0.5 * (n * log(2π) - log|P| + x^T P x)
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
    n_samples: Optional[int] = None,
    seed: Optional[int] = None,
    trace_estimator: Optional[str] = "exact",
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
    node_grad = -0.5 * (minv_diag - b**2)

    return node_grad if del_M_del_a is None else node_grad @ del_M_del_a


def gaussian_likelihood_hessian(
    pz: np.ndarray,
    M: PrecisionOperator,
    del_M_del_a: np.ndarray,
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
        del_M_del_a: Matrix of derivatives of M's diagonal elements wrt parameters a

    Returns:
        Matrix of second derivatives wrt parameters a
    """

    # Compute b = M^(-1) * pz
    b = M.solve(pz)

    # Compute b_scaled = b .* del_sigma_del_a
    b_scaled = b[:, np.newaxis] * del_M_del_a

    # Compute M^(-1) * b_scaled
    minv_b_scaled = M.solve(b_scaled)

    # Compute Hessian: -1/2 * b_scaled^T * M^(-1) * b_scaled
    hess = -0.5 * (b_scaled.T @ minv_b_scaled)

    return hess
