"""Functions for computing likelihoods in the LDGM model."""

from typing import Optional, Tuple
import numpy as np

from .precision import PrecisionOperator


def gaussian_likelihood(
    pz: np.ndarray,
    M: PrecisionOperator,
) -> float:
    """Compute log-likelihood of GWAS summary statistics under a Gaussian model.

    The model is:
        beta ~ MVN(0, D)
        z|beta ~ MVN(sqrt(n)*R*beta, R) where R is the LD matrix, n is the sample size
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
) -> np.ndarray:
    """Computes the score under a Gaussian model.

        The model is:
        beta ~ MVN(0, D)
        z|beta ~ MVN(sqrt(n)*R*beta, R) where R is the LD matrix, n is the sample size
        pz = inv(R) * z / sqrt(n)
        M = cov(pz) = D + inv(R)/n

    Args:
        pz: Array of precision-premultiplied GWAS effect size estimates
        M: PrecisionOperator. This should be the covariance of pz.
        del_M_del_a (optional): Matrix of derivatives of the diagonal elements of M with respect to parameters a
    
    Returns:
        Array of diagonal elements of the gradient with respect to the diagonal elements of M,
        or with respect to parameters a if del_M_del_a is provided
        
    """
    # Compute b = M^(-1) * pz
    b = M.solve(pz)
    print(f"b = M\\(P@z) = {b}")
    
    # Compute diagonal elements of M^(-1)
    method = "exact" if n_samples is None else "hutchinson"
    minv_diag = M.inverse_diagonal(method=method, n_samples=n_samples, seed=seed)
    print(f"MinvDiag = {minv_diag}")
    
    # Compute gradient diagonal elements
    print(f"b^2 = {b**2}")
    node_grad = -0.5 * (minv_diag - b**2)
    print(f"nodeGrad = 1/2 * (MinvDiag - b^2) = {node_grad}")
    
    return node_grad if del_M_del_a is None else node_grad @ del_M_del_a


def gaussian_likelihood_hessian(
    pz: np.ndarray,
    M: PrecisionOperator,
    del_M_del_a: np.ndarray,
) -> np.ndarray:
    """Computes the average information matrix of the Gaussian log-likelihood.

    The model is:
        beta ~ MVN(0, D)
        z|beta ~ MVN(sqrt(n)*R*beta, R) where R is the LD matrix, n is the sample size
        pz = inv(R) * z / sqrt(n)
        M = cov(pz) = D + inv(R)/n

    Args:
        pz: Array of precision-premultiplied GWAS effect size estimates
        M: PrecisionOperator. This should be the covariance of pz.
        del_M_del_a: Matrix of derivatives of the diagonal elements of M with respect to parameters a

    Returns:
        Matrix of second derivatives with respect to parameters a
    """
    
    # Compute b = M^(-1) * pz
    b = M.solve(pz)
    
    # Compute b_scaled = b .* del_sigma_del_a
    b_scaled = b[:, np.newaxis] * del_M_del_a
    
    # Compute M^(-1) * b_scaled
    minv_b_scaled = M.solve(b_scaled)
    
    # Compute Hessian: -1/2 * b_scaled^T * M^(-1) * b_scaled
    hess = -0.5 * (b_scaled.T @ minv_b_scaled)
    print(f"hess = {hess}")
    
    return hess
