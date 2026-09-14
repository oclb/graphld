"""Average-information log-determinant regularization for diagonal covariance models."""

import numpy as np
from scipy.linalg import cho_factor, cho_solve


def information_logdet(information: np.ndarray) -> tuple[float, np.ndarray]:
    """Return half log determinant and inverse of globally summed information.

    A singular matrix has no finite penalty. No ridge or eigenvalue truncation
    is applied, since either would change the regularizer.
    """
    information = (information + information.T) * 0.5
    if not np.isfinite(information).all():
        raise np.linalg.LinAlgError("Information is nonfinite")
    factor = cho_factor(information, lower=True, check_finite=False)
    value = float(np.log(np.diag(factor[0])).sum())
    inverse = cho_solve(factor, np.eye(len(information)), check_finite=False)
    return value, inverse


def information_terms(operator, observations, jacobian):
    """Compute information and intermediates reused by its logdet gradient."""
    b = np.asarray(operator.solve(observations)).reshape(-1)
    u = b[:, None] * jacobian
    v = operator.solve(u)
    # Accelerate can set spurious floating-point flags during finite BLAS work.
    # The global information/gradient validation checks actual output values.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        information = 0.5 * (u.T @ v)
    return information, b, v


def information_penalty_gradient(
    operator, annotations, variant_indices, link_second, jacobian, b, v, inverse
):
    """One block's gradient of half logdet of the GLOBAL information matrix.

    ``link_second`` is d²h/dη² for η = annotations @ parameters, including
    the link denominator. Variant-to-node aggregation is handled explicitly.
    ``inverse`` must be the inverse of the sum of block information matrices.
    The only additional covariance solve is for the vector r.
    """
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        t = v @ inverse
    r = np.einsum("ij,ij->i", jacobian, t)
    s = np.einsum("ij,ij->i", v, t)
    node_term = b * np.asarray(operator.solve(r)).reshape(-1)
    link_term = (
        b[variant_indices] * np.asarray(link_second).reshape(-1)
        * np.einsum("ij,ij->i", annotations, t[variant_indices])
    )
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        return (0.5 * (annotations.T @ link_term)
                - 0.5 * (jacobian.T @ node_term)
                - 0.25 * (jacobian.T @ s))
