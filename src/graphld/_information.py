"""Unregularized information solves with coordinate-invariant rank checks."""

import numpy as np
from scipy.linalg import cho_solve


def factor_information(information, rank_tolerance=1e-10):
    """Return diagonal scaling, a normalized Cholesky factor, and minimum eigenvalue.

    Positive diagonal rescaling changes coefficient units without changing the
    normalized rank decision. No proposal ridge is added to inferential curvature.
    """
    information = np.asarray(information, dtype=float)
    if (information.ndim != 2 or information.shape[0] == 0 or information.shape[0] != information.shape[1]
            or not np.isfinite(information).all()):
        raise np.linalg.LinAlgError("Invalid information matrix")
    diagonal = np.diag(information)
    if np.any(diagonal <= 0):
        raise np.linalg.LinAlgError("Information requires positive diagonal entries")
    scale = np.sqrt(diagonal)
    normalized = information / scale[:, None] / scale[None, :]
    if not np.isfinite(normalized).all() or not np.allclose(
            normalized, normalized.T, rtol=1e-10, atol=1e-12):
        raise np.linalg.LinAlgError("Information must be finite and symmetric")
    normalized = .5 * normalized + .5 * normalized.T
    eigenvalues = np.linalg.eigvalsh(normalized)
    if eigenvalues[0] <= max(1., float(eigenvalues[-1])) * rank_tolerance:
        raise np.linalg.LinAlgError("Singular normalized information")
    return scale, np.linalg.cholesky(normalized), float(eigenvalues[0])


def solve_information(information, score):
    """Solve information times update equals score, without regularization."""
    scale, factor, _ = factor_information(information)
    score = np.asarray(score, dtype=float)
    if score.shape != scale.shape or not np.isfinite(score).all():
        raise np.linalg.LinAlgError("Invalid information score")
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        normalized_score = score / scale
    if not np.isfinite(normalized_score).all():
        raise np.linalg.LinAlgError("Nonfinite normalized information score")
    update = cho_solve((factor, True), normalized_score, check_finite=False) / scale
    if not np.isfinite(update).all():
        raise np.linalg.LinAlgError("Nonfinite information update")
    return update
