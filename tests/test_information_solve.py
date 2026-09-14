"""Inferential solves preserve curvature and coefficient units."""

import numpy as np
import pytest

from graphld._information import factor_information, solve_information
from graphld.heritability import GraphREML


def test_tiny_curvature_does_not_reverse_newton_update():
    # An absolute +1e-12 ridge on the negative Hessian used to turn this
    # positive, large delete-block update into a negative, tiny update.
    result = GraphREML._compute_pseudojackknife(
        np.full((2, 1), 1e-20), np.full((2, 1, 1), -1e-40), np.zeros(1)
    )
    np.testing.assert_allclose(result, np.full((2, 1), 1e20), rtol=1e-14)


def quadratic_blocks():
    information = np.array([
        [[3., .4, .2], [.4, 2., .1], [.2, .1, 1.]],
        [[2., .1, .3], [.1, 4., .2], [.3, .2, 2.]],
        [[1., .2, .1], [.2, 3., .4], [.1, .4, 4.]],
    ])
    optima = np.array([[1., 2., -1.], [3., -2., .5], [-1., 4., 2.]])
    theta = np.array([.3, -.4, .7])
    score = np.einsum('bij,bj->bi', information, optima-theta)
    return information, optima, theta, score


def test_well_conditioned_delete_optima_are_preserved():
    information, optima, theta, score = quadratic_blocks()
    linear = np.einsum('bij,bj->bi', information, optima)
    expected = np.array([np.linalg.solve(information.sum(0)-a, linear.sum(0)-b)
                         for a, b in zip(information, linear)])
    actual = GraphREML._compute_pseudojackknife(score, -information, theta)
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)


def test_diagonal_coordinate_rescaling_preserves_delete_estimates_and_rank():
    information, _, theta, score = quadratic_blocks()
    expected = GraphREML._compute_pseudojackknife(score, -information, theta)
    # theta = units * beta; the first information diagonal is now about 1e-40.
    units = np.array([1e-20, 1e10, 3.])
    transformed = information * units[None, :, None] * units[None, None, :]
    actual = GraphREML._compute_pseudojackknife(score*units, -transformed, theta/units)
    np.testing.assert_allclose(actual*units, expected, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(factor_information(information.sum(0))[2],
                               factor_information(transformed.sum(0))[2], rtol=1e-13)


@pytest.mark.parametrize('information', [np.zeros((2, 2)), np.ones((2, 2)),
                                        np.array([[1., 2.], [2., 1.]])])
def test_singular_or_indefinite_information_is_rejected(information):
    with pytest.raises(np.linalg.LinAlgError):
        solve_information(information, np.ones(2))


def test_nonfinite_information_score_is_rejected():
    with pytest.raises(np.linalg.LinAlgError, match='score'):
        solve_information(np.eye(2), np.array([np.inf, 1.]))
