"""Absolute diagonal assignment avoids cancellation after extreme trial steps."""

import numpy as np
import polars as pl
import pytest
from scipy.sparse import csc_matrix

from graphld.precision import PrecisionOperator


@pytest.fixture
def operator():
    matrix = csc_matrix([[3., -.5, 0.], [-.5, 4., -.5], [0., -.5, 5.]])
    return PrecisionOperator(matrix, pl.DataFrame({'index': [0, 1, 2]}))


def test_huge_trial_roundtrip_restores_exact_diagonal_and_solution(operator):
    original = operator.matrix.toarray().copy()
    rhs = np.array([1., -.5, 2.])
    solution = operator.solve(rhs).copy()
    # Undoing this additive trial by subtraction cannot recover the original
    # diagonal bits. Direct assignment must restore the exact original matrix.
    operator.update_matrix(np.full(3, 1e20))
    operator.solve(rhs)
    operator.set_diagonal(np.diag(original))
    np.testing.assert_array_equal(operator.matrix.toarray(), original)
    np.testing.assert_array_equal(operator.solve(rhs), solution)


def test_assignment_respects_reordered_selection(operator):
    selected = operator[[2, 0]]
    original = operator.matrix.toarray().copy()
    selected.set_diagonal(np.array([11., 7.]))
    expected = original.copy()
    expected[2, 2], expected[0, 0] = 11., 7.
    np.testing.assert_array_equal(operator.matrix.toarray(), expected)
    rhs = np.array([1., 2.])
    expanded_rhs = np.array([2., 0., 1.])
    np.testing.assert_allclose(selected.solve(rhs), np.linalg.solve(expected, expanded_rhs)[[2, 0]])


@pytest.mark.parametrize('values', [
    [2.], [2., 3., 4.], [2., 0.], [2., -1.],
    [2., np.nan], [2., np.inf], [2., -np.inf],
])
def test_invalid_assignment_is_atomic_for_matrix_and_factors(operator, values):
    selected = operator[[2, 0]]
    operator.factor()
    selected.factor()
    original = operator.matrix.data.copy()
    version = operator._current_matrix_version()
    parent_factor, selected_factor = operator._solver, selected._solver
    with pytest.raises(ValueError):
        selected.set_diagonal(values)
    np.testing.assert_array_equal(operator.matrix.data, original)
    assert operator._current_matrix_version() == version
    assert operator._solver is parent_factor
    assert selected._solver is selected_factor
    assert operator._factor_is_current()
    assert selected._factor_is_current()


@pytest.mark.parametrize('alias_type', ['selected', 'direct'])
def test_assignment_invalidates_all_shared_alias_factors(operator, alias_type):
    alias = (operator[[2, 0]] if alias_type == 'selected'
             else PrecisionOperator(operator.matrix, operator.variant_info))
    operator.factor()
    alias.factor()
    previous_version = operator._current_matrix_version()
    alias.set_diagonal([11., 7.] if alias_type == 'selected' else [7., 4., 11.])
    assert operator._current_matrix_version() == previous_version + 1
    assert not operator._factor_is_current()
    assert not alias._factor_is_current()
    expected = np.linalg.solve(operator.matrix.toarray(), np.ones(3))
    np.testing.assert_allclose(operator.solve(np.ones(3)), expected)
    assert operator._factor_is_current()


def test_unchanged_assignment_preserves_factor_and_shared_version(operator):
    selected = operator[[2, 0]]
    operator.factor()
    selected.factor()
    version = operator._current_matrix_version()
    parent_factor, selected_factor = operator._solver, selected._solver
    selected.set_diagonal([5., 3.])
    assert operator._current_matrix_version() == version
    assert operator._solver is parent_factor
    assert selected._solver is selected_factor
    assert operator._factor_is_current()
    assert selected._factor_is_current()


def test_conversion_overflow_is_rejected_before_mutation(operator):
    # Validate the values actually representable in sparse matrix storage.
    operator = PrecisionOperator(operator.matrix.astype(np.float32), operator.variant_info)
    values = np.array([3., 4., 1e100])
    original = operator.matrix.data.copy()
    with pytest.raises(ValueError, match='finite and positive'):
        operator.set_diagonal(values)
    np.testing.assert_array_equal(operator.matrix.data, original)


def test_get_diagonal_returns_independent_copy_in_selection_order(operator):
    selected = operator[[2, 0]]
    diagonal = selected.get_diagonal()
    np.testing.assert_array_equal(diagonal, [5., 3.])
    diagonal[:] = 100.
    np.testing.assert_array_equal(operator.get_diagonal(), [3., 4., 5.])
    selected.set_diagonal([11., 7.])
    np.testing.assert_array_equal(diagonal, [100., 100.])
    np.testing.assert_array_equal(selected.get_diagonal(), [11., 7.])
