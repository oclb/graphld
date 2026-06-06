"""Tests for the PrecisionOperator class."""

import numpy as np
import polars as pl
import pytest
from scipy.sparse import csc_matrix

from graphld.precision import PrecisionOperator


def test_precision_operator_basic():
    """Test basic functionality of PrecisionOperator."""
    # Create a simple 3x3 precision matrix
    data = np.array([2.0, -1.0, -1.0, 2.0, -1.0, 2.0])
    indices = np.array([0, 1, 0, 1, 2, 2])
    indptr = np.array([0, 2, 4, 6])
    matrix = csc_matrix((data, indices, indptr), shape=(3, 3))

    # Create a simple variant info DataFrame
    variant_info = pl.DataFrame({
        'variant_id': ['rs1', 'rs2', 'rs3'],
        'position': [1, 2, 3],
        'chromosome': ['1', '1', '1']
    })

    # Create PrecisionOperator instance
    P = PrecisionOperator(matrix.copy(), variant_info)

    # Test shape property
    assert P.shape == (3, 3)

    # Test dtype property
    assert P.dtype == np.float64

    # Test matrix-vector multiplication with full matrix
    x = np.array([1.0, 1.0, 1.0])
    y = P @ x
    expected = matrix @ x
    np.testing.assert_array_almost_equal(y, expected)

    # Test matrix-vector multiplication with subset
    P_sub = P[[0, 2]]
    x_sub = np.array([1.0, 1.0])
    y_sub = P_sub @ x_sub
    # For this specific submatrix, result should be Schur complement
    # P22 = [[2, 0], [0, 2]], P12 = [-1, 0], P21 = [-1, 0]^T, P11 = [2]
    # Schur complement = P22 - P21 @ P11^(-1) @ P12
    expected_sub = np.array([1.5, 1.0])  # Verified manually
    np.testing.assert_array_almost_equal(y_sub, expected_sub)

def test_precision_operator_indexing():
    """Test indexing functionality of PrecisionOperator."""
    # Create a 4x4 precision matrix
    data = np.array([2.0, -1.0, -1.0, 2.0, -1.0, 2.0, -1.0, 2.0])
    indices = np.array([0, 1, 0, 1, 1, 2, 2, 3])
    indptr = np.array([0, 2, 4, 6, 8])
    matrix = csc_matrix((data, indices, indptr), shape=(4, 4))

    variant_info = pl.DataFrame({
        'variant_id': ['rs1', 'rs2', 'rs3', 'rs4'],
        'position': [1, 2, 3, 4],
        'chromosome': ['1', '1', '1', '1']
    })

    P = PrecisionOperator(matrix.copy(), variant_info)

    # Test boolean indexing
    P_sub = P[[True, True, False, True]]
    assert P_sub.shape == (3, 3)

    # Test slice indexing
    P_sub = P[1:3]
    assert P_sub.shape == (2, 2)

    # Test list indexing
    P_sub = P[[0, 2]]
    assert P_sub.shape == (2, 2)

def test_precision_operator_solve():
    """Test solve functionality of PrecisionOperator."""
    # Create a simple positive definite matrix
    # This matrix is tridiagonal with 2 on diagonal and -1 on off-diagonals
    # The solution to [2 -1 0; -1 2 -1; 0 -1 2]x = [1; 1; 1] is [1.5, 2.0, 1.5]
    data = np.array([2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0], dtype=np.float32)
    indices = np.array([0, 1, 0, 1, 2, 1, 2])  # Fixed indices for symmetric matrix
    indptr = np.array([0, 2, 5, 7])
    matrix = csc_matrix((data, indices, indptr), shape=(3, 3))

    # Verify solution with numpy for reference
    dense = matrix.toarray()
    dense_solution = np.linalg.solve(dense, np.ones(3, dtype=np.float32))

    variant_info = pl.DataFrame({
        'variant_id': ['rs1', 'rs2', 'rs3'],
        'position': [1, 2, 3],
        'chromosome': ['1', '1', '1']
    })

    P = PrecisionOperator(matrix.copy(), variant_info)

    # Test direct solver
    b = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    x_direct = P.solve(b, method='direct')

    # The solution should satisfy Px = b
    # For this matrix, the exact solution is [1.5, 2.0, 1.5]
    np.testing.assert_array_almost_equal(x_direct, dense_solution, decimal=5)

    # Test with subset
    P_sub = P[[0, 2]]
    b_sub = np.array([1.0, 1.0], dtype=np.float32)
    x_sub = P_sub.solve(b_sub, method='direct')
    expected_sub = np.array([1.0, 1.0], dtype=np.float32)  # For this specific submatrix
    np.testing.assert_array_almost_equal(x_sub, expected_sub, decimal=5)

    # Test residual with subset
    residual = np.linalg.norm(P_sub @ x_sub - b_sub)
    assert residual < 1e-5  # Use a more reasonable tolerance for float32

    # Function to count iterations
    def make_callback():
        iterations = [0]
        def callback(xk):
            iterations[0] += 1
        return callback, iterations

    # Test PCG solver
    x_pcg = P.solve(b, method='pcg')
    residual = np.linalg.norm(P @ x_pcg - b)
    assert residual < 1e-5  # PCG tolerance is typically larger

    callback, iterations = make_callback()
    x_pcg_callback = P.solve(b, method='pcg', callback=callback)
    np.testing.assert_array_almost_equal(x_pcg_callback, dense_solution, decimal=5)
    assert iterations[0] > 0

    # Test multiple right-hand sides
    B = np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]], dtype=np.float32)
    X = P.solve(B, method='direct')
    assert X.shape == (3, 2)
    residual = np.linalg.norm(P @ X - B)
    assert residual < 1e-5

    # Test with different perturbation sizes
    perturbations = [0.1, 1.0, 10.0]  # Small, medium, large perturbations
    for perturbation in perturbations:
        # Create perturbed matrix
        perturbed_data = data.copy()
        perturbed_data[0] += perturbation
        perturbed_matrix = csc_matrix((perturbed_data, indices, indptr), shape=(3, 3))

        # Create new operator with perturbed matrix
        P_perturbed = PrecisionOperator(perturbed_matrix, variant_info)

        # Test PCG solver with perturbed matrix
        callback, iterations = make_callback()
        x_pcg_perturbed = P_perturbed.solve(b, method='pcg', callback=callback)
        residual = np.linalg.norm(P_perturbed @ x_pcg_perturbed - b)
        assert residual < 1e-5  # Should still converge, might take more iterations

def test_precision_operator_pcg_subset_with_stale_factor():
    """Test PCG on subsetted operators after the preconditioner becomes stale."""
    data = np.array([
        4.0, -1.0,
        -1.0, 4.0, -1.0,
        -1.0, 4.0, -1.0,
        -1.0, 4.0,
    ], dtype=np.float64)
    indices = np.array([0, 1, 0, 1, 2, 1, 2, 3, 2, 3])
    indptr = np.array([0, 2, 5, 8, 10])
    matrix = csc_matrix((data, indices, indptr), shape=(4, 4))
    variant_info = pl.DataFrame({
        'variant_id': ['rs1', 'rs2', 'rs3', 'rs4'],
        'position': [1, 2, 3, 4],
        'chromosome': ['1', '1', '1', '1']
    })

    P = PrecisionOperator(matrix.copy(), variant_info)
    P_sub = P[[0, 2, 3]]
    P_sub.factor()
    P_sub.update_matrix(np.array([0.25, 0.5, 0.75]))

    b = np.array([1.0, -0.5, 2.0])
    x_pcg = P_sub.solve(b, method='pcg', tol=1e-10)

    assert x_pcg.shape == b.shape
    np.testing.assert_allclose(P_sub @ x_pcg, b, rtol=1e-8, atol=1e-8)

def test_precision_operator_pcg_preserves_subset_order():
    """Test PCG respects explicit list-index order for subsetted operators."""
    data = np.array([
        4.0, -1.0,
        -1.0, 4.0, -1.0,
        -1.0, 4.0, -1.0,
        -1.0, 4.0,
    ], dtype=np.float64)
    indices = np.array([0, 1, 0, 1, 2, 1, 2, 3, 2, 3])
    indptr = np.array([0, 2, 5, 8, 10])
    matrix = csc_matrix((data, indices, indptr), shape=(4, 4))
    variant_info = pl.DataFrame({
        'variant_id': ['rs1', 'rs2', 'rs3', 'rs4'],
        'position': [1, 2, 3, 4],
        'chromosome': ['1', '1', '1', '1']
    })

    P = PrecisionOperator(matrix.copy(), variant_info)
    P_sub = P[[2, 0]]
    b = np.array([2.0, -0.5])

    x_direct = P_sub.solve(b, method='direct')
    x_pcg = P_sub.solve(b, method='pcg', tol=1e-10)

    np.testing.assert_allclose(x_pcg, x_direct, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(P_sub @ x_pcg, b, rtol=1e-8, atol=1e-8)

def test_precision_operator_pcg_accepts_vector_initialization():
    """Test PCG reshapes 1-D initial guesses for vector right-hand sides."""
    data = np.array([2.0, -1.0, -1.0, 2.0], dtype=np.float64)
    indices = np.array([0, 1, 0, 1])
    indptr = np.array([0, 2, 4])
    matrix = csc_matrix((data, indices, indptr), shape=(2, 2))
    variant_info = pl.DataFrame({
        'variant_id': ['rs1', 'rs2'],
        'position': [1, 2],
        'chromosome': ['1', '1']
    })

    P = PrecisionOperator(matrix.copy(), variant_info)
    b = np.array([1.0, 0.5])
    initialization = np.zeros_like(b)

    x_pcg = P.solve(b, method='pcg', initialization=initialization)
    np.testing.assert_allclose(P @ x_pcg, b, rtol=1e-8, atol=1e-8)

def test_precision_operator_inverse_diagonal_subset_with_stale_factor():
    """Test stochastic inverse-diagonal estimates on a subsetted stale-factor operator."""
    data = np.array([
        4.0, -1.0,
        -1.0, 4.0, -1.0,
        -1.0, 4.0,
    ], dtype=np.float64)
    indices = np.array([0, 1, 0, 1, 2, 1, 2])
    indptr = np.array([0, 2, 5, 7])
    matrix = csc_matrix((data, indices, indptr), shape=(3, 3))
    variant_info = pl.DataFrame({
        'variant_id': ['rs1', 'rs2', 'rs3'],
        'position': [1, 2, 3],
        'chromosome': ['1', '1', '1']
    })

    P = PrecisionOperator(matrix.copy(), variant_info)
    P_sub = P[[0, 2]]
    P_sub.factor()
    P_sub.update_matrix(np.array([0.5, 0.25]))

    probes = np.array([[1.0, 1.0], [1.0, -1.0]])
    exact_diag = P_sub.inverse_diagonal(method='exact')
    hutch_diag, hutch_y = P_sub.inverse_diagonal(
        method='hutchinson',
        initialization=(probes, probes.copy()),
    )

    assert hutch_y.shape == probes.shape
    np.testing.assert_allclose(hutch_diag, exact_diag, rtol=1e-8, atol=1e-8)

def test_precision_operator_update():
    """Test updating the precision matrix."""
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
    P = PrecisionOperator(matrix.copy(), variant_info)

    # Test updating the precision matrix
    update = np.array([1.0, 1.0], dtype=np.float32)
    print("\ntest_precision_operator_update:")
    print("Original matrix:", matrix.toarray())
    P.update_matrix(update)
    print("After update:", P.matrix.toarray())

    expected = matrix.toarray()
    expected[0,0] = 3.0  # Original 2.0 + update 1.0
    expected[1,1] = 3.0  # Original 2.0 + update 1.0
    print("Expected:", expected)
    print("Diagonal indices:", P.diagonal_indices)
    np.testing.assert_array_almost_equal(P.matrix.toarray(), expected)

def test_precision_operator_update_with_indices():
    """Test updating the precision matrix with indices."""
    # Create a 4x4 precision matrix
    data = np.array([2.0, -1.0, -1.0, 2.0, -1.0, 2.0, -1.0, 2.0], dtype=np.float32)
    indices = np.array([0, 1, 0, 1, 1, 2, 2, 3])
    indptr = np.array([0, 2, 4, 6, 8])
    matrix = csc_matrix((data, indices, indptr), shape=(4, 4))

    variant_info = pl.DataFrame({
        'variant_id': ['rs1', 'rs2', 'rs3', 'rs4'],
        'position': [1, 2, 3, 4],
        'chromosome': ['1', '1', '1', '1']
    })

    P = PrecisionOperator(matrix.copy(), variant_info)

    # Test updating with boolean mask
    P_sub = P[[0, 2]]  # 2x2 submatrix
    update = np.array([1.0, 1.0], dtype=np.float32)
    print("\ntest_precision_operator_update_with_indices:")
    print("Original matrix:", matrix.toarray())
    print("P_sub shape:", P_sub.shape)
    print("P_sub which_indices:", P_sub._which_indices)
    P_sub.update_matrix(update)
    print("After update:", P_sub.matrix.toarray())

    # Check that only the selected elements were updated
    expected = matrix.toarray()
    expected[0, 0] = 3.0  # Original 2.0 + update 1.0
    expected[2, 2] = 3.0  # Original 2.0 + update 1.0
    print("Expected:", expected)
    print("Diagonal indices:", P_sub.diagonal_indices)
    np.testing.assert_array_almost_equal(P.matrix.toarray(), expected)

    # Test updating with integer array
    P_sub = P[1:3]  # Another 2x2 submatrix
    update = np.array([2.0, 2.0], dtype=np.float32)
    P_sub.update_matrix(update)

    # Check that only the selected elements were updated
    expected[1, 1] = 4.0  # Original 2.0 + update 2.0
    expected[2, 2] = 5.0  # Original 2.0 + first update 1.0 + second update 2.0
    print("Expected:", expected)
    print("Diagonal indices:", P_sub.diagonal_indices)
    np.testing.assert_array_almost_equal(P.matrix.toarray(), expected)

    # Test error on shape mismatch
    with pytest.raises(ValueError, match="Update vector length .* does not match matrix shape"):
        update = np.array([[1.0]], dtype=np.float32)  # Use dense array instead of sparse
        P_sub.update_matrix(update)

def test_precision_operator_update_matrix_is_atomic_on_invalid_update():
    """Test that invalid matrix updates do not partially mutate state."""
    matrix = csc_matrix(np.array([
        [2.0, -0.2, 0.0],
        [-0.2, 2.0, -0.2],
        [0.0, -0.2, 2.0],
    ]))
    variant_info = pl.DataFrame({
        'variant_id': ['rs1', 'rs2', 'rs3'],
        'position': [1, 2, 3],
        'chromosome': ['1', '1', '1'],
        'index': [0, 1, 2],
    })
    P = PrecisionOperator(matrix.copy(), variant_info)
    P.factor()
    solver = P._solver
    before = P.matrix.toarray().copy()

    with pytest.raises(ValueError, match="non-positive"):
        P.update_matrix(np.array([-3.0, 0.0, 0.0]))

    np.testing.assert_array_equal(P.matrix.toarray(), before)
    assert P._solver is solver
    assert P._cholesky_is_up_to_date

def test_precision_operator_copy_does_not_share_solver():
    """Test that copied operators lazily build independent Cholesky solvers."""
    matrix = csc_matrix(np.array([
        [2.0, -0.2, 0.0],
        [-0.2, 2.0, -0.2],
        [0.0, -0.2, 2.0],
    ]))
    variant_info = pl.DataFrame({
        'variant_id': ['rs1', 'rs2', 'rs3'],
        'position': [1, 2, 3],
        'chromosome': ['1', '1', '1'],
        'index': [0, 1, 2],
    })
    P = PrecisionOperator(matrix.copy(), variant_info)
    P.factor()

    copied = P.copy()
    assert copied._solver is None
    assert not copied._cholesky_is_up_to_date

    b = np.ones(3)
    np.testing.assert_allclose(copied.solve(b), P.solve(b))
    assert copied._solver is not None
    assert copied._solver is not P._solver

def test_precision_operator_set_which_indices_invalidates_selection_cache():
    """Test that repeated selection changes recompute cached mask and diagonal indices."""
    matrix = csc_matrix(np.array([
        [2.0, -0.2, 0.0],
        [-0.2, 2.0, -0.2],
        [0.0, -0.2, 2.0],
    ]))
    variant_info = pl.DataFrame({
        'variant_id': ['rs1', 'rs2', 'rs3'],
        'position': [1, 2, 3],
        'chromosome': ['1', '1', '1'],
        'index': [0, 1, 2],
    })
    P = PrecisionOperator(matrix.copy(), variant_info)

    P.set_which_indices([0, 2])
    np.testing.assert_array_equal(P._get_mask, [True, False, True])
    first_diagonal_indices = P.diagonal_indices.copy()

    P.set_which_indices([0])
    np.testing.assert_array_equal(P._get_mask, [True, False, False])
    assert len(P.diagonal_indices) == 1
    assert not np.array_equal(P.diagonal_indices, first_diagonal_indices)

    P.update_matrix(np.array([1.0]))
    expected = matrix.toarray()
    expected[0, 0] = 3.0
    np.testing.assert_array_equal(P.matrix.toarray(), expected)

def test_precision_operator_update_matrix_preserves_selection_order():
    """Test diagonal updates follow caller-visible subset order."""
    matrix = csc_matrix(np.array([
        [3.0, -0.5, 0.0],
        [-0.5, 3.0, -0.5],
        [0.0, -0.5, 3.0],
    ]))
    variant_info = pl.DataFrame({
        'variant_id': ['rs1', 'rs2', 'rs3'],
        'position': [1, 2, 3],
        'chromosome': ['1', '1', '1'],
        'index': [0, 1, 2],
    })
    P = PrecisionOperator(matrix.copy(), variant_info)
    P_sub = P[[2, 0]]

    P_sub.update_matrix(np.array([10.0, 1.0]))

    expected = matrix.toarray()
    expected[2, 2] = 13.0
    expected[0, 0] = 4.0
    np.testing.assert_array_equal(P.matrix.toarray(), expected)

def test_precision_operator_update_element_preserves_selection_order_with_factor():
    """Test rank-one factor updates target the selected global index."""
    matrix = csc_matrix(np.array([
        [3.0, -0.5, 0.0],
        [-0.5, 3.0, -0.5],
        [0.0, -0.5, 3.0],
    ]))
    variant_info = pl.DataFrame({
        'variant_id': ['rs1', 'rs2', 'rs3'],
        'position': [1, 2, 3],
        'chromosome': ['1', '1', '1'],
        'index': [0, 1, 2],
    })
    P = PrecisionOperator(matrix.copy(), variant_info)
    P_sub = P[[2, 0]]
    b = np.array([1.0, -0.5])
    P_sub.factor()

    P_sub.update_element(0, 10.0)

    expected = matrix.toarray()
    expected[2, 2] = 13.0
    np.testing.assert_array_equal(P.matrix.toarray(), expected)
    np.testing.assert_allclose(P_sub @ P_sub.solve(b), b, rtol=1e-8, atol=1e-8)
    assert P_sub._factor_is_current()

def test_precision_operator_subset_update_invalidates_parent_factor():
    """Test that shared-matrix updates through a subset stale parent factors."""
    matrix = csc_matrix(np.array([
        [3.0, -0.5, 0.0],
        [-0.5, 3.0, -0.5],
        [0.0, -0.5, 3.0],
    ]))
    variant_info = pl.DataFrame({
        'variant_id': ['rs1', 'rs2', 'rs3'],
        'position': [1, 2, 3],
        'chromosome': ['1', '1', '1'],
        'index': [0, 1, 2],
    })
    P = PrecisionOperator(matrix.copy(), variant_info)
    b = np.ones(3)
    P.factor()
    stale_solver = P._solver

    P_sub = P[[0, 2]]
    P_sub.update_matrix(np.array([2.0, 1.0]))

    P_fresh = PrecisionOperator(P.matrix.copy(), variant_info)
    np.testing.assert_allclose(P.solve(b), P_fresh.solve(b))
    assert P._solver is stale_solver
    assert P._factor_is_current()

def test_precision_operator_subset_update_element_invalidates_parent_factor():
    """Test parent factors are refreshed after a selected view updates one element."""
    matrix = csc_matrix(np.array([
        [3.0, -0.5, 0.0],
        [-0.5, 3.0, -0.5],
        [0.0, -0.5, 3.0],
    ]))
    variant_info = pl.DataFrame({
        'variant_id': ['rs1', 'rs2', 'rs3'],
        'position': [1, 2, 3],
        'chromosome': ['1', '1', '1'],
        'index': [0, 1, 2],
    })
    P = PrecisionOperator(matrix.copy(), variant_info)
    b = np.ones(3)
    P.factor()

    P_sub = P[[1]]
    P_sub.update_element(0, 2.0)

    P_fresh = PrecisionOperator(P.matrix.copy(), variant_info)
    np.testing.assert_allclose(P.solve(b), P_fresh.solve(b))
    assert P._factor_is_current()

def test_precision_operator_direct_shared_matrix_update_invalidates_peer_factor():
    """Test direct operators over the same matrix share mutation state."""
    matrix = csc_matrix(np.array([
        [3.0, -0.5, 0.0],
        [-0.5, 3.0, -0.5],
        [0.0, -0.5, 3.0],
    ]))
    variant_info = pl.DataFrame({
        'variant_id': ['rs1', 'rs2', 'rs3'],
        'position': [1, 2, 3],
        'chromosome': ['1', '1', '1'],
        'index': [0, 1, 2],
    })
    P1 = PrecisionOperator(matrix, variant_info)
    P2 = PrecisionOperator(matrix, variant_info)
    b = np.ones(3)
    P2.factor()

    P1.update_matrix(np.array([2.0, 0.0, 1.0]))

    P_fresh = PrecisionOperator(matrix.copy(), variant_info)
    np.testing.assert_allclose(P2.solve(b), P_fresh.solve(b))
    assert P2._factor_is_current()

def test_precision_operator_copy_matrix_alias_invalidates_copy_factor():
    """Test direct aliases of copied matrices share the copy's mutation state."""
    matrix = csc_matrix(np.array([
        [3.0, -0.5, 0.0],
        [-0.5, 3.0, -0.5],
        [0.0, -0.5, 3.0],
    ]))
    variant_info = pl.DataFrame({
        'variant_id': ['rs1', 'rs2', 'rs3'],
        'position': [1, 2, 3],
        'chromosome': ['1', '1', '1'],
        'index': [0, 1, 2],
    })
    P = PrecisionOperator(matrix, variant_info)
    copied = P.copy()
    alias = PrecisionOperator(copied.matrix, copied.variant_info)
    b = np.ones(3)
    copied.factor()

    alias.update_matrix(np.array([2.0, 0.0, 1.0]))

    fresh = PrecisionOperator(copied.matrix.copy(), variant_info)
    np.testing.assert_allclose(copied.solve(b), fresh.solve(b))
    assert copied._factor_is_current()

def test_precision_operator_update_element():
    """Test updating a single diagonal element."""
    # Create a 4x4 precision matrix that is better conditioned
    data = np.array([
        4.0, -1.0,       # row 0
        -1.0, 4.0, -1.0, # row 1
        -1.0, 4.0, -1.0, # row 2
        -1.0, 4.0        # row 3
    ], dtype=np.float32)
    indices = np.array([0, 1, 0, 1, 2, 1, 2, 3, 2, 3])
    indptr = np.array([0, 2, 5, 8, 10])
    matrix = csc_matrix((data, indices, indptr), shape=(4, 4))

    variant_info = pl.DataFrame({
        'variant_id': ['rs1', 'rs2', 'rs3', 'rs4'],
        'position': [1, 2, 3, 4],
        'chromosome': ['1', '1', '1', '1']
    })

    P = PrecisionOperator(matrix.copy(), variant_info)

    # First test: verify Cholesky update with positive value
    P.factor()
    assert P._cholesky_is_up_to_date

    # Get initial solution to a linear system
    b = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    x1 = P.solve(b)
    print("\nInitial solution:", x1)
    print("Initial matrix:\n", P.matrix.toarray())

    # Update element with positive value
    P.update_element(1, 1.0)
    assert P._cholesky_is_up_to_date  # Should still be true after update
    print("\nAfter update matrix:\n", P.matrix.toarray())

    # Solve same system with updated matrix
    x2 = P.solve(b)
    print("Updated solution:", x2)

    # Verify solution is correct by comparing with fresh factorization
    P_fresh = PrecisionOperator(P.matrix, variant_info)
    x3 = P_fresh.solve(b)
    print("Fresh solution:", x3)

    # Compute residuals
    r2 = b - P.matrix @ x2
    r3 = b - P.matrix @ x3
    print("\nResidual norms:")
    print("Updated Cholesky:", np.linalg.norm(r2))
    print("Fresh Cholesky:", np.linalg.norm(r3))

    np.testing.assert_array_almost_equal(x2, x3)

    # Second test: verify Cholesky update with negative value
    P = PrecisionOperator(matrix.copy(), variant_info)
    P.factor()
    assert P._cholesky_is_up_to_date

    # Get initial solution
    x1 = P.solve(b)
    print("\nInitial solution (negative update):", x1)
    print("Initial matrix:\n", P.matrix.toarray())

    # Update element with negative value (but not making it non-positive)
    P.update_element(1, -0.5)  # Original value is 4.0
    assert P._cholesky_is_up_to_date
    print("\nAfter negative update matrix:\n", P.matrix.toarray())

    # Solve same system with updated matrix
    x2 = P.solve(b)
    print("Updated solution:", x2)

    # Verify solution is correct by comparing with fresh factorization
    P_fresh = PrecisionOperator(P.matrix, variant_info)
    x3 = P_fresh.solve(b)
    print("Fresh solution:", x3)

    # Compute residuals
    r2 = b - P.matrix @ x2
    r3 = b - P.matrix @ x3
    print("\nResidual norms:")
    print("Updated Cholesky:", np.linalg.norm(r2))
    print("Fresh Cholesky:", np.linalg.norm(r3))

    np.testing.assert_array_almost_equal(x2, x3)

    # Third test: basic matrix updates
    P = PrecisionOperator(matrix.copy(), variant_info)

    # Test rank-one update (positive value)
    P.update_element(1, 1.0)
    expected = matrix.toarray()
    expected[1, 1] = 5.0  # Original 4.0 + update 1.0
    np.testing.assert_array_almost_equal(P.matrix.toarray(), expected)

    # Test rank-one downdate (negative value)
    P.update_element(0, -0.5)
    expected[0, 0] = 3.5  # Original 4.0 - update 0.5
    np.testing.assert_array_almost_equal(P.matrix.toarray(), expected)

    # Test with indices
    P_sub = P[1:]  # 3x3 submatrix
    P_sub.update_element(0, 1.0)  # Updates global index 1
    expected[1, 1] = 6.0  # Original 4.0 + first update 1.0 + second update 1.0
    np.testing.assert_array_almost_equal(P.matrix.toarray(), expected)

    # Test with mask
    mask = np.array([False, True, True, True])
    P_sub = P[mask]  # Another 3x3 submatrix
    P_sub.update_element(1, 2.0)  # Updates global index 2
    expected[2, 2] = 6.0  # Original 4.0 + update 2.0
    np.testing.assert_array_almost_equal(P.matrix.toarray(), expected)

    # Test error on out of bounds index
    with pytest.raises(IndexError):
        P.update_element(4, 1.0)

    # Test error on out of bounds index with indices
    with pytest.raises(IndexError):
        P_sub = P[1:]
        P_sub.update_element(3, 1.0)

def test_precision_operator_errors():
    """Test error handling in PrecisionOperator."""
    # Create a simple matrix
    data = np.array([2.0, -1.0, -1.0, 2.0])
    indices = np.array([0, 1, 0, 1])
    indptr = np.array([0, 2, 4])
    matrix = csc_matrix((data, indices, indptr), shape=(2, 2))

    variant_info = pl.DataFrame({
        'variant_id': ['rs1', 'rs2'],
        'position': [1, 2],
        'chromosome': ['1', '1']
    })

    P = PrecisionOperator(matrix.copy(), variant_info)

    # Test invalid solve method
    with pytest.raises(ValueError):
        P.solve(np.array([1.0, 1.0]), method='invalid')

    # Test mismatched dimensions in solve
    with pytest.raises(ValueError):
        P.solve(np.array([1.0, 1.0, 1.0]))

    # Test invalid indexing
    with pytest.raises(TypeError):
        P[0, 1]  # Only single-axis indexing is supported

def test_precision_operator_inverse_diagonal_methods():
    """Test different methods for computing inverse diagonal elements."""
    # Create a small test matrix to make exact computation feasible
    n = 200
    num_samples = 50
    rng = np.random.RandomState(42)

    # Create a random sparse positive definite matrix
    # Scale randn to match MATLAB's distribution
    A = np.sqrt(2) * rng.randn(n, n)  # MATLAB's randn has variance 1, numpy's has variance 1/2
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        A = A @ A.T + np.diag(np.arange(1, n+1))  # Make it positive definite
    A = csc_matrix(A)

    # Create variant info DataFrame
    variant_info = pl.DataFrame({
        "chrom": ["1"] * n,
        "pos": range(n),
        "ref": ["A"] * n,
        "alt": ["T"] * n,
    })

    # Create PrecisionOperator
    P = PrecisionOperator(A, variant_info)

    # Generate Rademacher random variables for xdiag
    v_xdiag = 2 * (rng.rand(n, num_samples) > 0.5).astype(float) - 1
    pv_xdiag = v_xdiag.copy()

    # Compute diagonal using all methods
    exact_diag = P.inverse_diagonal(method="exact")
    hutch_diag = P.inverse_diagonal(method="hutchinson", n_samples=num_samples, seed=42)
    xdiag_diag = P.inverse_diagonal(method="xdiag", initialization=(v_xdiag, pv_xdiag))

    # Compare trace estimates - they should be approximately equal
    np.testing.assert_allclose(np.sum(hutch_diag), np.sum(exact_diag), rtol=0.05)

    # For xdiag, we get back a tuple of (diag_estimate, Y) when initialization is provided
    xdiag_diag_est = xdiag_diag[0] if isinstance(xdiag_diag, tuple) else xdiag_diag
    np.testing.assert_allclose(np.sum(xdiag_diag_est), np.sum(exact_diag), rtol=0.05)

    # Test with initialization
    v = rng.choice([-1, 1], size=(n, num_samples))
    pv = v.copy()

    # Test hutchinson with initialization
    hutch_diag_init, hutch_y = P.inverse_diagonal(
        initialization=(v, pv),
        method="hutchinson"
    )

    np.testing.assert_allclose(np.sum(hutch_diag_init), np.sum(exact_diag), rtol=0.05)

    # Test xdiag with initialization
    xdiag_diag_init, xdiag_y = P.inverse_diagonal(
        initialization=(v_xdiag, pv_xdiag),
        method="xdiag"
    )

    np.testing.assert_allclose(np.sum(xdiag_diag_init), np.sum(exact_diag), rtol=0.05)

    # Verify shapes of returned values
    assert hutch_y.shape == v.shape
    assert xdiag_y.shape == v_xdiag.shape

    # Test error cases
    with pytest.raises(ValueError):
        P.inverse_diagonal(method="invalid")

    with pytest.raises(ValueError):
        P.inverse_diagonal(method="xnys")

    with pytest.raises(ValueError):
        P.inverse_diagonal(initialization=(v, pv), method="exact")

def test_precision_operator_scalar_multiplication():
    """Test scalar multiplication of PrecisionOperator."""
    # Create a simple 3x3 precision matrix
    data = np.array([2.0, -1.0, -1.0, 2.0, -1.0, 2.0])
    indices = np.array([0, 1, 0, 1, 2, 2])
    indptr = np.array([0, 2, 4, 6])
    matrix = csc_matrix((data, indices, indptr), shape=(3, 3))

    # Create variant info
    variant_info = pl.DataFrame({
        'variant_id': ['rs1', 'rs2', 'rs3'],
        'position': [1, 2, 3],
        'chromosome': ['1', '1', '1']
    })

    # Create PrecisionOperator instance
    P = PrecisionOperator(matrix, variant_info)

    # Test vector for multiplication
    x = np.array([1.0, 2.0, 3.0])
    scalar = 2.5

    # Test that scalar * (P @ x) equals (scalar * P) @ x
    result1 = scalar * (P @ x)
    result2 = (scalar * P) @ x
    np.testing.assert_array_almost_equal(result1, result2)

    # Test with subset of matrix
    P_sub = P[[0, 2]]
    x_sub = np.array([1.0, 3.0])
    result1_sub = scalar * (P_sub @ x_sub)
    result2_sub = (scalar * P_sub) @ x_sub
    np.testing.assert_array_almost_equal(result1_sub, result2_sub)

def test_precision_operator_xdiag_small():
    """Test xdiag method with a small 4x4 example that matches MATLAB output."""
    n = 4

    # Create the exact test matrix from MATLAB
    A = np.array([
        [1, 1, 1, 1],
        [1, 2, 0, 0],
        [1, 0, 3, 0],
        [1, 0, 0, 4]
    ])
    A = csc_matrix(A)

    # Create variant info DataFrame
    variant_info = pl.DataFrame({
        "chrom": ["1"] * n,
        "pos": range(n),
        "ref": ["A"] * n,
        "alt": ["T"] * n,
    })

    # Create PrecisionOperator
    P = PrecisionOperator(A, variant_info)

    # Generate the exact test vectors from MATLAB
    v = np.array([
        [ 1,  1, -1],
        [ 1, -1, -1],
        [-1,  1, -1],
        [-1,  1,  1]
    ])
    pv = v.copy()

    matlab_diag = np.array([-11.9946, -2.3328, -0.5468, -0.4516])

    # Compute diagonal using xdiag
    xdiag_diag = P.inverse_diagonal(method="xdiag", initialization=(v, pv))

    # For xdiag, we get back a tuple of (diag_estimate, Y) when initialization is provided
    xdiag_diag_est = xdiag_diag[0] if isinstance(xdiag_diag, tuple) else xdiag_diag

    # Verify that our final estimate matches MATLAB's
    np.testing.assert_allclose(xdiag_diag_est, matlab_diag, rtol=1e-4)

def test_precision_operator_nbytes():
    """Test memory usage calculation of PrecisionOperator."""
    # Create a simple 3x3 precision matrix
    data = np.array([2.0, -1.0, -1.0, 2.0, -1.0, 2.0])
    indices = np.array([0, 1, 0, 1, 2, 2])
    indptr = np.array([0, 2, 4, 6])
    matrix = csc_matrix((data, indices, indptr), shape=(3, 3))

    # Create variant info
    variant_info = pl.DataFrame({
        'variant_id': ['rs1', 'rs2', 'rs3'],
        'position': [1, 2, 3],
        'chromosome': ['1', '1', '1']
    })

    # Create PrecisionOperator instance
    P = PrecisionOperator(matrix, variant_info)

    # Get initial size
    initial_size = P.nbytes

    # Verify size includes matrix components
    matrix_size = matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes
    assert initial_size >= matrix_size

    # Factor the matrix and check size increases
    P.factor()
    factored_size = P.nbytes
    assert factored_size > initial_size

    # Create a subset and verify size changes
    P_sub = P[[0, 2]]
    assert P_sub.nbytes != P.nbytes

def test_precision_operator_delete_factor():
    """Test deleting Cholesky factorization."""
    # Create a simple 3x3 precision matrix
    data = np.array([2.0, -1.0, -1.0, 2.0, -1.0, 2.0])
    indices = np.array([0, 1, 0, 1, 2, 2])
    indptr = np.array([0, 2, 4, 6])
    matrix = csc_matrix((data, indices, indptr), shape=(3, 3))

    # Create variant info
    variant_info = pl.DataFrame({
        'variant_id': ['rs1', 'rs2', 'rs3'],
        'position': [1, 2, 3],
        'chromosome': ['1', '1', '1']
    })

    # Create PrecisionOperator instance
    P = PrecisionOperator(matrix, variant_info)

    # Get initial size
    initial_size = P.nbytes

    # Factor the matrix and check size increases
    P.factor()
    factored_size = P.nbytes
    assert factored_size > initial_size

    # Delete the factor and verify size returns to initial
    P.del_factor()
    freed_size = P.nbytes
    assert freed_size == initial_size

    # Verify we can re-factor after deleting
    P.factor()
    refactored_size = P.nbytes
    assert refactored_size == factored_size

def test_precision_operator_variant_solve():
    """Test variant_solve functionality of PrecisionOperator.

    This test verifies that variant_solve correctly handles cases where multiple variants
    share the same index (perfect LD). The function should:
    1. Sum the input values for variants with the same index
    2. Solve the system using these summed values
    3. Assign the solution back to all variants that share an index
    """
    # Create a 2x2 precision matrix (2 indices)
    data = np.array([2.0, -1.0, -1.0, 2.0], dtype=np.float32)
    indices = np.array([0, 1, 0, 1])
    indptr = np.array([0, 2, 4])
    matrix = csc_matrix((data, indices, indptr), shape=(2, 2))

    # Create variant info with 3 variants, where two share the same index
    variant_info = pl.DataFrame({
        'variant_id': ['rs1', 'rs2', 'rs3'],
        'position': [1, 2, 3],
        'chromosome': ['1', '1', '1']
    })
    indices = [0, 0, 1]  # First two variants share index 0
    variant_info = variant_info.with_columns(pl.Series('index', indices))

    # Create precision operator
    P = PrecisionOperator(matrix.copy(), variant_info)

    # Create input vector with different values for variants sharing index 0
    b = np.array([1.0, 2.0, 3.0], dtype=np.float32)  # Values for the three variants

    # Test variant_solve
    result = P.variant_solve(b)

    # Expected behavior:
    # 1. Values for index 0 should be summed: 1.0 + 2.0 = 3.0
    # 2. System solved with [3.0, 3.0]
    # 3. Solution for index 0 assigned back to both variants

    # Verify result has correct length (number of variants)
    assert len(result) == len(variant_info)

    # Verify variants sharing index 0 get the same value
    assert result[0] == result[1]

    # Verify the solution satisfies the original system
    # First, sum values for shared indices
    b_summed = np.array([3.0, 3.0], dtype=np.float32)

    # Then solve the system
    expected = P.solve(b_summed)[indices]

    # Finally, compare with summed result
    assert np.allclose(expected, result)

def test_precision_operator_chained_indexing():
    """Test that chained indexing operations compose correctly."""
    # Create a 4x4 precision matrix
    data = np.array([2.0, -1.0, -1.0, 2.0, -1.0, 2.0, -1.0, 2.0], dtype=np.float32)
    indices = np.array([0, 1, 0, 1, 1, 2, 2, 3])
    indptr = np.array([0, 2, 4, 6, 8])
    matrix = csc_matrix((data, indices, indptr), shape=(4, 4))

    variant_info = pl.DataFrame({
        'variant_id': ['rs1', 'rs2', 'rs3', 'rs4'],
        'position': [1, 2, 3, 4],
        'chromosome': ['1', '1', '1', '1']
    })

    P = PrecisionOperator(matrix.copy(), variant_info)

    # Test chained slicing
    P_sub1 = P[1:3]  # Should select indices [1, 2]
    P_sub2 = P_sub1[1]  # Should select index 2 from original matrix
    assert P_sub2._which_indices[0] == 2

    # Test chained indexing with lists
    P_sub3 = P[[0, 2, 3]]  # Select these indices
    P_sub4 = P_sub3[[1, 2]]  # Should select indices [2, 3] from original matrix
    np.testing.assert_array_equal(P_sub4._which_indices, [2, 3])

    # Test update through chained indexing
    P_sub4.update_element(0, 1.0)  # Should update index 2 in original matrix
    expected = matrix.toarray()
    expected[2, 2] = 3.0  # Original 2.0 + update 1.0
    np.testing.assert_array_almost_equal(P.matrix.toarray(), expected)
