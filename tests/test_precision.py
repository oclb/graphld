"""Tests for the PrecisionOperator class."""

import numpy as np
import polars as pl
from scipy.sparse import csr_matrix
import pytest

from graphld.precision import PrecisionOperator

def test_precision_operator_basic():
    """Test basic functionality of PrecisionOperator."""
    # Create a simple 3x3 precision matrix
    data = np.array([2.0, -1.0, -1.0, 2.0, -1.0, 2.0])
    indices = np.array([0, 1, 0, 1, 2, 2])
    indptr = np.array([0, 2, 4, 6])
    matrix = csr_matrix((data, indices, indptr), shape=(3, 3))
    
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
    matrix = csr_matrix((data, indices, indptr), shape=(4, 4))
    
    variant_info = pl.DataFrame({
        'variant_id': ['rs1', 'rs2', 'rs3', 'rs4'],
        'position': [1, 2, 3, 4],
        'chromosome': ['1', '1', '1', '1']
    })
    
    P = PrecisionOperator(matrix.copy(), variant_info)
    
    # Test integer indexing
    P_sub = P[1]
    assert P_sub.shape == (1, 1)
    
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
    matrix = csr_matrix((data, indices, indptr), shape=(3, 3))
    
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
    
    # Test PCG solver
    x_pcg = P.solve(b, method='pcg')
    residual = np.linalg.norm(P @ x_pcg - b)
    assert residual < 1e-5  # PCG tolerance is typically larger
    
    # Test multiple right-hand sides
    B = np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]], dtype=np.float32)
    X = P.solve(B, method='direct')
    assert X.shape == (3, 2)
    residual = np.linalg.norm(P @ X - B)
    assert residual < 1e-5
    
    # Function to count iterations
    def make_callback():
        iterations = [0]
        def callback(xk):
            iterations[0] += 1
        return callback, iterations
    
    # Test with different perturbation sizes
    perturbations = [0.1, 1.0, 10.0]  # Small, medium, large perturbations
    for perturbation in perturbations:
        # Create perturbed matrix
        perturbed_data = data.copy()
        perturbed_data[0] += perturbation
        perturbed_matrix = csr_matrix((perturbed_data, indices, indptr), shape=(3, 3))
        
        # Create new operator with perturbed matrix
        P_perturbed = PrecisionOperator(perturbed_matrix, variant_info)
        
        # Test PCG solver with perturbed matrix
        callback, iterations = make_callback()
        x_pcg_perturbed = P_perturbed.solve(b, method='pcg', callback=callback)
        residual = np.linalg.norm(P_perturbed @ x_pcg_perturbed - b)
        assert residual < 1e-5  # Should still converge, might take more iterations

def test_precision_operator_update():
    """Test updating the precision matrix."""
    # Create a simple positive definite matrix
    data = np.array([2.0, -1.0, -1.0, 2.0], dtype=np.float32)
    indices = np.array([0, 1, 0, 1])
    indptr = np.array([0, 2, 4])
    matrix = csr_matrix((data, indices, indptr), shape=(2, 2))
    
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
    matrix = csr_matrix((data, indices, indptr), shape=(4, 4))
    
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

def test_precision_operator_update_element():
    """Test updating a single diagonal element."""
    # Create a 3x3 precision matrix
    data = np.array([2.0, -1.0, -1.0, 2.0, -1.0, 2.0], dtype=np.float32)
    indices = np.array([0, 1, 0, 1, 2, 2])
    indptr = np.array([0, 2, 4, 6])
    matrix = csr_matrix((data, indices, indptr), shape=(3, 3))
    
    variant_info = pl.DataFrame({
        'variant_id': ['rs1', 'rs2', 'rs3'],
        'position': [1, 2, 3],
        'chromosome': ['1', '1', '1']
    })
    
    P = PrecisionOperator(matrix.copy(), variant_info)
    
    # Test rank-one update (positive value)
    print("\ntest_precision_operator_update_element:")
    print("Original matrix:", matrix.toarray())
    P.update_element(1, 1.0)
    print("After update:", P.matrix.toarray())
    expected = matrix.toarray()
    expected[1, 1] = 3.0  # Original 2.0 + update 1.0
    print("Expected:", expected)
    print("Diagonal indices:", P.diagonal_indices)
    np.testing.assert_array_almost_equal(P.matrix.toarray(), expected)
    
    # Verify Cholesky needs to be recomputed
    assert not P._cholesky_is_up_to_date
    
    # Test rank-one downdate (negative value)
    P.update_element(0, -0.5)
    expected[0, 0] = 1.5  # Original 2.0 - update 0.5
    print("After update:", P.matrix.toarray())
    print("Expected:", expected)
    print("Diagonal indices:", P.diagonal_indices)
    np.testing.assert_array_almost_equal(P.matrix.toarray(), expected)
    
    # Test with indices
    P_sub = P[1:]  # 2x2 submatrix
    P_sub.update_element(0, 1.0)  # Updates global index 1
    expected[1, 1] = 4.0  # Original 2.0 + first update 1.0 + second update 1.0
    print("After update:", P.matrix.toarray())
    print("Expected:", expected)
    print("Diagonal indices:", P_sub.diagonal_indices)
    np.testing.assert_array_almost_equal(P.matrix.toarray(), expected)
    
    # Test with mask
    mask = np.array([False, True, True])
    P_sub = P[mask]  # Another 2x2 submatrix
    P_sub.update_element(1, 2.0)  # Updates global index 2
    expected[2, 2] = 3.0  # Original 1.0 + update 2.0
    print("After update:", P.matrix.toarray())
    print("Expected:", expected)
    print("Diagonal indices:", P_sub.diagonal_indices)
    np.testing.assert_array_almost_equal(P.matrix.toarray(), expected)
    
    # Test error on out of bounds index
    with pytest.raises(IndexError):
        P.update_element(3, 1.0)
        
    # Test error on out of bounds index with indices
    with pytest.raises(IndexError):
        P_sub = P[1:]
        P_sub.update_element(2, 1.0)

def test_precision_operator_errors():
    """Test error handling in PrecisionOperator."""
    # Create a simple matrix
    data = np.array([2.0, -1.0, -1.0, 2.0])
    indices = np.array([0, 1, 0, 1])
    indptr = np.array([0, 2, 4])
    matrix = csr_matrix((data, indices, indptr), shape=(2, 2))
    
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
    with pytest.raises(ValueError):
        P[0, 1]  # Only single-axis indexing is supported

def test_precision_operator_inverse_diagonal_methods():
    """Test different methods for computing inverse diagonal elements."""
    # Create a small test matrix to make exact computation feasible
    n = 200
    num_samples = 50
    rng = np.random.RandomState(42)
    
    # Create a random sparse positive definite matrix
    A = rng.randn(n, n)
    A = A @ A.T + np.arange(1,n+1) * np.eye(n)  # Make it positive definite
    A = csr_matrix(A)
    
    # Create variant info DataFrame
    variant_info = pl.DataFrame({
        "chrom": ["1"] * n,
        "pos": range(n),
        "ref": ["A"] * n,
        "alt": ["T"] * n,
    })
    
    # Create PrecisionOperator
    P = PrecisionOperator(A, variant_info)
    
    # Compute diagonal using all three methods
    exact_diag = P.inverse_diagonal(method="exact")
    hutch_diag = P.inverse_diagonal(method="hutchinson", n_samples=num_samples, seed=42)
    # xnys_diag = P.inverse_diagonal(method="xnys", n_samples=num_samples, seed=42)
    
    # Compare trace estimates - they should be approximately equal
    np.testing.assert_allclose(np.sum(hutch_diag), np.sum(exact_diag), rtol=0.01)
    # np.testing.assert_allclose(xnys_diag, exact_diag, rtol=0.01)
    
    # Test with initialization
    v = rng.choice([-1, 1], size=(n, num_samples))
    pv = v.copy()
    
    # Test hutchinson with initialization
    hutch_diag_init, hutch_y = P.inverse_diagonal(
        initialization=(v, pv),
        method="hutchinson"
    )
    
    np.testing.assert_allclose(np.sum(hutch_diag_init), np.sum(exact_diag), rtol=0.01)

    # # Test xnys with initialization
    # xnys_diag_init, xnys_y = P.inverse_diagonal(
    #     initialization=(v, pv),
    #     method="xnys"
    # )
    
    # Verify shapes of returned values
    assert hutch_y.shape == v.shape
    # assert xnys_y.shape == v.shape
    
    
    # Test error cases
    with pytest.raises(ValueError):
        P.inverse_diagonal(method="invalid")
        
    with pytest.raises(ValueError):
        P.inverse_diagonal(initialization=(v, pv), method="exact")
