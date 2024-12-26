"""Shared test fixtures for graphld."""

import numpy as np
import polars as pl
import pytest
from scipy.sparse import csr_matrix

from graphld import PrecisionOperator

@pytest.fixture
def small_precision_matrix():
    """Create a small 3x3 precision matrix for testing."""
    data = np.array([2.0, -1.0, -1.0, 2.0, -1.0, 2.0])
    indices = np.array([0, 1, 0, 1, 2, 2])
    indptr = np.array([0, 2, 4, 6])
    return csr_matrix((data, indices, indptr), shape=(3, 3))

@pytest.fixture
def variant_info():
    """Create a sample variant info DataFrame."""
    return pl.DataFrame({
        'variant_id': ['rs1', 'rs2', 'rs3'],
        'position': [1, 2, 3],
        'chromosome': ['1', '1', '1'],
        'allele1': ['A', 'C', 'G'],
        'allele2': ['T', 'G', 'A']
    })

@pytest.fixture
def precision_operator(small_precision_matrix, variant_info):
    """Create a PrecisionOperator instance for testing."""
    return PrecisionOperator(small_precision_matrix, variant_info)

@pytest.fixture
def random_precision_matrix():
    """Create a larger random precision matrix for performance testing."""
    n = 100
    density = 0.1
    random_matrix = np.random.rand(n, n)
    random_matrix = (random_matrix + random_matrix.T) / 2  # Make symmetric
    np.fill_diagonal(random_matrix, np.abs(random_matrix).sum(axis=0) + 1)  # Make diagonally dominant
    return csr_matrix(random_matrix * (np.random.rand(n, n) < density))
