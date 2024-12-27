"""Test simulation functionality."""

import numpy as np
import polars as pl
from scipy.sparse import csc_matrix

from graphld import PrecisionOperator, Simulate


def test_simulate_basic():
    """Test basic simulation functionality with a simple precision matrix."""
    # Create a larger precision matrix
    n = 20  # More variants for more stable results
    data = []
    indices = []
    indptr = [0]
    pos = 0

    # Create block tridiagonal matrix
    for i in range(n):
        if i > 0:
            data.append(-1.0)  # Lower diagonal
            indices.append(i-1)
            pos += 1
        data.append(2.0)  # Diagonal
        indices.append(i)
        pos += 1
        if i < n-1:
            data.append(-1.0)  # Upper diagonal
            indices.append(i+1)
            pos += 1
        indptr.append(pos)

    matrix = csc_matrix((np.array(data, dtype=np.float32),
                        np.array(indices),
                        np.array(indptr)),
                       shape=(n, n))

    # Create variant info with mock annotations
    variant_info = pl.DataFrame({
        'variant_id': [f'rs{i}' for i in range(n)],
        'position': list(range(n)),
        'chromosome': ['1'] * n,
        'af': np.linspace(0.1, 0.5, n),  # Varying allele frequencies
        'annotation1': np.random.randn(n),  # Random annotations
        'annotation2': np.random.randn(n),
        'index': list(range(n))  # Add index column
    })

    # Create precision operator
    P = PrecisionOperator(matrix, variant_info)

    # Create simulator with specific settings
    sim = Simulate(
        sample_size=1000,
        heritability=0.5,  # Total h2
        component_variance=[1.0],  # Single component
        component_weight=[0.3],    # 30% of variants are causal
        alpha_param=-1,  # No allele frequency dependence
        component_random_seed=42  # For reproducibility
    )

    # Simulate summary statistics
    sumstats = sim.simulate([P])
    assert len(sumstats) == 1  # One block

    # Check basic properties
    stats = sumstats[0]
    assert len(stats) == n  # n variants
    assert all(col in stats.columns for col in ['Z', 'N', 'beta_true'])
    assert np.all(stats['N'].to_numpy() == 1000)  # Sample size is correct

    # Check that total heritability matches specification
    total_h2 = np.sum(stats['beta_true'].to_numpy()**2)
    np.testing.assert_allclose(total_h2, 0.5, rtol=1e-5)

    # Test with annotation-dependent polygenicity
    sim = Simulate(
        sample_size=1000,
        heritability=0.5,
        component_variance=[1.0],
        component_weight=[0.3],
        annotation_dependent_polygenicity=True,
        component_random_seed=42
    )

    sumstats = sim.simulate([P])
    stats = sumstats[0]
    total_h2 = np.sum(stats['beta_true'].to_numpy()**2)
    np.testing.assert_allclose(total_h2, 0.5, rtol=1e-5)

    # Test with allele frequency dependence
    sim = Simulate(
        sample_size=1000,
        heritability=0.5,
        component_variance=[1.0],
        component_weight=[0.3],
        alpha_param=0.75,  # Strong AF dependence
        component_random_seed=42
    )

    sumstats = sim.simulate([P])
    stats = sumstats[0]
    total_h2 = np.sum(stats['beta_true'].to_numpy()**2)
    np.testing.assert_allclose(total_h2, 0.5, rtol=1e-5)


def test_simulate_multiple_components():
    """Test simulation with multiple variance components."""
    # Create a larger precision matrix
    n = 10
    data = []
    indices = []
    indptr = [0]
    pos = 0

    # Create block tridiagonal matrix
    for i in range(n):
        if i > 0:
            data.append(-1.0)  # Lower diagonal
            indices.append(i-1)
            pos += 1
        data.append(2.0)  # Diagonal
        indices.append(i)
        pos += 1
        if i < n-1:
            data.append(-1.0)  # Upper diagonal
            indices.append(i+1)
            pos += 1
        indptr.append(pos)

    matrix = csc_matrix((np.array(data, dtype=np.float32),
                        np.array(indices),
                        np.array(indptr)),
                       shape=(n, n))

    # Create variant info
    variant_info = pl.DataFrame({
        'variant_id': [f'rs{i}' for i in range(n)],
        'position': list(range(n)),
        'chromosome': ['1'] * n,
        'af': np.linspace(0.1, 0.5, n),  # Varying allele frequencies
        'annotation1': np.random.randn(n),
        'annotation2': np.random.randn(n),
        'index': list(range(n))  # Add index column
    })

    # Create precision operator
    P = PrecisionOperator(matrix, variant_info)

    # Create simulator with multiple components
    sim = Simulate(
        sample_size=1000,
        heritability=0.3,
        component_variance=[1.0, 0.1],  # Two components with different variances
        component_weight=[0.2, 0.3],    # Different proportions of causal variants
        alpha_param=-1,
        component_random_seed=42
    )

    # Simulate summary statistics
    sumstats = sim.simulate([P])
    stats = sumstats[0]

    # Check that total heritability matches specification
    total_h2 = np.sum(stats['beta_true'].to_numpy()**2)
    np.testing.assert_allclose(total_h2, 0.3, rtol=1e-5)

    # Check that we have the expected number of causal variants
    n_causal = np.sum(np.abs(stats['beta_true'].to_numpy()) > 0)
    # Allow for some random variation in the number of causal variants
    # Expected proportion is 0.5, but allow up to 0.7 to account for random sampling
    assert n_causal <= n * 0.7  # Should be roughly 50% causal, but allow some variation
