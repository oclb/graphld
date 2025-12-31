import numpy as np
import polars as pl
import pytest
from score_test.score_test import VariantAnnot

def test_perturb_variant_annotations_binary():
    """Test perturbation of binary variant annotations."""
    n = 1000
    p = 0.3
    np.random.seed(42)
    vals = np.random.binomial(1, p, n)
    df = pl.DataFrame({'annot': vals, 'other': np.zeros(n)})
    annot_names = ['annot']
    
    # Perturb 100% of values
    # If we replace X with Y ~ Bernoulli(p), P(X != Y) = 2p(1-p)
    # For p=0.3, 2*0.3*0.7 = 0.42. So ~420 changed.
    
    annot = VariantAnnot(df.clone(), annot_names)
    annot.perturb(fraction=1.0, seed=42)
    df_perturbed = annot.df
    
    new_vals = df_perturbed['annot'].to_numpy()
    
    # Check changed
    n_changed = np.sum(vals != new_vals)
    expected_changed = n * 2 * p * (1-p)
    
    print(f"Changed: {n_changed}, Expected: {expected_changed}")
    assert n_changed > 0
    # Tolerance
    assert abs(n_changed - expected_changed) < n * 0.1
    
    # Check mean preserved
    assert abs(np.mean(new_vals) - p) < 0.05
    
    # Check 'other' column not changed
    assert np.all(df_perturbed['other'].to_numpy() == df['other'].to_numpy())


def test_perturb_variant_annotations_continuous():
    """Test that continuous annotations are dropped."""
    n = 100
    vals = np.random.normal(0, 1, n)
    df = pl.DataFrame({'annot': vals})
    
    # Even with fraction 1.0, continuous values should be skipped (dropped)
    annot = VariantAnnot(df.clone(), ['annot'])
    annot.perturb(fraction=1.0)
    
    # Should be empty
    assert len(annot.annot_names) == 0
    
    # Also check if we mix binary and continuous
    df = df.with_columns(pl.Series('binary', np.random.binomial(1, 0.5, n)))
    annot = VariantAnnot(df.clone(), ['annot', 'binary'])
    annot.perturb(fraction=1.0)
    
    # Only binary remains
    assert annot.annot_names == ['binary']
    
    # Binary changed (sanity check)
    # Not checking values here, assumed covered by other tests


def test_perturb_fraction():
    """Test partial perturbation."""
    n = 1000
    p = 0.5
    vals = np.random.binomial(1, p, n)
    df = pl.DataFrame({'annot': vals})
    
    fraction = 0.1
    # Expected changes: fraction * 2p(1-p)
    # 0.1 * 0.5 = 0.05 -> 50 changes
    
    annot = VariantAnnot(df.clone(), ['annot'])
    annot.perturb(fraction=fraction, seed=42)
    df_perturbed = annot.df
    new_vals = df_perturbed['annot'].to_numpy()
    
    n_changed = np.sum(vals != new_vals)
    expected = n * fraction * 2 * p * (1-p)
    
    print(f"Fraction {fraction}: Changed {n_changed}, Expected {expected}")
    assert abs(n_changed - expected) < 30


def test_perturb_nans_raise_error():
    """Test that annotations with NaNs raise ValueError."""
    n = 100
    vals = np.random.binomial(1, 0.5, n).astype(float)
    vals[0] = np.nan
    df = pl.DataFrame({'annot': vals})
    
    annot = VariantAnnot(df.clone(), ['annot'])
    
    with pytest.raises(ValueError, match="contains NaNs"):
        annot.perturb(fraction=0.1)
