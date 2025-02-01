"""Tests for heritability estimation."""

import numpy as np
import pytest
from typing import Optional, Union, List
from graphld import load_ldgm, Simulate, read_ldgm_metadata
from graphld.heritability import run_graphREML, _get_softmax_link_function, ModelOptions, MethodOptions
import polars as pl
import os

def test_run_graphREML(metadata_path, create_annotations, create_sumstats):
    """Test heritability estimation with simulated data."""
    sumstats = create_sumstats(str(metadata_path), 'EUR')
    annotations = create_annotations(metadata_path, 'EUR')

    # Run GraphREML
    model = ModelOptions(params=np.zeros((1,1)),
                sample_size=1000
    )
    method = MethodOptions(
        match_by_position=True
    )
    result = run_graphREML(
        model_options=model,
        method_options=method,
        summary_stats=sumstats,
        annotation_data=annotations,
        ldgm_metadata_path=metadata_path
    )

    assert result is not None


def test_softmax_link_function():
    """Test softmax link function against MATLAB output."""
    link_fn, link_fn_grad = _get_softmax_link_function(n_snps=10)
    
    # Test case from MATLAB: linkFnGrad((1:3),(2:4)')
    annot = np.array([1, 2, 3])
    theta = np.array([[2], [3], [4]])
    grad = link_fn_grad(annot, theta)
    assert grad.shape == (3,)
    assert np.allclose(grad, [0.1, 0.2, 0.3])

    # Test case from MATLAB: linkFn((1:3),(2:4)')
    val = link_fn(annot, theta)
    assert np.allclose(val, 2.0)
