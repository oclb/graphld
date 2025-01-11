"""Shared test fixtures for graphld."""

import numpy as np
import polars as pl
import pytest
from scipy.sparse import csr_matrix
from pathlib import Path
import os
from typing import Optional, Union, List

from graphld import PrecisionOperator
from graphld.io import read_ldgm_metadata


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
    np.fill_diagonal(random_matrix, np.abs(random_matrix).sum(axis=0) + 1)
    return csr_matrix(random_matrix * (np.random.rand(n, n) < density))

@pytest.fixture
def test_data_dir():
    """Get test data directory."""
    return Path("data/test")

@pytest.fixture
def metadata_path(test_data_dir):
    """Fixture for the metadata path."""
    return test_data_dir / "metadata.csv"

@pytest.fixture
def test_annotations_path(test_data_dir):
    """Fixture for the test annotations path."""
    return test_data_dir / "baselineLD.22.annot"

@pytest.fixture
def create_annotations():
    """Factory fixture for creating annotations."""
    def _create_annotations(
        metadata_path: Path, 
        populations: Optional[Union[str, List[str]]] = None
    ) -> pl.DataFrame:
        """Create annotations for testing."""
        metadata: pl.DataFrame = read_ldgm_metadata(str(metadata_path), populations=populations)
        annotations = {
            'SNP': [],
            'CHR': [],
            'POS': [],
            'A2': [],
            'A1': [],
            'base': []
        }
        for row in metadata.iter_rows(named=True):
            snplist_path = os.path.join(os.path.dirname(metadata_path), row['snplistName'])
            snplist = pl.read_csv(snplist_path, separator=',', has_header=True)
            chromosome = int(row['chrom'])
            annotations['CHR'].extend([chromosome] * len(snplist))
            annotations['SNP'].extend(snplist['site_ids'].to_list())
            annotations['POS'].extend(snplist['position'].to_list())
            annotations['A2'].extend(snplist['anc_alleles'].to_list())
            annotations['A1'].extend(snplist['deriv_alleles'].to_list())
            annotations['base'].extend([1] * len(snplist))

        return pl.DataFrame(annotations)
    
    return _create_annotations

@pytest.fixture
def create_sumstats():
    """Factory fixture for creating summary statistics."""
    def _create_sumstats(
        ldgm_metadata_path: str, 
        populations: Optional[Union[str, List[str]]]
    ) -> pl.DataFrame:
        """Create summary statistics for testing."""
        metadata: pl.DataFrame = read_ldgm_metadata(ldgm_metadata_path, populations=populations)
        sumstats = {
            'SNP': [],
            'CHR': [],
            'POS': [],
            'A1': [],
            'A2': [],
            'Z': [],
        }
        for row in metadata.iter_rows(named=True):
            snplist_path = os.path.join(os.path.dirname(ldgm_metadata_path), row['snplistName'])
            snplist = pl.read_csv(snplist_path, separator=',', has_header=True)
            chromosome = int(row['chrom'])
            
            sumstats['CHR'].extend([chromosome] * len(snplist))
            sumstats['SNP'].extend(snplist['site_ids'].to_list())
            sumstats['POS'].extend(snplist['position'].to_list())
            sumstats['A1'].extend(snplist['deriv_alleles'].to_list())
            sumstats['A2'].extend(snplist['anc_alleles'].to_list())
            
            # Generate random Z-scores and beta values for testing
            np.random.seed(42)
            sumstats['Z'].extend(np.random.normal(0, 1, len(snplist)))

        return pl.DataFrame(sumstats)
    
    return _create_sumstats
