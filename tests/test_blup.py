"""Test simulation functionality."""

import os
import numpy as np
import polars as pl
import pytest

from graphld import Simulate, read_ldgm_metadata, BLUP
from typing import Optional, Union, List

def test_blup():
    """Test simulation with variant annotations."""
    # Create simulator with specific settings
    sim = Simulate(
        sample_size=100_000,
        heritability=0.5,
        component_variance=[1.0],
        component_weight=[0.3],
        alpha_param=-1,
        random_seed=42
    )

    def create_sumstats(ldgm_metadata_path: str, populations: Optional[Union[str, List[str]]]) -> pl.DataFrame:
        metadata: pl.DataFrame = read_ldgm_metadata(ldgm_metadata_path, populations=populations)
        sumstats = {
            'CHR': [],
            'POS': [],
            'REF': [],
            'ALT': [],
            'Z': []
        }
        for row in metadata.iter_rows(named=True):
            snplist_path = os.path.join(os.path.dirname(ldgm_metadata_path), row['snplistName'])
            snplist = pl.read_csv(snplist_path, separator=',', has_header=True)
            chromosome = int(row['chrom'])
            sumstats['CHR'].extend([chromosome] * len(snplist))
            sumstats['POS'].extend(snplist['position'].to_list())
            sumstats['REF'].extend(snplist['anc_alleles'].to_list())
            sumstats['ALT'].extend(snplist['deriv_alleles'].to_list())
            sumstats['Z'].extend(list(np.random.randn(len(snplist))))

        return pl.DataFrame(sumstats)

    # Set random seed for reproducibility
    np.random.seed(42)

    # Create annotations from metadata
    metadata_path = "data/test/metadata.csv"
    sumstats = create_sumstats(metadata_path, populations="EUR")

    assert isinstance(sumstats, pl.DataFrame)
    print(sumstats.head())

    sigmasq = 0.01
    sample_size = 10_000
    blup = BLUP.compute_blup(metadata_path, 
                            sumstats, 
                            sigmasq, 
                            sample_size, 
                            populations="EUR",
                            run_in_serial=False,
                            match_by_position=True
                            )
    
    assert(len(blup) == np.sum(sumstats.select(pl.col('POS').is_first_distinct()).to_numpy()))