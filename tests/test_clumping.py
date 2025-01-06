"""Test LD clumping functionality."""

import os
import numpy as np
import polars as pl
import pytest

from graphld import read_ldgm_metadata, LDClumper, BLUP

def test_clumping():
    """Test LD clumping with simulated summary statistics."""
    def create_sumstats(ldgm_metadata_path: str, populations: str) -> pl.DataFrame:
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
            # Create some "causal" variants with large Z scores
            z_scores = np.random.randn(len(snplist))
            causal_idx = np.random.choice(len(snplist), size=int(0.01 * len(snplist)), replace=False)
            z_scores[causal_idx] *= 5  # Make some variants clearly significant
            sumstats['Z'].extend(list(z_scores))

        return pl.DataFrame(sumstats)

    # Set random seed for reproducibility
    np.random.seed(42)

    # Create sumstats from metadata
    metadata_path = "data/test/metadata.csv"
    sumstats = create_sumstats(metadata_path, populations="EUR")

    assert isinstance(sumstats, pl.DataFrame)
    print(sumstats.head())

    # Run clumping with reasonable thresholds
    rsq_threshold = 0.2  # Relatively stringent LD threshold
    chisq_threshold = 5.0  # Roughly p < 2.5e-2
    
    # sigmasq = 0.01
    # sample_size = 10_000
    # blup = BLUP.compute_blup(metadata_path, 
    #                         sumstats, 
    #                         sigmasq, 
    #                         sample_size, 
    #                         populations="EUR",
    #                         run_in_serial=False,
    #                         match_by_position=True
    #                         )

    clumped = LDClumper.clump(
        metadata_path, 
        sumstats, 
        rsq_threshold=rsq_threshold,
        chisq_threshold=chisq_threshold,
        populations="EUR",
        run_in_serial=True
    )
    
    # Basic sanity checks
    assert(len(clumped) == np.sum(sumstats.select(pl.col('POS').is_first_distinct()).to_numpy()))
    assert 'is_index' in clumped.columns
    assert clumped.select('is_index').to_numpy().dtype == bool
    
    # Check that we found some index variants
    n_index = clumped.select(pl.col('is_index')).sum().item()
    assert n_index > 0, "No index variants found"
    
    # Check that index variants have high chi-square values
    index_variants = clumped.filter(pl.col('is_index'))
    chisq = index_variants.select(pl.col('Z')**2).to_numpy()
    assert np.all(chisq >= chisq_threshold), "Found index variant below chi-square threshold"
    