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
            'Z': [],
            'SNP': [],  # Add variant IDs
            'ZSCORE': []  # Add alternative Z score column
        }
        for row in metadata.iter_rows(named=True):
            snplist_path = os.path.join(os.path.dirname(ldgm_metadata_path), row['snplistName'])
            snplist = pl.read_csv(snplist_path, separator=',', has_header=True)
            chromosome = int(row['chrom'])
            n_variants = len(snplist)
            sumstats['CHR'].extend([chromosome] * n_variants)
            sumstats['POS'].extend(snplist['position'].to_list())
            sumstats['REF'].extend(snplist['anc_alleles'].to_list())
            sumstats['ALT'].extend(snplist['deriv_alleles'].to_list())
            # Create some "causal" variants with large Z scores
            z_scores = np.random.randn(n_variants)
            causal_idx = np.random.choice(n_variants, size=int(0.01 * n_variants), replace=False)
            z_scores[causal_idx] *= 5  # Make some variants clearly significant
            sumstats['Z'].extend(list(z_scores))
            sumstats['ZSCORE'].extend(list(z_scores))  # Same Z scores in alternative column
            # Create variant IDs that match the site_ids in the snplist
            sumstats['SNP'].extend(snplist['site_ids'].to_list())

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

    # Test default options (match by position, Z column)
    clumped = LDClumper.clump(
        metadata_path,
        sumstats,
        rsq_threshold=rsq_threshold,
        chisq_threshold=chisq_threshold,
        populations="EUR",
        run_in_serial=False
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

    # Test matching by variant ID
    clumped_by_id = LDClumper.clump(
        metadata_path,
        sumstats,
        rsq_threshold=rsq_threshold,
        chisq_threshold=chisq_threshold,
        populations="EUR",
        run_in_serial=False,
        match_by_position=False,
        variant_id_col='SNP'
    )
    
    # Results should be identical since variant IDs match positions
    assert np.array_equal(
        clumped.select('is_index').to_numpy(),
        clumped_by_id.select('is_index').to_numpy()
    ), "Results differ when matching by variant ID"

    # Test alternative Z score column
    clumped_alt_z = LDClumper.clump(
        metadata_path,
        sumstats,
        rsq_threshold=rsq_threshold,
        chisq_threshold=chisq_threshold,
        populations="EUR",
        run_in_serial=False,
        z_col='ZSCORE'
    )
    
    # Results should be identical since Z scores are the same
    assert np.array_equal(
        clumped.select('is_index').to_numpy(),
        clumped_alt_z.select('is_index').to_numpy()
    ), "Results differ with alternative Z score column"