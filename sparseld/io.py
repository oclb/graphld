"""
Input/output operations for LDGM data.
"""

import os
from pathlib import Path
from typing import Optional, Tuple, List
import numpy as np
import polars as pl
from scipy.sparse import csr_matrix

def load_ldgm(filepath: str, snplist_path: Optional[str] = None, population: Optional[str] = None,
              snps_only: bool = False) -> Tuple[csr_matrix, pl.DataFrame]:
    """
    Load an LDGM from a single LD block's edgelist and snplist files.

    Args:
        filepath: Path to the .edgelist file or directory containing it
        snplist_path: Optional path to .snplist file or directory. If None, uses filepath
        population: Optional population name to filter files
        snps_only: Import snplist data for SNPs only (smaller memory usage)

    Returns:
        Tuple of (sparse matrix, variant info DataFrame) for LDGM construction
    """
    # Handle directory vs file input
    filepath = Path(filepath)
    if filepath.is_dir():
        pattern = "*.edgelist"
        if population:
            pattern = f"*{population}*.edgelist"
        edgelist_files = list(filepath.glob(pattern))
        if not edgelist_files:
            raise FileNotFoundError(f"No edgelist files found in {filepath}")
        filepath = edgelist_files[0]

    # Use provided snplist path or find corresponding snplist file
    if snplist_path is None:
        snplist_path = filepath.parent
        pattern = filepath.stem.split('.')[0]  # Remove all extensions
        if pattern.endswith(f".{population}"):
            pattern = pattern[:-len(f".{population}")]
        snplist_files = list(Path(snplist_path).glob(f"{pattern}*.snplist"))
        if not snplist_files:
            raise FileNotFoundError(f"No matching snplist file found for {filepath}")
        snplist_file = snplist_files[0]
    else:
        snplist_file = Path(snplist_path)
        if not snplist_file.exists():
            raise FileNotFoundError(f"Snplist file not found: {snplist_file}")

    # Read snplist
    variant_info = pl.read_csv(snplist_file, separator=',')
    if snps_only:
        # Filter for SNPs (single character alleles)
        variant_info = variant_info.filter(
            pl.col('anc_alleles').str.contains(r'^.$') &
            pl.col('deriv_alleles').str.contains(r'^.$')
        )

    # Read edgelist and create sparse matrix
    edges = pl.read_csv(filepath, separator=',', has_header=False)

    # Get matrix dimensions
    n = max(
        variant_info.select(pl.col('index')).max().item(),
        max(
            edges.select(pl.col('column_1')).max().item(),
            edges.select(pl.col('column_2')).max().item()
        )
    ) + 1

    # Create symmetric sparse matrix
    row_indices = edges['column_1'].to_numpy()
    col_indices = edges['column_2'].to_numpy()
    values = edges['column_3'].to_numpy()

    # Add symmetric entries where needed
    mask = row_indices != col_indices
    mask_indices = np.where(mask)[0]

    # Create symmetric entries
    row_indices = np.concatenate([row_indices, col_indices[mask_indices]])
    col_indices = np.concatenate([col_indices, row_indices[mask_indices]])
    values = np.concatenate([values, values[mask_indices]])

    # Create sparse matrix
    matrix = csr_matrix((values, (row_indices, col_indices)), shape=(n, n))

    return matrix, variant_info

def merge_alleles(anc_alleles: pl.Series, deriv_alleles: pl.Series,
                  ref_alleles: pl.Series, alt_alleles: pl.Series) -> pl.Series:
    """
    Compare alleles between two sources and return phase information.
    
    Args:
        anc_alleles: Ancestral alleles from PrecisionOperator
        deriv_alleles: Derived alleles from PrecisionOperator
        ref_alleles: Reference alleles from summary statistics
        alt_alleles: Alternative alleles from summary statistics
        
    Returns:
        Series of integers indicating phase:
         1: Alleles match exactly
        -1: Alleles match but are swapped
         0: Alleles do not match
    """
    # Convert to numpy arrays for faster comparison
    anc = anc_alleles.to_numpy()
    der = deriv_alleles.to_numpy()
    ref = ref_alleles.to_numpy()
    alt = alt_alleles.to_numpy()
    
    # Make case-insensitive
    anc = np.char.lower(anc.astype(str))
    der = np.char.lower(der.astype(str))
    ref = np.char.lower(ref.astype(str))
    alt = np.char.lower(alt.astype(str))
    
    # Check matches
    exact_match = (anc == ref) & (der == alt)
    flipped_match = (anc == alt) & (der == ref)
    
    # Convert to phase
    phase = np.zeros(len(anc), dtype=np.int64)
    phase[exact_match] = 1
    phase[flipped_match] = -1
    
    return pl.Series(phase)

def merge_snplists(precision_ops: list, sumstats: pl.DataFrame, *,
                   variant_id_col: str = 'SNP',
                   ref_allele_col: str = 'A1',
                   alt_allele_col: str = 'A2',
                   match_by_position: bool = False,
                   chr_col: str = 'CHR',
                   pos_col: str = 'POS',
                   table_format: str = '') -> list[pl.DataFrame]:
    """
    Merge a list of PrecisionOperator instances with a summary statistics DataFrame.
    
    Args:
        precision_ops: List of PrecisionOperator instances
        sumstats: Summary statistics DataFrame
        variant_id_col: Column name containing variant IDs
        ref_allele_col: Column name containing reference allele
        alt_allele_col: Column name containing alternative allele
        match_by_position: Whether to match SNPs by position instead of variant ID
        chr_col: Column name containing chromosome
        pos_col: Column name containing position
        table_format: Optional file format specification (e.g., 'vcf')
        
    Returns:
        List of merged summary statistics DataFrames, one per precision operator
    """
    # Handle VCF format
    if table_format.lower() == 'vcf':
        variant_id_col = 'ID'
        ref_allele_col = 'A1'
        alt_allele_col = 'A2'
    
    # Validate inputs
    if match_by_position:
        if not all(col in sumstats.columns for col in [chr_col, pos_col]):
            raise ValueError(f"Summary statistics must contain {chr_col} and {pos_col} columns for position matching. Found columns: {', '.join(sumstats.columns)}")
    else:
        if variant_id_col not in sumstats.columns:
            raise ValueError(f"Summary statistics must contain {variant_id_col} column. Found columns: {', '.join(sumstats.columns)}")
    
    # Concatenate all variant info
    variant_infos = [op.variant_info for op in precision_ops]
    block_sizes = [len(vi) for vi in variant_infos]
    cumulative_sizes = [0] + list(np.cumsum(block_sizes))
    
    # Add block index to each variant info
    for i, vi in enumerate(variant_infos):
        vi = vi.with_columns(pl.lit(i).alias('block_index'))
        variant_infos[i] = vi
    
    all_variants = pl.concat(variant_infos)
    
    # Match variants
    if match_by_position:
        # Join on chromosome and position
        merged = all_variants.join(
            sumstats,
            left_on=['chr', 'position'],
            right_on=[chr_col, pos_col],
            how='inner'
        )
    else:
        # Join on variant ID
        merged = all_variants.join(
            sumstats,
            left_on='site_ids',
            right_on=variant_id_col,
            how='inner'
        )
    
    # Check alleles if provided
    if all(col in sumstats.columns for col in [ref_allele_col, alt_allele_col]):
        phase = merge_alleles(
            merged['anc_alleles'],
            merged['deriv_alleles'],
            merged[ref_allele_col],
            merged[alt_allele_col]
        )
        # Add phase information and filter mismatches
        merged = merged.with_columns(phase)
        merged = merged.filter(phase != 0)
    
    # Split back into blocks and update which_indices
    result = []
    for i, op in enumerate(precision_ops):
        block_variants = merged.filter(pl.col('block_index') == i)
        
        # Update which_indices in the PrecisionOperator
        op._which_indices = block_variants['index'].to_numpy()
        
        result.append(block_variants)
    
    return result
