"""Functions for reading parquet summary statistics files.

This module supports the parquet format produced by linear_dag's GWAS pipeline,
which stores summary statistics with columns like {trait}_BETA and {trait}_SE.
Multiple traits can be stored in a single file.
"""

from pathlib import Path
from typing import Optional, Union

import polars as pl


def get_parquet_traits(file: Union[str, Path]) -> list[str]:
    """Get list of trait names from a parquet summary statistics file.
    
    Args:
        file: Path to parquet file
        
    Returns:
        List of trait names found in the file
    """
    schema = pl.read_parquet_schema(file)
    traits = set()
    for col in schema:
        if col.endswith('_BETA'):
            traits.add(col[:-5])  # Remove '_BETA' suffix
        elif col.endswith('_SE'):
            traits.add(col[:-3])  # Remove '_SE' suffix
    return sorted(traits)


def read_parquet_sumstats(
    file: Union[str, Path],
    trait: Optional[str] = None,
    maximum_missingness: float = 1.0,
) -> pl.DataFrame:
    """Read parquet summary statistics file and return a single-trait DataFrame.
    
    The parquet format stores columns as {trait}_BETA and {trait}_SE for each trait.
    This function extracts a single trait and converts to a standard format with
    columns: SNP, CHR, POS, REF, ALT, N, Z.
    
    Args:
        file: Path to parquet file
        trait: Name of trait to extract. If None, uses the first trait found.
        maximum_missingness: Maximum fraction of missing samples allowed (based on N column if present)
        
    Returns:
        DataFrame with columns: SNP, CHR, POS, REF, ALT, N, Z
        
    Raises:
        ValueError: If the specified trait is not found in the file
    """
    # Get available traits
    available_traits = get_parquet_traits(file)
    if not available_traits:
        raise ValueError(f"No traits found in parquet file {file}. "
                        "Expected columns ending in _BETA and _SE.")

    # Determine which trait to use
    if trait is None:
        trait = available_traits[0]
    elif trait not in available_traits:
        raise ValueError(f"Trait '{trait}' not found in parquet file. "
                        f"Available traits: {available_traits}")

    # Read only necessary columns
    beta_col = f"{trait}_BETA"
    se_col = f"{trait}_SE"

    # Read the schema to find variant info columns
    schema = pl.read_parquet_schema(file)

    # Standard variant info column mappings
    variant_cols = []
    col_renames = {}

    # Check for various possible column names
    if 'site_ids' in schema:
        variant_cols.append('site_ids')
        col_renames['site_ids'] = 'SNP'
    elif 'SNP' in schema:
        variant_cols.append('SNP')
    elif 'rsid' in schema:
        variant_cols.append('rsid')
        col_renames['rsid'] = 'SNP'

    if 'chrom' in schema:
        variant_cols.append('chrom')
        col_renames['chrom'] = 'CHR'
    elif 'CHR' in schema:
        variant_cols.append('CHR')

    if 'position' in schema:
        variant_cols.append('position')
        col_renames['position'] = 'POS'
    elif 'POS' in schema:
        variant_cols.append('POS')
    elif 'BP' in schema:
        variant_cols.append('BP')
        col_renames['BP'] = 'POS'

    if 'ref' in schema:
        variant_cols.append('ref')
        col_renames['ref'] = 'REF'
    elif 'REF' in schema:
        variant_cols.append('REF')
    elif 'A2' in schema:
        variant_cols.append('A2')
        col_renames['A2'] = 'REF'

    if 'alt' in schema:
        variant_cols.append('alt')
        col_renames['alt'] = 'ALT'
    elif 'ALT' in schema:
        variant_cols.append('ALT')
    elif 'A1' in schema:
        variant_cols.append('A1')
        col_renames['A1'] = 'ALT'

    # Check for sample size column
    has_n = False
    if 'N' in schema:
        variant_cols.append('N')
        has_n = True
    elif 'n' in schema:
        variant_cols.append('n')
        col_renames['n'] = 'N'
        has_n = True

    # Columns to read
    columns_to_read = variant_cols + [beta_col, se_col]

    # Read the parquet file
    df = pl.read_parquet(file, columns=columns_to_read)

    # Rename columns
    if col_renames:
        df = df.rename(col_renames)

    # Compute Z score from BETA and SE
    df = df.with_columns(
        (pl.col(beta_col) / pl.col(se_col)).cast(pl.Float64).alias('Z')
    )

    # Drop the trait-specific columns, keeping just Z
    df = df.drop([beta_col, se_col])

    # Filter out rows with missing or invalid Z scores
    df = df.filter(pl.col('Z').is_finite())

    # Apply missingness filter if N column is present
    if has_n and maximum_missingness < 1.0:
        max_n = df['N'].max()
        min_n = (1 - maximum_missingness) * max_n
        df = df.filter(pl.col('N') >= min_n)

    # If no N column, add a placeholder
    if 'N' not in df.columns:
        df = df.with_columns(pl.lit(None).cast(pl.Float64).alias('N'))

    return df


def read_parquet_sumstats_multi(
    file: Union[str, Path],
    traits: Optional[list[str]] = None,
    maximum_missingness: float = 1.0,
) -> dict[str, pl.DataFrame]:
    """Read multiple traits from a parquet summary statistics file.
    
    Args:
        file: Path to parquet file
        traits: List of trait names to extract. If None, extracts all traits.
        maximum_missingness: Maximum fraction of missing samples allowed
        
    Returns:
        Dictionary mapping trait names to DataFrames
        
    Raises:
        ValueError: If any specified trait is not found in the file
    """
    available_traits = get_parquet_traits(file)

    if traits is None:
        traits = available_traits
    else:
        # Validate all requested traits exist
        missing = set(traits) - set(available_traits)
        if missing:
            raise ValueError(f"Traits not found in parquet file: {missing}. "
                           f"Available traits: {available_traits}")

    result = {}
    for trait in traits:
        result[trait] = read_parquet_sumstats(file, trait=trait,
                                               maximum_missingness=maximum_missingness)

    return result
