"""Functions for reading LDSC sumstats files."""

from pathlib import Path
from typing import Union

import polars as pl

POSITIONS_FILE = 'data/rsid_position.csv'

def read_ldsc_sumstats(
    file: Union[str, Path],
    add_positions: bool = True,
    positions_file: str = POSITIONS_FILE,
    maximum_missingness: float = 1.0,
) -> pl.DataFrame:
    """Read LDSC sumstats file format.

    Args:
        file: Path to LDSC sumstats file
        add_positions: If True, merge with external file to add positions
        positions_file: File containing RSIDs and positions, defaults to data/rsid_position.csv

    Returns:
        DataFrame with columns: SNP, N, Z, A1, A2
        If add_positions=True, also includes: CHR, POS
    """
    print(f"Reading LDSC sumstats file: {file} with max missingness: {maximum_missingness}")
    # Dynamically detect columns
    with open(file, 'r') as f:
        header = f.readline().strip().split('\t')

    # Determine which columns to read
    columns_to_read = []
    if 'Z' in header:
        columns_to_read = ['SNP', 'N', 'Z', 'A1', 'A2']
    elif 'Beta' in header and 'se' in header:
        columns_to_read = ['SNP', 'N', 'Beta', 'se', 'A1', 'A2']
    else:
        # If no Z or Beta/se, try to read all columns
        columns_to_read = header

    # Read the file using polars
    df = pl.read_csv(
        file,
        separator='\t',
        columns=columns_to_read,
        infer_schema_length=10000,
    )

    # Compute Z score if needed
    if 'Beta' in df.columns and 'Z' not in df.columns:
        df = df.with_columns(
            (pl.col('Beta') / pl.col('se')).alias('Z')
        ).drop(['Beta', 'se'])

    # Validate required columns
    required_cols = {'SNP', 'N', 'Z', 'A1', 'A2'}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Rename A1, A2 to ALT, REF
    df = df.with_columns(
        pl.col('A1').alias('ALT'),
        pl.col('A2').alias('REF')
    ).drop(['A1', 'A2'])

    if add_positions:
        # Read positions file
        positions = pl.read_csv(
            positions_file,
            separator=',',
            columns=['chrom', 'site_ids', 'position']
        )

        # Merge with positions
        df = df.join(
            positions.rename({
                'chrom': 'CHR',
                'site_ids': 'SNP',
                'position': 'POS'
            }),
            on='SNP',
            how='inner'
        )

    # Drop variants with N < (1 - maximum_missingness) * max(N)
    print(f"Num variants before filtering by missingness: {len(df)}")
    df = df.filter(pl.col('N') >= (1 - maximum_missingness) * pl.col('N').max())
    print(f"Num variants after filtering by missingness: {len(df)}")


    return df
