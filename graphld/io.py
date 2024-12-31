"""
Input/output operations for LDGM data.
"""

from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import polars as pl
from scipy.sparse import csc_matrix

from .precision import PrecisionOperator


def load_ldgm(filepath: str, snplist_path: Optional[str] = None, population: Optional[str] = "EUR",
              snps_only: bool = False) -> Union[PrecisionOperator, List[PrecisionOperator]]:
    """
    Load an LDGM from a single LD block's edgelist and snplist files.

    Args:
        filepath: Path to the .edgelist file or directory containing it
        snplist_path: Optional path to .snplist file or directory. If None, uses filepath
        population: Optional population name to filter files and set allele frequency column. Defaults to "EUR"
        snps_only: Import snplist data for SNPs only (smaller memory usage)

    Returns:
        If filepath is a directory:
            List of PrecisionOperator instances, one for each edgelist file
        If filepath is a file:
            Single PrecisionOperator instance with loaded precision matrix and variant info
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

        # Load each file and return a list of PrecisionOperators
        operators = []
        for edgelist_file in edgelist_files:
            operator = load_ldgm(edgelist_file, snplist_path, population, snps_only)
            operators.append(operator)
        return operators

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

    # Load edgelist data
    edgelist = pl.read_csv(filepath, separator=',', has_header=False,
                          new_columns=['i', 'j', 'value'])

    # Create sparse matrix
    matrix = csc_matrix(
        (edgelist['value'].to_numpy(),
         (edgelist['i'].to_numpy(), edgelist['j'].to_numpy()))
    )

    # Make matrix symmetric
    matrix_t = matrix.T
    diag_vals = matrix.diagonal().copy()
    matrix = matrix + matrix_t
    matrix.setdiag(diag_vals, k=0)

    # Verify diagonal values
    assert np.allclose(matrix.diagonal(), diag_vals), "Diagonal values not set correctly"

    # Create mask for rows/cols with nonzeros on diagonal
    diag = matrix.diagonal()
    nonzero_mask = diag != 0
    n_nonzero = np.sum(nonzero_mask)

    # Create mapping from old indices to new indices
    rows = np.full(len(diag), -1)
    rows[nonzero_mask] = np.arange(n_nonzero)

    # Load and process variant info
    variant_info = pl.read_csv(snplist_file, separator=',')

    # If population is specified and exists as a column, rename it to 'af'
    if population and population in variant_info.columns:
        variant_info = variant_info.rename({population: 'af'})
    elif 'af' not in variant_info.columns:
        available_cols = ", ".join(variant_info.columns)
        raise ValueError(
            f"Neither 'af' column nor '{population}' column found in snplist. "
            f"Available columns: {available_cols}"
        )

    # Store original indices and update with new mapping
    variant_info = variant_info.with_columns([
        pl.col('index').alias('original_index'),
        pl.col('index').map_elements(lambda x: rows[x], return_dtype=pl.Int64).alias('index')
    ])

    # Filter out variants with no corresponding matrix row
    variant_info = variant_info.filter(pl.col('index') >= 0)

    # Subset matrix to rows/cols with nonzero diagonal
    matrix = matrix[nonzero_mask][:, nonzero_mask]

    return PrecisionOperator(matrix, variant_info)


def merge_alleles(anc_alleles: pl.Series, deriv_alleles: pl.Series,
                  ref_alleles: pl.Series, alt_alleles: pl.Series) -> pl.Series:
    """Compare alleles between two sources and return phase information.

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
    """Merge PrecisionOperator instances with summary statistics DataFrame.

    Args:
        precision_ops: List of PrecisionOperator instances
        sumstats: Summary statistics DataFrame
        variant_id_col: Column name containing variant IDs
        ref_allele_col: Column name containing reference allele
        alt_allele_col: Column name containing alternative allele
        match_by_position: Whether to match SNPs by position instead of ID
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
        cols = [chr_col, pos_col]
        if not all(col in sumstats.columns for col in cols):
            msg = (f"Summary statistics must contain {chr_col} and {pos_col} columns "
                  f"for position matching. Found columns: {', '.join(sumstats.columns)}")
            raise ValueError(msg)
    else:
        if variant_id_col not in sumstats.columns:
            msg = (f"Summary statistics must contain {variant_id_col} column. "
                  f"Found columns: {', '.join(sumstats.columns)}")
            raise ValueError(msg)

    # Concatenate all variant info
    variant_infos = [op.variant_info for op in precision_ops]

    # Add block index to each variant info
    for i, vi in enumerate(variant_infos):
        vi = vi.with_columns(pl.lit(i).alias('block_index'))
        variant_infos[i] = vi

    all_variants = pl.concat(variant_infos)

    # Match variants
    if match_by_position:
        merged = all_variants.join(
            sumstats,
            left_on=['chr', 'position'],
            right_on=[chr_col, pos_col],
            how='inner'
        )
    else:
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
        merged = merged.with_columns(phase)
        merged = merged.filter(phase != 0)

    # Split back into blocks and update which_indices
    result = []
    for i, op in enumerate(precision_ops):
        block_variants = merged.filter(pl.col('block_index') == i)
        op._which_indices = block_variants['index'].to_numpy()
        result.append(block_variants)

    return result


def partition_variants(
    ldgm_metadata: pl.DataFrame,
    variant_data: pl.DataFrame,
    *,
    chrom_col: Optional[str] = None,
    pos_col: Optional[str] = None
) -> List[pl.DataFrame]:
    """Partition variant data according to LDGM blocks.
    
    Args:
        ldgm_metadata: DataFrame from read_ldgm_metadata containing block info
        variant_data: DataFrame containing variant information
        chrom_col: Optional name of chromosome column. If None, tries common names
        pos_col: Optional name of position column. If None, tries common names
        
    Returns:
        List of DataFrames, one per row in ldgm_metadata, containing variants
        that fall within each block's coordinates
        
    Raises:
        ValueError: If chromosome or position columns cannot be found
    """
    # Find chromosome column
    chrom_options = ['chrom', 'chromosome', 'CHR']
    if chrom_col is not None:
        chrom_options.insert(0, chrom_col)
    chrom_col = next((col for col in chrom_options if col in variant_data.columns), None)
    if chrom_col is None:
        raise ValueError(
            f"Could not find chromosome column. Tried: {', '.join(chrom_options)}"
        )
        
    # Find position column
    pos_options = ['position', 'POS']
    if pos_col is not None:
        pos_options.insert(0, pos_col)
    pos_col = next((col for col in pos_options if col in variant_data.columns), None)
    if pos_col is None:
        raise ValueError(
            f"Could not find position column. Tried: {', '.join(pos_options)}"
        )
        
    # Convert chromosome column to integer if needed
    if variant_data[chrom_col].dtype != pl.Int64:
        variant_data = variant_data.with_columns(
            pl.col(chrom_col).cast(pl.Int64).alias(chrom_col)
        )
        
    # Create list to store partitioned data
    partitioned = []
    
    # Process each block
    for block in ldgm_metadata.iter_rows(named=True):
        # Filter variants for this block
        block_variants = variant_data.filter(
            (pl.col(chrom_col) == block['chrom']) &
            (pl.col(pos_col) >= block['chromStart']) &
            (pl.col(pos_col) < block['chromEnd'])
        )
        partitioned.append(block_variants)
        
    return partitioned


def create_ldgm_metadata(directory: Union[str, Path], output_file: Optional[str] = None) -> pl.DataFrame:
    """Create metadata file for LDGM files in a directory.
    
    Args:
        directory: Directory containing .snplist and .edgelist files
        output_file: Optional path to write CSV file. If None, only returns DataFrame
        
    Returns:
        Polars DataFrame containing metadata for each LDGM file
    """
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    # Find all edgelist files
    edgelist_files = list(directory.glob("*.edgelist"))
    if not edgelist_files:
        raise FileNotFoundError(f"No .edgelist files found in {directory}")
    
    # Process each file
    data = []
    for edgefile in edgelist_files:
        # Parse filename to get info
        name = edgefile.name
        parts = name.split('_')  # e.g. 1kg_chr1_2888443_4320284.EUR.edgelist
        if len(parts) < 4:
            print(f"Skipping {name}: unexpected filename format")
            continue
            
        # Get chromosome and positions
        try:
            chrom = int(parts[1].replace('chr', ''))
            chromStart = int(parts[2])
            chromEnd = int(parts[3].split('.')[0])  # Remove population/extension
        except (ValueError, IndexError):
            print(f"Skipping {name}: could not parse chromosome/position")
            continue
            
        # Get population
        try:
            population = name.split('.')[-2]  # Second to last part
        except IndexError:
            print(f"Skipping {name}: could not parse population")
            continue
            
        # Find corresponding snplist file
        base_name = name.split('.')[0]  # Remove population and extension
        snplist_files = list(directory.glob(f"{base_name}.snplist"))
        if not snplist_files:
            print(f"Skipping {name}: no matching .snplist file")
            continue
        snplist_name = snplist_files[0].name
        
        # Count variants in snplist
        try:
            snplist_df = pl.read_csv(snplist_files[0])
            num_variants = len(snplist_df)
        except Exception as e:
            print(f"Skipping {name}: error reading snplist: {e}")
            continue
            
        # Count entries in edgelist
        try:
            edgelist_df = pl.read_csv(edgefile, has_header=False, 
                                    new_columns=['i', 'j', 'value'])
            
            # Count unique diagonal indices
            diag_mask = edgelist_df['i'] == edgelist_df['j']
            num_indices = len(edgelist_df.filter(diag_mask)['i'].unique())
            
            # Total number of entries
            num_entries = len(edgelist_df)
            
        except Exception as e:
            print(f"Skipping {name}: error reading edgelist: {e}")
            continue
            
        # Add row to metadata
        data.append({
            'chrom': chrom,
            'chromStart': chromStart,
            'chromEnd': chromEnd,
            'name': name,
            'snplistName': snplist_name,
            'population': population,
            'numVariants': num_variants,
            'numIndices': num_indices,
            'numEntries': num_entries,
            'info': ''
        })
    
    # Create DataFrame
    if not data:
        raise ValueError("No valid LDGM files found")
        
    df = pl.DataFrame(data)
    
    # Sort by chromosome and start position
    df = df.sort(['chrom', 'chromStart'])
    
    # Write to file if requested
    if output_file:
        df.write_csv(output_file)
    
    return df


def read_ldgm_metadata(
    filepath: Union[str, Path],
    *,
    populations: Optional[Union[str, List[str]]] = None,
    chromosomes: Optional[Union[int, List[int]]] = None,
    max_blocks: Optional[int] = None
) -> pl.DataFrame:
    """Read LDGM metadata from CSV file.
    
    Args:
        filepath: Path to metadata CSV file
        populations: Optional population(s) to filter by
        chromosomes: Optional chromosome(s) to filter by
        max_blocks: Optional maximum number of blocks to return
        
    Returns:
        Polars DataFrame containing LDGM metadata, filtered by population and chromosome
        if specified, and limited to max_blocks if specified
    """
    try:
        df = pl.read_csv(filepath)
        required_cols = [
            'chrom', 'chromStart', 'chromEnd', 'name', 'snplistName',
            'population', 'numVariants', 'numIndices', 'numEntries', 'info'
        ]
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(missing)}")
            
        # Filter by population if specified
        if populations is not None:
            if isinstance(populations, str):
                populations = [populations]
            df = df.filter(pl.col('population').is_in(populations))
            if len(df) == 0:
                raise FileNotFoundError(f"No LDGM files found in metadata for populations: {populations}")
                
        # Filter by chromosome if specified
        if chromosomes is not None:
            if isinstance(chromosomes, int):
                chromosomes = [chromosomes]
            df = df.filter(pl.col('chrom').is_in(chromosomes))
            if len(df) == 0:
                raise ValueError(f"No blocks found for chromosomes: {chromosomes}")
                
        # Sort by chromosome and position
        df = df.sort(['chrom', 'chromStart'])
        
        # Limit number of blocks if specified
        if max_blocks is not None:
            df = df.head(max_blocks)
            
        return df
        
    except Exception as e:
        raise ValueError(f"Error reading metadata file: {e}")
