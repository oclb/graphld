"""
Input/output operations for LDGM data.
"""

from pathlib import Path
from typing import List, Optional, Union, Tuple
import numpy as np
import polars as pl


def load_ldgm(filepath: str, snplist_path: Optional[str] = None, population: Optional[str] = "EUR",
              snps_only: bool = False) -> Union["PrecisionOperator", List["PrecisionOperator"]]:
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
    from .precision import PrecisionOperator
    from scipy.sparse import csc_matrix

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
    nonzero_where = np.where(diag != 0)[0]
    n_nonzero = len(nonzero_where)

    # Load variant info
    variant_info = pl.read_csv(snplist_file, separator=',')
    num_rows = variant_info['index'].max() + 1

    # Create mapping from old indices to new indices
    rows = np.full(num_rows, -1)
    rows[nonzero_where] = np.arange(n_nonzero)

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
    matrix = matrix[nonzero_where][:, nonzero_where]

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

    # Null alleles are given NaN phase so that corresponding Z scores will be NaN
    null_match = (ref_alleles.is_null().to_numpy() & alt_alleles.is_null().to_numpy())

    # Convert to phase
    phase = np.zeros(len(anc), dtype=np.float32)
    phase[exact_match] = 1
    phase[flipped_match] = -1
    phase[null_match] = np.nan

    return pl.Series(phase)


def merge_snplists(precision_op: "PrecisionOperator",
                   sumstats: pl.DataFrame, *,
                   variant_id_col: str = 'SNP',
                   ref_allele_col: str = 'REF',
                   alt_allele_col: str = 'ALT',
                   match_by_position: bool = False,
                   pos_col: str = 'POS',
                   table_format: str = '',
                   add_cols: list[str] = None,
                   add_allelic_cols: list[str] = None,
                   representatives_only: bool = False,
                   modify_in_place: bool = False) -> Tuple["PrecisionOperator", np.ndarray]:
    """Merge a PrecisionOperator instance with summary statistics DataFrame.
    
    Args:
        precision_op: PrecisionOperator instance
        sumstats: Summary statistics DataFrame
        variant_id_col: Column name containing variant IDs
        ref_allele_col: Column name containing reference allele
        alt_allele_col: Column name containing alternative allele
        match_by_position: Whether to match SNPs by position instead of ID
        pos_col: Column name containing position
        table_format: Optional file format specification (e.g., 'vcf')
        add_cols: Optional list of column names from sumstats to append to variant_info
        add_allelic_cols: Optional list of column names from sumstats to append to variant_info,
            multiplied by the phase (-1 or 1) to align with ancestral/derived alleles.
            If no alleles are provided, these are added without sign-flipping.
        modify_in_place: Whether to modify the PrecisionOperator in place

    Returns:
        Tuple containing:
        - Modified PrecisionOperator with merged variant info and appended columns
        - Array of indices into sumstats DataFrame indicating which rows were successfully merged
    """
    # Handle VCF format
    if table_format.lower() == 'vcf':
        match_by_position = True
        pos_col = 'POS'
        ref_allele_col = 'REF'
        alt_allele_col = 'ALT'
    elif table_format.lower() == 'ldsc':
        match_by_position = False
        ref_allele_col = 'A2'
        alt_allele_col = 'A1'

    # Find position column
    pos_options = ['position', 'POS', 'BP']
    if pos_col is not None:
        pos_options.insert(0, pos_col)
    pos_col = next((col for col in pos_options if col in sumstats.columns), None)
    if pos_col is None:
        raise ValueError(
            f"Could not find position column. Tried: {', '.join(pos_options)}"
        )

    # Validate inputs
    if match_by_position:
        if pos_col not in sumstats.columns:
            msg = (f"Summary statistics must contain {pos_col} column "
                  f"for position matching. Found columns: {', '.join(sumstats.columns)}")
            raise ValueError(msg)
    else:
        if variant_id_col not in sumstats.columns:
            msg = (f"Summary statistics must contain {variant_id_col} column. "
                  f"Found columns: {', '.join(sumstats.columns)}")
            raise ValueError(msg)

    # Match variants
    match_by = ('position', pos_col) if match_by_position else ('site_ids', variant_id_col)
    merged = precision_op.variant_info.join(
        sumstats.with_row_index(name="row_nr"),
        left_on=[match_by[0]],
        right_on=[match_by[1]],
        suffix="_sumstats",
        how='inner'
    )

    # Check alleles if provided
    phase = 1
    if all(col in sumstats.columns for col in [ref_allele_col, alt_allele_col]):
        phase = merge_alleles(
            merged['anc_alleles'],
            merged['deriv_alleles'],
            merged[ref_allele_col],
            merged[alt_allele_col]
        ).alias('phase')
        merged = merged.with_columns(phase)
        
        # Update indices to only include variants with matching alleles
        merged = merged.filter(pl.col('phase') != 0)
        phase = merged['phase'].to_numpy()
    
    add_cols = add_cols or []
    add_allelic_cols = add_allelic_cols or []
    new_cols = {}
    
    # Check all columns exist
    missing_cols = [col for col in add_cols + add_allelic_cols if col not in sumstats.columns]
    if missing_cols:
        msg = (f"Requested columns not found in sumstats: {', '.join(missing_cols)}. "
              f"Available columns: {', '.join(sumstats.columns)}")
        raise ValueError(msg)
    
    # Add columns with appropriate transformations
    for col in add_cols:
        new_cols[col] = pl.col(col)
    
    for col in add_allelic_cols:
        new_cols[col] = pl.col(col) * phase

    # Add all new columns at once if any
    if new_cols:
        merged = merged.with_columns(**new_cols)
        
    # Sort by index and add is_representative column
    merged = (
        merged
        .sort('index')
        .with_columns(
            pl.col('index').is_first_distinct().cast(pl.Int8).alias('is_representative')
        )
    )
    
    # Create new PrecisionOperator with merged variant info
    unique_indices = np.unique(merged['index'].to_numpy())
    if modify_in_place:
        precision_op.set_which_indices(unique_indices)
    else:
        precision_op = precision_op[unique_indices]
    
    # Create mapping from old indices to new contiguous ones
    index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(unique_indices)}
    
    # Update indices in merged data to be contiguous using efficient replace_strict
    merged = merged.with_columns(
        pl.col('index').replace_strict(index_map).alias('index')
    )
    
    if representatives_only:
        merged = merged.filter(pl.col('is_representative') == 1)

    precision_op.variant_info = merged
    sumstat_indices = merged.select('row_nr').to_numpy().flatten().astype(int)

    return precision_op, sumstat_indices


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
    pos_options = ['position', 'POS', 'BP']
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
        
    # First sort variants by chromosome and position
    sorted_variants = variant_data.sort([chrom_col, pos_col])
    
    # Group blocks by chromosome
    chrom_blocks = {}
    for block in ldgm_metadata.iter_rows(named=True):
        chrom = block['chrom']
        if chrom not in chrom_blocks:
            chrom_blocks[chrom] = []
        chrom_blocks[chrom].append(block)
    
    # Process each chromosome's blocks at once
    partitioned = []
    for chrom, blocks in chrom_blocks.items():
        # Get all variants for this chromosome
        chrom_variants = sorted_variants.filter(pl.col(chrom_col) == chrom)
        if len(chrom_variants) == 0:
            partitioned.extend([pl.DataFrame()] * len(blocks))
            continue
            
        # Get positions array for binary search
        positions = chrom_variants.get_column(pos_col).to_numpy()
        
        # Process each block
        for block in blocks:
            # Binary search for block boundaries
            start_idx = np.searchsorted(positions, block['chromStart'])
            end_idx = np.searchsorted(positions, block['chromEnd'])
            
            # Extract variants for this block
            block_variants = chrom_variants.slice(start_idx, end_idx - start_idx)
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
                raise ValueError(f"No blocks found for populations: {populations}")
                
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


POSITIONS_FILE = 'data/rsid_position.csv'
def load_annotations(annot_path: str, 
                    chromosome: Optional[int] = None, 
                    infer_schema_length: int = 100_000,
                    add_alleles: bool = False,
                    add_positions: bool = True,
                    positions_file: str = POSITIONS_FILE,
                    file_pattern: str = "baselineLD.{chrom}.annot",
                    exclude_bed: bool = False
                    ) -> pl.DataFrame:
    """Load annotation data for specified chromosome(s) and merge with LDGMs data.
    
    Args:
        annot_path: Path to directory containing annotation files
        chromosome: Specific chromosome number, or None for all chromosomes
        infer_schema_length: Number of rows to infer schema from. Runs faster if this is smaller
        but will throw an error if too small because floating-point columns will be
        cast as integers.
        file_pattern: Filename pattern to match, with {chrom} as a placeholder for chromosome number
        exclude_bed: If True, skip loading .bed files from the annotations directory
    
    Returns:
        DataFrame containing annotations
    
    Raises:
        ValueError: If no matching annotation files are found
    """
    import glob
    import os

    # Determine which chromosomes to process
    if chromosome is not None:
        chromosomes = [chromosome]
    else:
        chromosomes = range(1, 23)  # Assuming chromosomes 1-22
    
    # Find matching files
    annotations = []
    for chromosome in chromosomes:
        file_pattern = f"*.{chromosome}.annot"
        matching_files = Path(annot_path).glob(file_pattern)
        
        # Read all matching files for this chromosome
        dfs = []
        seen_columns = set()
        
        for file_path in matching_files:
            df = pl.read_csv(
                file_path,
                separator='\t',
                infer_schema_length=infer_schema_length
            )
        
            # Find and remove any duplicate columns seen in previous files
            duplicate_cols = set(df.columns).intersection(seen_columns)
            df = df.drop(list(duplicate_cols))
            seen_columns.update(df.columns)
            dfs.append(df)
        
        # Horizontally concatenate all dataframes for this chromosome
        if dfs:
            combined_df = pl.concat(dfs, how="horizontal")
            annotations.append(combined_df)
    
    # Check if any files were found
    if not annotations:
        raise ValueError(f"No annotation files found in {annot_path} matching pattern {file_pattern}")
    
    # Concatenate all chromosome dataframes vertically
    annotations = pl.concat(annotations, how="vertical")

    # Convert binary columns to boolean to save memory
    numeric_types = (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.Float32, pl.Float64)
    binary_cols = []
    for col in annotations.columns:
        # Skip non-numeric columns
        if not isinstance(annotations[col].dtype, numeric_types):
            continue
        # Check if column only contains 0, 1 and null values
        unique_vals = set(annotations[col].unique().drop_nulls())
        if unique_vals == {0, 1}:
            binary_cols.append(col)
    
    # Convert binary columns to boolean
    if binary_cols:
        bool_exprs = [pl.col(col).cast(pl.Boolean) for col in binary_cols]
        annotations = annotations.with_columns(bool_exprs)

    if add_positions or add_alleles:
        snplist_data = pl.read_csv(
            positions_file,
            separator=',',
            columns=['chrom', 'site_ids', 'position', 'anc_alleles', 'deriv_alleles']
        )
        
        snplist_data = snplist_data.rename({
                'chrom': 'CHR',
                'site_ids': 'SNP',
                'position': 'POS',
                'anc_alleles': 'A2',
                'deriv_alleles': 'A1'
            })
        
        with_columns = ['SNP']
        if add_positions:
            with_columns += ['CHR', 'POS']
        if add_alleles:
            with_columns += ['A2', 'A1']
        
        snplist_data = snplist_data.select(with_columns)

        # Existing coordinates might be in wrong genome build
        annotations = annotations.drop(['CHR', 'BP'])

        # Merge with positions
        annotations = annotations.join(
            snplist_data,
            on='SNP',
            how='inner'
        )

    if not add_positions:
        annotations = annotations.rename({'BP': 'POS'})

    bed_files = glob.glob(os.path.join(annot_path, "*.bed"))
    if exclude_bed or not bed_files:
        return annotations

    # Process BED files if they exist
    
    # Create a list to store new annotation columns
    bed_annotations = []
    for bed_file in bed_files:
        # Get the name for this annotation from the filename
        bed_name = os.path.splitext(os.path.basename(bed_file))[0]
        
        # Read the BED file
        bed_df = read_bed(bed_file)
        
        # Convert chromosome names to match our format (e.g., "chr1" -> 1)
        bed_df = bed_df.with_columns([
            pl.col("chrom").str.replace("chr", "").cast(pl.Int64).alias("chrom")
        ])
        
        # For each chromosome, create a boolean mask for variants within regions
        mask = pl.Series(name=bed_name, values=[0] * len(annotations))
        
        # Group BED regions by chromosome for efficiency
        for chrom_group in bed_df.group_by("chrom"):
            chrom = chrom_group[0]
            regions = chrom_group[1]
            
            # Get variants for this chromosome
            chrom_mask = (annotations["CHR"] == chrom)
            chrom_pos = annotations.filter(chrom_mask)["POS"]
            
            # For each region in this chromosome
            for region in regions.iter_rows():
                start = region[1]  # chromStart
                end = region[2]    # chromEnd
                
                # Update mask for variants in this region
                region_mask = (chrom_pos >= start) & (chrom_pos < end)
                mask = mask.set(chrom_mask & region_mask, 1)
        
        bed_annotations.append(mask)
    
    # Add all BED annotations to the main DataFrame
    annotations = annotations.with_columns(bed_annotations)

    return annotations


def read_bed(bed_file: str,
             min_fields: int = 3,
             max_fields: int = 12,
             zero_based: bool = True) -> pl.DataFrame:
    """Read a UCSC BED format file.
    
    The BED format has 3 required fields and 9 optional fields:
    Required:
        1. chrom - Chromosome name
        2. chromStart - Start position (0-based)
        3. chromEnd - End position (not included in feature)
    Optional:
        4. name - Name of BED line
        5. score - Score from 0-1000
        6. strand - Strand: "+" or "-" or "."
        7. thickStart - Starting position at which feature is drawn thickly
        8. thickEnd - Ending position at which feature is drawn thickly
        9. itemRgb - RGB value (e.g., "255,0,0")
        10. blockCount - Number of blocks (e.g., exons)
        11. blockSizes - Comma-separated list of block sizes
        12. blockStarts - Comma-separated list of block starts relative to chromStart

    Args:
        bed_file: Path to BED format file
        min_fields: Minimum number of fields required (default: 3)
        max_fields: Maximum number of fields to read (default: 12)
        zero_based: If True (default), keeps positions 0-based. If False, adds 1 to start positions.

    Returns:
        Polars DataFrame containing the BED data with appropriate column names and types.
        
    Raises:
        ValueError: If min_fields < 3 or max_fields > 12 or if file has inconsistent number of fields
    """
    if min_fields < 3:
        raise ValueError("BED format requires at least 3 fields")
    if max_fields > 12:
        raise ValueError("BED format has at most 12 fields")
    if min_fields > max_fields:
        raise ValueError("min_fields cannot be greater than max_fields")

    # Define all possible BED columns with their types
    bed_columns = [
        ('chrom', pl.Utf8),
        ('chromStart', pl.Int64),
        ('chromEnd', pl.Int64),
        ('name', pl.Utf8),
        ('score', pl.Int64),
        ('strand', pl.Utf8),
        ('thickStart', pl.Int64),
        ('thickEnd', pl.Int64),
        ('itemRgb', pl.Utf8),
        ('blockCount', pl.Int64),
        ('blockSizes', pl.Utf8),
        ('blockStarts', pl.Utf8)
    ]

    # Read and parse the file
    data = []
    with open(bed_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(('browser', 'track', '#')):
                continue
            # Split on tabs or spaces and filter out empty strings
            fields = [f for f in line.split() if f]
            if not (min_fields <= len(fields) <= max_fields):
                raise ValueError(
                    f"BED line has {len(fields)} fields, expected between {min_fields} and {max_fields}: {line}"
                )
            # Pad with None if we have fewer than max_fields
            fields.extend([None] * (max_fields - len(fields)))
            data.append(fields[:max_fields])

    # Create schema for required fields
    schema = {name: dtype for name, dtype in bed_columns[:max_fields]}

    # Create DataFrame
    df = pl.from_records(
        data,
        schema=schema,
        orient="row"
    )

    # Convert 0-based to 1-based coordinates if requested
    if not zero_based:
        df = df.with_columns([
            pl.col('chromStart') + 1
        ])
        # Also convert thick positions if present
        if 'thickStart' in df.columns:
            df = df.with_columns([
                pl.col('thickStart') + 1
            ])

    return df

def read_concat_snplists(ldgm_metadata: pl.DataFrame, parent_dir: Path) -> pl.LazyFrame:
    """Read and concatenate snplists from LDGM metadata.
    
    Args:
        ldgm_metadata: DataFrame from read_ldgm_metadata containing block info
    
    Returns:
        LazyFrame containing variant information concatenated across blocks.
    """
    ldgms = [load_ldgm(parent_dir / Path(row["name"])) for row in ldgm_metadata.iter_rows(named=True)]
    lazy_frames = [ldgm.variant_info.lazy().with_columns(pl.lit(i).alias('block')) for i, ldgm in enumerate(ldgms)]
    return pl.concat(lazy_frames)
    