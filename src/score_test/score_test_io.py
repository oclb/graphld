"""I/O operations for score test."""

from ast import Attribute
import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple, TYPE_CHECKING

import h5py
import numpy as np
import polars as pl

if TYPE_CHECKING:
    from score_test import TraitData, GeneAnnot, VariantAnnot

def _load_hdf5_group(group: h5py.Group) -> dict:
    result = {}
    for key, value in group.items():
        if isinstance(value, h5py.Group):
            result[key] = _load_hdf5_group(value)
        elif isinstance(value, h5py.Dataset):
            result[key] = value[...]
        else: 
            raise ValueError(f"Unknown type: {type(value)}")

    for attr_name, attr_value in group.attrs.items():
        result['@' + attr_name] = attr_value
    return result

def _decode_bytes_array(arr: np.ndarray) -> np.ndarray:
    """Decode numpy array of bytes to strings."""
    if not hasattr(arr, 'dtype'):
        return arr
        
    if arr.dtype.kind == 'S':
        return arr.astype(str)
    elif arr.dtype.kind == 'O' and len(arr) > 0 and isinstance(arr[0], bytes):
        try:
            return np.array([x.decode('utf-8') for x in arr])
        except Exception as e:
            logging.warning(f"Failed to decode bytes array: {e}")
            return arr
    return arr

def is_gene_level_hdf5(hdf5_path: str) -> bool:
    """Check if HDF5 file contains gene-level data.
    
    Args:
        hdf5_path: Path to HDF5 file
        
    Returns:
        True if file contains gene-level data, False for variant-level
    """
    with h5py.File(hdf5_path, 'r') as f:
        # Check metadata attribute first
        if 'data_type' in f.attrs:
            return f.attrs['data_type'] == 'gene'
        # Fallback: check if row_data contains gene_id
        if 'row_data' not in f:
            raise ValueError(f"HDF5 file must contain 'row_data' group")
        return 'gene_id' in f['row_data']


def load_row_data(hdf5_path: str) -> pl.DataFrame:
    """
    Load row data table from HDF5 file format.

    Args:
        hdf5_path: Path to the HDF5 file

    Returns:
        Polars DataFrame containing row data (variants or genes)
    """

    if not os.path.exists(hdf5_path):
        raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")

    with h5py.File(hdf5_path, 'r') as f:
        if 'row_data' not in f:
            raise ValueError(f"HDF5 file must contain 'row_data' group")
        
        # Use helper to load all datasets
        data = _load_hdf5_group(f['row_data'])
        
        # Remove attributes (prefixed with @)
        data = {k: v for k, v in data.items() if not k.startswith('@')}
        
        # Process arrays: flatten 2D columns with shape (n, 1) and convert bytes to strings
        for key, arr in data.items():
            if hasattr(arr, 'shape'):
                # Flatten 2D arrays with single column
                if len(arr.shape) == 2 and arr.shape[1] == 1:
                    data[key] = arr.ravel()
                    arr = data[key] # update reference
                
                # Convert bytes to strings
                data[key] = _decode_bytes_array(arr)

    return pl.DataFrame(data)


# Backward compatibility alias
load_variant_data = load_row_data


def get_trait_names(hdf5_path: str, trait_name: Optional[str] = None) -> List[str]:
    """
    Get list of trait names from HDF5 file.
    
    Args:
        hdf5_path: Path to the HDF5 file containing trait data
        trait_name: Optional specific trait name. If provided, returns list with just this trait.
                   If None, returns all trait names in the file.
    
    Returns:
        List of trait names
    """
    with h5py.File(hdf5_path, 'r') as f:
        if trait_name:
            if trait_name not in f['traits'].keys():
                raise ValueError(f"Trait {trait_name} not found in HDF5 file")
            return [trait_name]
        else:
            return list(f['traits'].keys())


def load_trait_hdf5(hdf5_path: str, trait_name: str) -> dict:
    """
    Load trait data from HDF5 file format.
    
    Returns:
        Dictionary with keys: gradient (required), and optionally parameters, jackknife_parameters, other datasets
    """

    with h5py.File(hdf5_path, 'r') as f:
        trait_group = f[f'traits/{trait_name}']
        
        # Use helper to load everything recursively
        data = _load_hdf5_group(trait_group)

    return data


def load_trait_data(hdf5_path: str, trait_name: str, variant_table: pl.DataFrame) -> 'TraitData':
    """
    Load trait data and combine with variant data into a TraitData object.
    
    Args:
        hdf5_path: Path to HDF5 file containing variant and trait data
        trait_name: Name of the trait to load
        variant_table: Pre-loaded variant table DataFrame
        
    Returns:
        TraitData object containing variant dataframe with gradients/corrections and parameters
    """
    # Import at runtime to avoid circular import
    try:
        from .score_test import TraitData
    except ImportError:
        from score_test import TraitData
    
    trait_hdf5 = load_trait_hdf5(hdf5_path, trait_name)
    
    # Determine primary key column
    possible_keys = ['RSID', 'gene_id', 'gene_name', 'CHR', 'POS']
    keys = [key for key in possible_keys if key in variant_table.columns]
    if len(keys) == 0:
        raise ValueError("variant_table must have one of: " + ", ".join(possible_keys))
    
    # Add trait-specific columns (gradient, hessian, etc.)
    new_columns = []
    for key_name, value in trait_hdf5.items():
        if isinstance(value, np.ndarray):
            decoded_value = _decode_bytes_array(value)
            new_columns.append(pl.Series(name=key_name, values=decoded_value))
    
    df = variant_table.with_columns(new_columns)

    if 'parameters' in trait_hdf5:
        params = trait_hdf5['parameters']['parameters']
        jk_params = trait_hdf5['parameters']['jackknife_parameters']
    else:
        params, jk_params = None, None
    
    # Create TraitData - it will compute exclude_cols and annot_names via properties
    trait_data = TraitData(
        df=df,
        params=params,
        jk_params=jk_params,
        keys=keys,
    )
    
    # Compute annotation names using the property
    trait_data.annot_names = [col for col in df.columns if col not in trait_data.exclude_cols]
    
    return trait_data


def load_annotations(annot_path: str,
                    chromosome: Optional[int] = None,
                    infer_schema_length: int = 100_000,
                    add_positions: bool = True) -> pl.DataFrame:
    """Load annotation data for specified chromosome(s).

    Args:
        annot_path: Path to directory containing annotation files
        chromosome: Specific chromosome number, or None for all chromosomes
        infer_schema_length: Number of rows to infer schema from
        add_positions: If True, rename BP to POS

    Returns:
        DataFrame containing annotations

    Raises:
        ValueError: If no matching annotation files are found
    """
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
        cols_found = set()
        for file_path in matching_files:
            df = pl.scan_csv(
                file_path,
                separator='\t',
                infer_schema_length=infer_schema_length
            )
            df = df.select([col for col in df.columns if col not in cols_found])
            cols_found.update(df.columns)
            dfs.append(df)

        # Horizontally concatenate all dataframes for this chromosome
        if dfs:
            combined_df = pl.concat(dfs, how="horizontal")
            combined_df = combined_df.select(sorted(combined_df.columns))
            annotations.append(combined_df)

    # Check if any files were found
    if not annotations:
        raise ValueError(
            f"No annotation files found in {annot_path} matching pattern *.{{chrom}}.annot"
        )

    # Concatenate all chromosome dataframes vertically
    annotations = pl.concat(annotations, how="vertical").collect()

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

    if not add_positions:
        annotations = annotations.rename({'BP': 'POS'})

    return annotations


def load_gene_table(gene_table_path: str, chromosomes: list[int] | None = None) -> pl.DataFrame:
    """Load gene table and optionally filter to specific chromosomes.
    
    Args:
        gene_table_path: Path to gene table TSV
        chromosomes: Optional list of chromosome numbers to filter to
        
    Returns:
        Gene table DataFrame
    """
    schema = {
        'gene_id': pl.Utf8,
        'gene_id_version': pl.Utf8,
        'gene_name': pl.Utf8,
        'start': pl.Int64,
        'end': pl.Int64,
        'CHR': pl.Utf8,
    }
    gene_table = pl.scan_csv(gene_table_path, schema=schema, separator='\t', has_header=True) \
        .filter(pl.col('CHR').is_in([str(i) for i in range(1,23)])) \
        .filter(pl.col('gene_id').is_not_null()) \
        .with_columns(pl.col('gene_name').fill_null('NA')) \
        .with_columns(((pl.col('start') + pl.col('end')) / 2).alias('midpoint')) \
        .with_columns(pl.col('midpoint').alias('POS')) \
        .sort(pl.col('CHR').cast(pl.Int64), 'midpoint') \
        .collect()
    
    if chromosomes:
        # Convert chromosomes to integers if they're strings
        if isinstance(chromosomes[0], str):
            chromosomes = [int(c) for c in chromosomes if c.isdigit()]
        gene_table = gene_table.filter(pl.col('CHR').cast(pl.Int64).is_in(chromosomes))
    
    # Add POS column (using midpoint) for compatibility with position-based functions
    gene_table = gene_table.with_columns(
        pl.col('midpoint').cast(pl.Int64).alias('POS')
    )
    
    return gene_table


def load_gene_sets_from_gmt(gene_annot_dir: str) -> dict[str, list[str]]:
    """Load gene sets from GMT files in a directory.
    
    GMT format: set_name<tab>description<tab>gene1<tab>gene2<tab>...
    
    Returns:
        Dictionary mapping set names to lists of genes
    """
    import glob
    from pathlib import Path
    
    gmt_files = glob.glob(str(Path(gene_annot_dir) / '*.gmt'))
    if not gmt_files:
        raise FileNotFoundError(f"No .gmt files found in {gene_annot_dir}")
    
    gene_sets = {}
    for gmt_file in gmt_files:
        with open(gmt_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    set_name = parts[0]
                    genes = parts[2:]  # Skip description
                    gene_sets[set_name] = genes
    
    return gene_sets


def load_variant_annotations(annot_dir: str, annot_names: list[str] | None = None) -> 'VariantAnnot':
    """Load variant-level annotations from directory.
    
    Args:
        annot_dir: Directory containing annotation files
        annot_names: Optional list of specific annotation names to load
        
    Returns:
        VariantAnnot object
    """
    # Import at runtime to avoid circular import
    try:
        from .score_test import VariantAnnot
    except ImportError:
        from score_test import VariantAnnot
    
    df_annot = load_annotations(annot_dir, add_positions=False)
    
    # Rename SNP to RSID for consistency
    if 'SNP' in df_annot.columns:
        df_annot = df_annot.rename({'SNP': 'RSID'})
    
    # Note: load_annotations with add_positions=False renames BP to POS
    exclude_cols = ['CHR', 'BP', 'POS', 'RSID', 'CM']
    
    if annot_names:
        available = [col for col in df_annot.columns if col not in exclude_cols]
        annot_names = [name for name in annot_names if name in available]
    else:
        annot_names = [col for col in df_annot.columns if col not in exclude_cols]
    
    return VariantAnnot(df_annot, annot_names)


def load_gene_annotations(gene_annot_dir: str, 
                          variant_table: pl.DataFrame, 
                          gene_table_path: str, 
                          nearest_weights: np.ndarray, 
                          annot_names: list[str] | None = None,
                          ) -> 'GeneAnnot':
    """Load gene-level annotations as a GeneAnnot object.
    
    Args:
        gene_annot_dir: Directory containing GMT files with gene sets
        variant_table: Variant table DataFrame (used to determine chromosomes)
        gene_table_path: Path to gene table TSV file
        nearest_weights: Weights for k-nearest genes
        annot_names: Optional list of specific annotation names to load
        
    Returns:
        GeneAnnot object
    """
    # Import at runtime to avoid circular import
    try:
        from .score_test import GeneAnnot
    except ImportError:
        from score_test import GeneAnnot
    
    chromosomes = variant_table['CHR'].unique().sort().to_list()
    gene_table = load_gene_table(gene_table_path, chromosomes)
    gene_sets = load_gene_sets_from_gmt(gene_annot_dir)
    
    if annot_names:
        gene_sets = {name: genes for name, genes in gene_sets.items() if name in annot_names}
    
    return GeneAnnot(gene_sets)


def create_random_gene_annotations(data_table: pl.DataFrame, 
                                  gene_table_path: str, 
                                  probs: List[float],
                                  ) -> 'GeneAnnot':
    """Create random gene-level annotations as a GeneAnnot object.
    
    Args:
        data_table: Data table DataFrame (variants or genes)
        gene_table_path: Path to gene table TSV file
        probs: List of probabilities for random gene selection
        
    Returns:
        GeneAnnot object with random gene sets
    """
    # Import at runtime to avoid circular import
    try:
        from .score_test import GeneAnnot
    except ImportError:
        from score_test import GeneAnnot
    
    # Check if this is gene-level data (has gene_id column)
    if 'gene_id' in data_table.columns:
        # For gene-level data, use genes directly from the data table
        gene_names = data_table['gene_name'].to_list()
    else:
        # For variant-level data, load genes from file
        chromosomes = data_table['CHR'].unique().sort().to_list()
        gene_table = load_gene_table(gene_table_path, chromosomes)
        gene_names = gene_table['gene_name'].to_list()
    
    # Create random gene sets
    gene_sets = {}
    for i, p in enumerate(probs):
        set_name = f"random_gene_{i}"
        n_genes = int(len(gene_names) * p)
        gene_sets[set_name] = list(np.random.choice(gene_names, size=n_genes, replace=False))
    
    return GeneAnnot(gene_sets)


def create_random_variant_annotations(variant_table: pl.DataFrame, 
                                     probs: List[float],
                                     ) -> 'VariantAnnot':
    """Create random variant-level annotations as a VariantAnnot object.
    
    Args:
        variant_table: Variant table DataFrame
        probs: List of probabilities for random annotation
        
    Returns:
        VariantAnnot object with random variant annotations
    """
    # Import at runtime to avoid circular import
    try:
        from .score_test import VariantAnnot
    except ImportError:
        from score_test import VariantAnnot
    
    num_variants = len(variant_table)
    
    # Create random annotations
    variant_annots = {}
    annot_names = []
    
    for i, p in enumerate(probs):
        col_name = f"random_variant_{i}"
        annot_names.append(col_name)
        variant_annots[col_name] = np.random.binomial(1, p, size=num_variants).astype(np.float64)
    
    # Create output DataFrame
    # Use RSID for variants
    if 'RSID' not in variant_table.columns:
        raise ValueError("variant_table must have 'RSID' column for variant annotations")
    
    df_annot = pl.DataFrame({
        'CHR': variant_table['CHR'],
        'BP': variant_table['POS'],
        'RSID': variant_table['RSID'],
        'CM': pl.Series([0.0] * len(variant_table)),
        **variant_annots
    })
    
    return VariantAnnot(df_annot, annot_names)

def get_trait_groups(hdf5_path: str) -> dict[str, list[str]]:
    """Get trait groups for meta-analysis from HDF5 file.
    
    Args:
        hdf5_path: Path to HDF5 file containing trait groups
        
    Returns:
        Dictionary mapping group names to lists of trait names.
        Returns empty dict if no groups are defined.
    """
    with h5py.File(hdf5_path, 'r') as f:
        if 'groups' not in f:
            return {}
        
        groups = {}
        groups_group = f['groups']
        for group_name in groups_group.keys():
            trait_list = groups_group[group_name][:]
            # Convert bytes to strings if necessary
            if trait_list.dtype.kind == 'S':
                trait_list = [t.decode('utf-8') for t in trait_list]
            else:
                trait_list = trait_list.tolist()
            groups[group_name] = trait_list
        
        return groups


def save_trait_groups(hdf5_path: str, groups: dict[str, list[str]]) -> None:
    """Save trait groups for meta-analysis to HDF5 file.
    
    Args:
        hdf5_path: Path to HDF5 file to create/update
        groups: Dictionary mapping group names to lists of trait names
    """
    with h5py.File(hdf5_path, 'a') as f:
        if 'groups' in f:
            del f['groups']
        
        groups_group = f.create_group('groups')
        
        for group_name, trait_list in groups.items():
            # Convert strings to bytes for HDF5 storage
            trait_array = np.array(trait_list, dtype='S')
            groups_group.create_dataset(group_name, data=trait_array)


def save_trait_data(trait_data: 'TraitData', 
                    hdf5_path: str, 
                    trait_name: str,
                    ) -> None:
    """Save trait data to HDF5 file in new format.
    
    Args:
        trait_data: TraitData object to save
        hdf5_path: Path to HDF5 file to create/update
        trait_name: Name of the trait
    """
    with h5py.File(hdf5_path, 'a') as f:
        # Create metadata attribute if it doesn't exist
        if 'metadata' not in f.attrs:
            f.attrs['metadata'] = ''
        
        # Set data_type and keys attributes
        if 'gene_id' in trait_data.df.columns:
            f.attrs['data_type'] = 'gene'
            f.attrs['keys'] = ['gene_id', 'gene_name']
        else:
            f.attrs['data_type'] = 'variant'
            f.attrs['keys'] = ['RSID', 'POS']
        
        # Create row_data group if it doesn't exist, or recreate if size mismatch
        if 'row_data' in f:
            existing_size = len(f['row_data'][list(f['row_data'].keys())[0]])
            expected_size = len(trait_data.df)
            if existing_size != expected_size:
                # Size mismatch - delete and recreate
                del f['row_data']
        
        if 'row_data' not in f:
            data_group = f.create_group('row_data')
            
            # Save all columns except trait-specific ones (gradient, hessian, etc.)
            # Use TraitData's exclude_cols property
            row_data_cols = set(trait_data.df.columns) - trait_data.exclude_cols
            # Add back standard columns that should be in row_data
            row_data_cols.update({'CHR', 'POS', 'jackknife_blocks'})
            if 'RSID' in trait_data.df.columns:
                row_data_cols.add('RSID')
            if 'gene_id' in trait_data.df.columns:
                row_data_cols.update({'gene_id', 'gene_name'})
            
            for col in trait_data.df.columns:
                if col in row_data_cols:
                    data = trait_data.df[col].to_numpy()
                    # Handle string columns
                    if data.dtype == object:
                        data = data.astype('S')
                    data_group.create_dataset(col, data=data)
        
        # Create traits group if it doesn't exist
        if 'traits' not in f:
            f.create_group('traits')
        
        trait_path = f'traits/{trait_name}'
        if trait_path in f:
            del f[trait_path]
        
        trait_group = f.create_group(trait_path)
        
        # Save parameters in parameters/ subgroup
        if trait_data.params is not None or trait_data.jk_params is not None:
            params_group = trait_group.create_group('parameters')
            if trait_data.params is not None:
                params_group.create_dataset('parameters', data=trait_data.params)
            if trait_data.jk_params is not None:
                params_group.create_dataset('jackknife_parameters', data=trait_data.jk_params)
        
        # Save gradient (required)
        trait_group.create_dataset('gradient', data=trait_data.df['gradient'].to_numpy())
        
        # Save any other trait-specific datasets (e.g., hessian)
        standard_cols = {'CHR', 'POS', 'jackknife_blocks', 'RSID', 'gene_id', 'gene_name', 'gradient'}
        annot_cols = set(trait_data.annot_names) if trait_data.annot_names else set()
        for col in trait_data.df.columns:
            if col not in standard_cols and col not in annot_cols:
                trait_group.create_dataset(col, data=trait_data.df[col].to_numpy())