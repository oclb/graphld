"""I/O operations for score test."""

import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple, TYPE_CHECKING

import h5py
import numpy as np
import polars as pl

if TYPE_CHECKING:
    from score_test import TraitData, GeneAnnot, VariantAnnot


def is_gene_level_hdf5(hdf5_path: str) -> bool:
    """Check if HDF5 file contains gene-level data.
    
    Args:
        hdf5_path: Path to HDF5 file
        
    Returns:
        True if file contains gene-level data, False for variant-level
    """
    with h5py.File(hdf5_path, 'r') as f:
        return 'genes' in f


def load_data_table(hdf5_path: str) -> pl.DataFrame:
    """
    Load data table (variants or genes) from HDF5 file format.

    Args:
        hdf5_path: Path to the HDF5 file

    Returns:
        Polars DataFrame containing variant or gene data
    """

    if not os.path.exists(hdf5_path):
        raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")

    with h5py.File(hdf5_path, 'r') as f:
        # Check if this is gene-level or variant-level data
        if 'genes' in f:
            data_group = f['genes']
        elif 'variants' in f:
            data_group = f['variants']
        else:
            raise ValueError(f"HDF5 file must contain either 'genes' or 'variants' group")
        
        # Load all datasets from data group
        data = {}
        for key in data_group.keys():
            dataset = data_group[key]
            if hasattr(dataset, 'shape'):  # It's a dataset, not a group
                arr = dataset[:].ravel() if len(dataset.shape) == 2 and dataset.shape[1] == 1 else dataset[:]
                # Convert bytes to strings if needed
                if arr.dtype.kind == 'S':
                    arr = arr.astype(str)
                data[key] = arr
        
        # Handle old format with 'annotations' matrix
        if 'annotations' in data and len(data['annotations'].shape) == 2:
            annotations = data.pop('annotations')
            for i in range(annotations.shape[1]):
                data[f'annot_{i}'] = annotations[:, i]

    return pl.DataFrame(data)


# Backward compatibility alias
load_variant_data = load_data_table


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
        Dictionary with keys: gradient (required), and optionally parameters, jackknife_parameters, hessian
    """

    required_keys = ['gradient']
    optional_keys = ['parameters', 'jackknife_parameters', 'hessian']
    
    with h5py.File(hdf5_path, 'r') as f:
        trait_group = f[f'traits/{trait_name}']
        data = {key: trait_group[key][:] for key in required_keys}
        
        # Load optional keys if they exist
        for key in optional_keys:
            if key in trait_group:
                data[key] = trait_group[key][:]
            else:
                data[key] = None

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
    
    # Store gradient with variant table, and hessian if available
    new_columns = [pl.Series(name='gradient', values=trait_hdf5['gradient'])]
    
    # Determine key column (RSID for variants, gene_id for genes)
    if 'RSID' in variant_table.columns:
        key = 'RSID'
        exclude_cols = ['CHR', 'POS', 'RSID', 'jackknife_blocks', 'gradient']
    elif 'gene_id' in variant_table.columns:
        key = 'gene_id'
        exclude_cols = ['CHR', 'POS', 'gene_id', 'gene_name', 'jackknife_blocks', 'gradient']
    else:
        raise ValueError("variant_table must have either 'RSID' or 'gene_id' column")
    
    if trait_hdf5['hessian'] is not None and len(trait_hdf5['hessian']) > 0:
        new_columns.append(pl.Series(name='hessian', values=trait_hdf5['hessian']))
        exclude_cols.append('hessian')
    
    df = variant_table.with_columns(new_columns)
    
    # Get annotation column names (exclude standard columns)
    annot_names = [col for col in df.columns if col not in exclude_cols]
    
    return TraitData(
        df=df,
        params=trait_hdf5['parameters'],
        jk_params=trait_hdf5['jackknife_parameters'],
        annot_names=annot_names,
        key=key,
    )


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
    
    # Note: load_annotations with add_positions=False renames BP to POS
    exclude_cols = ['CHR', 'BP', 'POS', 'SNP', 'CM']
    
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
    # Use RSID for variants, gene_id for genes
    snp_col = variant_table['RSID'] if 'RSID' in variant_table.columns else variant_table['gene_id']
    
    df_annot = pl.DataFrame({
        'CHR': variant_table['CHR'],
        'BP': variant_table['POS'],
        'SNP': snp_col,
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
    """Save trait data to HDF5 file.
    
    Args:
        trait_data: TraitData object to save
        hdf5_path: Path to HDF5 file to create/update
        trait_name: Name of the trait
    """
    with h5py.File(hdf5_path, 'a') as f:
        # Determine if this is gene-level or variant-level data
        is_gene_level = 'gene_id' in trait_data.df.columns
        group_name = 'genes' if is_gene_level else 'variants'
        
        # Create data group if it doesn't exist, or recreate if size mismatch
        if group_name in f:
            existing_size = len(f[group_name][list(f[group_name].keys())[0]])
            expected_size = len(trait_data.df)
            if existing_size != expected_size:
                # Size mismatch - delete and recreate
                del f[group_name]
        
        if group_name not in f:
            data_group = f.create_group(group_name)
            
            # Save all columns except gradient, which is trait specific
            exclude_cols = {'gradient', 'hessian'}
            
            for col in trait_data.df.columns:
                if col not in exclude_cols:
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
        
        # Save trait data
        if trait_data.params is not None:
            trait_group.create_dataset('parameters', data=trait_data.params)

        if trait_data.jk_params is not None:
            trait_group.create_dataset('jackknife_parameters', data=trait_data.jk_params)

        trait_group.create_dataset('gradient', data=trait_data.df['gradient'].to_numpy())