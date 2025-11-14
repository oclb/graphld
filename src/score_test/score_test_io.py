"""I/O operations for score test."""

import os
from pathlib import Path
from typing import List, Optional, Tuple

import h5py
import numpy as np
import polars as pl


def load_variant_data(hdf5_path: str) -> pl.DataFrame:
    """
    Load variant statistics from HDF5 file format.

    Args:
        hdf5_path: Path to the HDF5 file containing variant statistics

    Returns:
        Polars DataFrame containing variant statistics
    """

    if not os.path.exists(hdf5_path):
        raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")

    with h5py.File(hdf5_path, 'r') as f:
        variants_group = f['variants']
        data = {
            'CHR': variants_group['CHR'][:].ravel(),
            'POS': variants_group['POS'][:].ravel(),
            'annotations': variants_group['annotations'][:,:],
            'jackknife_blocks': variants_group['jackknife_blocks'][:].ravel(),
            'RSID': variants_group['RSID'][:].ravel().astype(str)
        }

    return pl.DataFrame(data)


def load_trait_data(hdf5_path: str, trait_name: str) -> dict:
    """
    Load trait data from HDF5 file format.
    """

    keys = ['parameters', 'jackknife_parameters', 'gradient', 'hessian']
    with h5py.File(hdf5_path, 'r') as f:
        trait_group = f[f'traits/{trait_name}']
        data = {key: trait_group[key][:] for key in keys}

    return data


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


def load_variant_annotations(annot_dir: str, annot_names: list[str] | None = None) -> Tuple[pl.DataFrame, List[str]]:
    """Load variant-level annotations from directory.
    
    Args:
        annot_dir: Directory containing annotation files
        annot_names: Optional list of specific annotation names to load
        
    Returns:
        Tuple of (annotations_df, annotation_column_names)
    """
    df_annot = load_annotations(annot_dir, add_positions=False)
    
    # Note: load_annotations with add_positions=False renames BP to POS
    exclude_cols = ['CHR', 'BP', 'POS', 'SNP', 'CM']
    
    if annot_names:
        available = [col for col in df_annot.columns if col not in exclude_cols]
        annot_names = [name for name in annot_names if name in available]
    else:
        annot_names = [col for col in df_annot.columns if col not in exclude_cols]
    
    return df_annot, annot_names


def load_gene_annotations(gene_annot_dir: str, variant_data: pl.DataFrame, gene_table_path: str, 
                         nearest_weights: np.ndarray, annot_names: list[str] | None = None) -> Tuple[pl.DataFrame, List[str]]:
    """Load gene-level annotations and convert to variant-level.
    
    Args:
        gene_annot_dir: Directory containing GMT files with gene sets
        variant_data: Variant data DataFrame
        gene_table_path: Path to gene table TSV file
        nearest_weights: Weights for k-nearest genes
        annot_names: Optional list of specific annotation names to load
        
    Returns:
        Tuple of (variant_annotations_df, annotation_column_names)
    """

    # Handle imports when running either as a script or as a package
    try:
        from .genesets import (load_gene_table, load_gene_sets_from_gmt, 
                              convert_gene_sets_to_variant_annotations)
    except ImportError:
        from genesets import (load_gene_table, load_gene_sets_from_gmt, 
                              convert_gene_sets_to_variant_annotations)
    
    chromosomes = variant_data['CHR'].unique().sort().to_list()
    gene_table = load_gene_table(gene_table_path, chromosomes)
    gene_sets = load_gene_sets_from_gmt(gene_annot_dir)
    
    if annot_names:
        gene_sets = {name: genes for name, genes in gene_sets.items() if name in annot_names}
    
    df_annot = convert_gene_sets_to_variant_annotations(
        gene_sets, variant_data, gene_table, nearest_weights
    )
    annot_columns = list(gene_sets.keys())
    
    return df_annot, annot_columns


def create_random_gene_annotations(variant_data: pl.DataFrame, gene_table_path: str, 
                                  nearest_weights: np.ndarray, probs: List[float]) -> Tuple[pl.DataFrame, List[str]]:
    """Create random gene-level annotations and convert to variant-level.
    
    Args:
        variant_data: Variant data DataFrame
        gene_table_path: Path to gene table TSV file
        nearest_weights: Weights for k-nearest genes
        probs: List of probabilities for random gene selection
        
    Returns:
        Tuple of (variant_annotations_df, annotation_column_names)
    """
    try:
        from .genesets import (load_gene_table, convert_gene_sets_to_variant_annotations)
    except ImportError:
        from genesets import (load_gene_table, convert_gene_sets_to_variant_annotations)
    
    chromosomes = variant_data['CHR'].unique().sort().to_list()
    gene_table = load_gene_table(gene_table_path, chromosomes)
    
    # Create random gene sets
    gene_names = gene_table['gene_name'].to_list()
    gene_sets = {}
    for i, p in enumerate(probs):
        set_name = f"random_gene_{i}"
        # Randomly select genes with probability p
        selected = np.random.binomial(1, p, size=len(gene_names)).astype(bool)
        gene_sets[set_name] = [gene for gene, sel in zip(gene_names, selected) if sel]
    
    df_annot = convert_gene_sets_to_variant_annotations(
        gene_sets, variant_data, gene_table, nearest_weights
    )
    annot_names = list(gene_sets.keys())
    
    return df_annot, annot_names


def create_random_variant_annotations(variant_data: pl.DataFrame, probs: List[float]) -> Tuple[pl.DataFrame, List[str]]:
    """Create random variant-level annotations.
    
    Args:
        variant_data: Variant data DataFrame
        probs: List of probabilities for random annotation
        
    Returns:
        Tuple of (variant_annotations_df, annotation_column_names)
    """
    num_variants = len(variant_data)
    
    # Create random annotations
    variant_annots = {}
    annot_names = []
    
    for i, p in enumerate(probs):
        col_name = f"random_variant_{i}"
        annot_names.append(col_name)
        variant_annots[col_name] = np.random.binomial(1, p, size=num_variants).astype(np.float64)
    
    # Create output DataFrame
    df_annot = pl.DataFrame({
        'CHR': variant_data['CHR'],
        'BP': variant_data['POS'],
        'SNP': variant_data['RSID'],
        'CM': pl.Series([0.0] * len(variant_data)),
        **variant_annots
    })
    
    return df_annot, annot_names
