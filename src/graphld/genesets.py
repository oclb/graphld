"""Gene set annotation utilities for graphREML.

This module provides functions to load gene sets from GMT files and convert
gene-level annotations to variant-level annotations for use with graphREML.
"""

import glob
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl
from scipy.sparse import csr_matrix

POSITION_SCALE = 1e9  # Scale factor for chromosome positions


def _get_nearest_genes(
    var_pos: np.ndarray,
    gene_pos: np.ndarray,
    num_nearest: int,
) -> np.ndarray:
    """Find the k nearest genes for each variant.

    Args:
        var_pos: Variant positions (must be sorted)
        gene_pos: Gene positions (must be sorted)
        num_nearest: Number of nearest genes to find

    Returns:
        Array of shape (nvar, num_nearest) with indices of nearest genes
    """
    # Ensure 1D inputs (HDF5 may store as Nx1)
    var_pos = np.asarray(var_pos).ravel()
    gene_pos = np.asarray(gene_pos).ravel()
    if not np.array_equal(var_pos, np.sort(var_pos)):
        raise ValueError("Variant positions must be sorted")
    if not np.array_equal(gene_pos, np.sort(gene_pos)):
        raise ValueError("Gene positions must be sorted")
    nvar, ngene = var_pos.size, gene_pos.size

    # how many genes come before each variant?
    order = np.argsort(np.concatenate((var_pos, gene_pos)))
    perm = np.empty(nvar + ngene)
    perm[order] = np.arange(nvar + ngene)  # number of genes or variants before
    num_genes_before = perm[:nvar] - np.arange(nvar)

    dist_k_flanking = np.empty((nvar, 2 * num_nearest), dtype=np.int32)
    genes_k_flanking = np.empty((nvar, 2 * num_nearest), dtype=np.int32)
    for k in range(-num_nearest, num_nearest):
        # fancy indexing + wraparound
        genes_k_flanking[:, k] = (num_genes_before + k) % ngene
        dist_k_flanking[:, k] = abs(var_pos - gene_pos[genes_k_flanking[:, k]])

    dist_k_flanking = dist_k_flanking.ravel()
    genes_k_flanking = genes_k_flanking.ravel()

    result = np.empty((nvar, num_nearest), dtype=np.int32)
    left_contestant = np.arange(0, 2 * nvar * num_nearest, 2 * num_nearest, dtype=np.int32)
    right_contestant = left_contestant + num_nearest * 2 - 1
    for k in range(num_nearest):
        right_wins = dist_k_flanking[right_contestant] < dist_k_flanking[left_contestant]
        left_wins = ~right_wins
        result[right_wins, k] = genes_k_flanking[right_contestant[right_wins]]
        result[left_wins, k] = genes_k_flanking[left_contestant[left_wins]]
        right_contestant -= right_wins
        left_contestant += left_wins

    return result


def _get_gene_variant_matrix(
    var_pos: np.ndarray,
    gene_pos: np.ndarray,
    nearest_weights: np.ndarray,
    dtype: np.dtype = np.float64,
) -> csr_matrix:
    """Build a sparse variants x genes weighted matrix.

    M[i, j] = nearest_weights[k] if gene j is the k-th closest gene to variant i, else 0.

    Args:
        var_pos: Variant positions (1D array of length nvar)
        gene_pos: Gene positions (1D array of length ngene)
        nearest_weights: Weights for the k-nearest genes (1D array of length num_nearest)
        dtype: Data type of the sparse matrix values

    Returns:
        CSR matrix of shape (nvar, ngene) where entry (i,j) equals nearest_weights[k]
        if gene j is the k-th closest to variant i, otherwise 0.
    """
    nvar, ngene = len(var_pos), len(gene_pos)
    num_nearest = len(nearest_weights)
    assert num_nearest > 0 and num_nearest <= ngene

    nearest = _get_nearest_genes(var_pos, gene_pos, num_nearest)
    # Row indices: each variant repeated num_nearest times
    rows = np.repeat(np.arange(nvar, dtype=np.int32), num_nearest)
    cols = nearest.ravel()
    data = np.tile(nearest_weights, nvar)
    return csr_matrix((data, (rows, cols)), shape=(nvar, ngene), dtype=dtype)


def _compute_positions(table: pl.DataFrame) -> np.ndarray:
    """Compute global positions (CHR * POSITION_SCALE + POS)."""
    if table["POS"].max() >= POSITION_SCALE:
        raise ValueError(f"POS values must be less than POSITION_SCALE: {POSITION_SCALE}")
    return (table["POS"] + table["CHR"].cast(pl.Int64) * POSITION_SCALE).to_numpy().astype(np.int64)


def load_gene_table(
    gene_table_path: str, chromosomes: Optional[list[int]] = None
) -> pl.DataFrame:
    """Load gene table and optionally filter to specific chromosomes.

    Args:
        gene_table_path: Path to gene table TSV file with columns:
            gene_id, gene_id_version, gene_name, start, end, CHR
        chromosomes: Optional list of chromosome numbers to filter to

    Returns:
        Gene table DataFrame with POS column set to gene midpoint
    """
    schema = {
        "gene_id": pl.Utf8,
        "gene_id_version": pl.Utf8,
        "gene_name": pl.Utf8,
        "start": pl.Int64,
        "end": pl.Int64,
        "CHR": pl.Utf8,
    }
    gene_table = (
        pl.scan_csv(gene_table_path, schema=schema, separator="\t", has_header=True)
        .filter(pl.col("CHR").is_in([str(i) for i in range(1, 23)]))
        .filter(pl.col("gene_id").is_not_null())
        .with_columns(pl.col("gene_name").fill_null("NA"))
        .with_columns(((pl.col("start") + pl.col("end")) / 2).alias("midpoint"))
        .with_columns(pl.col("midpoint").alias("POS"))
        .sort(pl.col("CHR").cast(pl.Int64), "midpoint")
        .collect()
    )

    if chromosomes:
        # Convert chromosomes to integers if they're strings
        if isinstance(chromosomes[0], str):
            chromosomes = [int(c) for c in chromosomes if c.isdigit()]
        gene_table = gene_table.filter(pl.col("CHR").cast(pl.Int64).is_in(chromosomes))

    # Add POS column (using midpoint) for compatibility with position-based functions
    gene_table = gene_table.with_columns(pl.col("midpoint").cast(pl.Int64).alias("POS"))

    return gene_table


def load_gene_sets_from_gmt(gene_annot_dir: str) -> dict[str, list[str]]:
    """Load gene sets from GMT files in a directory.

    GMT format: set_name<tab>description<tab>gene1<tab>gene2<tab>...

    Args:
        gene_annot_dir: Directory containing .gmt files

    Returns:
        Dictionary mapping set names to lists of genes

    Raises:
        FileNotFoundError: If no .gmt files are found in the directory
    """
    gmt_files = glob.glob(str(Path(gene_annot_dir) / "*.gmt"))
    if not gmt_files:
        raise FileNotFoundError(f"No .gmt files found in {gene_annot_dir}")

    gene_sets = {}
    for gmt_file in gmt_files:
        with open(gmt_file) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 3:
                    set_name = parts[0]
                    genes = parts[2:]  # Skip description
                    gene_sets[set_name] = genes

    return gene_sets


def _is_gene_id(gene: str) -> bool:
    """Check if a gene identifier is an Ensembl ID (vs gene symbol)."""
    return "ENSG" in gene


def convert_gene_sets_to_variant_annotations(
    gene_sets: dict[str, list[str]],
    variant_table: pl.DataFrame,
    gene_table: pl.DataFrame,
    nearest_weights: np.ndarray,
) -> pl.DataFrame:
    """Convert gene sets to variant-level annotations.

    Args:
        gene_sets: Dictionary mapping gene set names to lists of genes (symbols or IDs)
        variant_table: Variant table DataFrame with CHR, POS, SNP columns
        gene_table: Gene table DataFrame with CHR, POS, gene_id, gene_name columns
        nearest_weights: Weights for k-nearest genes (e.g., [0.4, 0.2, 0.1, 0.1, 0.1, 0.05, 0.05])

    Returns:
        DataFrame with variant-level annotations in LDSC format (CHR, BP, SNP, CM, ...)
    """
    # Determine if using gene IDs or symbols
    first_set = next(iter(gene_sets.values()))
    use_gene_id = _is_gene_id(first_set[0]) if first_set else False
    gene_key = "gene_id" if use_gene_id else "gene_name"

    # Get gene-variant matrix
    variant_positions = _compute_positions(variant_table)
    gene_positions = _compute_positions(gene_table)
    gv_matrix = _get_gene_variant_matrix(variant_positions, gene_positions, nearest_weights)

    # Convert each gene set to variant-level annotation
    variant_annots = {}
    gene_identifiers = gene_table[gene_key].to_list()

    for set_name, genes in gene_sets.items():
        gene_set = set(genes)
        gene_values = np.array(
            [1.0 if gene in gene_set else 0.0 for gene in gene_identifiers], dtype=np.float64
        )
        variant_values = (gv_matrix @ gene_values.reshape(-1, 1)).ravel()
        variant_annots[set_name] = variant_values

    # Create output DataFrame in LDSC format
    df_annot = pl.DataFrame(
        {
            "CHR": variant_table["CHR"],
            "BP": variant_table["POS"],
            "SNP": variant_table["SNP"],
            "CM": pl.Series([0.0] * len(variant_table)),
            **variant_annots,
        }
    )

    return df_annot


def load_gene_annotations(
    gene_annot_dir: str,
    variant_table: pl.DataFrame,
    gene_table_path: str,
    nearest_weights: np.ndarray,
    annot_names: Optional[list[str]] = None,
) -> pl.DataFrame:
    """Load gene-level annotations and convert to variant-level.

    This is the main entry point for loading GMT-based gene annotations
    for use with graphREML.

    Args:
        gene_annot_dir: Directory containing GMT files with gene sets
        variant_table: Variant table DataFrame with CHR, POS, SNP columns
        gene_table_path: Path to gene table TSV file
        nearest_weights: Weights for k-nearest genes
        annot_names: Optional list of specific annotation names to load

    Returns:
        DataFrame with variant-level annotations
    """
    chromosomes = variant_table["CHR"].unique().sort().to_list()
    gene_table = load_gene_table(gene_table_path, chromosomes)
    gene_sets = load_gene_sets_from_gmt(gene_annot_dir)

    if annot_names:
        gene_sets = {name: genes for name, genes in gene_sets.items() if name in annot_names}

    return convert_gene_sets_to_variant_annotations(
        gene_sets, variant_table, gene_table, nearest_weights
    )
