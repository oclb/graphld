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


def _nearest_indices_for_position(
    position: np.integer,
    gene_pos: np.ndarray,
    gene_indices: np.ndarray,
    num_nearest: int,
) -> np.ndarray:
    """Find nearest original gene indices from a sorted gene-position slice."""
    ngene = gene_pos.size
    if num_nearest > ngene:
        raise ValueError("num_nearest cannot exceed number of genes")

    insertion = np.searchsorted(gene_pos, position, side="left")
    left = max(0, insertion - num_nearest)
    right = min(ngene, insertion + num_nearest)

    if right - left < num_nearest:
        if left == 0:
            right = min(ngene, num_nearest)
        else:
            left = max(0, ngene - num_nearest)

    candidates = np.arange(left, right)
    distances = np.abs(gene_pos[candidates] - position)
    order = np.lexsort((gene_indices[candidates], distances))
    return gene_indices[candidates[order[:num_nearest]]]


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
    if num_nearest <= 0 or num_nearest > ngene:
        raise ValueError("num_nearest must be between 1 and the number of genes")

    result = np.empty((nvar, num_nearest), dtype=np.int32)
    gene_indices = np.arange(ngene, dtype=np.int32)
    var_pos = var_pos.astype(np.int64, copy=False)
    gene_pos = gene_pos.astype(np.int64, copy=False)
    for i, position in enumerate(var_pos):
        result[i] = _nearest_indices_for_position(
            position, gene_pos, gene_indices, num_nearest
        )

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


def _get_chromosome_aware_nearest_genes(
    variant_table: pl.DataFrame,
    gene_table: pl.DataFrame,
    num_nearest: int,
) -> np.ndarray:
    """Find nearest gene indices, prioritizing same-chromosome genes."""
    nvar, ngene = len(variant_table), len(gene_table)
    if num_nearest <= 0 or num_nearest > ngene:
        raise ValueError("num_nearest must be between 1 and the number of genes")

    variant_chr = variant_table["CHR"].cast(pl.Int64).to_numpy()
    gene_chr = gene_table["CHR"].cast(pl.Int64).to_numpy()
    variant_positions = _compute_positions(variant_table)
    gene_positions = _compute_positions(gene_table)

    all_sorted_indices = np.argsort(gene_positions, kind="stable").astype(np.int32)
    genes_by_chr = {
        chromosome: np.flatnonzero(gene_chr == chromosome).astype(np.int32)
        for chromosome in np.unique(gene_chr)
    }
    for chromosome, indices in genes_by_chr.items():
        order = np.argsort(gene_positions[indices], kind="stable")
        genes_by_chr[chromosome] = indices[order]

    fallback_by_chr: dict[int, np.ndarray] = {}
    for chromosome in np.unique(variant_chr):
        other = all_sorted_indices[gene_chr[all_sorted_indices] != chromosome]
        fallback_by_chr[int(chromosome)] = other.astype(np.int32, copy=False)

    nearest = np.empty((nvar, num_nearest), dtype=np.int32)
    for i, (chromosome, position) in enumerate(zip(variant_chr, variant_positions)):
        same_chr_indices = genes_by_chr.get(chromosome, np.array([], dtype=np.int32))
        n_same = min(num_nearest, same_chr_indices.size)
        selected: list[int] = []

        if n_same:
            selected.extend(
                _nearest_indices_for_position(
                    position,
                    gene_positions[same_chr_indices],
                    same_chr_indices,
                    n_same,
                ).tolist()
            )

        if len(selected) < num_nearest:
            fallback_indices = fallback_by_chr[int(chromosome)]
            selected.extend(
                _nearest_indices_for_position(
                    position,
                    gene_positions[fallback_indices],
                    fallback_indices,
                    num_nearest - len(selected),
                ).tolist()
            )

        nearest[i] = np.asarray(selected, dtype=np.int32)

    return nearest


def gene_variant_matrix(
    variant_table: pl.DataFrame,
    gene_table: pl.DataFrame,
    nearest_weights: np.ndarray,
    dtype: np.dtype = np.float64,
) -> csr_matrix:
    """Compute a chromosome-aware variants x genes weighted matrix."""
    nvar, ngene = len(variant_table), len(gene_table)
    num_nearest = len(nearest_weights)
    if num_nearest <= 0 or num_nearest > ngene:
        raise ValueError("nearest_weights length must be between 1 and the number of genes")

    nearest = _get_chromosome_aware_nearest_genes(variant_table, gene_table, num_nearest)
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
    if not gene_sets:
        raise ValueError("gene_sets must contain at least one gene set")

    # Get gene-variant matrix
    gv_matrix = gene_variant_matrix(variant_table, gene_table, nearest_weights)

    # Convert each gene set to variant-level annotation
    variant_annots = {}
    gene_ids = gene_table["gene_id"].to_list()
    gene_names = gene_table["gene_name"].to_list()

    for set_name, genes in gene_sets.items():
        gene_set = set(genes)
        gene_values = np.array(
            [
                1.0 if gene_id in gene_set or gene_name in gene_set else 0.0
                for gene_id, gene_name in zip(gene_ids, gene_names)
            ],
            dtype=np.float64,
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
