"""Gene set annotation utilities for graphREML.

This module keeps the GraphLD-facing LDSC annotation contract while reusing the
lightweight score-test implementation for gene-table loading, GMT loading,
position encoding, nearest-gene mapping, and gene-to-variant conversion.
"""

from typing import Optional

import numpy as np
import polars as pl

from score_test.genesets import (
    POSITION_SCALE,
    _compute_positions,
    _get_chromosome_aware_nearest_genes,
    _get_gene_variant_matrix,
    _get_nearest_genes,
    _is_gene_id,
    _nearest_indices_for_position,
    gene_sets_to_variant_annotation_frame,
    gene_variant_matrix,
    load_gene_sets_from_gmt,
    load_gene_table,
)


def convert_gene_sets_to_variant_annotations(
    gene_sets: dict[str, list[str]],
    variant_table: pl.DataFrame,
    gene_table: pl.DataFrame,
    nearest_weights: np.ndarray,
) -> pl.DataFrame:
    """Convert gene sets to LDSC-style variant-level annotations.

    Args:
        gene_sets: Dictionary mapping gene set names to gene symbols or Ensembl IDs
        variant_table: Variant table DataFrame with CHR, POS, SNP columns
        gene_table: Gene table DataFrame with CHR, POS, gene_id, gene_name columns
        nearest_weights: Weights for k-nearest genes

    Returns:
        DataFrame with CHR, BP, SNP, CM, and one column per gene set.
    """
    return gene_sets_to_variant_annotation_frame(
        gene_sets,
        variant_table,
        gene_table,
        nearest_weights,
        variant_id_col="SNP",
        output_id_col="SNP",
    )


def load_gene_annotations(
    gene_annot_dir: str,
    variant_table: pl.DataFrame,
    gene_table_path: str,
    nearest_weights: np.ndarray,
    annot_names: Optional[list[str]] = None,
) -> pl.DataFrame:
    """Load GMT gene annotations and convert them to variant-level annotations."""
    chromosomes = variant_table["CHR"].unique().sort().to_list()
    gene_table = load_gene_table(gene_table_path, chromosomes)
    gene_sets = load_gene_sets_from_gmt(gene_annot_dir)

    if annot_names:
        gene_sets = {name: genes for name, genes in gene_sets.items() if name in annot_names}

    return convert_gene_sets_to_variant_annotations(
        gene_sets,
        variant_table,
        gene_table,
        nearest_weights,
    )


__all__ = [
    "POSITION_SCALE",
    "_compute_positions",
    "_get_chromosome_aware_nearest_genes",
    "_get_gene_variant_matrix",
    "_get_nearest_genes",
    "_is_gene_id",
    "_nearest_indices_for_position",
    "convert_gene_sets_to_variant_annotations",
    "gene_variant_matrix",
    "load_gene_annotations",
    "load_gene_sets_from_gmt",
    "load_gene_table",
]
