"""GraphLD Score Test - Fast heritability enrichment testing for genomic annotations.

This module provides a fast score test for testing genomic or gene annotations
for heritability enrichment conditional upon a null model. The test produces
Z scores where positive values indicate enrichment and negative values indicate
depletion.

Main Functions:
    run_score_test: Run the enrichment score test on annotations
    load_variant_data: Load precomputed variant-level score statistics
    load_trait_data: Load trait-specific score statistics
    load_annotations: Load annotation data from LDSC format files
    load_gene_table: Load gene position table
    load_gene_sets_from_gmt: Load gene sets from GMT format files
    convert_gene_to_variant_annotations: Convert gene-level to variant-level annotations

Example::

    from score_test import run_score_test, load_variant_data, load_annotations

    variant_data = load_variant_data("path/to/scores.h5")
    annotations = load_annotations("path/to/annot_dir/")
    results = run_score_test(variant_data, annotations)
"""

from .score_test import run_score_test
from .score_test_io import load_variant_data, load_trait_data, load_annotations, load_gene_table, load_gene_sets_from_gmt
from .genesets import convert_gene_to_variant_annotations, gene_variant_matrix

__all__ = [
    'run_score_test',
    'load_variant_data',
    'load_trait_data',
    'load_annotations',
    'load_gene_table',
    'load_gene_sets_from_gmt',
    'convert_gene_to_variant_annotations',
    'gene_variant_matrix',
]
