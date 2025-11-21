import numpy as np
import polars as pl
from scipy.sparse import csr_matrix
from typing import List
import click
import logging

try:
    from .score_test_io import get_trait_names, get_trait_groups, save_trait_groups, load_trait_data, load_variant_data, save_trait_data, load_gene_table
    from .score_test import TraitData, get_block_boundaries
    from .genesets import gene_variant_matrix
except ImportError:
    from score_test_io import get_trait_names, get_trait_groups, save_trait_groups, load_trait_data, load_variant_data, save_trait_data, load_gene_table
    from score_test import TraitData, get_block_boundaries
    from genesets import gene_variant_matrix

def convert_variant_to_gene_scores(trait_data: TraitData,
                                   G: csr_matrix,
                                   gene_table: pl.DataFrame,
                                   ) -> TraitData:
    """Convert variant-level scores and annotations to gene level scores."""
    assert G.shape[0] == len(trait_data.df) == len(G.indptr) - 1
    
    gradient = trait_data.df['gradient'].to_numpy() @ G
    
    variant_block_boundaries = get_block_boundaries(trait_data.df['jackknife_blocks'].to_numpy())
    gene_jk_blocks = np.zeros(G.shape[1], dtype=int)  # Number of genes
    for i in range(1, len(variant_block_boundaries)-1):
        gene_idx = G.indices[G.indptr[variant_block_boundaries[i]]]
        gene_jk_blocks[gene_idx:] = i

    assert np.all(np.sort(gene_jk_blocks) == gene_jk_blocks)

    # Create DataFrame with only the columns we need for gene-level analysis
    # POS is the midpoint for genes
    df = pl.DataFrame({
        'CHR': gene_table['CHR'],
        'POS': gene_table['midpoint'].cast(pl.Int64),
        'gene_id': gene_table['gene_id'],
        'gene_name': gene_table['gene_name'],
        'jackknife_blocks': gene_jk_blocks,
        'gradient': gradient,
    })

    return TraitData(df, key='gene_id')

def convert_hdf5(variant_stats_hdf5: str,
                 gene_stats_hdf5: str,
                 genes_file: str,
                 nearest_weights: List[float],
                 ):
    variant_table: pl.DataFrame = load_variant_data(variant_stats_hdf5)
    
    # Only load genes from chromosomes present in variant data
    chromosomes = variant_table['CHR'].unique().sort().to_list()
    gene_table: pl.DataFrame = load_gene_table(genes_file, chromosomes=chromosomes)
    
    G = gene_variant_matrix(variant_table, gene_table, np.array(nearest_weights))
    trait_names = get_trait_names(variant_stats_hdf5)
    
    # Convert each trait
    for name in trait_names:
        trait_data: TraitData = load_trait_data(variant_stats_hdf5, trait_name=name, variant_table=variant_table)
        td = convert_variant_to_gene_scores(trait_data, G, gene_table)
        save_trait_data(td, gene_stats_hdf5, trait_name=name)
    
    # Preserve trait groups from source file
    groups = get_trait_groups(variant_stats_hdf5)
    if groups:
        save_trait_groups(gene_stats_hdf5, groups)


@click.command()
@click.argument('variant_stats_hdf5', type=click.Path(exists=True))
@click.argument('gene_stats_hdf5', type=click.Path())
@click.option('--gene-table', default='data/genes.tsv', type=click.Path(exists=True),
              help="Path to gene table TSV file")
@click.option('--nearest-weights', default='0.3,0.2,0.1,0.1,0.1,0.04,0.04,0.04,0.04,0.04',
              help="Comma-separated weights for k-nearest genes")
@click.option('-v', '--verbose', is_flag=True,
              help='Enable verbose output')
def main(variant_stats_hdf5, gene_stats_hdf5, gene_table, nearest_weights, verbose):
    """Convert variant-level statistics to gene-level statistics."""
    log_level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    weights = [float(w) for w in nearest_weights.split(',')]
    convert_hdf5(variant_stats_hdf5, gene_stats_hdf5, gene_table, weights)
    logging.info("Conversion complete!")


if __name__ == '__main__':
    main()