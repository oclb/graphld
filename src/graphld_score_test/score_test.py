#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "click",
#     "h5py",
#     "numpy",
#     "polars",
#     "scipy",
# ]
# ///

#-------------------------------------------------------
# Score test to evaluate the significance of
# new annotations
#-------------------------------------------------------
import logging
import sys
import time
from typing import List, Tuple

import click
import h5py
import numpy as np
import polars as pl

from score_test_io import (
    load_variant_data,
    load_trait_data,
    load_variant_annotations,
    load_gene_annotations,
    create_random_gene_annotations,
    create_random_variant_annotations,
)


def _get_block_boundaries(blocks: np.ndarray) -> np.ndarray:
    temp = np.where(np.diff(blocks) != 0)[0]
    num_blocks = len(temp) + 1
    block_boundaries = np.zeros(num_blocks + 1, dtype=int)
    block_boundaries[1:-1] = temp
    block_boundaries[-1] = len(blocks)
    return block_boundaries


def _parse_probs(probs_str: str) -> List[float]:
    """Parse comma-separated probabilities and validate they are between 0 and 1."""
    probs = [float(p) for p in probs_str.split(',')]
    for p in probs:
        if p < 0 or p > 1:
            raise ValueError(f"Probability must be between 0 and 1, got {p}")
    return probs


def _project_out(y: np.ndarray, x: np.ndarray):
    """Projects out x from y in place."""
    beta = np.linalg.solve(x.T @ x, x.T @ y )
    y -= x @ beta


def run_score_test(df_snp: pl.DataFrame,
    df_annot: pl.DataFrame,
    params: np.ndarray,
    jk_params: np.ndarray,
    annot_test_list: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run score test for hypothesis testing of new annotation or functional category.

    Args:
        df_snp: DataFrame containing variant statistics (must have RSID column)
        df_annot: DataFrame containing annotations (must have SNP column)
        params: Parameter values from the null model
        jk_params: Jackknife estimates for the null model
        annot_test_list: List of annotations to test

    Returns:
        Tuple of (point_estimates, jackknife_estimates)
    """
    # Merge variant statistics with annotations
    df_merged = df_snp.join(
        df_annot,
        left_on='RSID',
        right_on='SNP',
        how='inner',
        maintain_order='left'
    )
    block_boundaries = _get_block_boundaries(df_merged['jackknife_blocks'].to_numpy())
    noBlocks = len(block_boundaries) - 1

    grad = df_merged['gradient'].to_numpy().astype(np.float64)
    test_annot = df_merged[annot_test_list].to_numpy().astype(np.float64)
    model_annot = df_merged['annotations'].to_numpy().astype(np.float64)
    hess = df_merged['hessian'].to_numpy().astype(np.float64)

    _project_out(test_annot, model_annot)

    # Compute single-block derivatives
    U_block = [] # Derivative of the log-likelihood for each test annotation
    J_block = [] # Derivative of U_block w.r.t. each model parameter
    for i in range(noBlocks):
        block = range(block_boundaries[i], block_boundaries[i+1])
        Ui = grad[block].reshape(1,-1) @ test_annot[block, :]
        U_block.append(Ui)
        Ji = model_annot[block, :].T @ (hess[block].reshape(-1,1) * test_annot[block, :])
        J_block.append(Ji)
    U_total = sum(U_block)
    J_total = sum(J_block)

    # Compute leave-one-out derivatives
    U_jackknife = np.zeros((noBlocks, U_total.shape[1]))
    for i in range(noBlocks):
        block = range(block_boundaries[i], block_boundaries[i+1])

        # deduct the score contributed from one LD block
        U_jackknife[i, :] = U_total - U_block[i].ravel()

        # correct for the change in parameters when leaving out this block
        Ji = J_total - J_block[i]
        U_jackknife[i, :] += ((jk_params[i,:] - params).reshape(1,-1) @ Ji).ravel()

    return U_total, U_jackknife


def _setup_logging(output_fp: str | None, verbose: bool):
    """Set up logging configuration."""
    log_format = '%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'

    handlers = []
    if output_fp:
        log_fp = f"{output_fp}_scoretest.log"
        handlers.append(logging.FileHandler(log_fp))
    if verbose:
        handlers.append(logging.StreamHandler(sys.stdout))
    if not handlers:
        handlers.append(logging.NullHandler())

    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        datefmt=date_format,
        handlers=handlers
    )


@click.command()
@click.argument('variant_stats_hdf5', type=click.Path(exists=True))
@click.option('-o', '--output', 'output_fp', default=None,
              help="Output file path prefix (e.g., 'results/my_test'). '.txt' and '.log' will be appended.")
@click.option('-a', '--variant-annot-dir', 'variant_annot_dir', type=click.Path(exists=True),
              help="Directory containing variant-level annotation files (.annot).")
@click.option('-g', '--gene-annot-dir', 'gene_annot_dir', type=click.Path(exists=True),
              help="Directory containing gene-level annotations to convert to variant-level.")
@click.option('--random-genes', 'random_genes',
              help="Comma-separated probabilities (0-1) for random gene-level annotations (e.g., '0.1,0.01').")
@click.option('--random-variants', 'random_variants',
              help="Comma-separated probabilities (0-1) for random variant-level annotations (e.g., '0.1,0.01').")
@click.option('--gene-table', default='data/genes.tsv', type=click.Path(exists=True),
              help="Path to gene table TSV file (required for gene-level options).")
@click.option('--nearest-weights', default='0.5,0.2,0.1,0.1,0.1',
              help="Comma-separated weights for k-nearest genes (for gene-level options).")
@click.option('--annotations',
              help="Optional comma-separated list of specific annotation names to test.")
@click.option('-n', '--name', 'trait_name',
              help="Specific trait name to process from HDF5 file. If omitted, all traits are processed.")
@click.option('-v', '--verbose', is_flag=True,
              help='Enable verbose output (log messages and results to console).')
@click.option('--seed', type=int, default=None,
              help='Seed for generating random annotations.')
def main(variant_stats_hdf5, output_fp, variant_annot_dir, gene_annot_dir, random_genes, 
         random_variants, gene_table, nearest_weights, annotations, trait_name, verbose, seed):
    """Run score test for annotation enrichment."""
    
    _setup_logging(output_fp, verbose)
    start_time = time.time()

    # Set random seed if provided
    if seed:
        np.random.seed(seed)

    # Check that exactly one annotation source is specified
    num_provided = sum([
        variant_annot_dir is not None,
        gene_annot_dir is not None,
        random_genes is not None,
        random_variants is not None
    ])
    if num_provided != 1:
        raise click.UsageError("Must specify exactly one of: --variant-annot-dir, --gene-annot-dir, --random-genes, --random-variants")

    # Parse annotation names if provided
    annot_names_filter = [a.strip() for a in annotations.split(',')] if annotations else None

    # Load variant data
    logging.info(f"Loading variant data from {variant_stats_hdf5}")
    variant_data = load_variant_data(variant_stats_hdf5)
    logging.info(f"Loaded {len(variant_data)} variants")
    
    # Load annotations based on source type
    weights = np.array([float(w) for w in nearest_weights.split(',')], dtype=np.float64)
    
    if variant_annot_dir:
        logging.info(f"Loading variant annotations from {variant_annot_dir}")
        df_annot, annot_names = load_variant_annotations(variant_annot_dir, annot_names_filter)
    elif gene_annot_dir:
        logging.info(f"Loading gene annotations from {gene_annot_dir}")
        df_annot, annot_names = load_gene_annotations(
            gene_annot_dir, variant_data, gene_table, weights
        )
    elif random_genes:
        logging.info(f"Creating random gene annotations with probabilities: {random_genes}")
        probs = _parse_probs(random_genes)
        df_annot, annot_names = create_random_gene_annotations(
            variant_data, gene_table, weights, probs
        )
    else:  # random_variants
        logging.info(f"Creating random variant annotations with probabilities: {random_variants}")
        probs = _parse_probs(random_variants)
        df_annot, annot_names = create_random_variant_annotations(
            variant_data, probs
        )
    
    logging.info(f"Testing {len(annot_names)} annotation(s): {', '.join(annot_names)}")
    
    with h5py.File(variant_stats_hdf5, 'r') as f:
        trait_names = [trait_name] if trait_name else list(f['traits'].keys())
    
    logging.info(f"Processing {len(trait_names)} trait(s)")
    run_meta_analysis = len(trait_names) > 1 and 'meta_analysis' not in trait_names

    # Run the score test
    results_dict = {'annotation' : annot_names}
    summed_scores = 0
    summed_jackknife_scores = 0
    for trait in trait_names:
        logging.info(f"Running score test for trait: {trait}")
        trait_data = load_trait_data(variant_stats_hdf5, trait_name=trait)
        variant_data_with_grad = variant_data.with_columns(
            pl.Series(name='gradient', values=trait_data['gradient']),
            pl.Series(name='hessian', values=trait_data['hessian']),
        )

        point_estimates, jackknife_estimates = run_score_test(
            df_snp=variant_data_with_grad,
            df_annot=df_annot,
            params=trait_data['parameters'],
            jk_params=trait_data['jackknife_parameters'],
            annot_test_list=annot_names,
        )
        summed_scores += point_estimates
        summed_jackknife_scores += jackknife_estimates

        # Compute Z-scores
        std_dev = np.std(jackknife_estimates, axis=0)
        n = jackknife_estimates.shape[0] - 1
        z_scores = point_estimates.ravel() / std_dev / np.sqrt(n)
        results_dict[trait] = z_scores

    if run_meta_analysis:
        logging.info("Computing meta-analysis across traits")
        std_dev = np.std(summed_jackknife_scores, axis=0)
        n = summed_jackknife_scores.shape[0] - 1
        results_dict['meta_analysis'] = summed_scores.ravel() / std_dev / np.sqrt(n)

    results_df = pl.DataFrame(results_dict)
    
    # Print results to console if no output file or verbose mode
    if not output_fp or verbose:
        print(results_df)

    # Write results to file if output path provided
    if output_fp:
        results_df.write_csv(output_fp + ".txt", separator='\t')
        logging.info(f'Results written to {output_fp}.txt')

    logging.info(f'Total time: {time.time()-start_time:.2f}s')


if __name__ == '__main__':
    main()
