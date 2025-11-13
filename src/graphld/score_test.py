#!/usr/bin/python
# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "h5py",
#     "numpy",
#     "polars",
# ]
# ///

#-------------------------------------------------------
# Score test to evaluate the significance of
# new annotations

#-------------------------------------------------------
import importlib.util
import logging
import os
import sys
import time
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import click
import h5py
import numpy as np
import polars as pl

SPECIAL_COLNAMES = {'SNP', 'CM', 'BP', 'CHR', 'POS', 'A1', 'A2', 'RSID', 'jackknife_blocks'}

# Import load_annotations function without triggering __init__.py,
# such that project dependencies are not required
io_path = Path(__file__).resolve().parent / "io.py"
spec = importlib.util.spec_from_file_location("graphld.io", io_path)
graphld_io = importlib.util.module_from_spec(spec)
spec.loader.exec_module(graphld_io)
load_annotations: Callable = graphld_io.load_annotations

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
    left_key: str,
    right_key: str,
    return_jackknife: bool = False,
) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
    """
    Run score test for hypothesis testing of new annotation or functional category.

    Args:
        df_snp: DataFrame containing variant or gene-level statistics
        df_annot: DataFrame containing annotations
        params: Parameter values from the null model
        jk_params: Jackknife estimates for the null model
        annot_test_list: List of annotations to test
        left_key: Column name in df_snp to merge on
        right_key: Column name in df_annot to merge on
        return_jackknife: If True, return (point_estimates, jackknife_estimates) instead of Z-scores

    Returns:
        Z score for enrichment of each annotation, or (point_estimates, jackknife_estimates) if return_jackknife=True
    """
    # Merge variant statistics with annotations
    df_merged = df_snp.join(
        df_annot,
        left_on=left_key,
        right_on=right_key,
        how='inner',
        maintain_order='left'
    )
    block_boundaries = _get_block_boundaries(df_merged['jackknife_blocks'].to_numpy())
    noBlocks = len(block_boundaries) - 1
    grad = df_merged['gradient'].to_numpy().astype(np.float64)
    test_annot = df_merged[annot_test_list].to_numpy().astype(np.float64)
    model_annot = df_merged['annotations'].to_numpy().astype(np.float64)
    hess = df_merged['hessian'].to_numpy().astype(np.float64)

    # Project out model annotations from test annotations
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

    # Return raw estimates or Z-scores
    if return_jackknife:
        return U_total, U_jackknife
    else:
        jack_mean = U_total.ravel()
        jack_se = np.std(U_jackknife, axis=0) * np.sqrt(U_jackknife.shape[0]-1)
        return jack_mean / jack_se


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


def _load_variant_annotations(annot_dir: str, annot_names: list[str] | None = None) -> Tuple[pl.DataFrame, List[str]]:
    """Load variant-level annotations from directory."""
    df_annot = load_annotations(annot_dir, add_positions=False)
    
    if annot_names:
        available = [col for col in df_annot.columns if col not in ['CHR', 'BP', 'SNP', 'CM']]
        annot_names = [name for name in annot_names if name in available]
    else:
        annot_names = [col for col in df_annot.columns if col not in ['CHR', 'BP', 'SNP', 'CM']]
    
    return df_annot, annot_names


def _load_gene_annotations(gene_annot_dir: str, variant_data: pl.DataFrame, gene_table_path: str, 
                           nearest_weights: np.ndarray) -> Tuple[pl.DataFrame, List[str]]:
    """Load gene-level annotations and convert to variant-level."""
    from graphld.genesets import (load_gene_table, load_gene_sets_from_gmt, 
                                  convert_gene_sets_to_variant_annotations)
    
    chromosomes = variant_data['CHR'].unique().sort().to_list()
    gene_table = load_gene_table(gene_table_path, chromosomes)
    gene_sets = load_gene_sets_from_gmt(gene_annot_dir)
    df_annot = convert_gene_sets_to_variant_annotations(
        gene_sets, variant_data, gene_table, nearest_weights
    )
    annot_columns = list(gene_sets.keys())
    
    return df_annot, annot_columns


def _create_random_gene_annotations(variant_data: pl.DataFrame, gene_table_path: str, 
                                   nearest_weights: np.ndarray, probs: List[float]) -> Tuple[pl.DataFrame, List[str]]:
    """Create random gene-level annotations and convert to variant-level."""
    from graphld.genesets import (load_gene_table, convert_gene_sets_to_variant_annotations)
    
    chromosomes = variant_data['CHR'].unique().sort().to_list()
    gene_table = load_gene_table(gene_table_path, chromosomes)
    
    # Create random gene sets
    import numpy as np
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


def _create_random_variant_annotations(variant_data: pl.DataFrame, probs: List[float]) -> Tuple[pl.DataFrame, List[str]]:
    """Create random variant-level annotations."""
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
        df_annot, annot_names = _load_variant_annotations(variant_annot_dir, annot_names_filter)
    elif gene_annot_dir:
        logging.info(f"Loading gene annotations from {gene_annot_dir}")
        df_annot, annot_names = _load_gene_annotations(
            gene_annot_dir, variant_data, gene_table, weights
        )
    elif random_genes:
        logging.info(f"Creating random gene annotations with probabilities: {random_genes}")
        probs = _parse_probs(random_genes)
        df_annot, annot_names = _create_random_gene_annotations(
            variant_data, gene_table, weights, probs
        )
    else:  # random_variants
        logging.info(f"Creating random variant annotations with probabilities: {random_variants}")
        probs = _parse_probs(random_variants)
        df_annot, annot_names = _create_random_variant_annotations(
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
            left_key='RSID',
            right_key='SNP',
            return_jackknife=True,
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
