#!/usr/bin/env -S uv run --script
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
from dataclasses import dataclass
import logging
import sys
import time
from typing import List, Tuple
import warnings

import click
import numpy as np
import polars as pl

# Suppress numpy runtime warnings globally
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Handle imports when running either as a script or as a package
try:
    from .score_test_io import (
        get_trait_names,
        get_trait_groups,
        is_gene_level_hdf5,
        load_row_data,
        load_gene_table,
        load_trait_data,
        load_variant_annotations,
        load_gene_annotations,
        create_random_gene_annotations,
        create_random_variant_annotations,
    )
    from .genesets import convert_gene_to_variant_annotations
    from .meta_analysis import MetaAnalysis
except ImportError:
    from score_test_io import (
        get_trait_names,
        get_trait_groups,
        is_gene_level_hdf5,
        load_row_data,
        load_gene_table,
        load_trait_data,
        load_variant_annotations,
        load_gene_annotations,
        create_random_gene_annotations,
        create_random_variant_annotations,
    )
    from genesets import convert_gene_to_variant_annotations
    from meta_analysis import MetaAnalysis


def get_block_boundaries(blocks: np.ndarray) -> np.ndarray:
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



@dataclass
class TraitData:
    df: pl.DataFrame  # Must contain 'gradient', optionally 'hessian' and annotation columns
    params: np.ndarray | None = None
    jk_params: np.ndarray | None = None
    annot_names: List[str] = None  # Names of annotation columns in df
    keys: List[str] = None  # Primary column names to use for merging

    @property
    def exclude_cols(self) -> set[str]:
        """Columns to exclude when determining annotation names."""
        base_cols = {'CHR', 'POS', 'jackknife_blocks', 'gradient', 'hessian'}
        # Add key columns
        if self.keys:
            base_cols.update(self.keys)
        return base_cols

class Annot:
    """Base class for annotations that can be merged with TraitData."""
    annot_names: List[str]
    other_key: str

    def __init__(self, annot_names: List[str], other_key: str):
        """
        Args:
            annot_names: List of annotation column names to test
            other_key: Column name to use for merging (e.g., 'SNP', 'gene_id', 'gene_name')
        """
        self.annot_names = annot_names
        self.other_key = other_key

    def merge(self, trait_data: TraitData) -> tuple[pl.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
        """Merge with TraitData and return extracted arrays.

        Returns:
            Tuple of (merged_df, test_annot, model_annot, block_boundaries)
        """
        raise NotImplementedError("Subclasses must implement merge()")


class VariantAnnot(Annot):
    """Variant-level annotations."""
    df: pl.DataFrame

    def __init__(self, df: pl.DataFrame, annot_names: List[str]):
        """
        Args:
            df: DataFrame with variant annotations (must have 'RSID' column)
            annot_names: List of annotation column names to test
        """
        super().__init__(annot_names, other_key='RSID')
        self.df = df

    def perturb(self, fraction: float, seed: int | None = None):
        """Perturb binary annotations."""

        def _perturb_binary_vector(vals: np.ndarray, fraction: float, rng: np.random.Generator) -> np.ndarray:
            p = np.mean(vals)
            assert 0 <= p <= 1, f"p must be between 0 and 1, got {p}"
            perturb_mask = (rng.random(len(vals)) < fraction)
            vals[perturb_mask] = rng.binomial(1, p, np.sum(perturb_mask)).astype(vals.dtype)
            return vals

        rng = np.random.default_rng(seed)
        new_cols = []
        kept_names = []
        for col in self.annot_names:
            vals = self.df[col].to_numpy().copy()

            if np.any(np.isnan(vals)):
                raise ValueError(f"Annotation '{col}' contains NaNs")

            unique = np.unique(vals)
            if np.all(np.isin(unique, [0, 1])):
                vals = _perturb_binary_vector(vals, fraction, rng)
                new_cols.append(pl.Series(col, vals))
                kept_names.append(col)

        if new_cols:
            self.df = self.df.with_columns(new_cols)

        self.annot_names = kept_names

    def merge(self, trait_data: TraitData) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray]:
        """Merge variant annotations with TraitData.

        Returns:
            Tuple of (grad, correction, test_annot, block_boundaries)
            Note: correction is None if hessian is not available
        """
        # Check if trait_data has the required key
        if self.other_key not in trait_data.keys:
            raise ValueError(f"TraitData does not have required key '{self.other_key}'. Available keys: {trait_data.keys}")

        # Verify merge key exists in annotation DataFrame
        if self.other_key not in self.df.columns:
            raise ValueError(f"Merge key '{self.other_key}' not found in annotation DataFrame. "
                           f"Available columns: {self.df.columns}")

        df_merged = trait_data.df.join(
            self.df,
            left_on=self.other_key,
            right_on=self.other_key,
            how='inner',
            maintain_order='left'
        )
        block_boundaries = get_block_boundaries(df_merged['jackknife_blocks'].to_numpy())

        # Extract arrays from merged dataframe
        grad = df_merged['gradient'].to_numpy().astype(np.float64)
        hessian = df_merged['hessian'].to_numpy().astype(np.float64) if 'hessian' in df_merged.columns else None
        model_annot = df_merged[trait_data.annot_names].to_numpy().astype(np.float64) if trait_data.annot_names else None
        test_annot = df_merged[self.annot_names].to_numpy().astype(np.float64)

        return grad, hessian, model_annot, test_annot, block_boundaries


class GeneAnnot(Annot):
    """Gene-level annotations."""
    gene_sets: dict[str, list[str]]

    def __init__(self, gene_sets: dict[str, list[str]]):
        """
        Args:
            gene_sets: Dictionary mapping set names to lists of genes
        """
        self.gene_sets = gene_sets

        # Determine gene key from first gene in first set
        # Import helper function
        try:
            from .genesets import _is_gene_id
        except ImportError:
            from genesets import _is_gene_id

        first_gene = next(iter(next(iter(gene_sets.values()))))
        self.other_key = 'gene_id' if _is_gene_id(first_gene) else 'gene_name'

        super().__init__(list(gene_sets.keys()), self.other_key)

    def merge(self, trait_data: TraitData) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Merge gene annotations with gene-level TraitData.

        Creates one-hot encodings for each gene set and returns unmodified grad/correction
        from the TraitData.

        Returns:
            Tuple of (grad, correction, test_annot, block_boundaries)
        """
        # Check if trait_data has the required key
        if self.other_key not in trait_data.keys:
            raise ValueError(f"TraitData does not have required key '{self.other_key}'. Available keys: {trait_data.keys}")
        # Use the appropriate key for merging (gene_id or gene_name)
        # For gene-level data, we need to match using the same identifier type as the gene sets
        merge_key = self.other_key

        # Verify merge key exists in trait_data
        if merge_key not in trait_data.df.columns:
            raise ValueError(f"Merge key '{merge_key}' not found in trait_data.df")

        # Get gene identifiers from trait_data
        gene_ids = trait_data.df[merge_key].to_list()

        # Create one-hot encoding for each gene set
        # TODO vectorize
        test_annot_dict = {}
        for set_name, gene_list in self.gene_sets.items():
            # Create binary indicator: 1 if gene is in set, 0 otherwise
            gene_set = set(gene_list)
            test_annot_dict[set_name] = [1.0 if gene in gene_set else 0.0 for gene in gene_ids]

        # Create DataFrame with one-hot encodings
        test_annot_df = pl.DataFrame(test_annot_dict)
        test_annot = test_annot_df.to_numpy().astype(np.float64)

        # Extract grad and correction (unmodified from TraitData)
        grad = trait_data.df['gradient'].to_numpy().astype(np.float64)

        # Get block boundaries
        block_boundaries = get_block_boundaries(trait_data.df['jackknife_blocks'].to_numpy())

        return grad, None, None, test_annot, block_boundaries


class GenomeAnnot(Annot):
    """Genome region annotations (from BED files)."""

    def __init__(self):
        """TODO: Implement GenomeAnnot for BED file annotations."""
        raise NotImplementedError("GenomeAnnot not yet implemented")

    def merge(self, trait_data: TraitData) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Merge genome region annotations with TraitData.

        Returns:
            Tuple of (grad, correction, test_annot, block_boundaries)
        """
        # TODO: Implement BED file annotation logic
        raise NotImplementedError("GenomeAnnot.merge() not yet implemented")


def run_score_test(trait_data: TraitData,
    annot: Annot,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run approximate score test for hypothesis testing of new annotation or functional category.

    Unlike run_score_test, this function does not adjust for uncertainty in the fitted model parameters.

    Args:
        trait_data: TraitData object containing variant data with gradients
        annot: Annot object (VariantAnnot or GeneAnnot) containing annotations to test

    Returns:
        Tuple of (point_estimates, jackknife_estimates)
    """
    # Merge trait data with annotations
    grad, _, _, test_annot, block_boundaries = annot.merge(trait_data)

    noBlocks = len(block_boundaries) - 1

    # Compute single-block derivatives
    U_block = [] # Derivative of the log-likelihood for each test annotation
    for i in range(noBlocks):
        block = range(block_boundaries[i], block_boundaries[i+1])
        Ui = grad[block].reshape(1,-1) @ test_annot[block, :]
        U_block.append(Ui)
    U_total = sum(U_block)

    # Compute leave-one-out derivatives
    U_jackknife = np.zeros((noBlocks, U_total.shape[1]))
    for i in range(noBlocks):
        block = range(block_boundaries[i], block_boundaries[i+1])

        # deduct the score contributed from one LD block
        U_jackknife[i, :] = U_total - U_block[i].ravel()

    return U_total, U_jackknife


def _setup_logging(output_fp: str | None, verbose: bool):
    """Set up logging configuration."""
    log_format = '%(levelname)s %(module)s - %(funcName)s: %(message)s'
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


@click.command(context_settings={'help_option_names': ['-h', '--help']})
@click.argument('variant_stats_hdf5', type=click.Path(exists=True))
@click.argument('output_fp', required=False, default=None)
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
@click.option('--nearest-weights', default='0.4,0.2,0.1,0.1,0.1,0.05,0.05',
              help="Comma-separated weights for k-nearest genes (for gene-level options).")
@click.option('--annotations',
              help="Optional comma-separated list of specific annotation names to test.")
@click.option('-n', '--name', 'trait_name',
              help="Specific trait name to process from HDF5 file. If omitted, all traits are processed.")
@click.option('-v', '--verbose', is_flag=True,
              help='Enable verbose output (log messages and results to console).')
@click.option('--seed', type=int, default=None,
              help='Seed for generating random annotations.')
@click.option('--perturb-annot', type=float, default=0,
              help='Fraction of variants to perturb for calibration testing.')
def main(variant_stats_hdf5, output_fp, variant_annot_dir, gene_annot_dir, random_genes,
         random_variants, gene_table, nearest_weights, annotations, trait_name, verbose,
         seed, perturb_annot):
    """Run score test for annotation enrichment."""

    _setup_logging(output_fp, verbose)
    logging.info(f"{time.strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info(f"Command: {' '.join(sys.argv)}")
    start_time = time.time()

    # Set random seed if provided
    if seed:
        np.random.seed(seed)

    # Parse annotation names if provided
    annot_names_filter = [a.strip() for a in annotations.split(',')] if annotations else None

    # Detect if this is gene-level or variant-level data
    is_gene_level = is_gene_level_hdf5(variant_stats_hdf5)
    data_type = "gene" if is_gene_level else "variant"

    # Load row data (variants or genes)
    data_table = load_row_data(variant_stats_hdf5)
    logging.info(f"Loaded {len(data_table)} {data_type}s from {variant_stats_hdf5}")

    # Load annotations based on source type
    weights = np.array([float(w) for w in nearest_weights.split(',')], dtype=np.float64)

    num_provided = 0
    annot = None

    if variant_annot_dir:
        if is_gene_level:
            raise click.UsageError("Cannot use --variant-annot-dir with gene-level HDF5 file")
        annot = load_variant_annotations(variant_annot_dir, annot_names_filter)
        logging.info(f"Loaded {len(annot.annot_names)} variant annotations from {variant_annot_dir}")
        num_provided += 1

    if gene_annot_dir:
        gene_annot: GeneAnnot = load_gene_annotations(
            gene_annot_dir, data_table, gene_table, weights, annot_names_filter
        )
        if is_gene_level:
            # For gene-level data, use gene annotations directly
            annot = gene_annot
            logging.info(f"Loaded {len(annot.annot_names)} gene annotations from {gene_annot_dir}")
        else:
            # For variant-level data, convert gene to variant annotations
            chromosomes = data_table['CHR'].unique().sort().to_list()
            gene_table_df = load_gene_table(gene_table, chromosomes)
            annot = convert_gene_to_variant_annotations(gene_annot, data_table, gene_table_df, weights)
            logging.info(f"Loaded {len(annot.annot_names)} gene annotations from {gene_annot_dir}")
        num_provided += 1

    if random_genes:
        probs = _parse_probs(random_genes)
        gene_annot: GeneAnnot = create_random_gene_annotations(
            data_table, gene_table, probs
        )
        if is_gene_level:
            # For gene-level data, use gene annotations directly
            annot = gene_annot
            logging.info(f"Created {len(annot.annot_names)} random gene annotations")
        else:
            # For variant-level data, convert gene to variant annotations
            chromosomes = data_table['CHR'].unique().sort().to_list()
            gene_table_df = load_gene_table(gene_table, chromosomes)
            annot = convert_gene_to_variant_annotations(gene_annot, data_table, gene_table_df, weights)
            logging.info(f"Created {len(annot.annot_names)} random gene annotations")
        num_provided += 1

    if random_variants:
        if is_gene_level:
            raise click.UsageError("Cannot use --random-variants with gene-level HDF5 file")
        probs = _parse_probs(random_variants)
        annot = create_random_variant_annotations(data_table, probs)
        logging.info(f"Created {len(annot.annot_names)} random variant annotations")
        num_provided += 1

    if num_provided != 1:
        msg = "Must specify exactly one of: --variant-annot-dir, " + \
            "--gene-annot-dir, --random-genes, --random-variants"
        raise click.UsageError(msg)

    # Apply perturbation if requested
    if perturb_annot > 0:
        if isinstance(annot, VariantAnnot):
             logging.info(f"Perturbing annotations with fraction {perturb_annot}")
             annot.perturb(perturb_annot, seed)
        else:
             logging.warning("Perturbation only supported for VariantAnnot")

    # Run the score test
    results_dict = {'annotation' : annot.annot_names}
    trait_names = get_trait_names(variant_stats_hdf5, trait_name)

    # Store results for each trait for meta-analysis
    trait_results = {}

    for trait in trait_names:
        trait_data = load_trait_data(variant_stats_hdf5, trait_name=trait, variant_table=data_table)

        point_estimates, jackknife_estimates = run_score_test(
                trait_data=trait_data,
                annot=annot,
            )

        # Store for meta-analysis
        trait_results[trait] = (point_estimates, jackknife_estimates)

        # Compute Z-scores
        std_dev = np.std(jackknife_estimates, axis=0)
        z_col = f"{trait}_Z"
        z_scores = point_estimates.ravel() / std_dev / np.sqrt(jackknife_estimates.shape[0] - 1)
        results_dict[z_col] = z_scores

    # Load trait groups and perform meta-analyses
    trait_groups = get_trait_groups(variant_stats_hdf5)
    for group_name, group_traits in trait_groups.items():
        # Filter to traits that were actually processed
        group_traits = [t for t in group_traits if t in trait_results]
        meta = MetaAnalysis()
        for trait in group_traits:
            meta.update(*trait_results[trait])
        z_col = f"{group_name}_Z"
        results_dict[z_col] = meta.z_scores.ravel()
        logging.info(f"Computed meta-analysis for group '{group_name}' with {len(group_traits)} traits")

    results_df = pl.DataFrame(results_dict)

    if verbose or not output_fp:
        with pl.Config(tbl_rows=-1, tbl_cols=-1):
            print(results_df)

    if random_genes or random_variants or perturb_annot > 0:
        print("\nRoot mean squared Z-scores:")
        for col in [c for c in results_df.columns if c.endswith('_Z')]:
            print(f"{col}: {np.sqrt(np.mean(results_df[col].to_numpy()**2)):.4f}")

    if output_fp:
        results_df.write_csv(output_fp + ".txt", separator='\t')
        logging.info(f'Results written to {output_fp}.txt')

    logging.info(f'Total time: {time.time()-start_time:.2f}s')



if __name__ == '__main__':
    main()
