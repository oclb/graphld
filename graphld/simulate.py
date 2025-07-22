"""
Simulate GWAS summary statistics.
"""

import os
from dataclasses import dataclass
from multiprocessing import Value
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import polars as pl

from .heritability import softmax_robust
from .io import partition_variants
from .multiprocessing_template import ParallelProcessor, SharedData, WorkerManager
from .precision import PrecisionOperator


def _default_link_fn(annotations: np.ndarray) -> np.ndarray:
    """Default link function mapping annotations to relative per-variant heritability."""
    return softmax_robust(np.sum(annotations, axis=1))

@dataclass
class _SimulationSpecification:
    """
    Holds parameters for simulating summary statistics from LDGM precision matrices.

    Attributes:
        sample_size: Sample size for the population
        heritability: Total heritability (h2) for the trait
        component_variance: Per-allele effect size variance for each mixture component
        component_weight: Mixture weight for each component (must sum to ≤ 1)
        alpha_param: Alpha parameter for allele frequency-dependent architecture
        annotation_dependent_polygenicity: If True, use annotations to modify proportion
            of causal variants instead of effect size magnitude
        link_fn: Function mapping annotation vector to relative per-variant heritability.
            Default is softmax: x -> log(1 + exp(sum(x)))
        component_random_seed: Random seed for component assignments
        annotation_columns: List of column names to use as annotations. Annotations are
            expected to be in the LDGM variant_info DataFrame
    """
    sample_size: int
    heritability: float = 0.5
    component_variance: Union[np.ndarray, List[float]] = None  # Will default to [1.0]
    component_weight: Union[np.ndarray, List[float]] = None    # Will default to [1.0]
    alpha_param: float = -1
    annotation_dependent_polygenicity: bool = False
    link_fn: Callable[[np.ndarray], np.ndarray] = _default_link_fn
    random_seed: Optional[int] = None
    annotation_columns: Optional[List[str]] = None


    def __post_init__(self):
        """Initialize default values and validate inputs."""
        if self.component_variance is None:
            self.component_variance = np.array([1.0])
        if self.component_weight is None:
            self.component_weight = np.array([1.0])

        # Validate inputs
        if not isinstance(self.component_variance, np.ndarray):
            self.component_variance = np.array(self.component_variance)
        if not isinstance(self.component_weight, np.ndarray):
            self.component_weight = np.array(self.component_weight)

        assert np.all(self.component_variance >= 0), "Component variances must be non-negative"
        assert np.all(self.component_weight >= 0), "Component weights must be non-negative"
        assert np.sum(self.component_weight) <= 1, "Component weights must sum to at most 1"


def _simulate_beta_block(ldgm: PrecisionOperator,
                        spec: _SimulationSpecification,
                        ) -> tuple[np.ndarray, np.ndarray]:
    """Simulate effect sizes for a single LD block.

    Args:
        ldgm: PrecisionOperator instance for this block
        spec: Simulation parameters
        block_data: Optional DataFrame containing variant annotations for this block

    Returns:
        Tuple of causal effect sizes, marginal effect sizes
    """
    if spec.annotation_dependent_polygenicity:  # TODO
        raise NotImplementedError

    # Use annotations from LDGM
    annotation_columns = spec.annotation_columns or ['af']
    annotations = ldgm.variant_info.select(annotation_columns).to_numpy()

    af = ldgm.variant_info.select("af").to_numpy().reshape(-1)  # Ensure 1D

    # variants and indices differ because there can be multiple variants per index
    num_variants = len(af)

    # Sample components
    weights = spec.component_weight
    variances = spec.component_variance

    # Normalize weights to sum to 1 (add null component)
    total_weight = np.sum(weights)
    if total_weight > 1:
        raise ValueError(f"Component weights sum to {total_weight} > 1")
    weights = np.append(weights, [1 - total_weight])  # Add null component
    variances = np.append(variances, [0])  # Zero variance for null component

    if spec.random_seed is not None:
        np.random.seed(spec.random_seed)
    component_assignments = np.random.choice(
        len(variances), num_variants, p=weights
    )

    # Compute per-variant heritabilities
    h2_per_variant = spec.link_fn(annotations)

    # Calculate allele frequency term
    af_term = 2 * af * (1 - af)
    af_term = af_term ** (1 + spec.alpha_param)

    h2_per_variant *= af_term

    h2_per_variant *= variances[component_assignments]

    # Generate effect sizes for each variant
    beta = np.random.randn(num_variants) * np.sqrt(h2_per_variant)
    alpha = ldgm.variant_solve(beta)

    assert np.sum(np.isnan(alpha)) == 0, 'Simulated NaN effect sizes'

    return beta.reshape(-1, 1), alpha.reshape(-1, 1)


def _simulate_noise_block(ldgm: PrecisionOperator,
                   random_seed: int
                   ) -> np.ndarray:
    """Simulate noise vector for a single LD block.

    Args:
        ldgm: PrecisionOperator instance for this block
        random_seed: Random seed for component assignments

    Returns:
        Noise vector
    """

    # Generate noise with variance equal to the LD matrix
    np.random.seed(random_seed)
    white_noise = np.random.randn(ldgm.shape[0])
    return ldgm.solve_Lt(white_noise)[ldgm.variant_indices].reshape(-1, 1)


def _create_block_annotations(
    metadata: pl.DataFrame,
    ldgm_metadata_path: str
) -> list[pl.DataFrame]:
    """Create block-specific annotation DataFrames from SNP lists.

    Args:
        metadata: DataFrame containing LDGM metadata
        ldgm_metadata_path: Path to LDGM metadata file

    Returns:
        List of DataFrames containing CHR and POS columns for each block
    """
    block_annotations = []
    for row in metadata.iter_rows(named=True):
        snplist_path = os.path.join(os.path.dirname(ldgm_metadata_path), row['snplistName'])
        snplist = pl.read_csv(snplist_path, separator=',', has_header=True)
        block_annotations.append(snplist.with_columns([
            pl.lit(int(row['chrom'])).alias('CHR'),
            pl.col('position').alias('POS'),
            pl.col('site_ids').alias('SNP'),
            pl.col('anc_alleles').alias('A2'),
            pl.col('deriv_alleles').alias('A1')
        ]))
    return block_annotations


class Simulate(ParallelProcessor, _SimulationSpecification):
    """Parallel processor for simulating GWAS summary statistics."""

    @staticmethod
    def create_shared_memory(
        metadata: pl.DataFrame,
        block_data: list[tuple],
        **kwargs
    ) -> SharedData:
        """Create shared memory arrays for simulation.

        Args:
            metadata: Metadata DataFrame containing block information
            block_data: List of tuples containing block-specific annotation DataFrames
            **kwargs: Additional keyword arguments
        """
        # Get total number of variants and indices
        num_variants = np.array([len(df) for df, _ in block_data])
        total_variants = int(sum(num_variants))  # Convert to Python int

        # Create shared arrays sized according to metadata
        shared = SharedData({
            'beta': total_variants,    # Causal effect sizes (one per index)
            'alpha': total_variants,   # Marginal effect sizes (one per index)
            'h2': total_variants,      # Per-variant heritability
            'noise': total_variants,   # Noise component
            'scale_param': 1,         # Single value
        })

        # Initialize arrays with zeros
        shared['beta'][:] = 0
        shared['alpha'][:] = 0
        shared['noise'][:] = 0
        shared['h2'][:] = 0
        shared['scale_param'][:] = 0

        return shared

    @classmethod
    def prepare_block_data(cls, metadata: pl.DataFrame, **kwargs) -> list[tuple]:
        """Prepare block-specific data for processing.

        Args:
            metadata: DataFrame containing LDGM metadata
            **kwargs: Additional arguments from run(), including:
                annotations: Optional DataFrame containing variant annotations

        Returns:
            List of block-specific annotation DataFrames, or None if no annotations
        """
        annotations = kwargs.get('annotations')
        if annotations is None:
            block_annotations = _create_block_annotations(
                metadata,
                kwargs.get('ldgm_metadata_path_duplicate')
            )
        else:
            block_annotations = partition_variants(metadata, annotations)

        cumulative_num_variants = np.cumsum(np.array([len(df) for df in block_annotations]))
        cumulative_num_variants = [0] + list(cumulative_num_variants[:-1])

        return list(zip(block_annotations, cumulative_num_variants, strict=False))

    @classmethod
    def process_block(cls, ldgm: PrecisionOperator, flag: Value,
                     shared_data: SharedData, block_offset: int,
                     block_data: Optional[tuple] = None,
                     worker_params: Optional[Dict] = None) -> None:
        """Process a single block."""
        # If we have block_data, merge it with LDGM variant info
        if block_data is not None:
            assert isinstance(block_data, tuple), "block_data must be a tuple"
            annotations, variant_offset = block_data
            num_variants = len(annotations)

            # Merge annotations with LDGM variant info and get indices of merged variants
            from .io import merge_snplists
            ldgm, sumstat_indices = merge_snplists(
                ldgm, annotations,
                match_by_position=True,
                pos_col='BP',
                ref_allele_col='REF',
                alt_allele_col='ALT'
            )
        else:
            variant_offset = block_offset
            num_variants = ldgm.shape[0]
            sumstat_indices = range(num_variants)

        # Get block slice using the number of indices
        block_slice = slice(variant_offset, variant_offset + num_variants)

        # Simulate effect sizes using the merged data
        beta, alpha = _simulate_beta_block(ldgm, worker_params)

        block_random_seed = None if worker_params.random_seed is None \
            else worker_params.random_seed + variant_offset
        noise = _simulate_noise_block(ldgm, random_seed=block_random_seed)

        # Create zero-filled arrays for all variants in sumstats
        beta_reshaped = np.zeros((num_variants, 1))
        alpha_reshaped = np.zeros((num_variants, 1))
        noise_reshaped = np.zeros((num_variants, 1))

        # Fill in values for successfully merged variants
        beta_reshaped[sumstat_indices, 0] = beta.flatten()
        alpha_reshaped[sumstat_indices, 0] = alpha.flatten()
        noise_reshaped[sumstat_indices, 0] = noise.flatten()

        # Update the shared memory arrays
        block_slice = slice(variant_offset, variant_offset + num_variants)
        shared_data['beta', block_slice] = beta_reshaped
        shared_data['alpha', block_slice] = alpha_reshaped
        shared_data['noise', block_slice] = noise_reshaped

    @classmethod
    def supervise(
        cls,
        manager: WorkerManager,
        shared_data: Dict[str, Any],
        block_data: list,
        **kwargs
    ) -> pl.DataFrame:
        """Supervise worker processes and collect results.

        Args:
            manager: Worker manager
            shared_data: Dictionary of shared memory arrays
            **kwargs: Additional arguments

        Returns:
            DataFrame containing simulated summary statistics
        """
        manager.start_workers()
        manager.await_workers()
        beta, alpha, noise = shared_data['beta'], shared_data['alpha'], shared_data['noise']

        # Compute scaling parameter to achieve desired heritability
        spec = kwargs['spec']
        h2 = spec.heritability
        current_h2 = np.dot(beta,alpha)
        assert current_h2 >= 0, "beta'*R*beta should be non-negative"
        scaling_param = np.sqrt(h2 / current_h2) if current_h2 > 0 else 1
        beta *= scaling_param
        alpha *= scaling_param

        # Concatenate block annotations and add simulation results
        result = pl.concat([
            df.select(['CHR', 'SNP', 'POS', 'A1', 'A2'])
            for df, _ in block_data
        ])

        return result.with_columns([
            pl.Series('Z', noise + np.sqrt(spec.sample_size) * alpha),
            pl.Series('beta', beta),
            pl.Series('beta_marginal', alpha),
            pl.Series('N', np.full(len(result), spec.sample_size)),
        ])

    def simulate(
        self,
        ldgm_metadata_path: str = 'data/ldgms/ldgm_metadata.csv',
        populations: Optional[Union[str, List[str]]] = 'EUR',
        chromosomes: Optional[Union[int, List[int]]] = None,
        run_in_serial: bool = False,
        num_processes: Optional[int] = None,
        annotations: Optional[pl.DataFrame] = None,
        verbose: bool = False,
    ) -> pl.DataFrame:
        """Simulate genetic data.

        Args:
            ldgm_metadata_path: Path to LDGM metadata file
            populations: Population(s) to filter
            chromosomes: Chromosome(s) to filter
            run_in_serial: Whether to run in serial mode
            annotations: Optional variant annotations

        Returns:
            Simulated genetic data DataFrame
        """
        run_fn = self.run_serial if run_in_serial else self.run
        result = run_fn(
            ldgm_metadata_path=ldgm_metadata_path,
            populations=populations,
            chromosomes=chromosomes,
            worker_params=self,  # Use instance itself as spec
            spec=self,
            annotations=annotations,
            # Pass path to prepare_block_data
            ldgm_metadata_path_duplicate=ldgm_metadata_path,
            num_processes=num_processes,
        )

        if verbose:
            print(f"Number of variants in summary statistics: {len(result)}")
            nonzero_count = (result['beta'] != 0).sum()
            print(f"Number of variants with nonzero beta: {nonzero_count}")

        return result

def run_simulate(
    sample_size: int,
    heritability: float = 0.5,
    component_variance: Optional[Union[np.ndarray, List[float]]] = None,
    component_weight: Optional[Union[np.ndarray, List[float]]] = None,
    alpha_param: float = -1,
    annotation_dependent_polygenicity: bool = False,
    link_fn: Callable[[np.ndarray], np.ndarray] = _default_link_fn,
    random_seed: Optional[int] = None,
    annotation_columns: Optional[List[str]] = None,
    ldgm_metadata_path: str = 'data/ldgms/ldgm_metadata.csv',
    populations: Optional[Union[str, List[str]]] = 'EUR',
    chromosomes: Optional[Union[int, List[int]]] = None,
    run_in_serial: bool = False,
    num_processes: Optional[int] = None,
    annotations: Optional[pl.DataFrame] = None,
    verbose: bool = False,
) -> pl.DataFrame:
    """Run GWAS summary statistics simulation with specified parameters.

    Args:
        sample_size: Sample size for the population
        heritability: Total heritability (h2) for the trait
        component_variance: Per-allele effect size variance for each mixture component
        component_weight: Mixture weight for each component (must sum to ≤ 1)
        alpha_param: Alpha parameter for allele frequency-dependent architecture
        annotation_dependent_polygenicity: If True, use annotations to modify proportion
            of causal variants instead of effect size magnitude
        link_fn: Function mapping annotation vector to relative per-variant heritability.
            Default is softmax: x -> log(1 + exp(sum(x)))
        random_seed: Random seed for reproducibility
        annotation_columns: List of column names to use as annotations
        ldgm_metadata_path: Path to LDGM metadata file
        populations: Population(s) to filter
        chromosomes: Chromosome(s) to filter
        run_in_serial: Whether to run in serial mode
        num_processes: Number of processes for parallel execution
        annotations: Optional variant annotations DataFrame
        verbose: Whether to print progress information

    Returns:
        DataFrame containing simulated summary statistics
    """
    sim = Simulate(
        sample_size=sample_size,
        heritability=heritability,
        component_variance=component_variance,
        component_weight=component_weight,
        alpha_param=alpha_param,
        annotation_dependent_polygenicity=annotation_dependent_polygenicity,
        link_fn=link_fn,
        random_seed=random_seed,
        annotation_columns=annotation_columns,
    )

    return sim.simulate(
        ldgm_metadata_path=ldgm_metadata_path,
        populations=populations,
        chromosomes=chromosomes,
        run_in_serial=run_in_serial,
        num_processes=num_processes,
        annotations=annotations,
        verbose=verbose,
    )
