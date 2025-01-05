"""
Simulate GWAS summary statistics.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from multiprocessing import Process, Value, Array, cpu_count
import os
import time
import traceback
from itertools import zip_longest

import numpy as np
import polars as pl
from .io import read_ldgm_metadata, partition_variants, load_ldgm
from .multiprocessing import ParallelProcessor, SharedData, WorkerManager
from .precision import PrecisionOperator
from .heritability import softmax_robust

def _default_link_fn(annotations: np.ndarray) -> np.ndarray:
    """Default link function mapping annotations to relative per-variant heritability."""
    from .heritability import softmax_robust
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
    num_indices = ldgm.shape[0]

    print(f"number of variants: {num_variants}")
    print(f"number of indices: {num_indices}")

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

    print(f"fraction of nonzero h2: {np.mean(h2_per_variant>0)}; fraction assigned: {np.mean(variances[component_assignments]>0)}")
    h2_per_variant *= variances[component_assignments]

    # Generate effect sizes for each variant
    beta = np.random.randn(num_variants) * np.sqrt(h2_per_variant)
    alpha = ldgm.variant_solve(beta)

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
    return ldgm.solve_L(white_noise)[ldgm.variant_indices].reshape(-1, 1)


class Simulate(ParallelProcessor, _SimulationSpecification):
    """Parallel processor for simulating GWAS summary statistics."""

    @staticmethod
    def create_shared_memory(metadata: pl.DataFrame, block_data: list[tuple], **kwargs) -> SharedData:
        """Create shared memory arrays for simulation.
        
        Args:
            metadata: Metadata DataFrame containing block information
            block_data: List of block-specific annotation DataFrames
            **kwargs:
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
            return [None] * len(metadata)
            
        # Partition annotations into blocks
        annotation_blocks: list[pl.DataFrame] = partition_variants(metadata, annotations)

        print("\nBlock sizes:")
        for i, df in enumerate(annotation_blocks):
            print(f"Block {i}: {len(df)} variants")

        cumulative_num_variants = np.cumsum(np.array([len(df) for df in annotation_blocks]))
        cumulative_num_variants = [0] + list(cumulative_num_variants[:-1])

        print("\nCumulative offsets:")
        for i, offset in enumerate(cumulative_num_variants):
            print(f"Block {i} starts at: {offset}")

        return list(zip(annotation_blocks, cumulative_num_variants))

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
                pos_col='POS',
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
        noise = _simulate_noise_block(ldgm, random_seed=worker_params.random_seed + variant_offset)

        # Create zero-filled arrays for all variants in sumstats
        beta_reshaped = np.zeros((num_variants, 1))
        alpha_reshaped = np.zeros((num_variants, 1))
        noise_reshaped = np.zeros((num_variants, 1))

        # Fill in values for successfully merged variants
        beta_reshaped[sumstat_indices, 0] = beta
        alpha_reshaped[sumstat_indices, 0] = alpha
        noise_reshaped[sumstat_indices, 0] = noise
        
        # Update the shared memory arrays
        block_slice = slice(variant_offset, variant_offset + num_variants)
        shared_data['beta', block_slice] = beta_reshaped
        shared_data['alpha', block_slice] = alpha_reshaped
        shared_data['noise', block_slice] = noise_reshaped

    @classmethod
    def supervise(cls, manager: WorkerManager, shared_data: Dict[str, Any], block_data: list, **kwargs) -> pl.DataFrame:
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
        spec = kwargs.get('spec')
        if spec is None:
            raise ValueError("spec must be provided in kwargs")
        heritability = spec.heritability
        
        scaling_param = np.sqrt(heritability / np.dot(beta,alpha)) if heritability > 0 else 1
        beta *= scaling_param
        alpha *= scaling_param
        
        # Return results
        return pl.DataFrame({
            'z_scores': noise + alpha,
            'beta': beta,
            'alpha': alpha,
        })

    def simulate(self, 
                ldgm_metadata_path: str,
                populations: Optional[Union[str, List[str]]] = None,
                chromosomes: Optional[Union[int, List[int]]] = None,
                annotations: Optional[pl.DataFrame] = None
                ) -> pl.DataFrame:
        """Simulate GWAS summary statistics for multiple LD blocks.
        
        Args:
            ldgm_metadata_path: Path to metadata CSV file
            populations: Optional list of populations to simulate
            chromosomes: Optional list of chromosomes to simulate
            annotations: Optional DataFrame containing variant annotations
            
        Returns:
            DataFrame containing simulated summary statistics
        """ 
        return self.run(
            ldgm_metadata_path=ldgm_metadata_path,
            populations=populations,
            chromosomes=chromosomes,
            worker_params=self,  # Use instance itself as spec since it inherits from _SimulationSpecification
            spec=self,  # Make spec available to supervisor
            annotations=annotations  # Pass annotations to prepare_block_data
        )
