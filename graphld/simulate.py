"""
Simulate summary statistics from specified prior distribution.
"""

from dataclasses import dataclass
from typing import Callable, List, Optional, Union

import numpy as np
import polars as pl

from graphld import PrecisionOperator


def default_link_fn(x: np.ndarray) -> np.ndarray:
    """Default link function mapping annotations to relative per-variant heritability."""
    # Assume first column is allele frequency, rest are annotations
    # Apply sigmoid to each annotation and take the mean
    annotations = x[:, 1:]  # Skip AF column
    return np.log(1 + np.exp(np.sum(annotations, axis=1)))


@dataclass
class Simulate:
    """
    Class for simulating summary statistics from LDGM precision matrices.

    This is a Python implementation of MATLAB's simulateSumstats.m, simplified to:
    1. Take PrecisionOperator instances directly instead of raw matrices
    2. Support only a single ancestry group
    3. Assume annotations are already merged with precision matrices

    Attributes:
        sample_size: Sample size for the population
        heritability: Total heritability (h2) for the trait
        component_variance: Per-allele effect size variance for each mixture component
        component_weight: Mixture weight for each component (must sum to ≤ 1)
        alpha_param: Alpha parameter for allele frequency-dependent architecture
        annotation_dependent_polygenicity: If True, use annotations to modify proportion
            of causal variants instead of effect size magnitude
        link_fn: Function mapping annotation vector to relative per-variant heritability.
            Default is softmax: x -> log(1 + exp(x))
        component_random_seed: Random seed for component assignments
    """
    sample_size: int
    heritability: float = 1.0
    component_variance: Union[np.ndarray, List[float]] = None  # Will default to [1.0]
    component_weight: Union[np.ndarray, List[float]] = None    # Will default to [1.0]
    alpha_param: float = -1
    annotation_dependent_polygenicity: bool = False
    link_fn: Callable[[np.ndarray], np.ndarray] = default_link_fn
    component_random_seed: Optional[int] = None

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

    def simulate(self, precision_ops: List[PrecisionOperator]) -> List[pl.DataFrame]:
        """Simulate summary statistics for a list of precision matrices.

        Args:
            precision_ops: List of PrecisionOperator instances, each representing an LD block

        Returns:
            List of DataFrames containing simulated summary statistics for each block
        """
        # Get dimensions
        n_components = len(self.component_weight)

        # Initialize output containers
        sumstats = []
        true_beta_perallele = []
        true_beta_persd = []
        true_alpha_persd = []
        per_variant_h2 = []  # Track per-variant heritability for enrichment calculations

        # Total probability of being causal
        total_p_causal = np.sum(self.component_weight)

        # First pass: simulate betas for all blocks to compute total variance
        all_betas = []
        all_indices = []

        # Set random seed if specified
        if self.component_random_seed is not None:
            np.random.seed(self.component_random_seed)

        # Simulate for each block
        for op in precision_ops:
            # Get indices and dimensions for this block
            indices = op._which_indices if op._which_indices is not None else np.arange(op.shape[0])
            n_variants = len(indices)
            all_indices.append(indices)

            # Get variant info and annotations
            variant_info = op.variant_info.filter(pl.col('index').is_in(indices))
            numeric_cols = ['af', 'annotation1', 'annotation2']  # Only use numeric columns
            annotations = variant_info.select(numeric_cols).to_numpy()

            if self.annotation_dependent_polygenicity:
                p_causal = self.link_fn(annotations)
                # Scale probabilities to achieve desired total probability
                scaling_factor = total_p_causal * n_variants / np.sum(p_causal)
                p_causal = np.minimum(1, p_causal * scaling_factor)
                block_per_variant_h2 = p_causal
            else:
                # Use fixed probability for each variant
                p_causal = np.full(n_variants, total_p_causal)
                block_per_variant_h2 = self.link_fn(annotations)

            # Assign variants to components
            which_causal = np.random.rand(n_variants) < p_causal
            which_component = -np.ones(n_variants, dtype=int)
            n_causal = np.sum(which_causal)
            if n_causal > 0:
                # Normalize component weights to sum to 1
                norm_weights = self.component_weight / np.sum(self.component_weight)
                which_component[which_causal] = np.random.choice(
                    n_components, size=n_causal, p=norm_weights
                )

            # Sample effect sizes
            beta = np.zeros(n_variants)
            for comp in range(n_components):
                mask = which_component == comp
                if np.any(mask):
                    beta[mask] = np.random.normal(
                        0, np.sqrt(self.component_variance[comp]), size=np.sum(mask)
                    )

            # Apply annotation and allele frequency effects
            if not self.annotation_dependent_polygenicity:
                # Scale effect sizes by annotation values
                beta *= np.sqrt(block_per_variant_h2)
                block_per_variant_h2[which_causal] = block_per_variant_h2[which_causal]

            # Apply allele frequency dependent architecture if specified
            if self.alpha_param != -1:
                af = variant_info.select(pl.col('af')).to_numpy().flatten()
                af_factor = np.power(af * (1 - af), 1 + self.alpha_param)
                if self.annotation_dependent_polygenicity:
                    p_causal *= af_factor
                    block_per_variant_h2 *= af_factor
                else:
                    beta *= np.sqrt(af_factor)
                    block_per_variant_h2 *= af_factor

            # Store results for this block
            all_betas.append(beta)
            per_variant_h2.append(block_per_variant_h2)

        # Scale effects to achieve desired total heritability
        total_variance = sum(np.sum(beta**2) for beta in all_betas)
        if total_variance > 0:  # Avoid division by zero
            scaling_factor = np.sqrt(self.heritability / total_variance)
            all_betas = [beta * scaling_factor for beta in all_betas]
            per_variant_h2 = [h2 * scaling_factor**2 for h2 in per_variant_h2]

        # Second pass: generate Z-scores using scaled betas
        for op, indices, beta in zip(precision_ops, all_indices, all_betas, strict=True):
            # Get true marginal effects (alpha) using precision matrix
            alpha = op.solve(beta, method="direct")
            true_alpha_persd.append(alpha)

            # Generate noise using Cholesky factorization
            # For P = L L^T, if z ~ N(0,I), then L^(-1) z ~ N(0,P^-1)
            z = np.random.standard_normal(op.matrix.shape[0])
            noise = op.solve_L(z)  # solve_L now handles float32 conversion

            # Z-scores are true effects * sqrt(N) + noise
            z_scores = alpha * np.sqrt(self.sample_size) + noise

            # Create summary statistics DataFrame
            block_stats = pl.DataFrame({
                'index': indices,
                'Z': z_scores,
                'N': np.full(len(indices), self.sample_size),
                'beta_true': beta
            })

            # Store results
            sumstats.append(block_stats)
            true_beta_perallele.append(beta)
            true_beta_persd.append(beta)  # TODO: Convert to per-SD units

        return sumstats
