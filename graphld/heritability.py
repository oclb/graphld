"""
GraphREML
"""
from multiprocessing import Value, Array
from .multiprocessing import ParallelProcessor, WorkerManager, SharedData
from dataclasses import dataclass, field
from typing import *

import numpy as np
import polars as pl
import scipy.stats as sps

from .precision import PrecisionOperator
from .likelihood import *
from .io import *

FLAGS = {
    'ERROR': -2,
    'SHUTDOWN': -1,
    'FINISHED': 0,
    'COMPUTE_ALL': 1,
    'COMPUTE_LIKELIHOOD_ONLY': 2,
    'INITIALIZE': 3
}

@dataclass
class ModelOptions:
    """Stores model parameters for graphREML.

    Attributes:
        annotation_columns: names of columns to be
            used as annotations
        params: Starting parameter values
        sample_size: GWAS sample size, only needed
            for heritability scaling
        intercept: LDSC intercept or 1
        link_fn_denominator: Scalar denominator for link function. 
            Recommended to be roughly the number of SNPs. Defaults to 6e6,
            roughly the number of common SNPs in Europeans.
    """
    annotation_columns: Optional[List[str]] = None
    params: Optional[np.ndarray] = None
    sample_size: float = 1.0
    intercept: float = 1.0
    link_fn_denominator: float = 6e6

    def __post_init__(self):
        if self.annotation_columns is None:
            self.annotation_columns = ['base']
        if self.params is None:
            self.params = np.zeros((len(self.annotation_columns),1))
        
        assert self.params.ndim == 2

@dataclass
class MethodOptions:
    """Stores method parameters for graphREML.

    Attributes:
        gradient_num_samples: Number of samples for gradient estimation
        match_by_position: Use position/allele instead of RSID
            for merging
        num_iterations: Optimization steps
        convergence_tol: Convergence tolerance (TODO)
        run_serial: Run in serial rather than parallel
        num_processes: If None, autodetect
        verbose: Flag for verbose output
        use_surrogate_markers: Whether to use surrogate markers for missing variants
        trust_region_size: Initial trust region size parameter
        trust_region_rho_lb: Lower bound for trust region ratio
        trust_region_rho_ub: Upper bound for trust region ratio
        trust_region_scalar: Scaling factor for trust region updates
        max_trust_iterations: Maximum number of trust region iterations
        reset_trust_region: Whether to reset trust region size at each iteration
        num_jackknife_blocks: Number of blocks to use for jackknife estimation
        max_chisq_threshold: Maximum allowed chi^2 value in a block. Blocks with chi^2 > threshold are excluded.
    """
    gradient_num_samples: int = 100
    match_by_position: bool = True
    num_iterations: int = 50
    convergence_tol: float = 0.01
    run_serial: bool = False
    num_processes: Optional[int] = None
    verbose: bool = False
    use_surrogate_markers: bool = True
    trust_region_size: float = 1e-1
    trust_region_rho_lb: float = 1e-4
    trust_region_rho_ub: float = .99
    trust_region_scalar: float = 5
    max_trust_iterations: int = 100
    reset_trust_region: bool = False
    num_jackknife_blocks: int = 100
    max_chisq_threshold: Optional[float] = None

def _surrogate_marker(ldgm: PrecisionOperator, missing_index: int, candidates: pl.DataFrame) -> int:
    """Find a surrogate marker for a missing variant.

    Args:
        ldgm: LDGM object
        missing_variant: Variant ID to be imputed
        candidates: Array of candidate variant IDs

    Returns:
        ID of the surrogate marker
    """
    indicator = np.zeros(ldgm.shape[0])
    indicator[missing_index] = 1
    index_correlations = ldgm.solve(indicator)
    candidate_correlations = index_correlations[candidates.select('index').to_numpy()]
    surrogate = np.argmax(candidate_correlations ** 2)

    return candidates.filter(pl.col('surrogate_nr') == surrogate).to_dicts()[0]

def softmax_robust(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax implementation."""
    y = x + np.log1p(np.exp(-x))
    mask = x < 0
    y[mask] = np.log1p(np.exp(x[mask]))
    return y
    
def _get_softmax_link_function(denominator: int) -> tuple[Callable, Callable]:
    """Create softmax link function and its gradient.

    Args:
        denominator: roughly num_snps / num_samples (if using Z scores) or num_snps (if using effect size estimates)

    Returns:
        Tuple containing:
        - Link function mapping (annot, theta) to per-SNP heritabilities
        - Gradient of the link function
    """
    np.seterr(over='ignore')

    def _link_fn(annot: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """Softmax link function."""
        return softmax_robust(annot @ theta) / denominator

    def _link_fn_grad(annot: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """Gradient of softmax link function."""
        x = annot @ theta
        result = annot / denominator / (1 + np.exp(-x))
        mask = x.flatten() < 0
        if mask.any():
            result[mask,:] = (annot[mask,:] * np.exp(x[mask]) / 
                            (1 + np.exp(x[mask])) / denominator)
        return result

    return _link_fn, _link_fn_grad


class GraphREML(ParallelProcessor):

    @classmethod
    def _sum_blocks(cls, array: np.ndarray, shape_each_block: tuple) -> np.ndarray:
        array = array.reshape((-1,*shape_each_block))
        return array.sum(axis=0)

    @classmethod
    def prepare_block_data(cls, metadata: pl.DataFrame, **kwargs) -> list[tuple]:
        """Prepare block-specific data for processing.

        Args:
            metadata: DataFrame containing LDGM metadata
            **kwargs: Additional arguments from run(), including:
                sumstats: DataFrame containing Z scores and variant info, optionally annotations
                annotation_columns: list of column names for the annotations in sumstats
                method: MethodOptions instance containing method parameters

        Returns:
            List of dictionaries containing block-specific data with keys:
                sumstats: DataFrame for this block, or None if max Z² exceeds threshold
                variant_offset: Cumulative number of variants before this block
                block_index: Index of this block
                Pz: Pre-computed precision-premultiplied Z scores for this block
        """
        sumstats: pl.DataFrame = kwargs.get('sumstats')
        method: MethodOptions = kwargs.get('method')
        sumstats_blocks: list[pl.DataFrame] = partition_variants(metadata, sumstats)

        # Filter blocks based on max Z² threshold
        if method.max_chisq_threshold is not None:
            max_z2s = [float(np.nanmax(block.select('Z').to_numpy() ** 2)) for block in sumstats_blocks]
            keep_block = [max_z2 <= method.max_chisq_threshold for max_z2 in max_z2s]
            sumstats_blocks = [block if keep_block 
                else pl.DataFrame([]) for block, max_z2 in zip(sumstats_blocks, max_z2s)]
            if method.verbose and not all(keep_block):
                print(f"{len(sumstats_blocks)-sum(keep_block)} out of {len(sumstats_blocks)} blocks discarded due to\n"
                      f"max chisq threshold of {method.max_chisq_threshold}")
            
        cumulative_num_variants = np.cumsum(np.array([len(df) for df in sumstats_blocks]))
        cumulative_num_variants = [0] + list(cumulative_num_variants[:-1])
        block_indices = list(range(len(sumstats_blocks)))
        block_Pz = [None for _ in block_indices]

        return [
            {
                'sumstats': block,
                'variant_offset': offset,
                'block_index': index,
                'Pz': Pz
            }
            for block, offset, index, Pz 
            in zip(sumstats_blocks, cumulative_num_variants, block_indices, block_Pz)
        ]
    
    @staticmethod
    def create_shared_memory(metadata: pl.DataFrame, block_data: list[tuple], **kwargs) -> SharedData:
        """Create output array.

        Args:
            metadata: Metadata DataFrame containing block information
            block_data: List of block-specific sumstats DataFrames
            **kwargs: Not used
        """
        num_params = kwargs.get("num_params")
        num_blocks = len(metadata)
        num_variants = sum([len(d['sumstats']) for d in block_data if d['sumstats'] is not None])

        result = SharedData({
            'params': num_params,
            'variant_h2': num_variants,
            'likelihood': num_blocks,
            'gradient': num_blocks * num_params,
            'hessian': num_blocks * num_params ** 2,
        })

        return result

    @staticmethod
    def _initialize_block_zscores(ldgm: PrecisionOperator, 
                                annot_df: pl.DataFrame,
                                annotation_columns: List[str],
                                match_by_position: bool,
                                verbose: bool = False
                                ):
        """Initialize Z-scores for a block by merging variants and computing Pz.
        
        Args:
            ldgm: LDGM object
            annot_df: Dataframe containing merged annotations and summary statistics
            annotation_columns: List of annotation column names

        Returns:
            Tuple containing:
            - z: Raw Z-scores after filtering
            - Pz: Z-scores premultiplied by precision matrix
        """
        # Merge annotations with LDGM variant info
        from .io import merge_snplists
        ldgm, annot_indices = merge_snplists(
            ldgm, annot_df,
            match_by_position=match_by_position,
            pos_col='POS',
            ref_allele_col='REF',
            alt_allele_col='ALT',
            add_allelic_cols=['Z'],
            add_cols=annotation_columns,
            modify_in_place=True
        )
        if verbose:
            print(f"Number of variants in sumstats before merging: {len(annot_df)}")
            print(f"Number of variants after merging: {len(ldgm.variant_info)}")
            if len(ldgm.variant_info) == 0:
                print("No variants left after merging annotations and sumstats")

        variant_info = ldgm.variant_info.with_row_index(name="vi_row_nr")
        variant_info_nonmissing = variant_info.filter(pl.col('Z').is_not_null()).with_row_index(name="surrogate_nr")
        variant_info_missing = variant_info.filter(pl.col('Z').is_null())
        indices_arr = np.array(variant_info.get_column('index').to_numpy())
        z_arr = np.array(variant_info.get_column('Z').to_numpy())
        for row in variant_info_missing.to_dicts():
            surrogate_row = _surrogate_marker(ldgm, row['index'], variant_info_nonmissing)
            indices_arr[row['vi_row_nr']] = surrogate_row['index']
            z_arr[row['vi_row_nr']] = surrogate_row['Z']

        ldgm.variant_info = variant_info.with_columns([
            pl.Series('index', indices_arr),
            pl.Series('Z', z_arr),
            pl.Series('annot_indices', annot_indices)
        ]).drop('vi_row_nr')

        if verbose:
            print(f"Number of missing rows assigned surrogates: {len(variant_info_missing)}")
        
        # Keep only first occurrence of each index for Z-scores
        first_index_mask = variant_info.select(pl.col('index').is_first_distinct()).to_numpy().flatten()
        z = ldgm.variant_info.select('Z').filter(first_index_mask).to_numpy()

        # Compute Pz
        Pz = ldgm @ z

        return ldgm, Pz


    @staticmethod
    def _compute_block_likelihood(ldgm: PrecisionOperator,
                           Pz: np.ndarray,
                           annotations: np.ndarray,
                           params: np.ndarray,
                           link_fn_denominator: float,
                           old_variant_h2: float,
                           num_samples: int,
                           likelihood_only: bool,
                           ) -> Tuple[float, Optional[np.ndarray], Optional[np.ndarray], np.ndarray]:
        """Compute likelihood, gradient, and hessian for a single block.

        Args:
            ldgm: LDGM object
            Pz: Z-scores premultiplied by precision matrix
            annotations: Annotation matrix
            params: Model parameters
            num_snps: Total number of SNPs
            old_variant_h2: Previous values of variant_h2, so that ldgm is updated using the difference

        Returns:
            Tuple containing:
            - likelihood: Log likelihood value
            - gradient: Gradient vector
            - hessian: Hessian matrix
            - per_variant_h2: With current parameters, heritability per variant
        """
        link_fn, link_fn_grad = _get_softmax_link_function(link_fn_denominator)

        # Compute change in diag(M), aggregating variants with same index
        per_variant_h2 = link_fn(annotations, params)
        delta_D = np.zeros(ldgm.shape[0])
        np.add.at(delta_D, ldgm.variant_indices, per_variant_h2.flatten() - old_variant_h2.flatten())
        
        # Update diag(M)
        ldgm.update_matrix(delta_D)

        # New log likelihood
        likelihood = gaussian_likelihood(Pz, ldgm)

        if likelihood_only:
            ldgm.del_factor() # To reduce memory usage
            return likelihood, None, None, per_variant_h2

        # Gradient of per-variant h2 wrt parameters
        del_h2_del_a = link_fn_grad(annotations, params)
        
        # Gradient of diag(M)
        del_M_del_a = np.zeros((ldgm.shape[0], params.shape[0]))
        np.add.at(del_M_del_a, ldgm.variant_indices, del_h2_del_a)

        # Gradient of log likelihood
        gradient = gaussian_likelihood_gradient(
            Pz, ldgm, del_M_del_a=del_M_del_a,
            n_samples=num_samples,
        )

        hessian = gaussian_likelihood_hessian(
            Pz, ldgm, del_M_del_a=del_M_del_a
        )

        ldgm.del_factor() # To reduce memory usage

        return likelihood, gradient, hessian, per_variant_h2

    @classmethod
    def process_block(cls, ldgm: PrecisionOperator,
                     flag: Value,
                     shared_data: SharedData,
                     block_offset: int,
                     block_data: Any = None,
                     worker_params: Tuple[ModelOptions, MethodOptions] = None) -> None:
        """Computes likelihood, gradient, and hessian for a single block.
        """
        model_options: ModelOptions
        method_options: MethodOptions
        model_options, method_options = worker_params
        
        sumstats: pl.DataFrame = block_data['sumstats']
        if len(sumstats) == 0:  # Skip blocks that were filtered out
            return
            
        variant_offset: int = block_data['variant_offset']
        block_index: int = block_data['block_index']
        num_annot = len(model_options.annotation_columns)
        
        Pz: np.ndarray
        if flag.value == FLAGS['INITIALIZE']:
            ldgm, Pz = cls._initialize_block_zscores(ldgm, 
                                                    sumstats, 
                                                    model_options.annotation_columns,
                                                    method_options.match_by_position,
                                                    method_options.verbose)
            
            # Work in effect-size as opposed to Z score units
            Pz /= np.sqrt(model_options.sample_size)
            ldgm.times_scalar(1.0 / model_options.sample_size)
            block_data['Pz'] = Pz

            # ldgm is modified in place and re-used in subsequent iterations
            
        else:
            Pz = block_data['Pz']

        annot: np.ndarray = ldgm.variant_info.select(model_options.annotation_columns).to_numpy()
        annot_indices: np.ndarray = ldgm.variant_info.select('annot_indices').to_numpy().flatten()
        max_index: int = np.max(annot_indices) + 1 if len(annot_indices) > 0 else 0
        block_variants: slice = slice(variant_offset, variant_offset + max_index)
        old_variant_h2: np.ndarray = shared_data['variant_h2', block_variants][annot_indices]

        likelihood_only = (flag.value == FLAGS['COMPUTE_LIKELIHOOD_ONLY'])
        likelihood, gradient, hessian, variant_h2 = cls._compute_block_likelihood(
            ldgm=ldgm,
            Pz=Pz,
            annotations=annot,
            params=shared_data['params'].reshape(-1,1),
            link_fn_denominator=model_options.link_fn_denominator,
            old_variant_h2=old_variant_h2,
            num_samples=method_options.gradient_num_samples,
            likelihood_only=likelihood_only
        )

        shared_data['likelihood', block_index] = likelihood
        variant_h2_padded = np.zeros(max_index)
        variant_h2_padded[annot_indices] = variant_h2.flatten()
        shared_data['variant_h2', block_variants] = variant_h2_padded
        if likelihood_only:
            return

        gradient_slice = slice(block_index * num_annot,
                             (block_index + 1) * num_annot)
        shared_data['gradient', gradient_slice] = gradient.flatten()

        hessian_slice = slice(block_index * num_annot**2,
                            (block_index + 1) * num_annot**2)
        shared_data['hessian', hessian_slice] = hessian.flatten()

    @staticmethod
    def _group_blocks(blocks: np.ndarray, num_groups: int) -> np.ndarray:
        """Group blocks into a smaller number of groups.
        
        Args:
            blocks: Array of shape (..., num_blocks, *block_shape)
            num_groups: Number of groups to create
            
        Returns:
            Array of shape (..., num_groups, *block_shape) containing summed blocks
        """
        num_blocks = blocks.shape[-2 if blocks.ndim > 2 else 0]
        num_groups = min(num_groups, num_blocks)
        block_size = num_blocks // num_groups
        remainder = num_blocks % num_groups
        
        # Initialize output array
        block_shape = blocks.shape[:-2] if blocks.ndim > 2 else ()
        group_shape = blocks.shape[-1:] if blocks.ndim > 1 else ()
        grouped = np.zeros(block_shape + (num_groups,) + group_shape)
        
        # Sum blocks within each group
        start_idx = 0
        for i in range(num_groups):
            # Add one extra block to some groups to handle remainder
            extra = 1 if i < remainder else 0
            end_idx = start_idx + block_size + extra
            
            # Sum along the block dimension (-2 if ndim > 2, else 0)
            if blocks.ndim > 2:
                grouped[..., i, :] = blocks[..., start_idx:end_idx, :].sum(axis=-2)
            else:
                grouped[i] = blocks[start_idx:end_idx].sum(axis=0)
            
            start_idx = end_idx
        
        return grouped

    @staticmethod
    def _compute_pseudojackknife(gradient_blocks: np.ndarray, hessian_blocks: np.ndarray, params: np.ndarray) -> np.ndarray:
        """Compute pseudo-jackknife estimates for parameters.
        
        Args:
            gradient_blocks: Array of shape (num_blocks, num_params) containing block-wise gradients
            hessian_blocks: Array of shape (num_blocks, num_params, num_params) containing block-wise Hessians
            params: Array of shape (num_params,) containing parameter values
            
        Returns:
            Array of shape (num_blocks, num_params) containing jackknife parameter estimates
        """
        num_blocks = gradient_blocks.shape[0]
        num_params = params.shape[0]
        
        # Sum blocks to get full gradient and Hessian
        gradient = gradient_blocks.sum(axis=0)  # Shape: (num_params,) - sum across blocks
        hessian = hessian_blocks.sum(axis=0)    # Shape: (num_params, num_params) - sum across blocks
        
        # Initialize output array
        jackknife = np.zeros((num_blocks, num_params))
        
        # Compute jackknife estimate for each block
        for block in range(num_blocks):
            # Remove current block from total gradient and Hessian
            block_gradient = gradient_blocks[block]  # Shape: (num_params,)
            block_hessian = hessian_blocks[block]  # Shape: (num_params, num_params)
            
            # Compute leave-one-out gradient and Hessian
            loo_gradient = gradient - block_gradient
            loo_hessian = hessian - block_hessian + 1e-12 * np.eye(num_params)
            
            # Compute jackknife estimate for this block
            jackknife[block] = params + np.linalg.solve(loo_hessian, loo_gradient)
            
        return jackknife

    @staticmethod
    def _compute_jackknife_heritability(block_data: list, jackknife_params: np.ndarray, model: ModelOptions) -> Tuple[np.ndarray, np.ndarray]:
        """Compute jackknife heritability estimates.
        
        Args:
            block_data: List of dictionaries containing block-specific data
            jackknife_params: Array of shape (num_jk_blocks, num_params) containing jackknife parameter estimates
            model: Model options containing annotation column names
            
        Returns:
            Tuple containing:
            - Array of shape (num_jk_blocks, num_annotations) containing jackknife heritability estimates
            - Array of shape (num_jk_blocks, num_annotations) containing jackknife annotation sums
        """
        num_blocks = len(block_data)
        num_jk = jackknife_params.shape[0]
        num_params = jackknife_params.shape[1]
        num_samples = model.sample_size
        
        # Get link function
        link_fn, _ = _get_softmax_link_function(model.link_fn_denominator)
        
        # Initialize output arrays
        jackknife_h2 = np.zeros((num_jk, num_params))
        jackknife_annot_sums = np.zeros((num_jk, num_params))
        
        # Pre-compute annotations for each block
        block_annotations = []
        for block in range(num_blocks):
            if len(block_data[block]['sumstats']) == 0:
                continue
            annotations = block_data[block]['sumstats'].select(model.annotation_columns).to_numpy()
            block_annotations.append(annotations)
            
        # Compute annotation sums for each block
        total_annot_sums = np.sum([annot.sum(axis=0) for annot in block_annotations], axis=0)
        
        # Compute heritability and annotation sums for each jackknife estimate
        for jk in range(num_jk):
            # Compute per-SNP heritability using jackknife parameters
            for annotations in block_annotations:
                per_snp_h2 = link_fn(annotations, jackknife_params[jk, :])
                # Add to total heritability (reshape per_snp_h2 to column vector)
                jackknife_h2[jk, :] += (per_snp_h2[:, np.newaxis] * annotations).sum(axis=0)
                # Add to annotation sums
                jackknife_annot_sums[jk, :] += annotations.sum(axis=0)
        
        return jackknife_h2, jackknife_annot_sums

    @staticmethod
    def _annotation_heritability(
        variant_h2: np.ndarray, annot: pl.DataFrame, ref_col: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Computes the heritability of each annotation based on the given h2 values.
        """
        annot_mat = annot.to_numpy().T
        annot_h2 = annot_mat @ variant_h2
        annot_size = np.sum(annot_mat, axis=1)
        annot_enrichment = annot_size[ref_col] * annot_h2 / (annot_h2[ref_col] * annot_size)

        return annot_h2, annot_enrichment

    @staticmethod
    def _wald_pvalue(jackknife_estimates: np.ndarray) -> Tuple[float, float, float]:
        """Compute Wald test p-value from jackknife estimates.

        Args:
            jackknife_estimates: Array of jackknife estimates

        Returns:
            Tuple containing:
            - point_estimate: Mean of jackknife estimates
            - standard_error: Standard error computed from jackknife formula
            - p_value: Two-sided p-value from Wald test. Returns 1.0 if all estimates are identical.
        """
        n_blocks = jackknife_estimates.shape[0]
        point_estimate = np.mean(jackknife_estimates)
        
        # Check if all estimates are identical (within numerical precision)
        if np.allclose(jackknife_estimates, jackknife_estimates[0]):
            return point_estimate, 0.0, 1.0
            
        standard_error = np.sqrt((n_blocks - 1) * np.var(jackknife_estimates, ddof=1))
        p_value = 2 * (1 - sps.norm.cdf(np.abs(point_estimate / standard_error)))
        return point_estimate, standard_error, p_value

    @classmethod
    def supervise(cls, manager: WorkerManager, shared_data: SharedData, block_data: list, **kwargs):
        """Runs graphREML.
        Args:
            manager: used to start parallel workers
            shared_data: used to communicate with workers
            block_data: annotation + gwas data passed to workers
            **kwargs: Additional arguments
        """
        
        num_iterations = kwargs.get('num_iterations')
        num_params = kwargs.get('num_params')
        verbose = kwargs.get('verbose')
        method: MethodOptions = kwargs.get('method')
        model: ModelOptions = kwargs.get('model')
        trust_region_lambda = method.trust_region_size
        log_likelihood_history = []
        
        def _trust_region_step(gradient: np.ndarray, hessian: np.ndarray, trust_region_lambda: float) -> np.ndarray:
            """Compute trust region step by solving (H + λD)x = -g.
            """
            hess_mod = hessian + trust_region_lambda * np.diag(np.diag(hessian))
            hess_mod += np.finfo(float).eps * np.eye(len(hessian))
            
            return np.linalg.solve(hess_mod, -gradient)

        for rep in range(num_iterations):
            if verbose:
                print(f"\n\tStarting iteration {rep}...")
            
            # Calculate likelihood, gradient, and hessian for each block
            flag = FLAGS['INITIALIZE'] if rep == 0 else FLAGS['COMPUTE_ALL']
            manager.start_workers(flag)
            manager.await_workers()

            likelihood = cls._sum_blocks(shared_data['likelihood'], (1,))[0]
            gradient = cls._sum_blocks(shared_data['gradient'], (num_params,))
            hessian = cls._sum_blocks(shared_data['hessian'], (num_params,num_params))
            
            old_params = shared_data['params'].copy()
            old_likelihood = likelihood
            
            # Reset trust region size if specified
            if method.reset_trust_region or rep == 0:
                trust_region_lambda = method.trust_region_size
                
            # Trust region optimization loop
            for trust_iter in range(method.max_trust_iterations):
                # Compute proposed step
                step = _trust_region_step(gradient, hessian, trust_region_lambda)
                shared_data['params'] = old_params + step
                
                # Evaluate proposed step
                manager.start_workers(FLAGS['COMPUTE_LIKELIHOOD_ONLY'])
                manager.await_workers()
                new_likelihood = cls._sum_blocks(shared_data['likelihood'], (1,))[0]
                
                # Compute actual vs predicted increase
                actual_increase = new_likelihood - old_likelihood
                predicted_increase = step.T @ gradient + 0.5 * step.T @ (hessian @ step)
                assert predicted_increase > -1e-100, f"Predicted increase must be greater than -epsilon but is {predicted_increase}."
                if verbose:
                    print(f"\tIncrease in log-likelihood: {actual_increase}, predicted increase: {predicted_increase}")
                
                # Check if step is acceptable and update trust region size if needed
                rho = actual_increase / predicted_increase
                if rho < method.trust_region_rho_lb:
                    # Reset trust region size to initial value if its below that
                    trust_region_lambda = max(method.trust_region_size, 
                                            trust_region_lambda * method.trust_region_scalar)
                    shared_data['params'] = old_params  # Revert step
                elif rho > method.trust_region_rho_ub:
                    trust_region_lambda /= method.trust_region_scalar
                    break  # Accept step and continue to next iteration
                else:
                    break  # Accept step with current trust region size
                    
                if trust_iter == method.max_trust_iterations - 1:
                    if verbose:
                        print("Warning: Maximum trust region iterations reached")
                                        
            log_likelihood_history.append(new_likelihood)

            if verbose:
                print(f"Trust region lambda: {trust_region_lambda}")
                if len(log_likelihood_history) >= 2:
                    print(f"Change in likelihood: {log_likelihood_history[-1] - log_likelihood_history[-2]}")
            
            # Check convergence
            if len(log_likelihood_history) >= 3:
                if abs(log_likelihood_history[-1] - log_likelihood_history[-3]) < (2 * method.convergence_tol):
                    break

        # After optimization
        annotations = pl.concat([dict['sumstats'].select(model.annotation_columns) for dict in block_data if len(dict['sumstats']) > 0])
        ref_col = 0 # Maybe TODO
        
        # Get block-wise gradients and Hessians for jackknife
        num_blocks = len(block_data)
        gradient_blocks = shared_data['gradient'].reshape((num_blocks, num_params))
        hessian_blocks = shared_data['hessian'].reshape((num_blocks, num_params, num_params))
        
        # Group blocks for jackknife
        jk_gradient_blocks = cls._group_blocks(gradient_blocks, method.num_jackknife_blocks)
        jk_hessian_blocks = cls._group_blocks(hessian_blocks, method.num_jackknife_blocks)
        
        # Compute jackknife estimates using the grouped blocks
        jackknife_params = cls._compute_pseudojackknife(jk_gradient_blocks, jk_hessian_blocks, shared_data['params'])
        
        # Compute jackknife heritability estimates and standard errors
        jackknife_h2, jackknife_annot_sums = cls._compute_jackknife_heritability(block_data, jackknife_params, model)
        
        # Compute standard errors using jackknife formula: SE = sqrt((n-1) * var(estimates))
        n_blocks = jackknife_params.shape[0]
        params_se = np.sqrt((n_blocks - 1) * np.var(jackknife_params, axis=0, ddof=1))
        h2_se = np.sqrt((n_blocks - 1) * np.var(jackknife_h2, axis=0, ddof=1))
        
        # Compute normalized heritability for each jackknife estimate
        jackknife_h2_normalized = jackknife_h2 / jackknife_annot_sums
        
        # Compute quotient for point estimates and SE
        jackknife_enrichment_quotient = jackknife_h2_normalized / jackknife_h2_normalized[:, [0]]
        enrichment_se = np.sqrt((n_blocks - 1) * np.var(jackknife_enrichment_quotient, axis=0, ddof=1))
        
        # Compute difference for p-values
        jackknife_enrichment_diff = jackknife_h2_normalized - jackknife_h2_normalized[:, [0]]
        
        # Point estimates
        annotation_heritability, annotation_enrichment = cls._annotation_heritability(
            shared_data['variant_h2'], annotations, ref_col)

        # Two-tailed p-values using jackknife estimates
        annotation_heritability_p = np.array([
            cls._wald_pvalue(jackknife_h2[:, i])[2] for i in range(jackknife_h2.shape[1])
        ])
        # Use differences for p-values
        annotation_enrichment_p = np.array([
            cls._wald_pvalue(jackknife_enrichment_diff[:, i])[2] for i in range(jackknife_enrichment_diff.shape[1])
        ])
        params_p = np.array([
            cls._wald_pvalue(jackknife_params[:, i])[2] for i in range(jackknife_params.shape[1])
        ])
        
        if verbose:
            num_annotations = len(annotation_heritability)
            print(f"Heritability: {annotation_heritability[:min(5, num_annotations)]}")
            print(f"Enrichment: {annotation_enrichment[:min(5, num_annotations)]}")
            print(f"Enrichment p-values: {annotation_enrichment_p[:min(5, num_annotations)]}")

        return {
            'parameters': shared_data['params'].copy(),
            'parameters_se': params_se,
            'parameters_p': params_p,
            'heritability': annotation_heritability,
            'heritability_se': h2_se,
            'heritability_p': annotation_heritability_p,
            'enrichment': annotation_enrichment,
            'enrichment_se': enrichment_se,
            'enrichment_p': annotation_enrichment_p,
            'likelihood_history': log_likelihood_history,
            'jackknife_h2': jackknife_h2,
            'jackknife_params': jackknife_params,
            'jackknife_enrichment': jackknife_enrichment_quotient,
        }

def run_graphREML(model_options: ModelOptions,
                  method_options: MethodOptions,
                  summary_stats: pl.DataFrame,
                  annotation_data: pl.DataFrame,
                  ldgm_metadata_path: str,
                  populations: Optional[Union[str, List[str]]] = None,
                  chromosomes: Optional[Union[int, List[int]]] = None,
                  ):
    """Wrapper function for GraphREML.

    Args:
        model: Model instance containing parameters and settings
        summary_stats: DataFrame containing GWAS summary statistics
        annotation_data: DataFrame containing variant annotations
        ldgm_metadata_path: Path to LDGM metadata file
        populations: Optional list of populations to include
        chromosomes: Optional list of chromosomes to include
        num_processes: Optional number of processes for parallel computation
        max_iterations: Maximum number of iterations
        convergence_tol: Convergence tolerance
        match_by_position: If True, match variants by position instead of variant ID
        run_in_serial: If True, run in serial instead of parallel
        num_iterations: Number of repetitions

    Returns:
        Dictionary containing:
        - estimated parameters
        - heritability estimates
        - standard errors
        - convergence diagnostics
    """
    # Merge summary stats with annotations
    if not method_options.match_by_position:
        raise NotImplementedError
        # TODO correctly handle !match_by_position; currently CHR and POS become CHR_right, POS_right
    
    join_cols = ['CHR', 'POS'] if method_options.match_by_position else ['SNP']
    merge_how = 'right' if method_options.use_surrogate_markers else 'inner'
    merged_data = summary_stats.join(
        annotation_data,
        on=join_cols,
        how=merge_how
    )
    
    run_fn = GraphREML.run_serial if method_options.run_serial else GraphREML.run
    return run_fn(
        ldgm_metadata_path,
        populations=populations,
        chromosomes=chromosomes,
        sumstats=merged_data,
        num_processes=method_options.num_processes,
        worker_params=(model_options,method_options),
        num_params = len(model_options.annotation_columns),
        model=model_options,
        method=method_options,
        num_iterations=method_options.num_iterations,
        verbose=method_options.verbose,
        convergence_tol=method_options.convergence_tol,
        sample_size=model_options.sample_size,
    )