"""
Functions and classes for estimating heritability using LDGM precision matrices.
"""
from multiprocessing import Value, Array
from .multiprocessing import ParallelProcessor, WorkerManager, SharedData
from dataclasses import dataclass, field
from typing import *

import numpy as np
import polars as pl

from .precision import PrecisionOperator
from .likelihood import *
from .io import *

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
        link_fn_denominator: Number of SNPs to be 
            used as denominator for link function
    """
    annotation_columns: Optional[List[str]] = None
    params: Optional[np.ndarray] = None
    sample_size: float = 1.0
    intercept: float = 1.0
    link_fn_denominator: float = 1.0

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
    """
    gradient_num_samples: int = 100
    match_by_position: bool = False
    num_iterations: int = 10
    convergence_tol: float = 1e-3
    run_serial: bool = False
    num_processes: Optional[int] = None
    verbose: bool = False
    use_surrogate_markers: bool = True
    trust_region_size: float = 1e-1
    trust_region_rho_lb: float = 1e-4
    trust_region_rho_ub: float = .99
    trust_region_scalar: float = 5
    max_trust_iterations: int = 20
    reset_trust_region: bool = True

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

def _newton_step(gradient: np.ndarray, hessian: np.ndarray) -> np.ndarray:
    """Compute Newton step: -H^{-1}g."""
    # print(f"Performing step with gradient: {gradient}, hessian diagonal: {np.diag(hessian)}")
    # print(hessian)
    
    ll = np.trace(hessian) / len(hessian) * 50
    return -np.linalg.solve(hessian + ll * np.eye(len(hessian)), gradient)

def softmax_robust(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax implementation."""
    y = x + np.log1p(np.exp(-x))
    mask = x < 0
    y[mask] = np.log1p(np.exp(x[mask]))
    return y
    
def _get_softmax_link_function(n_snps: int) -> tuple[Callable, Callable]:
    """Create softmax link function and its gradient.

    Args:
        n_snps: Total number of SNPs across all blocks

    Returns:
        Tuple containing:
        - Link function mapping (annot, theta) to per-SNP heritabilities
        - Gradient of the link function
    """
    np.seterr(over='ignore')

    def _link_fn(annot: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """Softmax link function."""
        return softmax_robust(annot @ theta) / n_snps

    def _link_fn_grad(annot: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """Gradient of softmax link function."""
        x = annot @ theta
        result = annot / n_snps / (1 + np.exp(-x))
        mask = x.flatten() < 0
        if mask.any():
            result[mask,:] = (annot[mask,:] * np.exp(x[mask]) / 
                            (1 + np.exp(x[mask])) / n_snps)
        return result

    return _link_fn, _link_fn_grad


class GraphREML(ParallelProcessor):

    @classmethod
    def _sum_blocks(cls, array: np.ndarray, shape_each_block: tuple) -> np.ndarray:
        array = array.reshape((*shape_each_block, -1))
        result = array.sum(axis=-1)
        assert result.shape == shape_each_block
        return result

    @classmethod
    def prepare_block_data(cls, metadata: pl.DataFrame, **kwargs) -> list[tuple]:
        """Prepare block-specific data for processing.
        
        Args:
            metadata: DataFrame containing LDGM metadata
            **kwargs: Additional arguments from run(), including:
                sumstats: DataFrame containing Z scores and variant info, optionally annotations
                annotation_columns: list of column names for the annotations in sumstats
                
        Returns:
            List of dictionaries containing block-specific data with keys:
                sumstats: DataFrame for this block
                variant_offset: Cumulative number of variants before this block
                block_index: Index of this block
                Pz: Pre-computed precision-premultiplied Z scores for this block
        """
        sumstats = kwargs.get('sumstats')
        # Partition annotations into blocks
        sumstats_blocks: list[pl.DataFrame] = partition_variants(metadata, sumstats)

        cumulative_num_variants = np.cumsum(np.array([len(df) for df in sumstats_blocks]))
        cumulative_num_variants = [0] + list(cumulative_num_variants[:-1])
        block_indices = list(range(len(sumstats_blocks)))
        block_Pz = [None for _ in block_indices]
        
        return [
            {
                'sumstats': sumstats,
                'variant_offset': offset,
                'block_index': index,
                'Pz': Pz
            }
            for sumstats, offset, index, Pz 
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
        num_variants = sum([len(d['sumstats']) for d in block_data])

        result = SharedData({
            'params': num_params,
            'variant_h2': num_variants,
            'likelihood': num_blocks,
            'gradient': num_blocks * num_params,
            'hessian': num_blocks * num_params ** 2,
            'is_first_iter': None
        })

        result['is_first_iter'] = 1

        return result

    @staticmethod
    def _initialize_block_zscores(ldgm: PrecisionOperator, 
                                sumstats: pl.DataFrame,
                                annotation_columns: List[str],
                                match_by_position: bool
                                ) -> Tuple[np.ndarray, np.ndarray]:
        """Initialize Z-scores for a block by merging variants and computing Pz.
        
        Args:
            ldgm: LDGM object
            sumstats: Summary statistics DataFrame
            annotation_columns: List of annotation column names

        Returns:
            Tuple containing:
            - z: Raw Z-scores after filtering
            - Pz: Z-scores premultiplied by precision matrix
        """
        # print(sumstats.select('Z').head(55))

        # Merge annotations with LDGM variant info
        from .io import merge_snplists
        ldgm, sumstat_indices = merge_snplists(
            ldgm, sumstats,
            match_by_position=match_by_position,
            pos_col='POS',
            ref_allele_col='REF',
            alt_allele_col='ALT',
            add_allelic_cols=['Z'],
            add_cols=annotation_columns,
            modify_in_place=True
        )

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
            pl.Series('Z', z_arr)
        ]).drop('vi_row_nr')

        print(f"Number of missing rows: {len(variant_info_missing)}")
        
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
                           num_snps: int,
                           old_variant_h2: float,
                           n_samples: int,
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
        # Get link function and its gradient
        link_fn, link_fn_grad = _get_softmax_link_function(num_snps)

        # Compute precision matrix diagonal, aggregating variants with same index
        per_variant_h2 = link_fn(annotations, params)
        delta_D = np.zeros(ldgm.shape[0])
        np.add.at(delta_D, ldgm.variant_indices, per_variant_h2.flatten() - old_variant_h2.flatten())
        ldgm.update_matrix(delta_D)

        likelihood = gaussian_likelihood(Pz, ldgm)

        if likelihood_only:
            ldgm.del_factor()
            return likelihood, None, None, per_variant_h2

        # Compute gradient of precision matrix wrt parameters
        del_h2_del_a = link_fn_grad(annotations, params)
        
        del_M_del_a = np.zeros((ldgm.shape[0], params.shape[0]))
        np.add.at(del_M_del_a, ldgm.variant_indices, del_h2_del_a)


        gradient = gaussian_likelihood_gradient(
            Pz, ldgm, del_M_del_a=del_M_del_a,
            n_samples=n_samples,
        )

        # Compute hessian
        hessian = gaussian_likelihood_hessian(
            Pz, ldgm, del_M_del_a=del_M_del_a
        )
        assert ~np.any(np.isnan(hessian))

        ldgm.del_factor()

        return likelihood, gradient, hessian, per_variant_h2

    @classmethod
    def process_block(cls, ldgm: PrecisionOperator,
                     flag: Value,
                     shared_data: SharedData,
                     block_offset: int,
                     block_data: Any = None,
                     worker_params: Tuple[ModelOptions, MethodOptions] = None) -> None:
        """Computes likelihood, gradient, and hessian for a single block. If flag is 2, it only computes the likelihood.
        """
        model_options, method_options = worker_params
        num_annot = len(model_options.annotation_columns)
        block_data_dict = block_data
        sumstats = block_data_dict['sumstats']
        variant_offset = block_data_dict['variant_offset']
        block_index = block_data_dict['block_index']
        
        if shared_data['is_first_iter'] == 1:
            # First iteration - initialize Z-scores
            ldgm, Pz = cls._initialize_block_zscores(ldgm, 
                                                    sumstats, 
                                                    model_options.annotation_columns,
                                                    method_options.match_by_position)
            block_data_dict['Pz'] = Pz
        else:
            Pz = block_data_dict['Pz']

        # Get annotations from sumstats
        annot = ldgm.variant_info.select(model_options.annotation_columns).to_numpy()

        block_variants = slice(variant_offset, variant_offset + len(ldgm.variant_info))
        variant_h2 = shared_data['variant_h2', block_variants]

        likelihood_only = (flag.value==2)
        likelihood, gradient, hessian, variant_h2 = cls._compute_block_likelihood(
            ldgm=ldgm,
            Pz=Pz,
            annotations=annot,
            params=shared_data['params'].reshape(-1,1),
            num_snps=model_options.link_fn_denominator,
            old_variant_h2=variant_h2,
            n_samples=method_options.gradient_num_samples,
            likelihood_only=likelihood_only
        )

        shared_data['likelihood', block_index] = likelihood
        shared_data['variant_h2', block_variants] = variant_h2
        if likelihood_only:
            return

        gradient_slice = slice(block_index * num_annot,
                             (block_index + 1) * num_annot)
        shared_data['gradient', gradient_slice] = gradient.flatten()

        hessian_slice = slice(block_index * num_annot**2,
                            (block_index + 1) * num_annot**2)
        shared_data['hessian', hessian_slice] = hessian.flatten()

    @staticmethod
    def _annotation_heritability(variant_h2: np.ndarray, annot: pl.DataFrame, ref_col: int) -> Tuple[np.ndarray, np.ndarray]:
        """Computes the heritability of each annotation based on the given h2 values.
        """
        annot_mat = annot.to_numpy().T
        annot_h2 = annot_mat @ variant_h2
        annot_size = np.sum(annot_mat, axis=1)
        annot_enrichment = annot_size[ref_col] * annot_h2 / (annot_h2[ref_col] * annot_size)

        return annot_h2, annot_enrichment


    @classmethod
    def supervise(cls, manager: WorkerManager, shared_data: SharedData, block_data: list, **kwargs):
        """Runs graphREML.
        Args:
            manager: used to start parallel workers
            shared_data: used to communicate with workers
            block_data: annotation + gwas data passed to workers
            **kwargs: Additional arguments
        """
        flags = {
            'ERROR': -2,
            'SHUTDOWN': -1,
            'FINISHED': 0,
            'COMPUTE_ALL': 1,
            'COMPUTE_LIKELIHOOD_ONLY': 2,
        }

        num_iterations = kwargs.get('num_iterations')
        num_params = kwargs.get('num_params')
        verbose = kwargs.get('verbose')
        sample_size = kwargs.get('sample_size')
        method = kwargs.get('method')
        model = kwargs.get('model')
        
        log_likelihood_history = []
        trust_region_lambda = method.trust_region_size
        
        def _trust_region_step(gradient: np.ndarray, hessian: np.ndarray, trust_region_lambda: float) -> np.ndarray:
            """Compute trust region step by solving (H + λD)x = -g.
            
            Args:
                gradient: Gradient vector
                hessian: Hessian matrix
                trust_region_lambda: Trust region parameter
                
            Returns:
                Step vector
            """
            hess_mod = hessian + trust_region_lambda * np.diag(np.diag(hessian))
            hess_mod += np.finfo(float).eps * np.eye(len(hessian))
            
            # TODO from MATLAB; commented out for now
            # hess_mod += 1e-2 * trust_region_lambda
            
            return np.linalg.solve(hess_mod, gradient)

        for rep in range(num_iterations):
            if verbose:
                print(f"starting iteration {rep} with params {shared_data['params']}")
            
            # Calculate likelihood, gradient, and hessian for each block
            manager.start_workers(flags['COMPUTE_ALL'])
            manager.await_workers()
            shared_data['is_first_iter'] = 0

            likelihood = cls._sum_blocks(shared_data['likelihood'], (1,))
            gradient = cls._sum_blocks(shared_data['gradient'], (num_params,))
            hessian = cls._sum_blocks(shared_data['hessian'], (num_params,num_params))
            
            old_params = shared_data['params'].copy()
            old_likelihood = likelihood[0]
            
            # Reset trust region size if specified
            if method.reset_trust_region or rep == 0:
                trust_region_lambda = method.trust_region_size
                
            # Trust region optimization loop
            for trust_iter in range(method.max_trust_iterations):
                # Compute proposed step
                step = _trust_region_step(gradient, hessian, trust_region_lambda)
                shared_data['params'] = old_params - step
                
                # Evaluate proposed step
                manager.start_workers(flags['COMPUTE_LIKELIHOOD_ONLY'])
                manager.await_workers()
                new_likelihood = cls._sum_blocks(shared_data['likelihood'], (1,))[0]
                
                # Compute actual vs predicted increase
                actual_increase = new_likelihood - old_likelihood
                predicted_increase = -(step.T @ gradient - 0.5 * step.T @ (hessian @ step))
                
                # Check if step is acceptable
                if actual_increase < 0:  # Likelihood decreased (worse)
                    rho = -1
                else:
                    rho = abs(actual_increase) / predicted_increase if predicted_increase > 0 else actual_increase / predicted_increase if predicted_increase > 0 else -1
                    
                # Update trust region size
                if rho < method.trust_region_rho_lb:
                    trust_region_lambda *= method.trust_region_scalar
                    shared_data['params'] = old_params  # Revert step
                elif rho > method.trust_region_rho_ub:
                    trust_region_lambda /= method.trust_region_scalar
                    break  # Accept step and continue to next iteration
                else:
                    break  # Accept step with current trust region size
                    
                if trust_iter == method.max_trust_iterations - 1:
                    if verbose:
                        print("Warning: Maximum trust region iterations reached")
                    shared_data['params'] = old_params - step  # Use last step
            
            log_likelihood_history.append(new_likelihood)
            heritability = np.sum(shared_data['variant_h2'] / sample_size)

            annotations = pl.concat([dict['sumstats'].select(model.annotation_columns) for dict in block_data])
            ref_col = 0 # Maybe TODO
            annotation_heritability, annotation_enrichment = cls._annotation_heritability(
                shared_data['variant_h2'], annotations, ref_col)
            
            if verbose:
                print(f"heritability: {heritability}")
                print(f"enrichment: {annotation_enrichment}")
                print(f"Variants with nonzero h2: {np.sum(shared_data['variant_h2'] != 0)}")
                print(f"Variants: {len(shared_data['variant_h2'])}")
                print(f"Trust region lambda: {trust_region_lambda}")
                print(f"max and min variant h2: {np.max(shared_data['variant_h2'])}, {np.min(shared_data['variant_h2'])}")
                
            # Check convergence
            if len(log_likelihood_history) >= 2:
                if abs(log_likelihood_history[-1] - log_likelihood_history[-2]) < method.convergence_tol:
                    if verbose:
                        print("Converged!")
                    break

        return heritability, log_likelihood_history

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